import torch
import os
import numpy as np
from pathlib import Path
import random
import argparse
import json
import warnings
import copy

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Zennit imports
from zennit.composites import EpsilonGammaBox, EpsilonAlpha2Beta1, EpsilonPlus, EpsilonPlusFlat
from zennit.attribution import Gradient
from zennit.torchvision import ResNetCanonizer

from model import ResNet50Predictor, finetune_net
from PIL import Image
import matplotlib.pyplot as plt
from eval_utils import insert_grey_circle


class ZennitSupervisedExplainer:
    """Explanation generator for supervised binary classifier using Zennit's LRP implementation."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Set circle parameters before _setup_experiment() which may use them
        self.circle_center_offset = -20
        self.circle_radius = 20
        self.circle_grey_value = 128
        self._setup_experiment()
        self._load_checkpoint()
        self._initialize_explainer()

    def _setup_experiment(self):
        """Set random seeds and directories."""
        self.seed = self.args.random_state
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Directory setup
        self.root_dir = Path(__file__).resolve().parents[1]
        base = os.path.join(self.root_dir, "adni_results")
        self.heatmap_dir = os.path.join(base, "heatmaps_classifier_zennit")
        self.param_dir = os.path.join(base, "params")
        self.overlay_dir = os.path.join(base, "overlay_classifier_zennit")

        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)

        self._save_param()
        self._load_test_set()

    def _save_param(self):
        """Save experiment parameters."""
        args_dict = vars(self.args)
        param_path = os.path.join(self.param_dir, f"zennit_lrp_{self.args.composite}")
        with open(f"{param_path}_params.json", "w") as f:
            json.dump(args_dict, f, indent=4)

    def _convert_to_tensor(self, group):
        """Convert numpy array to normalized tensor."""
        group_tensor = torch.tensor(group).unsqueeze(1).to(self.device).float()
        group_tensor = group_tensor / 255.0
        group_tensor = group_tensor.repeat(1, 3, 1, 1)

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        group_tensor = (group_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return group_tensor

    def _add_grey_circle_artifact_all_samples(self, group_np):
        # Apply grey circle to each sample in group_2
        print(f"Applying grey circle to each sample in group_2...")
        for i in range(len(group_np)):
            # Get image dimensions
            height, width = group_np[i].shape
            # Insert grey circle at center with specified radius and offset
            center_y = height // 2 + self.circle_center_offset
            center_x = width // 2 + self.circle_center_offset
            center = (center_y, center_x)
            group_np[i] = insert_grey_circle(group_np[i], center, self.circle_radius, self.circle_grey_value)
        return group_np

    def _load_test_set(self):
        """Load test dataset."""
        if self.args.dst == "corr" and self.args.deg == "zer-test":
            test_dir = self.root_dir / "adni_results" / "split" / "test" / "True"
            out_path = test_dir / "zer32" / "test_split.npz"

            with np.load(out_path) as f:
                self.group0 = f["test0"]
                self.group1 = f["test1"]

            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m]
        elif self.args.dst == "test":
            print(f"Using test set for classification explanation")
            test_dir = self.root_dir / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"

            with np.load(out_path) as f:
                print(f"Loaded keys: {f.files}")
                self.group0 = f["test0"]
                self.group1 = f["test1"]

            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m]

        elif self.args.dst == "faithfulness_eval":
            print(f"Using test set for getting embeddings")
            root_dir = "."
            test_dir = Path(root_dir) / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"
            with np.load(out_path) as f:
                print(f.files)  # -> ['test0', 'test1']
                self.group0 = f["test0"]
                self.group1 = copy.deepcopy(f["test0"])
            self.group0_np = self.group0[: self.args.n]
            self.group1_np = self.group1[: self.args.m].copy()

            self.group1_np = self._add_grey_circle_artifact_all_samples(self.group1_np)
        else:
            raise ValueError(f"Unsupported dst='{self.args.dst}' and deg='{self.args.deg}' combination")

        print(f"Loaded {len(self.group0_np)} samples from group 0")
        print(f"Loaded {len(self.group1_np)} samples from group 1")

    def _load_checkpoint(self):
        """Load model checkpoint."""
        print("Loading fine-tuned classifier model")
        checkpoint_path = os.path.join(self.root_dir, self.args.model_path)
        print(f"Checkpoint path: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
        self.model = finetune_net(net, num_classes=2).to(self.device)
        self.model.load_state_dict(state_dict["model_state"])
        self.model.eval()

        print("Model loaded successfully")

    def _initialize_explainer(self):
        """Initialize Zennit composite and canonizers for LRP."""
        composite_name = self.args.composite.lower()

        # Initialize canonizers for ResNet
        # ResNetCanonizer handles residual connections properly
        canonizers = [ResNetCanonizer()]

        # Select composite based on argument
        if composite_name == "epsilongammabox":
            self.composite = EpsilonGammaBox(
                low=self.args.low,
                high=self.args.high,
                gamma=self.args.gamma,
                epsilon=self.args.epsilon,
                canonizers=canonizers,
            )
            print(
                f"Initialized EpsilonGammaBox composite (low={self.args.low}, high={self.args.high}, "
                f"gamma={self.args.gamma}, epsilon={self.args.epsilon})"
            )
        elif composite_name == "epsilonplus":
            self.composite = EpsilonPlus(epsilon=self.args.epsilon, canonizers=canonizers)
            print(f"Initialized EpsilonPlus composite (epsilon={self.args.epsilon})")
        elif composite_name == "epsilonplusflat":
            self.composite = EpsilonPlusFlat(epsilon=self.args.epsilon, canonizers=canonizers)
            print(f"Initialized EpsilonPlusFlat composite (epsilon={self.args.epsilon})")

        elif composite_name == "epsilonalpha2beta1":
            self.composite = EpsilonAlpha2Beta1(epsilon=self.args.epsilon, canonizers=canonizers)
            print(f"Initialized EpsilonAlpha2Beta1 composite (epsilon={self.args.epsilon})")

        else:
            raise ValueError(
                f"Unknown composite: {self.args.composite}. "
                f"Supported composites: 'epsilongammabox', 'epsilonplusflat', 'epsilonalpha2beta1'"
            )

    def compute_attributions(self, images, target_class):
        """
        Compute LRP attributions for a batch of images using Zennit.

        Args:
            images: Tensor of images (B, C, H, W)
            target_class: Target class for attribution (0 or 1)

        Returns:
            attributions: Attribution maps (B, C, H, W)
            logits: Model predictions (B, num_classes)
        """
        # Get model predictions first (without grad)
        with torch.no_grad():
            logits = self.model(images)

        # Prepare target tensor for Zennit
        # For binary classification, we create a one-hot style target
        # Shape: (B, num_classes) with 1.0 at target_class
        target_tensor = torch.zeros_like(logits)
        target_tensor[:, target_class] = 1.0

        with Gradient(model=self.model, composite=self.composite) as attributor:
            # attributor returns (output, attribution)
            # We pass the target tensor to specify which output neuron to attribute to
            _, attributions = attributor(images, target_tensor)

        return attributions, logits

    def process_group(self, group_np, group_label, group_name):
        """
        Process a group of images and compute LRP attributions using Zennit.

        Args:
            group_np: Numpy array of images
            group_label: Class label for this group (0 or 1)
            group_name: Name for saving ("group0" or "group1")

        Returns:
            attributions_np: Numpy array of attributions
            logits_np: Numpy array of predictions
        """
        # Convert to tensor
        images = self._convert_to_tensor(group_np)

        # Process in batches
        batch_size = self.args.bs
        n_samples = len(images)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_attributions = []
        all_logits = []

        print(f"Processing {group_name} with {n_samples} samples in {n_batches} batches")

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_images = images[start_idx:end_idx]

            # Compute attributions
            attributions, logits = self.compute_attributions(batch_images, group_label)

            # Store results
            all_attributions.append(attributions.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

            # Clear GPU memory
            del attributions, logits, batch_images
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  Processed batch {i+1}/{n_batches}")

        # Concatenate all batches
        attributions_np = np.vstack(all_attributions)
        logits_np = np.vstack(all_logits)

        print(f"{group_name} processing complete")
        print(f"  Attributions shape: {attributions_np.shape}")
        print(f"  Logits shape: {logits_np.shape}")

        return attributions_np, logits_np

    def generate_heatmap(self, attributions, use_relu=False):
        """
        Generate heatmap from attributions.

        Args:
            attributions: Attributions (C, H, W) or (1, C, H, W)
            use_relu: If True, apply ReLU (keep only positive values). If False, keep raw values.

        Returns:
            heatmap: 2D heatmap (H, W) normalized to [0, 1]
                     - For use_relu=False: 0 in attribution maps to 0.5 in heatmap (middle of colormap)
                     - For use_relu=True: standard min-max normalization
        """
        # If batch dimension exists, squeeze it
        if attributions.ndim == 4:
            attributions = attributions.squeeze(0)

        # Sum across color channels
        heatmap = attributions.sum(axis=0)

        # Optionally apply ReLU
        if use_relu:
            heatmap = np.maximum(0, heatmap)
            # Standard min-max normalization for ReLU (all positive values)
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            if heatmap_max - heatmap_min > 1e-8:
                heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            # For raw attributions (can be negative), normalize symmetrically around zero
            # Zero attribution -> 0.5 (middle of colormap)
            # Max positive -> 1.0, Max negative -> 0.0
            max_abs = np.abs(heatmap).max()
            if max_abs > 1e-8:
                # Normalize to [-1, 1] then shift to [0, 1]
                heatmap = heatmap / max_abs  # Now in [-1, 1]
                heatmap = (heatmap + 1.0) / 2.0  # Now in [0, 1] with 0 -> 0.5
            else:
                # All values are essentially zero
                heatmap = np.full_like(heatmap, 0.5)

        return heatmap

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W) in range [0, 255]
            heatmap: Attribution heatmap (H, W) in range [0, 1]
            alpha: Transparency of heatmap overlay

        Returns:
            overlaid: RGB image with heatmap overlay
        """
        # Normalize image to [0, 1]
        image_norm = image.astype(np.float32) / 255.0

        # Apply colormap to heatmap
        colormap = plt.get_cmap("jet")
        heatmap_colored = colormap(heatmap)[:, :, :3]  # Remove alpha channel

        # Blend image and heatmap
        overlaid = (1 - alpha) * np.stack([image_norm] * 3, axis=-1) + alpha * heatmap_colored
        overlaid = (overlaid * 255).astype(np.uint8)

        return overlaid

    def run(self):
        """Main execution function."""
        print("=" * 50)
        print(f"Running Zennit LRP Explanation for Classification")
        print(f"Composite: {self.args.composite}")
        print("=" * 50)

        # Process group 0 (label 0)
        group0_attr, group0_logits = self.process_group(self.group0_np, 0, "group0")

        # Process group 1 (label 1)
        group1_attr, group1_logits = self.process_group(self.group1_np, 1, "group1")

        # Save raw attributions
        composite_suffix = self.args.composite.lower()
        attr_path0 = os.path.join(self.heatmap_dir, f"group0_zennit_lrp_{composite_suffix}_raw_attr.npy")
        attr_path1 = os.path.join(self.heatmap_dir, f"group1_zennit_lrp_{composite_suffix}_raw_attr.npy")
        np.save(attr_path0, group0_attr)
        np.save(attr_path1, group1_attr)

        # Save logits
        logits_path0 = os.path.join(self.heatmap_dir, f"group0_zennit_lrp_{composite_suffix}_logits.npy")
        logits_path1 = os.path.join(self.heatmap_dir, f"group1_zennit_lrp_{composite_suffix}_logits.npy")
        np.save(logits_path0, group0_logits)
        np.save(logits_path1, group1_logits)

        # Generate and save heatmaps
        print("\nGenerating heatmaps...")
        use_relu = self.args.use_relu
        group0_heatmaps = np.array([self.generate_heatmap(attr, use_relu=use_relu) for attr in group0_attr])
        group1_heatmaps = np.array([self.generate_heatmap(attr, use_relu=use_relu) for attr in group1_attr])

        heatmap_path0 = os.path.join(self.heatmap_dir, f"group0_zennit_lrp_{composite_suffix}_heatmaps.npy")
        heatmap_path1 = os.path.join(self.heatmap_dir, f"group1_zennit_lrp_{composite_suffix}_heatmaps.npy")
        np.save(heatmap_path0, group0_heatmaps)
        np.save(heatmap_path1, group1_heatmaps)

        print(f"\nSaved attributions and heatmaps:")
        print(f"  Group 0: {attr_path0}")
        print(f"  Group 1: {attr_path1}")

        # Print prediction statistics
        print("\n" + "=" * 50)
        print("Prediction Statistics")
        print("=" * 50)
        probs0 = torch.softmax(torch.from_numpy(group0_logits), dim=1).numpy()
        probs1 = torch.softmax(torch.from_numpy(group1_logits), dim=1).numpy()

        print(f"Group 0 (true label=0):")
        print(f"  Mean probability for class 0: {probs0[:, 0].mean():.4f}")
        print(f"  Mean probability for class 1: {probs0[:, 1].mean():.4f}")
        print(f"  Accuracy: {(probs0.argmax(axis=1) == 0).mean():.4f}")

        print(f"\nGroup 1 (true label=1):")
        print(f"  Mean probability for class 0: {probs1[:, 0].mean():.4f}")
        print(f"  Mean probability for class 1: {probs1[:, 1].mean():.4f}")
        print(f"  Accuracy: {(probs1.argmax(axis=1) == 1).mean():.4f}")

        return group0_heatmaps, group1_heatmaps

    def visualize_samples(self, n_samples=5):
        """
        Visualize sample images with their attribution heatmaps.

        Args:
            n_samples: Number of samples to visualize from each group
        """
        composite_suffix = self.args.composite.lower()

        # Load heatmaps
        heatmap_path0 = os.path.join(self.heatmap_dir, f"group0_zennit_lrp_{composite_suffix}_heatmaps.npy")
        heatmap_path1 = os.path.join(self.heatmap_dir, f"group1_zennit_lrp_{composite_suffix}_heatmaps.npy")

        if not os.path.exists(heatmap_path0) or not os.path.exists(heatmap_path1):
            print("Heatmaps not found. Run the main processing first.")
            return

        group0_heatmaps = np.load(heatmap_path0)
        group1_heatmaps = np.load(heatmap_path1)

        # Visualize samples from each group
        for images, heatmaps, group_name in [
            (self.group0_np, group0_heatmaps, "group0"),
            (self.group1_np, group1_heatmaps, "group1"),
        ]:
            print(f"\nVisualizing {n_samples} samples from {group_name}")

            for i in range(min(n_samples, len(images))):
                image = images[i]
                heatmap = heatmaps[i]

                # Save heatmap as standalone PNG
                heatmap_colormap = plt.get_cmap("jet")
                heatmap_colored = heatmap_colormap(heatmap)[:, :, :3]  # Remove alpha channel
                heatmap_img = (heatmap_colored * 255).astype(np.uint8)
                heatmap_path = os.path.join(
                    self.overlay_dir, f"{self.args.dst}_{group_name}_classification_{i}_heatmap_{composite_suffix}.png"
                )
                Image.fromarray(heatmap_img).save(heatmap_path)

                # Create and save overlay
                overlaid = self.overlay_heatmap(image, heatmap, alpha=self.args.alpha)
                overlay_path = os.path.join(
                    self.overlay_dir, f"{self.args.dst}_{group_name}_classification_{i}_overlay_{composite_suffix}.png"
                )
                Image.fromarray(overlaid).save(overlay_path)

            print(f"  Saved heatmaps and overlays to {self.overlay_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Classifier Explanation with Zennit LRP")
    parser.add_argument("--n", type=int, default=100, help="Number of samples in group 0")
    parser.add_argument("--m", type=int, default=100, help="Number of samples in group 1")
    parser.add_argument("--bs", type=int, default=16, help="Batch size for processing")
    parser.add_argument(
        "--dst",
        type=str,
        default="test",
        choices=["test", "corr", "faithfulness_eval"],
        help="Dataset type",
    )
    parser.add_argument(
        "--deg",
        type=str,
        default="None",
        help="Corruption type (e.g., 'zer-test' for corrupted data)",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_path",
        type=str,
        default="adni_results/ckps/model_finetun_best_12_False.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--composite",
        type=str,
        default="epsilonplusflat",
        choices=["epsilongammabox", "epsilonplus", "epsilonplusflat", "epsilonalpha2beta1"],
        help="Zennit composite to use for LRP",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon value for LRP stability")
    parser.add_argument("--gamma", type=float, default=0.25, help="Gamma value for LRP (used in EpsilonGammaBox)")
    parser.add_argument(
        "--low",
        type=float,
        default=0.0,
        help="Lower bound for input normalization (used in EpsilonGammaBox)",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=1.0,
        help="Upper bound for input normalization (used in EpsilonGammaBox)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for overlay visualization")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization overlays after processing")
    parser.add_argument("--n_vis", type=int, default=5, help="Number of samples to visualize from each group")
    parser.add_argument(
        "--use_relu",
        action="store_true",
        help="Apply ReLU to attributions (keep only positive values). If not set, keeps raw values including negative.",
    )

    args = parser.parse_args()

    # Create experiment instance
    experiment = ZennitSupervisedExplainer(args)

    # Run attribution
    group0_heatmaps, group1_heatmaps = experiment.run()

    # Optionally visualize samples
    if args.visualize:
        experiment.visualize_samples(n_samples=args.n_vis)

    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)
