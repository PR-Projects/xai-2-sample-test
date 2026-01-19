import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import os
import numpy as np
from pathlib import Path
import random
import argparse
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from model import SimCLR, ResNet50Predictor, finetune_net
from data import AdniMRIDataset2D


class SampleImportance:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._set_random_seed()
        self._setup_directories()
        self._load_checkpoint()
        self._load_images()

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        self.seed = self.args.random_state
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_directories(self):
        """Setup directory paths relative to parent directory."""
        self.root_dir = Path(__file__).resolve().parents[1]
        base = self.root_dir / "adni_results"
        self.embed_dir = base / "embeddings"
        self.param_dir = base / "params"
        self.inf_dir = base / "infscores"

        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)
        os.makedirs(self.inf_dir, exist_ok=True)

        self._save_param()

    def _save_param(self):
        """Save experiment parameters."""
        args_dict = vars(self.args)
        param_path = os.path.join(self.param_dir, f"{self.args.exp}_params.json")
        with open(param_path, "w") as f:
            json.dump(args_dict, f, indent=4)

    def _load_checkpoint(self):
        """Load model checkpoint based on checkpoint type."""
        if self.args.ckp == "simclr":
            print("Using self-supervised pre-trained model")
            checkpoint_dir = self.root_dir / "self_supervised" / "simclr" / "simclr_ckpts"
            pre_exp = 2
            sam_dir_last = os.path.join(checkpoint_dir, f"{pre_exp}_last_sclr.pt")
            state_dict = torch.load(sam_dir_last, weights_only=False, map_location=self.device)
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = SimCLR(backbone, hid_dim=2048, out_dim=128).to(self.device)
            model.load_state_dict(state_dict["model"])
            self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
            print(f"SimCLR model loaded from {sam_dir_last}")

        elif self.args.ckp == "fnt":
            print("Using fine-tuned model on two groups of data without corruption")
            checkpoint_dir = os.path.join(self.root_dir, self.args.model_path)
            state_dict = torch.load(checkpoint_dir, weights_only=False, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            print(f"Fine-tuned model loaded from {checkpoint_dir}")

        elif self.args.ckp == "fnt_bl":
            print("Using fine-tuned model on blurred images")
            checkpoint_dir = self.root_dir / "adni_results" / "ckps" / "model_finetun_last_7_True.pt"
            state_dict = torch.load(checkpoint_dir, weights_only=False, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            print(f"Fine-tuned (blurred) model loaded from {checkpoint_dir}")

        elif self.args.ckp == "fnt_zer":
            print("Using fine-tuned model on zero-patch corrupted images")
            checkpoint_dir = os.path.join(self.root_dir, self.args.model_path)
            state_dict = torch.load(checkpoint_dir, weights_only=False, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            model = finetune_net(net, num_classes=2).to(self.device)
            model.load_state_dict(state_dict["model_state"])
            backbone = model.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            print(f"Fine-tuned (zero-patch) model loaded from {checkpoint_dir}")

        elif self.args.ckp == "suppr":
            print("Using supervised pre-trained model without fine-tuning")
            checkpoint_dir = self.root_dir / "adni_results" / "ckps" / "resnet50_ukb_age_predict_epoch13.pth"
            weights = torch.load(checkpoint_dir, weights_only=False, map_location=self.device)
            net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(self.device)
            net.load_state_dict(weights)
            backbone = net.feature_extractor
            self.encoder = torch.nn.Sequential(*list(backbone.children()))
            print(f"Supervised pre-trained model loaded from {checkpoint_dir}")

        self.encoder.eval()

    def _convert_to_tensor(self, group):
        """Convert numpy array to normalized tensor."""
        group_tensor = torch.tensor(group).unsqueeze(1).to(self.device).float()
        group_tensor = group_tensor / 255.0  # Rescale to [0, 1]
        group_tensor = group_tensor.repeat(1, 3, 1, 1)  # Create 3 channels

        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        group_tensor = (group_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return group_tensor

    def _load_images(self):
        """Load test dataset from NPZ file."""
        if self.args.dst == "test":
            print("Loading test set from NPZ file")
            test_dir = self.root_dir / "adni_results" / "split" / "test" / "False" / "None"
            out_path = test_dir / "test_split.npz"
            with np.load(out_path) as f:
                print(f"Available arrays: {f.files}")
                group_1 = f["test0"]
                group_2 = f["test1"]

            self.group_1_np = group_1[: self.args.n]
            self.group_2_np = group_2[: self.args.m]

        elif self.args.dst == "full":
            print("Loading full dataset")
            dataset = AdniMRIDataset2D(
                annotations_file=self.args.annot_path,
                img_dir=self.args.img_path
            )
            group_1, group_2 = [], []
            for img, label in dataset:
                if label == 0:
                    group_1.append(img)
                else:
                    group_2.append(img)
            group_1 = np.concatenate(group_1, axis=0)
            group_2 = np.concatenate(group_2, axis=0)
            self.group_1_np = group_1[: self.args.n]
            self.group_2_np = group_2[: self.args.m]

        # Convert to tensors and create DataLoaders
        self.group_1 = self._convert_to_tensor(self.group_1_np)
        self.group_2 = self._convert_to_tensor(self.group_2_np)

        self.healthy_loader = DataLoader(self.group_1, batch_size=self.args.bs, shuffle=False, drop_last=False)
        self.unhealthy_loader = DataLoader(self.group_2, batch_size=self.args.bs, shuffle=False, drop_last=False)

        print(f"Loaded {len(self.group_1)} samples in group 1, {len(self.group_2)} samples in group 2")

    def _get_embeddings(self, dataloader):
        """Takes dataloader of each group, extract embedding vectors, return embeddings"""
        embeddings_list = []
        with torch.no_grad():
            for images in dataloader:
                images = images.to(self.device)
                embeddings = self.encoder.forward(images)
                embeddings = embeddings.view(embeddings.size()[0], -1)
                embeddings_list.append(embeddings.cpu().numpy())
        return np.vstack(embeddings_list)

    def _exclude_sample(self, i, tensor_data):
        """Exclude i'th sample from tensor data, return new DataLoader"""
        # Create indices excluding the specified sample
        indices = list(range(len(tensor_data)))
        indices.pop(i)
        # Select remaining samples
        remaining_data = tensor_data[indices]
        # Create a new DataLoader with the updated data
        updated_loader = DataLoader(remaining_data, batch_size=self.args.bs, shuffle=False, drop_last=False)
        return updated_loader

    def _get_orig_test_statistic(self):
        embed_X, embed_Y = self.load_embeddings()
        statistic = self._calculate_test_statistics(embed_X, embed_Y)
        print(f"test-statistic for whole dataset is:{statistic:0.4f}")
        return statistic

    def _calculate_test_statistics(self, embed_X, embed_Y):
        """Calculate the test statistic for different groups of DR."""
        # Calculate mean embeddings
        mean_X = embed_X.mean(0)
        mean_Y = embed_Y.mean(0)
        D = mean_X - mean_Y
        statistic = np.linalg.norm(D) ** 2
        return statistic

    def load_embeddings(self):
        """Load embeddings from saved files, or compute them if not available."""
        n = self.args.n
        m = self.args.m

        # Build file paths
        file_path = os.path.join(self.embed_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{n+m}")
        os.makedirs(file_path, exist_ok=True)

        path_X = os.path.join(file_path, "healthy_embed.npy")
        path_Y = os.path.join(file_path, "unhealthy_embed.npy")

        # Check if embedding files exist
        if os.path.exists(path_X) and os.path.exists(path_Y):
            print("Loading embeddings from saved numpy files...")
            embed_X = np.load(path_X, mmap_mode="r")
            embed_Y = np.load(path_Y, mmap_mode="r")
            return embed_X, embed_Y
        else:
            print("Computing embeddings...")
            embed_X = self._get_embeddings(self.healthy_loader)
            embed_Y = self._get_embeddings(self.unhealthy_loader)
            # Save embeddings
            np.save(path_X, embed_X)
            np.save(path_Y, embed_Y)
            print(f"Embeddings saved to {file_path}")
            return embed_X, embed_Y

    def _get_inf_score(self, tensor_data, D, embed_O):
        """Compute influence score for each sample in tensor_data."""
        inf_list = []
        for i in range(len(tensor_data)):
            new_dataloader = self._exclude_sample(i, tensor_data)
            embed_N = self._get_embeddings(new_dataloader)
            d_new = self._calculate_test_statistics(embed_O, embed_N)
            inf_score = D - d_new
            inf_list.append(inf_score)
        return inf_list

    def compute_all_inf_score(self):
        """Compute influence scores for all samples in both groups."""
        D = self._get_orig_test_statistic()
        embed_X, embed_Y = self.load_embeddings()

        print("Computing influence scores for group 1...")
        inf_list_X = self._get_inf_score(self.group_1, D, embed_Y)
        print("Computing influence scores for group 2...")
        inf_list_Y = self._get_inf_score(self.group_2, D, embed_X)

        print(f"inf_list_X: {inf_list_X}")
        print(f"inf_list_Y: {inf_list_Y}")

        # Build file paths
        n = len(inf_list_X)
        m = len(inf_list_Y)
        file_path = os.path.join(self.inf_dir, f"{self.seed}_{self.args.exp}_{self.args.ckp}_{n+m}")
        os.makedirs(file_path, exist_ok=True)

        path_X = os.path.join(file_path, "healthy_inf.npy")
        path_Y = os.path.join(file_path, "unhealthy_inf.npy")

        np.save(path_X, np.array(inf_list_X))
        np.save(path_Y, np.array(inf_list_Y))
        print(f"Influence scores saved to {file_path}")

        return inf_list_X, inf_list_Y


parser = argparse.ArgumentParser(description="Computing Sample_Importance")
parser.add_argument("--exp", type=str, default="sampleimp-fnt10-uncor", help="Experiment name")
parser.add_argument("--ckp", type=str, default="fnt", choices=["simclr", "fnt", "fnt_bl", "fnt_zer", "suppr"],
                    help="Checkpoint type")
parser.add_argument("--model_path", type=str, default="adni_results/ckps/model_finetun_last_10_False.pt",
                    help="Path to model checkpoint (relative to parent dir)")
parser.add_argument("--dst", type=str, default="test", choices=["test", "full", "corr"],
                    help="Dataset type")
parser.add_argument("--n", type=int, default=100, help="Number of samples in group 1")
parser.add_argument("--m", type=int, default=100, help="Number of samples in group 2")
parser.add_argument("--bs", type=int, default=16, help="Batch size")
parser.add_argument("--random_state", type=int, default=42, help="Random seed")
args = parser.parse_args()

if __name__ == "__main__":

    test = SampleImportance(args)
    inf_X, inf_Y = test.compute_all_inf_score()
