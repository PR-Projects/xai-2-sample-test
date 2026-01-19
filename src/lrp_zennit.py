import torch
import torch.nn as nn
import torch.nn.functional as F
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat, EpsilonAlpha2Beta1
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient


class EncoderWithProjection(nn.Module):
    """
    Wrapper that combines encoder with a projection onto the test statistic direction.

    This module:
    1. Passes input through the encoder to get embeddings
    2. Projects embeddings onto the direction vector D (group mean difference)
    3. Returns a scalar output representing contribution to test statistic
    """

    def __init__(self, encoder, direction_vector, sign=1.0, scaling=1.0):
        """
        Args:
            encoder: The feature extraction model
            direction_vector: D = mean_group0 - mean_group1 (tensor of shape [embed_dim])
            sign: +1 for group0, -1 for group1
            scaling: 2/n for group0, 2/m for group1
        """
        super().__init__()
        self.encoder = encoder
        # Register as buffer so it moves with model to device
        self.register_buffer("direction_vector", direction_vector)
        self.sign = sign
        self.scaling = scaling

    def forward(self, x):
        """
        Forward pass: encoder -> flatten -> project onto direction

        Returns:
            scalar tensor of shape [batch_size] representing each image's contribution
        """
        # Get embeddings from encoder
        embeddings = self.encoder(x)

        # Flatten embeddings
        embeddings = embeddings.view(embeddings.size(0), -1)

        # Project onto direction vector: contribution = sign * scaling * (embedding · D)
        # This represents how much each image contributes to the test statistic
        contribution = self.sign * self.scaling * torch.sum(embeddings * self.direction_vector, dim=1)

        return contribution


class LRPExplainer:
    """
    LRP-based explainer for statistical test using zennit library.

    This class provides an interface similar to GradCAM for computing LRP attributions
    that explain which image regions contribute to the test statistic.
    """

    def __init__(
        self,
        encoder,
        target_layer=None,
        relu=True,
        device="cuda",
        composite_type="epsilon_plus_flat",
        lrp_epsilon=1e-6,
        lrp_gamma=0.25,
        input_low=0.0,
        input_high=1.0,
    ):
        """
        Args:
            encoder: The feature extraction model
            target_layer: Target layer name (kept for API compatibility, not used in LRP)
            relu: Whether to apply ReLU to final attributions
            device: Device to run on
            composite_type: Type of LRP composite ('epsilon_gamma_box', 'epsilon_plus_flat', 'epsilon_alpha2beta1')
            lrp_epsilon: Epsilon value for numerical stability
            lrp_gamma: Gamma value for gamma rule
            input_low: Lower bound for input normalization (for ZBox rule)
            input_high: Upper bound for input normalization (for ZBox rule)
        """
        self.encoder = encoder
        self.device = device
        self.relu = relu
        self.target_layer = target_layer  # Kept for compatibility
        self.composite_type = composite_type
        self.lrp_epsilon = lrp_epsilon
        self.lrp_gamma = lrp_gamma
        self.input_low = input_low
        self.input_high = input_high

        self.encoder.to(self.device)
        self.encoder.eval()

        # Will be set during processing
        self.model_with_projection = None
        self.composite = None
        self.current_input = None

    def _create_composite(self):
        """Create the appropriate LRP composite based on configuration."""
        canonizers = [ResNetCanonizer()]

        if self.composite_type == "epsilon_gamma_box":
            composite = EpsilonGammaBox(
                low=torch.tensor(self.input_low).to(self.device),
                high=torch.tensor(self.input_high).to(self.device),
                epsilon=self.lrp_epsilon,
                gamma=self.lrp_gamma,
                canonizers=canonizers,
            )
        elif self.composite_type == "epsilon_plus":
            composite = EpsilonPlus(epsilon=self.lrp_epsilon, canonizers=canonizers)
        elif self.composite_type == "epsilon_plus_flat":
            #print(f"epsilon: {self.lrp_epsilon}")
            composite = EpsilonPlusFlat(epsilon=self.lrp_epsilon, canonizers=canonizers)
        elif self.composite_type == "epsilon_alpha2beta1":
            composite = EpsilonAlpha2Beta1(epsilon=self.lrp_epsilon, canonizers=canonizers)
        else:
            raise ValueError(f"Unknown composite type: {self.composite_type}")

        return composite

    def forward(self, x):
        """
        Forward pass through encoder to get embeddings.
        Stores input for later LRP computation.

        Args:
            x: Input images [batch_size, channels, height, width]

        Returns:
            embeddings: [batch_size, embed_dim]
        """
        self.current_input = x
        self.image_size = x.size(-1)

        with torch.no_grad():
            embeddings = self.encoder(x)
            embeddings = embeddings.view(embeddings.size(0), -1)

        return embeddings

    def compute_attributions(self, x, direction_vector, sign=1.0, scaling=1.0):
        """
        Compute LRP attributions for given images.

        Args:
            x: Input images [batch_size, channels, height, width]
            direction_vector: D = mean_group0 - mean_group1 [embed_dim]
            sign: +1 for group0, -1 for group1
            scaling: 2/n for group0, 2/m for group1

        Returns:
            attributions: [batch_size, 1, height, width]
        """
        # Ensure input requires gradient
        x = x.detach().clone().requires_grad_(True)

        # Create model that projects embeddings onto direction
        model_with_projection = EncoderWithProjection(self.encoder, direction_vector, sign=sign, scaling=scaling).to(
            self.device
        )

        # Create composite
        composite = self._create_composite()

        # Use zennit's Gradient attributor with LRP rules
        attributor = Gradient(model=model_with_projection, composite=composite)

        with attributor:
            # Forward pass - output is [batch_size] with one scalar per image
            output = model_with_projection(x)

            # Compute gradient w.r.t. input using LRP rules
            # Use grad_outputs to backprop from each sample independently
            (relevance,) = torch.autograd.grad(
                output.sum(),
                x,  # [batch_size, channels, height, width]
                create_graph=False,
            )

            # Sum over color channels to get spatial attribution [batch_size, 1, H, W]
            attributions = relevance.sum(dim=1, keepdim=True)

        return attributions

    def generate(self):
        """
        Generate heatmap from stored attributions.
        Called after compute_attributions to format output.

        Returns:
            heatmap: [batch_size, 1, image_size, image_size]
        """
        if not hasattr(self, "stored_attributions"):
            raise RuntimeError("Must call compute_attributions before generate()")

        heatmap = self.stored_attributions

        # Apply ReLU if requested (keep only positive relevance)
        if self.relu:
            heatmap = F.relu(heatmap)
        else:
            heatmap = heatmap  # .abs()

        # Ensure correct size (upsample/downsample if needed)
        if heatmap.size(-1) != self.image_size:
            heatmap = F.interpolate(heatmap, (self.image_size, self.image_size), mode="bilinear", align_corners=False)

        return heatmap


class LRPWrapper:
    """
    High-level wrapper for LRP explanations of the statistical test.
    Provides an interface compatible with GradCAM for easy integration.
    """

    def __init__(
        self,
        encoder,
        target_layer=None,
        relu=True,
        device="cuda",
        composite_type="epsilon_plus_flat",
        lrp_epsilon=1e-6,
        lrp_gamma=0.25,
        input_low=0.0,
        input_high=1.0,
    ):
        """
        Args:
            encoder: The feature extraction model
            target_layer: Target layer name (kept for API compatibility)
            relu: Whether to apply ReLU to final attributions
            device: Device to run on
            composite_type: Type of LRP composite
            lrp_epsilon: Epsilon value for numerical stability
            lrp_gamma: Gamma value for gamma rule
            input_low: Lower bound for input normalization
            input_high: Upper bound for input normalization
        """
        self.explainer = LRPExplainer(
            encoder=encoder,
            target_layer=target_layer,
            relu=relu,
            device=device,
            composite_type=composite_type,
            lrp_epsilon=lrp_epsilon,
            lrp_gamma=lrp_gamma,
            input_low=input_low,
            input_high=input_high,
        )
        self.model = encoder
        self.device = device

    def forward(self, x):
        """Forward pass to get embeddings."""
        return self.explainer.forward(x)

    def backward(self, statistic):
        """
        Backward pass - kept for API compatibility with GradCAM.
        For LRP, the actual computation happens in compute_attributions.
        """
        # Store statistic for reference, but LRP doesn't use gradients
        self.statistic = statistic

    def compute_attributions_for_batch(self, x, direction_vector, sign=1.0, scaling=1.0):
        """
        Compute LRP attributions for a batch of images.

        Args:
            x: Input images
            direction_vector: D = mean_group0 - mean_group1
            sign: +1 for group0, -1 for group1
            scaling: 2/n for group0, 2/m for group1

        Returns:
            attributions: [batch_size, 1, height, width]
        """
        attributions = self.explainer.compute_attributions(x, direction_vector, sign, scaling)

        # Store for generate() method
        self.explainer.stored_attributions = attributions

        return attributions

    def generate(self):
        """Generate final heatmap."""
        return self.explainer.generate()
