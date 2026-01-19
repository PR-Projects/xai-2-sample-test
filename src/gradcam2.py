import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# from attention import SpatialAttention, ChannelAttention


class ProbBase(object):
    def __init__(self, model, target_layer, relu, device, attention=None):
        self.model = model
        self.device = device
        self.relu = relu
        self.attention = (
            SpatialAttention().to(self.device)
            if attention == "spatial"
            else (ChannelAttention().to(self.device) if attention == "channel" else None)
        )
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        # gets filled by the backward hook => same as self.gradients
        self.outputs_backward = OrderedDict()
        # gets filled by the forward hook => same as self.activations
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    def forward(self, x):
        self.image_size = x.size(-1)
        # it gives the embedding before the final fc layer: [2048]
        # print(f'Input shape in GradCAM: {x.shape}')
        # print(f'self.model inside GradCAM:{self.model}')
        self.embed = self.model(x)
        # print(f'Embedding was extracted: {self.embed.shape}')
        return self.embed

    def backward(self, statistic):
        self.model.zero_grad()
        self.statistic = statistic.to(self.device)
        # print("Before backward, test statistic grad_fn:", self.statistic.grad_fn)
        self.statistic.backward(retain_graph=True)
        # checking if gradients are computed for parameters
        # for name, param in self.model.named_parameters():
        # if param.grad is not None:
        # print(f"Gradient for {name} after backward pass exists.")
        # else:
        # print(f"No gradient computed for {name}")

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid layer name: {target_layer}")


class GradCAM(ProbBase):
    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            # print("backward: ", grad_out[0].sum(), grad_out[0].shape)
            self.outputs_backward[id(module)] = grad_out[0]

        def func_f(module, input, f_output):
            # print("forward: ", f_output.sum(), f_output.shape)
            self.outputs_forward[id(module)] = f_output

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_full_backward_hook(func_b)
                module.register_forward_hook(func_f)

    def generate(self, flip_sign=False):
        # grads: dY/dA, activations: A
        # grads: [8,2048,8,8], when bs=8, C=2048, H=W=8 for resnet50 last conv layer
        # activations: [8,2048,8,8] bs=8, C=2048, H=W=8 for resnet50 last conv layer
        grads = self.get_conv_outputs(self.outputs_backward, self.target_layer)  # (B,C,H,W)
        activations = self.get_conv_outputs(self.outputs_forward, self.target_layer)  # (B,C,H,W)
        print(f"\nGetting gradient in generate:{grads.shape}")
        print(f"\nGetting activations in generate:{activations.shape}")

        # GAP over spatial dims -> weights (B,C,1,1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum
        # cam = [8, 1, 8, 8]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        print(f"\nGetting cam before relu in generate:{cam.shape}")

        # Flip sign if requested (for group 1 in MMD test statistic)
        if flip_sign:
            cam = -cam

        # ReLU or abs
        cam = F.relu(cam) if self.relu else cam  # .abs()

        #cam_pos = torch.clamp(cam, min=0.0) # Ensure non-negativity
        #cam_neg = torch.clamp(cam, max=0.0) # keeps values < 0, sets others to 0

        # Upsample to input size
        cam = F.interpolate(cam, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return cam


class GradCAMPlusPlus(ProbBase):
    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0]

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_full_backward_hook(func_b)
                module.register_forward_hook(func_f)

    def generate(self, flip_sign=False):
        # A is activations and g is gradients
        # A: [8,2048,8,8], g: [8,2048,8,8] when bs=8, C=2048, H=W=8 for resnet50 last conv layer
        # C: here is the number of channels in explainability method (2048 for resnet50 last conv layer)
        print(f"heatmaps from GradCAM++ are generated")
        A = self.get_conv_outputs(self.outputs_forward, self.target_layer)  # (B,C,H,W)
        g = self.get_conv_outputs(self.outputs_backward, self.target_layer)  # (B,C,H,W)

        print(f"A:{A.shape}")
        print(f"g:{g.shape}")

        g2 = g * g
        g3 = g2 * g
        sumA = A.sum(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        eps = 1e-8
        denom = 2.0 * g2 + (sumA * g3) + eps
        alpha = g2 / denom
        
        g = F.relu(g) if self.relu else g # we use this command when we do not want to apply relu

        weights = (alpha * g).sum(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        cam = (weights * A).sum(dim=1, keepdim=True)  # (B,1,H,W)

        # Flip sign if requested (for group 1 in MMD test statistic)
        if flip_sign:
            cam = -cam

        cam = F.relu(cam) if self.relu else cam # .abs()

        cam = F.interpolate(cam, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return cam


## This method does not use Gradients for hetamp visualisations
class ScoreCAM(ProbBase):
    def set_hook_func(self):
        def func_f(module, inp, out):
            self.outputs_forward[id(module)] = out

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(func_f)

    @torch.no_grad()
    def generate(self, x, target_class=None, max_channels=None, normalize=True):
        """
        x: input tensor (B,3,H,W) used to form masked inputs
        target_class: int or None; if None, uses argmax per sample
        max_channels: int or None; to subsample channels for speed
        """
        A = self.get_conv_outputs(self.outputs_forward, self.target_layer)  # (B,C,h,w)
        B, C, h, w = A.shape

        # Decide which channels to use (optional speed-up)
        if max_channels is not None and max_channels < C:
            idx = torch.linspace(0, C - 1, steps=max_channels).long().to(A.device)
            A = A[:, idx]  # (B, maxC, h, w)
            C = A.shape[1]

        # Prepare weights per-channel
        weights = torch.zeros((B, C, 1, 1), device=A.device)

        # Precompute logits on original input to get target class if needed
        base_logits = self.model(x)  # assumes logits; adapt if your model returns embeddings
        if target_class is None:
            target_class = base_logits.argmax(dim=1)  # (B,)

        # For each channel, build a mask and get score as weight
        for c in range(C):
            # Upsample that channel to input size
            mask = A[:, c : c + 1, :, :]
            mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            # Normalize to [0,1] to avoid killing the image
            if normalize:
                m_min = mask.amin(dim=(2, 3), keepdim=True)
                m_max = mask.amax(dim=(2, 3), keepdim=True)
                mask = (mask - m_min) / (m_max - m_min + 1e-8)

            # Mask the input
            x_masked = x * mask

            # Forward to get scores for the target class
            logits = self.model(x_masked)  # (B, num_classes)
            # Take target class scores and ensure positivity (original Score-CAM uses raw scores after softmax/logits)
            score = logits[torch.arange(B), target_class].view(B, 1, 1, 1)

            weights[:, c : c + 1, :, :] = score

        # Weighted sum of original (low-res) activations (as in paper)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (B,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return cam


# It does not have GAP, like GradCAM
class LayerCAM(ProbBase):
    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0]

        def func_f(module, inp, out):
            self.outputs_forward[id(module)] = out

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_full_backward_hook(func_b)
                module.register_forward_hook(func_f)

    def generate(self, flip_sign=False):
        print(f"heatmaps from LayerCAM are generated")
        A = self.get_conv_outputs(self.outputs_forward, self.target_layer)  # (B,C,H,W)
        g = self.get_conv_outputs(self.outputs_backward, self.target_layer)  # (B,C,H,W)

        print(f'relu in layercam: {self.relu}')
        
        g = F.relu(g) if self.relu else g
        cam = (g * A).sum(dim=1, keepdim=True)  # (B,1,H,W)

        # Flip sign if requested (for group 1 in MMD test statistic)
        if flip_sign:
            cam = -cam

        #cam = F.relu(cam) if self.relu else cam # .abs()
        # cam_pos = torch.clamp(cam, min=0.0) # instead of applying relu, lets take pos grads
        cam = F.interpolate(cam, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return cam
