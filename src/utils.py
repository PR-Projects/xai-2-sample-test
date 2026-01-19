import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


class ZeroGradientError(Exception):
    """Custom exception to handle cases where the gradient is zero."""
    pass


def save_cam_with_alpha(image, gcam, alpha=0.5):

    # Convert grayscale image to 3 channels if needed
    if len(image.shape) == 2:  # Grayscale image (H, W)
        image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)

    # Normalize the Grad-CAM values to [0, 1]
    gcam_min = np.min(gcam)
    gcam_max = np.max(gcam)

    try:
        if gcam_max == gcam_min:  # If all values are zero, raise an error
            raise ZeroGradientError(
                "Gradient map contains only zero values, cannot overlay.")
        # Normalize gradient map
        gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
    except ZeroGradientError as e:
        print(f"Error: {e}")
        # Handle the error (for example, return the original image or skip processing)
        return image, image  # Return the original image if error occurs

    # Resize Grad-CAM to match the image dimensions (224x224)
    # Get height and width (height, width) from image shape
    h, w = image.shape[:2]
    gcam_resized = np.array(Image.fromarray(
        gcam).resize((w, h), Image.BILINEAR))

    # Apply a colormap (similar to cv2.applyColorMap)
    # Apply colormap and select RGB channels
    gcam_colored = plt.cm.jet(gcam_resized)[:, :, :3] * 255
    gcam_colored = gcam_colored.astype(np.uint8)

    # Add Grad-CAM on top of the original image using alpha blending
    heatmap = gcam_colored.astype(np.float64)

    # checking dimension of image and heatmaps
    # print(f'heatmap:{heatmap.shape}')
    # print(f'image:{image.shape}')

    overlaid_image = (alpha * heatmap + (1 - alpha) *
                      image.astype(np.float64)).astype(np.uint8)

    return image, overlaid_image