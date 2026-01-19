import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage as ndi
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import argparse

### In this function we control that the path is inside the brain

from embeddingtest import *


# first function gives roi

def make_foreground_mask(img):
    """This function makes a binary mask of the foreground (brain)"""
    try:
        from skimage.filters import threshold_otsu
        t = threshold_otsu(img)
        m = img > t
    except Exception:
        # fallback if scikit-image isn't available
        m = img > np.percentile(img, 60)

    m = ndi.binary_opening(m, structure=np.ones((3,3)))
    m = ndi.binary_fill_holes(m)

    # keep largest connected component
    lab, n = ndi.label(m)
    if n > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        m = lab == np.argmax(counts)
    return m


def bbox_from_mask(roi_mask, thresh=0.5):
    """This function makes a bounding box from the roi mask"""
    m = roi_mask > thresh                 # works for float masks too
    if not m.any():
        raise ValueError("ROI mask is empty.")
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) # inclusive
    x0, x1 = int(xs.min()), int(xs.max()) # inclusive
    return y0, y1, x0, x1

def add_artifact_in_bbox(img, roi_mask, kind="blur", patch_size=32, sigma=8.0,
                         zero_value=0.0, rng=42, margin=0, clamp=True):
    """
    Place a square artifact fully inside the *bounding box* of roi_mask.
    Holes in the mask are ignored; only the box limits the placement.

    margin: keep this many pixels away from the bbox edges (optional).
    clamp:  if True, shrink patch_size to fit the bbox if needed.
    """
    H, W = img.shape
    rng = np.random.default_rng(rng)

    y0, y1, x0, x1 = bbox_from_mask(roi_mask)
    # shrink box by 'margin' (but keep valid)
    y0m = min(max(y0 + margin, 0), y1)
    x0m = min(max(x0 + margin, 0), x1)
    y1m = max(min(y1 - margin, H - 1), y0m)
    x1m = max(min(x1 - margin, W - 1), x0m)

    box_h = y1m - y0m + 1
    box_w = x1m - x0m + 1

    ps = int(patch_size)
    if clamp:
        ps = max(1, min(ps, box_h, box_w))
    if ps > box_h or ps > box_w:
        raise ValueError(
            f"Patch {patch_size}×{patch_size} doesn't fit in bbox {box_h}×{box_w}. "
            "Use smaller patch_size or set clamp=True."
        )

    # sample top-left so the patch fits entirely inside the bbox
    y_min = y0m
    y_max = y1m - ps + 1
    x_min = x0m
    x_max = x1m - ps + 1

    y = int(rng.integers(y_min, y_max + 1))
    x = int(rng.integers(x_min, x_max + 1))

    out = img.copy()
    patch_mask = np.zeros_like(img, dtype=np.uint8)
    patch_mask[y:y+ps, x:x+ps] = 1

    if kind == "blur":
        out[y:y+ps, x:x+ps] = gaussian_filter(out[y:y+ps, x:x+ps], sigma=sigma)
    elif kind == "zero":
        out[y:y+ps, x:x+ps] = zero_value
    else:
        raise ValueError("kind must be 'blur' or 'zero'")

    return out, patch_mask


def corrupt_images_with_artifacts(imgs, kind="blur", patch_size=32, sigma=8.0,
                         zero_value=0.0, rng=42, margin=10, clamp=True):
    """imgs: is a numpy array of shape [N,H,W]"""
    corrupted_images = []
    corrupted_masks = []
    for i in range(len(imgs)):
        img = imgs[i]
        roi_mask = make_foreground_mask(img)
        corrupted_img, mask = add_artifact_in_bbox(img, roi_mask, kind=kind, patch_size=patch_size,
                                               sigma=sigma, zero_value=zero_value,
                                               rng=rng, margin=margin, clamp=clamp)
        corrupted_images.append(corrupted_img)
        corrupted_masks.append(mask)

    # save corrupted images and masks as numpy arrays
    return np.array(corrupted_images), np.array(corrupted_masks)

def save_npy(img_corrupted, mask_corrupted, args):
    """Save a NumPy array to a .npy file."""
    path_corrupted = os.path.join(args.root_dir, 'AdniGithub','adni_results','corrupted', f'gr0_4591_{args.kind}_ps{args.ps}_sigma{args.sigma}.npz')
    os.makedirs(os.path.dirname(path_corrupted), exist_ok=True)
    np.savez_compressed(
    path_corrupted,
    image=img_corrupted.astype(np.float32),
    mask=mask_corrupted.astype(np.uint8),
    kind=str(args.kind),
    patch_size=int(args.ps),
    sigma=float(args.sigma)
)


parser = argparse.ArgumentParser(description='Preparing datasets for finetuning!')
parser.add_argument('--root_dir', type=str, default= '/path/to/data')
parser.add_argument('--ps', type=int, default=32, help='Size of path to corrupted area')
parser.add_argument('--sigma', type=float, default=0.8, help='the degree of corruption')
parser.add_argument('--kind', type=str, default='zero', help='If we want to blur or zero outsquares')
parser.add_argument('--save', type=bool, default=True, help='If we want to save the corrupted images or not')


if __name__=="__main__":
    
    args = parser.parse_args()
    # make a path to numpy group that we wnat to create corrupted images for it
    gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images', f'gr0_4591.npy')
    gr0 = np.load(gr0_dir)
    gr0_corr, gr0_cor_mask = corrupt_images_with_artifacts(imgs=gr0, kind=args.kind, patch_size=args.ps,
                                                            sigma=args.sigma,zero_value=0.0, rng=42,
                                                              margin=10, clamp=True)
    
    if args.save:
        save_npy(gr0_corr, gr0_cor_mask, args)
        print('corrupted images are saved!')
    


