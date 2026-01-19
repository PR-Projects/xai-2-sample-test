# Here I want to prepare data for finetuning a pre-trained model on ADNI
# This is a Binary classification problem
# We prepare two groups for this task: belonging to group 1 vs group0
# Where group1, group0 are defined as: control group vs AD group
# Or based on presence of hippocampal atrophy or not
# Split data to: train, val, test sets  
# We use train set for fine-tuning model and evaluate on val the test on test set 
# I determine the split for train, val and test sete based on desired proportion that we want 
# For both models I need to repeat chennels and I use ImgeNet statistics for normalisation 
# I define percentage of data that I want to use for training, validation and test set 


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.eval_utils import insert_grey_circle

from data import *

rng = np.random.default_rng(42)  # reproducible results


# This creats a PyTorch Dataset from NumPy arrays
class NpyImageDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray,
                 to_three_channels: bool = True,
                 imagenet_norm: bool = True):
        """
        X_np: (N, H, W) or (N, 1, H, W) NumPy array (uint8 or float)
        y_np: (N,) labels
        """
        self.X = X_np
        self.y = y_np.astype(np.int64)
        self.to_three = to_three_channels
        self.imagenet_norm = imagenet_norm

        if imagenet_norm:
            # Keep these on CPU; we’ll move to GPU per-batch later.
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()      # (H,W) or (1,H,W)
        if x.ndim == 2:
            x = x.unsqueeze(0)                         # -> (1,H,W)
        # If data are uint8 images 0..255, scale to [0,1]
        if x.max() > 1.0:
            x = x / 255.0

        if self.to_three:
            x = x.repeat(3, 1, 1)                      # -> (3,H,W)

        if self.imagenet_norm:
            # normalize per channel (on CPU)
            x = (x - self.mean) / self.std

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
# This function hanels both uint8 images and float images 
# Only robust_float_norm is added to hanle float images properly    
class NpyImageDataset2(Dataset):
    def __init__(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        to_three_channels: bool = True,
        imagenet_norm: bool = True,
        robust_float_norm: bool = True   # 🔹 NEW
    ):
        """
        X_np: (N,H,W) or (N,1,H,W)
              dtype can be uint8 or float
        y_np: (N,) labels
        """
        self.X = X_np
        self.y = y_np.astype(np.int64)

        #print(f'NpyImageDataset2 was used!')

        #print(f'type X: {self.X.dtype}')

        self.to_three = to_three_channels
        self.imagenet_norm = imagenet_norm
        self.robust_float_norm = robust_float_norm
        #print(f'float_norm:{self.robust_float_norm}')

        if imagenet_norm:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def _scale_to_01_float(self, x: torch.Tensor):
        """Robust scaling for float MRI images"""
        lo = torch.quantile(x, 0.01)
        hi = torch.quantile(x, 0.99)
        x = torch.clamp(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-8)

    def __getitem__(self, idx):
        x_np = self.X[idx]
        x = torch.from_numpy(x_np).float()   # (H,W) or (1,H,W)
        #print(f'type x: {x.dtype}')
        
        if x.ndim == 2:
            x = x.unsqueeze(0)               # -> (1,H,W)

        # 🔹 Correct intensity handling
        if x_np.dtype == np.uint8:
            # Old pipeline (safe)
            x = x / 255.0
        else:
            # New pipeline (float MRI)
            if self.robust_float_norm:
                #print(f'robust float normalization is applied')
                x = self._scale_to_01_float(x)

        # Channel handling
        if self.to_three and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)            # -> (3,H,W)

        # ImageNet normalization (for ResNet)
        if self.imagenet_norm:
            x = (x - self.mean) / self.std

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def make_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4,robust_float_norm=False):
    # robust_float_norm is added to hanle float images properly
    # robust_float_norm is passed to NpyImageDataset2 class
    # robust_float_norm is set to True when float images are used
    train_ds = NpyImageDataset2(X_train, y_train, to_three_channels=True, imagenet_norm=True, robust_float_norm=robust_float_norm)
    val_ds   = NpyImageDataset2(X_val,   y_val,   to_three_channels=True, imagenet_norm=True, robust_float_norm=robust_float_norm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# This function splits a numpy array into train, val, test sets with 60%, 20%, 20% proportions
def split_60_20_20_per_class(arr1,arr2,rng):
    "assuming two arrays have the same length"
    n = len(arr1)
    k_test = n // 5      # exact 20% via integer division
    k_val  = n // 5      # another 20%
    k_train = n - k_test - k_val  # remaining ~60% (exact if n%5==0)

    idx = rng.permutation(n)
    print(f'idx:{idx} in split_60_20_20_per_class')
    train0 = arr1[idx[:k_train]]
    val0   = arr1[idx[k_train:k_train+k_val]]
    test0  = arr1[idx[k_train+k_val:]]
    train1 = arr2[idx[:k_train]]
    val1   = arr2[idx[k_train:k_train+k_val]]
    test1  = arr2[idx[k_train+k_val:]]
    return train0, val0, test0, train1, val1, test1

# This function packs two arrays a0 and a1 with their corresponding labels: label0 and label1
def pack_and_shuffle(a0, a1, label0, label1, rng):
    X = np.concatenate([a0, a1], axis=0)
    y = np.concatenate([
        np.full(len(a0), label0, dtype=int),
        np.full(len(a1), label1, dtype=int),
    ])
    p = rng.permutation(len(y))
    return X[p], y[p]

def add_grey_circle_artifact_all_samples(group_np):
        # Apply grey circle to each sample in group_2
        circle_center_offset=-20
        circle_radius=20 
        circle_grey_value=128
        print(f"Applying grey circle to each sample in group_2...")
        for i in range(len(group_np)):
            # Get image dimensions
            height, width = group_np[i].shape
            # Insert grey circle at center with specified radius and offset
            center_y = height // 2 + circle_center_offset
            center_x = width // 2 + circle_center_offset
            center = (center_y, center_x)
            group_np[i] = insert_grey_circle(group_np[i], center, circle_radius, circle_grey_value)
        return group_np


# Dataset has (4591) samples 
def split_data(args):
    # first we load the dataset (either: gr0 vs gr1 or gr1 vs gr1_corrupted)
    # first I will take two groups of data: group 0 and group 1
    # load the numpy array for group0 and group1 (these two are two groups that are not corrupted)
    # we should replace them with corrupted images for group 1 if we want to use corrupted images
    # gives always label 0 to group 0 and label 1 to group 1 
    if not args.corrupted:
        if args.task=='hippo':
            gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images', f'gr0_4591.npy')
            gr1_dir = os.path.join(args.root_dir,'AdniGithub','adni_results','images', f'gr1_4592.npy')
            gr0 = np.load(gr0_dir)
            gr1 = np.load(gr1_dir)
            print(f'len gr0:{gr0.shape}', f'len gr1:{gr1.shape}')
            # make two groups of data from the same length 
            min_len = min(len(gr0), len(gr1))
            gr0, gr1 = gr0[:min_len], gr1[:min_len]
        elif args.task=='CN_AD':
            gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images','float', 'CN.npy')
            gr1_dir = os.path.join(args.root_dir,'AdniGithub','adni_results','images','float','AD.npy')
            gr0 = np.load(gr0_dir)
            gr1 = np.load(gr1_dir)
            print(f'len gr0:{gr0.shape}', f'len gr1:{gr1.shape}')
            # make two groups of data from the same length 
            min_len = min(len(gr0), len(gr1))
            gr0, gr1 = gr0[:min_len], gr1[:min_len]

    elif args.corrupted:
        if args.deg=='bl32':
            print('Using corrupted images with blurring with ps=32, sigma=0.8 for group 1')
            gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images', f'gr0_4591.npy')
            gr1_corrupted_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','corrupted', f'gr0_4591_blur_ps32_sigma0.8.npz')
            gr0 = np.load(gr0_dir)
            data = np.load(gr1_corrupted_dir)
            gr1 = data['image']
            min_len = min(len(gr0), len(gr1))
            gr0, gr1 = gr0[:min_len], gr1[:min_len]

        if args.deg=='zer32':
            print('Using corrupted images with zeroing with ps=32 for group 1')
            gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images', f'gr0_4591.npy')
            gr1_corrupted_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','corrupted', f'gr0_4591_zero_ps32_sigma0.8.npz')
            gr0 = np.load(gr0_dir)
            data = np.load(gr1_corrupted_dir)
            gr1 = data['image']
            min_len = min(len(gr0), len(gr1))
            gr0, gr1 = gr0[:min_len], gr1[:min_len]

        if args.deg=='circ':
            gr0_dir = os.path.join(args.root_dir, 'AdniGithub','adni_results','images', f'gr0_4591.npy')
            gr0 = np.load(gr0_dir)
            gr1 = copy.deepcopy(gr0)
            gr1 = add_grey_circle_artifact_all_samples(gr1)
            min_len = min(len(gr0), len(gr1))
            gr0, gr1 = gr0[:min_len], gr1[:min_len]

        

    # now we make the split for train, val and test sets
    train0, val0, test0, train1, val1, test1 = split_60_20_20_per_class(gr0, gr1, rng)
    print('creating train, val and test set from group1, group2')
    #train1, val1, test1 = split_60_20_20_per_class(gr1, rng)  
    
    # Here we make the packing and shuffling of the data
    # Shape of train, val sets: [N, 256, 256], intensies: [0-255]
    X_train, y_train = pack_and_shuffle(train0, train1, 0, 1, rng)   
    X_val,   y_val   = pack_and_shuffle(val0,   val1,   0, 1, rng)  


    # Now we save test_set for creating heat-maps and test-statistic 
    # Build the path
    if args.task=='hippo':
        # Handel hippocampal atrophy task
        test_dir = Path(args.root_dir) / "AdniGithub" /"adni_results" /"split"/ "test" / str(args.corrupted) / str(args.deg) 
    else:
        # Handel CN_AD task
        test_dir = Path(args.root_dir) / "AdniGithub" /"adni_results" /"split"/ "test" / str(args.corrupted) / str(args.deg) / args.task / 'float'
    test_dir.mkdir(parents=True, exist_ok=True)  # create folders if missing
    # Save both arrays into a single compressed file
    out_path = test_dir / "test_split.npz"
    np.savez_compressed(out_path, test0=test0, test1=test1)
    print(f"Saved: {out_path.resolve()}")


    # Making directory for train nad val sets
    if args.task=='hippo':
        # Handel hippocampal atrophy task
        train_dir = Path(args.root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted) / str(args.deg)
    else:
        # Handel CN_AD task, other tasks like MCI_AD can be added here
        train_dir = Path(args.root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted) / str(args.deg) / args.task / 'float'
    train_dir.mkdir(parents=True, exist_ok=True)  # create folders if missing
    # Save both arrays into a sigle cpmpressed file 
    train_path = train_dir / "train_split.npz"
    
    np.savez_compressed(
    train_dir / "train_val_splits.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val)
    
    print('train and val sets are saved!')



parser = argparse.ArgumentParser(description='Preparing datasets for finetuning!')
parser.add_argument('--root_dir', type=str, default= '/path/to/data')
parser.add_argument('--corrupted', type=str, default=True, help='Use corrupted images for group 1')
parser.add_argument('--deg', type=str, default='circ', help='Degree of corruption: 4 or 8', choices=('bl32', 'zer32','circ','None'))
parser.add_argument('--task', type=str, default='CN_AD', help='which task we want to use:hippo, CN_AD,MCI_AD', choices=('CN_AD','hippo','MCI_AD'))

if __name__ == "__main__":
    args = parser.parse_args()
    #split_data(args)
    #print('Data splitting is done!')

    outdir = Path('/path/to/data/AdniGithub/adni_results/split/train/False/None/CN_AD/float')

    data = np.load(outdir / "train_val_splits.npz")
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4, robust_float_norm=True)

    train_batch0 = next(iter(train_loader))

    print(len(train_batch0))

    val_batch0 = next(iter(val_loader))

    #print(len(val_batch0))

    #imgs, labels = next(iter(train_loader))
    #print(imgs.shape)                 # e.g., torch.Size([32, 3, 256, 256])
    #print(imgs.mean().item(), imgs.std().item())
    #print(labels.shape, labels[:8])         # e.g., torch.Size([32]) tensor([0, 1, 0, 0, 1, 1, 0, 0])



   

    

    








