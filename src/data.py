import torch
import numpy as np
import pandas as pd
import os

import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize

from pathlib import Path
import matplotlib.pyplot as plt


class AdniMRIDataset2D(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels["archive_fname"].iloc[idx])
        # print(img_path)
        image = self.read_image(img_path)
        label = self.img_labels["group"].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def read_image(self, path):
        img = nib.load(path).get_fdata().astype(np.uint8)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # HW -> HWC
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img
    

## Stratifyng by severity levels
class AdniMRIDatasetFull(Dataset):
    def __init__(self, annotations_file, img_dir=None, transform=None):
        df = pd.read_csv(annotations_file)
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        print(f'idx:{idx}')
        rel_path = self.img_labels["filepath_MNIlin"].iloc[idx]
        group = self.img_labels["Group"].iloc[idx]   # <-- CN, MCI, AD

        if self.img_dir:
            img_path = os.path.join(self.img_dir, rel_path)
        else:
            img_path = rel_path
        
        # load 3D MRI
        image = nib.load(img_path).get_fdata().astype(np.float32)
        # reorder axes: (coronal, sagittal, axial)
        # Middle coronal slice
        y_mid = image.shape[1] // 2
        slice_data = image[:, y_mid, :]

        # Rotate to match typical radiology visual orientation
        slice_data = np.rot90(slice_data)
        # optional: normalize 0–1 (similar to matplotlib auto-scaling)
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        
        # Resize to 256x256
        slice_data = resize(slice_data, (256, 256), preserve_range=True, anti_aliasing=True)
        if self.transform:
            slice_data = self.transform(slice_data)

        slice_data = slice_data[np.newaxis, ...] # add channel dimension =>(1,H,W)

        return slice_data, group
    



def save_severity_groups(dataset, out_dir="../adni_results/images", max_per_class=1000):
    os.makedirs(out_dir, exist_ok=True)

    CN, MCI, AD = [], [], []

    print(f'directory for saving: {out_dir}')

    i = 0

    for img, label in dataset:
        i += 1
        print(f'i:{i}')
        if label == "CN":
            CN.append(img)
        elif label == "MCI":
            MCI.append(img)
        elif label == "AD":
            AD.append(img)

        if (len(CN) >= max_per_class and
            len(MCI) >= max_per_class and
            len(AD) >= max_per_class):
            break

    if CN:
        np.save(os.path.join(out_dir, "CN.npy"), np.stack(CN))
        print("Saved CN.npy:", len(CN))
    if MCI:
        np.save(os.path.join(out_dir, "MCI.npy"), np.stack(MCI))
        print("Saved MCI.npy:", len(MCI))
    if AD:
        np.save(os.path.join(out_dir, "AD.npy"), np.stack(AD))
        print("Saved AD.npy:", len(AD))

if __name__ == "__main__":
    annotations_file = "./file_local.csv"
    dataset = AdniMRIDatasetFull(annotations_file)
    out_dir = "../"
    save_severity_groups(dataset, out_dir)

