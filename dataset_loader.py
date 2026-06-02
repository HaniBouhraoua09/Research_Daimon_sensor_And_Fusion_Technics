"""
dataset_loader.py - PyTorch Dataset for the tactile fruit classification task.

Loads trials from dataset/material=*/trial_*/ and returns synchronized
4-modality samples. Each sample uses the peak-deformation frame of the trial.

Splits are at the TRIAL level (not frame level) to prevent data leakage.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = "dataset"

# Handle the "orrange" folder typo gracefully — map both names to "orange"
MATERIAL_NAME_FIXES = {"orrange": "orange"}

# Exclude known bad trials (e.g., missed contact events)
EXCLUDE_TRIALS = {
    ("kiwi", 24),  # trial 024 was empty
}


# ============================================================
# DATASET
# ============================================================
class TactileDataset(Dataset):
    """One sample = peak-deformation frame of one trial, all 4 modalities."""
    
    def __init__(self, trial_dirs, label_map):
        self.trial_dirs = trial_dirs
        self.label_map = label_map  # {"apple": 0, "orange": 1, "kiwi": 2}
        # Pre-cache labels and peak frames for speed
        self.labels = []
        for td in trial_dirs:
            mat_folder = os.path.basename(os.path.dirname(td)).split("=")[1]
            mat = MATERIAL_NAME_FIXES.get(mat_folder, mat_folder)
            self.labels.append(label_map[mat])
    
    def __len__(self):
        return len(self.trial_dirs)
    
    def __getitem__(self, idx):
        trial_dir = self.trial_dirs[idx]
        
        # Load all four modalities
        rawimg = np.load(os.path.join(trial_dir, "rawimg.npy"))      # (T, H, W)
        depth  = np.load(os.path.join(trial_dir, "depth.npy"))       # (T, H, W)
        defo   = np.load(os.path.join(trial_dir, "deformation.npy")) # (T, H, W, 2)
        shear  = np.load(os.path.join(trial_dir, "shear.npy"))       # (T, H, W, 2)
        
        # Find peak-deformation frame (most informative)
        defo_mag = np.linalg.norm(defo, axis=-1)  # (T, H, W)
        peak_frame = int(defo_mag.reshape(len(defo_mag), -1).max(axis=1).argmax())
        
        # Extract single peak frame from each modality
        rawimg_f = rawimg[peak_frame].astype(np.float32) / 255.0   # (H, W) in [0,1]
        depth_f  = depth[peak_frame].astype(np.float32)            # (H, W)
        defo_f   = defo[peak_frame].astype(np.float32)             # (H, W, 2)
        shear_f  = shear[peak_frame].astype(np.float32)            # (H, W, 2)
        
        # Convert to PyTorch tensors with channel-first layout
        sample = {
            'rawimg':      torch.from_numpy(rawimg_f).unsqueeze(0),         # (1, H, W)
            'depth':       torch.from_numpy(depth_f).unsqueeze(0),          # (1, H, W)
            'deformation': torch.from_numpy(defo_f).permute(2, 0, 1),       # (2, H, W)
            'shear':       torch.from_numpy(shear_f).permute(2, 0, 1),      # (2, H, W)
            'label':       torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return sample


# ============================================================
# SPLITS
# ============================================================
def build_splits(dataset_root=DATASET_ROOT, test_size=0.2, val_size=0.15, seed=42):
    """Build train/val/test splits at TRIAL level (not frame level)."""
    
    # Find all materials
    material_dirs = sorted(glob.glob(os.path.join(dataset_root, "material=*")))
    materials_raw = [os.path.basename(d).split("=")[1] for d in material_dirs]
    materials = [MATERIAL_NAME_FIXES.get(m, m) for m in materials_raw]
    label_map = {mat: i for i, mat in enumerate(sorted(set(materials)))}
    
    # Find all trials with their labels, excluding bad ones
    all_trials = []
    all_labels = []
    for mat_dir, mat_raw, mat in zip(material_dirs, materials_raw, materials):
        trials = sorted(glob.glob(os.path.join(mat_dir, "trial_*")))
        for t in trials:
            trial_num = int(os.path.basename(t).split("_")[1])
            if (mat, trial_num) in EXCLUDE_TRIALS:
                print(f"  [skip] {mat}/trial_{trial_num:03d} (excluded)")
                continue
            all_trials.append(t)
            all_labels.append(label_map[mat])
    
    # Stratified split: train+val vs test
    trainval, test, trainval_y, _ = train_test_split(
        all_trials, all_labels, test_size=test_size,
        stratify=all_labels, random_state=seed,
    )
    # Stratified split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val, _, _ = train_test_split(
        trainval, trainval_y, test_size=val_ratio,
        stratify=trainval_y, random_state=seed,
    )
    
    print(f"\nLabel map: {label_map}")
    print(f"Total trials: {len(all_trials)}")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    
    return train, val, test, label_map


# ============================================================
# DATALOADERS
# ============================================================
def get_dataloaders(dataset_root=DATASET_ROOT, batch_size=8, num_workers=2, seed=42):
    train, val, test, label_map = build_splits(dataset_root, seed=seed)
    
    train_ds = TactileDataset(train, label_map)
    val_ds   = TactileDataset(val, label_map)
    test_ds  = TactileDataset(test, label_map)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, label_map


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_map = get_dataloaders(batch_size=4)
    print(f"\nLabel map: {label_map}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, val in batch.items():
        if hasattr(val, 'shape'):
            print(f"  {key:12s}: {tuple(val.shape)}   dtype={val.dtype}")
    print(f"\nFirst batch labels: {batch['label'].tolist()}")