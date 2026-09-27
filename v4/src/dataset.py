import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from skimage import exposure
try:
    from . import config
    from .augmentations import get_v4_transforms
except ImportError:
    import config
    from augmentations import get_v4_transforms

def apply_clahe_preprocessing(img: Image.Image) -> Image.Image:
    """
    Biological contrast enhancement for fish scales.
    Sharpens fine circuli and radii while blending to avoid amplifying background noise.
    """
    if not getattr(config, "USE_CLAHE", False):
        return img
    arr = np.array(img, dtype=np.float32) / 255.0
    res = exposure.equalize_adapthist(arr, clip_limit=getattr(config, "CLAHE_CLIP", 0.015))
    blended = (np.clip(0.5 * arr + 0.5 * res, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(blended)


class FishScaleDatasetV4(Dataset):
    """
    Next-Generation Fish Scale Dataset (v4) with Few-Shot and Long-Tail Stratification.
    """
    def __init__(self, root_dir: str, split: str = "train", seed: int = 42):
        self.root_dir = root_dir
        self.split = split
        self.transform = get_v4_transforms(config.IMAGE_SIZE, is_train=(split == "train"))
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dataset directory not found: {root_dir}")
            
        raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        
        # 1. Count valid images per class
        class_counts = []
        for class_name in raw_classes:
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            valid_files = [f for f in files if os.path.splitext(f.lower())[1] in supported_exts]
            if len(valid_files) > 0:
                class_counts.append((class_name, len(valid_files), valid_files))
                
        # 2. Sort by count descending and select target number of classes
        class_counts.sort(key=lambda x: x[1], reverse=True)
        if config.NUM_CLASSES is not None and config.NUM_CLASSES < len(class_counts):
            selected_classes_info = class_counts[:config.NUM_CLASSES]
        else:
            selected_classes_info = class_counts

        self.classes = [info[0] for info in selected_classes_info]
        self.classes.sort()  # Alphabetical for deterministic integer label mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        rng = random.Random(seed)
        self.train_counts = [0] * len(self.classes)
        
        # 3. Few-shot aware stratified splitting (70% Train, 15% Val, 15% Test)
        for class_name in self.classes:
            info = next(info for info in selected_classes_info if info[0] == class_name)
            valid_files = info[2].copy()
            valid_files.sort()
            rng.shuffle(valid_files)
            
            n = len(valid_files)
            class_idx = self.class_to_idx[class_name]
            
            if n >= 6:
                train_end = max(1, int(0.70 * n))
                val_end = max(train_end + 1, int(0.85 * n))
            elif n >= 4:
                train_end = n - 2
                val_end = n - 1
            elif n == 3:
                train_end = 2
                val_end = 2  # 2 train, 1 test
            else: # n == 2
                train_end = 1
                val_end = 1  # 1 train, 1 val
                
            if split == "train":
                selected_files = valid_files[:train_end]
            elif split == "val":
                selected_files = valid_files[train_end:val_end] if val_end > train_end else valid_files[train_end:]
            elif split == "test":
                selected_files = valid_files[val_end:] if val_end < n else valid_files[train_end:]
            else:
                selected_files = valid_files

            # Ensure minority classes are never empty
            if len(selected_files) == 0:
                selected_files = [valid_files[0]]

            for f in selected_files:
                self.image_paths.append(os.path.join(root_dir, class_name, f))
                self.labels.append(class_idx)

            if split == "train":
                self.train_counts[class_idx] = len(selected_files)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(path) as img:
            img = img.convert('RGB')
            if config.USE_CLAHE:
                img = apply_clahe_preprocessing(img)
            tensor_img = self.transform(img)

        return tensor_img, label, self.classes[label]


def get_dataloaders():
    """
    Creates train, validation, and test DataLoader instances with balanced sampling.
    """
    train_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    val_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)
    test_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="test", seed=config.RANDOM_SEED)

    # Class-Balanced Weighted Random Sampler for minority class amplification
    train_labels = np.array(train_dataset.labels)
    class_sample_counts = np.bincount(train_labels, minlength=len(train_dataset.classes))
    
    # Class weights: w_c = 1.0 / (N_c)^0.5
    class_weights = 1.0 / np.power(np.maximum(class_sample_counts, 1), 0.5)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.train_counts, train_dataset.classes
