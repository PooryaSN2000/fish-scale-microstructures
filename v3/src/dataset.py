import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

import config
from features import extract_classical_features
from skimage import exposure
from concurrent.futures import ThreadPoolExecutor

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

def get_transforms(split: str):
    """
    Get base image preprocessing (resizing and tensor conversion).
    Heavy augmentations and normalization are moved to the GPU in train.py.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

class FishScaleDataset(Dataset):
    """
    Dataset that dynamically selects the top NUM_CLASSES most populated classes (40 classes)
    and performs an image-level Train/Val/Test split.
    """
    def __init__(self, root_dir: str, split: str = "train", seed: int = 42):
        self.root_dir = root_dir
        self.split = split
        self.transform = get_transforms(split)
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dataset directory not found: {root_dir}")
            
        raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        
        # 1. Count images per class
        class_counts = []
        for class_name in raw_classes:
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            valid_files = [f for f in files if os.path.splitext(f.lower())[1] in supported_exts]
            if len(valid_files) > 0:
                class_counts.append((class_name, len(valid_files), valid_files))
                
        # 2. Sort by count descending and pick Top NUM_CLASSES
        class_counts.sort(key=lambda x: x[1], reverse=True)
        top_classes_info = class_counts[:config.NUM_CLASSES]
        
        self.classes = [info[0] for info in top_classes_info]
        self.classes.sort()  # Alphabetical for consistent label mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        rng = random.Random(seed)
        
        # 3. Split images WITHIN each class (60% Train, 20% Val, 20% Test)
        for class_name in self.classes:
            info = next(info for info in top_classes_info if info[0] == class_name)
            valid_files = info[2]
            
            # Sort then shuffle deterministically
            valid_files.sort()
            rng.shuffle(valid_files)
            
            n = len(valid_files)
            if n >= 5:
                train_end = max(1, int(0.6 * n))
                val_end = max(train_end + 1, int(0.8 * n))
            elif n >= 3:
                train_end = n - 2
                val_end = n - 1
            else:
                train_end = 1
                val_end = 1
                
            if split == "train":
                selected_files = valid_files[:train_end]
            elif split == "val":
                selected_files = valid_files[train_end:val_end]
            elif split == "test":
                selected_files = valid_files[val_end:]
            else:
                raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test']")
                
            for file_name in selected_files:
                self.image_paths.append(os.path.join(root_dir, class_name, file_name))
                self.labels.append(self.class_to_idx[class_name])
                
        # In-memory caching for high training speed
        self.image_cache = []
        self.classical_cache = []
        print(f"Preloading images and applying enhancements for split '{split}'...")
        
        def _load_single(path):
            with Image.open(path) as img:
                rgb_img = img.convert("RGB").resize(config.IMAGE_SIZE)
                if getattr(config, "USE_CLAHE", False):
                    rgb_img = apply_clahe_preprocessing(rgb_img)
                return rgb_img

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            self.image_cache = list(executor.map(_load_single, self.image_paths))

        if config.USE_CLASSICAL:
            for rgb_img in self.image_cache:
                self.classical_cache.append(extract_classical_features(rgb_img))
                    
        print(f"Loaded split '{split}': {len(self.classes)} classes, {len(self.image_paths)} images total.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        image = self.image_cache[idx]
        image_tensor = self.transform(image)
        
        if config.USE_CLASSICAL:
            classical_tensor = torch.tensor(self.classical_cache[idx], dtype=torch.float32)
            return image_tensor, label, classical_tensor
            
        return image_tensor, label


class EpisodicBatchSampler(Sampler):
    """
    A Sampler that yields batch indices representing an FSL episode.
    Safely samples with replacement if a class doesn't have enough images in the current split.
    """
    def __init__(self, labels: list, n_way: int, k_shot: int, q_query: int, num_episodes: int):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        self.classes = np.unique(self.labels)
        self.class_to_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            episode_indices = []
            for c in sampled_classes:
                available_indices = self.class_to_indices[c]
                needed = self.k_shot + self.q_query
                
                # Sample with replacement if the split has fewer images than needed
                replace_flag = len(available_indices) < needed
                
                sampled_indices = np.random.choice(
                    available_indices,
                    needed,
                    replace=replace_flag
                )
                episode_indices.extend(sampled_indices)
                
            yield episode_indices
