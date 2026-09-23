import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

import config
from features import extract_classical_features

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
    Dataset that dynamically selects the top N_WAY most populated classes
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
                
        # 2. Sort by count descending and pick Top N_WAY classes
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
            train_end = int(0.6 * n)
            val_end = int(0.8 * n)
            
            # Fallback for extremely small classes (ensure at least 1 image per split if possible)
            if n < 3:
                train_end = 1
                val_end = 2 if n > 1 else 1
                
            if split == "train":
                split_files = valid_files[:train_end]
            elif split == "val":
                split_files = valid_files[train_end:val_end]
            elif split == "test":
                split_files = valid_files[val_end:]
            else:
                raise ValueError(f"Unknown split: {split}")
                
            class_path = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for f in split_files:
                self.image_paths.append(os.path.join(class_path, f))
                self.labels.append(class_idx)
                
        self.image_cache = {}
        self.classical_cache = {}
        
        print(f"Preloading images and computing features for split '{split}'...")
        for idx, img_path in enumerate(self.image_paths):
            image = Image.open(img_path).convert("RGB")
            self.image_cache[idx] = image
            
            if config.USE_CLASSICAL:
                self.classical_cache[idx] = extract_classical_features(image)
                
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
