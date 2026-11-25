import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from skimage import exposure

import config
from features import extract_classical_features

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def apply_clahe_preprocessing(img: Image.Image) -> Image.Image:
    """
    Biological contrast enhancement for fish scales.
    Sharpens fine circuli and radii while blending 50/50 to preserve natural intensities.
    """
    if not getattr(config, "USE_CLAHE", False):
        return img

    if HAS_CV2:
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        blended = np.clip(0.5 * arr + 0.5 * res, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)
    else:
        arr = np.array(img, dtype=np.float32) / 255.0
        res = exposure.equalize_adapthist(arr, clip_limit=getattr(config, "CLAHE_CLIP", 0.015))
        blended = (np.clip(0.5 * arr + 0.5 * res, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(blended)


def get_transforms(split: str):
    """Base image resizing and tensor conversion."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])


class FishScaleDataset(Dataset):
    """
    Dataset selecting the 5 target Lutjanus species and performing a deterministic
    60% Train / 20% Val / 20% Test split per species.
    """
    def __init__(self, root_dir: str, split: str = "train", seed: int = 42, use_clahe: bool = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = get_transforms(split)
        self.use_clahe = config.USE_CLAHE if use_clahe is None else use_clahe

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dataset directory not found: {root_dir}")

        raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        TARGET_CLASSES = [
            "Lutjanus fulviflammus(Forsskal 1775)",
            "Lutjanus johni Bloch 1792",
            "Lutjanus lemniscatus (Valenciennes, 1828)",
            "Lutjanus lutjanus Bloch 1790",
            "Lutjanus malabaricus (Bloch & Schneider, 1801)",
            "lutjanus malabarius(Bloch and Schneider,1801)",
        ]

        class_counts = []
        for class_name in raw_classes:
            if class_name not in TARGET_CLASSES:
                continue
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            valid_files = [f for f in files if os.path.splitext(f.lower())[1] in supported_exts]
            if len(valid_files) > 0:
                class_counts.append((class_name, len(valid_files), valid_files))

        self.classes = sorted([info[0] for info in class_counts])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []
        rng = random.Random(seed)

        for class_name in self.classes:
            info = next(info for info in class_counts if info[0] == class_name)
            valid_files = sorted(info[2])
            rng.shuffle(valid_files)

            n = len(valid_files)
            train_end = int(0.6 * n)
            val_end = int(0.8 * n)
            if n < 3:
                train_end = 1
                val_end = 2 if n > 1 else 1

            if split == "train":
                split_files = valid_files[:train_end]
            elif split == "val":
                split_files = valid_files[train_end:val_end]
            elif split == "test":
                split_files = valid_files[val_end:]
            elif split == "all":
                split_files = valid_files
            else:
                raise ValueError(f"Unknown split: {split}")

            class_path = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for f in split_files:
                self.image_paths.append(os.path.join(class_path, f))
                self.labels.append(class_idx)

        self.image_cache = {}
        self.classical_cache = {}

        print(f"Preloading images for split '{split}' (CLAHE={self.use_clahe})...")
        for idx, img_path in enumerate(self.image_paths):
            image = Image.open(img_path).convert("RGB")
            if self.use_clahe:
                image = apply_clahe_preprocessing(image)
            self.image_cache[idx] = image
            if config.USE_CLASSICAL:
                self.classical_cache[idx] = extract_classical_features(image)

        print(f"Loaded split '{split}': {len(self.classes)} classes, {len(self.image_paths)} images total.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        image_tensor = self.transform(self.image_cache[idx])
        if config.USE_CLASSICAL:
            classical_tensor = torch.tensor(self.classical_cache[idx], dtype=torch.float32)
            return image_tensor, label, classical_tensor
        return image_tensor, label


class EpisodicBatchSampler(Sampler):
    """
    Sampler that yields batch indices representing an N-way K-shot Q-query FSL episode.
    """
    def __init__(self, labels: list, n_way: int, k_shot: int, q_query: int, num_episodes: int):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.classes = np.unique(self.labels)
        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)
            episode_indices = []
            for c in sampled_classes:
                available_indices = self.class_to_indices[c]
                needed = self.k_shot + self.q_query
                replace_flag = len(available_indices) < needed
                sampled_indices = np.random.choice(available_indices, needed, replace=replace_flag)
                episode_indices.extend(sampled_indices)
            yield episode_indices
