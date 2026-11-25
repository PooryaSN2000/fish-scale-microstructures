import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from skimage import exposure

try:
    from . import config
    from .augmentations import get_v4_transforms
    from .scale_detector import ScaleDetector
except ImportError:
    import config
    from augmentations import get_v4_transforms
    from scale_detector import ScaleDetector

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(path: str) -> Image.Image:
    """Loads RGB micrographs safely, handling 16-bit/grayscale TIFFs cleanly."""
    with Image.open(path) as raw_img:
        if raw_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(raw_img, dtype=np.float32)
            mx = float(arr.max()) if arr.size else 1.0
            if mx > 0:
                arr = (arr / mx * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            return Image.fromarray(arr).convert("RGB")
        return raw_img.convert("RGB")


def apply_clahe_preprocessing(img: Image.Image) -> Image.Image:
    """
    Biological contrast enhancement for fish scales.
    Sharpens fine circuli and radii while blending 50/50 to avoid amplifying noise.
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


def get_unseen_novel_species(root_dir: str, seen_classes: list[str]) -> dict[str, list[str]]:
    """
    Returns dictionary {species_name: sorted_file_paths} for all completely excluded
    taxa (< MIN_SAMPLES) in root_dir for strict Open-Set Few-Shot Novel Species benchmarking.
    """
    seen_set = set(seen_classes)
    novel = {}
    for d in sorted(os.listdir(root_dir)):
        dp = os.path.join(root_dir, d)
        if not os.path.isdir(dp) or d in seen_set:
            continue
        files = sorted([
            os.path.join(dp, f)
            for f in os.listdir(dp)
            if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS
        ])
        if files:
            novel[d] = files
    return novel


class FishScaleDatasetV4(Dataset):
    """
    Next-Generation Fish Scale Dataset (v4) with Unified Biological ROI Localization
    and Few-Shot / Long-Tail Stratification.
    """
    def __init__(self, root_dir: str, split: str = "train", seed: int = 42, use_roi_crop: bool = None, use_clahe: bool = None):
        self.root_dir = root_dir
        self.split = split
        self.use_roi_crop = getattr(config, "USE_ROI_CROP", True) if use_roi_crop is None else use_roi_crop
        self.use_clahe = getattr(config, "USE_CLAHE", True) if use_clahe is None else use_clahe
        self.transform = get_v4_transforms(config.IMAGE_SIZE, is_train=(split == "train"))
        self.detector = ScaleDetector(padding_ratio=getattr(config, "ROI_PADDING_RATIO", 0.12)) if self.use_roi_crop else None

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dataset directory not found: {root_dir}")

        raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # 1. Count valid images per class
        class_counts = []
        for class_name in raw_classes:
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            valid_files = [f for f in files if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS]
            if len(valid_files) > 0:
                class_counts.append((class_name, len(valid_files), valid_files))

        # 2. Sort by count descending and select target number of classes
        class_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
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
        self.full_counts = [0] * len(self.classes)

        # 3. Few-shot aware stratified splitting (70% Train, 15% Val, 15% Test)
        for class_name in self.classes:
            info = next(info for info in selected_classes_info if info[0] == class_name)
            valid_files = info[2].copy()
            valid_files.sort()
            rng.shuffle(valid_files)

            n = len(valid_files)
            class_idx = self.class_to_idx[class_name]
            self.full_counts[class_idx] = n

            if n >= 6:
                train_end = max(1, int(0.70 * n))
                val_end = max(train_end + 1, int(0.85 * n))
            elif n >= 4:
                train_end = n - 2
                val_end = n - 1
            elif n == 3:
                train_end = 2
                val_end = 2
            else:
                train_end = 1
                val_end = 1

            if split == "train":
                selected_files = valid_files[:train_end]
            elif split == "val":
                selected_files = valid_files[train_end:val_end] if val_end > train_end else valid_files[train_end:]
            elif split == "test":
                selected_files = valid_files[val_end:] if val_end < n else valid_files[train_end:]
            else:
                selected_files = valid_files

            if len(selected_files) == 0:
                selected_files = [valid_files[0]]

            for f in selected_files:
                self.image_paths.append(os.path.join(root_dir, class_name, f))
                self.labels.append(class_idx)

            self.train_counts[class_idx] = len(valid_files[:train_end])

        self._cache = {}

    def __len__(self):
        return len(self.image_paths)

    def load_processed_pil(self, path: str) -> Image.Image:
        if path in self._cache:
            return self._cache[path]
        img = load_rgb_image(path)
        if self.use_roi_crop and self.detector is not None:
            img = self.detector.crop(img, padding_ratio=getattr(config, "ROI_PADDING_RATIO", 0.12), square=True)
        if self.use_clahe:
            img = apply_clahe_preprocessing(img)
        # Store resized RGB in RAM cache to keep footprint tiny and throughput high
        cache_img = img.resize((max(config.IMAGE_SIZE) + 48, max(config.IMAGE_SIZE) + 48), Image.Resampling.BILINEAR)
        self._cache[path] = cache_img
        return cache_img

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = self.load_processed_pil(path)
        tensor_img = self.transform(img)
        return tensor_img, label, self.classes[label]


def get_dataloaders():
    """
    Creates train, validation, and test DataLoader instances with balanced sampling.
    """
    train_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    val_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)
    test_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="test", seed=config.RANDOM_SEED)

    train_labels = np.array(train_dataset.labels)
    class_sample_counts = np.bincount(train_labels, minlength=len(train_dataset.classes))

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
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.train_counts, train_dataset.classes
