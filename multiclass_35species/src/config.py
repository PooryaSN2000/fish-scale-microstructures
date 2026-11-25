import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR points to the dataset folder 'dataset' at project root
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "dataset")

# Hardware Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification Configuration
# Setting NUM_CLASSES = 35 filters out noisy species with < 7 samples,
# ensuring every class has statistical significance, robust prototypes, and high generalization.
NUM_CLASSES = 35
MIN_SAMPLES = 7

# Feature Extractor Backbone settings
BACKBONE = "convnext_tiny"
EMBEDDING_DIM = 384  # Dimension of projected deep embeddings

# Domain Generalization (Cross-Camera & Sensor Invariance via MixStyle)
USE_MIXSTYLE = True
MIXSTYLE_P = 0.25      # Balanced probability preserving cross-family optical traits
MIXSTYLE_ALPHA = 0.3   # Shape parameter of Beta distribution for mixing weights

# Class-Balanced ArcFace (CB-ArcFace) for Long-Tail Mitigation
# Margin for class c: m_c = m_base + m_delta * (1 - (N_c / N_max)^gamma)
ARCFACE_SCALE = 30.0        # Feature scale norm factor (s)
ARCFACE_MARGIN_BASE = 0.25  # Angular margin for dominant species
ARCFACE_MARGIN_DELTA = 0.15 # Additional angular margin for tail classes
ARCFACE_GAMMA = 0.25        # Nonlinear scaling parameter

# Frequency Domain & Cross-Sensor Augmentations
USE_FDA = True              # Fourier Domain Augmentation (Amplitude swap)
FDA_BETA = 0.06             # Window size ratio for low-frequency spectrum exchange

# Image Preprocessing & Biological Scale Detection
IMAGE_SIZE = (336, 336)
USE_ROI_CROP = True
ROI_PADDING_RATIO = 0.12
USE_CLAHE = True
CLAHE_CLIP = 0.015

# Training Hyperparameters
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 34
EPOCHS = EPOCHS_STAGE2
BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
LR_BACKBONE = 1e-4
LR_PROJECTOR = 5e-4
LR_ARCFACE = 1.5e-3
WEIGHT_DECAY = 1e-4

# Inference & Test-Time Augmentation (TTA)
USE_SPHERICAL_TTA = True    # Hyperspherical cosine-averaged multi-view TTA

# Output directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42
