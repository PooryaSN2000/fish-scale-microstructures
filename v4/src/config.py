import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR points to the dataset folder 'scale fish' at project root
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "scale fish")

# Hardware Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification Configuration
# Setting NUM_CLASSES = None automatically includes all species in the dataset (57 species),
# or specify an integer (e.g. 57 or 40)
NUM_CLASSES = 57

# Feature Extractor Backbone settings
BACKBONE = "convnext_tiny"
EMBEDDING_DIM = 384  # Dimension of projected deep embeddings

# Domain Generalization (Cross-Camera & Sensor Invariance via MixStyle)
USE_MIXSTYLE = True
MIXSTYLE_P = 0.5       # Probability of applying MixStyle on a training batch
MIXSTYLE_ALPHA = 0.3   # Shape parameter of Beta distribution for mixing weights

# Class-Balanced ArcFace (CB-ArcFace) for Long-Tail / Few-Shot Mitigation
# Margin for class c: m_c = m_base + m_delta * (1 - (N_c / N_max)^gamma)
ARCFACE_SCALE = 30.0        # Feature scale norm factor (s)
ARCFACE_MARGIN_BASE = 0.30  # Baseline angular margin for frequent (head) classes
ARCFACE_MARGIN_DELTA = 0.25 # Additional angular margin for few-shot (tail) classes
ARCFACE_GAMMA = 0.25        # Nonlinear scaling parameter

# Frequency Domain & Cross-Sensor Augmentations
USE_FDA = True              # Fourier Domain Augmentation (Amplitude swap)
FDA_BETA = 0.08             # Window size ratio for low-frequency spectrum exchange

# Image Preprocessing & Biological Scale Detection
IMAGE_SIZE = (336, 336)
USE_CLAHE = True
CLAHE_CLIP = 0.015

# Training Hyperparameters
EPOCHS = 35
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2      # Effective batch size = 8 * 2 = 16
LR_BACKBONE = 2e-5
LR_PROJECTOR = 3e-4
LR_ARCFACE = 1e-3
WEIGHT_DECAY = 1e-4

# Inference & Test-Time Augmentation (TTA)
USE_SPHERICAL_TTA = True    # Hyperspherical cosine-averaged multi-view TTA

# Output directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42
