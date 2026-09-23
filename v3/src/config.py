import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR is two levels up because BASE_DIR is v3/src
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "scale fish")

# Hardware Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Top-40 Classification Configuration
NUM_CLASSES = 40   # Total number of classes to select (Top 40 most populated species)

# Feature Extractor Backbone settings
BACKBONE = "convnext_tiny"
EMBEDDING_DIM = 384  # Dimension of projected deep embeddings

# ArcFace Metric Learning Configuration
ARCFACE_SCALE = 30.0   # Feature scale norm factor (s)
ARCFACE_MARGIN = 0.35  # Angular margin in radians (m)

# Classical Features Integration Configuration (Disabled in v3.1 to ensure rotation invariance)
USE_CLASSICAL = False

# Image Preprocessing dimensions (336x336 provides superior spatial resolution for fine circuli)
IMAGE_SIZE = (336, 336)

# Biological Preprocessing (CLAHE)
USE_CLAHE = True
CLAHE_CLIP = 0.015

# Training Hyperparameters
EPOCHS = 35
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
LR_BACKBONE = 2e-5
LR_PROJECTOR = 3e-4
LR_ARCFACE = 1e-3
WEIGHT_DECAY = 1e-4

# Outputs
CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42
