import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "dataset")

# Hardware Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-Species Lutjanidae Classification Configuration
NUM_CLASSES = 5

# Few-Shot Learning configuration (Stage 2)
N_WAY = 5
K_SHOT = 5
Q_QUERY = 5

# FSL Metric settings
USE_COSINE_DISTANCE = True
COSINE_TEMP_INIT = 10.0

# Feature Extractor Backbone & Domain Generalization
BACKBONE = "convnext_tiny"
EMBEDDING_DIM = 256
USE_MIXSTYLE = True
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.3
USE_FDA = True
FDA_BETA = 0.08

# Classical Features Integration (disabled for 360-deg rotation invariance)
USE_CLASSICAL = False
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 24
CLASSICAL_PROJ_DIM = 128

# Biological Scale Localization & Preprocessing
IMAGE_SIZE = (336, 336)
USE_ROI_CROP = True
USE_CLAHE = True
CLAHE_CLIP = 0.015
USE_TTA = True

# Training Hyperparameters
EPOCHS_STAGE1 = 22
BATCH_SIZE_STAGE1 = 16
LEARNING_RATE_STAGE1 = 5e-5

EPOCHS_STAGE2 = 20
EPISODES_PER_EPOCH = 30
VAL_EPISODES = 15
TEST_EPISODES = 50
LEARNING_RATE_STAGE2 = 2e-4

WEIGHT_DECAY = 1e-4

# Outputs
CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42
