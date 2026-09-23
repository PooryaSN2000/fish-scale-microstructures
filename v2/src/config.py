import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR is two levels up because BASE_DIR is v2/src
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "scale fish")

# Hardware Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Top-40 Classification Configuration
NUM_CLASSES = 5   # Total number of classes to select (Lutjanus species)

# Few-Shot Learning configuration (Stage 2)
N_WAY = 5         # Number of classes per episode (should match NUM_CLASSES)
K_SHOT = 5         # Number of support examples per class
Q_QUERY = 5        # Number of query examples per class

# FSL Metric settings
USE_COSINE_DISTANCE = True
COSINE_TEMP_INIT = 10.0  # Initial temperature scaling factor

# Feature Extractor Backbone settings
# Options: 'resnet18', 'efficientnet_b0', 'convnext_tiny'
BACKBONE = "convnext_tiny"
EMBEDDING_DIM = 256  # Dimension of projected deep embeddings

# Classical Features Integration Configuration
USE_CLASSICAL = True
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 24
CLASSICAL_PROJ_DIM = 128  # Dimension to project classical features to

# Image Preprocessing dimensions
IMAGE_SIZE = (224, 224)

# Training Hyperparameters
# Stage 1: Standard Supervised Pre-training
EPOCHS_STAGE1 = 30
BATCH_SIZE_STAGE1 = 32
LEARNING_RATE_STAGE1 = 1e-3

# Stage 2: Episodic Few-Shot Fine-tuning
EPOCHS_STAGE2 = 10
EPISODES_PER_EPOCH = 50
VAL_EPISODES = 20
TEST_EPISODES = 50
LEARNING_RATE_STAGE2 = 1e-4

WEIGHT_DECAY = 1e-4
SCHEDULER_STEP_SIZE = 15
SCHEDULER_GAMMA = 0.5

# Outputs
CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42
