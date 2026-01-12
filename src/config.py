"""
Configuration file for the image classification project.
All paths, constants, and hyperparameters are defined here.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Legacy data paths (for backward compatibility)
LEGACY_TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "seg_train" / "seg_train"
LEGACY_TEST_DIR = PROJECT_ROOT / "data" / "test" / "seg_test" / "seg_test"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
IMG_SIZE = (150, 150)  # Baseline and regularized models
TL_IMG_SIZE = (224, 224)  # Transfer learning (MobileNetV2)
BATCH_SIZE = 32
SEED = 42

# Training settings
BASELINE_EPOCHS = 10
REGULARIZED_EPOCHS = 20
TL_INITIAL_EPOCHS = 5
TL_FINETUNE_EPOCHS = 5
TL_UNFREEZE_LAYERS = 30

# Learning rates
BASELINE_LR = 1e-4
REGULARIZED_LR = 3e-4
TL_HEAD_LR = 1e-3
TL_FINETUNE_LR = 1e-4

# Regularization
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.5

# =============================================================================
# SPEED MODE (for quick testing)
# =============================================================================
SPEED_MODE = False
MAX_TRAIN_STEPS = 50
MAX_VAL_STEPS = 20
MAX_TEST_STEPS = 100
GRAD_CAM_N = 2

# =============================================================================
# CLASS NAMES
# =============================================================================
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# API SETTINGS
# =============================================================================
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, PREDICTIONS_DIR,
                     OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_train_dir():
    """Get the training directory (legacy or new structure)."""
    # Check if new structure has actual class folders (not nested seg_train)
    if TRAIN_DIR.exists():
        subdirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
        if subdirs and any(d.name in CLASS_NAMES for d in subdirs):
            return TRAIN_DIR
    # Use legacy nested path
    return LEGACY_TRAIN_DIR

def get_test_dir():
    """Get the test directory (legacy or new structure)."""
    # Check if new structure has actual class folders
    if TEST_DIR.exists():
        subdirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
        if subdirs and any(d.name in CLASS_NAMES for d in subdirs):
            return TEST_DIR
    # Use legacy nested path
    return LEGACY_TEST_DIR
