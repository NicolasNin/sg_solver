"""
Configuration for ML training infrastructure.
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# MLflow configuration
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT}/mlruns"
MLFLOW_EXPERIMENT_NAME = "board_detection"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATED_DIRS = [
    DATA_DIR / "my_photos",
]

# Model save directory (legacy, will migrate to MLflow)
LEGACY_MODELS_DIR = DATA_DIR / "models"

# Feature extraction settings
PATCH_SIZE = 80
NUM_CELLS = 48
