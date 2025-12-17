#Python# src/config.py
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data path - adjust if your file is in data/raw or elsewhere
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"

# Alternative if you have raw folder
# DATA_PATH = PROJECT_ROOT / "data" / "raw" / "training.csv"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS_RFM = 3