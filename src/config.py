# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Your actual file name
RAW_DATA_PATH = RAW_DATA_DIR / "data.csv"

PROCESSED_PATH = PROCESSED_DATA_DIR / "processed_with_target.csv"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS_RFM = 3