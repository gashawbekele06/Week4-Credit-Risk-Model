# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

# Ensure directories exist
(PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

PLOT_STYLE = "seaborn-v0_8"
PALETTE = "husl"
RANDOM_STATE = 42
SNAPSHOT_DATE = "2019-03-01"