#Python# src/config.py
from pathlib import Path
import os

# Project root (works regardless of where script is run)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Explicit data directory - ensures structure is clear
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)  # Create if missing

# Data path
DATA_PATH = DATA_DIR / "data.csv"

# Raise clear error if file missing
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found at {DATA_PATH}\n"
        f"Please place your CSV file in the 'data/' directory as 'data.csv'\n"
        f"Current directory structure expected:\n"
        f"  {PROJECT_ROOT.name}/\n"
        f"    data/\n"
        f"      data.csv\n"
        f"    src/\n"
        f"    notebooks/"
    )

# Plot style
PLOT_STYLE = "seaborn-v0_8"
PALETTE = "husl"