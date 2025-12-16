# src/data_loader.py
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import sys
import os

# Path fix for standalone run
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_PATH
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
        logger.info(f"Loaded {df.shape[0]:,} rows")
        return df