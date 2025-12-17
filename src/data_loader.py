# src/data_loader.py
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
<<<<<<< HEAD
import sys
import os

# Path fix for standalone run
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
=======
from config import DATA_PATH

logging.basicConfig(level=logging.INFO)
>>>>>>> task-3
logger = logging.getLogger(__name__)

class DataLoader:
<<<<<<< HEAD
    """Robust DataLoader with comprehensive error handling."""
    
=======
>>>>>>> task-3
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_PATH
        
        if not self.data_path.exists():
<<<<<<< HEAD
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Please ensure 'data/data.csv' exists in your project directory."
            )
        
        logger.info(f"DataLoader initialized with path: {self.data_path}")

    def load(self) -> pd.DataFrame:
        """Load data with multiple fallback strategies."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                df = pd.read_csv(self.data_path, encoding=enc, low_memory=False)
                logger.info(f"Successfully loaded with encoding: {enc}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to load with encoding: {enc}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error with encoding {enc}: {e}")
                raise
        else:
            raise ValueError(f"Could not read file with any supported encoding: {self.data_path}")

        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Safe datetime conversion
        if 'TransactionStartTime' in df.columns:
            original_count = len(df)
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
            invalid = df['TransactionStartTime'].isna().sum()
            if invalid > 0:
                logger.warning(f"{invalid}/{original_count} TransactionStartTime values invalid and set to NaT")

        # Safe numeric conversion
        for col in ['Amount', 'Value']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info("Data loading completed successfully")
=======
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
        logger.info(f"Loaded {df.shape[0]:,} rows")
>>>>>>> task-3
        return df