# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from typing import Optional
import logging

from src.config import Config
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.pipeline = None
        self.woe = None
        self.iv_table = None

    @staticmethod
    def filter_debits(df: pd.DataFrame) -> pd.DataFrame:
        """Filter only debit transactions (Amount > 0) for meaningful spending behavior."""
        logger.info("Filtering debit transactions (Amount > 0)")
        return df[df['Amount'] > 0].copy()

    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, day, month, year from TransactionStartTime."""
        logger.info("Extracting time features")
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        return df

    @staticmethod
    def aggregate_per_customer(df: pd.DataFrame) -> pd.DataFrame:
        """Create required aggregate features per CustomerId."""
        logger.info("Aggregating features per customer")
        
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'count'],
            'transaction_hour': ['mean'],
            'transaction_day': ['mean'],
            'transaction_month': ['nunique'],
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
        }
        
        agg_df = df.groupby('CustomerId').agg(agg_dict)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
        agg_df = agg_df.reset_index()
        
        # Rename to match requirements
        rename_map = {
            'Amount_sum': 'total_transaction_amount',
            'Amount_mean': 'average_transaction_amount',
            'Amount_std': 'standard_deviation_transaction_amounts',
            'Amount_count': 'transaction_count',
            'transaction_hour_mean': 'average_transaction_hour',
            'transaction_day_mean': 'average_transaction_day',
            'transaction_month_nunique': 'active_months',
            'ProductCategory_<lambda>': 'most_frequent_product',
            'ChannelId_<lambda>': 'most_frequent_channel'
        }
        agg_df = agg_df.rename(columns=rename_map)
        
        # Handle missing std (single transaction customers)
        agg_df['standard_deviation_transaction_amounts'] = agg_df['standard_deviation_transaction_amounts'].fillna(0)
        
        return agg_df

    def build_preprocessor(self) -> ColumnTransformer:
        """Preprocess numeric and categorical features."""
        numeric_features = [
            'total_transaction_amount', 'average_transaction_amount',
            'standard_deviation_transaction_amounts', 'transaction_count',
            'average_transaction_hour', 'average_transaction_day', 'active_months'
        ]
        
        categorical_features = ['most_frequent_product', 'most_frequent_channel']

        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Handle missing
            ('scaler', StandardScaler())  # Standardization
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding
        ])

        return ColumnTransformer(transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    def build_pipeline(self) -> Pipeline:
        """Full reproducible pipeline."""
        self.pipeline = Pipeline(steps=[
            ('filter_debits', FunctionTransformer(self.filter_debits)),
            ('time_features', FunctionTransformer(self.extract_time_features)),
            ('aggregate', FunctionTransformer(self.aggregate_per_customer)),
            ('preprocess', self.build_preprocessor())
        ])
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit pipeline and optionally apply WoE."""
        if self.pipeline is None:
            self.build_pipeline()

        X = self.pipeline.fit_transform(df)
        
        # Feature names
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        X_df = pd.DataFrame(X, columns=feature_names)

        if y is not None:
            logger.info("Applying WoE transformation")
            woe_features = ['most_frequent_product', 'most_frequent_channel']
            # Extract original categorical for WoE (before one-hot)
            agg_df = self.pipeline.named_steps['aggregate'].transform(
                self.pipeline.named_steps['time_features'].transform(
                    self.pipeline.named_steps['filter_debits'].transform(df)
                )
            )
            agg_df = pd.DataFrame(agg_df, columns=self.aggregate_per_customer(df.iloc[:1]).columns)  # Approximate
            
            self.woe = WOE()
            self.woe.fit(agg_df[woe_features], y)
            X_woe = self.woe.transform(agg_df[woe_features])
            X_df = pd.concat([X_df, X_woe.add_prefix('woe_')], axis=1)
            
            self.iv_table = self.woe.iv_df.sort_values('Information_Value', ascending=False)
            print("\nInformation Value Table:")
            print(self.iv_table)

        logger.info(f"Feature engineering complete. Shape: {X_df.shape}")
        return X_df

# Standalone test
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df)
    
    print("\nModel-ready features:")
    print(X.head())
    print(f"Shape: {X.shape}")