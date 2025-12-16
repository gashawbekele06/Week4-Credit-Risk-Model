# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer  # <-- Added FunctionTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from typing import Optional
import logging
import sys
import os

# Path fix
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.pipeline = None
        self.woe = None
        self.iv_table = None

    @staticmethod
    def filter_debits(df):
        return df[df['Amount'] > 0].copy()

    @staticmethod
    def extract_time_features(df):
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        return df

    @staticmethod
    def aggregate_per_customer(df):
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'count'],
            'transaction_hour': 'mean',
            'transaction_day': 'mean',
            'transaction_month': 'nunique',
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
        }
        aggregated = df.groupby('CustomerId').agg(agg_dict)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        aggregated = aggregated.reset_index()  # Preserve CustomerId
        
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
        aggregated = aggregated.rename(columns=rename_map)
        aggregated['standard_deviation_transaction_amounts'] = aggregated['standard_deviation_transaction_amounts'].fillna(0)
        return aggregated

    def build_preprocessor(self):
        numeric_features = [
            'total_transaction_amount', 'average_transaction_amount',
            'standard_deviation_transaction_amounts', 'transaction_count',
            'average_transaction_hour', 'average_transaction_day', 'active_months'
        ]
        categorical_features = ['most_frequent_product', 'most_frequent_channel']

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='drop')  # We handle CustomerId manually

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ('filter_debits', FunctionTransformer(self.filter_debits)),
            ('time_features', FunctionTransformer(self.extract_time_features)),
            ('aggregate', FunctionTransformer(self.aggregate_per_customer))
        ])
        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.pipeline is None:
            self.build_pipeline()
        
        # Run aggregation
        agg_df = self.pipeline.fit_transform(df)
        
        # Preprocess modeling features
        preprocessor = self.build_preprocessor()
        X_processed = preprocessor.fit_transform(agg_df)
        feature_names = preprocessor.get_feature_names_out()
        
        # Final DataFrame with CustomerId first
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        X_df.insert(0, 'CustomerId', agg_df['CustomerId'].values)
        
        if y is not None:
            woe_features = ['most_frequent_product', 'most_frequent_channel']
            self.woe = WOE()
            self.woe.fit(agg_df[woe_features], y)
            X_woe = self.woe.transform(agg_df[woe_features])
            X_df = pd.concat([X_df, X_woe.add_prefix('woe_')], axis=1)
            
            self.iv_table = self.woe.iv_df.sort_values('Information_Value', ascending=False)
            print("\nInformation Value Table:")
            print(self.iv_table)

        return X_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted")
        
        agg_df = self.pipeline.transform(df)
        
        preprocessor = self.build_preprocessor()
        X_processed = preprocessor.transform(agg_df)
        feature_names = preprocessor.get_feature_names_out()
        
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        X_df.insert(0, 'CustomerId', agg_df['CustomerId'].values)
        
        return X_df