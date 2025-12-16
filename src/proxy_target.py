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
            'average_transaction_hour'
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
        ], remainder='passthrough')  # Keep CustomerId

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('filter', FunctionTransformer(self.filter_debits)),
            ('time', FunctionTransformer(self.extract_time_features)),
            ('aggregate', FunctionTransformer(self.aggregate_per_customer)),
            ('preprocess', self.build_preprocessor())
        ])

    def fit_transform(self, df, y=None):
        if self.pipeline is None:
            self.build_pipeline()
        X = self.pipeline.fit_transform(df)
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        all_cols = ['CustomerId'] + list(feature_names)
        X_df = pd.DataFrame(X, columns=all_cols)
        return X_df

    def transform(self, df):
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted")
        X = self.pipeline.transform(df)
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        all_cols = ['CustomerId'] + list(feature_names)
        return pd.DataFrame(X, columns=all_cols)