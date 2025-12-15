# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from typing import Optional
from config import PROCESSED_PATH
from data_loader import DataLoader


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
        df['hour'] = df['TransactionStartTime'].dt.hour
        df['day'] = df['TransactionStartTime'].dt.day
        df['month'] = df['TransactionStartTime'].dt.month
        df['year'] = df['TransactionStartTime'].dt.year
        return df

    @staticmethod
    def aggregate_per_customer(df):
        agg = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'hour': 'mean',
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'unknown'
        })
        agg.columns = ['_'.join(col).strip() for col in agg.columns]
        agg = agg.reset_index()
        agg = agg.rename(columns={
            'Amount_sum': 'total_transaction_amount',
            'Amount_mean': 'average_transaction_amount',
            'Amount_std': 'standard_deviation_transaction_amounts',
            'Amount_count': 'transaction_count',
            'hour_mean': 'average_transaction_hour',
            'ProductCategory_<lambda>': 'most_frequent_product',
            'ChannelId_<lambda>': 'most_frequent_channel'
        })
        agg['standard_deviation_transaction_amounts'] = agg['standard_deviation_transaction_amounts'].fillna(0)
        return agg

    def build_pipeline(self):
        numeric_features = ['total_transaction_amount', 'average_transaction_amount', 'standard_deviation_transaction_amounts', 'transaction_count', 'average_transaction_hour']
        categorical_features = ['most_frequent_product', 'most_frequent_channel']

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        self.pipeline = Pipeline([
            ('filter', FunctionTransformer(self.filter_debits)),
            ('time', FunctionTransformer(self.extract_time_features)),
            ('aggregate', FunctionTransformer(self.aggregate_per_customer)),
            ('preprocess', preprocessor)
        ])

    def fit_transform(self, df, y=None):
        if self.pipeline is None:
            self.build_pipeline()
        X = self.pipeline.fit_transform(df)
        feature_names = self.pipeline.named_steps['preprocess'].get_feature_names_out()
        X_df = pd.DataFrame(X, columns=feature_names)

        if y is not None:
            woe_cols = ['most_frequent_product', 'most_frequent_channel']
            agg_df = self.pipeline.named_steps['aggregate'].transform(
                self.pipeline.named_steps['time'].transform(
                    self.pipeline.named_steps['filter'].transform(df)
                )
            )
            agg_df = pd.DataFrame(agg_df, columns=self.aggregate_per_customer(df.iloc[:1]).columns)
            self.woe = WOE()
            self.woe.fit(agg_df[woe_cols], y)
            X_woe = self.woe.transform(agg_df[woe_cols])
            X_df = pd.concat([X_df, X_woe.add_prefix('woe_')], axis=1)

        return X_df