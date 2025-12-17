# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from category_encoders import WOEEncoder

from src.config import RAW_DATA_PATH, PROCESSED_PATH, RANDOM_STATE, N_CLUSTERS_RFM


# ================================
# Custom Transformers
# ================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time components from TransactionStartTime."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level features per CustomerId."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = X.groupby('CustomerId').agg(
            TotalAmount=('Amount', 'sum'),
            AvgAmount=('Amount', 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdAmount=('Amount', 'std')
        ).reset_index()
        agg['StdAmount'] = agg['StdAmount'].fillna(0)  # Single transaction → std = 0
        return X.merge(agg, on='CustomerId', how='left')


class RFMProxyTarget(BaseEstimator, TransformerMixin):
    """
    Task 4: Calculate RFM, cluster with K-Means, and create is_high_risk proxy.
    """
    def __init__(self, n_clusters=N_CLUSTERS_RFM, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        # Compute RFM at customer level
        snapshot = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (snapshot - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        # Scale and fit K-Means
        rfm_scaled = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        self.kmeans.fit(rfm_scaled)

        # Identify high-risk cluster (lowest average Frequency = least engaged)
        rfm['Cluster'] = self.kmeans.labels_
        cluster_freq = rfm.groupby('Cluster')['Frequency'].mean()
        self.high_risk_cluster = cluster_freq.idxmin()

        return self

    def transform(self, X):
        X = X.copy()
        snapshot = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (snapshot - x.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        rfm_scaled = self.scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
        rfm['Cluster'] = self.kmeans.predict(rfm_scaled)
        rfm['is_high_risk'] = (rfm['Cluster'] == self.high_risk_cluster).astype(int)

        return X.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')


# ================================
# Pipelines
# ================================

# Feature engineering pipeline (no target needed)
feature_pipeline = Pipeline(steps=[
    ('time_extract', TimeFeatureExtractor()),
    ('aggregate', CustomerAggregator()),
    ('proxy_target', RFMProxyTarget())
])


# Preprocessing pipeline for modeling features (WoE needs y)
def build_preprocessor():
    numerical_features = [
        'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount'
    ]
    categorical_features = [
        'ProductCategory', 'ChannelId', 'PricingStrategy',
        'ProviderId', 'ProductId', 'CurrencyCode', 'CountryCode'
    ]

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Standardization as required
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('woe', WOEEncoder(handle_missing='value', handle_unknown='value'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


# ================================
# Main Execution
# ================================

if __name__ == "__main__":
    print(f"Loading raw data from: {RAW_DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Apply feature engineering + proxy target creation
    df_engineered = feature_pipeline.fit_transform(df)
    print(f"After feature engineering & proxy target: {df_engineered.shape}")

    # Extract target and features
    y = df_engineered['is_high_risk']                     # pandas Series
    X = df_engineered.drop(columns=['is_high_risk'])

    # Keep only modeling features
    modeling_features = [
        'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount',
        'ProductCategory', 'ChannelId', 'PricingStrategy',
        'ProviderId', 'ProductId', 'CurrencyCode', 'CountryCode'
    ]
    X = X[modeling_features]

    # Apply WoE encoding and scaling (requires y)
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X, y)

    # Reconstruct final dataset
    final_df = pd.DataFrame(X_processed, columns=modeling_features)
    
    # FIXED LINE: Use reset_index on the Series, not on .values
    final_df['is_high_risk'] = y.reset_index(drop=True)

    # Save processed data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_PATH, index=False)

    print(f"\nProcessing complete!")
    print(f"Model-ready data saved to: {PROCESSED_PATH}")
    print(f"Final shape: {final_df.shape}")
    print("\nProxy target distribution:")
    print(final_df['is_high_risk'].value_counts(normalize=True).round(3))