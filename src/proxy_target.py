# src/proxy_target.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os

# Path fix
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.config import SNAPSHOT_DATE, RANDOM_STATE

class ProxyTargetEngineer:
    def __init__(self):
        self.snapshot_date = pd.to_datetime(SNAPSHOT_DATE).tz_localize('UTC')

    def create_proxy_target(self, df):
        debits = df[df['Amount'] > 0].copy()
        debits['TransactionStartTime'] = pd.to_datetime(debits['TransactionStartTime'], utc=True)

        rfm = debits.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']

        rfm['monetary_log'] = np.log1p(rfm['monetary'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary_log']])

        kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        rfm['cluster'] = kmeans.fit_predict(X_scaled)

        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        risk_score = centers[:,0] - centers[:,1] - centers[:,2]
        high_risk_cluster = np.argmax(risk_score)

        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

        return rfm[['CustomerId', 'is_high_risk']]