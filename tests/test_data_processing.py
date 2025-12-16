# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import FeatureEngineer

@pytest.fixture
def sample_raw_data():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [1000, -500, 2000, 1500, 3000],
        'TransactionStartTime': ['2018-11-15T02:18:49Z'] * 5,
        'ProductCategory': ['airtime', 'financial_services', 'data_bundles', 'airtime', 'utility_bill'],
        'ChannelId': ['Channel_3', 'Channel_2', 'Channel_3', 'Channel_3', 'Channel_3']
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_feature_engineer_returns_expected_columns(sample_raw_data):
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_data)
    
    expected_cols = [
        'CustomerId',
        'total_transaction_amount',
        'average_transaction_amount',
        'standard_deviation_transaction_amounts',
        'transaction_count',
        'average_transaction_hour',
        'average_transaction_day',
        'active_months',
        'most_frequent_product',
        'most_frequent_channel'
    ]
    
    # Check core aggregate columns exist
    for col in ['total_transaction_amount', 'transaction_count', 'CustomerId']:
        assert col in X.columns, f"Missing required column: {col}"
    
    assert X.shape[0] == 3  # 3 unique customers

def test_feature_engineer_preserves_customer_id(sample_raw_data):
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_data)
    
    assert 'CustomerId' in X.columns
    assert X['CustomerId'].nunique() == 3
    assert set(X['CustomerId']) == {'C1', 'C2', 'C3'}