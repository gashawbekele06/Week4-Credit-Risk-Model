# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import FeatureEngineer

@pytest.fixture
def sample_raw_data():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [1000, 2000, 1500, 3000, 500],
        'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-11-15T03:18:49Z', '2018-11-15T02:44:21Z', '2018-11-15T04:44:21Z', '2018-11-15T05:44:21Z'],
        'ProductCategory': ['airtime', 'airtime', 'data_bundles', 'data_bundles', 'utility_bill'],
        'ChannelId': ['Channel_3', 'Channel_3', 'Channel_3', 'Channel_3', 'Channel_3']
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_feature_engineer_returns_expected_columns(sample_raw_data):
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_data)
    
    required_cols = [
        'total_transaction_amount',
        'average_transaction_amount',
        'standard_deviation_transaction_amounts',
        'transaction_count',
        'CustomerId'
    ]
    
    for col in required_cols:
        assert col in X.columns, f"Missing required column: {col}"
    
    assert X.shape[0] == 3  # 3 unique customers

def test_feature_engineer_preserves_customer_id(sample_raw_data):
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_data)
    
    assert 'CustomerId' in X.columns
    assert X['CustomerId'].nunique() == 3
    assert set(X['CustomerId']) == {'C1', 'C2', 'C3'}

def test_feature_engineer_handles_negative_amounts(sample_raw_data):
    # Test that negative amounts are filtered out
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_data)
    
    # All transactions should be debits (positive Amount)
    assert X['total_transaction_amount'].min() >= 0
    assert X['average_transaction_amount'].min() >= 0

def test_pipeline_is_reproducible(sample_raw_data):
    engineer1 = FeatureEngineer()
    X1 = engineer1.fit_transform(sample_raw_data)
    
    engineer2 = FeatureEngineer()
    X2 = engineer2.fit_transform(sample_raw_data)
    
    # Should be identical
    pd.testing.assert_frame_equal(X1, X2)