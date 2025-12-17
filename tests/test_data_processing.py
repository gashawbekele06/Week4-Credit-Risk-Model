# tests/test_data_processing.py
import pandas as pd
import numpy as np
from src.data_processing import (
    TimeFeatureExtractor,
    CustomerAggregator,
    RFMProxyTarget
)

def test_time_extractor():
    df = pd.DataFrame({
        'TransactionStartTime': ['2019-01-01 10:30:00', '2019-02-15 14:45:00']
    })
    transformer = TimeFeatureExtractor()
    result = transformer.transform(df)

    expected_cols = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    for col in expected_cols:
        assert col in result.columns

    assert result['TransactionHour'].iloc[0] == 10
    assert result['TransactionMonth'].iloc[1] == 2


def test_customer_aggregator():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 150],
        'TransactionId': ['a', 'b', 'c']
    })
    transformer = CustomerAggregator()
    result = transformer.transform(df)

    cust1 = result[result['CustomerId'] == 1]
    assert cust1['TotalAmount'].iloc[0] == 300
    assert cust1['AvgAmount'].iloc[0] == 150
    assert cust1['TransactionCount'].iloc[0] == 2
    # Correct std for [100, 200] is ≈70.71
    assert np.isclose(cust1['StdAmount'].iloc[0], 70.71067811865476, atol=1e-8)


def test_proxy_target_creates_column():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': pd.to_datetime(['2019-01-01', '2019-01-10', '2019-02-01']),
        'TransactionId': ['a', 'b', 'c'],
        'Amount': [100, 200, 50]
    })
    transformer = RFMProxyTarget(n_clusters=2, random_state=42)
    result = transformer.fit_transform(df)

    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].nunique() == 2  # Should have both 0 and 1