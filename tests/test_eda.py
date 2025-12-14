# tests/test_eda.py
import sys
import os
# FIX: Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.eda import EDA

@pytest.fixture
def sample_df():
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'Amount': [1000, -500, 2000, 0],
        'Value': [1000, 500, 2000, 0],
        'TransactionStartTime': ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'],
        'ProductCategory': ['airtime', 'data', 'airtime', 'utility'],
        'ChannelId': ['Channel_1', 'Channel_2', 'Channel_1', 'Channel_3'],
        'FraudResult': [0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_eda_overview(sample_df, capsys):
    eda = EDA(sample_df)
    eda.overview()
    captured = capsys.readouterr()
    assert "Number of rows: 4" in captured.out
    assert "Number of columns: 7" in captured.out

def test_numerical_distributions_runs(sample_df):
    eda = EDA(sample_df)
    eda.numerical_distributions()  # Should run without error

def test_categorical_distributions_handles_imbalance(sample_df):
    eda = EDA(sample_df)
    eda.categorical_distributions()  # Should run and use pie/bar appropriately

def test_top_insights_contains_business_linkage(sample_df, capsys):
    eda = EDA(sample_df)
    eda.top_insights()
    captured = capsys.readouterr()
    assert "BATI BANK" in captured.out or "proxy" in captured.out.lower()