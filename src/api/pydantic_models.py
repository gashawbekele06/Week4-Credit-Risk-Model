# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import List

class CustomerInput(BaseModel):
    total_transaction_amount: float
    average_transaction_amount: float
    standard_deviation_transaction_amounts: float
    transaction_count: int
    average_transaction_hour: float
    average_transaction_day: float
    active_months: int
    most_frequent_product: str
    most_frequent_channel: str

class PredictionOutput(BaseModel):
    risk_probability: float
    is_high_risk: int  # 1 if probability > 0.5, else 0