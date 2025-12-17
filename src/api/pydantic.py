from pydantic import BaseModel
from typing import List, Optional


class CustomerInput(BaseModel):
    Value: float
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int
    ProviderId: str
    ProductId: str
    CurrencyCode: str
    CountryCode: int
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    TotalAmount: float
    AvgAmount: float
    TransactionCount: int
    StdAmount: float


class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: float
    optimal_loan_amount: float
    optimal_duration_months: int