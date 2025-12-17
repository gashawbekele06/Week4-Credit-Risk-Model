# src/predict.py - FINAL WORKING VERSION (uses raw data for fitting)
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
from sklearn.pipeline import Pipeline

# Load model
print("Loading model from MLflow registry...")
model = mlflow.sklearn.load_model("models:/CreditRiskProxyBest/1")
print("Model loaded successfully!")

# Features
numerical_features = [
    'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
    'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount'
]
categorical_features = [
    'ProductCategory', 'ChannelId', 'PricingStrategy',
    'ProviderId', 'ProductId', 'CurrencyCode', 'CountryCode'
]

# Preprocessor
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('woe', WOEEncoder(handle_missing='value', handle_unknown='value'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# === ONE-TIME ONLY: Fit using data BEFORE WoE encoding ===
if True:  # Run this once
    print("Fitting preprocessor using raw data with strings...")
    df = pd.read_csv("data/raw/data.csv")
    # Add engineered features and proxy target temporarily
    # Use the full feature engineering from data_processing.py
    from src.data_processing import feature_pipeline
    df_engineered = feature_pipeline.fit_transform(df)
    y = df_engineered['is_high_risk']
    X = df_engineered[numerical_features + categorical_features]
    preprocessor.fit(X, y)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Preprocessor saved!")

# === NORMAL: Load saved preprocessor ===
print("Loading saved preprocessor...")
preprocessor = joblib.load("models/preprocessor.pkl")
print("Preprocessor loaded successfully!")


def predict_customer(raw_data: dict) -> dict:
    df = pd.DataFrame([raw_data])
    df = df[numerical_features + categorical_features]
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]

    return {
        "risk_probability": round(prob, 4),
        "credit_score": round(850 - 500 * prob),
        "optimal_loan_amount": round(1000 / (prob + 0.01), 2),
        "optimal_duration_months": 12 if prob < 0.5 else 6,
        "risk_level": "high" if prob >= 0.5 else "low"
    }


if __name__ == "__main__":
    sample = {
        'Value': 5000.0,
        'TransactionHour': 14,
        'TransactionDay': 10,
        'TransactionMonth': 3,
        'TransactionYear': 2019,
        'TotalAmount': 15000.0,
        'AvgAmount': 3000.0,
        'TransactionCount': 5,
        'StdAmount': 1200.0,
        'ProductCategory': 'airtime',
        'ChannelId': 'ChannelId_3',
        'PricingStrategy': 2,
        'ProviderId': 'ProviderId_4',
        'ProductId': 'ProductId_1',
        'CurrencyCode': 'UGX',
        'CountryCode': 256
    }

    result = predict_customer(sample)
    print("\n=== Credit Risk Prediction ===")
    for k, v in result.items():
        print(f"{k.replace('_', ' ').title():25}: {v}")