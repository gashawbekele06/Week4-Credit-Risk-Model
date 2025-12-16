# src/api/main.py
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from pydantic_models import CustomerInput, PredictionOutput

app = FastAPI(title="Bati Bank Credit Risk API", version="1.0")

# Load the best registered model
MODEL_NAME = "BatiBankCreditRiskModel"
MODEL_STAGE = "Production"  # or "Staging"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.get("/")
def home():
    return {"message": "Bati Bank Credit Risk API - /predict for inference"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: CustomerInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Predict probability
    prob = model.predict_proba(df)[0, 1]
    risk = int(prob > 0.5)
    
    return PredictionOutput(
        risk_probability=float(prob),
        is_high_risk=risk
    )