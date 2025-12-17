from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from .pydantic_models import CustomerInput, PredictionResponse

app = FastAPI(title="Bati Bank Credit Risk API")

# Load the best model from MLflow registry (you registered it in Task 5)
model = mlflow.sklearn.load_model("models:/CreditRiskProxyBest/1")   # change version if needed

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: CustomerInput):
    try:
        # Convert input to DataFrame (exactly the same columns as training)
        df = pd.DataFrame([input_data.dict()])

        # Make prediction
        prob = model.predict_proba(df)[:, 1][0]

        # Credit score (Task 4 requirement)
        credit_score = 850 - 500 * prob

        # Optimal loan (simple business rule)
        optimal_amount = 1000 / (prob + 0.01)   # avoid division by zero
        optimal_duration = 12 if prob < 0.5 else 6

        return PredictionResponse(
            risk_probability=round(prob, 4),
            credit_score=round(credit_score),
            optimal_loan_amount=round(optimal_amount),
            optimal_duration_months=optimal_duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}