# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd
import joblib
import os

from src.config import PROCESSED_PATH, RANDOM_STATE

# MLflow setup
mlflow.set_tracking_uri("mlruns")  # Local folder
mlflow.set_experiment("credit_risk_proxy_model")

# Load processed data
df = pd.read_csv(PROCESSED_PATH)
print(f"Loaded processed data: {df.shape}")

# Features and target
X = df.drop('is_high_risk', axis=1)
y = df['is_high_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target distribution in train: {y_train.mean():.3f}")

# Models and hyperparameter grids
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    }
}

best_auc = 0
best_model_name = None
best_run_id = None

for name, config in models.items():
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        # Hyperparameter tuning with GridSearchCV
        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # Best estimator
        best_estimator = grid.best_estimator_

        # Predictions
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # Log to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_estimator, "model")

        print(f"{name} - Best ROC-AUC: {metrics['roc_auc']:.4f}")

        # Track best model
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id

            # Save best model locally
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_estimator, f"models/best_model_{name}.pkl")

print(f"\nBest model: {best_model_name} with ROC-AUC = {best_auc:.4f}")

# Register best model in MLflow Registry
with mlflow.start_run(run_id=best_run_id):
    mlflow.register_model(f"runs:/{best_run_id}/model", "CreditRiskProxyBest")

print("Best model registered in MLflow Model Registry as 'CreditRiskProxyBest'")

# Start MLflow UI to compare runs
print("\nTo view experiments, run:")
print("mlflow ui")
print("Then open http://localhost:5000")