# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import logging
import sys
import os

# -------------------------------------------------
# PATH FIX: Add project root to sys.path when running as script
# -------------------------------------------------
if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[INFO] Added project root to sys.path: {project_root}")

from src.data_loader import DataLoader
from src.data_processing import FeatureEngineer
from src.proxy_target import ProxyTargetEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Switch to SQLite backend to avoid deprecation warning
mlflow.set_experiment("BatiBank_CreditRisk")

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_auc = 0.0  # Initialize best_auc to avoid AttributeError
        self.best_model_name = None

    def prepare_data(self):
        logger.info("Loading and preparing data...")
        loader = DataLoader()
        df = loader.load()
        
        engineer = FeatureEngineer()
        features_df = engineer.fit_transform(df)  # Has CustomerId
        
        proxy = ProxyTargetEngineer()
        target_df = proxy.create_proxy_target(df)
        
        merged = features_df.merge(target_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        merged['is_high_risk'] = merged['is_high_risk'].fillna(0).astype(int)
        
        X = merged.drop(columns=['CustomerId', 'is_high_risk'])
        y = merged['is_high_risk']
        
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

    def train_and_log(self, model, name, X_train, X_test, y_train, y_test):
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob)
            }
            
            mlflow.log_params({"model": name, "random_state": self.random_state})
            mlflow.log_metrics(metrics)
            
            # Use sklearn flavor for both (fixes XGBoost logging issue)
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"{name} - AUC: {metrics['roc_auc']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > self.best_auc:
                self.best_auc = metrics['roc_auc']
                self.best_model_name = name
                logger.info(f"New best model: {name} (AUC: {self.best_auc:.4f})")

    def run(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.train_and_log(lr, "LogisticRegression", X_train, X_test, y_train, y_test)
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.train_and_log(xgb, "XGBoost", X_train, X_test, y_train, y_test)
        
        # Register best model
        if self.best_model_name:
            # Get the best run
            runs = mlflow.search_runs(order_by=["metrics.roc_auc DESC"], max_results=1)
            best_run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{best_run_id}/model"
            registered = mlflow.register_model(model_uri, "BatiBankCreditRiskModel")
            logger.info(f"Best model ({self.best_model_name}) registered as BatiBankCreditRiskModel v{registered.version}")

        print("\nTraining complete. View results at http://127.0.0.1:5000")
        print("Run 'mlflow ui' to start the tracking server.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()