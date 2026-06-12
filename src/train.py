"""
Churn prediction training script.

Generates synthetic customer data, trains an XGBoost classifier,
and logs everything to MLflow.

Usage:
    python -m src.train
    python -m src.train --n-samples 2000 --max-depth 5
"""

import argparse

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import settings


def generate_data(n_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(settings.CONTRACT_TYPES, size=n_samples, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 72, size=n_samples)
    monthly_charges = rng.uniform(20, 120, size=n_samples).round(2)
    num_products = rng.integers(1, 6, size=n_samples)
    has_internet = rng.integers(0, 2, size=n_samples)

    # Customers on month-to-month contracts with high charges churn more
    churn_prob = (
        0.05
        + 0.35 * (contract == "month-to-month")
        + 0.15 * (monthly_charges > 80)
        - 0.10 * (tenure > 24)
        - 0.05 * (num_products > 3)
    ).clip(0.02, 0.95)

    churn = rng.binomial(1, churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "num_products": num_products,
        "has_internet": has_internet,
        "contract_type": contract,
        "churn": churn,
    })


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df = df.copy()
    df["contract_type"] = le.fit_transform(df["contract_type"])
    return df, le


def train(n_samples: int, max_depth: int, n_estimators: int) -> None:
    mlflow.set_experiment(settings.EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({
            "n_samples": n_samples,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
        })

        logger.info(f"Generating {n_samples} samples...")
        df = generate_data(n_samples, seed=settings.RANDOM_STATE)
        df, le = preprocess(df)

        X = df.drop(columns=[settings.TARGET])
        y = df[settings.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
        )

        model = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            eval_metric="logloss",
            random_state=settings.RANDOM_STATE,
        )
        logger.info("Training model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "label_encoder": le}, settings.MODEL_PATH)
        df.drop(columns=[settings.TARGET]).head(200).to_csv(settings.REFERENCE_PATH, index=False)

        logger.success(f"Model saved to {settings.MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    train(args.n_samples, args.max_depth, args.n_estimators)
