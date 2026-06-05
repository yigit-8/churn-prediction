"""
Churn prediction API.

Loads the model saved by train.py and exposes prediction endpoints.
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")

model_bundle = None


def load_model():
    global model_bundle
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")
    model_bundle = joblib.load(MODEL_PATH)
    print("Model loaded.")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            tenure           INTEGER,
            monthly_charges  REAL,
            num_products     INTEGER,
            has_internet     INTEGER,
            contract_type    TEXT,
            churn            INTEGER,
            probability      REAL,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(features: dict, churn: int, probability: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO predictions
               (tenure, monthly_charges, num_products, has_internet, contract_type, churn, probability)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                features["tenure"],
                features["monthly_charges"],
                features["num_products"],
                features["has_internet"],
                features["contract_type"],
                churn,
                probability,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def prepare_features(customer_dict: dict) -> pd.DataFrame:
    le = model_bundle["label_encoder"]
    df = pd.DataFrame([customer_dict])
    df["contract_type"] = le.transform(df["contract_type"])
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a customer will churn based on account features.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    num_products: int = Field(..., ge=1, le=10, description="Number of products subscribed")
    has_internet: Literal[0, 1] = Field(..., description="1 if the customer has internet service")
    contract_type: Literal["month-to-month", "one_year", "two_year"]


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    threshold: float


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    churn_count: int


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Churn probability threshold"),
):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = prepare_features(customer.model_dump())
    probability = float(model_bundle["model"].predict_proba(df)[0][1])
    churn = probability >= threshold

    log_prediction(customer.model_dump(), int(churn), probability)
    return PredictionResponse(churn=churn, probability=round(probability, 4), threshold=threshold)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    customers: list[CustomerFeatures],
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list cannot be empty.")

    results = []
    for customer in customers:
        df = prepare_features(customer.model_dump())
        probability = float(model_bundle["model"].predict_proba(df)[0][1])
        churn = probability >= threshold
        log_prediction(customer.model_dump(), int(churn), probability)
        results.append(PredictionResponse(churn=churn, probability=round(probability, 4), threshold=threshold))

    churn_count = sum(1 for r in results if r.churn)
    return BatchPredictionResponse(results=results, total=len(results), churn_count=churn_count)


@app.get("/feature-importance")
def feature_importance():
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model = model_bundle["model"]
    feature_names = ["tenure", "monthly_charges", "num_products", "has_internet", "contract_type"]
    importances = model.feature_importances_.tolist()

    ranked = sorted(
        [{"feature": f, "importance": round(i, 4)} for f, i in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return {"feature_importance": ranked}


@app.get("/logs")
def get_logs(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT tenure, monthly_charges, num_products, has_internet,
                  contract_type, churn, probability, timestamp
           FROM predictions ORDER BY timestamp DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    keys = ["tenure", "monthly_charges", "num_products", "has_internet",
            "contract_type", "churn", "probability", "timestamp"]
    return [dict(zip(keys, row)) for row in rows]


@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    churned = conn.execute("SELECT COUNT(*) FROM predictions WHERE churn = 1").fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "churn_count": churned,
        "retention_count": total - churned,
        "churn_rate": round(churned / total, 4) if total > 0 else 0.0,
    }
