"""
Churn prediction API.

Loads the model saved by train.py and exposes prediction endpoints.
"""

import sqlite3
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.config import settings
from src.metrics import metrics_endpoint, metrics_middleware, record_prediction

model_bundle = None


def load_model():
    global model_bundle

    if settings.USE_REGISTRY:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            uri = f"models:/{settings.REGISTERED_MODEL_NAME}@{settings.PRODUCTION_ALIAS}"
            model = mlflow.xgboost.load_model(uri)
            model_bundle = {"model": model, "source": f"registry@{settings.PRODUCTION_ALIAS}"}
            logger.info(f"Model loaded from registry: {uri}")
            return
        except Exception as exc:
            logger.warning(f"Registry load failed ({exc}); falling back to local file.")

    if not settings.MODEL_PATH.exists():
        logger.error(f"Model not found at {settings.MODEL_PATH}")
        raise RuntimeError("Model not found. Run python -m src.train first.")
    bundle = joblib.load(settings.MODEL_PATH)
    model_bundle = {"model": bundle["model"], "source": "local"}
    logger.info("Model loaded successfully.")


def init_db():
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
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
    logger.info(f"Database initialized at {settings.DB_PATH}")


def log_prediction(features: dict, churn: int, probability: float):
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        conn.execute(
            """INSERT INTO predictions
               (tenure, monthly_charges, num_products, has_internet,
                contract_type, churn, probability)
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
    df = pd.DataFrame([customer_dict])
    df["contract_type"] = df["contract_type"].map(settings.CONTRACT_MAP).astype(int)
    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title=settings.APP_TITLE,
    description="Predicts whether a customer will churn based on account features.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.middleware("http")(metrics_middleware)
app.get("/metrics", include_in_schema=False)(metrics_endpoint)


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


@app.get("/model-info")
def model_info():
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    info = {
        "active_source": model_bundle.get("source"),
        "registered_name": settings.REGISTERED_MODEL_NAME,
    }
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = MlflowClient()
        registered = client.get_registered_model(settings.REGISTERED_MODEL_NAME)
        aliases_by_version: dict[str, list[str]] = {}
        for alias, version in registered.aliases.items():
            aliases_by_version.setdefault(str(version), []).append(alias)
        versions = client.search_model_versions(f"name='{settings.REGISTERED_MODEL_NAME}'")
        info["versions"] = sorted(
            ({"version": int(v.version), "aliases": aliases_by_version.get(v.version, [])}
             for v in versions),
            key=lambda x: x["version"],
        )
    except Exception as exc:
        info["versions"] = f"registry unavailable: {exc}"
    return info


def resolve_threshold(requested: float | None) -> float:
    """Caller's threshold, or the one training chose on held-out data.

    Defaulting to 0.5 would be the wrong operating point here: churn is the
    minority class, so at 0.5 the model scores F1 0.13 against 0.49 at the
    threshold train.py selects. Older model files predate the field.
    """
    if requested is not None:
        return requested
    return float((model_bundle or {}).get("threshold", 0.5))


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures,
    threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Churn probability threshold. Defaults to the value chosen at training time.",
    ),
):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    threshold = resolve_threshold(threshold)
    df = prepare_features(customer.model_dump())
    probability = float(model_bundle["model"].predict_proba(df)[0][1])
    churn = probability >= threshold

    log_prediction(customer.model_dump(), int(churn), probability)
    record_prediction(churn, probability)
    return PredictionResponse(churn=churn, probability=round(probability, 4), threshold=threshold)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    customers: list[CustomerFeatures],
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list cannot be empty.")

    threshold = resolve_threshold(threshold)
    results = []
    for customer in customers:
        df = prepare_features(customer.model_dump())
        probability = float(model_bundle["model"].predict_proba(df)[0][1])
        churn = probability >= threshold
        log_prediction(customer.model_dump(), int(churn), probability)
        record_prediction(churn, probability)
        results.append(
            PredictionResponse(churn=churn, probability=round(probability, 4), threshold=threshold)
        )

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
    conn = sqlite3.connect(settings.DB_PATH)
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
    conn = sqlite3.connect(settings.DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    churned = conn.execute("SELECT COUNT(*) FROM predictions WHERE churn = 1").fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "churn_count": churned,
        "retention_count": total - churned,
        "churn_rate": round(churned / total, 4) if total > 0 else 0.0,
    }
