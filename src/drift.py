"""
Data drift detection for the churn pipeline.

Compares live prediction inputs against the reference dataset
saved during training and generates an HTML report.

Usage:
    python -m src.drift
"""

import sqlite3

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from loguru import logger

from src.config import settings


def load_reference() -> pd.DataFrame:
    return pd.read_csv(settings.REFERENCE_PATH)[settings.NUMERICAL_FEATURES]


def load_current(limit: int = 200) -> pd.DataFrame:
    if not settings.DB_PATH.exists():
        return pd.DataFrame(columns=settings.NUMERICAL_FEATURES)
    conn = sqlite3.connect(settings.DB_PATH)
    rows = conn.execute(
        f"SELECT {', '.join(settings.NUMERICAL_FEATURES)} FROM predictions "
        "ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=settings.NUMERICAL_FEATURES)


def run_drift_report(min_samples: int = 10) -> dict:
    if not settings.REFERENCE_PATH.exists():
        logger.warning("No reference data. Run training first.")
        return {"drift_detected": False, "reason": "no_reference_data"}

    reference = load_reference()
    current = load_current()

    if len(current) < min_samples:
        logger.warning(f"Not enough data ({len(current)} rows, need {min_samples}).")
        return {"drift_detected": False, "reason": "insufficient_data"}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(settings.REPORT_PATH))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    logger.info(f"Drift detected: {drift_detected}")
    logger.info(f"Report saved to {settings.REPORT_PATH}")
    return {"drift_detected": drift_detected, "current_rows": len(current)}


if __name__ == "__main__":
    run_drift_report()
