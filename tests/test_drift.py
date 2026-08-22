"""Drift detection tests.

The reference file, the prediction database and the report all live under
tmp_path, so nothing here touches the repo's data/ directory.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import drift
from src.config import settings

FEATURES = settings.NUMERICAL_FEATURES


def make_reference(rng, n: int = 120) -> pd.DataFrame:
    """Customers shaped like the ones train.py writes to reference.csv."""
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "monthly_charges": rng.normal(60.0, 10.0, n).round(2),
        "num_products": rng.integers(1, 6, n),
        "has_internet": rng.integers(0, 2, n),
        "contract_type": rng.integers(0, 3, n),
    })


def make_shifted(rng, n: int = 120) -> pd.DataFrame:
    """New, expensive, single-product customers who all have internet."""
    return pd.DataFrame({
        "tenure": rng.integers(1, 4, n),
        "monthly_charges": rng.normal(115.0, 3.0, n).round(2),
        "num_products": rng.integers(1, 3, n),
        "has_internet": (rng.random(n) < 0.95).astype(int),
        "contract_type": np.zeros(n, dtype=int),
    })


def write_current(db_path, frame: pd.DataFrame) -> None:
    """Persist rows the way serve.py's log_prediction would have."""
    placeholders = ", ".join("?" for _ in FEATURES)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE predictions (
                   id               INTEGER PRIMARY KEY AUTOINCREMENT,
                   tenure           INTEGER,
                   monthly_charges  REAL,
                   num_products     INTEGER,
                   has_internet     INTEGER,
                   contract_type    TEXT,
                   churn            INTEGER,
                   probability      REAL,
                   timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        conn.executemany(
            f"INSERT INTO predictions ({', '.join(FEATURES)}) VALUES ({placeholders})",
            frame[FEATURES].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    database = tmp_path / "predictions.db"
    report = tmp_path / "drift_report.html"
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "REFERENCE_PATH", reference)
    monkeypatch.setattr(settings, "DB_PATH", database)
    monkeypatch.setattr(settings, "REPORT_PATH", report)
    return {"reference": reference, "database": database, "report": report}


def test_drift_detected_when_customers_shift(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(paths["database"], make_shifted(rng))

    result = drift.run_drift_report()

    assert result["drift_detected"]
    assert result["current_rows"] == 120
    assert paths["report"].exists()


def test_no_drift_when_current_matches_reference(paths):
    rng = np.random.default_rng(0)
    reference = make_reference(rng)
    reference.to_csv(paths["reference"], index=False)
    write_current(paths["database"], reference.copy())

    result = drift.run_drift_report()

    assert not result["drift_detected"]


def test_missing_reference_is_reported_not_raised(paths):
    result = drift.run_drift_report()
    assert result == {"drift_detected": False, "reason": "no_reference_data"}


def test_too_few_rows_is_reported_not_raised(paths):
    rng = np.random.default_rng(0)
    make_reference(rng).to_csv(paths["reference"], index=False)
    write_current(paths["database"], make_shifted(rng, n=5))

    result = drift.run_drift_report()

    assert result == {"drift_detected": False, "reason": "insufficient_data"}
