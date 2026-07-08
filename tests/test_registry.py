"""End-to-end check of the registry workflow: train registers a version,
promotion moves it to Production, and the Production model loads and predicts.
Runs against an isolated SQLite registry in a temp directory.
"""

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src import train
from src.config import settings
from src.registry import get_production_version, promote


@pytest.fixture()
def isolated_registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", uri)
    # Keep mlruns artifacts inside the temp directory too.
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(uri)
    return uri


def test_train_registers_and_promotion_loads(isolated_registry):
    train.train(n_samples=300, max_depth=3, n_estimators=20)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.REGISTERED_MODEL_NAME}'")
    assert len(versions) >= 1

    latest = max(int(v.version) for v in versions)
    promote(latest, "Production")

    prod = get_production_version()
    assert prod is not None
    assert int(prod.version) == latest

    model = mlflow.xgboost.load_model(
        f"models:/{settings.REGISTERED_MODEL_NAME}@{settings.PRODUCTION_ALIAS}"
    )
    sample = pd.DataFrame(
        [{"tenure": 24, "monthly_charges": 65.0, "num_products": 2,
          "has_internet": 1, "contract_type": 0}]
    )
    proba = float(model.predict_proba(sample)[0][1])
    assert 0.0 <= proba <= 1.0
