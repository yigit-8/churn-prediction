import pytest
from fastapi.testclient import TestClient

from src.serve import app

VALID_CUSTOMER = {
    "tenure": 24,
    "monthly_charges": 65.0,
    "num_products": 2,
    "has_internet": 1,
    "contract_type": "month-to-month",
}

HIGH_RISK_CUSTOMER = {
    "tenure": 3,
    "monthly_charges": 99.0,
    "num_products": 1,
    "has_internet": 1,
    "contract_type": "month-to-month",
}

LOW_RISK_CUSTOMER = {
    "tenure": 60,
    "monthly_charges": 30.0,
    "num_products": 4,
    "has_internet": 0,
    "contract_type": "two_year",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probability" in data
    assert "threshold" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_with_custom_threshold(client):
    response = client.post("/predict?threshold=0.3", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    assert response.json()["threshold"] == 0.3


def test_high_risk_customer_has_higher_probability(client):
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()["probability"]
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()["probability"]
    assert high > low


def test_batch_predict(client):
    response = client.post("/predict/batch", json=[HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "churn_count" in data
    assert len(data["results"]) == 2


def test_batch_predict_empty_list_returns_400(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400


def test_feature_importance(client):
    response = client.get("/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    features = [f["feature"] for f in data["feature_importance"]]
    assert "tenure" in features
    assert "monthly_charges" in features


def test_logs_returns_list(client):
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_counts(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "churn_rate" in data
