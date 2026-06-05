import pytest
from fastapi.testclient import TestClient

from src.serve import app

client = TestClient(app)

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


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_response():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_high_risk_customer_churns():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    assert response.json()["churn"] is True


def test_low_risk_customer_stays():
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    assert response.json()["churn"] is False


def test_logs_returns_list():
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_counts():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "churn_rate" in data
