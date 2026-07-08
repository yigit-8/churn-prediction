import pytest
from fastapi.testclient import TestClient

from src.metrics import PREDICTIONS
from src.serve import app

CUSTOMER = {
    "tenure": 24,
    "monthly_charges": 65.0,
    "num_products": 2,
    "has_internet": 1,
    "contract_type": "month-to-month",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_request_metrics_use_route_template(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert 'http_requests_total{method="GET",path="/health",status="200"}' in body


def test_prediction_metrics_are_recorded(client):
    before = PREDICTIONS.labels(outcome="churn")._value.get() + PREDICTIONS.labels(
        outcome="retain"
    )._value.get()

    client.post("/predict", json=CUSTOMER)

    body = client.get("/metrics").text
    assert "churn_predictions_total" in body
    assert "churn_prediction_probability_bucket" in body

    after = PREDICTIONS.labels(outcome="churn")._value.get() + PREDICTIONS.labels(
        outcome="retain"
    )._value.get()
    assert after == before + 1


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")
    body = client.get("/metrics").text
    assert 'path="/metrics"' not in body
