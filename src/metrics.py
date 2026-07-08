"""
Prometheus metrics for the churn API.

Two kinds of signals are exposed on /metrics:

- Service metrics (request count and latency per endpoint) collected by an
  ASGI middleware. Labels use the route template ("/predict"), never raw
  paths, to keep label cardinality bounded.
- Model metrics (prediction outcomes and probability distribution) recorded
  where predictions happen. A drifting probability histogram or a shifting
  churn/retain ratio is often the first visible symptom of data drift.
"""

import time

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "churn_predictions_total",
    "Predictions served, labelled by outcome.",
    ["outcome"],
)

PREDICTION_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribution of predicted churn probabilities.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def record_prediction(churn: bool, probability: float) -> None:
    PREDICTIONS.labels(outcome="churn" if churn else "retain").inc()
    PREDICTION_PROBABILITY.observe(probability)


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    # Unmatched paths (404s) carry no route; skip them so scanners and typos
    # cannot grow the label space.
    if route is not None and route.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method, path=route.path, status=response.status_code
        ).inc()
        HTTP_REQUEST_DURATION.labels(method=request.method, path=route.path).observe(
            time.perf_counter() - start
        )
    return response


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
