# MLOps Churn Prediction

Predicts customer churn using XGBoost, with full MLOps tooling: experiment tracking via MLflow, data drift detection via Evidently, and automated CI/CD via GitHub Actions.

## How it works

A customer's account features go in, and the model outputs whether that customer is likely to churn along with a probability score. Every prediction is stored and monitored for drift over time.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Model | XGBoost Classifier |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |

## Quick Start

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Train the model**

```bash
python src/train.py
mlflow ui
```

Open http://localhost:5000 to browse experiments and compare runs.

**Serve the API**

```bash
uvicorn src.serve:app --reload
```

**Check for data drift**

```bash
python src/drift.py
```

**Run with Docker Compose**

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Readiness probe |
| POST | `/predict` | Predict churn for a customer |
| GET | `/logs` | Recent predictions |
| GET | `/stats` | Churn rate and counts |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 99.0,
    "num_products": 1,
    "has_internet": 1,
    "contract_type": "month-to-month"
  }'
```

```json
{"churn": true, "probability": 0.8731}
```

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` runs the test suite and then builds and smoke-tests the Docker image automatically.
