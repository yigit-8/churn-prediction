from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_PATH: Path = DATA_DIR / "model.joblib"
    REFERENCE_PATH: Path = DATA_DIR / "reference.csv"
    DB_PATH: Path = DATA_DIR / "predictions.db"
    REPORT_PATH: Path = DATA_DIR / "drift_report.html"

    # ML Settings
    EXPERIMENT_NAME: str = "churn-prediction"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

    # MLflow / Model Registry
    # A SQLite backend lets the registry track versions and stages without a server.
    MLFLOW_TRACKING_URI: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    REGISTERED_MODEL_NAME: str = "churn-predictor"
    # Promotion uses registry aliases (the modern replacement for the deprecated
    # stages API). "production" marks the version serving should load.
    PRODUCTION_ALIAS: str = "production"
    # When True, serving loads the @production model from the registry instead of
    # the local file. Off by default so the bundled demo image stays self-contained.
    USE_REGISTRY: bool = False

    # API Settings
    APP_TITLE: str = "Churn Prediction API"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Features
    CATEGORICAL_FEATURES: list[str] = ["contract_type"]
    NUMERICAL_FEATURES: list[str] = ["tenure", "monthly_charges", "num_products", "has_internet"]
    TARGET: str = "churn"

    CONTRACT_TYPES: list[str] = ["month-to-month", "one_year", "two_year"]
    # Fixed, deterministic encoding for contract_type. Replaces a pickled
    # LabelEncoder so the registered model is self-contained at serving time.
    CONTRACT_MAP: dict[str, int] = {"month-to-month": 0, "one_year": 1, "two_year": 2}


settings = Settings()
