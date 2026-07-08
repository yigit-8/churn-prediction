"""
MLflow Model Registry operations for the churn model.

New versions are created automatically by train.py (registered_model_name).
This module handles the deliberate part: promoting a version to a stage and
inspecting what is registered.

Promotion is implemented with registry aliases ("staging" / "production"),
the modern replacement for the stages API that MLflow deprecated in 2.9.
A version carries the "production" alias when it is the one serving should load.

Usage:
    python -m src.registry list
    python -m src.registry promote --version 3 --stage Production
"""

import argparse

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from src.config import settings

STAGE_ALIASES = {"Staging": "staging", "Production": "production"}


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    return MlflowClient()


def alias_map() -> dict[str, list[str]]:
    """Return {version: [aliases]} for the registered model.

    search_model_versions does not populate aliases in MLflow 2.13, so the
    authoritative source is the registered model's {alias: version} map.
    """
    client = get_client()
    model = client.get_registered_model(settings.REGISTERED_MODEL_NAME)
    result: dict[str, list[str]] = {}
    for alias, version in model.aliases.items():
        result.setdefault(str(version), []).append(alias)
    return result


def list_versions() -> list:
    client = get_client()
    versions = sorted(
        client.search_model_versions(f"name='{settings.REGISTERED_MODEL_NAME}'"),
        key=lambda v: int(v.version),
    )
    aliases_by_version = alias_map()
    for v in versions:
        aliases = ", ".join(aliases_by_version.get(v.version, [])) or "-"
        logger.info(f"version={v.version} aliases=[{aliases}] run_id={v.run_id}")
    if not versions:
        logger.warning(f"No versions registered for '{settings.REGISTERED_MODEL_NAME}'.")
    return versions


def promote(version: int, stage: str) -> None:
    alias = STAGE_ALIASES[stage]
    client = get_client()
    client.set_registered_model_alias(
        name=settings.REGISTERED_MODEL_NAME,
        alias=alias,
        version=str(version),
    )
    logger.success(
        f"Version {version} of '{settings.REGISTERED_MODEL_NAME}' is now @{alias} ({stage})."
    )


def get_production_version():
    client = get_client()
    try:
        return client.get_model_version_by_alias(
            settings.REGISTERED_MODEL_NAME, settings.PRODUCTION_ALIAS
        )
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the churn model registry.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List all registered versions and their aliases.")
    promote_parser = sub.add_parser("promote", help="Promote a version to a stage.")
    promote_parser.add_argument("--version", type=int, required=True)
    promote_parser.add_argument("--stage", default="Production", choices=list(STAGE_ALIASES))
    args = parser.parse_args()

    if args.command == "list":
        list_versions()
    elif args.command == "promote":
        promote(args.version, args.stage)
