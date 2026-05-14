#!/usr/bin/env python3
"""Verify MLflow tracking and registry state for the wildfire models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow


REQUIRED_MODELS = {
    "rf_tuned": "wildfire-rf-tuned",
    "gbt": "wildfire-gbt",
}

def expected_production_model(metrics: dict[str, object]) -> str:
    best_model = metrics.get("best_model")
    if isinstance(best_model, str) and best_model in REQUIRED_MODELS:
        return REQUIRED_MODELS[best_model]
    return REQUIRED_MODELS["gbt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MLflow wildfire model registry state.")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "wildfire-prediction"))
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--min-runs", type=int, default=8)
    return parser.parse_args()


def model_versions(client, name: str) -> list[dict[str, object]]:
    versions = sorted(client.search_model_versions("name='{}'".format(name)), key=lambda item: int(item.version))
    return [
        {
            "version": version.version,
            "stage": version.current_stage,
            "status": version.status,
            "source": version.source,
            "run_id": version.run_id,
            "tags": dict(version.tags),
        }
        for version in versions
    ]


def main() -> int:
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    runs = client.search_runs([experiment.experiment_id], max_results=1000) if experiment else []
    registered_model_names = {model.name for model in client.search_registered_models()}
    metrics = json.loads(args.metrics_path.read_text(encoding="utf-8")) if args.metrics_path.exists() else {}
    production_model_name = expected_production_model(metrics)

    summary = {
        "experiment": args.experiment_name,
        "experiment_id": experiment.experiment_id if experiment else None,
        "run_count": len(runs),
        "run_names": sorted(run.data.tags.get("mlflow.runName", "") for run in runs),
        "registered_models": sorted(registered_model_names),
        "versions": {name: model_versions(client, name) for name in REQUIRED_MODELS.values()},
        "metrics_best_model": metrics.get("best_model"),
        "expected_production_model": production_model_name,
        "metrics_registered_uris": {
            key: metrics.get(key, {}).get("registered_model_uri")
            for key in REQUIRED_MODELS
            if isinstance(metrics.get(key), dict)
        },
    }

    failures = []
    if experiment is None:
        failures.append("missing experiment")
    if len(runs) < args.min_runs:
        failures.append(f"run count {len(runs)} < {args.min_runs}")
    for model_name in REQUIRED_MODELS.values():
        if model_name not in registered_model_names:
            failures.append(f"missing registered model {model_name}")
    production_versions = summary["versions"].get(production_model_name, [])
    if not any(version["stage"] == "Production" and version["status"] == "READY" for version in production_versions):
        failures.append(f"{production_model_name} has no READY Production version")
    rf_versions = summary["versions"].get(REQUIRED_MODELS["rf_tuned"], [])
    if not any(version["status"] == "READY" for version in rf_versions):
        failures.append("wildfire-rf-tuned has no READY version")

    summary["status"] = "ok" if not failures else "fail"
    summary["failures"] = failures
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
