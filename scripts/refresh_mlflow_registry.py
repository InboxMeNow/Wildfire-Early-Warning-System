#!/usr/bin/env python3
"""Refresh MLflow Registry versions from the trained Spark models in MinIO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


DEFAULT_MINIO_ENDPOINT = "http://minio:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"
REGISTERED_MODEL_NAMES = {
    "rf_tuned": "wildfire-rf-tuned",
    "gbt": "wildfire-gbt",
}
COMMON_ARTIFACTS = [
    Path("reports/model_metrics_week1.json"),
    Path("reports/model_comparison_week1.csv"),
    Path("reports/feature_importance_week1.csv"),
    Path("reports/feature_importance_week1.png"),
    Path("reports/threshold_optimization_week1.csv"),
    Path("reports/threshold_optimization_week1.png"),
    Path("reports/calibration_curves_week1.csv"),
    Path("reports/calibration_curves_week1.png"),
]


def s3a_path(bucket: str, prefix: str) -> str:
    return f"s3a://{bucket}/{prefix.strip('/')}/"


def build_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName(args.spark_app_name)
    if args.spark_master:
        builder = builder.master(args.spark_master)
    if args.hadoop_aws_package:
        builder = builder.config("spark.jars.packages", args.hadoop_aws_package)
    builder = builder.config("spark.sql.session.timeZone", args.spark_timezone)
    spark = builder.getOrCreate()

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", args.minio_endpoint)
    hadoop_conf.set("fs.s3a.access.key", args.minio_access_key)
    hadoop_conf.set("fs.s3a.secret.key", args.minio_secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.sparkContext.setLogLevel(args.spark_log_level)
    return spark


def log_metrics_with_aliases(model_metrics: dict[str, object]) -> None:
    logged = {}
    for key, value in model_metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            logged[key] = float(value)
    if "auc_roc" in logged:
        logged["AUC"] = logged["auc_roc"]
    if "auc_pr" in logged:
        logged["AUC_PR"] = logged["auc_pr"]
    if "validation_auc_roc" in logged:
        logged["validation_AUC"] = logged["validation_auc_roc"]
    if "validation_auc_pr" in logged:
        logged["validation_AUC_PR"] = logged["validation_auc_pr"]
    mlflow.log_metrics(logged)


def transition_to_production(client, registered_model_name: str, version: str) -> None:
    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(registered_model_name, version, "deployment_stage", "Production")
    client.set_registered_model_tag(registered_model_name, "production_version", version)
    try:
        client.set_registered_model_alias(registered_model_name, "production", version)
    except Exception:
        pass

def register_spark_pipeline_model(
    client,
    registered_model_name: str,
    source_uri: str,
    run_id: str,
) -> str:
    try:
        client.create_registered_model(registered_model_name)
    except Exception:
        pass
    version = client.create_model_version(
        name=registered_model_name,
        source=source_uri.rstrip("/") + "/",
        run_id=run_id,
    )
    return str(version.version)


def refresh_model(
    model_key: str,
    metrics: dict[str, object],
) -> None:
    model_metrics = metrics[model_key]
    if not isinstance(model_metrics, dict):
        raise RuntimeError(f"Missing metrics block for {model_key}")

    registered_name = REGISTERED_MODEL_NAMES[model_key]
    source_uri = str(model_metrics["model_output"])
    model = PipelineModel.load(source_uri)
    client = mlflow.tracking.MlflowClient()
    feature_columns = metrics.get("feature_columns", [])

    with mlflow.start_run(run_name=f"{model_key}-registry-refresh"):
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tags({"run_role": "model_registry_refresh", "model_family": model_key})
        params = dict(model_metrics.get("params", {}))
        params.update(
            {
                "model_output": source_uri,
                "feature_count": len(feature_columns) if isinstance(feature_columns, list) else 0,
                "feature_columns": json.dumps(feature_columns),
                "registry_refresh": True,
            }
        )
        mlflow.log_params(params)
        log_metrics_with_aliases(model_metrics)
        for artifact_path in COMMON_ARTIFACTS:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        registered_version = register_spark_pipeline_model(client, registered_name, source_uri, run_id)
        client.set_model_version_tag(registered_name, registered_version, "spark_pipeline_uri", source_uri)
        if metrics.get("best_model") == model_key:
            transition_to_production(client, registered_name, registered_version)
            model_metrics["registered_model_stage"] = "Production"
            model_metrics["production_model_uri"] = f"models:/{registered_name}/Production"
        else:
            model_metrics.pop("registered_model_stage", None)
            model_metrics.pop("production_model_uri", None)

        model_metrics.update(
            {
                "mlflow_run_id": run_id,
                "registered_model_name": registered_name,
                "registered_model_source": source_uri,
                "registered_model_version": registered_version,
                "registered_model_uri": f"models:/{registered_name}/{registered_version}",
            }
        )
        model_metrics.pop("mlflow_model_uri", None)
        print(f"{model_key}: registered {registered_name} version {registered_version} from run {run_id}")


def log_metrics_artifact_to_run(run_id: str | None, metrics_path: Path) -> None:
    if not run_id or not metrics_path.exists():
        return
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(metrics_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh MLflow Registry from trained Spark models.")
    parser.add_argument("--spark-app-name", default="Wildfire MLflow Registry Refresh")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--mlflow-registry-uri", default=os.getenv("MLFLOW_REGISTRY_URI"))
    parser.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "wildfire-prediction"))
    parser.add_argument("--spark-timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))
    parser.add_argument("--spark-log-level", default="WARN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spark = build_spark(args)
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_registry_uri(args.mlflow_registry_uri or args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    try:
        metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))
        refresh_model("rf_tuned", metrics)
        refresh_model("gbt", metrics)
        args.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for model_key in ("rf_tuned", "gbt"):
            model_metrics = metrics.get(model_key, {})
            if isinstance(model_metrics, dict):
                run_id = model_metrics.get("mlflow_run_id")
                log_metrics_artifact_to_run(run_id if isinstance(run_id, str) else None, args.metrics_path)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
