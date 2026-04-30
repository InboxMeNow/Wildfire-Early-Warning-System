"""Weekly Spark feature rebuild, model retraining, and downstream reports."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG

from wildfire_airflow_utils import DEFAULT_ARGS, MINIO_ARGS, PROJECT_DIR, python_script_task, spark_script_task


with DAG(
    dag_id="model_retrain_dag",
    default_args=DEFAULT_ARGS,
    description="Retrain wildfire risk models weekly and refresh derived artifacts.",
    schedule="0 2 * * 0",
    start_date=datetime(2026, 4, 28),
    catchup=False,
    max_active_runs=1,
    tags=["wildfire", "model", "spark"],
) as dag:
    clean_data = spark_script_task(
        task_id="clean_with_spark",
        script_name="04_etl_clean.py",
        extra_args=["--country-boundary", str(PROJECT_DIR / "geo" / "vietnam_boundary.geojson")],
    )

    build_features = spark_script_task(
        task_id="build_features",
        script_name="05_feature_engineering.py",
        extra_args=["--print-counts"],
    )

    train_models = spark_script_task(
        task_id="train_models",
        script_name="07_train_model.py",
        extra_args=[
            "--metrics-output",
            str(PROJECT_DIR / "reports" / "model_metrics_week1.json"),
            "--importance-output",
            str(PROJECT_DIR / "reports" / "feature_importance_week1.csv"),
        ],
    )

    data_quality = python_script_task(
        task_id="data_quality_report",
        script_name="06_data_quality_and_heatmap.py",
        extra_args=MINIO_ARGS,
    )

    cluster_recent_fires = python_script_task(
        task_id="cluster_recent_fires",
        script_name="07_dbscan_clustering.py",
        extra_args=MINIO_ARGS,
    )

    detect_anomalies = python_script_task(
        task_id="detect_anomalies",
        script_name="08_anomaly_detection.py",
        extra_args=MINIO_ARGS,
    )

    next_day_inference = spark_script_task(
        task_id="next_day_inference",
        script_name="09_inference.py",
        extra_args=[
            "--geojson-output",
            str(PROJECT_DIR / "reports" / "fire_risk_forecast_latest.geojson"),
            "--metadata-output",
            str(PROJECT_DIR / "reports" / "fire_risk_forecast_latest.json"),
            "--print-progress",
        ],
    )

    clean_data >> build_features
    build_features >> [data_quality, cluster_recent_fires, detect_anomalies, train_models]
    train_models >> next_day_inference
