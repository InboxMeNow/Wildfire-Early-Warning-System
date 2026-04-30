"""Refresh daily weather data and rebuild Spark features."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG

from wildfire_airflow_utils import DEFAULT_ARGS, MINIO_ARGS, PROJECT_DIR, python_script_task, spark_script_task


with DAG(
    dag_id="weather_update_dag",
    default_args=DEFAULT_ARGS,
    description="Fetch daily weather observations and refresh cleaned/features datasets.",
    schedule="15 1 * * *",
    start_date=datetime(2026, 4, 28),
    catchup=False,
    max_active_runs=1,
    tags=["wildfire", "weather", "spark"],
) as dag:
    fetch_weather = python_script_task(
        task_id="fetch_weather",
        script_name="02_fetch_weather.py",
        extra_args=[
            "--start-date",
            "{{ ds }}",
            "--end-date",
            "{{ ds }}",
            "--local-output",
            "data/raw/meteostat_daily_vietnam_{{ ds }}.parquet",
            "--parts-dir",
            "data/raw/weather_meteostat_vietnam_parts_{{ ds_nodash }}",
            "--object-name",
            "meteostat_daily_vietnam_{{ ds }}.parquet",
            "--request-delay-seconds",
            "0",
            *MINIO_ARGS,
        ],
    )

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

    fetch_weather >> clean_data >> build_features
