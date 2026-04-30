"""Ingest recent FIRMS detections, publish to Kafka, and archive to MinIO."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = Path(os.getenv("AIRFLOW_PROJECT_DIR", "/opt/airflow/project"))
SRC_DIR = PROJECT_DIR / "src"
for candidate in (PROJECT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from orchestration.airflow_tasks import (  # noqa: E402
    archive_recent_fires_to_minio,
    fetch_recent_firms_to_file,
    push_recent_fires_to_kafka,
)
from wildfire_airflow_utils import DEFAULT_ARGS  # noqa: E402


with DAG(
    dag_id="fire_ingestion_dag",
    default_args=DEFAULT_ARGS,
    description="Fetch recent FIRMS fire detections every 3 hours.",
    schedule="0 */3 * * *",
    start_date=datetime(2026, 4, 28),
    catchup=False,
    max_active_runs=1,
    tags=["wildfire", "firms", "kafka"],
) as dag:
    fetch = PythonOperator(
        task_id="fetch_firms",
        python_callable=fetch_recent_firms_to_file,
    )

    to_kafka = PythonOperator(
        task_id="push_to_kafka",
        python_callable=push_recent_fires_to_kafka,
    )

    to_minio = PythonOperator(
        task_id="archive_to_minio",
        python_callable=archive_recent_fires_to_minio,
    )

    fetch >> [to_kafka, to_minio]
