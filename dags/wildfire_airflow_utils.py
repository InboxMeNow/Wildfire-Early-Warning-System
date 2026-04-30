"""Shared DAG construction helpers for the wildfire Airflow deployment."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


PROJECT_DIR = Path(os.getenv("AIRFLOW_PROJECT_DIR", "/opt/airflow/project"))
PYTHON_BIN = os.getenv("AIRFLOW_PROJECT_PYTHON", "python")
SPARK_CONN_ID = os.getenv("AIRFLOW_SPARK_CONN_ID", "spark_default")
HADOOP_AWS_PACKAGE = os.getenv("SPARK_HADOOP_AWS_PACKAGE", "org.apache.hadoop:hadoop-aws:3.3.4")

MINIO_ARGS = [
    "--minio-endpoint",
    os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
    "--minio-access-key",
    os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "--minio-secret-key",
    os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "--minio-bucket",
    os.getenv("MINIO_BUCKET", "wildfire-data"),
]

DEFAULT_ARGS = {
    "owner": "wildfire-system",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def project_script(script_name: str) -> str:
    return str(PROJECT_DIR / script_name)


def project_python_command(script_name: str, extra_args: list[str] | None = None) -> str:
    args = " ".join(extra_args or [])
    return f"cd {PROJECT_DIR} && {PYTHON_BIN} {script_name} {args}".strip()


def python_script_task(
    task_id: str,
    script_name: str,
    extra_args: list[str] | None = None,
) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=project_python_command(script_name, extra_args),
        env={
            "PYTHONPATH": f"{PROJECT_DIR / 'src'}:{PROJECT_DIR}",
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET", "wildfire-data"),
        },
        append_env=True,
    )


def spark_script_task(
    task_id: str,
    script_name: str,
    extra_args: list[str] | None = None,
    packages: str | None = HADOOP_AWS_PACKAGE,
) -> SparkSubmitOperator:
    return SparkSubmitOperator(
        task_id=task_id,
        conn_id=SPARK_CONN_ID,
        application=project_script(script_name),
        application_args=[*MINIO_ARGS, *(extra_args or [])],
        packages=packages,
        verbose=False,
        env_vars={
            "PYTHONPATH": f"{PROJECT_DIR / 'src'}:{PROJECT_DIR}",
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET", "wildfire-data"),
        },
    )
