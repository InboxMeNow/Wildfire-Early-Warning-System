"""Daily forecast risk alerts: publish all high-risk grid cells for today.

Runs at 00:00 Asia/Ho_Chi_Minh every day, reading the latest inference
GeoJSON and pushing one Kafka alert per high-risk grid cell whose
``date == today`` (Asia/Ho_Chi_Minh). The notification worker dispatches
the alerts to Telegram and email subscribers.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG

from wildfire_airflow_utils import DEFAULT_ARGS, PROJECT_DIR, python_module_task


with DAG(
    dag_id="daily_forecast_alerts_dag",
    default_args=DEFAULT_ARGS,
    description="Send Telegram/email alerts for today's high-risk forecast cells.",
    # Airflow scheduler runs in UTC. 17:00 UTC == 00:00 Asia/Ho_Chi_Minh.
    schedule="0 17 * * *",
    start_date=datetime(2026, 5, 1),
    catchup=False,
    max_active_runs=1,
    tags=["wildfire", "alerts", "forecast"],
) as dag:
    publish_today = python_module_task(
        task_id="publish_forecast_alerts_today",
        module_name="src.alerts.forecast_publisher",
        extra_args=[
            "--geojson",
            str(PROJECT_DIR / "reports" / "fire_risk_forecast_latest.geojson"),
            "--min-severity",
            "high",
            "--target-date",
            "today",
            "--target-timezone",
            "Asia/Ho_Chi_Minh",
            "--max-alerts",
            "200",
        ],
    )

    publish_today
