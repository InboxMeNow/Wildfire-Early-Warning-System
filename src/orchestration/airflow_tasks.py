"""Small Python callables used by the Airflow DAGs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config

from streaming.firms_producer import (
    DEFAULT_BBOX,
    DEFAULT_HOURS,
    DEFAULT_SOURCE,
    DEFAULT_TOPIC,
    create_producer,
    event_key,
    fetch_firms_recent,
)


PROJECT_DIR = Path(os.getenv("AIRFLOW_PROJECT_DIR", ".")).resolve()
DEFAULT_STAGE_DIR = PROJECT_DIR / "data" / "airflow" / "firms_recent"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in {None, ""} else default


def _required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value != "PASTE_YOUR_FIRMS_MAP_KEY_HERE":
            return value
    joined = " or ".join(names)
    raise RuntimeError(f"Missing required environment variable: {joined}")


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _logical_date(context: dict[str, Any]) -> datetime:
    value = context.get("logical_date") or context.get("execution_date")
    if isinstance(value, datetime):
        return value
    return datetime.now(timezone.utc)


def fetch_recent_firms_to_file(**context: Any) -> str:
    """Fetch recent FIRMS NRT rows once and persist them for downstream tasks."""

    map_key = _required_env("FIRMS_MAP_KEY", "MAP_KEY")
    source = _env("FIRMS_SOURCE", DEFAULT_SOURCE)
    bbox = _env("FIRMS_BBOX", DEFAULT_BBOX)
    hours = int(_env("FIRMS_RECENT_HOURS", str(DEFAULT_HOURS)))
    timeout_seconds = int(_env("FIRMS_TIMEOUT_SECONDS", "60"))

    events = fetch_firms_recent(
        map_key=map_key,
        source=source,
        bbox=bbox,
        hours=hours,
        timeout_seconds=timeout_seconds,
    )

    stage_dir = Path(_env("AIRFLOW_FIRMS_STAGE_DIR", str(DEFAULT_STAGE_DIR)))
    stage_dir.mkdir(parents=True, exist_ok=True)

    logical_date = _logical_date(context)
    run_id = _safe_token(str(context.get("run_id", logical_date.isoformat())))
    output_path = stage_dir / f"firms_recent_{logical_date:%Y%m%dT%H%M%S}_{run_id}.json"
    output_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fetched {len(events):,} FIRMS event(s) into {output_path}")
    return str(output_path)


def _load_events(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def push_recent_fires_to_kafka(**context: Any) -> int:
    """Publish the fetched FIRMS JSON payload into the fire-events Kafka topic."""

    path = context["ti"].xcom_pull(task_ids="fetch_firms")
    events = _load_events(path)
    bootstrap_servers = _env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = _env("KAFKA_FIRE_TOPIC", DEFAULT_TOPIC)

    producer = create_producer(bootstrap_servers)
    try:
        for event in events:
            producer.send(topic, key=event_key(event), value=event)
        producer.flush()
    finally:
        producer.close(timeout=10)

    print(f"Published {len(events):,} FIRMS event(s) to Kafka topic {topic!r}")
    return len(events)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=_env("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=_env("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=_env("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def archive_recent_fires_to_minio(**context: Any) -> str:
    """Archive the fetched FIRMS payload in MinIO for replay/debugging."""

    path = context["ti"].xcom_pull(task_ids="fetch_firms")
    bucket = _env("MINIO_BUCKET", "wildfire-data")
    prefix = _env("FIRMS_NRT_ARCHIVE_PREFIX", "firms_nrt_archive").strip("/")
    logical_date = _logical_date(context)
    object_key = (
        f"{prefix}/{logical_date:%Y/%m/%d}/"
        f"{Path(path).stem}.json"
    )

    client = _s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)
    client.upload_file(path, bucket, object_key)

    uri = f"s3://{bucket}/{object_key}"
    print(f"Archived FIRMS payload to {uri}")
    return uri
