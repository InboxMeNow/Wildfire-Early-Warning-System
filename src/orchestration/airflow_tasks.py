"""Small Python callables used by the Airflow DAGs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config

from streaming.firms_producer import (
    DEFAULT_BBOX,
    DEFAULT_SOURCE,
    DEFAULT_TOPIC,
    create_producer,
    event_key,
    fetch_firms_recent,
)


PROJECT_DIR = Path(os.getenv("AIRFLOW_PROJECT_DIR", ".")).resolve()
DEFAULT_STAGE_DIR = PROJECT_DIR / "data" / "airflow" / "firms_recent"
DEFAULT_TRAINING_STAGE_DIR = PROJECT_DIR / "data" / "airflow" / "firms_training"

FIRMS_TRAINING_SCHEMA = pa.schema(
    [
        ("firms_source", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("brightness", pa.float64()),
        ("bright_ti4", pa.float64()),
        ("bright_ti5", pa.float64()),
        ("bright_t31", pa.float64()),
        ("scan", pa.float64()),
        ("track", pa.float64()),
        ("acq_date", pa.date32()),
        ("acq_time", pa.int32()),
        ("satellite", pa.string()),
        ("instrument", pa.string()),
        ("confidence", pa.string()),
        ("version", pa.string()),
        ("frp", pa.float64()),
        ("daynight", pa.string()),
        ("type", pa.string()),
        ("query_start", pa.date32()),
        ("query_end", pa.date32()),
    ]
)

FIRMS_COVERAGE_SCHEMA = pa.schema(
    [
        ("firms_source", pa.string()),
        ("query_start", pa.date32()),
        ("query_end", pa.date32()),
        ("event_count", pa.int32()),
        ("fetched_at_utc", pa.timestamp("us", tz="UTC")),
        ("run_id", pa.string()),
    ]
)


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
    hours = int(_env("FIRMS_RECENT_HOURS", "6"))
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


def _ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def _upload_file_to_minio(path: Path, object_key: str) -> str:
    bucket = _env("MINIO_BUCKET", "wildfire-data")
    client = _s3_client()
    _ensure_bucket(client, bucket)
    client.upload_file(str(path), bucket, object_key)
    return f"s3://{bucket}/{object_key}"


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
    _ensure_bucket(client, bucket)
    client.upload_file(path, bucket, object_key)

    uri = f"s3://{bucket}/{object_key}"
    print(f"Archived FIRMS payload to {uri}")
    return uri

def _empty_firms_training_frame() -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in FIRMS_TRAINING_SCHEMA})

def normalize_recent_fires_for_training(events: list[dict[str, Any]], default_source: str) -> pd.DataFrame:
    """Convert FIRMS NRT JSON events into the same Parquet schema used by history."""

    if not events:
        frame = _empty_firms_training_frame()
    else:
        frame = pd.DataFrame(events)

    normalized = frame.copy()
    if "firms_source" not in normalized:
        normalized["firms_source"] = default_source
    normalized["firms_source"] = normalized["firms_source"].fillna(default_source)

    for column in [
        "latitude",
        "longitude",
        "brightness",
        "bright_ti4",
        "bright_ti5",
        "bright_t31",
        "scan",
        "track",
        "frp",
    ]:
        if column not in normalized:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    if "brightness" in normalized and "bright_ti4" in normalized:
        normalized["brightness"] = normalized["brightness"].fillna(normalized["bright_ti4"])

    if "acq_date" not in normalized:
        normalized["acq_date"] = pd.NaT
    normalized["acq_date"] = pd.to_datetime(normalized["acq_date"], errors="coerce").dt.date

    if "acq_time" not in normalized:
        normalized["acq_time"] = pd.NA
    normalized["acq_time"] = pd.to_numeric(normalized["acq_time"], errors="coerce").astype("Int32")

    for column in ["satellite", "instrument", "confidence", "version", "daynight", "type"]:
        if column not in normalized:
            normalized[column] = pd.NA
        normalized[column] = normalized[column].astype("string")

    acquired_dates = pd.Series(normalized["acq_date"]).dropna()
    query_start = acquired_dates.min() if not acquired_dates.empty else datetime.now(timezone.utc).date()
    query_end = acquired_dates.max() if not acquired_dates.empty else query_start
    normalized["query_start"] = query_start
    normalized["query_end"] = query_end

    return normalized[[field.name for field in FIRMS_TRAINING_SCHEMA]]


def _event_fetched_at(events: list[dict[str, Any]]) -> datetime:
    fetched_values = [event.get("fetched_at_utc") for event in events if event.get("fetched_at_utc")]
    if fetched_values:
        parsed = pd.to_datetime(fetched_values, utc=True, errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().to_pydatetime()
    return datetime.now(timezone.utc)


def _write_firms_coverage_marker(events: list[dict[str, Any]], context: dict[str, Any]) -> str:
    source = _env("FIRMS_SOURCE", DEFAULT_SOURCE)
    hours = int(_env("FIRMS_RECENT_HOURS", "6"))
    fetched_at = _event_fetched_at(events)
    window_start = (fetched_at - timedelta(hours=hours)).date()
    window_end = fetched_at.date()

    event_dates = pd.to_datetime(
        [event.get("acq_date") for event in events if event.get("acq_date")],
        utc=False,
        errors="coerce",
    ).dropna()
    if not event_dates.empty:
        window_start = min(window_start, event_dates.min().date())
        window_end = max(window_end, event_dates.max().date())

    logical_date = _logical_date(context)
    run_id = _safe_token(str(context.get("run_id", logical_date.isoformat())))
    stage_dir = Path(_env("AIRFLOW_FIRMS_TRAINING_STAGE_DIR", str(DEFAULT_TRAINING_STAGE_DIR)))
    stage_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = stage_dir / f"firms_coverage_{logical_date:%Y%m%dT%H%M%S}_{run_id}.parquet"
    coverage_frame = pd.DataFrame(
        [
            {
                "firms_source": source,
                "query_start": window_start,
                "query_end": window_end,
                "event_count": len(events),
                "fetched_at_utc": fetched_at,
                "run_id": run_id,
            }
        ]
    )
    coverage_table = pa.Table.from_pandas(coverage_frame, schema=FIRMS_COVERAGE_SCHEMA, preserve_index=False)
    pq.write_table(coverage_table, coverage_path, compression="snappy")

    coverage_prefix = _env("FIRMS_COVERAGE_PREFIX", "firms_coverage").strip("/")
    return _upload_file_to_minio(coverage_path, f"{coverage_prefix}/{coverage_path.name}")


def write_recent_fires_training_parquet(**context: Any) -> str:
    """Write recent FIRMS detections to the raw training prefix consumed by Spark ETL."""

    path = context["ti"].xcom_pull(task_ids="fetch_firms")
    events = _load_events(path)
    coverage_uri = _write_firms_coverage_marker(events, context)
    if not events:
        print(f"No recent FIRMS events to add to the training raw prefix. Wrote coverage marker to {coverage_uri}")
        return f"skipped:no-events coverage={coverage_uri}"

    default_source = _env("FIRMS_SOURCE", DEFAULT_SOURCE)
    frame = normalize_recent_fires_for_training(events, default_source=default_source)

    stage_dir = Path(_env("AIRFLOW_FIRMS_TRAINING_STAGE_DIR", str(DEFAULT_TRAINING_STAGE_DIR)))
    stage_dir.mkdir(parents=True, exist_ok=True)

    logical_date = _logical_date(context)
    run_id = _safe_token(str(context.get("run_id", logical_date.isoformat())))
    parquet_path = stage_dir / f"firms_recent_training_{logical_date:%Y%m%dT%H%M%S}_{run_id}.parquet"
    table = pa.Table.from_pandas(frame, schema=FIRMS_TRAINING_SCHEMA, preserve_index=False)
    pq.write_table(table, parquet_path, compression="snappy")

    prefix = _env("FIRMS_RAW_PREFIX", "firms").strip("/")
    object_key = f"{prefix}/{parquet_path.name}"
    uri = _upload_file_to_minio(parquet_path, object_key)
    print(f"Wrote {len(frame):,} recent FIRMS row(s) into training raw data at {uri}")
    print(f"Wrote FIRMS observation coverage marker to {coverage_uri}")
    return uri
