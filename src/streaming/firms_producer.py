#!/usr/bin/env python3
"""Stream recent NASA FIRMS fire detections into Kafka."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError


FIRMS_AREA_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_SOURCE = "VIIRS_NOAA20_NRT"
DEFAULT_BBOX = "95,5,115,25"
DEFAULT_TOPIC = "fire-events"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_POLL_SECONDS = 180
DEFAULT_HOURS = 1
MAX_NRT_DAYS = 5


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_firms_url(map_key: str, source: str, bbox: str, days: int) -> str:
    return f"{FIRMS_AREA_API}/{map_key}/{source}/{bbox}/{days}"


def parse_acquisition_time(row: dict[str, str]) -> datetime | None:
    date_value = row.get("acq_date")
    time_value = str(row.get("acq_time", "0000")).strip().zfill(4)
    if not date_value:
        return None

    try:
        hour = int(time_value[:2])
        minute = int(time_value[2:4])
        acquired_date = datetime.strptime(date_value, "%Y-%m-%d").date()
        return datetime(
            acquired_date.year,
            acquired_date.month,
            acquired_date.day,
            hour,
            minute,
            tzinfo=timezone.utc,
        )
    except (TypeError, ValueError):
        return None


def coerce_value(value: str) -> Any:
    if value == "":
        return None
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            continue
    return value


def normalize_fire(row: dict[str, str], source: str, fetched_at: datetime) -> dict[str, Any]:
    event = {key: coerce_value(value) for key, value in row.items()}
    acquired_at = parse_acquisition_time(row)
    event["firms_source"] = source
    event["acquired_at_utc"] = acquired_at.isoformat() if acquired_at else None
    event["fetched_at_utc"] = fetched_at.isoformat()
    return event


def fetch_firms_recent(
    map_key: str,
    source: str,
    bbox: str,
    hours: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    days = max(1, min(MAX_NRT_DAYS, (hours + 23) // 24))
    fetched_at = datetime.now(timezone.utc)
    cutoff = fetched_at - timedelta(hours=hours)
    url = build_firms_url(map_key=map_key, source=source, bbox=bbox, days=days)

    response = requests.get(
        url,
        headers={"User-Agent": "wildfire-kafka-firms-producer/1.0"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    text = response.text.strip()
    if not text or text.startswith("Invalid API call") or text.startswith("Error"):
        raise RuntimeError(f"Unexpected FIRMS response: {text[:300]}")

    events: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(text)):
        acquired_at = parse_acquisition_time(row)
        if acquired_at is None or acquired_at < cutoff:
            continue
        events.append(normalize_fire(row, source=source, fetched_at=fetched_at))
    return events


def event_key(event: dict[str, Any]) -> bytes:
    parts = [
        event.get("firms_source", ""),
        event.get("acq_date", ""),
        event.get("acq_time", ""),
        event.get("latitude", ""),
        event.get("longitude", ""),
    ]
    return "|".join(str(part) for part in parts).encode("utf-8")


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        key_serializer=lambda value: value,
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        retries=5,
        linger_ms=100,
    )


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Fetch recent FIRMS detections and publish them to Kafka."
    )
    parser.add_argument(
        "--map-key",
        default=os.getenv("FIRMS_MAP_KEY") or os.getenv("MAP_KEY"),
        help="NASA FIRMS MAP_KEY. Defaults to FIRMS_MAP_KEY or MAP_KEY from environment/.env.",
    )
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS))
    parser.add_argument("--topic", default=os.getenv("KAFKA_FIRE_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--source", default=os.getenv("FIRMS_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--bbox", default=os.getenv("FIRMS_BBOX", DEFAULT_BBOX))
    parser.add_argument("--hours", type=int, default=int(os.getenv("FIRMS_RECENT_HOURS", DEFAULT_HOURS)))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("FIRMS_POLL_SECONDS", DEFAULT_POLL_SECONDS)))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Publish one batch and exit.")
    args = parser.parse_args()

    if not args.map_key:
        parser.error("Missing FIRMS MAP_KEY. Set FIRMS_MAP_KEY/MAP_KEY or pass --map-key.")
    if args.map_key == "PASTE_YOUR_FIRMS_MAP_KEY_HERE":
        parser.error("Replace placeholder MAP_KEY in .env or pass --map-key.")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def publish_batch(producer: KafkaProducer, args: argparse.Namespace) -> int:
    events = fetch_firms_recent(
        map_key=args.map_key,
        source=args.source,
        bbox=args.bbox,
        hours=args.hours,
        timeout_seconds=args.timeout_seconds,
    )
    for event in events:
        producer.send(args.topic, key=event_key(event), value=event)
    producer.flush()
    print(
        f"Published {len(events)} FIRMS events to topic '{args.topic}' "
        f"from last {args.hours} hour(s)."
    )
    return len(events)


def main() -> int:
    args = parse_args()
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        producer = create_producer(args.bootstrap_servers)
    except KafkaError as exc:
        print(f"Could not connect to Kafka at {args.bootstrap_servers}: {exc}", file=sys.stderr)
        return 1

    try:
        while not stop_requested:
            publish_batch(producer, args)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except requests.RequestException as exc:
        print(f"FIRMS API request failed: {exc}", file=sys.stderr)
        return 1
    except KafkaError as exc:
        print(f"Kafka publish failed: {exc}", file=sys.stderr)
        return 1
    finally:
        producer.close(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
