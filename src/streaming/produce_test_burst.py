#!/usr/bin/env python3
"""Publish a synthetic FIRMS-like burst to validate hot-zone alerts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaProducer


DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "fire-events"


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


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Publish a synthetic fire-event burst.")
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS))
    parser.add_argument("--topic", default=os.getenv("KAFKA_FIRE_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--latitude", type=float, default=10.75)
    parser.add_argument("--longitude", type=float, default=106.75)
    parser.add_argument("--event-time", default=None, help="ISO timestamp. Defaults to current UTC time.")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    return args


def main() -> int:
    args = parse_args()
    acquired_at = (
        datetime.fromisoformat(args.event_time.replace("Z", "+00:00"))
        if args.event_time
        else datetime.now(timezone.utc)
    )
    acquired_at = acquired_at.astimezone(timezone.utc)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        key_serializer=lambda value: value.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
    )

    for index in range(args.count):
        event = {
            "firms_source": "SYNTHETIC_BURST",
            "latitude": args.latitude + (index % 3) * 0.001,
            "longitude": args.longitude + (index % 3) * 0.001,
            "acq_date": acquired_at.strftime("%Y-%m-%d"),
            "acq_time": int(acquired_at.strftime("%H%M")),
            "acquired_at_utc": acquired_at.isoformat(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "satellite": "TEST",
            "instrument": "TEST",
            "confidence": "h",
            "frp": 25.0 + index,
            "daynight": "D",
        }
        key = f"synthetic-burst|{acquired_at.isoformat()}|{index}"
        producer.send(args.topic, key=key, value=event)

    producer.flush()
    producer.close()
    print(
        f"Published {args.count} synthetic fires to '{args.topic}' near "
        f"({args.latitude}, {args.longitude}) at {acquired_at.isoformat()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
