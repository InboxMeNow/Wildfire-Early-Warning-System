#!/usr/bin/env python3
"""Read Kafka messages from a wildfire topic for streaming smoke tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kafka import KafkaConsumer
from kafka.errors import KafkaError


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

    parser = argparse.ArgumentParser(description="Consume Kafka messages for verification.")
    parser.add_argument("--topic", default=os.getenv("KAFKA_FIRE_TOPIC", DEFAULT_TOPIC))
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS),
    )
    parser.add_argument("--group-id", default="wildfire-test-consumer")
    parser.add_argument("--max-messages", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    if args.max_messages <= 0:
        parser.error("--max-messages must be positive")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    return args


def main() -> int:
    args = parse_args()

    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_servers,
            group_id=args.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=args.timeout_ms,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            key_deserializer=lambda value: value.decode("utf-8") if value else None,
        )
    except KafkaError as exc:
        print(f"Could not connect to Kafka at {args.bootstrap_servers}: {exc}")
        return 1

    count = 0
    try:
        for message in consumer:
            count += 1
            event = message.value
            summary = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "key": message.key,
            }
            if "alert_id" in event:
                summary.update(
                    {
                        "alert_id": event.get("alert_id"),
                        "grid_id": event.get("grid_id"),
                        "fire_count": event.get("fire_count"),
                        "severity": event.get("severity"),
                    }
                )
            else:
                summary.update(
                    {
                        "acq_date": event.get("acq_date"),
                        "acq_time": event.get("acq_time"),
                        "latitude": event.get("latitude"),
                        "longitude": event.get("longitude"),
                        "confidence": event.get("confidence"),
                    }
                )
            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                )
            )
            if count >= args.max_messages:
                break
    finally:
        consumer.close()

    if count == 0:
        print(f"No messages received from topic '{args.topic}' within {args.timeout_ms} ms.")
        return 1

    print(f"Consumed {count} message(s) from topic '{args.topic}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
