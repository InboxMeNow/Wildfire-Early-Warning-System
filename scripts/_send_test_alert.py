"""Send one synthetic alert into the Kafka 'alerts' topic for end-to-end testing."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from kafka import KafkaProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="alerts")
    parser.add_argument("--lat-index", type=int, default=42)  # 42 * 0.5 = 21.0 -> Hanoi area
    parser.add_argument("--lon-index", type=int, default=211)  # 211 * 0.5 = 105.5
    parser.add_argument("--severity", default="high")
    parser.add_argument("--fire-count", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window_end = now
    window_start = window_end - timedelta(hours=1)

    alert_id = (
        f"{args.lat_index}_{args.lon_index}|"
        f"{window_start.strftime('%Y-%m-%dT%H:%M:%S')}|"
        f"{window_end.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    payload = {
        "alert_id": alert_id,
        "alert_type": "HOT_ZONE",
        "severity": args.severity,
        "grid_id": f"{args.lat_index}_{args.lon_index}",
        "grid_lat_index": args.lat_index,
        "grid_lon_index": args.lon_index,
        "fire_count": args.fire_count,
        "threshold": args.threshold,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": window_end.isoformat().replace("+00:00", "Z"),
        "created_at_utc": now.isoformat().replace("+00:00", "Z"),
        "spark_batch_id": -1,
    }

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    future = producer.send(args.topic, key=alert_id, value=payload)
    metadata = future.get(timeout=10)
    producer.flush()
    producer.close()
    print(json.dumps({"sent": True, "topic": metadata.topic, "partition": metadata.partition, "offset": metadata.offset, "alert_id": alert_id}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
