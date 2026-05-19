"""Kafka consumer that fans wildfire alerts out to Telegram and email subscribers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

from kafka import KafkaConsumer

from . import db
from .dispatcher import (
    fetch_map_image,
    normalize_alert,
    render_email,
    render_telegram,
    render_telegram_caption,
    send_email,
    send_telegram,
    send_telegram_photo,
)


LOGGER = logging.getLogger("wildfire.alerts.worker")
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "alerts"
DEFAULT_GROUP_ID = "wildfire-notification-worker"
SHUTDOWN = False


def _load_env_file(path: Path = Path(".env")) -> None:
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


def _install_signal_handlers() -> None:
    def _handle(signum, frame):  # noqa: ARG001
        global SHUTDOWN
        LOGGER.info("Received signal %s, shutting down", signum)
        SHUTDOWN = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def parse_args() -> argparse.Namespace:
    _load_env_file()
    parser = argparse.ArgumentParser(description="Wildfire alert notification worker")
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS))
    parser.add_argument("--topic", default=os.getenv("KAFKA_ALERT_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--group-id", default=os.getenv("KAFKA_NOTIFICATION_GROUP_ID", DEFAULT_GROUP_ID))
    parser.add_argument("--starting-offsets", default=os.getenv("KAFKA_NOTIFICATION_OFFSETS", "latest"))
    parser.add_argument(
        "--grid-size",
        type=float,
        default=float(os.getenv("ALERT_GRID_SIZE", "0.5")),
        help="Grid size (degrees) used to derive lat/lon from grid_lat_index/grid_lon_index.",
    )
    parser.add_argument("--once", action="store_true", help="Process whatever is currently buffered and exit.")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def deliver(alert: dict[str, object]) -> dict[str, int]:
    counts = {"telegram": 0, "email": 0, "skipped": 0, "failed": 0}
    subscribers = db.matching_subscribers(
        severity_value=alert["severity_label"],
        latitude=alert["latitude"],
        longitude=alert["longitude"],
    )

    if not subscribers:
        LOGGER.info("No subscribers matched alert %s", alert["alert_id"])
        return counts

    map_image_url = alert.get("map_image_url")
    map_image_bytes: bytes | None = None
    if map_image_url:
        map_image_bytes = fetch_map_image(map_image_url)
        if map_image_bytes is None:
            LOGGER.warning("Mapbox image download failed for %s, falling back to text-only", alert["alert_id"])
        else:
            alert["has_inline_map"] = True

    telegram_text = render_telegram(alert)
    telegram_caption = render_telegram_caption(alert)
    subject, email_text, email_html = render_email(alert)
    email_inline = {"wildfire-map": map_image_bytes} if map_image_bytes else None

    for subscriber in subscribers:
        sub_id = int(subscriber["id"])
        if db.already_delivered(sub_id, alert["alert_id"]):
            counts["skipped"] += 1
            continue
        try:
            if subscriber["channel"] == "telegram":
                if map_image_bytes is not None:
                    send_telegram_photo(subscriber["address"], map_image_bytes, telegram_caption)
                else:
                    send_telegram(subscriber["address"], telegram_text)
                counts["telegram"] += 1
            elif subscriber["channel"] == "email":
                send_email(
                    subscriber["address"],
                    subject,
                    email_text,
                    email_html,
                    inline_images=email_inline,
                )
                counts["email"] += 1
            else:
                LOGGER.warning("Unknown channel %s for subscriber %s", subscriber["channel"], sub_id)
                counts["skipped"] += 1
                continue
            db.record_delivery(sub_id, alert["alert_id"], subscriber["channel"], "sent")
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            LOGGER.exception(
                "Failed to deliver alert %s to %s/%s",
                alert["alert_id"],
                subscriber["channel"],
                subscriber["address"],
            )
            db.record_delivery(sub_id, alert["alert_id"], subscriber["channel"], "failed", str(exc))
    return counts


def consume(args: argparse.Namespace) -> None:
    LOGGER.info(
        "Starting notification worker: topic=%s bootstrap=%s group=%s",
        args.topic,
        args.bootstrap_servers,
        args.group_id,
    )
    db.init_schema()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers.split(","),
        group_id=args.group_id,
        auto_offset_reset=args.starting_offsets,
        enable_auto_commit=True,
        value_deserializer=lambda value: value.decode("utf-8") if value else "",
        consumer_timeout_ms=1500 if args.once else float("inf"),
    )

    processed = 0
    try:
        while not SHUTDOWN:
            poll = consumer.poll(timeout_ms=1000, max_records=50)
            if not poll:
                if args.once and processed > 0:
                    break
                continue
            for _topic_partition, records in poll.items():
                for record in records:
                    if not record.value:
                        continue
                    try:
                        payload = json.loads(record.value)
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping non-JSON alert: %s", record.value[:200])
                        continue
                    alert = normalize_alert(payload, grid_size=args.grid_size)
                    if not alert["alert_id"]:
                        LOGGER.warning("Skipping alert without alert_id: %s", payload)
                        continue
                    counts = deliver(alert)
                    processed += 1
                    LOGGER.info("Dispatched alert %s -> %s", alert["alert_id"], counts)
    finally:
        consumer.close()
        LOGGER.info("Notification worker stopped after processing %d alerts", processed)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _install_signal_handlers()
    consume(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
