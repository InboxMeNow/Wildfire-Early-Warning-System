"""Publish daily forecast risk alerts from the inference GeoJSON into Kafka.

Reads ``reports/fire_risk_forecast_latest.geojson`` (produced by ``09_inference.py``),
filters cells whose ``risk_label`` meets a minimum severity, and emits one
``alert_type=FORECAST_RISK`` event per ``(grid_id, date)`` into the configured
Kafka topic. The notification worker picks the events up and dispatches them
to Telegram and email subscribers using the same matching rules as hot-zone
alerts.

Usage:
    python -m src.alerts.forecast_publisher --bootstrap-servers kafka:29092

The script is idempotent at the consumer level: ``notification_log`` already
deduplicates ``(subscriber_id, alert_id)`` pairs, so re-running this publisher
will not produce duplicate notifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kafka import KafkaProducer


LOGGER = logging.getLogger("wildfire.alerts.forecast_publisher")
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "alerts"
DEFAULT_GEOJSON_PATH = Path("reports/fire_risk_forecast_latest.geojson")
SEVERITY_RANKS = {"low": 0, "medium": 1, "high": 2}


try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Pythons
    ZoneInfo = None  # type: ignore[assignment]


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


def parse_args() -> argparse.Namespace:
    _load_env_file()
    parser = argparse.ArgumentParser(description="Publish forecast risk alerts to Kafka.")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS),
    )
    parser.add_argument("--topic", default=os.getenv("KAFKA_ALERT_TOPIC", DEFAULT_TOPIC))
    parser.add_argument(
        "--geojson",
        type=Path,
        default=Path(os.getenv("FORECAST_GEOJSON_PATH", str(DEFAULT_GEOJSON_PATH))),
    )
    parser.add_argument(
        "--min-severity",
        choices=["low", "medium", "high"],
        default=os.getenv("FORECAST_MIN_SEVERITY", "high"),
        help="Minimum severity to publish (default: high)",
    )
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=int(os.getenv("FORECAST_MAX_ALERTS", "50")),
        help="Cap on the number of alerts emitted per run, top-K by risk_score.",
    )
    parser.add_argument(
        "--include-dates",
        default=os.getenv("FORECAST_DATES"),
        help="Optional comma-separated allowlist of forecast dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--target-date",
        default=os.getenv("FORECAST_TARGET_DATE"),
        help=(
            "Convenience: publish only alerts for this single forecast date. "
            "Use 'today' to resolve at runtime in --target-timezone."
        ),
    )
    parser.add_argument(
        "--target-timezone",
        default=os.getenv("FORECAST_TARGET_TIMEZONE", "Asia/Ho_Chi_Minh"),
        help="Timezone used to resolve --target-date=today (default: Asia/Ho_Chi_Minh).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the alerts that would be emitted without writing to Kafka.",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def feature_centroid(feature: dict[str, Any]) -> tuple[float | None, float | None]:
    properties = feature.get("properties") or {}
    if "lat" in properties and "lon" in properties:
        try:
            return float(properties["lat"]), float(properties["lon"])
        except (TypeError, ValueError):
            pass
    geometry = feature.get("geometry") or {}
    if geometry.get("type") == "Polygon":
        ring = geometry.get("coordinates", [[]])[0] or []
        coords = [(lon, lat) for lon, lat in ring if lon is not None and lat is not None]
        if coords:
            lons, lats = zip(*coords)
            return sum(lats) / len(lats), sum(lons) / len(lons)
    return None, None


def build_alert(feature: dict[str, Any], *, model_version: str | None) -> dict[str, Any] | None:
    properties = feature.get("properties") or {}
    grid_id = properties.get("grid_id")
    forecast_date = properties.get("date")
    risk_label = properties.get("risk_label")
    if not (grid_id and forecast_date and risk_label):
        return None

    latitude, longitude = feature_centroid(feature)
    risk_score = properties.get("risk_score")
    risk_level = properties.get("risk_level")
    alert_id = f"FORECAST|{grid_id}|{forecast_date}"
    payload: dict[str, Any] = {
        "alert_id": alert_id,
        "alert_type": "FORECAST_RISK",
        "severity": str(risk_label).lower(),
        "grid_id": grid_id,
        "grid_lat_index": properties.get("grid_lat_index"),
        "grid_lon_index": properties.get("grid_lon_index"),
        "latitude": latitude,
        "longitude": longitude,
        "forecast_date": forecast_date,
        "risk_label": risk_label,
        "risk_level": risk_level,
        "risk_score": float(risk_score) if isinstance(risk_score, (int, float)) else None,
        "temperature_2m_max": properties.get("temperature_2m_max"),
        "relative_humidity_2m_min": properties.get("relative_humidity_2m_min"),
        "wind_speed_10m_max": properties.get("wind_speed_10m_max"),
        "precipitation_sum": properties.get("precipitation_sum"),
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return payload


def filter_features(
    features: Iterable[dict[str, Any]],
    *,
    min_severity_rank: int,
    include_dates: set[str] | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        risk_label = str(properties.get("risk_label") or "").lower()
        if SEVERITY_RANKS.get(risk_label, -1) < min_severity_rank:
            continue
        if include_dates is not None and properties.get("date") not in include_dates:
            continue
        selected.append(feature)
    selected.sort(
        key=lambda feature: (
            SEVERITY_RANKS.get(str((feature.get("properties") or {}).get("risk_label") or "").lower(), -1),
            float((feature.get("properties") or {}).get("risk_score") or 0.0),
        ),
        reverse=True,
    )
    return selected


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.geojson.exists():
        LOGGER.error("Forecast GeoJSON not found: %s", args.geojson)
        return 2

    geojson = json.loads(args.geojson.read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    if not isinstance(features, list):
        LOGGER.error("GeoJSON has no FeatureCollection features array")
        return 2

    metadata_path = args.geojson.with_suffix(".json")
    model_version: str | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_version = metadata.get("model_version") or metadata.get("model_input")
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse forecast metadata at %s", metadata_path)

    include_dates = None
    if args.include_dates:
        include_dates = {chunk.strip() for chunk in args.include_dates.split(",") if chunk.strip()}

    if args.target_date:
        target_date = args.target_date.strip().lower()
        if target_date == "today":
            if ZoneInfo is not None:
                try:
                    target_date = datetime.now(ZoneInfo(args.target_timezone)).date().isoformat()
                except Exception:  # noqa: BLE001
                    target_date = datetime.now(timezone.utc).date().isoformat()
            else:
                target_date = datetime.now(timezone.utc).date().isoformat()
        else:
            # Validate YYYY-MM-DD
            datetime.strptime(target_date, "%Y-%m-%d")
        include_dates = {target_date}
        LOGGER.info("Restricting publishing to forecast date %s", target_date)

    min_rank = SEVERITY_RANKS[args.min_severity]
    selected = filter_features(features, min_severity_rank=min_rank, include_dates=include_dates)
    if args.max_alerts > 0:
        selected = selected[: args.max_alerts]
    LOGGER.info(
        "Selected %d/%d features at severity >= %s%s",
        len(selected),
        len(features),
        args.min_severity,
        f" within dates {sorted(include_dates)}" if include_dates else "",
    )

    if not selected:
        LOGGER.info("Nothing to publish; exiting")
        return 0

    if args.dry_run:
        for feature in selected:
            payload = build_alert(feature, model_version=model_version)
            if payload:
                print(json.dumps(payload, ensure_ascii=False))
        return 0

    producer = KafkaProducer(
        bootstrap_servers=[server.strip() for server in args.bootstrap_servers.split(",") if server.strip()],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8") if key else None,
    )

    sent = 0
    try:
        for feature in selected:
            payload = build_alert(feature, model_version=model_version)
            if payload is None:
                continue
            future = producer.send(args.topic, key=payload["alert_id"], value=payload)
            metadata = future.get(timeout=15)
            sent += 1
            LOGGER.info(
                "Published forecast alert %s -> %s/%s offset=%s",
                payload["alert_id"],
                metadata.topic,
                metadata.partition,
                metadata.offset,
            )
        producer.flush()
    finally:
        producer.close()

    LOGGER.info("Published %d forecast alerts to %s", sent, args.topic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
