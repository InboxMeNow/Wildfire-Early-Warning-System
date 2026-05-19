"""Database helpers for the wildfire alert notification subsystem."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

import psycopg


DEFAULT_DATABASE_URL = "postgresql://wildfire:wildfire@postgres:5432/wildfire"

SEVERITY_TO_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
RANK_TO_SEVERITY: dict[int, str] = {0: "low", 1: "medium", 2: "high"}


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


@contextmanager
def connect(retries: int = 12, delay_seconds: float = 1.0) -> Iterator[psycopg.Connection]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            conn = psycopg.connect(database_url())
            try:
                yield conn
            finally:
                conn.close()
            return
        except psycopg.OperationalError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not connect to PostgreSQL: {last_error}") from last_error


def init_schema() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_subscribers (
                id BIGSERIAL PRIMARY KEY,
                channel TEXT NOT NULL CHECK (channel IN ('telegram', 'email')),
                address TEXT NOT NULL,
                min_severity_rank INTEGER NOT NULL DEFAULT 2,
                bbox_min_lon DOUBLE PRECISION,
                bbox_min_lat DOUBLE PRECISION,
                bbox_max_lon DOUBLE PRECISION,
                bbox_max_lat DOUBLE PRECISION,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (channel, address)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_log (
                id BIGSERIAL PRIMARY KEY,
                subscriber_id BIGINT NOT NULL REFERENCES notification_subscribers(id) ON DELETE CASCADE,
                alert_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (subscriber_id, alert_id)
            )
            """
        )
        conn.commit()


def severity_rank(severity: str | int | None) -> int:
    if severity is None:
        return 0
    if isinstance(severity, int):
        if severity in RANK_TO_SEVERITY:
            return severity
        raise ValueError(f"unsupported severity rank: {severity}")
    normalized = str(severity).strip().lower()
    if normalized in SEVERITY_TO_RANK:
        return SEVERITY_TO_RANK[normalized]
    if normalized in {"0", "1", "2"}:
        return int(normalized)
    raise ValueError(f"unsupported severity: {severity!r}")


def severity_label(rank: int) -> str:
    if rank not in RANK_TO_SEVERITY:
        raise ValueError(f"unsupported severity rank: {rank}")
    return RANK_TO_SEVERITY[rank]


def upsert_subscriber(
    *,
    channel: str,
    address: str,
    min_severity: str | int = "high",
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    rank = severity_rank(min_severity)
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon = min_lat = max_lon = max_lat = None

    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO notification_subscribers (
                channel, address, min_severity_rank,
                bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (channel, address) DO UPDATE SET
                min_severity_rank = EXCLUDED.min_severity_rank,
                bbox_min_lon = EXCLUDED.bbox_min_lon,
                bbox_min_lat = EXCLUDED.bbox_min_lat,
                bbox_max_lon = EXCLUDED.bbox_max_lon,
                bbox_max_lat = EXCLUDED.bbox_max_lat,
                status = 'active',
                updated_at = NOW()
            RETURNING id, channel, address, min_severity_rank,
                      bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, status
            """,
            (channel, address, rank, min_lon, min_lat, max_lon, max_lat),
        ).fetchone()
        conn.commit()
    return _row_to_subscriber(row)


def update_severity(channel: str, address: str, severity: str | int) -> bool:
    rank = severity_rank(severity)
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE notification_subscribers
            SET min_severity_rank = %s, updated_at = NOW()
            WHERE channel = %s AND address = %s
            """,
            (rank, channel, address),
        )
        conn.commit()
        return cur.rowcount > 0


def update_region(channel: str, address: str, bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE notification_subscribers
            SET bbox_min_lon = %s, bbox_min_lat = %s, bbox_max_lon = %s, bbox_max_lat = %s, updated_at = NOW()
            WHERE channel = %s AND address = %s
            """,
            (min_lon, min_lat, max_lon, max_lat, channel, address),
        )
        conn.commit()
        return cur.rowcount > 0


def deactivate_subscriber(channel: str, address: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE notification_subscribers
            SET status = 'inactive', updated_at = NOW()
            WHERE channel = %s AND address = %s
            """,
            (channel, address),
        )
        conn.commit()
        return cur.rowcount > 0


def get_subscriber(channel: str, address: str) -> dict[str, object] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, channel, address, min_severity_rank,
                   bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, status
            FROM notification_subscribers
            WHERE channel = %s AND address = %s
            """,
            (channel, address),
        ).fetchone()
    return _row_to_subscriber(row) if row else None


def matching_subscribers(
    severity_value: str | int,
    latitude: float | None,
    longitude: float | None,
) -> list[dict[str, object]]:
    rank = severity_rank(severity_value)
    sql = """
        SELECT id, channel, address, min_severity_rank,
               bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, status
        FROM notification_subscribers
        WHERE status = 'active'
          AND min_severity_rank <= %s
    """
    params: list[object] = [rank]
    if latitude is not None and longitude is not None:
        sql += """
          AND (
            bbox_min_lon IS NULL
            OR (
                %s BETWEEN bbox_min_lon AND bbox_max_lon
                AND %s BETWEEN bbox_min_lat AND bbox_max_lat
            )
          )
        """
        params.extend([longitude, latitude])

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_subscriber(row) for row in rows]


def record_delivery(subscriber_id: int, alert_id: str, channel: str, status: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO notification_log (subscriber_id, alert_id, channel, status, error)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (subscriber_id, alert_id) DO UPDATE SET
                status = EXCLUDED.status,
                error = EXCLUDED.error,
                sent_at = NOW()
            """,
            (subscriber_id, alert_id, channel, status, error),
        )
        conn.commit()


def already_delivered(subscriber_id: int, alert_id: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM notification_log
            WHERE subscriber_id = %s AND alert_id = %s AND status = 'sent'
            """,
            (subscriber_id, alert_id),
        ).fetchone()
    return row is not None


def recent_alerts_for_subscriber(subscriber: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
    rank = int(subscriber["min_severity_rank"])
    bbox_min_lon = subscriber.get("bbox_min_lon")
    sql = """
        SELECT grid_id, latitude, longitude, prediction_date, risk_label, risk_score, model_version, payload
        FROM prediction_cache
        WHERE risk_level >= %s
    """
    params: list[object] = [rank]
    if bbox_min_lon is not None:
        sql += """
          AND latitude BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
        """
        params.extend(
            [
                subscriber["bbox_min_lat"],
                subscriber["bbox_max_lat"],
                subscriber["bbox_min_lon"],
                subscriber["bbox_max_lon"],
            ]
        )
    sql += " ORDER BY risk_level DESC, risk_score DESC, prediction_date DESC LIMIT %s"
    params.append(limit)

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        {
            "grid_id": row[0],
            "latitude": float(row[1]),
            "longitude": float(row[2]),
            "date": row[3].isoformat(),
            "risk_label": row[4],
            "risk_score": float(row[5]),
            "model_version": row[6],
            "payload": row[7],
        }
        for row in rows
    ]


def _row_to_subscriber(row: object) -> dict[str, object]:
    return {
        "id": int(row[0]),
        "channel": row[1],
        "address": row[2],
        "min_severity_rank": int(row[3]),
        "min_severity": severity_label(int(row[3])),
        "bbox_min_lon": float(row[4]) if row[4] is not None else None,
        "bbox_min_lat": float(row[5]) if row[5] is not None else None,
        "bbox_max_lon": float(row[6]) if row[6] is not None else None,
        "bbox_max_lat": float(row[7]) if row[7] is not None else None,
        "status": row[8],
    }
