#!/usr/bin/env python3
"""FastAPI REST service for wildfire prediction consumers."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator


APP_DIR = Path(__file__).resolve().parent
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", APP_DIR / "reports"))
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data"))
PREDICTION_GEOJSON_PATH = Path(os.getenv("PREDICTION_GEOJSON_PATH", REPORTS_DIR / "fire_risk_forecast_latest.geojson"))
PREDICTION_METADATA_PATH = Path(os.getenv("PREDICTION_METADATA_PATH", REPORTS_DIR / "fire_risk_forecast_latest.json"))
HISTORICAL_FIRES_PATH = Path(os.getenv("HISTORICAL_FIRES_PATH", DATA_DIR / "raw" / "firms_history_vietnam_2020_2024.parquet"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://wildfire:wildfire@postgres:5432/wildfire",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "wildfire-gbt")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")


app = FastAPI(
    title="Wildfire Warning API",
    version="1.0.0",
    description="REST API for wildfire risk predictions, active alerts, subscriptions, and historical FIRMS fire detections.",
)


class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    date: str
    requested_date: str
    risk_level: str
    risk_score: float
    model_version: str
    grid_id: str


class Alert(BaseModel):
    grid_id: str
    latitude: float
    longitude: float
    risk_level: str
    risk_score: float
    date: str
    model_version: str


class ActiveAlertsResponse(BaseModel):
    count: int
    alerts: list[Alert]


class FireDetection(BaseModel):
    latitude: float
    longitude: float
    acq_date: str
    acq_time: int | None = None
    satellite: str | None = None
    instrument: str | None = None
    confidence: str | None = None
    frp: float | None = None
    source: str | None = None


class HistoricalFiresResponse(BaseModel):
    count: int
    limit: int
    fires: list[FireDetection]


class SubscriptionRequest(BaseModel):
    email: EmailStr
    min_risk_level: str | int = Field(default="high", description="low, medium, high, or numeric 0-2")
    bbox: str | None = Field(default=None, description="min_lon,min_lat,max_lon,max_lat")
    webhook_url: HttpUrl | None = None

    @field_validator("min_risk_level", mode="before")
    @classmethod
    def validate_risk_level(cls, value: str | int) -> str | int:
        normalize_risk_level(value)
        return value

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str | None) -> str | None:
        if value is not None:
            parse_bbox(value)
        return value


class SubscriptionResponse(BaseModel):
    id: int
    email: EmailStr
    min_risk_level: str
    bbox: str | None = None
    webhook_url: str | None = None
    status: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    cached_predictions: int


def connect_db(retries: int = 12, delay_seconds: float = 1.0):
    last_error = None
    for _ in range(retries):
        try:
            return psycopg.connect(DATABASE_URL)
        except psycopg.OperationalError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not connect to PostgreSQL: {last_error}") from last_error


def init_db() -> None:
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_cache (
                grid_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                grid_lat DOUBLE PRECISION NOT NULL,
                grid_lon DOUBLE PRECISION NOT NULL,
                latitude DOUBLE PRECISION NOT NULL,
                longitude DOUBLE PRECISION NOT NULL,
                risk_level INTEGER NOT NULL,
                risk_label TEXT NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (grid_id, prediction_date)
            )
            """
        )
        conn.execute("ALTER TABLE prediction_cache DROP CONSTRAINT IF EXISTS prediction_cache_pkey")
        conn.execute(
            """
            ALTER TABLE prediction_cache
            ADD CONSTRAINT prediction_cache_pkey PRIMARY KEY (grid_id, prediction_date)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_subscriptions (
                id BIGSERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                min_risk_level INTEGER NOT NULL,
                bbox TEXT,
                webhook_url TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.commit()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_version() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, [MLFLOW_MODEL_STAGE])
        if versions:
            version = versions[0]
            return f"{version.name}/{version.version}"
    except Exception:
        pass

    metadata = load_json(PREDICTION_METADATA_PATH)
    model_input = metadata.get("model_input")
    if isinstance(model_input, str) and model_input:
        return model_input.replace("models:/", "")
    return f"{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"


def seed_prediction_cache() -> int:
    if not PREDICTION_GEOJSON_PATH.exists():
        return 0

    geojson = load_json(PREDICTION_GEOJSON_PATH)
    features = geojson.get("features", [])
    if not isinstance(features, list):
        return 0

    model_version = resolve_model_version()
    rows = []
    for feature in features:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        if not isinstance(props, dict):
            continue
        rows.append(
            (
                str(props["grid_id"]),
                date.fromisoformat(str(props["date"])),
                float(props["grid_lat"]),
                float(props["grid_lon"]),
                float(props["lat"]),
                float(props["lon"]),
                int(props["risk_level"]),
                str(props["risk_label"]),
                float(props["risk_score"]),
                model_version,
                json.dumps(props),
            )
        )

    if not rows:
        return 0

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_cache")
            cur.executemany(
                """
                INSERT INTO prediction_cache (
                    grid_id, prediction_date, grid_lat, grid_lon, latitude, longitude,
                    risk_level, risk_label, risk_score, model_version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                ON CONFLICT (grid_id, prediction_date) DO UPDATE SET
                    grid_lat = EXCLUDED.grid_lat,
                    grid_lon = EXCLUDED.grid_lon,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    risk_level = EXCLUDED.risk_level,
                    risk_label = EXCLUDED.risk_label,
                    risk_score = EXCLUDED.risk_score,
                    model_version = EXCLUDED.model_version,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                rows,
            )
        conn.commit()
    return len(rows)


@app.on_event("startup")
def startup() -> None:
    init_db()
    app.state.cached_predictions = seed_prediction_cache()
    app.state.model_version = resolve_model_version()


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        min_lon, min_lat, max_lon, max_lat = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat") from exc
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox min values must be smaller than max values")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("bbox coordinates are outside valid latitude/longitude bounds")
    return min_lon, min_lat, max_lon, max_lat


def normalize_risk_level(value: str | int) -> int:
    mapping = {"low": 0, "medium": 1, "high": 2}
    if isinstance(value, bool):
        raise ValueError("risk level must be 0, 1, 2, low, medium, or high")
    if isinstance(value, int):
        if value not in (0, 1, 2):
            raise ValueError("risk level must be 0, 1, 2, low, medium, or high")
        return value
    normalized = str(value).strip().lower()
    if normalized in mapping:
        return mapping[normalized]
    if normalized in {"0", "1", "2"}:
        return int(normalized)
    raise ValueError("risk level must be 0, 1, 2, low, medium, or high")


def risk_level_label(value: str | int) -> str:
    labels = {0: "low", 1: "medium", 2: "high"}
    return labels[normalize_risk_level(value)]


def nearest_prediction(lat: float, lon: float, requested_date: date) -> dict[str, Any] | None:
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT grid_id, latitude, longitude, prediction_date, risk_label, risk_score, model_version
            FROM prediction_cache
            ORDER BY
                CASE WHEN prediction_date = %s THEN 0 ELSE 1 END ASC,
                prediction_date DESC,
                ((latitude - %s) * (latitude - %s) + (longitude - %s) * (longitude - %s)) ASC
            LIMIT 1
            """,
            (requested_date, lat, lat, lon, lon),
        ).fetchone()
    if row is None:
        return None
    return {
        "grid_id": row[0],
        "latitude": float(row[1]),
        "longitude": float(row[2]),
        "prediction_date": row[3].isoformat(),
        "risk_level": row[4],
        "risk_score": float(row[5]),
        "model_version": row[6],
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    with connect_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_cache").fetchone()[0]
    return HealthResponse(
        status="ok",
        model_name=MLFLOW_MODEL_NAME,
        model_version=getattr(app.state, "model_version", resolve_model_version()),
        cached_predictions=int(count),
    )


@app.get("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict_risk(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date_value: date = Query(..., alias="date"),
) -> PredictionResponse:
    prediction = nearest_prediction(lat, lon, date_value)
    if prediction is None:
        raise HTTPException(status_code=503, detail="Prediction cache is empty")
    return PredictionResponse(
        latitude=lat,
        longitude=lon,
        date=prediction["prediction_date"],
        requested_date=date_value.isoformat(),
        risk_level=prediction["risk_level"],
        risk_score=prediction["risk_score"],
        model_version=prediction["model_version"],
        grid_id=prediction["grid_id"],
    )


@app.get("/alerts/active", response_model=ActiveAlertsResponse, tags=["alerts"])
def get_active_alerts(
    min_risk_level: str = Query("high", description="low, medium, high, or numeric 0-2"),
    limit: int = Query(25, ge=1, le=200),
) -> ActiveAlertsResponse:
    try:
        risk_floor = normalize_risk_level(min_risk_level)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT grid_id, latitude, longitude, prediction_date, risk_label, risk_score, model_version
            FROM prediction_cache
            WHERE risk_level >= %s
            ORDER BY risk_level DESC, risk_score DESC
            LIMIT %s
            """,
            (risk_floor, limit),
        ).fetchall()

    alerts = [
        Alert(
            grid_id=row[0],
            latitude=float(row[1]),
            longitude=float(row[2]),
            date=row[3].isoformat(),
            risk_level=row[4],
            risk_score=float(row[5]),
            model_version=row[6],
        )
        for row in rows
    ]
    return ActiveAlertsResponse(count=len(alerts), alerts=alerts)


@app.get("/historical/fires", response_model=HistoricalFiresResponse, tags=["historical"])
def get_historical_fires(
    bbox: str = Query(..., description="min_lon,min_lat,max_lon,max_lat"),
    start: date = Query(...),
    end: date = Query(...),
    limit: int = Query(500, ge=1, le=5000),
) -> HistoricalFiresResponse:
    try:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")
    if not HISTORICAL_FIRES_PATH.exists():
        return HistoricalFiresResponse(count=0, limit=limit, fires=[])

    columns = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence", "frp", "firms_source"]
    filters = [
        ("longitude", ">=", min_lon),
        ("longitude", "<=", max_lon),
        ("latitude", ">=", min_lat),
        ("latitude", "<=", max_lat),
        ("acq_date", ">=", start),
        ("acq_date", "<=", end),
    ]
    frame = pd.read_parquet(HISTORICAL_FIRES_PATH, columns=columns, filters=filters)
    if frame.empty:
        return HistoricalFiresResponse(count=0, limit=limit, fires=[])
    frame = frame.sort_values(["acq_date", "acq_time"], ascending=False).head(limit)
    fires = []
    for row in frame.to_dict("records"):
        frp = row.get("frp")
        fires.append(
            FireDetection(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                acq_date=row["acq_date"].isoformat() if hasattr(row["acq_date"], "isoformat") else str(row["acq_date"]),
                acq_time=None if pd.isna(row.get("acq_time")) else int(row["acq_time"]),
                satellite=None if pd.isna(row.get("satellite")) else str(row["satellite"]),
                instrument=None if pd.isna(row.get("instrument")) else str(row["instrument"]),
                confidence=None if pd.isna(row.get("confidence")) else str(row["confidence"]),
                frp=None if frp is None or (isinstance(frp, float) and math.isnan(frp)) else float(frp),
                source=None if pd.isna(row.get("firms_source")) else str(row["firms_source"]),
            )
        )
    return HistoricalFiresResponse(count=len(fires), limit=limit, fires=fires)


@app.post("/subscribe", response_model=SubscriptionResponse, status_code=201, tags=["subscriptions"])
def subscribe(request: SubscriptionRequest) -> SubscriptionResponse:
    min_risk_level = normalize_risk_level(request.min_risk_level)
    with connect_db() as conn:
        row = conn.execute(
            """
            INSERT INTO alert_subscriptions (email, min_risk_level, bbox, webhook_url)
            VALUES (%s, %s, %s, %s)
            RETURNING id, status
            """,
            (
                request.email,
                min_risk_level,
                request.bbox,
                str(request.webhook_url) if request.webhook_url else None,
            ),
        ).fetchone()
        conn.commit()
    return SubscriptionResponse(
        id=int(row[0]),
        email=request.email,
        min_risk_level=risk_level_label(request.min_risk_level),
        bbox=request.bbox,
        webhook_url=str(request.webhook_url) if request.webhook_url else None,
        status=row[1],
    )
