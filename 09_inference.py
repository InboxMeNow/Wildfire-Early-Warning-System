#!/usr/bin/env python3
"""
Run next-day wildfire risk inference from Open-Meteo forecast data.

Input:
    s3a://wildfire-data/features/
    s3a://wildfire-data/models/random_forest_fire_baseline/

Output:
    reports/fire_risk_forecast_latest.geojson
    reports/fire_risk_forecast_latest.json
    s3a://wildfire-data/predictions/fire_risk_forecast/latest.geojson
    s3a://wildfire-data/predictions/fire_risk_forecast/latest.json

Run inside this docker-compose network:
    docker compose exec -T spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /workspace/09_inference.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"
DEFAULT_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_FEATURE_COLUMNS = [
    "grid_lat_index",
    "grid_lon_index",
    "grid_lat",
    "grid_lon",
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
    "precipitation_sum_7days",
    "dry_days_count",
    "station_distance_km",
    "weather_points_count",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]
DAILY_VARIABLES = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
]


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


def s3a_path(bucket: str, prefix: str) -> str:
    return f"s3a://{bucket}/{prefix.strip('/')}/"


def default_target_date(timezone: str) -> str:
    if ZoneInfo is not None:
        try:
            current_day = datetime.now(ZoneInfo(timezone)).date()
            return (current_day + timedelta(days=1)).isoformat()
        except Exception:
            pass

    offset_hours = 7 if timezone == "Asia/Ho_Chi_Minh" else 0
    current_day = (datetime.utcnow() + timedelta(hours=offset_hours)).date()
    return (current_day + timedelta(days=1)).isoformat()


def build_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName(args.spark_app_name)

    if args.spark_master:
        builder = builder.master(args.spark_master)

    if args.hadoop_aws_package:
        builder = builder.config("spark.jars.packages", args.hadoop_aws_package)

    builder = (
        builder.config("spark.hadoop.fs.s3a.endpoint", args.minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", args.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", args.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(args.minio_endpoint.startswith("https://")).lower())
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.session.timeZone", args.spark_timezone)
    )

    return builder.getOrCreate()


def load_grid(features: DataFrame) -> list[dict[str, object]]:
    stats = (
        features.groupBy("grid_id", "grid_lat_index", "grid_lon_index", "grid_lat", "grid_lon")
        .agg(
            F.avg("station_distance_km").alias("station_distance_km"),
            F.avg("weather_points_count").alias("weather_points_count"),
        )
        .orderBy("grid_lat_index", "grid_lon_index")
    )
    return [row.asDict(recursive=True) for row in stats.collect()]


def open_meteo_daily(
    args: argparse.Namespace,
    latitude: float,
    longitude: float,
) -> dict[str, list[object]]:
    query = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": args.forecast_timezone,
        "past_days": str(args.past_days_for_rolling),
        "forecast_days": str(args.forecast_days),
    }
    url = f"{args.open_meteo_url}?{urllib.parse.urlencode(query)}"

    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=args.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "daily" not in payload:
                raise RuntimeError(f"Open-Meteo response missing daily block: {payload}")
            return payload["daily"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_delay_seconds * attempt)

    raise RuntimeError(f"Open-Meteo request failed for lat={latitude}, lon={longitude}: {last_error}") from last_error


def fetch_forecast_rows(args: argparse.Namespace, grid_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target = date.fromisoformat(args.target_date)
    rows: list[dict[str, object]] = []

    for index, grid in enumerate(grid_rows, start=1):
        latitude = float(grid["grid_lat"])
        longitude = float(grid["grid_lon"])
        daily = open_meteo_daily(args, latitude, longitude)
        dates = [date.fromisoformat(value) for value in daily["time"]]

        for day_index, current_date in enumerate(dates):
            if current_date > target:
                continue
            rows.append(
                {
                    "grid_id": str(grid["grid_id"]),
                    "grid_lat_index": int(grid["grid_lat_index"]),
                    "grid_lon_index": int(grid["grid_lon_index"]),
                    "grid_lat": latitude,
                    "grid_lon": longitude,
                    "date": current_date.isoformat(),
                    "temperature_2m_max": daily_value(daily, "temperature_2m_max", day_index),
                    "relative_humidity_2m_min": daily_value(daily, "relative_humidity_2m_min", day_index),
                    "wind_speed_10m_max": daily_value(daily, "wind_speed_10m_max", day_index),
                    "precipitation_sum": daily_value(daily, "precipitation_sum", day_index),
                    "station_distance_km": float(grid["station_distance_km"] or 0.0),
                    "weather_points_count": float(grid["weather_points_count"] or 1.0),
                }
            )

        if args.request_delay_seconds > 0 and index < len(grid_rows):
            time.sleep(args.request_delay_seconds)

        if args.print_progress and (index % 25 == 0 or index == len(grid_rows)):
            print(f"Fetched forecast for {index:,}/{len(grid_rows):,} grid cells")

    if not any(row["date"] == args.target_date for row in rows):
        raise RuntimeError(f"Open-Meteo forecast did not include target_date={args.target_date}")

    return rows


def daily_value(daily: dict[str, list[object]], column: str, index: int) -> float | None:
    values = daily.get(column)
    if values is None or index >= len(values):
        return None
    value = values[index]
    return None if value is None else float(value)


def add_rolling_features(features: DataFrame) -> DataFrame:
    grid_by_date = Window.partitionBy("grid_id").orderBy("date")
    rolling_7_days = grid_by_date.rowsBetween(-6, 0)

    features = features.withColumn(
        "precipitation_sum_7days",
        F.sum(F.coalesce(F.col("precipitation_sum"), F.lit(0.0))).over(rolling_7_days),
    )

    is_dry = F.when(F.col("precipitation_sum") == F.lit(0.0), F.lit(1)).otherwise(F.lit(0))
    wet_group = F.sum(F.when(is_dry == F.lit(0), F.lit(1)).otherwise(F.lit(0))).over(
        grid_by_date.rowsBetween(Window.unboundedPreceding, 0)
    )

    return (
        features.withColumn("_is_dry", is_dry)
        .withColumn("_wet_group", wet_group)
        .withColumn(
            "dry_days_count",
            F.when(
                F.col("_is_dry") == F.lit(1),
                F.sum("_is_dry").over(
                    Window.partitionBy("grid_id", "_wet_group")
                    .orderBy("date")
                    .rowsBetween(Window.unboundedPreceding, 0)
                ),
            ).otherwise(F.lit(0)),
        )
        .drop("_is_dry", "_wet_group")
    )


def add_time_features(frame: DataFrame) -> DataFrame:
    month = F.month("date")
    dayofyear = F.dayofyear("date")
    return (
        frame.withColumn("month", month.cast("double"))
        .withColumn("dayofyear", dayofyear.cast("double"))
        .withColumn("month_sin", F.sin(2 * F.pi() * month / F.lit(12.0)))
        .withColumn("month_cos", F.cos(2 * F.pi() * month / F.lit(12.0)))
        .withColumn("dayofyear_sin", F.sin(2 * F.pi() * dayofyear / F.lit(366.0)))
        .withColumn("dayofyear_cos", F.cos(2 * F.pi() * dayofyear / F.lit(366.0)))
    )


def build_scoring_frame(spark: SparkSession, rows: list[dict[str, object]], target_date: str) -> DataFrame:
    frame = spark.createDataFrame(rows).withColumn("date", F.to_date("date"))
    frame = add_rolling_features(frame)
    frame = frame.filter(F.col("date") == F.to_date(F.lit(target_date)))
    frame = add_time_features(frame)
    return frame.fillna(0.0, subset=DEFAULT_FEATURE_COLUMNS)


def risk_level_column(score_col: str, medium_threshold: float, high_threshold: float):
    return (
        F.when(F.col(score_col) >= F.lit(high_threshold), F.lit(2))
        .when(F.col(score_col) >= F.lit(medium_threshold), F.lit(1))
        .otherwise(F.lit(0))
    )


def predict_risk(scoring: DataFrame, model_path: str, args: argparse.Namespace) -> DataFrame:
    model = PipelineModel.load(model_path)
    predictions = model.transform(scoring)
    probability_to_score = F.udf(
        lambda probability: float(probability[1]) if probability is not None and len(probability) > 1 else None,
        DoubleType(),
    )
    return (
        predictions.withColumn("risk_score", probability_to_score("probability"))
        .withColumn("risk_level", risk_level_column("risk_score", args.medium_threshold, args.high_threshold))
        .withColumn(
            "risk_label",
            F.when(F.col("risk_level") == F.lit(2), F.lit("high"))
            .when(F.col("risk_level") == F.lit(1), F.lit("medium"))
            .otherwise(F.lit("low")),
        )
    )


def build_geojson(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    features: list[dict[str, object]] = []
    half_step = args.grid_size / 2.0

    for row in rows:
        lat0 = float(row["grid_lat"])
        lon0 = float(row["grid_lon"])
        lat1 = lat0 + args.grid_size
        lon1 = lon0 + args.grid_size
        center_lat = lat0 + half_step
        center_lon = lon0 + half_step
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "grid_id": row["grid_id"],
                    "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
                    "lat": round(center_lat, 6),
                    "lon": round(center_lon, 6),
                    "grid_lat": lat0,
                    "grid_lon": lon0,
                    "risk_score": float(row["risk_score"]),
                    "risk_level": int(row["risk_level"]),
                    "risk_label": row["risk_label"],
                    "model_prediction": float(row["prediction"]),
                    "temperature_2m_max": none_or_float(row["temperature_2m_max"]),
                    "relative_humidity_2m_min": none_or_float(row["relative_humidity_2m_min"]),
                    "wind_speed_10m_max": none_or_float(row["wind_speed_10m_max"]),
                    "precipitation_sum": none_or_float(row["precipitation_sum"]),
                    "precipitation_sum_7days": none_or_float(row["precipitation_sum_7days"]),
                    "dry_days_count": int(row["dry_days_count"]),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon0, lat0],
                            [lon1, lat0],
                            [lon1, lat1],
                            [lon0, lat1],
                            [lon0, lat0],
                        ]
                    ],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def none_or_float(value) -> float | None:
    return None if value is None else float(value)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_risk_levels(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["risk_level"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def copy_local_to_s3a(spark: SparkSession, local_path: Path, destination: str) -> None:
    jvm = spark.sparkContext._jvm
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI(destination), conf)
    source_path = jvm.org.apache.hadoop.fs.Path(local_path.resolve().as_uri())
    destination_path = jvm.org.apache.hadoop.fs.Path(destination)
    fs.copyFromLocalFile(False, True, source_path, destination_path)


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Predict next-day wildfire risk from Open-Meteo forecast data.")
    parser.add_argument("--spark-app-name", default="Wildfire Risk Inference")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--model-prefix", default=os.getenv("RF_MODEL_OUTPUT_PREFIX", "models/random_forest_fire_baseline"))
    parser.add_argument("--predictions-prefix", default=os.getenv("PREDICTIONS_PREFIX", "predictions/fire_risk_forecast"))
    parser.add_argument("--target-date", default=os.getenv("INFERENCE_TARGET_DATE"))
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.5")))
    parser.add_argument("--forecast-timezone", default=os.getenv("FORECAST_TIMEZONE", "Asia/Ho_Chi_Minh"))
    parser.add_argument("--spark-timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))
    parser.add_argument("--open-meteo-url", default=os.getenv("OPEN_METEO_FORECAST_URL", DEFAULT_OPEN_METEO_URL))
    parser.add_argument("--past-days-for-rolling", type=int, default=int(os.getenv("FORECAST_PAST_DAYS", "30")))
    parser.add_argument("--forecast-days", type=int, default=int(os.getenv("FORECAST_DAYS", "2")))
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--medium-threshold", type=float, default=0.33)
    parser.add_argument("--high-threshold", type=float, default=0.66)
    parser.add_argument("--geojson-output", type=Path, default=Path("reports/fire_risk_forecast_latest.geojson"))
    parser.add_argument("--metadata-output", type=Path, default=Path("reports/fire_risk_forecast_latest.json"))
    parser.add_argument("--print-progress", action="store_true")

    args = parser.parse_args()
    if args.target_date is None:
        args.target_date = default_target_date(args.forecast_timezone)
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if args.past_days_for_rolling < 6:
        parser.error("--past-days-for-rolling must be at least 6")
    if args.forecast_days < 1:
        parser.error("--forecast-days must be positive")
    if not 0 <= args.medium_threshold <= args.high_threshold <= 1:
        parser.error("--medium-threshold and --high-threshold must satisfy 0 <= medium <= high <= 1")
    date.fromisoformat(args.target_date)
    return args


def main() -> int:
    args = parse_args()
    features_input = s3a_path(args.minio_bucket, args.features_prefix)
    model_input = s3a_path(args.minio_bucket, args.model_prefix)
    geojson_destination = s3a_path(args.minio_bucket, args.predictions_prefix).rstrip("/") + "/latest.geojson"
    metadata_destination = s3a_path(args.minio_bucket, args.predictions_prefix).rstrip("/") + "/latest.json"

    spark = build_spark(args)
    try:
        features = spark.read.parquet(features_input)
        grid_rows = load_grid(features)
        forecast_rows = fetch_forecast_rows(args, grid_rows)
        scoring = build_scoring_frame(spark, forecast_rows, args.target_date)
        predictions = predict_risk(scoring, model_input, args)

        output_rows = (
            predictions.select(
                "grid_id",
                "date",
                "grid_lat",
                "grid_lon",
                "risk_score",
                "risk_level",
                "risk_label",
                "prediction",
                "temperature_2m_max",
                "relative_humidity_2m_min",
                "wind_speed_10m_max",
                "precipitation_sum",
                "precipitation_sum_7days",
                "dry_days_count",
            )
            .orderBy("grid_lat", "grid_lon")
            .collect()
        )
        output_dicts = [row.asDict(recursive=True) for row in output_rows]

        geojson = build_geojson(output_dicts, args)
        metadata = {
            "status": "ok",
            "target_date": args.target_date,
            "grid_count": len(output_dicts),
            "model_input": model_input,
            "features_input": features_input,
            "open_meteo_url": args.open_meteo_url,
            "daily_variables": DAILY_VARIABLES,
            "medium_threshold": args.medium_threshold,
            "high_threshold": args.high_threshold,
            "risk_level_counts": count_risk_levels(output_dicts),
        }

        write_json(args.geojson_output, geojson)
        write_json(args.metadata_output, metadata)
        copy_local_to_s3a(spark, args.geojson_output, geojson_destination)
        copy_local_to_s3a(spark, args.metadata_output, metadata_destination)

        print(f"Target date: {args.target_date}")
        print(f"Predicted grid cells: {len(output_dicts):,}")
        print(f"Saved GeoJSON to {args.geojson_output}")
        print(f"Uploaded GeoJSON to {geojson_destination}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
