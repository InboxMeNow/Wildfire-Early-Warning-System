#!/usr/bin/env python3
"""
Build ML-ready wildfire risk features from cleaned FIRMS and weather data in MinIO.

Input:
    s3a://wildfire-data/firms_clean/
    s3a://wildfire-data/weather_clean/

Output:
    s3a://wildfire-data/features/

Run inside this docker-compose network:
    docker compose exec spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /workspace/05_feature_engineering.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType


DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"


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
        .config("spark.sql.session.timeZone", args.timezone)
    )

    return builder.getOrCreate()


def ensure_date(frame: DataFrame, column: str) -> DataFrame:
    if column in frame.columns and not isinstance(frame.schema[column].dataType, DateType):
        return frame.withColumn(column, F.to_date(F.col(column)))
    return frame


def with_grid_columns(frame: DataFrame, grid_size: float) -> DataFrame:
    lat_index = F.floor(F.col("latitude") / F.lit(grid_size)).cast("int")
    lon_index = F.floor(F.col("longitude") / F.lit(grid_size)).cast("int")

    return (
        frame.withColumn("grid_lat_index", lat_index)
        .withColumn("grid_lon_index", lon_index)
        .withColumn("grid_id", F.concat_ws("_", F.col("grid_lat_index"), F.col("grid_lon_index")))
        .withColumn("grid_lat", F.round(F.col("grid_lat_index") * F.lit(grid_size), 6))
        .withColumn("grid_lon", F.round(F.col("grid_lon_index") * F.lit(grid_size), 6))
    )


def aggregate_weather(weather: DataFrame, grid_size: float) -> DataFrame:
    weather = ensure_date(weather, "date")
    weather = with_grid_columns(weather, grid_size)

    grouped = weather.groupBy(
        "grid_id",
        "grid_lat_index",
        "grid_lon_index",
        "grid_lat",
        "grid_lon",
        "date",
    )

    return grouped.agg(
        F.avg("temperature_2m_max").alias("temperature_2m_max"),
        F.avg("relative_humidity_2m_min").alias("relative_humidity_2m_min"),
        F.avg("wind_speed_10m_max").alias("wind_speed_10m_max"),
        F.avg("precipitation_sum").alias("precipitation_sum"),
        F.avg("station_distance_km").alias("station_distance_km"),
        F.countDistinct("point_id").alias("weather_points_count"),
    )


def aggregate_fires(fires: DataFrame, grid_size: float) -> DataFrame:
    fires = ensure_date(fires, "acq_date")
    fires = with_grid_columns(fires, grid_size)

    return (
        fires.groupBy("grid_id", F.col("acq_date").alias("date"))
        .agg(
            F.count(F.lit(1)).cast("int").alias("fire_count"),
            F.avg("confidence_score").alias("avg_fire_confidence"),
            F.avg("frp").alias("avg_frp"),
            F.max("frp").alias("max_frp"),
        )
    )


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


def build_features(fires: DataFrame, weather: DataFrame, grid_size: float) -> DataFrame:
    weather_daily = aggregate_weather(weather, grid_size)
    fire_daily = aggregate_fires(fires, grid_size)

    features = (
        weather_daily.join(fire_daily, on=["grid_id", "date"], how="left")
        .fillna(
            {
                "fire_count": 0,
                "avg_fire_confidence": 0.0,
                "avg_frp": 0.0,
                "max_frp": 0.0,
            }
        )
        .withColumn("fire_occurred", F.when(F.col("fire_count") > 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "risk_level",
            F.when(F.col("fire_count") == 0, F.lit(0))
            .when(F.col("fire_count") <= 5, F.lit(1))
            .otherwise(F.lit(2)),
        )
    )

    features = add_rolling_features(features)

    return features.select(
        "grid_id",
        "grid_lat_index",
        "grid_lon_index",
        "grid_lat",
        "grid_lon",
        "date",
        "temperature_2m_max",
        "relative_humidity_2m_min",
        "wind_speed_10m_max",
        "precipitation_sum",
        "precipitation_sum_7days",
        "dry_days_count",
        "station_distance_km",
        "weather_points_count",
        "fire_count",
        "avg_fire_confidence",
        "avg_frp",
        "max_frp",
        "fire_occurred",
        "risk_level",
    )


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Build grid-day wildfire ML features from cleaned FIRMS and weather data."
    )
    parser.add_argument("--spark-app-name", default="Wildfire Feature Engineering")
    parser.add_argument(
        "--spark-master",
        default=os.getenv("SPARK_MASTER"),
        help="Optional Spark master URL, for example spark://localhost:7077.",
    )
    parser.add_argument(
        "--hadoop-aws-package",
        default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE),
        help="Maven package used by Spark for S3A. Set empty string if jars are already installed.",
    )
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--fires-input-prefix", default=os.getenv("FIRES_CLEAN_PREFIX", "firms_clean"))
    parser.add_argument("--weather-input-prefix", default=os.getenv("WEATHER_CLEAN_PREFIX", "weather_clean"))
    parser.add_argument("--features-output-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.5")))
    parser.add_argument(
        "--write-mode",
        choices=["append", "error", "errorifexists", "ignore", "overwrite"],
        default=os.getenv("SPARK_WRITE_MODE", "overwrite"),
    )
    parser.add_argument("--timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))
    parser.add_argument(
        "--print-counts",
        action="store_true",
        help="Print row and grid counts after writing. This triggers extra Spark jobs.",
    )

    args = parser.parse_args()
    if not args.minio_endpoint:
        parser.error("--minio-endpoint must not be empty")
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    return args


def main() -> int:
    args = parse_args()

    fires_input = s3a_path(args.minio_bucket, args.fires_input_prefix)
    weather_input = s3a_path(args.minio_bucket, args.weather_input_prefix)
    features_output = s3a_path(args.minio_bucket, args.features_output_prefix)

    spark = build_spark(args)
    try:
        fires = spark.read.parquet(fires_input)
        weather = spark.read.parquet(weather_input)

        features = build_features(fires, weather, args.grid_size)
        features.write.mode(args.write_mode).parquet(features_output)

        print(f"Wrote feature dataset to {features_output}")
        if args.print_counts:
            written = spark.read.parquet(features_output)
            print(f"Feature rows: {written.count():,}")
            print(f"Grid cells: {written.select('grid_id').distinct().count():,}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
