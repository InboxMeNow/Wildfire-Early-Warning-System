#!/usr/bin/env python3
"""
Clean raw wildfire and weather Parquet datasets from MinIO with Spark.

Defaults match docker-compose.yml:
    MINIO_ENDPOINT=http://localhost:9000
    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin
    MINIO_BUCKET=wildfire-data

Input:
    s3a://wildfire-data/firms/
    s3a://wildfire-data/weather/

Output:
    s3a://wildfire-data/firms_clean/
    s3a://wildfire-data/weather_clean/

Run locally:
    python 04_etl_clean.py

Run with a Spark cluster:
    spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 \
        04_etl_clean.py --spark-master spark://localhost:7077

Run inside this docker-compose network:
    docker compose exec spark-master /opt/spark/bin/spark-submit \
        --packages org.apache.hadoop:hadoop-aws:3.3.4 \
        --master spark://spark-master:7077 \
        /workspace/04_etl_clean.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType


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

    # S3A/MinIO settings.
    builder = (
        builder.config("spark.hadoop.fs.s3a.endpoint", args.minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", args.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", args.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(args.minio_endpoint.startswith("https://")).lower())
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    )

    # Keep timestamps deterministic across local machines and Docker containers.
    builder = builder.config("spark.sql.session.timeZone", args.timezone)

    return builder.getOrCreate()


def existing_columns(frame: DataFrame, columns: list[str]) -> list[str]:
    available = set(frame.columns)
    return [column for column in columns if column in available]


def convert_date_columns(frame: DataFrame, columns: list[str]) -> DataFrame:
    for column in existing_columns(frame, columns):
        if isinstance(frame.schema[column].dataType, DateType):
            continue
        frame = frame.withColumn(column, F.to_date(F.col(column)))
    return frame


def drop_duplicates(frame: DataFrame, preferred_subset: list[str]) -> DataFrame:
    subset = existing_columns(frame, preferred_subset)
    if subset:
        return frame.dropDuplicates(subset)
    return frame.dropDuplicates()


def clean_fires(fires: DataFrame, confidence_threshold: int) -> DataFrame:
    fires = convert_date_columns(fires, ["acq_date", "query_start", "query_end", "date"])

    if "confidence" not in fires.columns:
        raise ValueError("FIRMS input is missing required column: confidence")

    confidence_text = F.lower(F.trim(F.col("confidence").cast(StringType())))
    numeric_confidence = F.regexp_extract(confidence_text, r"^([0-9]+(?:\.[0-9]+)?)$", 1).cast("double")
    confidence_score = (
        F.when(confidence_text.isin("h", "high"), F.lit(100.0))
        .when(confidence_text.isin("n", "nominal", "medium", "m"), F.lit(50.0))
        .when(confidence_text.isin("l", "low"), F.lit(0.0))
        .otherwise(numeric_confidence)
    )

    fires = (
        fires.withColumn("confidence_score", confidence_score)
        .filter(F.col("confidence_score") >= F.lit(float(confidence_threshold)))
    )

    return drop_duplicates(
        fires,
        [
            "firms_source",
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
        ],
    )


def clean_weather(weather: DataFrame) -> DataFrame:
    weather = convert_date_columns(weather, ["date"])
    return drop_duplicates(weather, ["point_id", "latitude", "longitude", "date"])


def write_parquet(frame: DataFrame, output_path: str, mode: str) -> None:
    frame.write.mode(mode).parquet(output_path)


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Clean FIRMS and weather Parquet datasets in MinIO using Spark."
    )
    parser.add_argument("--spark-app-name", default="Wildfire ETL")
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
    parser.add_argument(
        "--minio-endpoint",
        default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT),
    )
    parser.add_argument(
        "--minio-access-key",
        default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY),
    )
    parser.add_argument(
        "--minio-secret-key",
        default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY),
    )
    parser.add_argument(
        "--minio-bucket",
        default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET),
    )
    parser.add_argument(
        "--fires-input-prefix",
        default=os.getenv("FIRES_INPUT_PREFIX", "firms"),
    )
    parser.add_argument(
        "--weather-input-prefix",
        default=os.getenv("WEATHER_INPUT_PREFIX", "weather"),
    )
    parser.add_argument(
        "--fires-output-prefix",
        default=os.getenv("FIRES_CLEAN_PREFIX", "firms_clean"),
    )
    parser.add_argument(
        "--weather-output-prefix",
        default=os.getenv("WEATHER_CLEAN_PREFIX", "weather_clean"),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=int(os.getenv("FIRES_CONFIDENCE_THRESHOLD", "30")),
    )
    parser.add_argument(
        "--write-mode",
        choices=["append", "error", "errorifexists", "ignore", "overwrite"],
        default=os.getenv("SPARK_WRITE_MODE", "overwrite"),
    )
    parser.add_argument(
        "--timezone",
        default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"),
    )

    args = parser.parse_args()
    if not args.minio_endpoint:
        parser.error("--minio-endpoint must not be empty")
    if not 0 <= args.confidence_threshold <= 100:
        parser.error("--confidence-threshold must be in 0..100")
    return args


def main() -> int:
    args = parse_args()

    fires_input = s3a_path(args.minio_bucket, args.fires_input_prefix)
    weather_input = s3a_path(args.minio_bucket, args.weather_input_prefix)
    fires_output = s3a_path(args.minio_bucket, args.fires_output_prefix)
    weather_output = s3a_path(args.minio_bucket, args.weather_output_prefix)

    spark = build_spark(args)
    try:
        fires = spark.read.parquet(fires_input)
        weather = spark.read.parquet(weather_input)

        fires_clean = clean_fires(fires, args.confidence_threshold)
        weather_clean = clean_weather(weather)

        write_parquet(fires_clean, fires_output, args.write_mode)
        write_parquet(weather_clean, weather_output, args.write_mode)

        print(f"Wrote cleaned FIRMS data to {fires_output}")
        print(f"Wrote cleaned weather data to {weather_output}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
