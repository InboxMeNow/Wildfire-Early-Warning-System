#!/usr/bin/env python3
"""Detect real-time wildfire hot zones from Kafka fire events with Spark."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    concat,
    count,
    current_timestamp,
    date_format,
    floor,
    from_json,
    lit,
    struct,
    to_json,
    to_timestamp,
    window,
)
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_INPUT_TOPIC = "fire-events"
DEFAULT_ALERT_TOPIC = "alerts"
DEFAULT_CHECKPOINT = "checkpoints/streaming/fire-hot-zones"
DEFAULT_GRID_SIZE_DEGREES = 0.5
DEFAULT_THRESHOLD = 5
DEFAULT_WATERMARK = "10 minutes"
DEFAULT_WINDOW = "1 hour"
DEFAULT_TRIGGER = "10 seconds"
DEFAULT_KAFKA_PACKAGE = "auto"


FIRE_EVENT_SCHEMA = StructType(
    [
        StructField("firms_source", StringType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("acq_date", StringType()),
        StructField("acq_time", IntegerType()),
        StructField("acquired_at_utc", StringType()),
        StructField("fetched_at_utc", StringType()),
        StructField("satellite", StringType()),
        StructField("instrument", StringType()),
        StructField("confidence", StringType()),
        StructField("frp", DoubleType()),
        StructField("daynight", StringType()),
    ]
)


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

    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming job for Kafka fire hot-zone alerts."
    )
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS))
    parser.add_argument("--input-topic", default=os.getenv("KAFKA_FIRE_TOPIC", DEFAULT_INPUT_TOPIC))
    parser.add_argument("--alert-topic", default=os.getenv("KAFKA_ALERT_TOPIC", DEFAULT_ALERT_TOPIC))
    parser.add_argument("--checkpoint-location", default=os.getenv("SPARK_STREAM_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--grid-size-degrees", type=float, default=DEFAULT_GRID_SIZE_DEGREES)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--watermark", default=DEFAULT_WATERMARK)
    parser.add_argument("--window", default=DEFAULT_WINDOW)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--starting-offsets", default=os.getenv("KAFKA_STARTING_OFFSETS", "latest"))
    parser.add_argument(
        "--kafka-package",
        default=os.getenv("SPARK_KAFKA_PACKAGE", DEFAULT_KAFKA_PACKAGE),
        help="Spark Kafka connector package. Use 'auto' to match the detected Spark version.",
    )
    args = parser.parse_args()

    if args.grid_size_degrees <= 0:
        parser.error("--grid-size-degrees must be positive")
    if args.threshold < 1:
        parser.error("--threshold must be at least 1")
    return args


def detect_spark_version() -> str:
    spark_home = os.getenv("SPARK_HOME", "")
    match = re.search(r"spark[-_](\d+\.\d+\.\d+)", spark_home)
    if match:
        return match.group(1)
    return pyspark.__version__


def resolve_kafka_package(package: str) -> str | None:
    if package.lower() in {"", "none", "false"}:
        return None
    if package.lower() == "auto":
        spark_version = detect_spark_version()
        return f"org.apache.spark:spark-sql-kafka-0-10_2.12:{spark_version}"
    return package


def create_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName("FireStreamProcessor")
    kafka_package = resolve_kafka_package(args.kafka_package)
    if kafka_package:
        builder = builder.config("spark.jars.packages", kafka_package)
    return builder.getOrCreate()


def parse_fire_events(raw_events: DataFrame, grid_size_degrees: float) -> DataFrame:
    parsed = raw_events.select(
        from_json(col("value").cast("string"), FIRE_EVENT_SCHEMA).alias("event"),
        col("timestamp").alias("kafka_timestamp"),
    ).select("event.*", "kafka_timestamp")

    grid_factor = 1.0 / grid_size_degrees
    return (
        parsed.filter(col("latitude").isNotNull() & col("longitude").isNotNull())
        .withColumn(
            "event_time",
            to_timestamp(col("acquired_at_utc")),
        )
        .withColumn("event_time", col("event_time").cast("timestamp"))
        .filter(col("event_time").isNotNull())
        .withColumn("grid_lat_index", floor(col("latitude") * lit(grid_factor)).cast("int"))
        .withColumn("grid_lon_index", floor(col("longitude") * lit(grid_factor)).cast("int"))
        .withColumn("grid_id", concat(col("grid_lat_index"), lit("_"), col("grid_lon_index")))
    )


def build_hot_zone_alerts(events: DataFrame, args: argparse.Namespace) -> DataFrame:
    hot_zones = (
        events.withWatermark("event_time", args.watermark)
        .groupBy(
            col("grid_id"),
            col("grid_lat_index"),
            col("grid_lon_index"),
            window(col("event_time"), args.window).alias("window"),
        )
        .agg(count("*").alias("fire_count"))
        .filter(col("fire_count") > lit(args.threshold))
    )

    return (
        hot_zones.withColumn("alert_type", lit("HOT_ZONE"))
        .withColumn("severity", lit("high"))
        .withColumn("threshold", lit(args.threshold))
        .withColumn("created_at_utc", current_timestamp())
        .withColumn(
            "alert_id",
            concat(
                col("grid_id"),
                lit("|"),
                date_format(col("window.start"), "yyyy-MM-dd'T'HH:mm:ss"),
                lit("|"),
                date_format(col("window.end"), "yyyy-MM-dd'T'HH:mm:ss"),
            ),
        )
    )


def write_alerts_to_kafka(batch_df: DataFrame, batch_id: int, bootstrap_servers: str, alert_topic: str) -> None:
    if batch_df.rdd.isEmpty():
        return

    kafka_df = batch_df.select(
        col("alert_id").cast("string").alias("key"),
        to_json(
            struct(
                col("alert_id"),
                col("alert_type"),
                col("severity"),
                col("grid_id"),
                col("grid_lat_index"),
                col("grid_lon_index"),
                col("fire_count"),
                col("threshold"),
                col("window.start").alias("window_start_utc"),
                col("window.end").alias("window_end_utc"),
                col("created_at_utc"),
                lit(batch_id).alias("spark_batch_id"),
            )
        ).alias("value"),
    )

    kafka_df.write.format("kafka").option("kafka.bootstrap.servers", bootstrap_servers).option(
        "topic", alert_topic
    ).save()


def main() -> int:
    args = parse_args()
    spark = create_spark(args)
    spark.sparkContext.setLogLevel("WARN")

    raw_events = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.input_topic)
        .option("startingOffsets", args.starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    fire_events = parse_fire_events(raw_events, grid_size_degrees=args.grid_size_degrees)
    alerts = build_hot_zone_alerts(fire_events, args)

    query = (
        alerts.writeStream.outputMode("update")
        .foreachBatch(
            lambda batch_df, batch_id: write_alerts_to_kafka(
                batch_df=batch_df,
                batch_id=batch_id,
                bootstrap_servers=args.bootstrap_servers,
                alert_topic=args.alert_topic,
            )
        )
        .option("checkpointLocation", args.checkpoint_location)
        .trigger(processingTime=args.trigger)
        .start()
    )

    print(
        "FireStreamProcessor started: "
        f"{args.input_topic} -> {args.alert_topic}, "
        f"threshold > {args.threshold} fires/grid/{args.window}."
    )
    query.awaitTermination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
