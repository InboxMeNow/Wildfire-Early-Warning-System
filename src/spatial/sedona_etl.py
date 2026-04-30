#!/usr/bin/env python3
"""Sedona-backed spatial ETL for wildfire fire and weather points."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from sedona.spark import SedonaContext
from shapely.geometry import shape
from shapely.ops import unary_union
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType


DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"
DEFAULT_SEDONA_PACKAGES = (
    "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.5.1,"
    "org.datasyslab:geotools-wrapper:1.5.1-28.2"
)
DEFAULT_COUNTRY_BOUNDARY = Path("geo/vietnam_boundary.geojson")


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


def boundary_wkt(path: Path) -> str:
    return load_boundary_geometry(path).wkt


def load_boundary_geometry(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = [shape(feature["geometry"]) for feature in data.get("features", []) if feature.get("geometry")]
    if not geometries:
        raise ValueError(f"No geometry features found in {path}")
    return unary_union(geometries)


def build_sedona(args: argparse.Namespace) -> SparkSession:
    packages = ",".join(
        package.strip()
        for package in [args.hadoop_aws_package, args.sedona_packages]
        if package and package.strip()
    )

    builder = SedonaContext.builder().appName(args.spark_app_name)
    if args.spark_master:
        builder = builder.master(args.spark_master)
    if packages:
        builder = builder.config("spark.jars.packages", packages)
        builder = builder.config("spark.jars.excludes", "edu.ucar:cdm-core")

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

    return SedonaContext.create(builder.getOrCreate())


def ensure_date(frame: DataFrame, column: str) -> DataFrame:
    if column in frame.columns and not isinstance(frame.schema[column].dataType, DateType):
        return frame.withColumn(column, F.to_date(F.col(column)))
    return frame


def drop_duplicates(frame: DataFrame, preferred_subset: list[str]) -> DataFrame:
    subset = [column for column in preferred_subset if column in frame.columns]
    if subset:
        return frame.dropDuplicates(subset)
    return frame.dropDuplicates()


def clean_fires(fires: DataFrame, confidence_threshold: int) -> DataFrame:
    fires = ensure_date(fires, "acq_date")
    fires = ensure_date(fires, "query_start")
    fires = ensure_date(fires, "query_end")

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

    fires = fires.withColumn("confidence_score", confidence_score).filter(
        F.col("confidence_score") >= F.lit(float(confidence_threshold))
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
    weather = ensure_date(weather, "date")
    return drop_duplicates(weather, ["point_id", "latitude", "longitude", "date"])


def with_point_geometry(frame: DataFrame, latitude_column: str = "latitude", longitude_column: str = "longitude") -> DataFrame:
    return frame.withColumn(
        "geometry",
        F.expr(f"ST_Point(CAST({longitude_column} AS DOUBLE), CAST({latitude_column} AS DOUBLE))"),
    )


def filter_to_boundary(frame: DataFrame, boundary: DataFrame) -> DataFrame:
    boundary.createOrReplaceTempView("_sedona_boundary")
    frame.createOrReplaceTempView("_sedona_points")
    return frame.sparkSession.sql(
        """
        SELECT p.*
        FROM _sedona_points p
        CROSS JOIN _sedona_boundary b
        WHERE ST_Contains(b.geometry, p.geometry)
        """
    )


def create_grid_polygons(spark: SparkSession, boundary: DataFrame, boundary_path: Path, grid_size: float) -> DataFrame:
    min_lon, min_lat, max_lon, max_lat = load_boundary_geometry(boundary_path).bounds
    lat_start = math.floor(min_lat / grid_size)
    lat_end = math.ceil(max_lat / grid_size)
    lon_start = math.floor(min_lon / grid_size)
    lon_end = math.ceil(max_lon / grid_size)

    lat_indexes = spark.range(lat_start, lat_end).select(F.col("id").cast("int").alias("grid_lat_index"))
    lon_indexes = spark.range(lon_start, lon_end).select(F.col("id").cast("int").alias("grid_lon_index"))
    grid = (
        lat_indexes.crossJoin(lon_indexes)
        .withColumn("grid_id", F.concat_ws("_", F.col("grid_lat_index"), F.col("grid_lon_index")))
        .withColumn("grid_lat", F.round(F.col("grid_lat_index") * F.lit(grid_size), 6))
        .withColumn("grid_lon", F.round(F.col("grid_lon_index") * F.lit(grid_size), 6))
        .withColumn("min_lon", F.col("grid_lon"))
        .withColumn("min_lat", F.col("grid_lat"))
        .withColumn("max_lon", F.round(F.col("grid_lon") + F.lit(grid_size), 6))
        .withColumn("max_lat", F.round(F.col("grid_lat") + F.lit(grid_size), 6))
    )
    grid = grid.withColumn(
        "geometry",
        F.expr("ST_PolygonFromEnvelope(min_lon, min_lat, max_lon, max_lat)"),
    )
    grid.createOrReplaceTempView("_sedona_grid_all")
    boundary.createOrReplaceTempView("_sedona_boundary")
    return spark.sql(
        """
        SELECT g.grid_id, g.grid_lat_index, g.grid_lon_index, g.grid_lat, g.grid_lon, g.geometry
        FROM _sedona_grid_all g
        CROSS JOIN _sedona_boundary b
        WHERE ST_Intersects(g.geometry, b.geometry)
        """
    )


def attach_grid(points: DataFrame, grid: DataFrame) -> DataFrame:
    points.createOrReplaceTempView("_sedona_points_for_grid")
    grid.createOrReplaceTempView("_sedona_grid")
    original_columns = ", ".join(f"p.`{column}`" for column in points.columns if column != "geometry")
    return points.sparkSession.sql(
        f"""
        SELECT /*+ BROADCAST(g) */
            {original_columns},
            g.grid_id,
            g.grid_lat_index,
            g.grid_lon_index,
            g.grid_lat,
            g.grid_lon
        FROM _sedona_points_for_grid p
        JOIN _sedona_grid g
          ON ST_Contains(g.geometry, p.geometry)
        """
    )


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Run Sedona spatial ETL for wildfire datasets.")
    parser.add_argument("--spark-app-name", default="Wildfire Sedona Spatial ETL")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--sedona-packages", default=os.getenv("SPARK_SEDONA_PACKAGES", DEFAULT_SEDONA_PACKAGES))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--fires-input-prefix", default=os.getenv("FIRES_INPUT_PREFIX", "firms"))
    parser.add_argument("--weather-input-prefix", default=os.getenv("WEATHER_INPUT_PREFIX", "weather"))
    parser.add_argument("--fires-output-prefix", default=os.getenv("SEDONA_FIRES_CLEAN_PREFIX", "firms_sedona_clean"))
    parser.add_argument("--weather-output-prefix", default=os.getenv("SEDONA_WEATHER_CLEAN_PREFIX", "weather_sedona_clean"))
    parser.add_argument("--grid-output-prefix", default=os.getenv("SEDONA_GRID_PREFIX", "grid_sedona"))
    parser.add_argument("--country-boundary", type=Path, default=Path(os.getenv("COUNTRY_BOUNDARY", str(DEFAULT_COUNTRY_BOUNDARY))))
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.5")))
    parser.add_argument("--confidence-threshold", type=int, default=int(os.getenv("FIRES_CONFIDENCE_THRESHOLD", "30")))
    parser.add_argument("--write-mode", choices=["append", "error", "errorifexists", "ignore", "overwrite"], default=os.getenv("SPARK_WRITE_MODE", "overwrite"))
    parser.add_argument("--timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))
    parser.add_argument("--print-counts", action="store_true")
    args = parser.parse_args()

    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if not 0 <= args.confidence_threshold <= 100:
        parser.error("--confidence-threshold must be in 0..100")
    if not args.country_boundary.exists():
        parser.error(f"--country-boundary not found: {args.country_boundary}")
    return args


def main() -> int:
    args = parse_args()
    spark = build_sedona(args)

    fires_input = s3a_path(args.minio_bucket, args.fires_input_prefix)
    weather_input = s3a_path(args.minio_bucket, args.weather_input_prefix)
    fires_output = s3a_path(args.minio_bucket, args.fires_output_prefix)
    weather_output = s3a_path(args.minio_bucket, args.weather_output_prefix)
    grid_output = s3a_path(args.minio_bucket, args.grid_output_prefix)

    try:
        boundary = spark.createDataFrame([(boundary_wkt(args.country_boundary),)], ["wkt"]).selectExpr(
            "ST_GeomFromWKT(wkt) AS geometry"
        )
        grid = create_grid_polygons(spark, boundary, args.country_boundary, args.grid_size).cache()

        fires = clean_fires(spark.read.parquet(fires_input), args.confidence_threshold)
        weather = clean_weather(spark.read.parquet(weather_input))

        fires_points = filter_to_boundary(with_point_geometry(fires), boundary)
        weather_points = filter_to_boundary(with_point_geometry(weather), boundary)

        fires_with_grid = attach_grid(fires_points, grid)
        weather_with_grid = attach_grid(weather_points, grid)

        grid.write.mode(args.write_mode).parquet(grid_output)
        fires_with_grid.write.mode(args.write_mode).parquet(fires_output)
        weather_with_grid.write.mode(args.write_mode).parquet(weather_output)

        print(f"Wrote Sedona grid polygons to {grid_output}")
        print(f"Wrote Sedona FIRMS data to {fires_output}")
        print(f"Wrote Sedona weather data to {weather_output}")

        if args.print_counts:
            print(f"Grid cells: {grid.count():,}")
            print(f"FIRMS rows: {fires_with_grid.count():,}")
            print(f"Weather rows: {weather_with_grid.count():,}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
