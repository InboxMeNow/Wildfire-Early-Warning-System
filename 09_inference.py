#!/usr/bin/env python3
"""
Run multi-day wildfire risk inference from Open-Meteo forecast data.

Input:
    s3a://wildfire-data/features/
    MLflow registry Production model by default

Output:
    reports/fire_risk_forecast_latest.geojson
    reports/fire_risk_forecast_latest.json
    s3a://wildfire-data/predictions/fire_risk_forecast/latest.geojson
    s3a://wildfire-data/predictions/fire_risk_forecast/latest.json

Run inside this docker-compose network:
    docker compose exec -T spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /workspace/09_inference.py --forecast-horizon-days 5
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
MAX_FORECAST_HORIZON_DAYS = 5
REGISTERED_MODEL_NAMES = {
    "rf_tuned": "wildfire-rf-tuned",
    "gbt": "wildfire-gbt",
}
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
    "mean_ndvi_30days",
    "station_distance_km",
    "weather_points_count",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "precip_lag_1",
    "precip_lag_3",
    "precip_lag_7",
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_7",
    "humidity_lag_1",
    "wind_lag_1",
    "ndvi_lag_1",
    "temp_7d_avg",
    "temp_7d_std",
    "temp_30d_avg",
    "temp_30d_std",
    "humidity_7d_avg",
    "humidity_30d_avg",
    "wind_7d_avg",
    "precip_14d_sum",
    "precip_30d_sum",
    "dry_days_14d",
    "dry_days_30d",
    "ndvi_30d_avg",
    "ndvi_30d_min",
    "heat_dryness_index",
    "wind_dryness_index",
    "vegetation_dryness_index",
    "lat_band",
    "lon_band",
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


def load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_uri(args: argparse.Namespace, prefix_or_uri: str) -> str:
    if prefix_or_uri.startswith(("models:/", "runs:/")):
        return prefix_or_uri
    if prefix_or_uri.startswith("s3a://"):
        return prefix_or_uri.rstrip("/") + "/"
    return s3a_path(args.minio_bucket, prefix_or_uri)

def infer_model_name(args: argparse.Namespace, metrics: dict[str, object], model_input: str) -> str | None:
    if "wildfire-gbt" in model_input:
        return "gbt"
    if "wildfire-rf-tuned" in model_input:
        return "rf_tuned"

    normalized_input = model_input.rstrip("/")
    for model_name in ("rf_baseline", "rf_tuned", "gbt"):
        model_metrics = metrics.get(model_name, {})
        if not isinstance(model_metrics, dict):
            continue
        model_output = model_metrics.get("model_output")
        if isinstance(model_output, str) and model_uri(args, model_output).rstrip("/") == normalized_input:
            return model_name

    if args.model_prefix:
        token = args.model_prefix.strip("/").split("/")[-1]
        if token in metrics:
            return token

    best_model = metrics.get("best_model")
    return best_model if isinstance(best_model, str) else None


def resolve_model_input(args: argparse.Namespace, metrics: dict[str, object]) -> str:
    if args.model_prefix:
        return model_uri(args, args.model_prefix)

    if not args.disable_mlflow_registry:
        if args.registry_model_uri:
            return model_uri(args, args.registry_model_uri)

        best_model = metrics.get("best_model")
        if isinstance(best_model, str):
            model_metrics = metrics.get(best_model, {})
            if isinstance(model_metrics, dict):
                production_model_uri = model_metrics.get("production_model_uri")
                if isinstance(production_model_uri, str) and production_model_uri:
                    return model_uri(args, production_model_uri)
                registered_model_uri = model_metrics.get("registered_model_uri")
                if isinstance(registered_model_uri, str) and registered_model_uri:
                    return model_uri(args, registered_model_uri)
                registered_model_name = model_metrics.get("registered_model_name")
                if isinstance(registered_model_name, str) and registered_model_name:
                    return f"models:/{registered_model_name}/Production"

        return f"models:/{REGISTERED_MODEL_NAMES['gbt']}/Production"

    best_model = metrics.get("best_model")
    if isinstance(best_model, str):
        model_metrics = metrics.get(best_model, {})
        if isinstance(model_metrics, dict):
            model_output = model_metrics.get("model_output")
            if isinstance(model_output, str) and model_output:
                return model_uri(args, model_output)

    return model_uri(args, "models/gbt")


def apply_threshold_defaults(args: argparse.Namespace, metrics: dict[str, object], model_name: str | None) -> None:
    if args.medium_threshold is not None and args.high_threshold is not None:
        return

    optimal_threshold = None
    if isinstance(model_name, str):
        model_metrics = metrics.get(model_name, {})
        if isinstance(model_metrics, dict):
            value = model_metrics.get("optimal_threshold")
            if isinstance(value, (int, float)):
                optimal_threshold = float(value)

    if args.medium_threshold is None:
        args.medium_threshold = optimal_threshold if optimal_threshold is not None else 0.33
    if args.high_threshold is None:
        args.high_threshold = max(0.75, min(0.95, float(args.medium_threshold) + 0.15))


def load_prediction_model(model_input: str, args: argparse.Namespace):
    if model_input.startswith(("models:/", "runs:/")):
        try:
            import mlflow
            import mlflow.spark
        except ImportError as exc:
            raise RuntimeError("MLflow is required to load registry model URIs.") from exc

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_registry_uri(args.mlflow_registry_uri or args.mlflow_tracking_uri)
        try:
            return mlflow.spark.load_model(model_input)
        except Exception as exc:
            if not model_input.startswith("models:/"):
                raise
            sparkml_path = resolve_registry_sparkml_path(mlflow, model_input)
            print(
                "mlflow.spark.load_model failed for registry URI; "
                f"loading Spark ML pipeline from registered source instead: {sparkml_path}. "
                f"Original error: {exc}"
            )
            return PipelineModel.load(sparkml_path)

    return PipelineModel.load(model_input)


def resolve_registry_sparkml_path(mlflow_module, model_input: str) -> str:
    client = mlflow_module.tracking.MlflowClient()
    name, selector = parse_models_uri(model_input)
    if selector.isdigit():
        version = client.get_model_version(name, selector)
    else:
        versions = client.get_latest_versions(name, [selector])
        if not versions:
            raise RuntimeError(f"No MLflow model version found for {model_input}")
        version = versions[0]

    tags = getattr(version, "tags", {}) or {}
    spark_pipeline_uri = tags.get("spark_pipeline_uri")
    if isinstance(spark_pipeline_uri, str) and spark_pipeline_uri:
        return spark_pipeline_uri.rstrip("/") + "/"

    source = version.source.rstrip("/")
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "file":
        source_path = Path(urllib.parse.unquote(parsed.path))
        sparkml_path = source_path / "sparkml"
        if spark_metadata_has_data(sparkml_path):
            return str(sparkml_path)
        if spark_metadata_has_data(source_path):
            return str(source_path)
        return str(sparkml_path)
    if source.endswith("/model"):
        return source + "/sparkml"
    return source.rstrip("/") + "/"


def parse_models_uri(model_input: str) -> tuple[str, str]:
    if not model_input.startswith("models:/"):
        raise ValueError(f"Invalid MLflow models URI: {model_input}")
    suffix = model_input[len("models:/"):].strip("/")
    parts = suffix.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid MLflow models URI: {model_input}")
    return "/".join(parts[:-1]), parts[-1]

def spark_metadata_has_data(path: Path) -> bool:
    metadata_path = path / "metadata"
    if not metadata_path.exists():
        return False
    return any(child.name.startswith("part-") for child in metadata_path.iterdir())


def current_date_for_timezone(timezone: str) -> date:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(timezone)).date()
        except Exception:
            pass

    offset_hours = 7 if timezone == "Asia/Ho_Chi_Minh" else 0
    return (datetime.utcnow() + timedelta(hours=offset_hours)).date()

def default_target_date(timezone: str) -> str:
    return (current_date_for_timezone(timezone) + timedelta(days=1)).isoformat()

def forecast_target_dates(args: argparse.Namespace) -> list[date]:
    start = date.fromisoformat(args.target_date)
    return [start + timedelta(days=offset) for offset in range(args.forecast_horizon_days)]

def forecast_end_date(args: argparse.Namespace) -> date:
    return forecast_target_dates(args)[-1]

def required_forecast_days(args: argparse.Namespace) -> int:
    today = current_date_for_timezone(args.forecast_timezone)
    return max(1, (forecast_end_date(args) - today).days + 1)


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
    ndvi_latest = None
    if "mean_ndvi_30days" in features.columns:
        latest_window = Window.partitionBy("grid_id").orderBy(F.col("date").desc())
        ndvi_latest = (
            features.select("grid_id", "date", "mean_ndvi_30days")
            .filter(F.col("mean_ndvi_30days").isNotNull())
            .withColumn("_row_number", F.row_number().over(latest_window))
            .filter(F.col("_row_number") == F.lit(1))
            .select(
                "grid_id",
                F.col("mean_ndvi_30days").cast("double").alias("mean_ndvi_30days"),
                F.col("date").alias("ndvi_source_date"),
            )
        )

    stats = (
        features.groupBy("grid_id", "grid_lat_index", "grid_lon_index", "grid_lat", "grid_lon")
        .agg(
            F.avg("station_distance_km").alias("station_distance_km"),
            F.avg("weather_points_count").alias("weather_points_count"),
        )
        .orderBy("grid_lat_index", "grid_lon_index")
    )
    if ndvi_latest is not None:
        stats = stats.join(ndvi_latest, on="grid_id", how="left")
    else:
        stats = stats.withColumn("mean_ndvi_30days", F.lit(0.0).cast("double"))
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
    target_dates = {target.isoformat() for target in forecast_target_dates(args)}
    target_end = forecast_end_date(args)
    rows: list[dict[str, object]] = []

    for index, grid in enumerate(grid_rows, start=1):
        latitude = float(grid["grid_lat"])
        longitude = float(grid["grid_lon"])
        daily = open_meteo_daily(args, latitude, longitude)
        dates = [date.fromisoformat(value) for value in daily["time"]]

        for day_index, current_date in enumerate(dates):
            if current_date > target_end:
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
                    "mean_ndvi_30days": float(grid.get("mean_ndvi_30days") or 0.0),
                    "station_distance_km": float(grid["station_distance_km"] or 0.0),
                    "weather_points_count": float(grid["weather_points_count"] or 1.0),
                }
            )

        if args.request_delay_seconds > 0 and index < len(grid_rows):
            time.sleep(args.request_delay_seconds)

        if args.print_progress and (index % 25 == 0 or index == len(grid_rows)):
            print(f"Fetched forecast for {index:,}/{len(grid_rows):,} grid cells")

    available_target_dates = {row["date"] for row in rows if row["date"] in target_dates}
    missing_dates = sorted(target_dates - available_target_dates)
    if missing_dates:
        raise RuntimeError(f"Open-Meteo forecast did not include target forecast date(s): {', '.join(missing_dates)}")

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


def add_advanced_features(frame: DataFrame) -> DataFrame:
    frame = add_time_features(frame)

    grid_by_date = Window.partitionBy("grid_id").orderBy("date")
    prev_7 = grid_by_date.rowsBetween(-7, -1)
    prev_14 = grid_by_date.rowsBetween(-14, -1)
    prev_30 = grid_by_date.rowsBetween(-30, -1)
    is_dry_day = F.when(F.col("precipitation_sum") < F.lit(1.0), F.lit(1.0)).otherwise(F.lit(0.0))

    return (
        frame.withColumn("precip_lag_1", F.lag("precipitation_sum", 1).over(grid_by_date))
        .withColumn("precip_lag_3", F.lag("precipitation_sum", 3).over(grid_by_date))
        .withColumn("precip_lag_7", F.lag("precipitation_sum", 7).over(grid_by_date))
        .withColumn("temp_lag_1", F.lag("temperature_2m_max", 1).over(grid_by_date))
        .withColumn("temp_lag_3", F.lag("temperature_2m_max", 3).over(grid_by_date))
        .withColumn("temp_lag_7", F.lag("temperature_2m_max", 7).over(grid_by_date))
        .withColumn("humidity_lag_1", F.lag("relative_humidity_2m_min", 1).over(grid_by_date))
        .withColumn("wind_lag_1", F.lag("wind_speed_10m_max", 1).over(grid_by_date))
        .withColumn("ndvi_lag_1", F.lag("mean_ndvi_30days", 1).over(grid_by_date))
        .withColumn("temp_7d_avg", F.avg("temperature_2m_max").over(prev_7))
        .withColumn("temp_7d_std", F.stddev("temperature_2m_max").over(prev_7))
        .withColumn("temp_30d_avg", F.avg("temperature_2m_max").over(prev_30))
        .withColumn("temp_30d_std", F.stddev("temperature_2m_max").over(prev_30))
        .withColumn("humidity_7d_avg", F.avg("relative_humidity_2m_min").over(prev_7))
        .withColumn("humidity_30d_avg", F.avg("relative_humidity_2m_min").over(prev_30))
        .withColumn("wind_7d_avg", F.avg("wind_speed_10m_max").over(prev_7))
        .withColumn("precip_14d_sum", F.sum("precipitation_sum").over(prev_14))
        .withColumn("precip_30d_sum", F.sum("precipitation_sum").over(prev_30))
        .withColumn("dry_days_14d", F.sum(is_dry_day).over(prev_14))
        .withColumn("dry_days_30d", F.sum(is_dry_day).over(prev_30))
        .withColumn("ndvi_30d_avg", F.avg("mean_ndvi_30days").over(prev_30))
        .withColumn("ndvi_30d_min", F.min("mean_ndvi_30days").over(prev_30))
        .withColumn(
            "heat_dryness_index",
            F.col("temperature_2m_max") * (F.lit(100.0) - F.col("relative_humidity_2m_min")),
        )
        .withColumn(
            "wind_dryness_index",
            F.col("wind_speed_10m_max") * (F.lit(100.0) - F.col("relative_humidity_2m_min")),
        )
        .withColumn(
            "vegetation_dryness_index",
            F.coalesce(F.col("mean_ndvi_30days"), F.lit(0.0)) * F.coalesce(F.col("dry_days_30d"), F.lit(0.0)),
        )
        .withColumn("lat_band", F.floor(F.col("grid_lat") / F.lit(5.0)).cast("double"))
        .withColumn("lon_band", F.floor(F.col("grid_lon") / F.lit(5.0)).cast("double"))
    )


def build_scoring_frame(
    spark: SparkSession,
    rows: list[dict[str, object]],
    target_start_date: str,
    target_end_date: str,
) -> DataFrame:
    frame = spark.createDataFrame(rows).withColumn("date", F.to_date("date"))
    frame = add_rolling_features(frame)
    frame = add_advanced_features(frame)
    frame = frame.filter(
        (F.col("date") >= F.to_date(F.lit(target_start_date)))
        & (F.col("date") <= F.to_date(F.lit(target_end_date)))
    )
    return frame.fillna(0.0, subset=DEFAULT_FEATURE_COLUMNS)


def risk_level_column(score_col: str, medium_threshold: float, high_threshold: float):
    return (
        F.when(F.col(score_col) >= F.lit(high_threshold), F.lit(2))
        .when(F.col(score_col) >= F.lit(medium_threshold), F.lit(1))
        .otherwise(F.lit(0))
    )


def predict_risk(scoring: DataFrame, model_path: str, args: argparse.Namespace) -> DataFrame:
    model = load_prediction_model(model_path, args)
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


def _polygon_coords(polygon) -> list[list[list[float]]]:
    exterior = [[float(x), float(y)] for x, y in polygon.exterior.coords]
    rings = [exterior]
    for interior in polygon.interiors:
        rings.append([[float(x), float(y)] for x, y in interior.coords])
    return rings


def _geometry_to_geojson(geom) -> dict[str, object] | None:
    if geom is None or geom.is_empty:
        return None
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return {"type": "Polygon", "coordinates": _polygon_coords(geom)}
    if geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [_polygon_coords(part) for part in geom.geoms if not part.is_empty],
        }
    if geom_type == "GeometryCollection":
        polygons = [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty]
        if not polygons:
            return None
        coordinates: list[list[list[list[float]]]] = []
        for part in polygons:
            if part.geom_type == "Polygon":
                coordinates.append(_polygon_coords(part))
            else:
                coordinates.extend(_polygon_coords(sub) for sub in part.geoms if not sub.is_empty)
        if len(coordinates) == 1:
            return {"type": "Polygon", "coordinates": coordinates[0]}
        return {"type": "MultiPolygon", "coordinates": coordinates}
    return None


def _load_boundary(path: Path):
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("shapely is required for grid clipping. Install shapely or pass --no-clip-to-boundary.") from exc

    if not path.exists():
        raise FileNotFoundError(f"Country boundary not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        geometries = [shape(feature["geometry"]) for feature in payload.get("features", []) if feature.get("geometry")]
    elif payload.get("type") == "Feature":
        geometries = [shape(payload["geometry"])] if payload.get("geometry") else []
    else:
        geometries = [shape(payload)]

    if not geometries:
        raise RuntimeError(f"Country boundary file is empty: {path}")
    return unary_union(geometries)


def build_geojson(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    features: list[dict[str, object]] = []
    half_step = args.grid_size / 2.0

    boundary = None
    if args.clip_to_boundary:
        try:
            boundary = _load_boundary(args.country_boundary)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"WARN: clipping disabled - {exc}")

    if boundary is not None:
        from shapely.geometry import box

    for row in rows:
        lat0 = float(row["grid_lat"])
        lon0 = float(row["grid_lon"])
        lat1 = lat0 + args.grid_size
        lon1 = lon0 + args.grid_size
        center_lat = lat0 + half_step
        center_lon = lon0 + half_step

        if boundary is not None:
            cell = box(lon0, lat0, lon1, lat1)
            clipped = cell.intersection(boundary)
            if clipped.is_empty:
                continue
            geometry = _geometry_to_geojson(clipped)
            if geometry is None:
                continue
        else:
            geometry = {
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
            }

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
                    "mean_ndvi_30days": none_or_float(row["mean_ndvi_30days"]),
                },
                "geometry": geometry,
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


def count_risk_levels_by_date(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        row_date = row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"])
        level = str(row["risk_level"])
        counts.setdefault(row_date, {})
        counts[row_date][level] = counts[row_date].get(level, 0) + 1
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

    parser = argparse.ArgumentParser(description="Predict multi-day wildfire risk from Open-Meteo forecast data.")
    parser.add_argument("--spark-app-name", default="Wildfire Risk Inference")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--model-prefix", default=os.getenv("MODEL_OUTPUT_PREFIX"))
    parser.add_argument("--disable-mlflow-registry", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--mlflow-registry-uri", default=os.getenv("MLFLOW_REGISTRY_URI"))
    parser.add_argument("--registry-model-uri", default=os.getenv("MLFLOW_MODEL_URI"))
    parser.add_argument("--metrics-input", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--predictions-prefix", default=os.getenv("PREDICTIONS_PREFIX", "predictions/fire_risk_forecast"))
    parser.add_argument("--target-date", default=os.getenv("INFERENCE_TARGET_DATE"))
    parser.add_argument(
        "--forecast-horizon-days",
        type=int,
        default=int(os.getenv("FORECAST_HORIZON_DAYS", str(MAX_FORECAST_HORIZON_DAYS))),
        help=f"Number of consecutive forecast dates to score, capped at {MAX_FORECAST_HORIZON_DAYS}.",
    )
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.25")))
    parser.add_argument(
        "--country-boundary",
        type=Path,
        default=Path(os.getenv("COUNTRY_BOUNDARY", "geo/vietnam_boundary.geojson")),
        help="GeoJSON Polygon/MultiPolygon used to clip risk grid cells to the country footprint.",
    )
    parser.add_argument(
        "--no-clip-to-boundary",
        dest="clip_to_boundary",
        action="store_false",
        help="Skip clipping risk grid cells to the country boundary polygon.",
    )
    parser.set_defaults(clip_to_boundary=True)
    parser.add_argument("--forecast-timezone", default=os.getenv("FORECAST_TIMEZONE", "Asia/Ho_Chi_Minh"))
    parser.add_argument("--spark-timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))
    parser.add_argument("--open-meteo-url", default=os.getenv("OPEN_METEO_FORECAST_URL", DEFAULT_OPEN_METEO_URL))
    parser.add_argument("--past-days-for-rolling", type=int, default=int(os.getenv("FORECAST_PAST_DAYS", "30")))
    parser.add_argument("--forecast-days", type=int, default=int(os.getenv("FORECAST_DAYS", "2")))
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--medium-threshold", type=float)
    parser.add_argument("--high-threshold", type=float)
    parser.add_argument("--geojson-output", type=Path, default=Path("reports/fire_risk_forecast_latest.geojson"))
    parser.add_argument("--metadata-output", type=Path, default=Path("reports/fire_risk_forecast_latest.json"))
    parser.add_argument("--print-progress", action="store_true")

    args = parser.parse_args()
    if args.target_date is None:
        args.target_date = default_target_date(args.forecast_timezone)
    date.fromisoformat(args.target_date)
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if args.past_days_for_rolling < 6:
        parser.error("--past-days-for-rolling must be at least 6")
    if not 1 <= args.forecast_horizon_days <= MAX_FORECAST_HORIZON_DAYS:
        parser.error(f"--forecast-horizon-days must be between 1 and {MAX_FORECAST_HORIZON_DAYS}")
    if args.forecast_days < 1:
        parser.error("--forecast-days must be positive")
    args.forecast_days = max(args.forecast_days, required_forecast_days(args))
    return args


def main() -> int:
    args = parse_args()
    features_input = s3a_path(args.minio_bucket, args.features_prefix)
    metrics = load_json_if_exists(args.metrics_input)
    model_input = resolve_model_input(args, metrics)
    model_name = infer_model_name(args, metrics, model_input)
    apply_threshold_defaults(args, metrics, model_name)
    if not 0 <= args.medium_threshold <= args.high_threshold <= 1:
        raise ValueError("--medium-threshold and --high-threshold must satisfy 0 <= medium <= high <= 1")
    geojson_destination = s3a_path(args.minio_bucket, args.predictions_prefix).rstrip("/") + "/latest.geojson"
    metadata_destination = s3a_path(args.minio_bucket, args.predictions_prefix).rstrip("/") + "/latest.json"
    forecast_dates = [target.isoformat() for target in forecast_target_dates(args)]
    forecast_end = forecast_dates[-1]

    spark = build_spark(args)
    try:
        features = spark.read.parquet(features_input)
        grid_rows = load_grid(features)
        forecast_rows = fetch_forecast_rows(args, grid_rows)
        scoring = build_scoring_frame(spark, forecast_rows, args.target_date, forecast_end)
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
                "mean_ndvi_30days",
            )
            .orderBy("date", "grid_lat", "grid_lon")
            .collect()
        )
        output_dicts = [row.asDict(recursive=True) for row in output_rows]
        unique_grid_count = len({row["grid_id"] for row in output_dicts})

        geojson = build_geojson(output_dicts, args)
        metadata = {
            "status": "ok",
            "target_date": args.target_date,
            "forecast_start_date": forecast_dates[0],
            "forecast_end_date": forecast_end,
            "forecast_horizon_days": args.forecast_horizon_days,
            "forecast_dates": forecast_dates,
            "grid_count": unique_grid_count,
            "prediction_count": len(output_dicts),
            "model_input": model_input,
            "model_name": model_name,
            "model_source": "mlflow_registry" if model_input.startswith("models:/") else "spark_path",
            "features_input": features_input,
            "open_meteo_url": args.open_meteo_url,
            "daily_variables": DAILY_VARIABLES,
            "medium_threshold": args.medium_threshold,
            "high_threshold": args.high_threshold,
            "risk_level_counts": count_risk_levels(output_dicts),
            "risk_level_counts_by_date": count_risk_levels_by_date(output_dicts),
        }

        write_json(args.geojson_output, geojson)
        write_json(args.metadata_output, metadata)
        copy_local_to_s3a(spark, args.geojson_output, geojson_destination)
        copy_local_to_s3a(spark, args.metadata_output, metadata_destination)

        print(f"Forecast dates: {forecast_dates[0]} to {forecast_end} ({args.forecast_horizon_days} day(s))")
        print(f"Predicted grid cells: {unique_grid_count:,}")
        print(f"Prediction rows: {len(output_dicts):,}")
        print(f"Saved GeoJSON to {args.geojson_output}")
        print(f"Uploaded GeoJSON to {geojson_destination}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
