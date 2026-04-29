#!/usr/bin/env python3
"""
Fetch Meteostat hourly weather, aggregate to daily values for a 0.5 degree
Vietnam grid, and save Parquet to MinIO.

Default output:
    s3://wildfire-data/weather/meteostat_daily_vietnam_2020_2024.parquet

Run:
    python 02_fetch_weather.py

MinIO defaults match docker-compose.yml:
    MINIO_ENDPOINT=http://localhost:9000
    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin
    MINIO_BUCKET=wildfire-data
    MINIO_PREFIX=weather
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config
from botocore.exceptions import ClientError
from meteostat import config, hourly, stations


config.block_large_requests = False


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Vietnam bounding box, matching the FIRMS Vietnam area used in 01_explore_firms.py.
DEFAULT_LAT_MIN = 5.0
DEFAULT_LAT_MAX = 25.0
DEFAULT_LON_MIN = 95.0
DEFAULT_LON_MAX = 115.0
DEFAULT_GRID_STEP = 0.5

DEFAULT_LOCAL_OUTPUT = Path("data/raw/meteostat_daily_vietnam_2020_2024.parquet")
DEFAULT_PARTS_DIR = Path("data/raw/weather_meteostat_vietnam_parts")
DEFAULT_OBJECT_NAME = "meteostat_daily_vietnam_2020_2024.parquet"
WEATHER_SCHEMA = pa.schema(
    [
        ("point_id", pa.int64()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("station_id", pa.string()),
        ("station_latitude", pa.float64()),
        ("station_longitude", pa.float64()),
        ("station_distance_km", pa.float64()),
        ("date", pa.date32()),
        ("temperature_2m_max", pa.float64()),
        ("relative_humidity_2m_min", pa.float64()),
        ("wind_speed_10m_max", pa.float64()),
        ("precipitation_sum", pa.float64()),
    ]
)


@dataclass(frozen=True)
class GridPoint:
    point_id: int
    latitude: float
    longitude: float


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


def inclusive_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--grid-step must be positive")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 6) for i in range(count)]


def build_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float,
) -> list[GridPoint]:
    latitudes = inclusive_range(lat_min, lat_max, step)
    longitudes = inclusive_range(lon_min, lon_max, step)
    points: list[GridPoint] = []

    for latitude in latitudes:
        for longitude in longitudes:
            points.append(
                GridPoint(
                    point_id=len(points),
                    latitude=latitude,
                    longitude=longitude,
                )
            )

    return points


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    radius_km = 6371.0088
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = lat2.map(math.radians)
    lon2_rad = lon2.map(math.radians)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (dlat / 2).map(math.sin) ** 2 + math.cos(lat1_rad) * lat2_rad.map(math.cos) * (
        (dlon / 2).map(math.sin) ** 2
    )
    return 2 * radius_km * a.map(math.sqrt).map(math.asin)


def load_weather_stations(country: str) -> pd.DataFrame:
    station_frame = stations.query(
        "select id, country, region, latitude, longitude, elevation, timezone "
        "from stations where country = ?",
        params=(country,),
    )
    if station_frame.empty:
        raise RuntimeError(f"Meteostat returned no stations for country={country!r}")
    return station_frame.reset_index(drop=True)


def assign_nearest_stations(
    grid: list[GridPoint],
    station_frame: pd.DataFrame,
    max_station_distance_km: float | None,
) -> pd.DataFrame:
    assignments: list[dict[str, object]] = []

    for point in grid:
        distances = haversine_km(
            point.latitude,
            point.longitude,
            station_frame["latitude"],
            station_frame["longitude"],
        )
        nearest_index = distances.idxmin()
        nearest = station_frame.loc[nearest_index]
        distance_km = float(distances.loc[nearest_index])

        if max_station_distance_km is not None and distance_km > max_station_distance_km:
            continue

        assignments.append(
            {
                "point_id": point.point_id,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "station_id": nearest["id"],
                "station_latitude": float(nearest["latitude"]),
                "station_longitude": float(nearest["longitude"]),
                "station_distance_km": round(distance_km, 3),
            }
        )

    if not assignments:
        raise RuntimeError("No grid points were assigned to a Meteostat station")

    return pd.DataFrame(assignments)


def fetch_station_daily(
    station_id: str,
    start_date: str,
    end_date: str,
    timezone: str | None,
    yearly_requests: bool,
) -> pd.DataFrame:
    start = datetime.combine(date.fromisoformat(start_date), datetime_time.min)
    end = datetime.combine(date.fromisoformat(end_date), datetime_time.max)
    all_dates = pd.DataFrame(
        {"date": pd.date_range(start=start_date, end=end_date, freq="D").date}
    )

    if yearly_requests:
        hourly_frames = []
        for year in range(start.year, end.year + 1):
            chunk_start = max(start, datetime(year, 1, 1))
            chunk_end = min(end, datetime(year, 12, 31, 23, 59, 59, 999999))
            chunk_frame = hourly(station_id, chunk_start, chunk_end, timezone=timezone).fetch()
            if chunk_frame is not None and not chunk_frame.empty:
                hourly_frames.append(chunk_frame)
        hourly_frame = (
            pd.concat(hourly_frames).sort_index()
            if hourly_frames
            else None
        )
    else:
        hourly_frame = hourly(station_id, start, end, timezone=timezone).fetch()

    if hourly_frame is None or hourly_frame.empty:
        return all_dates.assign(
            temperature_2m_max=pd.NA,
            relative_humidity_2m_min=pd.NA,
            wind_speed_10m_max=pd.NA,
            precipitation_sum=pd.NA,
        )

    frame = hourly_frame.reset_index()
    frame["date"] = pd.to_datetime(frame["time"]).dt.date

    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            temperature_2m_max=("temp", "max"),
            relative_humidity_2m_min=("rhum", "min"),
            wind_speed_10m_max=("wspd", "max"),
            precipitation_sum=("prcp", lambda values: values.sum(min_count=1)),
        )
    )

    return all_dates.merge(daily, on="date", how="left")


def expand_station_daily_to_grid(
    assignment_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
) -> pd.DataFrame:
    points = assignment_frame.copy()
    points["_join_key"] = 1
    daily = daily_frame.copy()
    daily["_join_key"] = 1

    result = points.merge(daily, on="_join_key", how="inner").drop(columns="_join_key")
    return result[
        [
            "point_id",
            "latitude",
            "longitude",
            "station_id",
            "station_latitude",
            "station_longitude",
            "station_distance_km",
            "date",
            "temperature_2m_max",
            "relative_humidity_2m_min",
            "wind_speed_10m_max",
            "precipitation_sum",
        ]
    ]


def normalize_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["point_id"] = normalized["point_id"].astype("int64")
    normalized["latitude"] = normalized["latitude"].astype("float64")
    normalized["longitude"] = normalized["longitude"].astype("float64")
    normalized["station_id"] = normalized["station_id"].astype("string")
    normalized["station_latitude"] = normalized["station_latitude"].astype("float64")
    normalized["station_longitude"] = normalized["station_longitude"].astype("float64")
    normalized["station_distance_km"] = normalized["station_distance_km"].astype("float64")
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date

    for column in [
        "temperature_2m_max",
        "relative_humidity_2m_min",
        "wind_speed_10m_max",
        "precipitation_sum",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    return normalized


def safe_part_name(station_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", station_id)


def part_path(parts_dir: Path, station_id: str) -> Path:
    return parts_dir / f"station-{safe_part_name(station_id)}.parquet"


def write_weather_parquet(path: Path, args: argparse.Namespace) -> int:
    grid = build_grid(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        step=args.grid_step,
    )
    station_frame = load_weather_stations(args.station_country)
    assignments = assign_nearest_stations(
        grid,
        station_frame,
        args.max_station_distance_km,
    )

    unique_stations = sorted(assignments["station_id"].unique())
    print(
        f"Fetching Meteostat data for {len(grid):,} grid points, "
        f"{len(assignments):,} assigned points, {len(unique_stations):,} stations "
        f"for {args.start_date}..{args.end_date}"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    args.parts_dir.mkdir(parents=True, exist_ok=True)

    for index, station_id in enumerate(unique_stations, start=1):
        current_part_path = part_path(args.parts_dir, station_id)
        if args.resume and current_part_path.exists():
            print(f"Skipping existing station {index:,}/{len(unique_stations):,}: {station_id}")
            continue

        station_assignments = assignments[assignments["station_id"] == station_id]
        daily = fetch_station_daily(
            station_id=station_id,
            start_date=args.start_date,
            end_date=args.end_date,
            timezone=args.timezone,
            yearly_requests=args.yearly_requests,
        )
        expanded = expand_station_daily_to_grid(station_assignments, daily)
        expanded = normalize_weather_frame(expanded)
        expanded.to_parquet(
            current_part_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        print(
            f"Fetched station {index:,}/{len(unique_stations):,}: "
            f"{station_id} -> {len(expanded):,} rows"
        )

        if index < len(unique_stations) and args.request_delay_seconds > 0:
            time.sleep(args.request_delay_seconds)

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for station_id in unique_stations:
            current_part_path = part_path(args.parts_dir, station_id)
            if not current_part_path.exists():
                raise RuntimeError(f"Missing checkpoint part: {current_part_path}")

            part_frame = pd.read_parquet(current_part_path)
            part_frame = normalize_weather_frame(part_frame)
            table = pa.Table.from_pandas(
                part_frame,
                schema=WEATHER_SCHEMA,
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    where=path,
                    schema=WEATHER_SCHEMA,
                    compression="snappy",
                )
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return row_count


def create_s3_client(args: argparse.Namespace):
    return boto3.client(
        "s3",
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=args.minio_access_key,
        aws_secret_access_key=args.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3_client, bucket: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in {404, 400}:
            raise
        s3_client.create_bucket(Bucket=bucket)


def upload_to_minio(args: argparse.Namespace, parquet_path: Path) -> str:
    object_key = "/".join(
        part.strip("/")
        for part in [args.minio_prefix, args.object_name]
        if part and part.strip("/")
    )

    s3_client = create_s3_client(args)
    ensure_bucket(s3_client, args.minio_bucket)
    s3_client.upload_file(str(parquet_path), args.minio_bucket, object_key)
    return f"s3://{args.minio_bucket}/{object_key}"


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Fetch Meteostat hourly weather for Vietnam grid, aggregate daily, and upload Parquet to MinIO."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP)
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for Meteostat hourly records before daily aggregation.",
    )
    parser.add_argument(
        "--single-request-per-station",
        action="store_false",
        dest="yearly_requests",
        help="Fetch the full date range in one Meteostat request per station instead of one request per year.",
    )
    parser.add_argument(
        "--station-country",
        default="VN",
        help="Meteostat station country code used for nearest-station assignment.",
    )
    parser.add_argument(
        "--max-station-distance-km",
        type=float,
        default=None,
        help="Optional maximum nearest-station distance; farther grid points are dropped.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between Meteostat station fetches.",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=DEFAULT_LOCAL_OUTPUT,
        help="Local Parquet cache path before/alongside MinIO upload.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=DEFAULT_PARTS_DIR,
        help="Directory for per-station checkpoint Parquet files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing station checkpoint files and refetch all stations.",
    )
    parser.add_argument(
        "--skip-local-copy",
        action="store_true",
        help="Delete local output after successful MinIO upload.",
    )
    parser.add_argument(
        "--skip-minio-upload",
        action="store_true",
        help="Only write local Parquet; do not upload to MinIO.",
    )
    parser.add_argument(
        "--minio-endpoint",
        default=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    )
    parser.add_argument(
        "--minio-access-key",
        default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    )
    parser.add_argument(
        "--minio-secret-key",
        default=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    )
    parser.add_argument(
        "--minio-bucket",
        default=os.getenv("MINIO_BUCKET", "wildfire-data"),
        help="S3/MinIO bucket name. docker-compose.yml creates wildfire-data by default.",
    )
    parser.add_argument(
        "--minio-prefix",
        default=os.getenv("MINIO_PREFIX", "weather"),
        help="Object prefix/folder inside the bucket.",
    )
    parser.add_argument(
        "--object-name",
        default=os.getenv("WEATHER_OBJECT_NAME", DEFAULT_OBJECT_NAME),
    )

    args = parser.parse_args()
    if args.grid_step <= 0:
        parser.error("--grid-step must be positive")
    if args.request_delay_seconds < 0:
        parser.error("--request-delay-seconds must be non-negative")
    if args.max_station_distance_km is not None and args.max_station_distance_km <= 0:
        parser.error("--max-station-distance-km must be positive")
    if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
        parser.error("--start-date must be on or before --end-date")

    return args


def main() -> int:
    args = parse_args()
    row_count = write_weather_parquet(args.local_output, args)

    if not args.skip_local_copy:
        print(f"Saved local Parquet: {args.local_output} ({row_count:,} rows)")

    if not args.skip_minio_upload:
        minio_uri = upload_to_minio(args, args.local_output)
        print(f"Uploaded Parquet to MinIO: {minio_uri} ({row_count:,} rows)")

    if args.skip_local_copy:
        args.local_output.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
