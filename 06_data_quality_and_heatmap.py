#!/usr/bin/env python3
"""
Generate a week-1 data quality report and a Vietnam fire heatmap.

Input defaults:
    s3://wildfire-data/firms_clean/
    s3://wildfire-data/weather_clean/
    s3://wildfire-data/features/

Output defaults:
    reports/data_quality_week1.md
    maps/fires_heatmap_2020_2024.html
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import boto3
import folium
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from botocore.client import Config
from folium.plugins import HeatMap


DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"


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


def normalize_endpoint(endpoint: str) -> tuple[str, str]:
    if endpoint.startswith("https://"):
        return endpoint.removeprefix("https://"), "https"
    if endpoint.startswith("http://"):
        return endpoint.removeprefix("http://"), "http"
    return endpoint, "http"


def build_s3_filesystem(args: argparse.Namespace) -> pafs.S3FileSystem:
    endpoint, scheme = normalize_endpoint(args.minio_endpoint)
    return pafs.S3FileSystem(
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        endpoint_override=endpoint,
        scheme=scheme,
        region="us-east-1",
    )


def build_boto_client(args: argparse.Namespace):
    return boto3.client(
        "s3",
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=args.minio_access_key,
        aws_secret_access_key=args.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def dataset_path(bucket: str, prefix: str) -> str:
    return f"{bucket}/{prefix.strip('/')}"


def read_dataset(
    filesystem: pafs.S3FileSystem,
    bucket: str,
    prefix: str,
) -> ds.Dataset:
    return ds.dataset(dataset_path(bucket, prefix), filesystem=filesystem, format="parquet")


def list_parquet_objects(args: argparse.Namespace, prefix: str) -> tuple[int, int]:
    client = build_boto_client(args)
    parquet_count = 0
    parquet_bytes = 0
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=args.minio_bucket, Prefix=f"{prefix.strip('/')}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_count += 1
                parquet_bytes += obj["Size"]

    return parquet_count, parquet_bytes


def table_profile(dataset: ds.Dataset, columns: list[str]) -> dict[str, Any]:
    table = dataset.to_table(columns=[column for column in columns if column in dataset.schema.names])
    profile: dict[str, Any] = {"rows": table.num_rows, "columns": len(dataset.schema.names)}

    for column in table.column_names:
        array = table[column]
        profile[f"{column}_nulls"] = array.null_count
        if table.num_rows and column not in {"grid_id"}:
            try:
                profile[f"{column}_min"] = pc.min(array).as_py()
                profile[f"{column}_max"] = pc.max(array).as_py()
            except Exception:
                pass

    return profile


def value_counts(table: pd.DataFrame, column: str) -> dict[int, int]:
    counts = table[column].value_counts(dropna=False).sort_index()
    return {int(key): int(value) for key, value in counts.items()}


def profile_features(features_dataset: ds.Dataset) -> tuple[dict[str, Any], list[str]]:
    columns = [
        "grid_id",
        "date",
        "fire_count",
        "fire_occurred",
        "risk_level",
        "precipitation_sum",
        "precipitation_sum_7days",
        "dry_days_count",
    ]
    table = features_dataset.to_table(columns=columns)
    frame = table.to_pandas()

    profile: dict[str, Any] = {
        "rows": len(frame),
        "columns": len(features_dataset.schema.names),
        "grid_cells": int(frame["grid_id"].nunique()),
        "date_min": frame["date"].min(),
        "date_max": frame["date"].max(),
        "duplicate_grid_dates": int(frame.duplicated(["grid_id", "date"]).sum()),
        "risk_counts": value_counts(frame, "risk_level"),
        "fire_occurred_counts": value_counts(frame, "fire_occurred"),
    }

    expected_fire_occurred = (frame["fire_count"] > 0).astype("int32")
    expected_risk = pd.Series(0, index=frame.index, dtype="int32")
    expected_risk[(frame["fire_count"] >= 1) & (frame["fire_count"] <= 5)] = 1
    expected_risk[frame["fire_count"] > 5] = 2

    profile["bad_fire_occurred_labels"] = int((frame["fire_occurred"] != expected_fire_occurred).sum())
    profile["bad_risk_level_labels"] = int((frame["risk_level"] != expected_risk).sum())
    profile["negative_fire_count_rows"] = int((frame["fire_count"] < 0).sum())
    profile["negative_rolling_precip_rows"] = int((frame["precipitation_sum_7days"] < 0).sum())
    profile["negative_dry_days_rows"] = int((frame["dry_days_count"] < 0).sum())

    issues: list[str] = []
    if profile["duplicate_grid_dates"]:
        issues.append("features has duplicate (grid_id, date) rows")
    if profile["bad_fire_occurred_labels"]:
        issues.append("fire_occurred does not match fire_count > 0")
    if profile["bad_risk_level_labels"]:
        issues.append("risk_level does not match fire_count thresholds")
    if profile["negative_fire_count_rows"]:
        issues.append("features has negative fire_count values")
    if profile["negative_rolling_precip_rows"]:
        issues.append("features has negative precipitation_sum_7days values")
    if profile["negative_dry_days_rows"]:
        issues.append("features has negative dry_days_count values")

    return profile, issues


def build_heatmap(
    firms_dataset: ds.Dataset,
    grid_size: float,
    output_path: Path,
) -> pd.DataFrame:
    table = firms_dataset.to_table(columns=["latitude", "longitude", "acq_date"])
    frame = table.to_pandas()
    frame = frame[(frame["acq_date"] >= date(2020, 1, 1)) & (frame["acq_date"] <= date(2024, 12, 31))]

    frame["grid_lat"] = (frame["latitude"] / grid_size).map(math.floor) * grid_size
    frame["grid_lon"] = (frame["longitude"] / grid_size).map(math.floor) * grid_size

    heat = (
        frame.groupby(["grid_lat", "grid_lon"], as_index=False)
        .size()
        .rename(columns={"size": "fire_count"})
    )
    heat["center_lat"] = heat["grid_lat"] + grid_size / 2
    heat["center_lon"] = heat["grid_lon"] + grid_size / 2

    fire_map = folium.Map(location=[16.0, 106.0], zoom_start=5, tiles="CartoDB positron")
    HeatMap(
        heat[["center_lat", "center_lon", "fire_count"]].values.tolist(),
        name="Fire detections 2020-2024",
        radius=18,
        blur=22,
        min_opacity=0.25,
        max_zoom=8,
    ).add_to(fire_map)

    for row in heat.nlargest(100, "fire_count").itertuples(index=False):
        folium.CircleMarker(
            location=[row.center_lat, row.center_lon],
            radius=4,
            weight=1,
            color="#7f1d1d",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.65,
            tooltip=f"{int(row.fire_count):,} fires",
        ).add_to(fire_map)

    folium.LayerControl().add_to(fire_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fire_map.save(output_path)

    return heat


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.1f} {unit}"


def write_report(
    args: argparse.Namespace,
    output_path: Path,
    object_profiles: dict[str, tuple[int, int]],
    firms_profile: dict[str, Any],
    weather_profile: dict[str, Any],
    features_profile: dict[str, Any],
    heat: pd.DataFrame,
    issues: list[str],
) -> None:
    status = "PASS" if not issues else "WARN"
    lines = [
        "# Week 1 Data Quality Report",
        "",
        f"Status: {status}",
        "",
        "## MinIO Objects",
        "",
        "| Dataset | Parquet parts | Size |",
        "|---|---:|---:|",
    ]

    for name, (parts, bytes_count) in object_profiles.items():
        lines.append(f"| {name} | {parts:,} | {format_bytes(bytes_count)} |")

    lines.extend(
        [
            "",
            "## Dataset Shape",
            "",
            "| Dataset | Rows | Columns | Date range |",
            "|---|---:|---:|---|",
            f"| firms_clean | {firms_profile['rows']:,} | {firms_profile['columns']:,} | {firms_profile.get('acq_date_min')} to {firms_profile.get('acq_date_max')} |",
            f"| weather_clean | {weather_profile['rows']:,} | {weather_profile['columns']:,} | {weather_profile.get('date_min')} to {weather_profile.get('date_max')} |",
            f"| features | {features_profile['rows']:,} | {features_profile['columns']:,} | {features_profile['date_min']} to {features_profile['date_max']} |",
            "",
            "## Feature Integrity",
            "",
            f"- Grid cells: {features_profile['grid_cells']:,}",
            f"- Duplicate `(grid_id, date)` rows: {features_profile['duplicate_grid_dates']:,}",
            f"- Bad `fire_occurred` labels: {features_profile['bad_fire_occurred_labels']:,}",
            f"- Bad `risk_level` labels: {features_profile['bad_risk_level_labels']:,}",
            f"- `risk_level` counts: {features_profile['risk_counts']}",
            f"- `fire_occurred` counts: {features_profile['fire_occurred_counts']}",
            "",
            "## Heatmap",
            "",
            f"- Heatmap grid cells with fires: {len(heat):,}",
            f"- Max fires in one 0.5 degree cell: {int(heat['fire_count'].max()):,}",
            f"- Output: `{args.heatmap_output}`",
            "",
            "## Issues",
            "",
        ]
    )

    if issues:
        lines.extend([f"- {issue}" for issue in issues])
    else:
        lines.append("- No blocking data quality issues found in week-1 outputs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Generate week-1 data quality report and heatmap.")
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--firms-prefix", default=os.getenv("FIRES_CLEAN_PREFIX", "firms_clean"))
    parser.add_argument("--weather-prefix", default=os.getenv("WEATHER_CLEAN_PREFIX", "weather_clean"))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", "0.5")))
    parser.add_argument("--report-output", type=Path, default=Path("reports/data_quality_week1.md"))
    parser.add_argument("--heatmap-output", type=Path, default=Path("maps/fires_heatmap_2020_2024.html"))

    args = parser.parse_args()
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    filesystem = build_s3_filesystem(args)

    firms = read_dataset(filesystem, args.minio_bucket, args.firms_prefix)
    weather = read_dataset(filesystem, args.minio_bucket, args.weather_prefix)
    features = read_dataset(filesystem, args.minio_bucket, args.features_prefix)

    firms_profile = table_profile(firms, ["acq_date", "latitude", "longitude", "confidence_score"])
    weather_profile = table_profile(weather, ["date", "latitude", "longitude", "precipitation_sum"])
    features_profile, issues = profile_features(features)

    heat = build_heatmap(firms, args.grid_size, args.heatmap_output)

    object_profiles = {
        args.firms_prefix: list_parquet_objects(args, args.firms_prefix),
        args.weather_prefix: list_parquet_objects(args, args.weather_prefix),
        args.features_prefix: list_parquet_objects(args, args.features_prefix),
    }

    write_report(
        args=args,
        output_path=args.report_output,
        object_profiles=object_profiles,
        firms_profile=firms_profile,
        weather_profile=weather_profile,
        features_profile=features_profile,
        heat=heat,
        issues=issues,
    )

    print(f"Wrote data quality report to {args.report_output}")
    print(f"Wrote heatmap to {args.heatmap_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
