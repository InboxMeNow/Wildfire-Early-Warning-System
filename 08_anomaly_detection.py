#!/usr/bin/env python3
"""
Detect anomalous daily grid fire counts with a simple historical z-score rule.

Rule:
    count_today > historical_mean + 3 * historical_std

Input:
    s3://wildfire-data/features/

Output:
    reports/fire_anomalies_latest.geojson
    reports/fire_anomaly_detector_stats.csv
    s3://wildfire-data/models/fire_anomaly_detector/latest.geojson
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from botocore.client import Config


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
        return endpoint[len("https://") :], "https"
    if endpoint.startswith("http://"):
        return endpoint[len("http://") :], "http"
    return endpoint, "http"


def prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def s3_filesystem(args: argparse.Namespace) -> pafs.S3FileSystem:
    endpoint, scheme = normalize_endpoint(args.minio_endpoint)
    return pafs.S3FileSystem(
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        endpoint_override=endpoint,
        scheme=scheme,
        region="us-east-1",
    )


def s3_client(args: argparse.Namespace):
    return boto3.client(
        "s3",
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=args.minio_access_key,
        aws_secret_access_key=args.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_file(args: argparse.Namespace, path: Path, key: str) -> None:
    s3_client(args).upload_file(str(path), args.minio_bucket, key)


def load_features(args: argparse.Namespace) -> pd.DataFrame:
    dataset = ds.dataset(
        f"{args.minio_bucket}/{args.features_prefix.strip('/')}",
        filesystem=s3_filesystem(args),
        format="parquet",
    )
    table = dataset.to_table(columns=["grid_id", "grid_lat", "grid_lon", "date", "fire_count"])
    frame = table.to_pandas()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["fire_count"] = pd.to_numeric(frame["fire_count"], errors="coerce").fillna(0)
    return frame


def detect_anomalies(frame: pd.DataFrame, z_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    latest_date = frame["date"].max()
    history = frame[frame["date"] < latest_date].copy()
    today = frame[frame["date"] == latest_date].copy()

    stats = (
        history.groupby("grid_id", as_index=False)["fire_count"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "historical_mean", "std": "historical_std", "count": "history_days"})
    )
    stats["historical_std"] = stats["historical_std"].fillna(0.0)
    scored = today.merge(stats, on="grid_id", how="left")
    scored["historical_mean"] = scored["historical_mean"].fillna(0.0)
    scored["historical_std"] = scored["historical_std"].fillna(0.0)
    scored["history_days"] = scored["history_days"].fillna(0).astype(int)
    scored["threshold"] = scored["historical_mean"] + z_threshold * scored["historical_std"]
    scored["z_score"] = 0.0
    nonzero_std = scored["historical_std"] > 0
    scored.loc[nonzero_std, "z_score"] = (
        scored.loc[nonzero_std, "fire_count"] - scored.loc[nonzero_std, "historical_mean"]
    ) / scored.loc[nonzero_std, "historical_std"]
    scored["is_anomaly"] = (scored["fire_count"] > scored["threshold"]) & (scored["fire_count"] > 0)
    anomalies = scored[scored["is_anomaly"]].copy()

    metadata = {
        "latest_date": latest_date.date().isoformat(),
        "grid_count": int(len(today)),
        "anomaly_count": int(len(anomalies)),
        "z_threshold": z_threshold,
        "rule": "count_today > historical_mean + z_threshold * historical_std",
    }
    return scored, anomalies, metadata


def anomalies_to_geojson(anomalies: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    features: list[dict[str, object]] = []
    for row in anomalies.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "grid_id": row.grid_id,
                    "date": row.date.date().isoformat(),
                    "fire_count": int(row.fire_count),
                    "historical_mean": float(row.historical_mean),
                    "historical_std": float(row.historical_std),
                    "threshold": float(row.threshold),
                    "z_score": float(row.z_score),
                    "history_days": int(row.history_days),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row.grid_lon), float(row.grid_lat)],
                },
            }
        )
    return {"type": "FeatureCollection", "metadata": metadata, "features": features}


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Detect anomalous fire-count grid days.")
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--geojson-output", type=Path, default=Path("reports/fire_anomalies_latest.geojson"))
    parser.add_argument("--stats-output", type=Path, default=Path("reports/fire_anomaly_detector_stats.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("reports/fire_anomaly_detector_latest.json"))
    parser.add_argument("--minio-output-prefix", default="models/fire_anomaly_detector")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = load_features(args)
    scored, anomalies, metadata = detect_anomalies(frame, args.z_threshold)
    geojson = anomalies_to_geojson(anomalies, metadata)

    prepare_output_path(args.geojson_output)
    prepare_output_path(args.metadata_output)
    prepare_output_path(args.stats_output)
    args.geojson_output.write_text(json.dumps(geojson, indent=2) + "\n", encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scored.to_csv(args.stats_output, index=False)

    upload_file(args, args.geojson_output, f"{args.minio_output_prefix.strip('/')}/latest.geojson")
    upload_file(args, args.metadata_output, f"{args.minio_output_prefix.strip('/')}/latest.json")
    upload_file(args, args.stats_output, f"{args.minio_output_prefix.strip('/')}/detector_stats.csv")

    print(f"Latest date: {metadata['latest_date']}")
    print(f"Anomalies: {metadata['anomaly_count']} / {metadata['grid_count']} grids")
    print(f"Saved anomalies to {args.geojson_output}")
    print(f"Uploaded anomalies to s3://{args.minio_bucket}/{args.minio_output_prefix.strip('/')}/latest.geojson")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
