#!/usr/bin/env python3
"""
Cluster latest 24h FIRMS fire points with DBSCAN and write fire-area GeoJSON.

Input:
    s3://wildfire-data/firms_clean/

Output:
    reports/dbscan_fire_clusters_latest.geojson
    reports/dbscan_fire_clusters_latest.json
    s3://wildfire-data/models/dbscan_fire_clusters/latest.geojson
    s3://wildfire-data/models/dbscan_fire_clusters/latest.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from botocore.client import Config
from sklearn.cluster import DBSCAN


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


def load_fire_points(args: argparse.Namespace) -> pd.DataFrame:
    dataset = ds.dataset(
        f"{args.minio_bucket}/{args.firms_prefix.strip('/')}",
        filesystem=s3_filesystem(args),
        format="parquet",
    )
    table = dataset.to_table(
        columns=["latitude", "longitude", "acq_date", "acq_time", "confidence_score", "frp"]
    )
    frame = table.to_pandas()
    frame["acq_date"] = pd.to_datetime(frame["acq_date"])
    frame["acq_time"] = pd.to_numeric(frame["acq_time"], errors="coerce").fillna(0).astype(int)
    hours = frame["acq_time"] // 100
    minutes = frame["acq_time"] % 100
    frame["acq_datetime"] = frame["acq_date"] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
    return frame


def monotonic_chain(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[float, float],
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def polygon_from_points(points: list[tuple[float, float]], fallback_radius: float) -> list[list[float]]:
    hull = monotonic_chain(points)
    if len(hull) >= 3:
        ring = [[lon, lat] for lon, lat in hull]
        ring.append(ring[0])
        return ring

    lons = [lon for lon, _ in points]
    lats = [lat for _, lat in points]
    min_lon = min(lons) - fallback_radius
    max_lon = max(lons) + fallback_radius
    min_lat = min(lats) - fallback_radius
    max_lat = max(lats) + fallback_radius
    return [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]


def build_geojson(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    if frame.empty:
        empty = {"type": "FeatureCollection", "features": []}
        metadata = {"status": "empty", "point_count": 0, "cluster_count": 0}
        return empty, metadata

    latest = frame["acq_datetime"].max()
    cutoff = latest - timedelta(hours=args.window_hours)
    window_frame = frame[frame["acq_datetime"] >= cutoff].copy()

    if window_frame.empty:
        empty = {"type": "FeatureCollection", "features": []}
        metadata = {
            "status": "empty",
            "latest_timestamp": latest.isoformat(),
            "cutoff_timestamp": cutoff.isoformat(),
            "point_count": 0,
            "cluster_count": 0,
        }
        return empty, metadata

    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(
        window_frame[["latitude", "longitude"]]
    )
    window_frame["cluster_id"] = labels
    clustered = window_frame[window_frame["cluster_id"] >= 0].copy()

    features: list[dict[str, object]] = []
    for cluster_id, cluster in clustered.groupby("cluster_id"):
        points = list(zip(cluster["longitude"], cluster["latitude"]))
        ring = polygon_from_points(points, args.eps / 2.0)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "cluster_id": int(cluster_id),
                    "point_count": int(len(cluster)),
                    "mean_confidence_score": float(cluster["confidence_score"].mean()),
                    "max_frp": float(cluster["frp"].max()) if cluster["frp"].notna().any() else None,
                    "start_time": cluster["acq_datetime"].min().isoformat(),
                    "end_time": cluster["acq_datetime"].max().isoformat(),
                    "eps_degrees": args.eps,
                    "min_samples": args.min_samples,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    metadata = {
        "status": "ok",
        "latest_timestamp": latest.isoformat(),
        "cutoff_timestamp": cutoff.isoformat(),
        "window_hours": args.window_hours,
        "point_count": int(len(window_frame)),
        "clustered_point_count": int(len(clustered)),
        "noise_point_count": int((window_frame["cluster_id"] == -1).sum()),
        "cluster_count": len(features),
        "eps_degrees": args.eps,
        "min_samples": args.min_samples,
    }
    return geojson, metadata


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Cluster latest FIRMS points with DBSCAN.")
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--firms-prefix", default=os.getenv("FIRES_CLEAN_PREFIX", "firms_clean"))
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--geojson-output", type=Path, default=Path("reports/dbscan_fire_clusters_latest.geojson"))
    parser.add_argument("--metadata-output", type=Path, default=Path("reports/dbscan_fire_clusters_latest.json"))
    parser.add_argument("--minio-output-prefix", default="models/dbscan_fire_clusters")
    args = parser.parse_args()
    if args.eps <= 0:
        parser.error("--eps must be positive")
    if args.min_samples <= 0:
        parser.error("--min-samples must be positive")
    if args.window_hours <= 0:
        parser.error("--window-hours must be positive")
    return args


def main() -> int:
    args = parse_args()
    frame = load_fire_points(args)
    geojson, metadata = build_geojson(frame, args)

    prepare_output_path(args.geojson_output)
    prepare_output_path(args.metadata_output)
    args.geojson_output.write_text(json.dumps(geojson, indent=2) + "\n", encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    upload_file(args, args.geojson_output, f"{args.minio_output_prefix.strip('/')}/latest.geojson")
    upload_file(args, args.metadata_output, f"{args.minio_output_prefix.strip('/')}/latest.json")

    print(f"Latest timestamp: {metadata.get('latest_timestamp')}")
    print(f"Window points: {metadata['point_count']}")
    print(f"Clusters: {metadata['cluster_count']}")
    print(f"Saved GeoJSON to {args.geojson_output}")
    print(f"Uploaded GeoJSON to s3://{args.minio_bucket}/{args.minio_output_prefix.strip('/')}/latest.geojson")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
