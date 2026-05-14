#!/usr/bin/env python3
"""
Request and process MODIS MOD13Q1 NDVI composites from NASA AppEEARS.

Outputs:
    Local Parquet: data/raw/ndvi/mod13q1_ndvi_by_grid.parquet
    MinIO: s3://wildfire-data/ndvi/mod13q1_ndvi_by_grid.parquet

The Parquet rows are 16-day composite grid aggregates:
    grid_id, grid_lat_index, grid_lon_index, grid_lat, grid_lon,
    composite_date, mean_ndvi, ndvi_pixel_count, source_file

Typical run after setting Earthdata credentials:
    python src/etl/ndvi_processing.py --mode all --earthdata-username USER --earthdata-password PASS
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import boto3
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
from botocore.client import Config
from botocore.exceptions import ClientError


APPEEARS_API = "https://appeears.earthdatacloud.nasa.gov/api"
DEFAULT_PRODUCT = "MOD13Q1.061"
DEFAULT_LAYER = "_250m_16_days_NDVI"
DEFAULT_BBOX = "95,5,115,25"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_GRID_SIZE = 0.5
DEFAULT_LOCAL_DIR = Path("data/raw/ndvi")
DEFAULT_LOCAL_OUTPUT = DEFAULT_LOCAL_DIR / "mod13q1_ndvi_by_grid.parquet"
DEFAULT_TASK_FILE = DEFAULT_LOCAL_DIR / "appeears_task.json"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_MINIO_PREFIX = "ndvi"
DEFAULT_OBJECT_NAME = "mod13q1_ndvi_by_grid.parquet"


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


def iso_to_appeears(value: str) -> str:
    parsed = date.fromisoformat(value)
    return f"{parsed:%m-%d-%Y}"


def bbox_to_feature_collection(bbox: str) -> dict[str, Any]:
    parts = [float(part.strip()) for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = parts
    coordinates = [
        [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": coordinates},
            }
        ],
        "fileName": "se_asia_bbox",
    }


def build_area_task(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_type": "area",
        "task_name": args.task_name,
        "params": {
            "dates": [
                {
                    "startDate": iso_to_appeears(args.start_date),
                    "endDate": iso_to_appeears(args.end_date),
                }
            ],
            "layers": [{"product": args.product, "layer": args.layer}],
            "geo": bbox_to_feature_collection(args.bbox),
            "output": {
                "format": {"type": "geotiff", "filename_date": "calendar"},
                "projection": args.projection,
            },
        },
    }


def appeears_login(args: argparse.Namespace, session: requests.Session) -> str:
    username = args.earthdata_username or os.getenv("EARTHDATA_USERNAME")
    password = args.earthdata_password or os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing Earthdata credentials. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD "
            "or pass --earthdata-username/--earthdata-password."
        )

    response = session.post(f"{args.appeears_api}/login", auth=(username, password), timeout=args.timeout_seconds)
    response.raise_for_status()
    token = response.json().get("token")
    if not token:
        raise RuntimeError(f"AppEEARS login response did not include a token: {response.text[:500]}")
    return str(token)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def submit_task(args: argparse.Namespace, session: requests.Session, token: str) -> str:
    task = build_area_task(args)
    response = session.post(
        f"{args.appeears_api}/task",
        json=task,
        headers=auth_headers(token),
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    task_id = payload.get("task_id")
    if not task_id:
        raise RuntimeError(f"AppEEARS submit response did not include task_id: {payload}")

    args.task_file.parent.mkdir(parents=True, exist_ok=True)
    args.task_file.write_text(json.dumps({"task_id": task_id, "request": task}, indent=2) + "\n", encoding="utf-8")
    return str(task_id)


def task_id_from_file(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_id = payload.get("task_id")
    if not task_id:
        raise RuntimeError(f"Task file does not include task_id: {path}")
    return str(task_id)


def wait_for_task(args: argparse.Namespace, session: requests.Session, token: str, task_id: str) -> None:
    deadline = time.time() + args.wait_timeout_seconds
    while True:
        response = session.get(
            f"{args.appeears_api}/status/{task_id}",
            headers=auth_headers(token),
            timeout=args.timeout_seconds,
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "done":
            return
        if status == "error":
            raise RuntimeError(f"AppEEARS task failed: {payload}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for AppEEARS task {task_id}; last status={status!r}")
        if args.print_progress:
            summary = payload.get("progress", {}).get("summary")
            suffix = f" ({summary}% complete)" if summary is not None else ""
            print(f"AppEEARS task {task_id} status: {status}{suffix}")
        time.sleep(args.poll_seconds)


def list_bundle_files(args: argparse.Namespace, session: requests.Session, token: str, task_id: str) -> list[dict[str, Any]]:
    response = session.get(
        f"{args.appeears_api}/bundle/{task_id}",
        headers=auth_headers(token),
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    files = response.json().get("files", [])
    if not isinstance(files, list):
        raise RuntimeError(f"Unexpected AppEEARS bundle response: {response.text[:500]}")
    return files


def download_bundle_files(args: argparse.Namespace, session: requests.Session, token: str, task_id: str) -> list[Path]:
    args.download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for file_info in list_bundle_files(args, session, token, task_id):
        file_name = str(file_info.get("file_name", file_info.get("file_id", "")))
        file_id = file_info.get("file_id")
        if not file_id:
            continue
        if not file_name.lower().endswith((".tif", ".tiff")):
            continue
        if args.layer not in file_name:
            continue

        target = args.download_dir / safe_filename(file_name)
        expected_size = file_info.get("file_size")
        if args.resume and target.exists():
            local_size = target.stat().st_size
            if local_size > 0 and (expected_size is None or local_size == int(expected_size)):
                downloaded.append(target)
                continue
            if args.print_progress:
                print(
                    f"Re-downloading incomplete file {target.name}: "
                    f"local={local_size:,}, expected={int(expected_size):,}"
                )

        response = session.get(
            f"{args.appeears_api}/bundle/{task_id}/{file_id}",
            headers=auth_headers(token),
            timeout=args.timeout_seconds,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
        downloaded.append(target)
        if args.print_progress:
            print(f"Downloaded {target}")

    if not downloaded:
        raise RuntimeError(f"No GeoTIFF files were downloaded for task {task_id}")
    return downloaded


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def infer_composite_date(path: Path) -> date:
    name = path.name
    calendar = re.search(r"(?<!\d)(20\d{6})T?\d*(?!\d)", name)
    if calendar:
        return date.fromisoformat(f"{calendar.group(1)[:4]}-{calendar.group(1)[4:6]}-{calendar.group(1)[6:8]}")

    doy = re.search(r"doy(20\d{2})(\d{3})", name, flags=re.IGNORECASE)
    if doy:
        return date.fromordinal(date(int(doy.group(1)), 1, 1).toordinal() + int(doy.group(2)) - 1)

    raise ValueError(f"Cannot infer MOD13Q1 composite date from filename: {path.name}")


def raster_xy_names(raster) -> tuple[str, str]:
    x_name = "x" if "x" in raster.coords else "lon"
    y_name = "y" if "y" in raster.coords else "lat"
    return x_name, y_name


def normalize_ndvi(values: np.ndarray) -> np.ndarray:
    result = values.astype("float64", copy=False)
    result = np.where(np.isfinite(result), result, np.nan)
    result = np.where((result == -3000) | (result < -3000), np.nan, result)
    finite = result[np.isfinite(result)]
    if finite.size and (np.nanmax(finite) > 2.0 or np.nanmin(finite) < -1.0):
        result = result * 0.0001
    return np.where((result >= -0.2) & (result <= 1.0), result, np.nan)


def iter_chunks(length: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def aggregate_geotiff_to_grid(path: Path, grid_size: float, chunk_rows: int) -> pd.DataFrame:
    raster = rxr.open_rasterio(path, masked=True).squeeze(drop=True)
    if raster.rio.crs is not None and raster.rio.crs.to_epsg() != 4326:
        raster = raster.rio.reproject("EPSG:4326")

    x_name, y_name = raster_xy_names(raster)
    x_values = np.asarray(raster[x_name].values, dtype="float64")
    y_values = np.asarray(raster[y_name].values, dtype="float64")
    lon_index = np.floor(x_values / grid_size).astype("int64")
    min_lon_index = int(lon_index.min())
    max_lon_index = int(lon_index.max())
    lon_bins = max_lon_index - min_lon_index + 1
    composite = infer_composite_date(path)

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    for start, end in iter_chunks(len(y_values), chunk_rows):
        block = normalize_ndvi(np.asarray(raster.isel({y_name: slice(start, end)}).values))
        valid = np.isfinite(block)
        if not valid.any():
            continue

        lat_index = np.floor(y_values[start:end] / grid_size).astype("int64")
        row_codes = (lat_index[:, None] * lon_bins) + (lon_index[None, :] - min_lon_index)
        codes = row_codes[valid]
        values = block[valid]

        unique_codes, inverse = np.unique(codes, return_inverse=True)
        block_sums = np.bincount(inverse, weights=values)
        block_counts = np.bincount(inverse)
        for code, value_sum, value_count in zip(unique_codes, block_sums, block_counts):
            code_int = int(code)
            sums[code_int] = sums.get(code_int, 0.0) + float(value_sum)
            counts[code_int] = counts.get(code_int, 0) + int(value_count)

    rows: list[dict[str, Any]] = []
    for code, value_sum in sums.items():
        lat_index = math.floor(code / lon_bins)
        lon_idx = int(code - lat_index * lon_bins + min_lon_index)
        count = counts[code]
        rows.append(
            {
                "grid_id": f"{lat_index}_{lon_idx}",
                "grid_lat_index": int(lat_index),
                "grid_lon_index": lon_idx,
                "grid_lat": round(lat_index * grid_size, 6),
                "grid_lon": round(lon_idx * grid_size, 6),
                "composite_date": composite,
                "mean_ndvi": value_sum / count if count else np.nan,
                "ndvi_pixel_count": count,
                "source_file": path.name,
            }
        )

    return pd.DataFrame(rows)


def process_geotiffs(args: argparse.Namespace) -> int:
    tif_paths = sorted(
        path
        for pattern in ("*.tif", "*.tiff")
        for path in args.download_dir.glob(pattern)
        if args.layer in path.name
    )
    if not tif_paths:
        raise RuntimeError(f"No GeoTIFF files found under {args.download_dir}")

    frames = []
    for index, path in enumerate(tif_paths, start=1):
        frame = aggregate_geotiff_to_grid(path, args.grid_size, args.chunk_rows)
        frames.append(frame)
        if args.print_progress:
            print(f"Processed {index:,}/{len(tif_paths):,}: {path.name} -> {len(frame):,} grid rows")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["grid_lat_index", "grid_lon_index", "composite_date"])
    args.local_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.local_output, index=False, engine="pyarrow", compression="snappy")
    return len(combined)


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
        if status not in {400, 404}:
            raise
        s3_client.create_bucket(Bucket=bucket)


def upload_to_minio(args: argparse.Namespace) -> str:
    object_key = "/".join(
        part.strip("/")
        for part in [args.minio_prefix, args.object_name]
        if part and part.strip("/")
    )
    s3_client = create_s3_client(args)
    ensure_bucket(s3_client, args.minio_bucket)
    s3_client.upload_file(str(args.local_output), args.minio_bucket, object_key)
    return f"s3://{args.minio_bucket}/{object_key}"


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Fetch/process MOD13Q1 NDVI and upload grid aggregates to MinIO.")
    parser.add_argument("--mode", choices=["submit", "download", "process", "all"], default="process")
    parser.add_argument("--appeears-api", default=os.getenv("APPEEARS_API", APPEEARS_API))
    parser.add_argument("--earthdata-username", default=os.getenv("EARTHDATA_USERNAME"))
    parser.add_argument("--earthdata-password", default=os.getenv("EARTHDATA_PASSWORD"))
    parser.add_argument("--task-id", default=os.getenv("APPEEARS_TASK_ID"))
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--task-name", default=os.getenv("APPEEARS_TASK_NAME", "wildfire-mod13q1-ndvi-se-asia"))
    parser.add_argument("--product", default=os.getenv("NDVI_PRODUCT", DEFAULT_PRODUCT))
    parser.add_argument("--layer", default=os.getenv("NDVI_LAYER", DEFAULT_LAYER))
    parser.add_argument("--start-date", default=os.getenv("NDVI_START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.getenv("NDVI_END_DATE", DEFAULT_END_DATE))
    parser.add_argument("--bbox", default=os.getenv("NDVI_BBOX", DEFAULT_BBOX))
    parser.add_argument("--projection", default=os.getenv("NDVI_PROJECTION", "sinu_modis"))
    parser.add_argument("--download-dir", type=Path, default=Path(os.getenv("NDVI_DOWNLOAD_DIR", str(DEFAULT_LOCAL_DIR / "geotiffs"))))
    parser.add_argument("--local-output", type=Path, default=DEFAULT_LOCAL_OUTPUT)
    parser.add_argument("--grid-size", type=float, default=float(os.getenv("GRID_SIZE", str(DEFAULT_GRID_SIZE))))
    parser.add_argument("--chunk-rows", type=int, default=int(os.getenv("NDVI_CHUNK_ROWS", "512")))
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("APPEEARS_POLL_SECONDS", "60")))
    parser.add_argument("--wait-timeout-seconds", type=float, default=float(os.getenv("APPEEARS_WAIT_TIMEOUT_SECONDS", "86400")))
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--skip-minio-upload", action="store_true")
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--minio-prefix", default=os.getenv("NDVI_MINIO_PREFIX", DEFAULT_MINIO_PREFIX))
    parser.add_argument("--object-name", default=os.getenv("NDVI_OBJECT_NAME", DEFAULT_OBJECT_NAME))
    parser.add_argument("--print-progress", action="store_true")

    args = parser.parse_args()
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
        parser.error("--start-date must be on or before --end-date")
    return args


def main() -> int:
    args = parse_args()
    task_id = args.task_id

    if args.mode in {"submit", "download", "all"}:
        session = requests.Session()
        session.headers.update({"User-Agent": "wildfire-mod13q1-ndvi/1.0"})
        token = appeears_login(args, session)

        if args.mode in {"submit", "all"}:
            task_id = submit_task(args, session, token)
            print(f"Submitted AppEEARS task: {task_id}")

        if args.mode in {"download", "all"}:
            if not task_id:
                task_id = task_id_from_file(args.task_file)
            wait_for_task(args, session, token, task_id)
            downloaded = download_bundle_files(args, session, token, task_id)
            print(f"Downloaded {len(downloaded):,} GeoTIFF file(s) to {args.download_dir}")

    if args.mode in {"process", "all"}:
        row_count = process_geotiffs(args)
        print(f"Saved NDVI grid Parquet to {args.local_output} ({row_count:,} rows)")
        if not args.skip_minio_upload:
            uri = upload_to_minio(args)
            print(f"Uploaded NDVI grid Parquet to {uri}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
