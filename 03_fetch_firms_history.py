#!/usr/bin/env python3
"""
Fetch historical NASA FIRMS active fire data for Vietnam and save Parquet to MinIO.

The interactive archive downloader at https://firms.modaps.eosdis.nasa.gov/download/
requires Earthdata/email authentication. This script uses the FIRMS area API loop
with the same Vietnam bounding box used by 01_explore_firms.py.

Default output:
    s3://wildfire-data/firms/firms_history_vietnam_2020_2024.parquet

Run:
    python 03_fetch_firms_history.py
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from botocore.client import Config
from botocore.exceptions import ClientError


FIRMS_AREA_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
VIETNAM_BBOX = "95,5,115,25"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_DAY_RANGE = 5
DEFAULT_SOURCES = ["VIIRS_NOAA20_SP", "VIIRS_SNPP_SP", "MODIS_SP"]
DEFAULT_LOCAL_OUTPUT = Path("data/raw/firms_history_vietnam_2020_2024.parquet")
DEFAULT_PARTS_DIR = Path("data/raw/firms_history_vietnam_parts")
DEFAULT_OBJECT_NAME = "firms_history_vietnam_2020_2024.parquet"

FIRMS_SCHEMA = pa.schema(
    [
        ("firms_source", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("brightness", pa.float64()),
        ("bright_ti4", pa.float64()),
        ("bright_ti5", pa.float64()),
        ("bright_t31", pa.float64()),
        ("scan", pa.float64()),
        ("track", pa.float64()),
        ("acq_date", pa.date32()),
        ("acq_time", pa.int32()),
        ("satellite", pa.string()),
        ("instrument", pa.string()),
        ("confidence", pa.string()),
        ("version", pa.string()),
        ("frp", pa.float64()),
        ("daynight", pa.string()),
        ("type", pa.string()),
        ("query_start", pa.date32()),
        ("query_end", pa.date32()),
    ]
)


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    @property
    def day_range(self) -> int:
        return (self.end - self.start).days + 1


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


def date_windows(start_date: str, end_date: str, day_range: int) -> Iterable[DateWindow]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=day_range - 1), end)
        yield DateWindow(current, window_end)
        current = window_end + timedelta(days=1)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def part_path(parts_dir: Path, source: str, window: DateWindow) -> Path:
    return parts_dir / f"{safe_name(source)}_{window.start}_{window.end}.parquet"


def build_area_url(map_key: str, source: str, bbox: str, window: DateWindow) -> str:
    return (
        f"{FIRMS_AREA_API}/{map_key}/{source}/{bbox}/"
        f"{window.day_range}/{window.start.isoformat()}"
    )


def fetch_firms_csv(
    session: requests.Session,
    map_key: str,
    source: str,
    bbox: str,
    window: DateWindow,
    timeout_seconds: int,
    max_retries: int,
    retry_sleep_seconds: int,
) -> pd.DataFrame:
    url = build_area_url(map_key, source, bbox, window)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                response.raise_for_status()
            if not response.ok:
                if 400 <= response.status_code < 500:
                    raise ValueError(
                        f"FIRMS HTTP {response.status_code}: {response.text[:500]}"
                    )
                raise RuntimeError(
                    f"FIRMS HTTP {response.status_code}: {response.text[:500]}"
                )

            text = response.text.strip()
            if not text:
                return pd.DataFrame()
            if text.startswith("Invalid API call") or text.startswith("Error"):
                raise ValueError(text[:500])

            return pd.read_csv(io.StringIO(text))
        except ValueError:
            raise
        except (requests.RequestException, RuntimeError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = retry_sleep_seconds * (attempt + 1)
            print(
                f"FIRMS request failed for {source} {window.start}..{window.end}: "
                f"{exc}; retrying in {sleep_seconds}s",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"FIRMS request failed after retries for {source} "
        f"{window.start}..{window.end}: {last_error}"
    )


def empty_normalized_frame() -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in FIRMS_SCHEMA})


def normalize_firms_frame(
    frame: pd.DataFrame,
    source: str,
    window: DateWindow,
) -> pd.DataFrame:
    if frame.empty:
        normalized = empty_normalized_frame()
    else:
        normalized = frame.copy()

    normalized["firms_source"] = source
    normalized["query_start"] = window.start
    normalized["query_end"] = window.end

    for column in [
        "latitude",
        "longitude",
        "brightness",
        "bright_ti4",
        "bright_ti5",
        "bright_t31",
        "scan",
        "track",
        "frp",
    ]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")

    if "brightness" in normalized.columns and "bright_ti4" in normalized.columns:
        normalized["brightness"] = normalized["brightness"].fillna(normalized["bright_ti4"])

    if "acq_date" not in normalized.columns:
        normalized["acq_date"] = pd.NaT
    normalized["acq_date"] = pd.to_datetime(normalized["acq_date"], errors="coerce").dt.date

    if "acq_time" not in normalized.columns:
        normalized["acq_time"] = pd.NA
    normalized["acq_time"] = pd.to_numeric(normalized["acq_time"], errors="coerce").astype("Int32")

    for column in ["satellite", "instrument", "confidence", "version", "daynight", "type"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = normalized[column].astype("string")

    normalized["firms_source"] = normalized["firms_source"].astype("string")
    normalized["query_start"] = pd.to_datetime(normalized["query_start"]).dt.date
    normalized["query_end"] = pd.to_datetime(normalized["query_end"]).dt.date

    return normalized[[field.name for field in FIRMS_SCHEMA]]


def write_part(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, schema=FIRMS_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def fetch_parts(args: argparse.Namespace) -> tuple[list[Path], int]:
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    windows = list(date_windows(args.start_date, args.end_date, args.day_range))
    total_requests = len(args.sources) * len(windows)
    request_number = 0
    row_count = 0
    part_paths: list[Path] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "wildfire-firms-history/1.0"})

    print(
        f"Fetching FIRMS history for bbox {args.bbox}, {len(args.sources)} sources, "
        f"{len(windows)} date windows ({total_requests} requests)"
    )

    for source in args.sources:
        for window in windows:
            request_number += 1
            current_part_path = part_path(args.parts_dir, source, window)
            part_paths.append(current_part_path)

            if args.resume and current_part_path.exists():
                table = pq.read_table(current_part_path, columns=["latitude"])
                row_count += table.num_rows
                if request_number == 1 or request_number % args.progress_every == 0:
                    print(
                        f"Skipping existing {request_number:,}/{total_requests:,}: "
                        f"{source} {window.start}..{window.end}"
                    )
                continue

            raw = fetch_firms_csv(
                session=session,
                map_key=args.map_key,
                source=source,
                bbox=args.bbox,
                window=window,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            normalized = normalize_firms_frame(raw, source, window)
            write_part(current_part_path, normalized)
            row_count += len(normalized)

            if request_number == 1 or request_number % args.progress_every == 0:
                print(
                    f"Fetched {request_number:,}/{total_requests:,}: "
                    f"{source} {window.start}..{window.end} ({len(normalized):,} rows)"
                )

            if request_number < total_requests and args.request_delay_seconds > 0:
                time.sleep(args.request_delay_seconds)

    return part_paths, row_count


def merge_parts(output_path: Path, part_paths: list[Path]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0

    try:
        for current_part_path in part_paths:
            if not current_part_path.exists():
                raise RuntimeError(f"Missing checkpoint part: {current_part_path}")

            table = pq.read_table(current_part_path, schema=FIRMS_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(
                    where=output_path,
                    schema=FIRMS_SCHEMA,
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


def parse_sources(value: str) -> list[str]:
    sources = [source.strip() for source in value.split(",") if source.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("At least one source is required")
    return sources


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Fetch historical NASA FIRMS active fire data and upload Parquet to MinIO."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--bbox", default=VIETNAM_BBOX)
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=DEFAULT_SOURCES,
        help="Comma-separated FIRMS sources. Default: VIIRS_NOAA20_SP,VIIRS_SNPP_SP,MODIS_SP",
    )
    parser.add_argument(
        "--map-key",
        default=os.getenv("FIRMS_MAP_KEY") or os.getenv("MAP_KEY"),
        help="NASA FIRMS MAP_KEY. Defaults to FIRMS_MAP_KEY or MAP_KEY environment variable.",
    )
    parser.add_argument("--day-range", type=int, default=DEFAULT_DAY_RANGE)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=int, default=10)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=20)
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
        help="Directory for per-request checkpoint Parquet files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing checkpoint files and refetch all request windows.",
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
    )
    parser.add_argument(
        "--minio-prefix",
        default=os.getenv("FIRMS_MINIO_PREFIX", "firms"),
    )
    parser.add_argument(
        "--object-name",
        default=os.getenv("FIRMS_OBJECT_NAME", DEFAULT_OBJECT_NAME),
    )

    args = parser.parse_args()
    if not args.map_key:
        parser.error("Missing FIRMS MAP_KEY. Set FIRMS_MAP_KEY/MAP_KEY or pass --map-key.")
    if args.map_key == "PASTE_YOUR_FIRMS_MAP_KEY_HERE":
        parser.error("Replace placeholder MAP_KEY in .env or pass --map-key.")
    if not 1 <= args.day_range <= 5:
        parser.error("--day-range must be in 1..5 for this FIRMS API endpoint")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.request_delay_seconds < 0:
        parser.error("--request-delay-seconds must be non-negative")
    if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
        parser.error("--start-date must be on or before --end-date")

    return args


def main() -> int:
    args = parse_args()
    part_paths, _ = fetch_parts(args)
    row_count = merge_parts(args.local_output, part_paths)

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
