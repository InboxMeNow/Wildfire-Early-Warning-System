#!/usr/bin/env python3
"""
Download NASA FIRMS VIIRS NOAA-20 NRT fire detections for Vietnam.

Usage:
    $env:FIRMS_MAP_KEY = "your_map_key"
    python 01_explore_firms.py

Optional:
    python 01_explore_firms.py --output data/raw/firms_vietnam.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCE = "VIIRS_NOAA20_NRT"
VIETNAM_BBOX = "95,5,115,25"
DEFAULT_DAYS = 5
MAX_DAYS = 5
DEFAULT_OUTPUT = Path("data/raw/firms_viirs_noaa20_vietnam_last5days.csv")
DEFAULT_ENV_FILE = Path(".env")


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
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


def build_url(map_key: str, days: int) -> str:
    return (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{SOURCE}/{VIETNAM_BBOX}/{days}"
    )


def download_csv(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "wildfire-firms-explorer/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Download FIRMS VIIRS NOAA-20 NRT CSV data for Vietnam, last 7 days."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--map-key",
        default=os.getenv("FIRMS_MAP_KEY") or os.getenv("MAP_KEY"),
        help="NASA FIRMS MAP_KEY. Defaults to FIRMS_MAP_KEY or MAP_KEY environment variable.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of recent days to download. FIRMS currently accepts 1..{MAX_DAYS} for this NRT endpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.map_key:
        print(
            "Missing NASA FIRMS MAP_KEY. Set FIRMS_MAP_KEY or MAP_KEY, "
            "or pass --map-key.",
            file=sys.stderr,
        )
        return 2
    if args.map_key == "PASTE_YOUR_FIRMS_MAP_KEY_HERE":
        print(
            "Please replace PASTE_YOUR_FIRMS_MAP_KEY_HERE in .env with your real FIRMS MAP_KEY.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.days <= MAX_DAYS:
        print(
            f"Invalid --days {args.days}. FIRMS currently accepts 1..{MAX_DAYS} "
            f"for {SOURCE}.",
            file=sys.stderr,
        )
        return 2

    try:
        csv_bytes = download_csv(build_url(args.map_key, args.days))
    except HTTPError as exc:
        print(f"FIRMS API returned HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        if error_body:
            print(error_body, file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Could not reach FIRMS API: {exc.reason}", file=sys.stderr)
        return 1
    except TimeoutError:
        print("FIRMS API request timed out.", file=sys.stderr)
        return 1

    if not csv_bytes.strip():
        print("FIRMS API returned an empty CSV response.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(csv_bytes)

    print(f"Saved {csv_bytes.count(b'\n')} CSV lines to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
