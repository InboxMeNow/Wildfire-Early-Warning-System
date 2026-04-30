#!/usr/bin/env python3
"""Benchmark single-node GeoPandas spatial join versus Spark/Sedona."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from sedona.spark import SedonaContext
from shapely.geometry import box
from pyspark.sql import functions as F


DEFAULT_BOUNDARY = Path("geo/vietnam_boundary.geojson")
DEFAULT_REPORT = Path("reports/sedona_spatial_benchmark.md")
DEFAULT_CSV = Path("reports/sedona_spatial_benchmark.csv")
DEFAULT_ROWS = 1_000_000
DEFAULT_GRID_SIZE = 0.5
DEFAULT_SEDONA_PACKAGES = (
    "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.5.1,"
    "org.datasyslab:geotools-wrapper:1.5.1-28.2"
)


def boundary_bounds(path: Path) -> tuple[float, float, float, float]:
    series = gpd.read_file(path)
    min_lon, min_lat, max_lon, max_lat = series.total_bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def grid_records(bounds: tuple[float, float, float, float], grid_size: float) -> list[dict[str, float | int | str]]:
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_indexes = range(math.floor(min_lon / grid_size), math.ceil(max_lon / grid_size))
    lat_indexes = range(math.floor(min_lat / grid_size), math.ceil(max_lat / grid_size))
    records = []
    for lat_index in lat_indexes:
        for lon_index in lon_indexes:
            grid_lat = round(lat_index * grid_size, 6)
            grid_lon = round(lon_index * grid_size, 6)
            records.append(
                {
                    "grid_id": f"{lat_index}_{lon_index}",
                    "grid_lat_index": lat_index,
                    "grid_lon_index": lon_index,
                    "grid_lat": grid_lat,
                    "grid_lon": grid_lon,
                    "min_lon": grid_lon,
                    "min_lat": grid_lat,
                    "max_lon": round(grid_lon + grid_size, 6),
                    "max_lat": round(grid_lat + grid_size, 6),
                }
            )
    return records


def benchmark_geopandas(rows: int, bounds: tuple[float, float, float, float], grid_size: float) -> tuple[int, float]:
    started = time.perf_counter()
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    point_ids = pd.Series(range(rows), name="id")
    fires = pd.DataFrame(
        {
            "id": point_ids,
            "longitude": min_lon + ((point_ids * 37) % 1_000_000) / 1_000_000 * lon_range,
            "latitude": min_lat + ((point_ids * 91) % 1_000_000) / 1_000_000 * lat_range,
        }
    )
    gdf_fires = gpd.GeoDataFrame(
        fires,
        geometry=gpd.points_from_xy(fires["longitude"], fires["latitude"]),
        crs="EPSG:4326",
    )
    grid_frame = pd.DataFrame(grid_records(bounds, grid_size))
    grid = gpd.GeoDataFrame(
        grid_frame,
        geometry=[
            box(row.min_lon, row.min_lat, row.max_lon, row.max_lat)
            for row in grid_frame.itertuples(index=False)
        ],
        crs="EPSG:4326",
    )

    joined = gdf_fires.sjoin(grid[["grid_id", "geometry"]], how="inner", predicate="within")
    elapsed = time.perf_counter() - started
    return len(joined), elapsed


def create_sedona(master: str, packages: str):
    builder = SedonaContext.builder().appName("Sedona Spatial Join Benchmark").master(master)
    if packages:
        builder = builder.config("spark.jars.packages", packages)
        builder = builder.config("spark.jars.excludes", "edu.ucar:cdm-core")
    builder = builder.config("spark.sql.shuffle.partitions", "8")
    return SedonaContext.create(builder.getOrCreate())


def benchmark_sedona(
    rows: int,
    bounds: tuple[float, float, float, float],
    grid_size: float,
    master: str,
    packages: str,
) -> tuple[int, float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    spark = create_sedona(master, packages)
    try:
        started = time.perf_counter()
        lat_start = math.floor(min_lat / grid_size)
        lat_end = math.ceil(max_lat / grid_size)
        lon_start = math.floor(min_lon / grid_size)
        lon_end = math.ceil(max_lon / grid_size)
        lat_indexes = spark.range(lat_start, lat_end).select(F.col("id").cast("int").alias("grid_lat_index"))
        lon_indexes = spark.range(lon_start, lon_end).select(F.col("id").cast("int").alias("grid_lon_index"))
        grid = (
            lat_indexes.crossJoin(lon_indexes)
            .withColumn("grid_id", F.concat_ws("_", F.col("grid_lat_index"), F.col("grid_lon_index")))
            .withColumn("min_lon", F.round(F.col("grid_lon_index") * F.lit(grid_size), 6))
            .withColumn("min_lat", F.round(F.col("grid_lat_index") * F.lit(grid_size), 6))
            .withColumn("max_lon", F.round(F.col("min_lon") + F.lit(grid_size), 6))
            .withColumn("max_lat", F.round(F.col("min_lat") + F.lit(grid_size), 6))
            .withColumn("geometry", F.expr("ST_PolygonFromEnvelope(min_lon, min_lat, max_lon, max_lat)"))
        )
        grid.createOrReplaceTempView("grid")
        grid.cache().count()

        fires = spark.range(rows).selectExpr(
            "id",
            f"{min_lon}D + (PMOD(id * 37L, 1000000L) / 1000000D) * {lon_range}D AS longitude",
            f"{min_lat}D + (PMOD(id * 91L, 1000000L) / 1000000D) * {lat_range}D AS latitude",
        )
        fires = fires.withColumn("geometry", F.expr("ST_Point(longitude, latitude)"))
        fires.createOrReplaceTempView("fires")

        joined_count = spark.sql(
            """
            SELECT /*+ BROADCAST(g) */ count(*) AS joined_count
            FROM fires f
            JOIN grid g
              ON ST_Contains(g.geometry, f.geometry)
            """
        ).collect()[0]["joined_count"]
        elapsed = time.perf_counter() - started
        return int(joined_count), elapsed
    finally:
        spark.stop()


def write_report(
    report_path: Path,
    csv_path: Path,
    rows: int,
    grid_size: float,
    geopandas_result: tuple[int, float],
    sedona_result: tuple[int, float],
    sedona_master: str,
) -> None:
    geopandas_count, geopandas_seconds = geopandas_result
    sedona_count, sedona_seconds = sedona_result
    speedup = geopandas_seconds / sedona_seconds if sedona_seconds else float("inf")
    pass_latency = sedona_seconds < 30
    pass_speedup = speedup >= 3
    status = "PASS" if pass_latency and pass_speedup else "NEEDS REVIEW"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"engine": "GeoPandas", "rows": rows, "joined_rows": geopandas_count, "seconds": geopandas_seconds},
            {"engine": "Sedona", "rows": rows, "joined_rows": sedona_count, "seconds": sedona_seconds},
        ]
    ).to_csv(csv_path, index=False)

    report_path.write_text(
        "\n".join(
            [
                "# Sedona Spatial Join Benchmark",
                "",
                f"- Input points: {rows:,}",
                f"- Grid size: {grid_size:g} degrees",
                f"- Sedona master: `{sedona_master}`",
                f"- Result: **{status}**",
                "",
                "| Engine | Joined rows | Seconds |",
                "|---|---:|---:|",
                f"| GeoPandas | {geopandas_count:,} | {geopandas_seconds:.3f} |",
                f"| Sedona | {sedona_count:,} | {sedona_seconds:.3f} |",
                "",
                f"Sedona speedup: **{speedup:.2f}x**",
                "",
                "Acceptance criteria:",
                f"- Sedona under 30 seconds: {'PASS' if pass_latency else 'FAIL'}",
                f"- Sedona at least 3x faster than GeoPandas: {'PASS' if pass_speedup else 'FAIL'}",
                "",
                "Notes:",
                f"- This benchmark uses a simple {grid_size:g}-degree rectangular grid over the Vietnam bounding box.",
                "- GeoPandas/Shapely 2.x is highly optimized for this small-polygon local case, so Spark startup and execution overhead dominate.",
                "- Sedona remains the scalable path for distributed joins against larger point volumes, richer polygons, or cluster-backed ETL.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GeoPandas vs Sedona spatial joins.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--sedona-master", default="local[4]")
    parser.add_argument("--sedona-packages", default=DEFAULT_SEDONA_PACKAGES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if not args.boundary.exists():
        parser.error(f"--boundary not found: {args.boundary}")
    return args


def main() -> int:
    args = parse_args()
    bounds = boundary_bounds(args.boundary)

    geopandas_result = benchmark_geopandas(args.rows, bounds, args.grid_size)
    sedona_result = benchmark_sedona(
        rows=args.rows,
        bounds=bounds,
        grid_size=args.grid_size,
        master=args.sedona_master,
        packages=args.sedona_packages,
    )
    write_report(
        args.report,
        args.csv,
        args.rows,
        args.grid_size,
        geopandas_result,
        sedona_result,
        args.sedona_master,
    )

    print(args.report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
