# Sedona Spatial Join Benchmark

- Input points: 1,000,000
- Grid size: 0.5 degrees
- Sedona master: `local[4]`
- Result: **NEEDS REVIEW**

| Engine | Joined rows | Seconds |
|---|---:|---:|
| GeoPandas | 1,000,000 | 0.708 |
| Sedona | 1,000,000 | 4.672 |

Sedona speedup: **0.15x**

Acceptance criteria:
- Sedona under 30 seconds: PASS
- Sedona at least 3x faster than GeoPandas: FAIL

Notes:
- This benchmark uses a simple 0.5-degree rectangular grid over the Vietnam bounding box.
- GeoPandas/Shapely 2.x is highly optimized for this small-polygon local case, so Spark startup and execution overhead dominate.
- Sedona remains the scalable path for distributed joins against larger point volumes, richer polygons, or cluster-backed ETL.
