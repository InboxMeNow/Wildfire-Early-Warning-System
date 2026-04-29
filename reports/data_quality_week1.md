# Week 1 Data Quality Report

Status: PASS

## MinIO Objects

| Dataset | Parquet parts | Size |
|---|---:|---:|
| firms_clean | 2 | 16.6 MB |
| weather_clean | 2 | 974.0 KB |
| features | 2 | 1.8 MB |

## Dataset Shape

| Dataset | Rows | Columns | Date range |
|---|---:|---:|---|
| firms_clean | 607,106 | 21 | 2020-01-01 to 2024-12-31 |
| weather_clean | 206,451 | 12 | 2020-01-01 to 2024-12-31 |
| features | 206,451 | 20 | 2020-01-01 to 2024-12-31 |

## Feature Integrity

- Grid cells: 113
- Duplicate `(grid_id, date)` rows: 0
- Bad `fire_occurred` labels: 0
- Bad `risk_level` labels: 0
- `risk_level` counts: {0: 145269, 1: 41531, 2: 19651}
- `fire_occurred` counts: {0: 145269, 1: 61182}

## Heatmap

- Heatmap grid cells with fires: 167
- Max fires in one 0.5 degree cell: 18,680
- Output: `maps\fires_heatmap_2020_2024.html`

## Issues

- No blocking data quality issues found in week-1 outputs.
