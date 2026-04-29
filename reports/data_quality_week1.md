# Week 1 Data Quality Report

Status: PASS

## MinIO Objects

| Dataset | Parquet parts | Size |
|---|---:|---:|
| firms_clean | 13 | 217.9 MB |
| weather_clean | 2 | 14.2 MB |
| features | 2 | 18.6 MB |

## Dataset Shape

| Dataset | Rows | Columns | Date range |
|---|---:|---:|---|
| firms_clean | 8,044,822 | 21 | 2020-01-01 to 2024-12-31 |
| weather_clean | 3,071,187 | 12 | 2020-01-01 to 2024-12-31 |
| features | 3,071,187 | 20 | 2020-01-01 to 2024-12-31 |

## Feature Integrity

- Grid cells: 1,681
- Duplicate `(grid_id, date)` rows: 0
- Bad `fire_occurred` labels: 0
- Bad `risk_level` labels: 0
- `risk_level` counts: {0: 2549298, 1: 313441, 2: 208448}
- `fire_occurred` counts: {0: 2549298, 1: 521889}

## Heatmap

- Heatmap grid cells with fires: 1,035
- Max fires in one 0.5 degree cell: 82,524
- Output: `maps\fires_heatmap_2020_2024.html`

## Issues

- No blocking data quality issues found in week-1 outputs.
