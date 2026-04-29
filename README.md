# Wildfire Early Warning System

Week 1 data and ML pipeline for a wildfire risk early-warning project. The pipeline collects NASA FIRMS fire detections and weather data, cleans them with Spark on MinIO, builds 0.5 degree grid-day features, creates a first visualization, and trains a baseline Spark ML model.

## Architecture

- `MinIO`: local object storage. The default bucket is `wildfire-data`.
- `Spark`: PySpark ETL and ML jobs that read/write MinIO through `s3a://`.
- `NASA FIRMS`: historical active fire detections for the area around Vietnam, 2020-2024.
- `Meteostat/Open-Meteo`: daily weather data on a grid around Vietnam.
- `geo/vietnam_boundary.geojson`: Vietnam boundary mask used during cleaning so downstream datasets only keep records inside Vietnam.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
```

MinIO:

- API: `http://localhost:9000`
- Console: `http://localhost:9001`
- User/password: `minioadmin` / `minioadmin`

Spark UI:

- Master: `http://localhost:8080`
- Worker: `http://localhost:8081`

The local Spark image is built from `docker/spark/Dockerfile` and installs `numpy`, which PySpark MLlib needs for model training.

## Configuration

Create a `.env` file if you need to fetch FIRMS data again:

```text
FIRMS_MAP_KEY=your_firms_map_key
```

Default MinIO settings used by the scripts:

```text
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=wildfire-data
```

When running inside the Docker network, `docker-compose.yml` sets `MINIO_ENDPOINT=http://minio:9000` for Spark.

## Week 1 Pipeline

### 1. FIRMS NRT Exploration

```powershell
python 01_explore_firms.py
```

Local output:

- `data/raw/firms_viirs_noaa20_vietnam_last5days.csv`

### 2. Fetch Weather

```powershell
python 02_fetch_weather.py
```

Output:

- Local cache: `data/raw/meteostat_daily_vietnam_2020_2024.parquet`
- MinIO: `s3://wildfire-data/weather/meteostat_daily_vietnam_2020_2024.parquet`

### 3. Fetch FIRMS History

```powershell
python 03_fetch_firms_history.py
```

Output:

- Local cache: `data/raw/firms_history_vietnam_2020_2024.parquet`
- MinIO: `s3://wildfire-data/firms/firms_history_vietnam_2020_2024.parquet`

### 4. Spark Clean ETL

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/04_etl_clean.py
```

Processing steps:

- Read `s3a://wildfire-data/firms/` and `s3a://wildfire-data/weather/`.
- Filter points with `geo/vietnam_boundary.geojson` to keep only records inside Vietnam.
- Filter FIRMS records to `confidence >= 30`.
- Convert date columns.
- Drop duplicates.
- Write `s3a://wildfire-data/firms_clean/` and `s3a://wildfire-data/weather_clean/`.

### 5. Feature Engineering

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/05_feature_engineering.py --print-counts
```

Output:

- MinIO: `s3a://wildfire-data/features/`

Each row in the final dataset is one `(grid_id, date)`:

- Grid cell: `floor(latitude / 0.5)`, `floor(longitude / 0.5)`
- Fire aggregate: `fire_count`
- Weather features: temperature, humidity, wind, precipitation
- Rolling features: `precipitation_sum_7days`, `dry_days_count`
- Labels: `fire_occurred`, `risk_level`

### 6. Data Quality And Heatmap

```powershell
python 06_data_quality_and_heatmap.py
```

Output:

- Report: `reports/data_quality_week1.md`
- Heatmap: `maps/fires_heatmap_2020_2024.html`

### 7. Baseline ML Training

Notebook:

```text
notebooks/06_train_model.ipynb
```

Equivalent Spark script for reproducible command-line training:

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/07_train_model.py
```

Processing steps:

- Read `s3a://wildfire-data/features/`.
- Time-based split: train on 2020-2023, test on 2024.
- Compute class weights for `fire_occurred`.
- Train Spark MLlib `RandomForestClassifier`.
- Evaluate AUC-ROC, precision, recall, and F1.
- Save the model to `s3a://wildfire-data/models/random_forest_fire_baseline/`.
- Save metrics and feature importance artifacts to `reports/`.

## Current Results

Latest run after applying the Vietnam boundary mask:

- `firms_clean`: 607,106 rows
- `weather_clean`: 206,451 rows
- `features`: 206,451 rows, 113 grid cells, 20 columns
- Date range: `2020-01-01` to `2024-12-31`
- Data quality: PASS, no duplicate `(grid_id, date)` rows, labels match `fire_count`
- Baseline RF AUC-ROC on 2024 test data: 0.8369

Label distribution:

- `risk_level = 0`: 145,269 rows
- `risk_level = 1`: 41,531 rows
- `risk_level = 2`: 19,651 rows

## File Structure

```text
01_explore_firms.py              # FIRMS NRT quick fetch
02_fetch_weather.py              # Daily weather fetch/upload
03_fetch_firms_history.py        # FIRMS history fetch/upload
04_etl_clean.py                  # Spark cleaning ETL
05_feature_engineering.py        # Spark spatial join + ML features
06_data_quality_and_heatmap.py   # Data quality report + heatmap
07_train_model.py                # Spark MLlib RandomForest baseline
geo_utils.py                     # GeoJSON point-in-polygon helpers
geo/vietnam_boundary.geojson     # Vietnam country boundary mask
docker-compose.yml               # Local MinIO + Spark stack
docker/spark/Dockerfile          # Spark image with numpy for MLlib
spark-conf/spark-defaults.conf   # S3A config for Spark
maps/                            # HTML visualizations
reports/                         # Data quality and model reports
```

## Notes

- `data/raw/` is gitignored because it contains large local cache files.
- The FIRMS area API accepts a bounding box, so raw fetches can include neighboring countries. `04_etl_clean.py` is the step that clips records back to the Vietnam boundary.
- `spark-conf/spark-defaults.conf` configures `hadoop-aws` and S3A settings so Spark can read/write MinIO reliably.
- If Spark runs outside Docker, the MinIO endpoint should usually be `http://localhost:9000`. If Spark runs inside the Docker network, the endpoint should be `http://minio:9000`.
