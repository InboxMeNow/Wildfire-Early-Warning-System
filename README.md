# Wildfire Early Warning System

Data and ML pipeline for a wildfire risk early-warning project in Vietnam. The pipeline collects NASA FIRMS fire detections and weather data, cleans them with Spark on MinIO, builds 0.5 degree grid-day features, visualizes historical fire density, trains Spark ML risk models, and produces near-real-time DBSCAN clusters plus anomaly alerts.

## Architecture

- `MinIO`: local object storage. The default bucket is `wildfire-data`.
- `Spark`: PySpark ETL, feature engineering, and ML jobs that read/write MinIO through `s3a://`.
- `NASA FIRMS`: historical active fire detections for the area around Vietnam, 2020-2024.
- `Meteostat/Open-Meteo`: daily weather data on a grid around Vietnam.
- `geo/vietnam_boundary.geojson`: Vietnam boundary mask used during cleaning so downstream datasets only keep records inside Vietnam.
- `scikit-learn`: local DBSCAN clustering for recent fire points and lightweight anomaly-detection utilities.

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

- Master: `http://localhost:8082`
- Worker: `http://localhost:8083`

The local Spark image is built from `docker/spark/Dockerfile` and installs `numpy`, Apache Sedona, and Shapely for MLlib and distributed spatial ETL.

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

## Pipeline

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

### 7. Spark ML Training

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
- Tune Spark MLlib `RandomForestClassifier` with `CrossValidator`.
- Train and compare a Spark MLlib `GBTClassifier`.
- Evaluate AUC-ROC, precision, recall, and F1.
- Save the RF model to `s3a://wildfire-data/models/random_forest_fire_baseline/`.
- Save the GBT model to `s3a://wildfire-data/models/gbt_fire_baseline/`.
- Save metrics and feature importance artifacts to `reports/`.

### 8. Recent Fire Clustering

```powershell
python 07_dbscan_clustering.py
```

Processing steps:

- Read cleaned fire points from `s3://wildfire-data/firms_clean/`.
- Select points from the latest 24-hour window in the dataset.
- Run `DBSCAN(eps=0.05, min_samples=3)` on latitude/longitude.
- Build a convex hull for each cluster.
- Save GeoJSON and metadata locally under `reports/`.
- Upload the latest cluster outputs to `s3://wildfire-data/models/dbscan_fire_clusters/`.

### 9. Fire Anomaly Detection

```powershell
python 08_anomaly_detection.py
```

Processing steps:

- Read feature rows from `s3://wildfire-data/features/`.
- Compute each grid cell's historical daily fire-count mean and standard deviation.
- Score the latest date with a simple z-score rule: `fire_count > mean + 3 * std`.
- Save anomaly GeoJSON, detector stats, and metadata locally under `reports/`.
- Upload the latest detector outputs to `s3://wildfire-data/models/fire_anomaly_detector/`.

### 10. Next-Day Inference

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/09_inference.py
```

Processing steps:

- Read Vietnam grid cells from `s3a://wildfire-data/features/`.
- Fetch Open-Meteo forecast data for the target day, which defaults to tomorrow in `Asia/Ho_Chi_Minh`.
- Rebuild the same model features used during training, including time features, 7-day precipitation, and dry-day streak.
- Load the RF pipeline from `s3a://wildfire-data/models/random_forest_fire_baseline/`.
- Predict fire occurrence probability for each grid cell.
- Convert probability to `risk_level`: low (`0`), medium (`1`), high (`2`).
- Save GeoJSON and metadata under `reports/`.
- Upload the latest outputs to `s3a://wildfire-data/predictions/fire_risk_forecast/`.

The forecast API variables used by the script are documented in the official Open-Meteo Weather Forecast API docs: https://open-meteo.com/en/docs

Daily automation on Windows can call:

```powershell
.\scripts\run_daily_inference.ps1
```

### 11. Streamlit Dashboard

```powershell
streamlit run app.py
```

If `streamlit` is not on `PATH`, run:

```powershell
python -m streamlit run app.py
```

Dashboard views:

- Risk Map: Folium forecast choropleth and heatmap from `reports/fire_risk_forecast_latest.geojson`.
- Active Fires: FIRMS fire-point scatter from MinIO plus DBSCAN cluster polygons from `reports/dbscan_fire_clusters_latest.geojson`.
- Statistics: Plotly charts for risk levels, daily fire detections, model metrics, and anomaly output.

## Current Results

Latest run after applying the Vietnam boundary mask:

- `firms_clean`: 607,106 rows
- `weather_clean`: 206,451 rows
- `features`: 206,451 rows, 113 grid cells, 20 columns
- Date range: `2020-01-01` to `2024-12-31`
- Data quality: PASS, no duplicate `(grid_id, date)` rows, labels match `fire_count`
- Tuned RF AUC-ROC on 2024 test data: 0.8516
- GBT AUC-ROC on 2024 test data: 0.8614
- Best current Spark risk model by AUC-ROC: `gbt`
- Latest DBSCAN run: 86 fire points in the latest 24-hour window, 8 clusters
- Latest anomaly run: 2 anomalous grids on `2024-12-31`
- Latest inference run: 113 grid cells predicted for `2026-04-30`
- Latest inference risk levels: 24 low, 53 medium, 36 high

Label distribution:

- `risk_level = 0`: 145,269 rows
- `risk_level = 1`: 41,531 rows
- `risk_level = 2`: 19,651 rows

### 12. Kafka Streaming

Start Kafka, Kafka UI, and the topic initializer:

```powershell
docker compose up -d kafka kafka-init kafka-ui
```

Kafka runs in KRaft mode on `localhost:9092`. The compose file uses
`bitnamilegacy/kafka:3.6` because the original `bitnami/kafka:3.6` tag is no
longer published on Docker Hub. Kafka UI is available at:

```text
http://localhost:8080
```

The compose initializer creates these topics automatically:

- `fire-events`
- `weather-updates`
- `alerts`

You can also create a topic manually:

```powershell
docker exec kafka kafka-topics.sh --create --if-not-exists --topic fire-events --bootstrap-server localhost:9092
```

Install Python dependencies, then publish recent FIRMS detections:

```powershell
pip install -r requirements.txt
python src/streaming/firms_producer.py --once
```

Run continuously every 3 minutes:

```powershell
python src/streaming/firms_producer.py
```

Verify messages:

```powershell
python src/streaming/test_consumer.py --max-messages 10
```

Note: Kafka UI uses host port `8080`; Spark master UI is mapped to `localhost:8082`.

### 13. Spark Structured Streaming Alerts

Run the Spark streaming processor in the Docker Spark image against Kafka:

```powershell
docker compose run -d --name wildfire-spark-stream --no-deps spark-master `
  /opt/spark/bin/spark-submit `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6 `
  src/streaming/spark_streaming_job.py `
  --bootstrap-servers kafka:29092 `
  --starting-offsets latest `
  --checkpoint-location /tmp/spark-checkpoints/fire-hot-zones `
  --kafka-package none
```

The job reads `fire-events`, computes 0.5-degree grid cells, applies a 10-minute
watermark, counts fires in 1-hour windows, and writes hot-zone alerts to
`alerts` when a cell has more than 5 fires.

For a deterministic smoke test, start the Spark job in one terminal, then publish
10 synthetic FIRMS-like events into the same grid:

```powershell
python src/streaming/produce_test_burst.py --count 10
```

Read the alert topic:

```powershell
python src/streaming/test_consumer.py --topic alerts --group-id wildfire-alert-check --max-messages 1
```

For host-local Linux/macOS runs, this also works after `pip install -r requirements.txt`:

```powershell
python src/streaming/spark_streaming_job.py
```

### 14. Airflow Orchestration

Airflow runs the scheduled pipeline DAGs with the web UI on:

```text
http://localhost:8081
```

Default local login:

```text
admin / admin
```

Start Airflow with the backing services:

```powershell
docker compose up -d minio create-bucket spark-master spark-worker kafka kafka-init airflow
```

The compose service builds `docker/airflow/Dockerfile` from
`apache/airflow:2.8.0` with Java for `spark-submit`, then installs PySpark, the
Spark provider, and runtime Python dependencies at container startup. It mounts
`./dags`, `./src`, and the project root, and configures `spark_default` to
submit jobs to `spark://spark-master:7077`.

Available DAGs:

- `fire_ingestion_dag`: every 3 hours, fetches recent FIRMS detections, publishes
  to Kafka `fire-events`, and archives the same batch to MinIO.
- `weather_update_dag`: daily, fetches weather for the execution date, then runs
  Spark cleaning and feature refresh.
- `model_retrain_dag`: weekly, runs Spark cleaning, feature engineering, model
  training, data quality/heatmap, DBSCAN clustering, anomaly detection, and
  next-day inference.

Useful checks:

```powershell
docker compose exec airflow airflow dags list
docker compose exec airflow airflow dags trigger fire_ingestion_dag
docker compose exec airflow airflow dags trigger weather_update_dag
docker compose exec airflow airflow dags trigger model_retrain_dag
```

`fire_ingestion_dag` needs `FIRMS_MAP_KEY` or `MAP_KEY` in the host environment
or `.env` before the Airflow container starts.

### 15. Apache Sedona Spatial ETL

The Sedona migration keeps spatial work inside Spark instead of using a
single-node GeoPandas join. It builds Sedona geometries with
`ST_Point`, `ST_PolygonFromEnvelope`, `ST_Contains`, and `ST_Intersects`, then
writes spatially clipped/grid-attached Parquet outputs back to MinIO.

Install local dependencies:

```powershell
pip install -r requirements.txt
```

Run the Sedona ETL locally against MinIO:

```powershell
python src/spatial/sedona_etl.py --print-counts
```

When running through the Docker Spark image, rebuild it once so the Sedona
Python package is available in the container:

```powershell
docker compose build spark-master spark-worker
```

Output prefixes:

- `s3a://wildfire-data/firms_sedona_clean/`
- `s3a://wildfire-data/weather_sedona_clean/`
- `s3a://wildfire-data/grid_sedona/`

Run the 1M-point spatial join benchmark:

```powershell
python scripts/benchmark_spatial_join.py --rows 1000000
```

Latest local benchmark report:

- Report: `reports/sedona_spatial_benchmark.md`
- CSV: `reports/sedona_spatial_benchmark.csv`
- Sedona: 4.672s for 1,000,000 points on `local[4]`
- GeoPandas/Shapely 2.x: 0.708s for the same simple rectangular-grid workload
- Status: Sedona is under 30s, but the 3x speedup criterion is not met on this
  local synthetic benchmark because Spark startup/execution overhead dominates.

## File Structure

```text
01_explore_firms.py              # FIRMS NRT quick fetch
02_fetch_weather.py              # Daily weather fetch/upload
03_fetch_firms_history.py        # FIRMS history fetch/upload
04_etl_clean.py                  # Spark cleaning ETL
05_feature_engineering.py        # Spark spatial join + ML features
06_data_quality_and_heatmap.py   # Data quality report + heatmap
07_train_model.py                # Spark MLlib RF tuning + GBT comparison
07_dbscan_clustering.py          # DBSCAN recent fire clusters + GeoJSON
08_anomaly_detection.py          # Grid-level z-score anomaly detection
09_inference.py                  # Open-Meteo next-day RF inference
app.py                           # Streamlit dashboard
geo_utils.py                     # GeoJSON point-in-polygon helpers
geo/vietnam_boundary.geojson     # Vietnam country boundary mask
docker-compose.yml               # Local MinIO + Spark + Kafka stack
docker/airflow/Dockerfile        # Airflow 2.8 image with Java for spark-submit
docker/spark/Dockerfile          # Spark image with numpy, Sedona, and Shapely
spark-conf/spark-defaults.conf   # S3A config for Spark
dags/                            # Airflow DAGs for ingestion, weather, retraining
src/orchestration/               # Airflow Python task helpers
src/spatial/                     # Apache Sedona spatial ETL
src/streaming/                   # Kafka producers, consumers, Spark streaming alerts
scripts/benchmark_spatial_join.py # GeoPandas vs Sedona benchmark
scripts/                         # Operational helper scripts
maps/                            # HTML visualizations
reports/                         # Data quality and model reports
reports/sedona_spatial_benchmark.md # Latest Sedona benchmark report
```

## Notes

- `data/raw/` is gitignored because it contains large local cache files.
- The FIRMS area API accepts a bounding box, so raw fetches can include neighboring countries. `04_etl_clean.py` is the step that clips records back to the Vietnam boundary.
- `spark-conf/spark-defaults.conf` configures `hadoop-aws` and S3A settings so Spark can read/write MinIO reliably.
- If Spark runs outside Docker, the MinIO endpoint should usually be `http://localhost:9000`. If Spark runs inside the Docker network, the endpoint should be `http://minio:9000`.
