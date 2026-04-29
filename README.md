# Wildfire Early Warning System

Pipeline du lieu tuan 1 cho bai toan canh bao nguy co chay rung: thu thap NASA FIRMS, thu thap weather, lam sach bang Spark tren MinIO, feature engineering theo grid 0.5 do, va tao visualization so bo.

## Kien Truc

- `MinIO`: object storage local, bucket mac dinh `wildfire-data`
- `Spark`: xu ly ETL bang PySpark, doc/ghi MinIO qua `s3a://`
- `NASA FIRMS`: diem chay lich su Vietnam 2020-2024
- `Meteostat/Open-Meteo`: weather daily theo grid Vietnam

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
```

MinIO console:

- API: `http://localhost:9000`
- Console: `http://localhost:9001`
- User/password: `minioadmin` / `minioadmin`

Spark UI:

- Master: `http://localhost:8080`
- Worker: `http://localhost:8081`

## Cau Hinh

Tao file `.env` neu can fetch lai FIRMS:

```text
FIRMS_MAP_KEY=your_firms_map_key
```

Gia tri MinIO mac dinh trong cac script:

```text
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=wildfire-data
```

Khi chay trong Docker network, `docker-compose.yml` tu dong set `MINIO_ENDPOINT=http://minio:9000` cho Spark.

## Pipeline Tuan 1

### 1. Kham Pha FIRMS NRT

```powershell
python 01_explore_firms.py
```

Output local:

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

### 4. Clean ETL Bang Spark

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/04_etl_clean.py
```

Xu ly:

- Doc `s3a://wildfire-data/firms/` va `s3a://wildfire-data/weather/`
- Filter FIRMS `confidence >= 30`
- Convert date columns
- Drop duplicates
- Ghi `s3a://wildfire-data/firms_clean/` va `s3a://wildfire-data/weather_clean/`

### 5. Feature Engineering

```powershell
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /workspace/05_feature_engineering.py --print-counts
```

Output:

- MinIO: `s3a://wildfire-data/features/`

Moi dong trong dataset cuoi la mot `(grid_id, date)`:

- Grid cell: `floor(latitude / 0.5)`, `floor(longitude / 0.5)`
- Fire aggregate: `fire_count`
- Weather features: temperature, humidity, wind, precipitation
- Rolling features: `precipitation_sum_7days`, `dry_days_count`
- Labels: `fire_occurred`, `risk_level`

### 6. Data Quality Va Heatmap

```powershell
python 06_data_quality_and_heatmap.py
```

Output:

- Report: `reports/data_quality_week1.md`
- Heatmap: `maps/fires_heatmap_2020_2024.html`

## Ket Qua Hien Tai

Tu lan chay gan nhat:

- `firms_clean`: 8,044,822 rows
- `weather_clean`: 3,071,187 rows
- `features`: 3,071,187 rows, 1,681 grid cells, 20 columns
- Date range: `2020-01-01` den `2024-12-31`
- Data quality: PASS, khong co duplicate `(grid_id, date)`, labels khop voi `fire_count`

Label distribution:

- `risk_level = 0`: 2,549,298 rows
- `risk_level = 1`: 313,441 rows
- `risk_level = 2`: 208,448 rows

## Cau Truc File

```text
01_explore_firms.py              # FIRMS NRT quick fetch
02_fetch_weather.py              # Weather daily fetch/upload
03_fetch_firms_history.py        # FIRMS history fetch/upload
04_etl_clean.py                  # Spark clean ETL
05_feature_engineering.py        # Spark spatial join + ML features
06_data_quality_and_heatmap.py   # DQ report + heatmap
docker-compose.yml               # MinIO + Spark local stack
spark-conf/spark-defaults.conf   # S3A config for Spark
maps/                            # HTML visualizations
reports/                         # Data quality reports
```

## Ghi Chu

- Thu muc `data/raw/` duoc gitignore vi chua du lieu cache lon.
- `spark-conf/spark-defaults.conf` khai bao `hadoop-aws` va S3A settings de Spark doc/ghi MinIO on dinh.
- Neu chay Spark ngoai Docker, endpoint MinIO nen la `http://localhost:9000`; neu chay trong Docker network, endpoint nen la `http://minio:9000`.
