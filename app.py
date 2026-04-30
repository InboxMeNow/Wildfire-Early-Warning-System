from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Wildfire Early Warning",
    layout="wide",
)


REPORTS_DIR = Path("reports")
DATA_DIR = Path("data") / "raw"
RISK_GEOJSON_PATH = REPORTS_DIR / "fire_risk_forecast_latest.geojson"
RISK_METADATA_PATH = REPORTS_DIR / "fire_risk_forecast_latest.json"
DBSCAN_GEOJSON_PATH = REPORTS_DIR / "dbscan_fire_clusters_latest.geojson"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics_week1.json"
ANOMALY_STATS_PATH = REPORTS_DIR / "fire_anomaly_detector_stats.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance_week1.csv"
HISTORICAL_FIRES_PATH = DATA_DIR / "firms_history_vietnam_2020_2024.parquet"
RECENT_FIRES_PATH = DATA_DIR / "firms_viirs_noaa20_vietnam_last5days.csv"

DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"

REGIONS = {
    "Vietnam": {
        "center": [15.9, 106.2],
        "zoom": 5,
        "bbox": (8.0, 24.5, 102.0, 110.8),
    },
    "SE Asia": {
        "center": [13.5, 104.5],
        "zoom": 4,
        "bbox": (-5.0, 28.0, 90.0, 125.0),
    },
    "All area": {
        "center": [15.0, 106.0],
        "zoom": 4,
        "bbox": (-15.0, 35.0, 85.0, 130.0),
    },
}

RISK_COLORS = {
    "low": "#22c55e",
    "medium": "#f59e0b",
    "high": "#ef4444",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#e5e7eb"),
    title_font=dict(color="#f8fafc"),
    xaxis=dict(gridcolor="rgba(148, 163, 184, 0.18)", zerolinecolor="rgba(148, 163, 184, 0.20)"),
    yaxis=dict(gridcolor="rgba(148, 163, 184, 0.18)", zerolinecolor="rgba(148, 163, 184, 0.20)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=12, r=12, t=54, b=18),
)


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


def normalize_endpoint(endpoint: str) -> tuple[str, str]:
    if endpoint.startswith("https://"):
        return endpoint[len("https://") :], "https"
    if endpoint.startswith("http://"):
        return endpoint[len("http://") :], "http"
    return endpoint, "http"


def minio_settings() -> dict[str, str]:
    load_env_file()
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT),
        "access_key": os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY),
        "secret_key": os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY),
        "bucket": os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET),
    }


def s3_filesystem(settings: dict[str, str]) -> pafs.S3FileSystem:
    endpoint, scheme = normalize_endpoint(settings["endpoint"])
    return pafs.S3FileSystem(
        access_key=settings["access_key"],
        secret_key=settings["secret_key"],
        endpoint_override=endpoint,
        scheme=scheme,
        region="us-east-1",
    )


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict[str, object]:
    current_path = Path(path)
    if not current_path.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(current_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, object]:
    current_path = Path(path)
    if not current_path.exists():
        return {}
    return json.loads(current_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    current_path = Path(path)
    if not current_path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.read_csv(current_path)
    if {"feature", "importance"}.issubset(frame.columns):
        frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0.0)
        frame = frame.sort_values("importance", ascending=True)
    return frame


@st.cache_data(show_spinner="Loading historical fire trend...")
def load_monthly_fire_trends(path: str, region: str) -> pd.DataFrame:
    current_path = Path(path)
    if not current_path.exists():
        return pd.DataFrame()

    dataset = ds.dataset(str(current_path), format="parquet")
    columns = ["latitude", "longitude", "acq_date", "frp"]
    available_columns = [name for name in columns if name in dataset.schema.names]
    if not {"latitude", "longitude", "acq_date"}.issubset(available_columns):
        return pd.DataFrame()

    lat_min, lat_max, lon_min, lon_max = REGIONS[region]["bbox"]
    chunks = []
    scanner = dataset.scanner(columns=available_columns, batch_size=250_000)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame = frame[
            frame["latitude"].between(lat_min, lat_max)
            & frame["longitude"].between(lon_min, lon_max)
        ].copy()
        if frame.empty:
            continue

        frame["month"] = pd.to_datetime(frame["acq_date"]).dt.to_period("M").dt.to_timestamp()
        if "frp" in frame.columns:
            frame["frp"] = pd.to_numeric(frame["frp"], errors="coerce").fillna(0.0)
            chunk = frame.groupby("month", as_index=False).agg(
                fire_points=("acq_date", "size"),
                frp_total=("frp", "sum"),
            )
        else:
            chunk = frame.groupby("month", as_index=False).agg(fire_points=("acq_date", "size"))
            chunk["frp_total"] = 0.0
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    monthly = pd.concat(chunks, ignore_index=True)
    monthly = monthly.groupby("month", as_index=False).agg(
        fire_points=("fire_points", "sum"),
        frp_total=("frp_total", "sum"),
    )
    monthly["avg_frp"] = monthly["frp_total"] / monthly["fire_points"].clip(lower=1)
    return monthly.sort_values("month")


@st.cache_data(show_spinner=False)
def load_dataset_info(history_path: str) -> dict[str, object]:
    info: dict[str, object] = {
        "history_rows": None,
        "history_date_min": None,
        "history_date_max": None,
    }
    current_path = Path(history_path)
    if not current_path.exists():
        return info

    try:
        parquet_file = pq.ParquetFile(current_path)
        info["history_rows"] = parquet_file.metadata.num_rows
        date_mins = []
        date_maxes = []
        schema_names = parquet_file.schema.names
        if "acq_date" in schema_names:
            column_index = schema_names.index("acq_date")
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                stats = parquet_file.metadata.row_group(row_group_index).column(column_index).statistics
                if stats and stats.has_min_max:
                    date_mins.append(pd.to_datetime(stats.min).date())
                    date_maxes.append(pd.to_datetime(stats.max).date())
        if date_mins and date_maxes:
            info["history_date_min"] = min(date_mins).isoformat()
            info["history_date_max"] = max(date_maxes).isoformat()
    except Exception:
        return info
    return info


@st.cache_data(show_spinner="Loading fire detections...")
def load_fire_points() -> pd.DataFrame:
    settings = minio_settings()
    dataset = ds.dataset(
        f"{settings['bucket']}/firms_clean",
        filesystem=s3_filesystem(settings),
        format="parquet",
    )
    columns = ["latitude", "longitude", "acq_date", "acq_time", "confidence_score", "frp", "satellite"]
    available_columns = [name for name in columns if name in dataset.schema.names]
    table = dataset.to_table(columns=available_columns)
    frame = table.to_pandas()
    if frame.empty:
        return frame

    frame["acq_date"] = pd.to_datetime(frame["acq_date"]).dt.date
    frame["confidence_score"] = pd.to_numeric(frame.get("confidence_score", 0), errors="coerce").fillna(0.0)
    frame["frp"] = pd.to_numeric(frame.get("frp", 0), errors="coerce").fillna(0.0)
    if "acq_time" in frame.columns:
        frame["acq_time"] = pd.to_numeric(frame["acq_time"], errors="coerce").fillna(0).astype(int)
    return frame


@st.cache_data(show_spinner=False)
def load_recent_fire_points(path: str) -> pd.DataFrame:
    current_path = Path(path)
    if not current_path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(current_path)
    if frame.empty:
        return frame

    frame["acq_date"] = pd.to_datetime(frame["acq_date"]).dt.date
    frame["frp"] = pd.to_numeric(frame.get("frp", 0), errors="coerce").fillna(0.0)
    confidence_map = {"l": 30, "n": 60, "h": 90}
    if "confidence_score" not in frame.columns:
        frame["confidence_score"] = (
            frame.get("confidence", pd.Series(index=frame.index, dtype="object"))
            .astype(str)
            .str.lower()
            .map(confidence_map)
            .fillna(0)
        )
    if "acq_time" in frame.columns:
        frame["acq_time"] = pd.to_numeric(frame["acq_time"], errors="coerce").fillna(0).astype(int)
    return frame


@st.cache_data(show_spinner=False)
def load_anomaly_stats(path: str) -> pd.DataFrame:
    current_path = Path(path)
    if not current_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(current_path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1120;
            --panel: #111827;
            --panel-2: #172033;
            --border: rgba(248, 250, 252, 0.10);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #f97316;
            --danger: #ef4444;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(239, 68, 68, 0.18), transparent 28%),
                linear-gradient(180deg, #0b1120 0%, #111827 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080d19 0%, #111827 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(31, 41, 55, 0.96), rgba(17, 24, 39, 0.96));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.85rem;
        }
        [data-testid="stMetricValue"] {
            color: #fff7ed;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: #111827;
            border: 1px solid var(--border);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            color: #cbd5e1;
            height: 42px;
            padding: 0 14px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(249, 115, 22, 0.24), rgba(239, 68, 68, 0.16));
            color: #ffffff;
            border-color: rgba(249, 115, 22, 0.55);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .dashboard-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 18px 20px;
            margin: 0 0 18px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(127, 29, 29, 0.62));
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .logo-mark {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            background: linear-gradient(135deg, #f97316, #dc2626);
            color: #ffffff;
            font-weight: 800;
            letter-spacing: 0;
            box-shadow: 0 12px 28px rgba(239, 68, 68, 0.24);
        }
        .dashboard-title {
            margin: 0;
            color: #ffffff;
            font-size: 1.72rem;
            line-height: 1.15;
            letter-spacing: 0;
        }
        .dashboard-subtitle {
            margin: 4px 0 0;
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .status-pill {
            border: 1px solid rgba(249, 115, 22, 0.55);
            border-radius: 999px;
            color: #fed7aa;
            padding: 7px 12px;
            background: rgba(249, 115, 22, 0.12);
            white-space: nowrap;
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 12px 0 8px;
        }
        .footer {
            color: #94a3b8;
            border-top: 1px solid var(--border);
            padding: 16px 0 4px;
            margin-top: 26px;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(risk_metadata: dict[str, object]) -> None:
    target = risk_metadata.get("target_date", "n/a")
    status = str(risk_metadata.get("status", "offline")).upper()
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div class="brand-lockup">
                <div class="logo-mark">WF</div>
                <div>
                    <h1 class="dashboard-title">Wildfire Early Warning</h1>
                    <p class="dashboard-subtitle">Vietnam fire-risk monitoring, model diagnostics, and daily alerts</p>
                </div>
            </div>
            <div class="status-pill">{status} forecast · {target}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_region(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    if frame.empty or "latitude" not in frame or "longitude" not in frame:
        return frame

    lat_min, lat_max, lon_min, lon_max = REGIONS[region]["bbox"]
    return frame[
        frame["latitude"].between(lat_min, lat_max)
        & frame["longitude"].between(lon_min, lon_max)
    ].copy()


def filter_risk_geojson(geojson: dict[str, object], selected_day: date, region: str) -> dict[str, object]:
    lat_min, lat_max, lon_min, lon_max = REGIONS[region]["bbox"]
    features = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        feature_date = pd.to_datetime(properties.get("date")).date()
        lat = float(properties.get("lat", properties.get("grid_lat", 0)))
        lon = float(properties.get("lon", properties.get("grid_lon", 0)))
        if feature_date == selected_day and lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def risk_style(feature: dict[str, object]) -> dict[str, object]:
    label = feature.get("properties", {}).get("risk_label", "low")
    color = RISK_COLORS.get(str(label), RISK_COLORS["low"])
    score = float(feature.get("properties", {}).get("risk_score", 0.0))
    return {
        "fillColor": color,
        "color": "#1f2937",
        "weight": 0.7,
        "fillOpacity": min(0.78, 0.22 + score * 0.7),
    }


def make_base_map(region: str) -> folium.Map:
    details = REGIONS[region]
    return folium.Map(
        location=details["center"],
        zoom_start=details["zoom"],
        tiles="CartoDB dark_matter",
        control_scale=True,
    )


def render_risk_map(risk_geojson: dict[str, object], region: str) -> None:
    fmap = make_base_map(region)
    features = risk_geojson.get("features", [])
    if features:
        folium.GeoJson(
            risk_geojson,
            name="Risk forecast",
            style_function=risk_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "grid_id",
                    "risk_label",
                    "risk_score",
                    "temperature_2m_max",
                    "relative_humidity_2m_min",
                    "wind_speed_10m_max",
                    "precipitation_sum",
                ],
                aliases=[
                    "Grid",
                    "Risk",
                    "Score",
                    "Temp max",
                    "Humidity min",
                    "Wind max",
                    "Precipitation",
                ],
                localize=True,
            ),
        ).add_to(fmap)

        heat_points = [
            [
                feature["properties"]["lat"],
                feature["properties"]["lon"],
                feature["properties"]["risk_score"],
            ]
            for feature in features
        ]
        HeatMap(
            heat_points,
            name="Risk heatmap",
            radius=32,
            blur=24,
            min_opacity=0.18,
            max_zoom=8,
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    st_folium(fmap, height=680, use_container_width=True)


def render_active_fire_map(fires: pd.DataFrame, clusters: dict[str, object], region: str) -> None:
    fmap = make_base_map(region)

    if not fires.empty:
        for row in fires.itertuples(index=False):
            confidence = float(getattr(row, "confidence_score", 0.0))
            frp = float(getattr(row, "frp", 0.0))
            folium.CircleMarker(
                location=[float(row.latitude), float(row.longitude)],
                radius=max(3, min(8, 3 + frp / 15)),
                color="#7f1d1d",
                weight=1,
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.78,
                tooltip=f"Confidence: {confidence:.0f} | FRP: {frp:.2f}",
            ).add_to(fmap)

    if clusters.get("features"):
        folium.GeoJson(
            clusters,
            name="DBSCAN clusters",
            style_function=lambda _: {
                "fillColor": "#f97316",
                "color": "#9a3412",
                "weight": 2,
                "fillOpacity": 0.22,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["cluster_id", "point_count", "mean_confidence_score", "max_frp"],
                aliases=["Cluster", "Points", "Mean confidence", "Max FRP"],
                localize=True,
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    st_folium(fmap, height=680, use_container_width=True)


def metric_card(label: str, value: object) -> None:
    st.metric(label, value)


def risk_dataframe(risk_geojson: dict[str, object]) -> pd.DataFrame:
    rows = []
    for feature in risk_geojson.get("features", []):
        props = feature.get("properties", {})
        rows.append(props)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def top_alerts(risk_frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if risk_frame.empty:
        return pd.DataFrame()

    columns = [
        "grid_id",
        "risk_label",
        "risk_score",
        "lat",
        "lon",
        "temperature_2m_max",
        "relative_humidity_2m_min",
        "wind_speed_10m_max",
        "precipitation_sum",
        "dry_days_count",
    ]
    available = [column for column in columns if column in risk_frame.columns]
    alerts = risk_frame.sort_values(["risk_score", "risk_level"], ascending=False).head(limit)[available].copy()
    rename_map = {
        "grid_id": "Grid",
        "risk_label": "Risk",
        "risk_score": "Score",
        "lat": "Lat",
        "lon": "Lon",
        "temperature_2m_max": "Temp max",
        "relative_humidity_2m_min": "Humidity min",
        "wind_speed_10m_max": "Wind max",
        "precipitation_sum": "Rain",
        "dry_days_count": "Dry days",
    }
    alerts = alerts.rename(columns=rename_map)
    if "Risk" in alerts:
        alerts["Risk"] = alerts["Risk"].astype(str).str.title()
    for column in ["Score", "Lat", "Lon", "Temp max", "Humidity min", "Wind max", "Rain"]:
        if column in alerts:
            alerts[column] = pd.to_numeric(alerts[column], errors="coerce").round(3)
    return alerts


def render_confusion_matrix(model_metrics: dict[str, object], model_name: str) -> None:
    metrics = model_metrics.get(model_name, {})
    matrix = [
        [metrics.get("true_negative", 0), metrics.get("false_positive", 0)],
        [metrics.get("false_negative", 0), metrics.get("true_positive", 0)],
    ]
    fig = px.imshow(
        matrix,
        labels=dict(x="Predicted", y="Actual", color="Rows"),
        x=["No fire", "Fire"],
        y=["No fire", "Fire"],
        text_auto=True,
        color_continuous_scale=["#111827", "#f97316", "#ef4444"],
        title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
    )
    plotly_theme(fig)
    fig.update_traces(textfont=dict(color="#ffffff", size=16))
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(feature_importance: pd.DataFrame) -> None:
    if feature_importance.empty:
        st.info("Feature importance file is unavailable.")
        return

    fig = px.bar(
        feature_importance.tail(16),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#facc15", "#f97316", "#ef4444"],
        title="RandomForest Feature Importance",
    )
    plotly_theme(fig)
    fig.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)


def app() -> None:
    apply_theme()

    risk_geojson_all = load_geojson(str(RISK_GEOJSON_PATH))
    risk_metadata = load_json(str(RISK_METADATA_PATH))
    clusters_geojson = load_geojson(str(DBSCAN_GEOJSON_PATH))
    model_metrics = load_json(str(MODEL_METRICS_PATH))
    anomaly_stats = load_anomaly_stats(str(ANOMALY_STATS_PATH))
    feature_importance = load_feature_importance(str(FEATURE_IMPORTANCE_PATH))
    dataset_info = load_dataset_info(str(HISTORICAL_FIRES_PATH))

    render_header(risk_metadata)

    try:
        fires_all = load_fire_points()
        fire_source = "MinIO firms_clean"
        fire_error = None
    except Exception as exc:
        fires_all = load_recent_fire_points(str(RECENT_FIRES_PATH))
        fire_source = "Local recent FIRMS CSV" if not fires_all.empty else "Unavailable"
        fire_error = exc

    risk_all = risk_dataframe(risk_geojson_all)
    available_dates = sorted(
        {
            *risk_all.get("date", pd.Series(dtype="object")).dropna().tolist(),
            *fires_all.get("acq_date", pd.Series(dtype="object")).dropna().tolist(),
        }
    )
    default_day = pd.to_datetime(risk_metadata.get("target_date")).date() if risk_metadata.get("target_date") else None
    if default_day not in available_dates:
        default_day = available_dates[-1] if available_dates else date.today()

    with st.sidebar:
        st.markdown("### Controls")
        selected_day = st.date_input("Date", value=default_day)
        selected_region = st.selectbox("Region", list(REGIONS.keys()), index=0)
        min_confidence = st.slider("Min fire confidence", 0, 100, 30, 5)

        st.divider()
        st.markdown("### Dataset")
        if dataset_info.get("history_rows"):
            st.metric("Historical FIRMS rows", f"{int(dataset_info['history_rows']):,}")
        st.caption(
            f"History window: `{dataset_info.get('history_date_min', 'n/a')}` to "
            f"`{dataset_info.get('history_date_max', 'n/a')}`"
        )
        st.caption(f"Forecast grids: `{risk_metadata.get('grid_count', len(risk_all))}`")
        st.caption(f"Active fire source: `{fire_source}`")

        st.divider()
        st.markdown("### Model")
        st.caption(f"Best model: `{model_metrics.get('best_model', 'n/a')}`")
        if risk_metadata:
            st.caption(f"Forecast target: `{risk_metadata.get('target_date', 'n/a')}`")
        st.caption(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M')}`")

    selected_day = pd.to_datetime(selected_day).date()
    risk_geojson = filter_risk_geojson(risk_geojson_all, selected_day, selected_region)
    risk_frame = risk_dataframe(risk_geojson)

    fires_region = filter_region(fires_all, selected_region)
    fires_day = fires_region[
        (fires_region.get("acq_date") == selected_day)
        & (fires_region.get("confidence_score", 0) >= min_confidence)
    ].copy() if not fires_region.empty else pd.DataFrame()

    tabs = st.tabs(
        [
            "Risk Map",
            "Active Fires",
            "Statistics",
            "Historical Trends",
            "Model Insights",
            "Alerts",
        ]
    )

    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Cells", len(risk_frame))
        with col2:
            metric_card("Avg risk", f"{risk_frame['risk_score'].mean():.3f}" if not risk_frame.empty else "n/a")
        with col3:
            metric_card("High risk", int((risk_frame.get("risk_level", pd.Series(dtype=int)) == 2).sum()))
        with col4:
            metric_card("Forecast date", selected_day.isoformat())

        if risk_frame.empty:
            st.warning("No forecast risk cells are available for the selected date and region.")
        render_risk_map(risk_geojson, selected_region)

    with tabs[1]:
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Active fire points", len(fires_day))
        with col2:
            metric_card("DBSCAN clusters", len(clusters_geojson.get("features", [])))
        with col3:
            metric_card("Mean FRP", f"{fires_day['frp'].mean():.2f}" if not fires_day.empty else "n/a")

        if fire_error is not None and fires_all.empty:
            st.warning(f"Could not load MinIO fire detections: {fire_error}")
        elif fire_error is not None:
            st.info("MinIO is unavailable, so Active Fires is using the local FIRMS CSV fallback.")
        elif fires_day.empty:
            st.info("No active fire detections matched the current date, region, and confidence filter.")
        render_active_fire_map(fires_day.head(1500), clusters_geojson, selected_region)

    with tabs[2]:
        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            if not risk_frame.empty:
                risk_counts = (
                    risk_frame.assign(risk_label=risk_frame["risk_label"].str.title())
                    .groupby("risk_label", as_index=False)
                    .size()
                    .rename(columns={"size": "cells"})
                )
                fig = px.bar(
                    risk_counts,
                    x="risk_label",
                    y="cells",
                    color="risk_label",
                    color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                    title="Forecast Risk Levels",
                )
                plotly_theme(fig)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk forecast statistics are unavailable for this date.")

        with stat_col2:
            if not anomaly_stats.empty and "is_anomaly" in anomaly_stats:
                anomaly_counts = (
                    anomaly_stats.assign(is_anomaly=anomaly_stats["is_anomaly"].astype(str))
                    .groupby("is_anomaly", as_index=False)
                    .size()
                    .rename(columns={"size": "grid_days"})
                )
                fig = px.pie(
                    anomaly_counts,
                    names="is_anomaly",
                    values="grid_days",
                    title="Latest Anomaly Detector Output",
                    hole=0.45,
                    color_discrete_sequence=["#f97316", "#ef4444", "#22c55e"],
                )
                plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Anomaly detector statistics are unavailable.")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            model_rows = []
            for model_name in ["random_forest", "gbt"]:
                metrics = model_metrics.get(model_name, {})
                if metrics:
                    model_rows.append(
                        {
                            "model": model_name,
                            "auc_roc": metrics.get("auc_roc"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "f1": metrics.get("f1"),
                        }
                    )
            if model_rows:
                model_frame = pd.DataFrame(model_rows)
                fig = px.bar(
                    model_frame,
                    x="model",
                    y=["auc_roc", "precision", "recall", "f1"],
                    barmode="group",
                    title="Model Metrics",
                    color_discrete_sequence=["#f97316", "#ef4444", "#facc15", "#38bdf8"],
                )
                plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with metric_col2:
            if not risk_frame.empty:
                weather_summary = risk_frame[
                    [
                        "temperature_2m_max",
                        "relative_humidity_2m_min",
                        "wind_speed_10m_max",
                        "precipitation_sum",
                    ]
                ].mean(numeric_only=True)
                summary_frame = weather_summary.rename(
                    {
                        "temperature_2m_max": "Temp max",
                        "relative_humidity_2m_min": "Humidity min",
                        "wind_speed_10m_max": "Wind max",
                        "precipitation_sum": "Rain",
                    }
                ).reset_index()
                summary_frame.columns = ["variable", "average"]
                fig = px.bar(
                    summary_frame,
                    x="variable",
                    y="average",
                    title="Weather Averages In Selected Risk Cells",
                    color="average",
                    color_continuous_scale=["#38bdf8", "#f97316", "#ef4444"],
                )
                plotly_theme(fig)
                fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Average")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        monthly_trends = load_monthly_fire_trends(str(HISTORICAL_FIRES_PATH), selected_region)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Historical points", f"{monthly_trends['fire_points'].sum():,}" if not monthly_trends.empty else "n/a")
        with col2:
            metric_card("Peak month", monthly_trends.loc[monthly_trends["fire_points"].idxmax(), "month"].strftime("%Y-%m") if not monthly_trends.empty else "n/a")
        with col3:
            metric_card("Monthly avg", f"{monthly_trends['fire_points'].mean():,.0f}" if not monthly_trends.empty else "n/a")

        if monthly_trends.empty:
            st.info("Historical fire trend data is unavailable.")
        else:
            fig = px.line(
                monthly_trends,
                x="month",
                y="fire_points",
                markers=True,
                title="Monthly Fire Detections",
            )
            fig.update_traces(line=dict(color="#f97316", width=3), marker=dict(color="#ef4444", size=6))
            plotly_theme(fig)
            fig.update_layout(xaxis_title=None, yaxis_title="Fire points")
            st.plotly_chart(fig, use_container_width=True)

            annual = monthly_trends.assign(year=monthly_trends["month"].dt.year)
            annual = annual.groupby("year", as_index=False)["fire_points"].sum()
            fig = px.bar(
                annual,
                x="year",
                y="fire_points",
                title="Annual Fire Detections",
                color="fire_points",
                color_continuous_scale=["#facc15", "#f97316", "#ef4444"],
            )
            plotly_theme(fig)
            fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title="Fire points")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        insight_col1, insight_col2 = st.columns([1.25, 1])
        model_options = [
            model_name for model_name in ["gbt", "random_forest"] if model_metrics.get(model_name)
        ]
        default_model = model_metrics.get("best_model", model_options[0] if model_options else "")
        if model_options:
            selected_model = st.radio(
                "Model",
                model_options,
                index=model_options.index(default_model) if default_model in model_options else 0,
                horizontal=True,
            )
        else:
            selected_model = ""

        with insight_col1:
            render_feature_importance(feature_importance)

        with insight_col2:
            if selected_model:
                selected_metrics = model_metrics.get(selected_model, {})
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    metric_card("AUC ROC", f"{selected_metrics.get('auc_roc', 0):.3f}")
                with m2:
                    metric_card("Precision", f"{selected_metrics.get('precision', 0):.3f}")
                with m3:
                    metric_card("Recall", f"{selected_metrics.get('recall', 0):.3f}")
                with m4:
                    metric_card("F1", f"{selected_metrics.get('f1', 0):.3f}")
                render_confusion_matrix(model_metrics, selected_model)
            else:
                st.info("Model metrics are unavailable.")

    with tabs[5]:
        alerts = top_alerts(risk_frame, limit=10)
        a1, a2, a3 = st.columns(3)
        with a1:
            metric_card("Top alert cells", len(alerts))
        with a2:
            metric_card("Max risk score", f"{alerts['Score'].max():.3f}" if not alerts.empty and "Score" in alerts else "n/a")
        with a3:
            metric_card("High risk cells", int((risk_frame.get("risk_level", pd.Series(dtype=int)) == 2).sum()))

        if alerts.empty:
            st.info("No alert cells are available for the selected date and region.")
        else:
            st.markdown('<div class="section-title">Top 10 Highest-Risk Grid Cells Today</div>', unsafe_allow_html=True)
            st.dataframe(
                alerts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    )
                },
            )

            fig = px.bar(
                alerts.sort_values("Score"),
                x="Score",
                y="Grid",
                orientation="h",
                color="Risk",
                color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
                title="Alert Ranking",
            )
            plotly_theme(fig)
            fig.update_layout(xaxis_title="Risk score", yaxis_title=None, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="footer">Wildfire Early Warning · Local Streamlit dashboard · Data sources: FIRMS, weather features, Spark ML reports</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    app()
