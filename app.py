from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Wildfire Early Warning",
    layout="wide",
)


REPORTS_DIR = Path("reports")
RISK_GEOJSON_PATH = REPORTS_DIR / "fire_risk_forecast_latest.geojson"
RISK_METADATA_PATH = REPORTS_DIR / "fire_risk_forecast_latest.json"
DBSCAN_GEOJSON_PATH = REPORTS_DIR / "dbscan_fire_clusters_latest.geojson"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics_week1.json"
ANOMALY_STATS_PATH = REPORTS_DIR / "fire_anomaly_detector_stats.csv"

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
    "low": "#2ca25f",
    "medium": "#feb24c",
    "high": "#de2d26",
}


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
        return endpoint.removeprefix("https://"), "https"
    if endpoint.startswith("http://"):
        return endpoint.removeprefix("http://"), "http"
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
def load_anomaly_stats(path: str) -> pd.DataFrame:
    current_path = Path(path)
    if not current_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(current_path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


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
        tiles="CartoDB positron",
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


def app() -> None:
    st.title("Wildfire Early Warning Dashboard")

    risk_geojson_all = load_geojson(str(RISK_GEOJSON_PATH))
    risk_metadata = load_json(str(RISK_METADATA_PATH))
    clusters_geojson = load_geojson(str(DBSCAN_GEOJSON_PATH))
    model_metrics = load_json(str(MODEL_METRICS_PATH))
    anomaly_stats = load_anomaly_stats(str(ANOMALY_STATS_PATH))

    try:
        fires_all = load_fire_points()
        fire_error = None
    except Exception as exc:
        fires_all = pd.DataFrame()
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
        st.header("Controls")
        selected_day = st.date_input("Date", value=default_day)
        selected_region = st.selectbox("Region", list(REGIONS.keys()), index=0)
        min_confidence = st.slider("Min fire confidence", 0, 100, 30, 5)

        st.divider()
        st.caption("Model")
        st.write(f"Best model: `{model_metrics.get('best_model', 'n/a')}`")
        if risk_metadata:
            st.write(f"Forecast target: `{risk_metadata.get('target_date', 'n/a')}`")

    selected_day = pd.to_datetime(selected_day).date()
    risk_geojson = filter_risk_geojson(risk_geojson_all, selected_day, selected_region)
    risk_frame = risk_dataframe(risk_geojson)

    fires_region = filter_region(fires_all, selected_region)
    fires_day = fires_region[
        (fires_region.get("acq_date") == selected_day)
        & (fires_region.get("confidence_score", 0) >= min_confidence)
    ].copy() if not fires_region.empty else pd.DataFrame()

    tabs = st.tabs(["Risk Map", "Active Fires", "Statistics"])

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

        if fire_error is not None:
            st.warning(f"Could not load MinIO fire detections: {fire_error}")
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
                    color_discrete_map={"Low": "#2ca25f", "Medium": "#feb24c", "High": "#de2d26"},
                    title="Forecast Risk Levels",
                )
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk forecast statistics are unavailable for this date.")

        with stat_col2:
            if not fires_region.empty:
                daily_fires = (
                    fires_region.groupby("acq_date", as_index=False)
                    .size()
                    .rename(columns={"acq_date": "date", "size": "fire_points"})
                    .sort_values("date")
                )
                fig = px.line(daily_fires, x="date", y="fire_points", title="Daily Fire Detections")
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Fire history statistics are unavailable.")

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
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with metric_col2:
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
                )
                fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app()
