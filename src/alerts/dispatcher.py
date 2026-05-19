"""Format and dispatch wildfire alerts to Telegram and email channels."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


LOGGER = logging.getLogger(__name__)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _grid_to_latlon(grid_lat_index: int | None, grid_lon_index: int | None, grid_size: float = 0.5) -> tuple[float | None, float | None]:
    if grid_lat_index is None or grid_lon_index is None:
        return None, None
    lat = round((grid_lat_index + 0.5) * grid_size, 5)
    lon = round((grid_lon_index + 0.5) * grid_size, 5)
    return lat, lon


def normalize_alert(payload: dict[str, Any], grid_size: float = 0.5) -> dict[str, Any]:
    """Convert a raw Kafka alert into the dict consumed by templates and matchers.

    Supports two alert shapes:
      - HOT_ZONE: realtime hot-zones from Spark Structured Streaming (fire_count + window).
      - FORECAST_RISK: model-predicted risk per grid cell from 09_inference.
    """
    grid_lat_index = payload.get("grid_lat_index")
    grid_lon_index = payload.get("grid_lon_index")
    latitude, longitude = _grid_to_latlon(grid_lat_index, grid_lon_index, grid_size)

    # Allow forecast alerts to override location with their own lat/lon (already centered).
    if payload.get("latitude") is not None and payload.get("longitude") is not None:
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])

    alert_type = str(payload.get("alert_type") or "HOT_ZONE")
    severity_value = str(payload.get("severity") or payload.get("risk_label") or "high").lower()
    severity_rank = {"low": 0, "medium": 1, "high": 2}.get(severity_value, 2)

    map_image_url = build_mapbox_image_url(latitude, longitude)
    map_link = build_map_link(latitude, longitude)
    map_url = map_image_url or map_link

    risk_score = payload.get("risk_score")
    forecast_date = payload.get("forecast_date") or payload.get("date")

    return {
        "alert_id": str(payload.get("alert_id") or ""),
        "alert_type": alert_type,
        "severity_label": severity_value,
        "severity_rank": severity_rank,
        "grid_id": payload.get("grid_id") or f"{grid_lat_index}_{grid_lon_index}",
        "grid_lat_index": grid_lat_index,
        "grid_lon_index": grid_lon_index,
        "fire_count": int(payload.get("fire_count") or 0),
        "threshold": int(payload.get("threshold") or 0),
        "risk_score": float(risk_score) if isinstance(risk_score, (int, float)) else None,
        "forecast_date": forecast_date,
        "window_start": payload.get("window_start_utc") or payload.get("window_start"),
        "window_end": payload.get("window_end_utc") or payload.get("window_end"),
        "created_at": payload.get("created_at_utc") or payload.get("created_at"),
        "latitude": latitude,
        "longitude": longitude,
        "map_url": map_url,
        "map_image_url": map_image_url,
        "map_link": map_link,
        "raw": payload,
    }


def build_mapbox_image_url(latitude: float | None, longitude: float | None) -> str | None:
    """Static Mapbox image URL with a fire pin. Returns None when token missing."""
    if latitude is None or longitude is None:
        return None

    mapbox_token = os.getenv("MAPBOX_TOKEN")
    if not mapbox_token:
        return None

    style = os.getenv("MAPBOX_STYLE", "mapbox/satellite-streets-v12")
    zoom = os.getenv("MAPBOX_ZOOM", "9")
    size = os.getenv("MAPBOX_SIZE", "640x400")
    marker_name = os.getenv("MAPBOX_MARKER", "fire-station")
    marker_color = os.getenv("MAPBOX_MARKER_COLOR", "ff5722").lstrip("#")
    marker = f"pin-l-{marker_name}+{marker_color}({longitude},{latitude})"
    return (
        f"https://api.mapbox.com/styles/v1/{style}/static/"
        f"{marker}/{longitude},{latitude},{zoom}/{size}@2x?access_token={mapbox_token}"
    )


def build_map_link(latitude: float | None, longitude: float | None) -> str | None:
    """Browser-friendly map link used as fallback when no Mapbox token is set."""
    if latitude is None or longitude is None:
        return None
    return f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=10/{latitude}/{longitude}"


def fetch_map_image(map_image_url: str, *, timeout: float = 15.0) -> bytes | None:
    """Download the static Mapbox image so it can be sent as a Telegram photo or email attachment."""
    if not map_image_url:
        return None
    try:
        with urllib.request.urlopen(map_image_url, timeout=timeout) as response:
            content = response.read()
        if not content:
            return None
        return content
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to download Mapbox image: %s", map_image_url)
        return None


def build_map_url(latitude: float | None, longitude: float | None) -> str | None:
    """Backward-compatible URL used in templates: prefer Mapbox image, else OSM link."""
    return build_mapbox_image_url(latitude, longitude) or build_map_link(latitude, longitude)


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html",)),
        keep_trailing_newline=True,
    )


def render_email(alert: dict[str, Any]) -> tuple[str, str, str]:
    env = _jinja_env()
    subject = f"Wildfire alert ({alert['severity_label'].upper()}): grid {alert['grid_id']}"
    text_body = env.get_template("email_alert.txt").render(alert=alert)
    html_body = env.get_template("email_alert.html").render(alert=alert)
    return subject, text_body, html_body


def render_telegram(alert: dict[str, Any]) -> str:
    env = _jinja_env()
    return env.get_template("telegram_alert.txt").render(alert=alert)


def render_telegram_caption(alert: dict[str, Any]) -> str:
    env = _jinja_env()
    return env.get_template("telegram_caption.txt").render(alert=alert)


def send_telegram(chat_id: str, text: str, *, bot_token: str | None = None, timeout: float = 10.0) -> None:
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "false",
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        payload = json.loads(body)
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")


def _telegram_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----wildfire{int.from_bytes(os.urandom(8), 'big'):x}"
    eol = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))
    for name, (filename, content, mime_type) in files.items():
        parts.append(f"--{boundary}".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {mime_type}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = eol.join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def send_telegram_photo(
    chat_id: str,
    photo: bytes,
    caption: str | None = None,
    *,
    filename: str = "alert.png",
    bot_token: str | None = None,
    timeout: float = 20.0,
) -> None:
    """Upload a static map image as a Telegram photo with optional Markdown caption."""
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    fields: dict[str, str] = {"chat_id": chat_id}
    if caption is not None:
        # Telegram captions are limited to 1024 chars.
        fields["caption"] = caption[:1024]
        fields["parse_mode"] = "HTML"

    body, content_type = _telegram_multipart(
        fields,
        {"photo": (filename, photo, "image/png")},
    )

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API error (sendPhoto): {payload}")


def send_email(
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
    *,
    inline_images: dict[str, bytes] | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_username: str | None = None,
    smtp_password: str | None = None,
    smtp_use_tls: bool | None = None,
    sender: str | None = None,
    timeout: float = 15.0,
) -> None:
    host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    username = smtp_username or os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    password = smtp_password or os.getenv("SMTP_PASSWORD")
    use_tls = smtp_use_tls if smtp_use_tls is not None else os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    sender_address = sender or os.getenv("SMTP_SENDER") or username

    if not username or not password:
        raise RuntimeError("SMTP credentials are not configured (SMTP_USERNAME/SMTP_PASSWORD)")
    if not sender_address:
        raise RuntimeError("SMTP_SENDER (or SMTP_USERNAME) must be set")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_address
    message["To"] = recipient
    message.set_content(text_body)
    if html_body:
        message.add_alternative(html_body, subtype="html")
        if inline_images:
            html_part = message.get_payload()[-1]
            for cid, content in inline_images.items():
                html_part.add_related(
                    content,
                    maintype="image",
                    subtype="png",
                    cid=f"<{cid}>",
                    filename=f"{cid}.png",
                )

    with smtplib.SMTP(host, port, timeout=timeout) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(message)
