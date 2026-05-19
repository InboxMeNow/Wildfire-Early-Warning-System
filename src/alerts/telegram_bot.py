"""Telegram bot for managing wildfire alert subscriptions.

Implements long-polling against the Telegram Bot HTTP API so the project does
not need to add the python-telegram-bot dependency footprint. Supports the
commands /start, /region, /severity, /unsubscribe, /status.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import db


LOGGER = logging.getLogger("wildfire.alerts.bot")
DEFAULT_API_BASE = "https://api.telegram.org"
SHUTDOWN = False
HELP_TEXT = (
    "🔥 <b>Wildfire Alert Bot</b>\n\n"
    "Available commands:\n"
    "• /start - subscribe to alerts\n"
    "• /region &lt;min_lon,min_lat,max_lon,max_lat&gt; - set region of interest\n"
    "• /severity &lt;low|medium|high&gt; - minimum severity to receive\n"
    "• /status - latest 5 alerts in your region\n"
    "• /unsubscribe - stop alerts\n\n"
    "Tip: copy a bbox from Google Maps as <code>lon,lat,lon,lat</code>."
)


def _load_env_file(path: Path = Path(".env")) -> None:
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


def _install_signal_handlers() -> None:
    def _handle(signum, frame):  # noqa: ARG001
        global SHUTDOWN
        LOGGER.info("Signal %s received, shutting down", signum)
        SHUTDOWN = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def parse_args() -> argparse.Namespace:
    _load_env_file()
    parser = argparse.ArgumentParser(description="Telegram bot for wildfire alerts")
    parser.add_argument("--bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--api-base", default=os.getenv("TELEGRAM_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--poll-timeout", type=int, default=int(os.getenv("TELEGRAM_POLL_TIMEOUT", "25")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    if not args.bot_token:
        parser.error("TELEGRAM_BOT_TOKEN is required")
    return args


def telegram_request(api_base: str, bot_token: str, method: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/bot{bot_token}/{method}"
    data = urllib.parse.urlencode(params or {}).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API error for {method}: {payload}")
    return payload


def send_message(api_base: str, bot_token: str, chat_id: str, text: str) -> None:
    telegram_request(
        api_base,
        bot_token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        },
        timeout=15,
    )


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [chunk.strip() for chunk in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 4 numbers separated by commas: min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("bbox values must be numbers") from exc
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox min values must be smaller than max values")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("bbox coordinates are out of range")
    return min_lon, min_lat, max_lon, max_lat


def _format_subscriber(sub: dict[str, Any]) -> str:
    lines = [
        f"Status: <b>{sub['status']}</b>",
        f"Min severity: <b>{sub['min_severity']}</b>",
    ]
    if sub.get("bbox_min_lon") is not None:
        lines.append(
            f"Region: <code>{sub['bbox_min_lon']},{sub['bbox_min_lat']},{sub['bbox_max_lon']},{sub['bbox_max_lat']}</code>"
        )
    else:
        lines.append("Region: any")
    return "\n".join(lines)


def handle_command(api_base: str, bot_token: str, chat_id: str, command: str, args_text: str) -> None:
    cmd = command.lower()
    address = chat_id

    if cmd in {"/start", "/help"}:
        sub = db.upsert_subscriber(channel="telegram", address=address)
        send_message(
            api_base,
            bot_token,
            chat_id,
            HELP_TEXT + "\n\n" + _format_subscriber(sub),
        )
        return

    if cmd == "/region":
        if not args_text.strip():
            sub = db.get_subscriber("telegram", address)
            if sub is None:
                send_message(api_base, bot_token, chat_id, "You are not subscribed yet. Use /start first.")
                return
            send_message(
                api_base,
                bot_token,
                chat_id,
                "Send <code>/region min_lon,min_lat,max_lon,max_lat</code> to set a region.\n\n" + _format_subscriber(sub),
            )
            return
        try:
            bbox = parse_bbox(args_text)
        except ValueError as exc:
            send_message(api_base, bot_token, chat_id, f"❌ {exc}")
            return
        if not db.update_region("telegram", address, bbox):
            db.upsert_subscriber(channel="telegram", address=address, bbox=bbox)
        sub = db.get_subscriber("telegram", address)
        send_message(api_base, bot_token, chat_id, "✅ Region updated.\n\n" + _format_subscriber(sub or {}))
        return

    if cmd == "/severity":
        level = args_text.strip().lower()
        if level not in {"low", "medium", "high"}:
            send_message(api_base, bot_token, chat_id, "Usage: <code>/severity low|medium|high</code>")
            return
        if not db.update_severity("telegram", address, level):
            db.upsert_subscriber(channel="telegram", address=address, min_severity=level)
        sub = db.get_subscriber("telegram", address)
        send_message(api_base, bot_token, chat_id, "✅ Severity updated.\n\n" + _format_subscriber(sub or {}))
        return

    if cmd == "/unsubscribe":
        if db.deactivate_subscriber("telegram", address):
            send_message(api_base, bot_token, chat_id, "👋 You are unsubscribed. Send /start to resubscribe.")
        else:
            send_message(api_base, bot_token, chat_id, "You were not subscribed.")
        return

    if cmd == "/status":
        sub = db.get_subscriber("telegram", address)
        if sub is None or sub["status"] != "active":
            send_message(api_base, bot_token, chat_id, "You are not subscribed. Use /start first.")
            return
        recent = db.recent_alerts_for_subscriber(sub, limit=5)
        if not recent:
            send_message(
                api_base,
                bot_token,
                chat_id,
                "No alerts at or above your severity threshold yet.\n\n" + _format_subscriber(sub),
            )
            return
        lines = [f"📋 <b>Latest {len(recent)} alerts in your region:</b>"]
        for item in recent:
            lines.append(
                f"• {item['date']} | grid <code>{item['grid_id']}</code> | "
                f"{item['risk_label'].upper()} (score {item['risk_score']:.2f}) "
                f"@ {item['latitude']:.3f},{item['longitude']:.3f}"
            )
        lines.append("\n" + _format_subscriber(sub))
        send_message(api_base, bot_token, chat_id, "\n".join(lines))
        return

    send_message(api_base, bot_token, chat_id, f"Unknown command: {command}\n\n" + HELP_TEXT)


def process_update(api_base: str, bot_token: str, update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0].split("@")[0]
        args_text = parts[1] if len(parts) == 2 else ""
        try:
            handle_command(api_base, bot_token, chat_id, command, args_text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to handle command %s for %s", command, chat_id)
            send_message(api_base, bot_token, chat_id, f"⚠️ Internal error: {exc}")
    else:
        send_message(api_base, bot_token, chat_id, HELP_TEXT)


def poll_loop(args: argparse.Namespace) -> None:
    db.init_schema()
    LOGGER.info("Wildfire Telegram bot started, long-polling every %ss", args.poll_timeout)
    offset: int | None = None
    while not SHUTDOWN:
        params: dict[str, Any] = {"timeout": args.poll_timeout, "allowed_updates": json.dumps(["message"])}
        if offset is not None:
            params["offset"] = offset
        try:
            payload = telegram_request(
                args.api_base,
                args.bot_token,
                "getUpdates",
                params,
                timeout=args.poll_timeout + 10,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Polling failed; retrying in 5s")
            time.sleep(5)
            continue

        for update in payload.get("result", []) or []:
            offset = max(offset or 0, int(update.get("update_id", 0)) + 1)
            process_update(args.api_base, args.bot_token, update)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _install_signal_handlers()
    poll_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
