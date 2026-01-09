import argparse
import csv
import os
import json
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import requests

# -------------------------
# Config (put in env vars or directly here)
# -------------------------
STATE_FILE = "alert_state.json"  # prevents duplicate alerts

# Example tickers (change as needed)
SYMBOLS = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "TQQQ",  # Nasdaq 100 3x leveraged
]

# -------------------------
# Telegram
# -------------------------
def normalize_chat_id(chat_id: str) -> str:
    chat_id = chat_id.strip()
    if chat_id.lstrip("-").isdigit():
        return chat_id
    if not chat_id.startswith("@"):
        return f"@{chat_id}"
    return chat_id


def send_telegram(bot_token: str, chat_id: str, message: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY_RUN] Telegram message to {chat_id}:\n{message}\n")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

# -------------------------
# State (anti-spam)
# -------------------------
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# -------------------------
# Data + SMA logic
# -------------------------
STOOQ_DAILY_CSV_URL = "https://stooq.com/q/d/l/"
SMA_WINDOW_DAYS = 150


@dataclass(frozen=True)
class SMA150Snapshot:
    date: datetime
    close: float
    sma150: float


def fetch_price_history(symbol: str, years: int = 3) -> list[tuple[datetime, float]]:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("symbol is empty")

    stooq_symbol = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    params = {"s": stooq_symbol, "i": "d"}
    r = requests.get(STOOQ_DAILY_CSV_URL, params=params, timeout=20)
    r.raise_for_status()

    content = r.text.strip()
    if not content or content.startswith("No data"):
        raise ValueError(f"No data for {symbol} ({stooq_symbol})")

    cutoff = datetime.now(timezone.utc) - timedelta(days=366 * years)
    history: list[tuple[datetime, float]] = []
    reader = csv.DictReader(StringIO(content))
    for row in reader:
        date_str = (row.get("Date") or "").strip()
        close_str = (row.get("Close") or "").strip()
        if not date_str or not close_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            close = float(close_str)
        except ValueError:
            continue
        if dt < cutoff:
            continue
        history.append((dt, close))

    if not history:
        raise ValueError(f"No recent price history for {symbol} ({stooq_symbol})")
    return history


def compute_sma(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(values) < window:
        return [None] * len(values)

    sma: list[float | None] = [None] * len(values)
    rolling_sum = sum(values[:window])
    sma[window - 1] = rolling_sum / window

    for i in range(window, len(values)):
        rolling_sum += values[i] - values[i - window]
        sma[i] = rolling_sum / window

    return sma


def fetch_last_two_with_sma150(symbol: str, years: int = 3) -> tuple[SMA150Snapshot, SMA150Snapshot]:
    history = fetch_price_history(symbol, years=years)
    closes = [close for _, close in history]
    sma = compute_sma(closes, SMA_WINDOW_DAYS)

    indices = [i for i, v in enumerate(sma) if v is not None]
    if len(indices) < 2:
        raise ValueError(f"Not enough data for SMA{SMA_WINDOW_DAYS} for {symbol}")

    i_prev, i_last = indices[-2], indices[-1]
    prev_dt, prev_close = history[i_prev]
    last_dt, last_close = history[i_last]
    return (
        SMA150Snapshot(date=prev_dt, close=prev_close, sma150=float(sma[i_prev])),
        SMA150Snapshot(date=last_dt, close=last_close, sma150=float(sma[i_last])),
    )


def classify_signal(prev: SMA150Snapshot, last: SMA150Snapshot) -> str | None:
    """
    Returns one of: "cross_up", "cross_down", "touch"
    Or None if no signal.
    """
    # Cross up (below -> above)
    if prev.close < prev.sma150 and last.close >= last.sma150:
        return "cross_up"

    # Cross down (above -> below)
    if prev.close > prev.sma150 and last.close <= last.sma150:
        return "cross_down"

    # Touch (close very near SMA150) - configurable threshold
    dist_pct = abs((last.close / last.sma150) - 1.0) * 100
    if dist_pct <= 0.15:  # 0.15% proximity
        return "touch"

    return None

def format_message(symbol: str, last: SMA150Snapshot, signal: str) -> str:
    date_str = last.date.strftime("%Y-%m-%d")
    close = last.close
    sma = last.sma150
    dist_pct = (close / sma - 1.0) * 100

    label = {
        "cross_up": "CROSS UP (מתחת -> מעל)",
        "cross_down": "CROSS DOWN (מעל -> מתחת)",
        "touch": "TOUCH (נגיעה/קרוב מאוד)",
    }[signal]

    return (
        f"SMA150 Alert: {symbol}\n"
        f"Signal: {label}\n"
        f"Date: {date_str}\n"
        f"Close: {close:.2f}\n"
        f"SMA150: {sma:.2f}\n"
        f"Distance: {dist_pct:.2f}%"
    )


def format_daily_summary(
    summary_date: str,
    rows: list[tuple[str, SMA150Snapshot, str | None, float]],
    errors: list[str],
) -> str:
    lines: list[str] = [f"SMA150 Daily Update ({summary_date})"]
    if not rows and not errors:
        lines.append("No data.")
        return "\n".join(lines)

    any_signal = any(signal is not None for _, _, signal, _ in rows)
    if not any_signal:
        lines.append("No signals today.")

    for symbol, last, signal, dist_pct in rows:
        signal_text = signal or "none"
        lines.append(
            f"{symbol} ({last.date.strftime('%Y-%m-%d')}): "
            f"close={last.close:.2f} sma150={last.sma150:.2f} dist={dist_pct:+.2f}% signal={signal_text}"
        )

    if errors:
        lines.append("Errors:")
        lines.extend(errors)

    return "\n".join(lines)

# -------------------------
# Runner
# -------------------------
def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'").strip()
            if not key:
                continue
            os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    parser = argparse.ArgumentParser(description="SMA150 alerts to Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Print message(s) instead of sending to Telegram")
    parser.add_argument("--test", action="store_true", help="Send a test Telegram message and exit")
    parser.add_argument("--status", action="store_true", help="Send status message (always), then exit")
    args = parser.parse_args()

    load_env_file("secrets.env")
    load_env_file(".env")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    dry_run = args.dry_run or env_bool("DRY_RUN", default=False)

    if not bot_token or bot_token == "PUT_YOUR_BOT_TOKEN":
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN (in env vars, secrets.env, or .env).")
    if not chat_id_raw or chat_id_raw == "PUT_YOUR_CHAT_ID":
        raise RuntimeError("Please set TELEGRAM_CHAT_ID (in env vars, secrets.env, or .env).")

    chat_id = normalize_chat_id(chat_id_raw)

    if args.test:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        send_telegram(bot_token, chat_id, f"Bot test OK ({now_utc})", dry_run=dry_run)
        return

    state = load_state()
    errors: list[str] = []
    status_rows: list[tuple[str, SMA150Snapshot, str | None, float]] = []
    sent_any_alert = False

    for symbol in SYMBOLS:
        try:
            prev, last = fetch_last_two_with_sma150(symbol)
            signal = classify_signal(prev, last)
            dist_pct = (last.close / last.sma150 - 1.0) * 100
            status_rows.append((symbol, last, signal, dist_pct))
        except Exception as e:
            errors.append(f"{symbol}: {e}")
            continue

        if not signal:
            continue

        # Anti-spam key: symbol + signal + date
        last_date = last.date.strftime("%Y-%m-%d")
        key = f"{symbol}:{signal}:{last_date}"

        if state.get(key):
            continue  # already sent today

        msg = format_message(symbol, last, signal)
        send_telegram(bot_token, chat_id, msg, dry_run=dry_run)
        sent_any_alert = True

        state[key] = {
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": signal,
            "date": last_date
        }

    if args.status:
        summary_date = (
            status_rows[-1][1].date.strftime("%Y-%m-%d")
            if status_rows
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        msg = format_daily_summary(summary_date, status_rows, errors)
        send_telegram(bot_token, chat_id, msg, dry_run=dry_run)
        return

    send_daily_summary = env_bool("SEND_DAILY_SUMMARY", default=True)
    if send_daily_summary and not sent_any_alert:
        summary_date = (
            status_rows[-1][1].date.strftime("%Y-%m-%d")
            if status_rows
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        summary_key = f"summary:{summary_date}"
        if not state.get(summary_key):
            msg = format_daily_summary(summary_date, status_rows, errors)
            send_telegram(bot_token, chat_id, msg, dry_run=dry_run)
            state[summary_key] = {
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "date": summary_date,
                "type": "daily_summary",
            }

    save_state(state)

if __name__ == "__main__":
    main()
