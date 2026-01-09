import argparse
import csv
import os
import json
from io import StringIO
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import requests

# -------------------------
# Config (put in env vars or directly here)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "alert_state.json")  # prevents duplicate alerts

# Example tickers (change as needed)
WATCHLIST: dict[str, list[str]] = {
    "ETFs": [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "TQQQ",  # Nasdaq 100 3x leveraged
    ],
    "Magnificent Seven": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "AMZN",   # Amazon
        "GOOGL",  # Alphabet (Google)
        "META",   # Meta (Facebook)
        "NVDA",   # Nvidia
        "TSLA",   # Tesla
    ],
    "Semis / hardware": [
        "AVGO",  # Broadcom
        "AMD",   # AMD
        "TSM",   # Taiwan Semi (ADR)
        "ASML",  # ASML (ADR)
        "AMAT",  # Applied Materials
        "LRCX",  # Lam Research
        "KLAC",  # KLA
        "MU",    # Micron
        "QCOM",  # Qualcomm
        "INTC",  # Intel
        "TXN",   # Texas Instruments
        "ADI",   # Analog Devices
        "MRVL",  # Marvell
        "ARM",   # Arm
        "SMCI",  # Super Micro Computer
        "SNPS",  # Synopsys
        "CDNS",  # Cadence
    ],
    "Software / cloud": [
        "ADBE",  # Adobe
        "ORCL",  # Oracle
        "CRM",   # Salesforce
        "NOW",   # ServiceNow
        "TEAM",  # Atlassian
        "SHOP",  # Shopify
        "SNOW",  # Snowflake
        "DDOG",  # Datadog
        "MDB",   # MongoDB
        "NET",   # Cloudflare
        "SPOT",  # Spotify
        "UBER",  # Uber
        "ABNB",  # Airbnb
        "PLTR",  # Palantir
    ],
    "Cybersecurity": [
        "CRWD",  # CrowdStrike
        "PANW",  # Palo Alto Networks
        "FTNT",  # Fortinet
        "ZS",    # Zscaler
        "OKTA",  # Okta
        "CHKP",  # Check Point
    ],
    "Consumer": [
        "COST",  # Costco
        "NFLX",  # Netflix
        "WMT",   # Walmart
        "HD",    # Home Depot
        "LOW",   # Lowe's
        "NKE",   # Nike
        "SBUX",  # Starbucks
        "MCD",   # McDonald's
        "DIS",   # Disney
        "KO",    # Coca-Cola
        "PEP",   # PepsiCo
    ],
    "Financials": [
        "JPM",    # JPMorgan
        "BRK-B",  # Berkshire Hathaway (B)
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "V",      # Visa
        "MA",     # Mastercard
        "BLK",    # BlackRock
    ],
    "Healthcare": [
        "LLY",   # Eli Lilly
        "UNH",   # UnitedHealth
        "JNJ",   # Johnson & Johnson
        "MRK",   # Merck
        "ABBV",  # AbbVie
        "PFE",   # Pfizer
        "TMO",   # Thermo Fisher
        "ISRG",  # Intuitive Surgical
        "ABT",   # Abbott
        "AMGN",  # Amgen
    ],
    "Industrials / energy": [
        "CAT",  # Caterpillar
        "DE",   # Deere
        "GE",   # GE Aerospace
        "RTX",  # RTX
        "LMT",  # Lockheed Martin
        "XOM",  # Exxon Mobil
        "CVX",  # Chevron
        "COP",  # ConocoPhillips
    ],
}


def flatten_watchlist(watchlist: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    symbols: list[str] = []
    symbol_to_group: dict[str, str] = {}
    for group, group_symbols in watchlist.items():
        for symbol in group_symbols:
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            if symbol in symbol_to_group:
                continue
            symbol_to_group[symbol] = group
            symbols.append(symbol)
    return symbols, symbol_to_group


SYMBOLS, SYMBOL_TO_GROUP = flatten_watchlist(WATCHLIST)

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
DEFAULT_MAX_ABS_DISTANCE_PCT = 5.0
MAX_TELEGRAM_TEXT_LEN = 3900


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
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=366 * years)
    params = {
        "s": stooq_symbol,
        "i": "d",
        "d1": start_date.strftime("%Y%m%d"),
        "d2": end_date.strftime("%Y%m%d"),
    }
    r = requests.get(STOOQ_DAILY_CSV_URL, params=params, timeout=20)
    r.raise_for_status()

    content = r.text.strip()
    if not content or content.startswith("No data"):
        raise ValueError(f"No data for {symbol} ({stooq_symbol})")

    cutoff = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
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


def observed_if_weekend(d: date) -> date:
    # Saturday -> Friday, Sunday -> Monday
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    # weekday: Monday=0 ... Sunday=6
    if n <= 0:
        raise ValueError("n must be > 0")
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d = d + timedelta(days=offset + 7 * (n - 1))
    return d


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    # weekday: Monday=0 ... Sunday=6
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def easter_sunday(year: int) -> date:
    # Meeus/Jones/Butcher algorithm (Gregorian calendar)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nyse_holidays(year: int) -> set[date]:
    # Observed (market-closed) holidays for NYSE (common set).
    holidays: set[date] = set()

    holidays.add(observed_if_weekend(date(year, 1, 1)))  # New Year's Day
    holidays.add(nth_weekday_of_month(year, 1, weekday=0, n=3))  # MLK Day (3rd Mon Jan)
    holidays.add(nth_weekday_of_month(year, 2, weekday=0, n=3))  # Presidents' Day (3rd Mon Feb)
    holidays.add(easter_sunday(year) - timedelta(days=2))  # Good Friday
    holidays.add(last_weekday_of_month(year, 5, weekday=0))  # Memorial Day (last Mon May)

    if year >= 2022:
        holidays.add(observed_if_weekend(date(year, 6, 19)))  # Juneteenth

    holidays.add(observed_if_weekend(date(year, 7, 4)))  # Independence Day
    holidays.add(nth_weekday_of_month(year, 9, weekday=0, n=1))  # Labor Day (1st Mon Sep)
    holidays.add(nth_weekday_of_month(year, 11, weekday=3, n=4))  # Thanksgiving (4th Thu Nov)
    holidays.add(observed_if_weekend(date(year, 12, 25)))  # Christmas Day

    return holidays


def is_nyse_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    holidays = nyse_holidays(d.year) | nyse_holidays(d.year + 1)
    return d not in holidays


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


DailyRow = tuple[str, SMA150Snapshot, str | None, float]


def format_row_line(symbol: str, last: SMA150Snapshot, signal: str | None, dist_pct: float) -> str:
    signal_text = signal or "none"
    return (
        f"{symbol} ({last.date.strftime('%Y-%m-%d')}): "
        f"close={last.close:.2f} sma150={last.sma150:.2f} dist={dist_pct:+.2f}% signal={signal_text}"
    )


def pack_sections_into_messages(
    header_lines: list[str],
    sections: list[list[str]],
    *,
    max_text_len: int,
) -> list[str]:
    messages: list[str] = []
    current_lines = header_lines.copy()

    def flush() -> None:
        nonlocal current_lines
        if len(current_lines) > len(header_lines):
            messages.append("\n".join(current_lines).strip())
        current_lines = header_lines.copy()

    for section_lines in sections:
        candidate = current_lines + [""] + section_lines
        if len("\n".join(candidate)) <= max_text_len:
            current_lines = candidate
            continue

        flush()
        candidate = current_lines + [""] + section_lines
        if len("\n".join(candidate)) <= max_text_len:
            current_lines = candidate
            continue

        # Section is too large by itself -> chunk it
        section_header = section_lines[0]
        chunk: list[str] = [section_header]
        for line in section_lines[1:]:
            candidate = header_lines + [""] + chunk + [line]
            if len("\n".join(candidate)) <= max_text_len:
                chunk.append(line)
                continue
            messages.append("\n".join(header_lines + [""] + chunk).strip())
            chunk = [section_header, line]

        current_lines = header_lines + [""] + chunk

    flush()
    if not messages:
        messages.append("\n".join(header_lines).strip())
    return messages


def build_daily_messages(
    summary_date: str,
    rows: list[DailyRow],
    errors: list[str],
    max_abs_distance_pct: float,
    scanned_count: int,
) -> list[str]:
    header_lines: list[str] = [
        f"SMA150 Daily Update ({summary_date})",
        f"Filter: |dist| <= {max_abs_distance_pct:.2f}% (showing {len(rows)}/{scanned_count})",
    ]

    if not rows and not errors:
        return ["\n".join(header_lines + [f"No symbols within ±{max_abs_distance_pct:.2f}% of SMA150."]).strip()]

    sections: list[list[str]] = []

    if rows:
        rows_by_group: dict[str, list[DailyRow]] = {}
        for row in rows:
            symbol = row[0]
            group = SYMBOL_TO_GROUP.get(symbol, "Other")
            rows_by_group.setdefault(group, []).append(row)

        for group_rows in rows_by_group.values():
            group_rows.sort(key=lambda r: abs(r[3]))

        group_order = list(WATCHLIST.keys())
        ordered_groups = [g for g in group_order if g in rows_by_group] + sorted(
            (g for g in rows_by_group.keys() if g not in group_order)
        )

        for group in ordered_groups:
            group_rows = rows_by_group[group]
            section_lines = [f"== {group} ({len(group_rows)}) =="]
            any_signal = any(signal is not None for _, _, signal, _ in group_rows)
            if not any_signal:
                section_lines.append("No signals in this group.")
            for symbol, last, signal, dist_pct in group_rows:
                section_lines.append(format_row_line(symbol, last, signal, dist_pct))
            sections.append(section_lines)
    else:
        sections.append([f"No symbols within ±{max_abs_distance_pct:.2f}% of SMA150."])

    if errors:
        sections.append(["== Errors =="] + errors)

    return pack_sections_into_messages(header_lines, sections, max_text_len=MAX_TELEGRAM_TEXT_LEN)

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


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def main():
    parser = argparse.ArgumentParser(description="SMA150 alerts to Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Print message(s) instead of sending to Telegram")
    parser.add_argument("--test", action="store_true", help="Send a test Telegram message and exit")
    parser.add_argument("--status", action="store_true", help="Send daily status message, then exit")
    parser.add_argument("--force", action="store_true", help="Ignore once-per-day guard (manual use)")
    args = parser.parse_args()

    load_env_file(os.path.join(BASE_DIR, "secrets.env"))
    load_env_file(os.path.join(BASE_DIR, ".env"))

    bot_token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
    chat_id_raw = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_USER") or "").strip()
    dry_run = args.dry_run or env_bool("DRY_RUN", default=False)
    max_abs_distance_pct = env_float("SMA150_MAX_DIST_PCT", default=DEFAULT_MAX_ABS_DISTANCE_PCT)

    if not bot_token or bot_token == "PUT_YOUR_BOT_TOKEN":
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN) in env vars, secrets.env, or .env.")
    if not chat_id_raw or chat_id_raw == "PUT_YOUR_CHAT_ID":
        raise RuntimeError("Please set TELEGRAM_CHAT_ID (or TELEGRAM_USER) in env vars, secrets.env, or .env.")

    chat_id = normalize_chat_id(chat_id_raw)

    if args.test:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        send_telegram(bot_token, chat_id, f"Bot test OK ({now_utc})", dry_run=dry_run)
        return

    market_tz = ZoneInfo(os.getenv("MARKET_TZ", "America/New_York"))
    market_date = datetime.now(market_tz).date()
    if not args.force and not is_nyse_trading_day(market_date):
        return

    state = load_state()
    errors: list[str] = []
    status_rows: list[tuple[str, SMA150Snapshot, str | None, float]] = []

    for symbol in SYMBOLS:
        try:
            prev, last = fetch_last_two_with_sma150(symbol)
            signal = classify_signal(prev, last)
            dist_pct = (last.close / last.sma150 - 1.0) * 100
            status_rows.append((symbol, last, signal, dist_pct))
        except Exception as e:
            errors.append(f"{symbol}: {e}")
            continue

    near_rows = [row for row in status_rows if abs(row[3]) <= max_abs_distance_pct]
    near_rows.sort(key=lambda r: abs(r[3]))

    summary_date = (
        max(row[1].date for row in status_rows).strftime("%Y-%m-%d")
        if status_rows
        else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    summary_key = f"summary:{summary_date}"

    send_daily_summary = env_bool("SEND_DAILY_SUMMARY", default=True)
    should_send_summary = args.status or send_daily_summary

    if should_send_summary:
        if dry_run:
            messages = build_daily_messages(
                summary_date,
                near_rows,
                errors,
                max_abs_distance_pct=max_abs_distance_pct,
                scanned_count=len(status_rows),
            )
            for msg in messages:
                send_telegram(bot_token, chat_id, msg, dry_run=True)
        else:
            already_sent = bool(state.get(summary_key))
            if args.force or not already_sent:
                messages = build_daily_messages(
                    summary_date,
                    near_rows,
                    errors,
                    max_abs_distance_pct=max_abs_distance_pct,
                    scanned_count=len(status_rows),
                )
                for msg in messages:
                    send_telegram(bot_token, chat_id, msg, dry_run=False)
                state[summary_key] = {
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "date": summary_date,
                    "type": "daily_summary",
                }

    save_state(state)

if __name__ == "__main__":
    main()
