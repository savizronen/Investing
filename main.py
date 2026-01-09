from __future__ import annotations

import argparse
import csv
import os
import json
import html
from io import StringIO
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
import time
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
import requests

# -------------------------
# Config (put in env vars or directly here)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "alert_state.json")  # prevents duplicate alerts
HISTORY_CACHE_DIR = os.path.join(BASE_DIR, "history_cache")

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
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = _HTTP_SESSION.post(url, json=payload, timeout=20)
    r.raise_for_status()

# -------------------------
# State (anti-spam)
# -------------------------
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# -------------------------
# Data + SMA logic
# -------------------------
STOOQ_DAILY_CSV_URL = "https://stooq.com/q/d/l/"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_MAX_ABS_DISTANCE_PCT = 5.0
MAX_TELEGRAM_TEXT_LEN = 3900
DEFAULT_HISTORY_CACHE_TTL_HOURS = 20
DEFAULT_HISTORY_PROVIDERS = "yahoo,stooq"
DEFAULT_YAHOO_USER_AGENT = "Mozilla/5.0"
DEFAULT_YAHOO_RETRY_COUNT = 2
MAX_ERROR_LINES = 20
DEFAULT_SMA_WINDOW_DAYS = 150

_STOOQ_RATE_LIMITED = False
_YAHOO_RATE_LIMITED = False
_HTTP_SESSION = requests.Session()
_ENV_FILE_KEYS_SET: set[str] = set()


class RateLimitError(RuntimeError):
    pass


class StooqRateLimitError(RateLimitError):
    pass


class YahooRateLimitError(RateLimitError):
    pass


def is_stooq_rate_limit_message(text: str) -> bool:
    t = text.strip().lower()
    return (
        "exceeded the daily hits limit" in t
        or "przekroczony dzienny limit" in t
        or "limit wywolan" in t
    )


@dataclass(frozen=True)
class SMASnapshot:
    date: datetime
    close: float
    sma: float


def parse_history_csv(content: str, *, cutoff: datetime) -> list[tuple[datetime, float]]:
    history: list[tuple[datetime, float]] = []
    reader = csv.DictReader(StringIO(content.strip()))
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
    history.sort(key=lambda x: x[0])
    return history


def history_to_csv(history: list[tuple[datetime, float]]) -> str:
    # Minimal cache format (only what's needed for SMA calc).
    out = ["Date,Close"]
    for dt, close in sorted(history, key=lambda x: x[0]):
        out.append(f"{dt.strftime('%Y-%m-%d')},{close}")
    return "\n".join(out) + "\n"


def save_cache_csv(cache_path: str, content: str) -> None:
    try:
        os.makedirs(HISTORY_CACHE_DIR, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content.rstrip())
            f.write("\n")
        os.replace(tmp_path, cache_path)
    except OSError:
        pass


def fetch_history_from_stooq(
    *,
    symbol: str,
    stooq_symbol: str,
    start_date: date,
    end_date: date,
) -> str:
    global _STOOQ_RATE_LIMITED
    if _STOOQ_RATE_LIMITED:
        raise StooqRateLimitError("Stooq daily hits limit exceeded.")

    params = {
        "s": stooq_symbol,
        "i": "d",
        "d1": start_date.strftime("%Y%m%d"),
        "d2": end_date.strftime("%Y%m%d"),
    }
    r = _HTTP_SESSION.get(STOOQ_DAILY_CSV_URL, params=params, timeout=20)
    r.raise_for_status()

    content = r.text.strip()
    if is_stooq_rate_limit_message(content):
        _STOOQ_RATE_LIMITED = True
        raise StooqRateLimitError("Stooq daily hits limit exceeded (try again tomorrow).")
    if not content or content.startswith("No data"):
        raise ValueError(f"No data for {symbol} ({stooq_symbol})")

    return content


def fetch_history_from_yahoo(*, symbol: str, start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, float]]:
    global _YAHOO_RATE_LIMITED
    if _YAHOO_RATE_LIMITED:
        raise YahooRateLimitError("Yahoo rate limit exceeded.")

    url = YAHOO_CHART_URL.format(symbol=quote_plus(symbol))
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": os.getenv("YAHOO_USER_AGENT", DEFAULT_YAHOO_USER_AGENT),
        "Accept": "application/json, text/plain, */*",
    }

    retries = int(env_float("YAHOO_RETRY_COUNT", default=float(DEFAULT_YAHOO_RETRY_COUNT)))
    backoff = 1.0
    last_status: int | None = None

    for attempt in range(retries + 1):
        r = _HTTP_SESSION.get(url, params=params, headers=headers, timeout=20)
        last_status = r.status_code
        if r.status_code == 429:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2.0
                continue
            _YAHOO_RATE_LIMITED = True
            raise YahooRateLimitError("Yahoo rate limit exceeded (HTTP 429).")
        r.raise_for_status()

        try:
            data = r.json()
        except ValueError:
            preview = (r.text or "").strip().replace("\n", " ")[:200]
            raise ValueError(f"Yahoo returned non-JSON for {symbol}: {preview}")

        chart = data.get("chart") or {}
        err = chart.get("error")
        if err:
            desc = (err.get("description") or str(err)).strip()
            if "too many requests" in desc.lower():
                _YAHOO_RATE_LIMITED = True
                raise YahooRateLimitError(f"Yahoo rate limit exceeded: {desc}")
            raise ValueError(f"Yahoo error for {symbol}: {desc}")

        result = (chart.get("result") or [None])[0]
        if not result:
            raise ValueError(f"Yahoo: No result for {symbol}")

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}

        closes: list[float | None] | None = None
        adjclose_list = indicators.get("adjclose") or []
        quote_list = indicators.get("quote") or []
        if adjclose_list and isinstance(adjclose_list, list):
            closes = adjclose_list[0].get("adjclose")
        if closes is None and quote_list and isinstance(quote_list, list):
            closes = quote_list[0].get("close")

        if not timestamps or closes is None:
            raise ValueError(f"Yahoo: Missing time series for {symbol}")

        history: list[tuple[datetime, float]] = []
        for ts, close in zip(timestamps, closes):
            if ts is None or close is None:
                continue
            try:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                if dt < start_dt or dt > end_dt:
                    continue
                history.append((dt, float(close)))
            except (ValueError, OSError, OverflowError):
                continue

        if not history:
            raise ValueError(f"Yahoo: No recent price history for {symbol}")

        history.sort(key=lambda x: x[0])
        return history

    _YAHOO_RATE_LIMITED = True
    raise YahooRateLimitError(f"Yahoo request failed (last status={last_status}).")


def fetch_price_history(symbol: str, years: int = 3) -> list[tuple[datetime, float]]:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("symbol is empty")

    stooq_symbol = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=366 * years)
    cutoff = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    start_dt = cutoff
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    cache_ttl_hours = env_float("HISTORY_CACHE_TTL_HOURS", default=DEFAULT_HISTORY_CACHE_TTL_HOURS)
    cache_path = os.path.join(HISTORY_CACHE_DIR, f"{stooq_symbol}.csv")
    cached_content: str | None = None
    cache_is_fresh = False

    if os.path.exists(cache_path):
        try:
            cached_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - cached_mtime).total_seconds() / 3600
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_content = f.read().strip()
            cache_is_fresh = bool(cached_content) and (age_hours <= max(cache_ttl_hours, 0.0))
        except OSError:
            cached_content = None
            cache_is_fresh = False

    if cache_is_fresh and cached_content is not None:
        history = parse_history_csv(cached_content, cutoff=cutoff)
        if history:
            return history

    providers_raw = (os.getenv("HISTORY_PROVIDERS") or DEFAULT_HISTORY_PROVIDERS).strip()
    providers = [p.strip().lower() for p in providers_raw.split(",") if p.strip()]
    providers = [p for p in providers if p in {"yahoo", "stooq"}]
    if not providers:
        providers = [p.strip() for p in DEFAULT_HISTORY_PROVIDERS.split(",")]

    provider_errors: list[str] = []
    rate_limited: list[str] = []

    for provider in providers:
        try:
            if provider == "yahoo":
                history = fetch_history_from_yahoo(symbol=symbol, start_dt=start_dt, end_dt=end_dt)
                save_cache_csv(cache_path, history_to_csv(history))
                return history

            if provider == "stooq":
                stooq_csv = fetch_history_from_stooq(
                    symbol=symbol,
                    stooq_symbol=stooq_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                history = parse_history_csv(stooq_csv, cutoff=cutoff)
                if not history:
                    raise ValueError(f"No recent price history for {symbol} ({stooq_symbol})")
                save_cache_csv(cache_path, history_to_csv(history))
                return history
        except RateLimitError as e:
            rate_limited.append(provider)
            provider_errors.append(f"{provider}: {e}")
            continue
        except Exception as e:
            provider_errors.append(f"{provider}: {e}")
            continue

    if cached_content is not None:
        history = parse_history_csv(cached_content, cutoff=cutoff)
        if history:
            return history

    if rate_limited:
        providers_str = ", ".join(rate_limited)
        raise RateLimitError(f"Rate limit reached for provider(s): {providers_str}")

    if provider_errors:
        raise ValueError(f"No recent price history for {symbol} ({stooq_symbol}). Last errors: {provider_errors[-1]}")

    raise ValueError(f"No recent price history for {symbol} ({stooq_symbol})")


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


def fetch_last_two_with_sma(symbol: str, *, window_days: int, years: int = 3) -> tuple[SMASnapshot, SMASnapshot]:
    history = fetch_price_history(symbol, years=years)
    closes = [close for _, close in history]
    sma = compute_sma(closes, window_days)

    indices = [i for i, v in enumerate(sma) if v is not None]
    if len(indices) < 2:
        raise ValueError(f"Not enough data for SMA{window_days} for {symbol}")

    i_prev, i_last = indices[-2], indices[-1]
    prev_dt, prev_close = history[i_prev]
    last_dt, last_close = history[i_last]
    return (
        SMASnapshot(date=prev_dt, close=prev_close, sma=float(sma[i_prev])),
        SMASnapshot(date=last_dt, close=last_close, sma=float(sma[i_last])),
    )


def classify_signal(prev: SMASnapshot, last: SMASnapshot) -> str | None:
    """
    Returns one of: "cross_up", "cross_down", "touch"
    Or None if no signal.
    """
    # Cross up (below -> above)
    if prev.close < prev.sma and last.close >= last.sma:
        return "cross_up"

    # Cross down (above -> below)
    if prev.close > prev.sma and last.close <= last.sma:
        return "cross_down"

    # Touch (close very near SMA) - configurable threshold
    dist_pct = abs((last.close / last.sma) - 1.0) * 100
    if dist_pct <= 0.15:  # 0.15% proximity
        return "touch"

    return None

def format_message(symbol: str, last: SMASnapshot, signal: str, *, window_days: int) -> str:
    date_str = last.date.strftime("%Y-%m-%d")
    close = last.close
    sma = last.sma
    dist_pct = (close / sma - 1.0) * 100

    label = {
        "cross_up": "CROSS UP (מתחת -> מעל)",
        "cross_down": "CROSS DOWN (מעל -> מתחת)",
        "touch": "TOUCH (נגיעה/קרוב מאוד)",
    }[signal]

    return (
        f"SMA{window_days} Alert: {symbol}\n"
        f"Signal: {label}\n"
        f"Date: {date_str}\n"
        f"Close: {close:.2f}\n"
        f"SMA{window_days}: {sma:.2f}\n"
        f"Distance: {dist_pct:.2f}%"
    )


DailyRow = tuple[str, SMASnapshot, str | None, float]


def google_chart_url(symbol: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(f'{symbol} stock')}"


def escape_html(text: str) -> str:
    return html.escape(text, quote=True)


def format_row_line(symbol: str, last: SMASnapshot, signal: str | None, dist_pct: float, *, window_days: int) -> str:
    chart_url = google_chart_url(symbol)
    symbol_link = f'<a href="{escape_html(chart_url)}">{escape_html(symbol)}</a>'

    close = last.close
    sma = last.sma
    dist_indicator = "🟩⬆️" if dist_pct > 0 else ("🟥⬇️" if dist_pct < 0 else "⬜️➡️")
    dist_text = f"{dist_indicator} {dist_pct:+.2f}%"

    signal_labels = {
        "cross_up": "↑ Cross Up",
        "cross_down": "↓ Cross Down",
        "touch": "≈ Touch",
    }

    line = f"• {symbol_link} — Close {close:.2f} | SMA{window_days} {sma:.2f} | Dist {dist_text}"
    if signal is not None:
        line += f" | <b>{escape_html(signal_labels.get(signal, signal))}</b>"
    return line


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
    only_above: bool,
    window_days: int,
    scanned_count: int,
) -> list[str]:
    filter_text = (
        f"dist ∈ [0, +{max_abs_distance_pct:.2f}%] (above SMA{window_days})"
        if only_above
        else f"|dist| ≤ {max_abs_distance_pct:.2f}%"
    )
    header_lines: list[str] = [
        f"<b>SMA{window_days} Daily Update</b> ({escape_html(summary_date)})",
        f"<i>Filter:</i> {filter_text} (showing {len(rows)}/{scanned_count})",
    ]

    if not rows and not errors:
        no_rows_line = (
            f"<i>No symbols within 0..+{max_abs_distance_pct:.2f}% above SMA{window_days}.</i>"
            if only_above
            else f"<i>No symbols within ±{max_abs_distance_pct:.2f}% of SMA{window_days}.</i>"
        )
        return ["\n".join(header_lines + [no_rows_line]).strip()]

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
            signal_count = sum(1 for _, _, signal, _ in group_rows if signal is not None)
            section_lines = [f"<b>{escape_html(group)}</b> ({len(group_rows)}) — signals: {signal_count}"]
            for symbol, last, signal, dist_pct in group_rows:
                section_lines.append(format_row_line(symbol, last, signal, dist_pct, window_days=window_days))
            sections.append(section_lines)
    else:
        no_rows_line = (
            f"<i>No symbols within 0..+{max_abs_distance_pct:.2f}% above SMA{window_days}.</i>"
            if only_above
            else f"<i>No symbols within ±{max_abs_distance_pct:.2f}% of SMA{window_days}.</i>"
        )
        sections.append([no_rows_line])

    if errors:
        sections.append(["<b>Errors</b>"] + [f"• {escape_html(e)}" for e in errors])

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
            value = value.strip()

            # Remove inline comments (unless inside quotes).
            in_single = False
            in_double = False
            out_chars: list[str] = []
            for ch in value:
                if ch == "'" and not in_double:
                    in_single = not in_single
                    out_chars.append(ch)
                    continue
                if ch == '"' and not in_single:
                    in_double = not in_double
                    out_chars.append(ch)
                    continue
                if ch == "#" and not in_single and not in_double:
                    break
                out_chars.append(ch)

            value = "".join(out_chars).strip()
            if len(value) >= 2 and ((value[0] == value[-1] == "'") or (value[0] == value[-1] == '"')):
                value = value[1:-1]
            value = value.strip()
            if not key:
                continue
            # Precedence: OS env > later env-file > earlier env-file.
            if key not in os.environ or key in _ENV_FILE_KEYS_SET:
                os.environ[key] = value
                _ENV_FILE_KEYS_SET.add(key)


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


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def main():
    parser = argparse.ArgumentParser(description="SMA alerts to Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Print message(s) instead of sending to Telegram")
    parser.add_argument("--test", action="store_true", help="Send a test Telegram message and exit")
    parser.add_argument("--status", action="store_true", help="Send daily status message, then exit")
    parser.add_argument("--force", action="store_true", help="Ignore once-per-day guard (manual use)")
    args = parser.parse_args()

    load_env_file(os.path.join(BASE_DIR, "config.env"))
    load_env_file(os.path.join(BASE_DIR, "secrets.env"))
    load_env_file(os.path.join(BASE_DIR, ".env"))

    bot_token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
    chat_id_raw = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_USER") or "").strip()
    dry_run = args.dry_run or env_bool("DRY_RUN", default=False)
    max_abs_distance_pct = env_float("SMA150_MAX_DIST_PCT", default=DEFAULT_MAX_ABS_DISTANCE_PCT)
    sma_window_days = env_int("SMA_WINDOW_DAYS", default=DEFAULT_SMA_WINDOW_DAYS)
    if sma_window_days <= 0:
        sma_window_days = DEFAULT_SMA_WINDOW_DAYS

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
    raw_errors: list[str] = []
    status_rows: list[DailyRow] = []
    rate_limit_missing_count = 0
    rate_limit_missing_preview: list[str] = []
    rate_limit_first_error: str | None = None

    for symbol in SYMBOLS:
        try:
            prev, last = fetch_last_two_with_sma(symbol, window_days=sma_window_days)
            signal = classify_signal(prev, last)
            dist_pct = (last.close / last.sma - 1.0) * 100
            status_rows.append((symbol, last, signal, dist_pct))
        except RateLimitError as e:
            rate_limit_missing_count += 1
            if rate_limit_first_error is None:
                rate_limit_first_error = str(e).strip() or "Rate limit reached."
            if len(rate_limit_missing_preview) < 12:
                rate_limit_missing_preview.append(symbol)
            continue
        except Exception as e:
            raw_errors.append(f"{symbol}: {e}")
            continue

    only_above = env_bool("SMA150_ONLY_ABOVE", default=False)
    if only_above:
        near_rows = [row for row in status_rows if 0.0 <= row[3] <= max_abs_distance_pct]
    else:
        near_rows = [row for row in status_rows if abs(row[3]) <= max_abs_distance_pct]
    near_rows.sort(key=lambda r: abs(r[3]))

    errors_for_message: list[str] = []
    if rate_limit_missing_count:
        prefix = rate_limit_first_error or "Rate limit reached."
        errors_for_message.append(f"{prefix} Missing data for {rate_limit_missing_count} symbol(s).")
        if rate_limit_missing_preview:
            preview = ", ".join(rate_limit_missing_preview)
            suffix = ", ..." if rate_limit_missing_count > len(rate_limit_missing_preview) else ""
            errors_for_message.append(f"Missing examples: {preview}{suffix}")

    if raw_errors:
        errors_for_message.extend(raw_errors[:MAX_ERROR_LINES])
        if len(raw_errors) > MAX_ERROR_LINES:
            errors_for_message.append(f"... and {len(raw_errors) - MAX_ERROR_LINES} more.")

    if status_rows:
        summary_date = max(row[1].date for row in status_rows).strftime("%Y-%m-%d")
        summary_key = f"summary:{summary_date}"
        summary_type = "daily_summary"
    else:
        # If we have no data at all (provider down / rate limit), don't block
        # a future successful summary for the same date.
        summary_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary_key = f"error:{summary_date}"
        summary_type = "daily_summary_error"

    send_daily_summary = env_bool("SEND_DAILY_SUMMARY", default=True)
    should_send_summary = args.status or send_daily_summary

    if should_send_summary:
        if dry_run:
            messages = build_daily_messages(
                summary_date,
                near_rows,
                errors_for_message,
                max_abs_distance_pct=max_abs_distance_pct,
                only_above=only_above,
                window_days=sma_window_days,
                scanned_count=len(SYMBOLS),
            )
            for msg in messages:
                send_telegram(bot_token, chat_id, msg, dry_run=True)
        else:
            already_sent = bool(state.get(summary_key))
            if args.force or not already_sent:
                messages = build_daily_messages(
                    summary_date,
                    near_rows,
                    errors_for_message,
                    max_abs_distance_pct=max_abs_distance_pct,
                    only_above=only_above,
                    window_days=sma_window_days,
                    scanned_count=len(SYMBOLS),
                )
                for msg in messages:
                    send_telegram(bot_token, chat_id, msg, dry_run=False)
                state[summary_key] = {
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "date": summary_date,
                    "type": summary_type,
                }

    save_state(state)

if __name__ == "__main__":
    main()
