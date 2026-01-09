# SMA150 Telegram Bot

Python script that sends a daily Telegram update with tickers that are close to SMA150 (default: within ±5%).

## Quick start (macOS)

From the project folder:

```bash
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install requests
```

Create `secrets.env` next to `main.py`:

```env
TELEGRAM_BOT_TOKEN='PASTE_TOKEN_HERE'
TELEGRAM_CHAT_ID='123456789'   # or @username, or a group id (often starts with -100)
```

Edit `config.env` next to `main.py` (non-secret settings):

```env
SMA150_MAX_DIST_PCT=5   # change to 1 for +/- 1%
SMA_WINDOW_DAYS=150     # change to 200 / 300
```

Verify the bot can message you:

```bash
./venv/bin/python main.py --test
```

Send a status message now (manual test):

```bash
./venv/bin/python main.py --status --force
```

## What it does

- Downloads daily price data (default: Yahoo Finance; fallback: Stooq).
- Calculates `SMA150` (150-day simple moving average).
- Sends a “Daily Update” message once per NYSE trading day, but only includes tickers within a configurable distance from SMA150 (default: ±5%). Includes distance and a signal when relevant (`cross_up` / `cross_down` / `touch`). If the output is long, it’s split into multiple Telegram messages grouped by sector/interest.
- Writes `alert_state.json` to avoid sending more than once per day.

## Setup

Requirements: Python 3.9+ (recommended 3.11+).

```bash
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install requests
```

## Telegram config

The script automatically loads `config.env`, `secrets.env` and `.env` from the same directory as `main.py` (OS environment variables still override everything).

Alternative env var names are also supported:
- `TELEGRAM_TOKEN` instead of `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_USER` instead of `TELEGRAM_CHAT_ID`

## Run

Send a test message:

```bash
./venv/bin/python main.py --test
```

Normal run (silent; sends at most once per trading day):

```bash
./venv/bin/python main.py
```

Send status now (still once-per-day):

```bash
./venv/bin/python main.py --status
```

Force sending even if it already sent today (manual testing):

```bash
./venv/bin/python main.py --status --force
```

Dry-run (prints instead of sending, useful for debugging):

```bash
./venv/bin/python main.py --dry-run --status
```

First run tip: the first `--status` run may take ~30–60 seconds because it populates `history_cache/`. After that, re-runs are fast until the cache expires.

## Optional settings

- `SEND_DAILY_SUMMARY=1` (default) — if `0`, disables the daily summary message.
- `SMA150_MAX_DIST_PCT=5` (default) — absolute distance threshold (e.g. `5` means only `-5%..+5%`).
- `SMA_WINDOW_DAYS=150` (default) — moving average window (e.g. `200`, `300`).
- `SMA150_ONLY_ABOVE=0/1` (default: `0`) — if `1`, only includes tickers above the SMA (distance `0..+SMA150_MAX_DIST_PCT`).
- `HISTORY_CACHE_TTL_HOURS=20` (default) — caches downloads to avoid hitting rate limits when you re-run the script.
- `HISTORY_PROVIDERS=yahoo,stooq` (default) — provider order (try the first, then fall back).
- `YAHOO_USER_AGENT='Mozilla/5.0'` — override if Yahoo blocks your requests.
- `MARKET_TZ=America/New_York` (default) — timezone used to decide “trading day”.
- `DRY_RUN=1` — same as `--dry-run`.

## Tickers

Edit the `WATCHLIST` groups in `main.py` (the `SYMBOLS` list is generated automatically from it).

## macOS automation (launchd)

If you want it to run automatically, use `launchd` to run the script once per day (Mon–Fri). The script itself skips non-trading days (NYSE holiday calendar), and also won’t send twice on the same day.

- Plist: `~/Library/LaunchAgents/com.ronen.investing.sma150.plist`
- Logs:
  - `~/Library/Logs/investing-sma150.log`
  - `~/Library/Logs/investing-sma150.err.log`

Load / enable:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ronen.investing.sma150.plist
launchctl enable gui/$(id -u)/com.ronen.investing.sma150
```

Run immediately:

```bash
launchctl kickstart -k gui/$(id -u)/com.ronen.investing.sma150
```

Unload:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.ronen.investing.sma150.plist
```

To change the time/days, edit `StartCalendarInterval` in the plist and then `bootout` + `bootstrap` again.

## Troubleshooting

- “`python main.py` does nothing”: this is expected when it already sent today, or when it’s not a NYSE trading day. Use `./venv/bin/python main.py --dry-run --status --force` to see what it would send.
- “Rate limit reached”: a data provider blocked you temporarily. Wait and try again (the cache helps a lot once it’s populated).
- “Requests is missing”: run `./venv/bin/pip install requests`.
- Telegram chat id: personal chats are numeric, usernames start with `@`, and group ids often start with `-100`.

## Notes

- Data comes from a public source (Stooq) and may be delayed/limited.
- Not financial advice.
