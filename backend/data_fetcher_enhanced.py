"""Upstox-only NSE market data adapter used by the AI Studio terminal.

Credentials are read from UPSTOX_ACCESS_TOKEN (preferred) or UPSTOX_API_KEY.
The adapter deliberately uses the REST API directly so it works in the
lightweight AI Studio runtime without requiring the Upstox SDK.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

IST = "Asia/Kolkata"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# Display name -> Upstox instrument key. Add instruments here without changing
# the rest of the application.
NSE_SYMBOLS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFC BANK": "NSE_EQ|INE040A01034",
    "INFOSYS": "NSE_EQ|INE009A01021",
    "ICICI BANK": "NSE_EQ|INE090A01021",
    "KOTAK BANK": "NSE_EQ|INE237A01028",
    "AXIS BANK": "NSE_EQ|INE238A01034",
    "SBI": "NSE_EQ|INE062A01020",
    "LT": "NSE_EQ|INE018A01030",
    "WIPRO": "NSE_EQ|INE075A01022",
    "HCL TECH": "NSE_EQ|INE860A01027",
    "BAJAJ FINANCE": "NSE_EQ|INE296A01024",
    "MARUTI": "NSE_EQ|INE585B01010",
    "TITAN": "NSE_EQ|INE280A01028",
}

INTERVALS = {"1m": "1minute", "5m": "5minute", "15m": "15minute", "30m": "30minute", "60m": "60minute", "1d": "day"}
YF_PERIOD = {k: "upstox" for k in INTERVALS}


def _token() -> str:
    token = os.getenv("UPSTOX_ACCESS_TOKEN") or os.getenv("UPSTOX_API_KEY")
    if not token:
        raise RuntimeError("Set UPSTOX_ACCESS_TOKEN in AI Studio Secrets before fetching market data.")
    return token


def _get(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json", "Authorization": f"Bearer {_token()}"})
    try:
        with urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstox HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach Upstox: {exc.reason}") from exc
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("errors", payload.get("message", "Upstox request failed")))
    return payload


def _url(key: str, interval: str, end: date, start: date) -> str:
    iv = INTERVALS.get(interval, interval)
    encoded = quote(key, safe="")
    if iv == "day":
        return f"https://api.upstox.com/v2/historical-candle/{encoded}/day/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
    return f"https://api.upstox.com/v2/historical-candle/{encoded}/{iv}/{end:%Y-%m-%d}/{start:%Y-%m-%d}"


def fetch_ohlcv(symbol: str, interval: str = "5m", source: str = "upstox"):
    key = symbol if "|" in symbol else NSE_SYMBOLS.get(symbol)
    if not key:
        raise ValueError(f"Unknown NSE symbol: {symbol}")
    # Upstox historical endpoints support a bounded lookback. Keep requests
    # small and predictable for AI Studio; daily data gets a longer window.
    days = 365 * 5 if interval == "1d" else (30 if interval == "1m" else 180)
    payload = _get(_url(key, interval, date.today(), date.today() - timedelta(days=days)))
    rows = list(reversed(payload.get("data", {}).get("candles", [])))
    if not rows:
        raise RuntimeError(f"Upstox returned no candles for {symbol} @ {interval}")
    records = []
    for candle in rows:
        if len(candle) < 6:
            continue
        records.append({"timestamps": pd.to_datetime(candle[0]).tz_localize(None), "open": float(candle[1]), "high": float(candle[2]), "low": float(candle[3]), "close": float(candle[4]), "volume": float(candle[5] or 0)})
    df = pd.DataFrame(records).dropna().sort_values("timestamps").drop_duplicates("timestamps").reset_index(drop=True)
    if interval != "1d":
        times = df.timestamps.dt.strftime("%H:%M")
        df = df[(times >= MARKET_OPEN) & (times <= MARKET_CLOSE)].reset_index(drop=True)
    return df, "upstox"


def is_upstox_connected() -> bool:
    return bool(os.getenv("UPSTOX_ACCESS_TOKEN") or os.getenv("UPSTOX_API_KEY"))


def set_upstox_session(*_args, **_kwargs):
    """Compatibility hook: AI Studio credentials are supplied via env vars."""
    return is_upstox_connected()


def calculate_orb(df: pd.DataFrame, orb_minutes: int = 15) -> dict:
    out = {}
    work = df.copy()
    work["_date"] = work.timestamps.dt.date
    work["_mins"] = work.timestamps.dt.hour * 60 + work.timestamps.dt.minute
    start = 9 * 60 + 15
    for d, group in work.groupby("_date"):
        window = group[(group._mins >= start) & (group._mins < start + orb_minutes)]
        if not window.empty:
            out[str(d)] = {"high": float(window.high.max()), "low": float(window.low.min())}
    return out


def safe_float(value) -> float:
    try:
        value = float(value)
        return 0.0 if not math.isfinite(value) else round(value, 4)
    except (TypeError, ValueError):
        return 0.0


def df_to_records(df: pd.DataFrame) -> list:
    return [{"timestamp": r.timestamps.isoformat(), "open": safe_float(r.open), "high": safe_float(r.high), "low": safe_float(r.low), "close": safe_float(r.close), "volume": safe_float(r.volume)} for r in df.itertuples()]
