"""Upstox-only market data adapter for the AI Studio terminal."""
from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

IST = "Asia/Kolkata"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
NSE_SYMBOLS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50", "BANK NIFTY": "NSE_INDEX|Nifty Bank", "NIFTY IT": "NSE_INDEX|Nifty IT",
    "RELIANCE": "NSE_EQ|INE002A01018", "TCS": "NSE_EQ|INE467B01029", "HDFC BANK": "NSE_EQ|INE040A01034",
    "INFOSYS": "NSE_EQ|INE009A01021", "ICICI BANK": "NSE_EQ|INE090A01021", "KOTAK BANK": "NSE_EQ|INE237A01028",
    "AXIS BANK": "NSE_EQ|INE238A01034", "SBI": "NSE_EQ|INE062A01020", "LT": "NSE_EQ|INE018A01030",
    "WIPRO": "NSE_EQ|INE075A01022", "HCL TECH": "NSE_EQ|INE860A01027", "BAJAJ FINANCE": "NSE_EQ|INE296A01024",
    "MARUTI": "NSE_EQ|INE585B01010", "TITAN": "NSE_EQ|INE280A01028",
}
INTERVALS = {"1m":"1minute", "5m":"5minute", "15m":"15minute", "30m":"30minute", "60m":"60minute", "1d":"day"}
YF_PERIOD = {key: "upstox" for key in INTERVALS}


def _token() -> str:
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is required. Add a valid Upstox OAuth access token to AI Studio Secrets.")
    return token


def _get(url: str) -> dict:
    request = Request(url, headers={"Accept":"application/json", "Authorization":f"Bearer {_token()}"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstox HTTP {exc.code}: {detail[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach Upstox: {exc.reason}") from exc
    if payload.get("status") != "success":
        errors = payload.get("errors") or payload.get("message") or "Upstox request failed"
        raise RuntimeError(str(errors))
    return payload


def _url(key: str, interval: str, end: date, start: date) -> str:
    encoded = quote(key, safe="")
    api_interval = INTERVALS.get(interval, interval)
    return f"https://api.upstox.com/v2/historical-candle/{encoded}/{api_interval}/{end:%Y-%m-%d}/{start:%Y-%m-%d}"


def fetch_ohlcv(symbol: str, interval: str = "5m", source: str = "upstox"):
    key = symbol if "|" in symbol else NSE_SYMBOLS.get(symbol)
    if not key:
        raise ValueError(f"Unknown NSE symbol: {symbol}")
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    days = 365 * 5 if interval == "1d" else (7 if interval == "1m" else 180)
    candles = list(reversed(_get(_url(key, interval, date.today(), date.today() - timedelta(days=days))).get("data", {}).get("candles", [])))
    records = []
    for candle in candles:
        if len(candle) < 6:
            continue
        timestamp = pd.to_datetime(candle[0], utc=True).tz_convert(IST).tz_localize(None)
        records.append({"timestamps": timestamp, "open": float(candle[1]), "high": float(candle[2]), "low": float(candle[3]), "close": float(candle[4]), "volume": float(candle[5] or 0)})
    df = pd.DataFrame(records).dropna().drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Upstox returned no candles for {symbol} @ {interval}")
    if interval != "1d":
        times = df.timestamps.dt.strftime("%H:%M")
        df = df[(times >= MARKET_OPEN) & (times <= MARKET_CLOSE)].reset_index(drop=True)
    return df, "upstox"


def is_upstox_connected() -> bool:
    return bool(os.getenv("UPSTOX_ACCESS_TOKEN"))


def set_upstox_session(*_args, **_kwargs):
    return is_upstox_connected()


def calculate_orb(df: pd.DataFrame, orb_minutes: int = 15) -> dict:
    work = df.copy(); work["_date"] = work.timestamps.dt.date; work["_mins"] = work.timestamps.dt.hour * 60 + work.timestamps.dt.minute
    result = {}; start = 9 * 60 + 15
    for day, group in work.groupby("_date"):
        window = group[(group._mins >= start) & (group._mins < start + orb_minutes)]
        if not window.empty: result[str(day)] = {"high": float(window.high.max()), "low": float(window.low.min())}
    return result


def safe_float(value) -> float:
    try:
        number = float(value)
        return 0.0 if not math.isfinite(number) else round(number, 4)
    except (TypeError, ValueError):
        return 0.0


def df_to_records(df: pd.DataFrame) -> list:
    return [{"timestamp": r.timestamps.isoformat(), "open": safe_float(r.open), "high": safe_float(r.high), "low": safe_float(r.low), "close": safe_float(r.close), "volume": safe_float(r.volume)} for r in df.itertuples()]
