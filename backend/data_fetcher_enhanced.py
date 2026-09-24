"""Upstox-only OHLCV provider for the NSE terminal.

This provider reads the bearer token from the environment at runtime, so no
secret is stored in source or logs. The terminal intentionally supports only
Upstox for market data; legacy Kite/yfinance references are compatibility
aliases.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

NSE_SYMBOLS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "FIN NIFTY": "NSE_INDEX|Nifty Financial Services",
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
    "HCL TECH": "NSE_EQ|INE860A01011",
    "BAJAJ FINANCE": "NSE_EQ|INE296A01024",
    "MARUTI": "NSE_EQ|INE585B01010",
    "TITAN": "NSE_EQ|INE280A01028",
    "ASIAN PAINTS": "NSE_EQ|INE021A01026",
}

UPSTOX_INTERVALS = {
    "1m": "1minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1d": "day",
}

UPSTOX_PERIOD_DAYS = {
    "1m": 7,
    "5m": 90,
    "15m": 180,
    "30m": 180,
    "60m": 365,
    "1d": 365 * 3,
}


def _upstox_access_token() -> str:
    token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured in AI Studio Secrets.")
    return token


def _normalize_symbol(symbol: str) -> str:
    if "|" in symbol:
        return symbol
    symbol = symbol.strip()
    return NSE_SYMBOLS.get(symbol, symbol)


def _fetch_url(symbol: str, interval: str) -> str:
    key = _normalize_symbol(symbol)
    if interval == "1m":
        return f"https://api.upstox.com/v2/historical-candle/intraday/{quote(key, safe='')}/1minute"
    return f"https://api.upstox.com/v2/historical-candle/{quote(key, safe='')}/{UPSTOX_INTERVALS[interval]}"


def _request_json(url: str):
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + _upstox_access_token(),
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upstox HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Upstox connection failed: {exc.reason}") from exc

    if payload.get("status") != "success":
        msg = payload.get("message") or payload.get("error") or payload.get("errors") or "Upstox request failed"
        raise RuntimeError(str(msg))
    return payload


def _df_from_candles(raw: list) -> pd.DataFrame:
    rows = []
    for candle in raw:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue
        ts_raw, o, h, l, c, v = candle[:6]
        try:
            ts = pd.to_datetime(ts_raw, utc=True).tz_convert(IST).tz_localize(None)
            rows.append(
                {
                    "timestamps": ts,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v or 0),
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
    return df


def fetch_ohlcv(symbol: str, interval: str = "5m", source: str = "auto") -> tuple[pd.DataFrame, str]:
    """Return (df, source_label) using Upstox as the only market data source."""
    if source not in {"auto", "upstox", "yfinance", "kite"}:
        raise ValueError(f"Unsupported source: {source}")

    key = _normalize_symbol(symbol)
    if not key:
        raise ValueError(f"Unknown NSE symbol: {symbol}")
    if interval not in UPSTOX_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    # Compatibility: legacy callers may still send yfinance/kite; they map to Upstox.
    if source in {"yfinance", "kite"}:
        source = "upstox"

    try:
        if interval == "1m":
            payload = _request_json(_fetch_url(key, interval))
            candles = payload.get("data", {}).get("candles", [])
        else:
            payload = _request_json(_fetch_url(key, interval))
            candles = payload.get("data", {}).get("candles", [])

        df = _df_from_candles(candles)
        if df.empty:
            raise RuntimeError(f"Upstox returned no candles for {symbol} @ {interval}")

        if interval != "1d":
            df = df[(df["timestamps"].dt.time >= MARKET_OPEN) & (df["timestamps"].dt.time <= MARKET_CLOSE)].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(f"Upstox returned no market-hours candles for {symbol} @ {interval}")

        return df, "upstox"
    except Exception as exc:
        raise RuntimeError(f"All data sources failed for {symbol} @ {interval}: {exc}") from exc


def is_upstox_connected() -> bool:
    return bool(os.getenv("UPSTOX_ACCESS_TOKEN", "").strip())


def set_upstox_session(*_args, **_kwargs) -> bool:
    return is_upstox_connected()


# Compatibility aliases for older code paths.
def is_kite_connected() -> bool:
    return is_upstox_connected()


def set_kite_session(*_args, **_kwargs) -> bool:
    return set_upstox_session(*_args, **_kwargs)


def calculate_orb(df: pd.DataFrame, orb_minutes: int = 15) -> dict:
    """Opening range breakout levels. Returns {date_str: {high, low}}."""
    view = df.copy()
    view["_date"] = view["timestamps"].dt.date
    view["_mins"] = view["timestamps"].dt.hour * 60 + view["timestamps"].dt.minute
    start_min = 9 * 60 + 15
    output = {}
    for day, grp in view.groupby("_date"):
        window = grp[(grp["_mins"] >= start_min) & (grp["_mins"] < start_min + orb_minutes)]
        if not window.empty:
            output[str(day)] = {"high": float(window["high"].max()), "low": float(window["low"].min())}
    return output


def safe_float(value) -> float:
    try:
        raw = float(value)
        return 0.0 if not math.isfinite(raw) else round(raw, 4)
    except (TypeError, ValueError):
        return 0.0


def df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "timestamp": row["timestamps"].isoformat(),
            "open": safe_float(row["open"]),
            "high": safe_float(row["high"]),
            "low": safe_float(row["low"]),
            "close": safe_float(row["close"]),
            "volume": safe_float(row.get("volume", 0)),
        }
        for _, row in df.iterrows()
    ]


# Legacy alias for the original project's naming.
fetch_data = fetch_ohlcv
