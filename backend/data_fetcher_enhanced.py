"""
Enhanced NSE data fetcher — yfinance primary, Kite live backup.

Priority:
  1. yfinance  — always tried first (no auth required, historical data)
  2. Kite      — used when:
       a. yfinance returns stale/empty data (market is live)
       b. caller explicitly requests source='kite'
       c. interval is '1m' (yfinance only keeps 7 days of 1m; Kite is real-time)

Kite session is managed as a module-level singleton so credentials are set
once via /api/kite-connect and reused across all requests.
"""

import math
import warnings
from datetime import datetime, time as dt_time, timedelta

import pandas as pd
import pytz

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN  = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)

NSE_SYMBOLS = {
    # Indices
    "NIFTY 50":       "^NSEI",
    "BANK NIFTY":     "^NSEBANK",
    "NIFTY IT":       "^CNXIT",
    "FIN NIFTY":      "NIFTY_FIN_SERVICE.NS",
    "MIDCAP NIFTY":   "^NSEMDCP50",
    # Large cap equities
    "RELIANCE":       "RELIANCE.NS",
    "TCS":            "TCS.NS",
    "HDFC BANK":      "HDFCBANK.NS",
    "INFOSYS":        "INFY.NS",
    "ICICI BANK":     "ICICIBANK.NS",
    "KOTAK BANK":     "KOTAKBANK.NS",
    "AXIS BANK":      "AXISBANK.NS",
    "SBI":            "SBIN.NS",
    "LT":             "LT.NS",
    "WIPRO":          "WIPRO.NS",
    "HCL TECH":       "HCLTECH.NS",
    "BAJAJ FINANCE":  "BAJFINANCE.NS",
    "MARUTI":         "MARUTI.NS",
    "TITAN":          "TITAN.NS",
    "ASIAN PAINTS":   "ASIANPAINT.NS",
}

# yfinance period per interval
YF_PERIOD = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1d":  "max",
}

# Kite interval strings
KITE_INTERVAL = {
    "1m":  "minute",
    "5m":  "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1d":  "day",
}

# ── Kite session singleton ────────────────────────────────────────────────────

_kite_session = None          # KiteConnect instance
_kite_instruments = None      # cached instrument list


def set_kite_session(api_key: str, access_token: str):
    """Call once after Kite OAuth to store the authenticated session."""
    global _kite_session, _kite_instruments
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    _kite_session = kite
    _kite_instruments = None  # reset cache


def get_kite_session():
    return _kite_session


def is_kite_connected() -> bool:
    return _kite_session is not None


def _kite_token(symbol_ns: str) -> int | None:
    """Resolve NSE trading symbol → Kite instrument_token."""
    global _kite_instruments
    if _kite_session is None:
        return None
    if _kite_instruments is None:
        _kite_instruments = pd.DataFrame(_kite_session.instruments("NSE"))

    sym = symbol_ns.replace(".NS", "").upper()
    # Handle indices differently — Kite uses NSE:NIFTY 50 etc.
    index_map = {
        "^NSEI":    ("NSE", "NIFTY 50"),
        "^NSEBANK": ("NSE", "NIFTY BANK"),
        "^CNXIT":   ("NSE", "NIFTY IT"),
    }
    if symbol_ns in index_map:
        exch, name = index_map[symbol_ns]
        match = _kite_instruments[_kite_instruments["name"].str.upper() == name.upper()]
    else:
        match = _kite_instruments[_kite_instruments["tradingsymbol"] == sym]

    return int(match.iloc[0]["instrument_token"]) if not match.empty else None


# ── yfinance fetch ────────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    period = YF_PERIOD.get(interval, "60d")
    raw = yf.download(symbol, period=period, interval=interval,
                      progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(IST)

    # Filter to market hours for intraday
    if interval != "1d":
        raw = raw.between_time("09:15", "15:30")

    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw = raw.reset_index().rename(columns={"Datetime": "timestamps", "index": "timestamps"})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    raw = raw[["timestamps", "open", "high", "low", "close", "volume"]]
    return raw.sort_values("timestamps").reset_index(drop=True)


# ── Kite fetch ────────────────────────────────────────────────────────────────

def _fetch_kite(symbol: str, interval: str, days: int = 60) -> pd.DataFrame:
    if _kite_session is None:
        raise RuntimeError("Kite not connected — call set_kite_session() first.")

    token = _kite_token(symbol)
    if token is None:
        raise ValueError(f"Could not resolve Kite instrument token for {symbol}")

    kite_interval = KITE_INTERVAL.get(interval, "5minute")
    end   = datetime.now()
    start = end - timedelta(days=days)
    chunks = []

    # Kite max window per call: 60 days for minute data
    step = 60
    cur  = start
    while cur < end:
        nxt  = min(cur + timedelta(days=step), end)
        data = _kite_session.historical_data(token, cur, nxt, kite_interval)
        chunks.extend(data)
        cur  = nxt

    if not chunks:
        raise ValueError(f"Kite returned no data for {symbol} @ {interval}")

    df = pd.DataFrame(chunks)
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
    return df.sort_values("timestamps").reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str = "5m",
                source: str = "auto") -> tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV data with fallback strategy.

    Parameters
    ----------
    symbol   : yfinance-compatible symbol (e.g. '^NSEI', 'RELIANCE.NS')
    interval : '1m' | '5m' | '15m' | '30m' | '60m' | '1d'
    source   : 'auto' | 'yfinance' | 'kite'

    Returns
    -------
    (df, source_used)  where source_used is 'yfinance' or 'kite'
    """
    errors = []

    # ── explicit source ────────────────────────────────────────────────────
    if source == "kite":
        if not is_kite_connected():
            raise RuntimeError("Kite not connected. Use /api/kite-connect first.")
        return _fetch_kite(symbol, interval), "kite"

    if source == "yfinance":
        return _fetch_yfinance(symbol, interval), "yfinance"

    # ── auto: try yfinance first ───────────────────────────────────────────
    try:
        df = _fetch_yfinance(symbol, interval)
        if df.empty:
            raise ValueError("yfinance returned empty dataframe")

        # If Kite is connected and market is live, top-up with Kite's latest bar
        if is_kite_connected() and _market_is_open():
            try:
                kite_df = _fetch_kite(symbol, "1m" if interval == "1m" else interval, days=1)
                if not kite_df.empty:
                    df = _merge_topup(df, kite_df)
            except Exception:
                pass  # Kite top-up is best-effort

        return df, "yfinance"

    except Exception as e:
        errors.append(f"yfinance: {e}")

    # ── fallback: Kite ─────────────────────────────────────────────────────
    if is_kite_connected():
        try:
            return _fetch_kite(symbol, interval), "kite"
        except Exception as e:
            errors.append(f"kite: {e}")

    raise RuntimeError(
        f"All data sources failed for {symbol} @ {interval}:\n" +
        "\n".join(errors)
    )


def _market_is_open() -> bool:
    now = datetime.now(IST).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def _merge_topup(hist: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Append live rows that are newer than the last historical row."""
    cutoff = hist["timestamps"].max()
    new = live[live["timestamps"] > cutoff]
    if new.empty:
        return hist
    combined = pd.concat([hist, new], ignore_index=True)
    return combined.sort_values("timestamps").reset_index(drop=True)


# ── ORB ───────────────────────────────────────────────────────────────────────

def calculate_orb(df: pd.DataFrame, orb_minutes: int = 15) -> dict:
    """
    Opening Range Breakout levels per trading day.
    NSE open: 09:15 IST. Window = first orb_minutes after open.
    Returns {date_str: {'high': float, 'low': float}}.
    """
    df = df.copy()
    df["_date"] = df["timestamps"].dt.date
    df["_time"] = df["timestamps"].dt.time

    end_time = (datetime.combine(datetime.today(), MARKET_OPEN)
                + timedelta(minutes=orb_minutes)).time()

    orb_levels = {}
    for date, day_df in df.groupby("_date"):
        window = day_df[
            (day_df["_time"] >= MARKET_OPEN) & (day_df["_time"] < end_time)
        ]
        if not window.empty:
            orb_levels[str(date)] = {
                "high": float(window["high"].max()),
                "low":  float(window["low"].min()),
            }
    return orb_levels


def safe_float(v) -> float:
    """JSON-safe float — replaces NaN/Inf with 0.0."""
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def df_to_records(df: pd.DataFrame) -> list:
    return [
        {
            "timestamp": row["timestamps"].isoformat(),
            "open":   safe_float(row["open"]),
            "high":   safe_float(row["high"]),
            "low":    safe_float(row["low"]),
            "close":  safe_float(row["close"]),
            "volume": safe_float(row.get("volume", 0)),
        }
        for _, row in df.iterrows()
    ]
