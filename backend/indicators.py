"""
Technical indicators for NSE Dashboard.

All functions accept a pandas DataFrame with OHLCV columns and return
a dict of {indicator_name: list[float | None]} ready for JSON serialisation.
"""

import math
import numpy as np
import pandas as pd


def _safe(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)


def compute_macd(close: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = ema(close, fast)
    ema_slow   = ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series,
                      period: int = 20, std_mult: float = 2.0
                      ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = sma(close, period)
    std   = close.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP reset per trading day."""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series([None] * len(df), index=df.index)

    df = df.copy()
    df["_date"] = df["timestamps"].dt.date
    df["_tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"]  = df["_tp"] * df["volume"]

    vwap = []
    for _, grp in df.groupby("_date", sort=False):
        cum_tpv = grp["_tpv"].cumsum()
        cum_vol = grp["volume"].cumsum().replace(0, np.nan)
        vwap.extend((cum_tpv / cum_vol).tolist())
    return pd.Series(vwap, index=df.index)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c  = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_supertrend(df: pd.DataFrame,
                       period: int = 10, multiplier: float = 3.0
                       ) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction: 1=up, -1=down."""
    atr  = compute_atr(df, period)
    hl2  = (df["high"] + df["low"]) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band  = upper_basic.copy()
    lower_band  = lower_basic.copy()
    supertrend  = pd.Series([None] * len(df), dtype=object)
    direction   = pd.Series([1]   * len(df))

    for i in range(1, len(df)):
        # Upper band
        if upper_basic.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Lower band
        if lower_basic.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Direction
        prev_st = supertrend.iloc[i - 1]
        if prev_st is None or prev_st == upper_band.iloc[i - 1]:
            if df["close"].iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i]  = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i]  = 1
        else:
            if df["close"].iloc[i] >= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i]  = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i]  = -1

    return supertrend.astype(float), direction


def get_all_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all indicators and return as JSON-ready dict.
    Each value is a list aligned to df rows (None where not enough history).
    """
    close = df["close"]
    result = {}

    # ── Trend ────────────────────────────────────────────────────────────
    result["ema9"]  = [_safe(v) for v in ema(close, 9)]
    result["ema21"] = [_safe(v) for v in ema(close, 21)]
    result["ema50"] = [_safe(v) for v in ema(close, 50)]
    result["vwap"]  = [_safe(v) for v in compute_vwap(df)]

    # ── Momentum ─────────────────────────────────────────────────────────
    result["rsi"]   = [_safe(v) for v in compute_rsi(close)]

    macd_line, signal_line, histogram = compute_macd(close)
    result["macd"]        = [_safe(v) for v in macd_line]
    result["macd_signal"] = [_safe(v) for v in signal_line]
    result["macd_hist"]   = [_safe(v) for v in histogram]

    # ── Volatility ───────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = compute_bollinger(close)
    result["bb_upper"] = [_safe(v) for v in bb_upper]
    result["bb_mid"]   = [_safe(v) for v in bb_mid]
    result["bb_lower"] = [_safe(v) for v in bb_lower]
    result["atr"]      = [_safe(v) for v in compute_atr(df)]

    # ── SuperTrend ───────────────────────────────────────────────────────
    try:
        st_line, st_dir = compute_supertrend(df)
        result["supertrend"]  = [_safe(v) for v in st_line]
        result["st_direction"] = st_dir.tolist()
    except Exception:
        result["supertrend"]   = [None] * len(df)
        result["st_direction"] = [1]    * len(df)

    return result


def get_latest_summary(df: pd.DataFrame) -> dict:
    """Return a single-row summary of latest indicator values + signals."""
    inds = get_all_indicators(df)
    close = float(df["close"].iloc[-1])

    def last(key):
        vals = [v for v in inds.get(key, []) if v is not None]
        return vals[-1] if vals else None

    rsi      = last("rsi")
    macd     = last("macd")
    macd_sig = last("macd_signal")
    ema9     = last("ema9")
    ema21    = last("ema21")
    bb_upper = last("bb_upper")
    bb_lower = last("bb_lower")
    st_dir   = inds["st_direction"][-1] if inds["st_direction"] else 1

    signals = []
    if rsi is not None:
        if rsi < 30:   signals.append({"name": "RSI Oversold",   "bias": "BUY"})
        elif rsi > 70: signals.append({"name": "RSI Overbought", "bias": "SELL"})

    if macd is not None and macd_sig is not None:
        if macd > macd_sig: signals.append({"name": "MACD Bullish Cross", "bias": "BUY"})
        else:               signals.append({"name": "MACD Bearish Cross", "bias": "SELL"})

    if ema9 is not None and ema21 is not None:
        if ema9 > ema21: signals.append({"name": "EMA9 > EMA21", "bias": "BUY"})
        else:            signals.append({"name": "EMA9 < EMA21", "bias": "SELL"})

    if bb_upper and bb_lower:
        if close > bb_upper: signals.append({"name": "Price > BB Upper", "bias": "SELL"})
        elif close < bb_lower: signals.append({"name": "Price < BB Lower", "bias": "BUY"})

    signals.append({"name": "SuperTrend", "bias": "BUY" if st_dir == 1 else "SELL"})

    buy_count  = sum(1 for s in signals if s["bias"] == "BUY")
    sell_count = sum(1 for s in signals if s["bias"] == "SELL")
    overall    = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "NEUTRAL"

    return {
        "rsi":        rsi,
        "macd":       macd,
        "macd_signal": macd_sig,
        "ema9":       ema9,
        "ema21":      ema21,
        "bb_upper":   bb_upper,
        "bb_lower":   bb_lower,
        "st_direction": st_dir,
        "signals":    signals,
        "overall":    overall,
        "buy_count":  buy_count,
        "sell_count": sell_count,
    }
