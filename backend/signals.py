"""
Trading signal generator — converts Kronos forecast + indicators into
structured buy/sell signals with entry, stop-loss, and target levels.

Signal structure
----------------
{
  direction:    "BUY" | "SELL" | "NEUTRAL"
  confidence:   float 0-100           (% of indicators agreeing)
  entry:        float                  (suggested entry price)
  stop_loss:    float                  (1.5× ATR below/above entry)
  target_1:     float                  (1× ATR move from entry)
  target_2:     float                  (2× ATR move from entry)
  risk_reward:  float                  (target_2 / stop distance)
  reasoning:    list[str]              (plain English explanation)
  forecast_chg: float                  (Kronos predicted % change)
  trend_bias:   str                    (indicator consensus)
}
"""

import math
import pandas as pd
from indicators import get_latest_summary, compute_atr, _safe


def generate_signal(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    orb_levels: dict | None = None,
) -> dict:
    """
    Generate a structured trading signal.

    Parameters
    ----------
    df       : historical OHLCV DataFrame
    pred_df  : Kronos forecast DataFrame (output of KronosPredictor.predict)
    orb_levels : ORB dict from data_fetcher_enhanced.calculate_orb

    Returns
    -------
    Signal dict (JSON-serialisable)
    """
    summary = get_latest_summary(df)
    last_close = float(df["close"].iloc[-1])

    # ── Kronos forecast direction ─────────────────────────────────────────
    if pred_df is not None and not pred_df.empty:
        pred_close  = float(pred_df["close"].iloc[-1])
        forecast_chg = (pred_close - last_close) / last_close * 100
        kronos_dir   = "BUY" if pred_close > last_close else "SELL"
    else:
        pred_close   = last_close
        forecast_chg = 0.0
        kronos_dir   = "NEUTRAL"

    # ── ATR for stop/target sizing ────────────────────────────────────────
    atr_series = compute_atr(df)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else last_close * 0.005

    # ── Combine Kronos + indicator signals ────────────────────────────────
    all_signals = summary["signals"] + [{"name": "Kronos Forecast", "bias": kronos_dir}]
    buy_votes   = sum(1 for s in all_signals if s["bias"] == "BUY")
    sell_votes  = sum(1 for s in all_signals if s["bias"] == "SELL")
    total_votes = len(all_signals)

    if buy_votes > sell_votes:
        direction  = "BUY"
        confidence = round(buy_votes / total_votes * 100, 1)
    elif sell_votes > buy_votes:
        direction  = "SELL"
        confidence = round(sell_votes / total_votes * 100, 1)
    else:
        direction  = "NEUTRAL"
        confidence = 50.0

    # ── Entry, stop-loss, targets ─────────────────────────────────────────
    entry = last_close
    if direction == "BUY":
        stop_loss = round(entry - 1.5 * atr, 2)
        target_1  = round(entry + 1.0 * atr, 2)
        target_2  = round(entry + 2.5 * atr, 2)
    elif direction == "SELL":
        stop_loss = round(entry + 1.5 * atr, 2)
        target_1  = round(entry - 1.0 * atr, 2)
        target_2  = round(entry - 2.5 * atr, 2)
    else:
        stop_loss = round(entry - atr, 2)
        target_1  = round(entry + atr, 2)
        target_2  = round(entry + 2 * atr, 2)

    stop_dist  = abs(entry - stop_loss)
    target_dist = abs(target_2 - entry)
    risk_reward = round(target_dist / stop_dist, 2) if stop_dist > 0 else 0.0

    # ── ORB context ───────────────────────────────────────────────────────
    orb_note = []
    if orb_levels:
        latest_orb = orb_levels.get(max(orb_levels.keys()), {})
        orb_h = latest_orb.get("high", 0)
        orb_l = latest_orb.get("low", 0)
        if orb_h and last_close > orb_h:
            orb_note.append(f"Price above ORB High (₹{orb_h:,.2f}) — bullish breakout")
        elif orb_l and last_close < orb_l:
            orb_note.append(f"Price below ORB Low (₹{orb_l:,.2f}) — bearish breakdown")
        elif orb_h and orb_l:
            orb_note.append(f"Price inside ORB range (₹{orb_l:,.2f}–₹{orb_h:,.2f}) — wait for breakout")

    # ── Plain English reasoning ───────────────────────────────────────────
    reasoning = []

    reasoning.append(
        f"Kronos forecast: {kronos_dir} ({forecast_chg:+.2f}% predicted change)"
    )

    rsi = summary.get("rsi")
    if rsi:
        if rsi < 30:   reasoning.append(f"RSI at {rsi:.1f} — oversold, potential bounce")
        elif rsi > 70: reasoning.append(f"RSI at {rsi:.1f} — overbought, possible pullback")
        else:          reasoning.append(f"RSI at {rsi:.1f} — neutral momentum")

    if summary.get("ema9") and summary.get("ema21"):
        if summary["ema9"] > summary["ema21"]:
            reasoning.append("EMA9 above EMA21 — short-term trend bullish")
        else:
            reasoning.append("EMA9 below EMA21 — short-term trend bearish")

    if summary.get("st_direction") == 1:
        reasoning.append("SuperTrend signal: uptrend")
    else:
        reasoning.append("SuperTrend signal: downtrend")

    reasoning.extend(orb_note)
    reasoning.append(
        f"{buy_votes}/{total_votes} indicators bullish, {sell_votes}/{total_votes} bearish"
    )

    return {
        "direction":    direction,
        "confidence":   confidence,
        "entry":        round(entry, 2),
        "stop_loss":    stop_loss,
        "target_1":     target_1,
        "target_2":     target_2,
        "risk_reward":  risk_reward,
        "atr":          round(atr, 2),
        "forecast_chg": round(forecast_chg, 3),
        "kronos_dir":   kronos_dir,
        "trend_bias":   summary.get("overall", "NEUTRAL"),
        "buy_votes":    buy_votes,
        "sell_votes":   sell_votes,
        "total_votes":  total_votes,
        "reasoning":    reasoning,
        "indicator_signals": all_signals,
    }
