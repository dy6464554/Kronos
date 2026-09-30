"""
Walk-forward backtesting engine for Kronos NSE forecasts.

Methodology
-----------
Walk-forward testing:  at each step we use a rolling window of `lookback`
candles as context, generate a Kronos forecast for `pred_len` candles,
then compare the predicted close at horizon H against the actual close.

Metrics computed
----------------
  hit_rate        : % of forecasts where predicted direction == actual direction
  mae             : mean absolute error of close price (₹)
  mape            : mean absolute percentage error (%)
  rmse            : root mean squared error
  avg_return      : average simulated trade return per signal
  sharpe_ratio    : annualised Sharpe of trade returns
  max_drawdown    : maximum peak-to-trough drawdown in the simulated equity curve
  total_pnl       : total simulated P&L assuming fixed position size of ₹1,00,000

All metrics are computed on out-of-sample windows only.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    lookback:        int   = 200    # historical candles fed to Kronos
    pred_len:        int   = 20     # candles forecast per step
    horizon:         int   = 10     # which forecast candle to evaluate (1..pred_len)
    step:            int   = 10     # how many candles to slide between steps
    temperature:     float = 1.0
    top_p:           float = 0.9
    sample_count:    int   = 1
    position_size:   float = 100_000.0   # ₹ per trade
    transaction_cost: float = 0.0003     # 0.03% per side (typical NSE brokerage)


@dataclass
class BacktestResult:
    n_trades:       int   = 0
    hit_rate:       float = 0.0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    net_points:     float = 0.0
    gross_profit:   float = 0.0
    gross_loss:     float = 0.0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    mae:            float = 0.0
    mape:           float = 0.0
    rmse:           float = 0.0
    avg_return:     float = 0.0
    sharpe_ratio:   float = 0.0
    max_drawdown:   float = 0.0
    total_pnl:      float = 0.0
    equity_curve:   list  = field(default_factory=list)
    trade_log:      list  = field(default_factory=list)
    error:          Optional[str] = None



# ── Core engine ───────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    predictor,
    config: BacktestConfig = None,
    progress_cb=None,
) -> BacktestResult:
    """
    Run a walk-forward backtest.

    Parameters
    ----------
    df          : full historical OHLCV DataFrame (must be long enough)
    predictor   : KronosPredictor instance (already loaded)
    config      : BacktestConfig
    progress_cb : optional callable(pct: float, msg: str) for UI updates

    Returns
    -------
    BacktestResult with all metrics populated
    """
    cfg = config or BacktestConfig()
    result = BacktestResult()

    min_rows = cfg.lookback + cfg.pred_len
    if len(df) < min_rows:
        result.error = f"Need ≥{min_rows} candles for backtesting, have {len(df)}."
        return result

    # ── Determine test windows ──────────────────────────────────────────
    # We leave the last pred_len candles as the final out-of-sample window.
    # Walk starts at index lookback, steps by cfg.step.
    start_indices = range(cfg.lookback, len(df) - cfg.pred_len, cfg.step)
    total = len(start_indices)
    if total == 0:
        result.error = "No test windows available with current config."
        return result

    errors_sq, abs_errors, pct_errors, returns = [], [], [], []
    correct_dir = 0
    equity = cfg.position_size
    equity_curve = [equity]
    trade_log = []

    cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")

    for i, start_idx in enumerate(start_indices):
        if progress_cb:
            progress_cb(i / total, f"Backtesting window {i+1}/{total}…")

        # Context window
        ctx  = df.iloc[start_idx - cfg.lookback : start_idx].copy()
        x_df = ctx[cols].copy()
        x_ts = ctx["timestamps"].reset_index(drop=True)

        # Future timestamps
        freq = _infer_freq(df)
        y_ts = pd.Series(
            pd.date_range(start=x_ts.iloc[-1] + freq, periods=cfg.pred_len, freq=freq)
        )

        # Forecast
        try:
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=cfg.pred_len,
                T=cfg.temperature, top_p=cfg.top_p,
                sample_count=cfg.sample_count,
            )
            pred_df = pred_df.fillna(0.0)
        except Exception as e:
            continue

        # Actual candle at horizon H
        actual_idx = start_idx + cfg.horizon - 1
        if actual_idx >= len(df):
            continue

        pred_close   = float(pred_df["close"].iloc[cfg.horizon - 1])
        actual_close = float(df["close"].iloc[actual_idx])
        last_close   = float(ctx["close"].iloc[-1])

        # Skip degenerate predictions
        if pred_close == 0 or last_close == 0:
            continue

        # ── Errors ──────────────────────────────────────────────────────
        err   = pred_close - actual_close
        errors_sq.append(err ** 2)
        abs_errors.append(abs(err))
        pct_errors.append(abs(err) / actual_close * 100)

        # ── Directional accuracy ─────────────────────────────────────────
        pred_dir   = 1 if pred_close > last_close else -1
        actual_dir = 1 if actual_close > last_close else -1
        if pred_dir == actual_dir:
            correct_dir += 1

        # ── Simulated trade ──────────────────────────────────────────────
        # Enter at last_close in direction of prediction,
        # exit at actual_close at horizon H.
        # Apply round-trip transaction cost.
        raw_ret     = pred_dir * (actual_close - last_close) / last_close
        cost        = 2 * cfg.transaction_cost
        net_ret     = raw_ret - cost
        trade_pnl   = cfg.position_size * net_ret
        trade_pts   = pred_dir * (actual_close - last_close)
        equity     += trade_pnl
        returns.append(net_ret)
        equity_curve.append(equity)

        trade_log.append({
            "window":       i,
            "entry_ts":     str(x_ts.iloc[-1]),
            "last_close":   round(last_close, 2),
            "pred_close":   round(pred_close, 2),
            "actual_close": round(actual_close, 2),
            "direction":    "UP" if pred_dir == 1 else "DN",
            "correct":      pred_dir == actual_dir,
            "net_points":   round(trade_pts, 2),
            "net_return":   round(net_ret * 100, 3),
            "pnl":          round(trade_pnl, 2),
        })

    n = len(trade_log)
    if n == 0:
        result.error = "No completed trades in backtest."
        return result

    returns_arr = np.array(returns)
    pnls = [t["pnl"] for t in trade_log]
    pts = [t["net_points"] for t in trade_log]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    # ── Aggregate metrics ────────────────────────────────────────────────
    result.n_trades       = n
    result.hit_rate       = round(correct_dir / n * 100, 2)
    result.win_rate       = round(len(wins) / n * 100, 2)
    result.winning_trades = len(wins)
    result.losing_trades  = len(losses)
    result.gross_profit   = round(sum(wins), 2)
    result.gross_loss     = round(sum(losses), 2)
    result.profit_factor  = round(sum(wins) / sum(losses), 2) if sum(losses) > 0 else (999.0 if len(wins) > 0 else 0.0)
    result.net_points     = round(float(sum(pts)), 2)
    result.mae            = round(float(np.mean(abs_errors)), 2)
    result.mape           = round(float(np.mean(pct_errors)), 3)
    result.rmse           = round(float(np.sqrt(np.mean(errors_sq))), 2)
    result.avg_return     = round(float(np.mean(returns_arr)) * 100, 3)
    result.total_pnl      = round(equity - cfg.position_size, 2)
    result.equity_curve   = [round(v, 2) for v in equity_curve]
    result.trade_log      = trade_log

    # Sharpe (annualised assuming ~75 candles/day for 5-min)
    if returns_arr.std() > 0:
        candles_per_year  = 75 * 250
        steps_per_year    = candles_per_year / cfg.step
        result.sharpe_ratio = round(
            float(returns_arr.mean() / returns_arr.std() * math.sqrt(steps_per_year)), 3
        )

    # Max drawdown
    eq = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    result.max_drawdown = round(float(dd.min()) * 100, 2)

    return result


def _infer_freq(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(minutes=5)
    diffs = df["timestamps"].diff().dropna()
    short = diffs[diffs < pd.Timedelta(hours=4)]
    mode  = short.mode()
    return mode.iloc[0] if len(mode) else diffs.median()


def result_to_dict(r: BacktestResult) -> dict:
    """Serialise BacktestResult to a JSON-safe dict."""
    return {
        "n_trades":       r.n_trades,
        "hit_rate":       r.hit_rate,
        "win_rate":       r.win_rate,
        "profit_factor":  r.profit_factor,
        "net_points":     r.net_points,
        "gross_profit":   r.gross_profit,
        "gross_loss":     r.gross_loss,
        "winning_trades": r.winning_trades,
        "losing_trades":  r.losing_trades,
        "mae":            r.mae,
        "mape":           r.mape,
        "rmse":           r.rmse,
        "avg_return":     r.avg_return,
        "sharpe_ratio":   r.sharpe_ratio,
        "max_drawdown":   r.max_drawdown,
        "total_pnl":      r.total_pnl,
        "equity_curve":   r.equity_curve,
        "trade_log":      r.trade_log,
        "error":          r.error,
    }

