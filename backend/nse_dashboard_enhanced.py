"""
Enhanced NSE Dashboard — Flask backend (port 3000)

Features:
  - Dual data: yfinance primary, Kite live backup
  - Kronos AI forecasting (async job polling)
  - Multi-sample confidence bands
  - Technical indicators: EMA, BB, VWAP, RSI, MACD, SuperTrend
  - Walk-forward backtesting with Sharpe / drawdown / P&L
  - Structured trading signals (entry / stop-loss / targets)
  - MCP venv integration (shares ~/kronos_mcp_venv)

Run:
    KRONOS_REPO_PATH=~/kronos_repo \
    /Users/dnyaneshchandewar/kronos_mcp_venv/bin/python nse_dashboard_enhanced.py
    open http://localhost:3000
"""

import json, math, os, sys, uuid, threading, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── Kronos path ───────────────────────────────────────────────────────────────
KRONOS_REPO = os.environ.get("KRONOS_REPO_PATH", os.path.expanduser("~/kronos_repo"))
if KRONOS_REPO not in sys.path:
    sys.path.insert(0, KRONOS_REPO)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from data_fetcher_enhanced import (
    NSE_SYMBOLS, YF_PERIOD, fetch_ohlcv, calculate_orb,
    safe_float, is_kite_connected, set_kite_session,
)
from indicators import get_all_indicators, get_latest_summary
from signals import generate_signal
from backtester import BacktestConfig, run_backtest, result_to_dict

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
_predictor        = None
_loaded_model_key = None
_cached_df        : pd.DataFrame | None = None
_cached_symbol    = ""
_cached_interval  = ""
_cached_source    = ""
_cached_indicators: dict | None = None

_jobs      : dict = {}
_jobs_lock       = threading.Lock()

AVAILABLE_MODELS = {
    "kronos-mini":  {"name":"Kronos-mini",  "model_id":"NeoQuasar/Kronos-mini",
                     "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-2k",  "context_length":2048, "params":"4.1M"},
    "kronos-small": {"name":"Kronos-small", "model_id":"NeoQuasar/Kronos-small",
                     "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base","context_length":512,  "params":"24.7M"},
    "kronos-base":  {"name":"Kronos-base",  "model_id":"NeoQuasar/Kronos-base",
                     "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base","context_length":512,  "params":"102.3M"},
}

# ── Chart helpers ─────────────────────────────────────────────────────────────

def _bar_freq(df):
    if len(df) < 2: return pd.Timedelta(minutes=5)
    d = df["timestamps"].diff().dropna()
    s = d[d < pd.Timedelta(hours=4)]
    m = s.mode()
    return m.iloc[0] if len(m) else d.median()

def _tstr(s):
    if isinstance(s, pd.DatetimeIndex): return s.strftime("%Y-%m-%d %H:%M").tolist()
    if isinstance(s, pd.Series):        return s.dt.strftime("%Y-%m-%d %H:%M").tolist()
    return [str(t) for t in s]

def _nz(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except: return None

def _strip_none(lst):
    """Replace None with None (Plotly handles None as gap)."""
    return [_nz(v) for v in lst]

# ── Chart builder with subplots ───────────────────────────────────────────────

def build_chart(df, pred_df, orb_levels, lookback,
                source_label="", indicators=None,
                show_rsi=True, show_macd=True,
                pred_bands=None):
    """
    Build a Plotly figure with:
      Row 1 (main): candlesticks + EMA + BB + VWAP + volume + prediction + ORB
      Row 2:        RSI (if show_rsi)
      Row 3:        MACD (if show_macd)

    pred_bands: dict with keys 'upper', 'lower', 'mean' (lists) for confidence shading
    """
    hist    = df.iloc[-lookback:].copy().reset_index(drop=True)
    hist_ts = _tstr(hist["timestamps"])
    inds    = indicators or {}

    # Decide subplot layout
    row_specs, row_heights = [], []
    row_specs.append([{"secondary_y": True}])
    row_heights.append(0.60)
    if show_rsi:
        row_specs.append([{"secondary_y": False}])
        row_heights.append(0.20)
    if show_macd:
        row_specs.append([{"secondary_y": False}])
        row_heights.append(0.20)

    n_rows  = len(row_specs)
    fig     = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        specs=row_specs,
    )

    # ── 1. Candlesticks ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=hist_ts,
        open=hist["open"].tolist(), high=hist["high"].tolist(),
        low=hist["low"].tolist(),   close=hist["close"].tolist(),
        name="Price",
        increasing=dict(line=dict(color="#26de81",width=1), fillcolor="#26de81"),
        decreasing=dict(line=dict(color="#ff4757",width=1), fillcolor="#ff4757"),
        whiskerwidth=0,
    ), row=1, col=1, secondary_y=False)

    # ── 2. Volume ────────────────────────────────────────────────────────
    if "volume" in hist.columns and hist["volume"].sum() > 0:
        vol_c = ["rgba(38,222,129,0.25)" if c >= o else "rgba(255,71,87,0.25)"
                 for o,c in zip(hist["open"], hist["close"])]
        fig.add_trace(go.Bar(
            x=hist_ts, y=hist["volume"].tolist(),
            marker_color=vol_c, name="Volume", showlegend=False,
        ), row=1, col=1, secondary_y=True)

    # ── 3. EMA lines ─────────────────────────────────────────────────────
    n = len(hist)
    _ema_cfg = [("ema9","#ffd32a","EMA 9"),("ema21","#58a6ff","EMA 21"),("ema50","#bf8aff","EMA 50")]
    for key, color, label in _ema_cfg:
        if key in inds:
            vals = _strip_none(inds[key][-n:])
            fig.add_trace(go.Scatter(
                x=hist_ts, y=vals, name=label, mode="lines",
                line=dict(color=color, width=1.2),
            ), row=1, col=1, secondary_y=False)

    # ── 4. Bollinger Bands ───────────────────────────────────────────────
    if "bb_upper" in inds and "bb_lower" in inds:
        bu = _strip_none(inds["bb_upper"][-n:])
        bm = _strip_none(inds["bb_mid"][-n:])
        bl = _strip_none(inds["bb_lower"][-n:])
        fig.add_trace(go.Scatter(
            x=hist_ts+hist_ts[::-1],
            y=bu+bl[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.06)",
            line=dict(color="rgba(88,166,255,0.0)"),
            name="BB Band", showlegend=False,
        ), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hist_ts, y=bm, name="BB Mid", mode="lines",
            line=dict(color="#377dff", width=1, dash="dot"),
        ), row=1, col=1, secondary_y=False)

    # ── 5. VWAP ──────────────────────────────────────────────────────────
    if "vwap" in inds:
        vv = _strip_none(inds["vwap"][-n:])
        if any(v is not None for v in vv):
            fig.add_trace(go.Scatter(
                x=hist_ts, y=vv, name="VWAP", mode="lines",
                line=dict(color="#ff9f43", width=1.5, dash="dot"),
            ), row=1, col=1, secondary_y=False)

    # ── 6. SuperTrend ────────────────────────────────────────────────────
    if "supertrend" in inds and "st_direction" in inds:
        st = _strip_none(inds["supertrend"][-n:])
        sd = inds["st_direction"][-n:]
        st_colors = ["rgba(38,222,129,0.7)" if d==1 else "rgba(255,71,87,0.7)" for d in sd]
        fig.add_trace(go.Scatter(
            x=hist_ts, y=st, name="SuperTrend", mode="markers",
            marker=dict(size=3, color=st_colors),
        ), row=1, col=1, secondary_y=False)

    # ── 7. Prediction candles ────────────────────────────────────────────
    pred_ts_strs = []
    if pred_df is not None and not pred_df.empty:
        freq = _bar_freq(df)
        last_ts = hist["timestamps"].iloc[-1]
        pred_dates = pd.date_range(start=last_ts + freq, periods=len(pred_df), freq=freq)
        pred_ts_strs = _tstr(pred_dates)

        fig.add_trace(go.Candlestick(
            x=pred_ts_strs,
            open=pred_df["open"].values.tolist(),
            high=pred_df["high"].values.tolist(),
            low=pred_df["low"].values.tolist(),
            close=pred_df["close"].values.tolist(),
            name="Kronos Forecast",
            increasing=dict(line=dict(color="#ffd32a",width=1), fillcolor="rgba(255,211,42,0.2)"),
            decreasing=dict(line=dict(color="#ffa502",width=1), fillcolor="rgba(255,165,2,0.2)"),
            whiskerwidth=0,
        ), row=1, col=1, secondary_y=False)

        # Confidence bands
        if pred_bands:
            ub = pred_bands.get("upper", [])
            lb = pred_bands.get("lower", [])
            if ub and lb:
                fig.add_trace(go.Scatter(
                    x=pred_ts_strs+pred_ts_strs[::-1],
                    y=ub+lb[::-1],
                    fill="toself", fillcolor="rgba(255,211,42,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Forecast band", showlegend=False,
                ), row=1, col=1, secondary_y=False)

    # ── 8. ORB lines ─────────────────────────────────────────────────────
    shapes, annotations = [], []
    if orb_levels:
        latest = orb_levels[max(orb_levels.keys())]
        for label, val, color in [("ORB H", latest["high"],"#00d4ff"),
                                   ("ORB L", latest["low"], "#ff6b81")]:
            shapes.append(dict(type="line",xref="paper",yref="y",
                               x0=0,x1=1,y0=val,y1=val,
                               line=dict(color=color,width=1.2,dash="dot")))
            annotations.append(dict(xref="paper",yref="y",x=1.002,y=val,
                                    text=f"<b>{label}</b> {val:,.0f}",
                                    showarrow=False,font=dict(color=color,size=9),
                                    xanchor="left"))

    if source_label:
        annotations.append(dict(xref="paper",yref="paper",x=0.01,y=0.99,
                                 text=f"SRC:{source_label.upper()}",
                                 showarrow=False,font=dict(color="#444",size=9),
                                 xanchor="left"))

    # ── 9. RSI subplot ───────────────────────────────────────────────────
    rsi_row = None
    if show_rsi and "rsi" in inds:
        rsi_row = 2
        rsi_vals = _strip_none(inds["rsi"][-n:])
        fig.add_trace(go.Scatter(
            x=hist_ts, y=rsi_vals, name="RSI 14", mode="lines",
            line=dict(color="#ffd32a", width=1.5),
        ), row=rsi_row, col=1)
        fig.add_hline(y=70, line=dict(color="#ff4757",width=0.8,dash="dot"), row=rsi_row,col=1)
        fig.add_hline(y=30, line=dict(color="#26de81",width=0.8,dash="dot"), row=rsi_row,col=1)
        fig.add_hline(y=50, line=dict(color="#444c56",width=0.5,dash="dot"), row=rsi_row,col=1)

    # ── 10. MACD subplot ─────────────────────────────────────────────────
    macd_row = None
    if show_macd and "macd" in inds:
        macd_row = (3 if show_rsi else 2)
        macd_v   = _strip_none(inds["macd"][-n:])
        msig_v   = _strip_none(inds["macd_signal"][-n:])
        mhist_v  = _strip_none(inds["macd_hist"][-n:])
        hist_colors = ["rgba(38,222,129,0.7)" if (v or 0) >= 0 else "rgba(255,71,87,0.7)"
                       for v in mhist_v]
        fig.add_trace(go.Bar(
            x=hist_ts, y=mhist_v, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=hist_ts, y=macd_v, name="MACD", mode="lines",
            line=dict(color="#58a6ff",width=1.2),
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=hist_ts, y=msig_v, name="Signal", mode="lines",
            line=dict(color="#ff9f43",width=1.2),
        ), row=macd_row, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    all_x = hist_ts + pred_ts_strs
    fig.update_layout(
        paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
        font=dict(color="#c9d1d9", family="JetBrains Mono,Consolas,monospace", size=11),
        margin=dict(l=55,r=110,t=28,b=28),
        height=580 + (120 if show_rsi else 0) + (120 if show_macd else 0),
        dragmode="pan",
        legend=dict(bgcolor="rgba(13,15,26,0.85)",bordercolor="#21262d",borderwidth=1,
                    x=0.01,y=0.99,xanchor="left",yanchor="top",font=dict(size=10)),
        shapes=shapes, annotations=annotations,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22",bordercolor="#30363d",font=dict(color="#c9d1d9",size=11)),
        xaxis=dict(
            type="category", rangeslider=dict(visible=False),
            gridcolor="#161b22", tickangle=-30, tickfont=dict(size=9), nticks=14,
            range=[max(0, len(all_x) - len(hist_ts)), len(all_x)-1],
        ),
    )

    # Axis styling per row
    fig.update_yaxes(gridcolor="#161b22",tickformat=",.0f",side="right",row=1,col=1,secondary_y=False)
    fig.update_yaxes(showgrid=False,showticklabels=False,row=1,col=1,secondary_y=True)
    if rsi_row:
        fig.update_yaxes(gridcolor="#161b22",tickformat=".0f",range=[0,100],
                         side="right",row=rsi_row,col=1)
    if macd_row:
        fig.update_yaxes(gridcolor="#161b22",tickformat=".2f",side="right",row=macd_row,col=1)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("nse_dashboard_enhanced.html")

@app.route("/api/symbols")
def get_symbols():
    return jsonify({"symbols": dict(NSE_SYMBOLS), "intervals": list(YF_PERIOD.keys())})

@app.route("/api/data-source-status")
def data_source_status():
    return jsonify({"kite_connected": is_kite_connected(), "last_source": _cached_source or "none"})

@app.route("/api/kite-login")
def kite_login():
    api_key = request.args.get("api_key","")
    if not api_key: return jsonify({"error":"api_key required"}), 400
    try:
        from kiteconnect import KiteConnect
        return jsonify({"login_url": KiteConnect(api_key=api_key).login_url()})
    except ImportError:
        return jsonify({"error":"pip install kiteconnect"}), 400

@app.route("/api/kite-connect", methods=["POST"])
def kite_connect():
    data = request.get_json()
    api_key = data.get("api_key",""); api_secret = data.get("api_secret","")
    request_token = data.get("request_token",""); access_token = data.get("access_token","")
    if not api_key: return jsonify({"error":"api_key required"}), 400
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        if access_token:
            kite.set_access_token(access_token)
        elif request_token and api_secret:
            s = kite.generate_session(request_token, api_secret=api_secret)
            access_token = s["access_token"]; kite.set_access_token(access_token)
        else:
            return jsonify({"error":"Provide access_token or (request_token+api_secret)"}), 400
        set_kite_session(api_key, access_token)
        profile = kite.profile()
        return jsonify({"success":True,"user":profile.get("user_name",""),"message":f"Kite connected as {profile.get('user_name','')}"})
    except ImportError: return jsonify({"error":"pip install kiteconnect"}), 400
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/api/fetch-data", methods=["POST"])
def fetch_data():
    global _cached_df, _cached_symbol, _cached_interval, _cached_source, _cached_indicators
    data = request.get_json()
    symbol = data.get("symbol","^NSEI"); interval = data.get("interval","5m")
    orb_minutes = int(data.get("orb_minutes",15)); source_pref = data.get("source","auto")
    show_rsi  = bool(data.get("show_rsi", True))
    show_macd = bool(data.get("show_macd", True))

    try:
        df, source_used = fetch_ohlcv(symbol, interval, source=source_pref)
        orb_levels = calculate_orb(df, orb_minutes)

        _cached_df = df; _cached_symbol = symbol
        _cached_interval = interval; _cached_source = source_used

        # Compute indicators
        inds = get_all_indicators(df)
        _cached_indicators = inds

        lookback   = min(200, len(df))
        chart_json = build_chart(df, None, orb_levels, lookback, source_used,
                                 indicators=inds, show_rsi=show_rsi, show_macd=show_macd)

        latest_orb = None
        if orb_levels:
            d = max(orb_levels.keys())
            latest_orb = {**orb_levels[d], "date": d}

        last  = df.iloc[-1]; prev = df.iloc[-2] if len(df) > 1 else last
        chg   = float(last["close"] - prev["close"])
        pct   = chg / float(prev["close"]) * 100 if float(prev["close"]) else 0

        # Indicator summary for sidebar
        ind_summary = get_latest_summary(df)

        return jsonify({
            "success": True, "rows": len(df),
            "start_date": df["timestamps"].min().isoformat(),
            "end_date":   df["timestamps"].max().isoformat(),
            "last_price": round(float(last["close"]),2),
            "change": round(chg,2), "change_pct": round(pct,2),
            "orb": latest_orb, "chart": chart_json,
            "source": source_used, "indicators": ind_summary,
            "message": f"{len(df):,} candles · {symbol} @ {interval} [{source_used}]",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/refresh-chart", methods=["POST"])
def refresh_chart():
    """Rebuild chart with new indicator toggle settings (no re-fetch)."""
    global _cached_indicators
    if _cached_df is None: return jsonify({"error":"No data loaded"}), 400
    data = request.get_json()
    orb_minutes = int(data.get("orb_minutes",15))
    show_rsi  = bool(data.get("show_rsi", True))
    show_macd = bool(data.get("show_macd", True))
    lookback  = min(int(data.get("lookback",200)), len(_cached_df))

    try:
        orb_levels = calculate_orb(_cached_df, orb_minutes)
        inds = _cached_indicators or get_all_indicators(_cached_df)
        chart_json = build_chart(_cached_df, None, orb_levels, lookback,
                                 _cached_source, indicators=inds,
                                 show_rsi=show_rsi, show_macd=show_macd)
        return jsonify({"success":True, "chart": chart_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/signal", methods=["POST"])
def get_signal():
    if _cached_df is None: return jsonify({"error":"No data loaded"}), 400
    data = request.get_json()
    orb_minutes = int(data.get("orb_minutes",15))
    try:
        orb_levels = calculate_orb(_cached_df, orb_minutes)
        sig = generate_signal(_cached_df, None, orb_levels)
        return jsonify({"success":True, "signal": sig})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/available-models")
def available_models():
    return jsonify({"models": AVAILABLE_MODELS,
                    "model_available": MODEL_AVAILABLE,
                    "loaded_model": _loaded_model_key})

@app.route("/api/model-status")
def model_status():
    if MODEL_AVAILABLE and _predictor:
        dev = str(next(_predictor.model.parameters()).device)
        return jsonify({"available":True,"loaded":True,"device":dev,"model":_loaded_model_key})
    elif MODEL_AVAILABLE:
        return jsonify({"available":True,"loaded":False,"message":"Library ready, model not loaded"})
    else:
        return jsonify({"available":False,"loaded":False,"message":"Kronos not found — check KRONOS_REPO_PATH"})

@app.route("/api/load-model", methods=["POST"])
def load_model():
    global _predictor, _loaded_model_key
    if not MODEL_AVAILABLE: return jsonify({"error":"Kronos library not available"}), 400
    data = request.get_json()
    key  = data.get("model_key","kronos-small"); device = data.get("device","cpu")
    if key not in AVAILABLE_MODELS: return jsonify({"error":f"Unknown: {key}"}), 400
    cfg = AVAILABLE_MODELS[key]
    try:
        tok = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        mdl = Kronos.from_pretrained(cfg["model_id"])
        _predictor = KronosPredictor(mdl, tok, max_context=cfg["context_length"])
        if device != "cpu": mdl.to(device)
        _loaded_model_key = key
        return jsonify({"success":True,"message":f"{cfg['name']} ({cfg['params']}) ready on {device}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Async prediction ──────────────────────────────────────────────────────────

def _multi_sample_bands(predictor, x_df, x_ts, y_ts, pred_len, temp, top_p, n=5):
    """Run n sample paths, return (mean_df, upper_list, lower_list)."""
    paths = []
    for _ in range(n):
        try:
            p = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                  pred_len=pred_len, T=temp, top_p=top_p, sample_count=1)
            paths.append(p["close"].values)
        except: pass
    if not paths:
        return None, None, None
    arr = np.array(paths)
    return (np.mean(arr,axis=0).tolist(),
            np.percentile(arr,95,axis=0).tolist(),
            np.percentile(arr,5,axis=0).tolist())

def _run_prediction(job_id, df, x_df, x_ts, y_ts,
                    freq, pred_len, temperature, top_p, sample_count,
                    orb_minutes, lookback, source_label, compute_bands):
    try:
        with _jobs_lock: _jobs[job_id]["status"] = "running"

        pred_df = _predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=sample_count,
        )
        pred_df = pred_df.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

        # Confidence bands (multiple sample paths)
        pred_bands = None
        if compute_bands and sample_count > 1:
            _, upper, lower = _multi_sample_bands(
                _predictor, x_df, x_ts, y_ts, pred_len, temperature, top_p, n=sample_count)
            if upper and lower:
                pred_bands = {"upper": upper, "lower": lower}

        orb_levels  = calculate_orb(df, orb_minutes)
        inds        = _cached_indicators or get_all_indicators(df)
        chart_json  = build_chart(df, pred_df, orb_levels, lookback, source_label,
                                  indicators=inds, show_rsi=True, show_macd=True,
                                  pred_bands=pred_bands)

        # Signal with forecast
        sig = generate_signal(df, pred_df, orb_levels)

        freq_td    = _bar_freq(df) if freq is None else freq
        pred_ts    = pd.date_range(start=x_ts.iloc[-1] + freq_td, periods=pred_len, freq=freq_td)
        records    = [{"timestamp": pred_ts[i].isoformat(),
                       "open":  safe_float(pred_df["open"].values[i]),
                       "high":  safe_float(pred_df["high"].values[i]),
                       "low":   safe_float(pred_df["low"].values[i]),
                       "close": safe_float(pred_df["close"].values[i])}
                      for i in range(pred_len)]

        last_close = float(df["close"].iloc[-1])
        pred_close = safe_float(pred_df["close"].values[-1])
        pct_change = (pred_close - last_close) / last_close * 100 if last_close else 0
        trend      = "BULLISH" if pred_close > last_close else "BEARISH"
        valid_h    = [safe_float(v) for v in pred_df["high"].values if not math.isnan(float(v))]
        valid_l    = [safe_float(v) for v in pred_df["low"].values  if not math.isnan(float(v))]

        with _jobs_lock:
            _jobs[job_id] = {"status":"done","result":{
                "success":True, "chart":chart_json,
                "prediction_results":records,
                "signal": sig,
                "stats":{
                    "last_close":round(last_close,2), "pred_close":pred_close,
                    "pct_change":round(pct_change,2), "trend":trend,
                    "pred_high":max(valid_h) if valid_h else 0,
                    "pred_low": min(valid_l) if valid_l else 0,
                },
                "source":source_label,
                "message":f"Predicted {pred_len} candles · {trend} ({pct_change:+.2f}%)",
            }}
    except Exception as e:
        with _jobs_lock: _jobs[job_id] = {"status":"error","error":str(e)}

@app.route("/api/predict", methods=["POST"])
def predict():
    if _cached_df is None: return jsonify({"error":"No data loaded"}), 400
    if not MODEL_AVAILABLE or _predictor is None: return jsonify({"error":"Model not loaded"}), 400
    data = request.get_json()
    lookback     = int(data.get("lookback",200))
    pred_len     = int(data.get("pred_len",60))
    temperature  = float(data.get("temperature",1.0))
    top_p        = float(data.get("top_p",0.9))
    sample_count = int(data.get("sample_count",1))
    orb_minutes  = int(data.get("orb_minutes",15))
    compute_bands= bool(data.get("confidence_bands", sample_count > 1))

    df = _cached_df
    if len(df) < lookback: return jsonify({"error":f"Need {lookback} candles, have {len(df)}"}), 400

    cols = ["open","high","low","close"] + (["volume"] if "volume" in df.columns else [])
    x_df = df.iloc[-lookback:][cols].copy()
    x_ts = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
    freq = _bar_freq(df)
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + freq, periods=pred_len, freq=freq))

    job_id = str(uuid.uuid4())
    with _jobs_lock: _jobs[job_id] = {"status":"pending"}
    threading.Thread(
        target=_run_prediction,
        args=(job_id, df.copy(), x_df, x_ts, y_ts,
              freq, pred_len, temperature, top_p, sample_count,
              orb_minutes, lookback, _cached_source, compute_bands),
        daemon=True,
    ).start()
    return jsonify({"job_id":job_id,"status":"pending"})

@app.route("/api/predict-status/<job_id>")
def predict_status(job_id):
    with _jobs_lock: job = _jobs.get(job_id)
    if job is None: return jsonify({"error":"Unknown job"}), 404
    if job["status"] == "done":   return jsonify({"status":"done",   **job["result"]})
    if job["status"] == "error":  return jsonify({"status":"error",  "error":job["error"]}), 500
    return jsonify({"status": job["status"]})

# ── Async backtest ────────────────────────────────────────────────────────────

def _run_backtest_job(job_id, df, cfg):
    try:
        with _jobs_lock: _jobs[job_id]["status"] = "running"

        def progress(pct, msg):
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = round(pct*100)
                    _jobs[job_id]["msg"] = msg

        result = run_backtest(df, _predictor, cfg, progress_cb=progress)
        rd     = result_to_dict(result)

        # Build equity curve chart
        eq_chart = None
        if rd["equity_curve"]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=rd["equity_curve"], mode="lines", name="Equity",
                line=dict(color="#26de81" if rd["total_pnl"] >= 0 else "#ff4757", width=2),
                fill="tonexty", fillcolor="rgba(38,222,129,0.08)" if rd["total_pnl"]>=0
                               else "rgba(255,71,87,0.08)",
            ))
            fig.add_hline(y=rd["equity_curve"][0],
                          line=dict(color="#444c56",width=1,dash="dot"))
            fig.update_layout(
                paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
                font=dict(color="#c9d1d9",family="JetBrains Mono,monospace",size=10),
                margin=dict(l=50,r=20,t=20,b=30), height=200,
                xaxis=dict(gridcolor="#161b22",showticklabels=False),
                yaxis=dict(gridcolor="#161b22",tickformat=",.0f",side="right"),
                showlegend=False,
            )
            eq_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        with _jobs_lock:
            _jobs[job_id] = {"status":"done","result":{**rd, "equity_chart":eq_chart}}
    except Exception as e:
        with _jobs_lock: _jobs[job_id] = {"status":"error","error":str(e)}

@app.route("/api/backtest", methods=["POST"])
def backtest():
    if _cached_df is None: return jsonify({"error":"No data loaded"}), 400
    if not MODEL_AVAILABLE or _predictor is None: return jsonify({"error":"Model not loaded"}), 400
    data = request.get_json()
    cfg = BacktestConfig(
        lookback      = int(data.get("lookback",     150)),
        pred_len      = int(data.get("pred_len",      20)),
        horizon       = int(data.get("horizon",       10)),
        step          = int(data.get("step",          10)),
        temperature   = float(data.get("temperature", 1.0)),
        top_p         = float(data.get("top_p",       0.9)),
        sample_count  = int(data.get("sample_count",  1)),
        position_size = float(data.get("position_size", 100_000)),
        transaction_cost = float(data.get("transaction_cost", 0.0003)),
    )
    min_needed = cfg.lookback + cfg.pred_len
    if len(_cached_df) < min_needed:
        return jsonify({"error":f"Need ≥{min_needed} candles for backtest, have {len(_cached_df)}"}),400

    job_id = str(uuid.uuid4())
    with _jobs_lock: _jobs[job_id] = {"status":"pending","progress":0,"msg":"Queued"}
    threading.Thread(
        target=_run_backtest_job, args=(job_id, _cached_df.copy(), cfg), daemon=True
    ).start()
    return jsonify({"job_id":job_id,"status":"pending"})

@app.route("/api/backtest-status/<job_id>")
def backtest_status(job_id):
    with _jobs_lock: job = _jobs.get(job_id)
    if job is None: return jsonify({"error":"Unknown job"}), 404
    if job["status"] == "done":  return jsonify({"status":"done",  **job["result"]})
    if job["status"] == "error": return jsonify({"status":"error", "error":job["error"]}), 500
    return jsonify({"status":job["status"],"progress":job.get("progress",0),"msg":job.get("msg","")})

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("-"*55)
    print("  Kronos NSE Terminal (Enhanced v2)")
    print(f"  Kronos : {'✓' if MODEL_AVAILABLE else '✗ check KRONOS_REPO_PATH'}")
    print(f"  Kite   : {'✓ connected' if is_kite_connected() else '○ not connected'}")
    print("  URL    : http://localhost:3000")
    print("-"*55)
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
