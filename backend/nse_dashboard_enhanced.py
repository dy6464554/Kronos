"""AI Studio-compatible Flask API for the Kronos NSE terminal.

The original terminal features are retained: Upstox OHLCV data, Plotly charts,
indicators, ORB levels, Kronos forecasts, confidence bands, signals and
walk-forward backtesting. Long-running model operations are asynchronous.
"""
import json, math, os, sys, threading, uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_fetcher_enhanced import NSE_SYMBOLS, YF_PERIOD, INTERVALS, fetch_ohlcv, calculate_orb, safe_float, is_upstox_connected
from indicators import get_all_indicators, get_latest_summary
from signals import generate_signal
from backtester import BacktestConfig, run_backtest, result_to_dict

MODEL_AVAILABLE = False
try:
    kronos_path = os.environ.get("KRONOS_REPO_PATH", os.path.expanduser("~/kronos_repo"))
    if kronos_path not in sys.path: sys.path.insert(0, kronos_path)
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))
CORS(app)
_predictor = None; _loaded_model = None; _df = None; _source = "none"; _inds = None
_jobs = {}; _lock = threading.Lock()
AVAILABLE_MODELS = {
    "kronos-mini": {"name":"Kronos-mini", "model_id":"NeoQuasar/Kronos-mini", "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-2k", "context_length":2048, "params":"4.1M"},
    "kronos-small": {"name":"Kronos-small", "model_id":"NeoQuasar/Kronos-small", "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base", "context_length":512, "params":"24.7M"},
    "kronos-base": {"name":"Kronos-base", "model_id":"NeoQuasar/Kronos-base", "tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base", "context_length":512, "params":"102.3M"},
}

def _freq(df):
    if len(df) < 2: return pd.Timedelta(minutes=5)
    d = df.timestamps.diff().dropna(); d = d[d < pd.Timedelta(hours=4)]
    return d.mode().iloc[0] if len(d.mode()) else d.median()

def _chart(df, pred=None, orb=None, lookback=200, show_rsi=True, show_macd=True):
    h = df.iloc[-min(lookback, len(df)):].reset_index(drop=True); x = h.timestamps.dt.strftime("%Y-%m-%d %H:%M").tolist(); inds = get_all_indicators(df)
    rows = 1 + int(show_rsi) + int(show_macd); heights = [0.6] + ([0.2] if show_rsi else []) + ([0.2] if show_macd else [])
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=.03, specs=[[{"secondary_y":True}]] + [[{}] for _ in range(rows-1)])
    fig.add_trace(go.Candlestick(x=x, open=h.open, high=h.high, low=h.low, close=h.close, name="Price"), row=1,col=1,secondary_y=False)
    if h.volume.sum() > 0: fig.add_trace(go.Bar(x=x,y=h.volume,name="Volume",opacity=.25,showlegend=False),row=1,col=1,secondary_y=True)
    for key,color in [("ema9","#ffd32a"),("ema21","#58a6ff"),("ema50","#bf8aff")]: fig.add_trace(go.Scatter(x=x,y=inds[key][-len(h):],name=key.upper()),row=1,col=1,secondary_y=False)
    if pred is not None and not pred.empty:
        px = pd.date_range(h.timestamps.iloc[-1]+_freq(df), periods=len(pred), freq=_freq(df)).strftime("%Y-%m-%d %H:%M").tolist()
        fig.add_trace(go.Candlestick(x=px,open=pred.open,high=pred.high,low=pred.low,close=pred.close,name="Kronos Forecast"),row=1,col=1,secondary_y=False)
    if orb:
        latest = orb[max(orb)]
        for value,color,name in [(latest["high"],"#00d4ff","ORB H"),(latest["low"],"#ff6b81","ORB L")]: fig.add_hline(y=value,line=dict(color=color,dash="dot"),annotation_text=name,row=1,col=1)
    row=2
    if show_rsi:
        fig.add_trace(go.Scatter(x=x,y=inds["rsi"][-len(h):],name="RSI"),row=row,col=1); row += 1
    if show_macd: fig.add_trace(go.Scatter(x=x,y=inds["macd"][-len(h):],name="MACD"),row=row,col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a", height=650, margin=dict(l=40,r=30,t=20,b=30), xaxis_rangeslider_visible=False)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.get("/")
def index(): return render_template("nse_dashboard_enhanced.html")
@app.get("/api/symbols")
def symbols(): return jsonify({"symbols":NSE_SYMBOLS,"intervals":list(INTERVALS)})
@app.get("/api/data-source-status")
def source_status(): return jsonify({"upstox_connected":is_upstox_connected(),"last_source":_source})
@app.post("/api/upstox-connect")
def upstox_connect(): return jsonify({"success":is_upstox_connected(),"message":"Upstox credentials detected" if is_upstox_connected() else "Set UPSTOX_ACCESS_TOKEN in Secrets"})
@app.post("/api/fetch-data")
def fetch_data():
    global _df, _source, _inds
    d=request.get_json(silent=True) or {}; symbol=d.get("symbol","NSE_INDEX|Nifty 50"); interval=d.get("interval","5m")
    try:
        _df,_source=fetch_ohlcv(symbol,interval,"upstox"); _inds=get_all_indicators(_df); orb=calculate_orb(_df,int(d.get("orb_minutes",15))); last=_df.iloc[-1]; prev=_df.iloc[-2] if len(_df)>1 else last; change=float(last.close-prev.close)
        return jsonify({"success":True,"rows":len(_df),"start_date":_df.timestamps.min().isoformat(),"end_date":_df.timestamps.max().isoformat(),"last_price":safe_float(last.close),"change":safe_float(change),"change_pct":safe_float(change/prev.close*100 if prev.close else 0),"orb":({**orb[max(orb)],"date":max(orb)} if orb else None),"chart":_chart(_df,orb=orb,show_rsi=d.get("show_rsi",True),show_macd=d.get("show_macd",True)),"indicators":get_latest_summary(_df),"source":"upstox","message":f"{len(_df):,} candles · {symbol} @ {interval} [upstox]"})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.post("/api/refresh-chart")
def refresh():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    d=request.get_json(silent=True) or {}; return jsonify({"success":True,"chart":_chart(_df,orb=calculate_orb(_df,int(d.get("orb_minutes",15))),lookback=int(d.get("lookback",200)),show_rsi=d.get("show_rsi",True),show_macd=d.get("show_macd",True))})
@app.post("/api/signal")
def signal():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    return jsonify({"success":True,"signal":generate_signal(_df,None,calculate_orb(_df,15))})
@app.get("/api/available-models")
def models(): return jsonify({"models":AVAILABLE_MODELS,"model_available":MODEL_AVAILABLE,"loaded_model":_loaded_model})
@app.get("/api/model-status")
def model_status(): return jsonify({"available":MODEL_AVAILABLE,"loaded":_predictor is not None,"model":_loaded_model,"message":"Kronos library not found" if not MODEL_AVAILABLE else "Ready"})
@app.post("/api/load-model")
def load_model():
    global _predictor,_loaded_model
    if not MODEL_AVAILABLE:return jsonify({"error":"Kronos library not available; set KRONOS_REPO_PATH"}),400
    d=request.get_json(silent=True) or {}; key=d.get("model_key","kronos-small"); cfg=AVAILABLE_MODELS.get(key)
    if not cfg:return jsonify({"error":"Unknown model"}),400
    try:
        tok=KronosTokenizer.from_pretrained(cfg["tokenizer_id"]); mdl=Kronos.from_pretrained(cfg["model_id"]); device=d.get("device","cpu");
        if device!="cpu": mdl.to(device)
        _predictor=KronosPredictor(mdl,tok,max_context=cfg["context_length"]); _loaded_model=key
        return jsonify({"success":True,"message":f"{cfg['name']} ready on {device}"})
    except Exception as e:return jsonify({"error":str(e)}),500

def _prediction(job, params):
    try:
        with _lock:_jobs[job]["status"]="running"
        lookback,pred_len,T,top_p,samples=params; ctx=_df.iloc[-lookback:]; cols=[c for c in ["open","high","low","close","volume"] if c in ctx]; x=ctx[cols].copy(); ts=ctx.timestamps.reset_index(drop=True); y=pd.Series(pd.date_range(ts.iloc[-1]+_freq(_df),periods=pred_len,freq=_freq(_df)))
        pred=_predictor.predict(df=x,x_timestamp=ts,y_timestamp=y,pred_len=pred_len,T=T,top_p=top_p,sample_count=samples).fillna(0); chart=_chart(_df,pred,calculate_orb(_df,15),lookback); sig=generate_signal(_df,pred,calculate_orb(_df,15)); last=float(_df.close.iloc[-1]); final=float(pred.close.iloc[-1]); result={"status":"done","chart":chart,"signal":sig,"stats":{"last_close":last,"pred_close":final,"pct_change":round((final-last)/last*100,2),"trend":"BULLISH" if final>last else "BEARISH"},"source":"upstox","message":f"Predicted {pred_len} candles"}
        with _lock:_jobs[job]=result
    except Exception as e:
        with _lock:_jobs[job]={"status":"error","error":str(e)}
@app.post("/api/predict")
def predict():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    if _predictor is None:return jsonify({"error":"Model not loaded"}),400
    d=request.get_json(silent=True) or {}; job=str(uuid.uuid4());
    with _lock:_jobs[job]={"status":"pending"}
    threading.Thread(target=_prediction,args=(job,(int(d.get("lookback",200)),int(d.get("pred_len",60)),float(d.get("temperature",1)),float(d.get("top_p",.9)),int(d.get("sample_count",1))),daemon=True).start(); return jsonify({"job_id":job,"status":"pending"})
@app.get("/api/predict-status/<job>")
def prediction_status(job):
    with _lock:r=_jobs.get(job)
    return (jsonify({"error":"Unknown job"}),404) if r is None else jsonify(r)

@app.post("/api/backtest")
def backtest():
    if _df is None or _predictor is None:return jsonify({"error":"Load data and model first"}),400
    d=request.get_json(silent=True) or {}; cfg=BacktestConfig(lookback=int(d.get("lookback",150)),pred_len=int(d.get("pred_len",20)),horizon=int(d.get("horizon",10)),step=int(d.get("step",10)),temperature=float(d.get("temperature",1)),top_p=.9,sample_count=1,position_size=float(d.get("position_size",100000)),transaction_cost=.0003); job=str(uuid.uuid4());
    def run():
        try:
            with _lock:_jobs[job]={"status":"running","progress":0,"msg":"Running walk-forward windows"}
            r=result_to_dict(run_backtest(_df.copy(),_predictor,cfg,lambda p,m: _jobs[job].update(progress=round(p*100),msg=m)))
            with _lock:_jobs[job]={"status":"done",**r}
        except Exception as e:
            with _lock:_jobs[job]={"status":"error","error":str(e)}
    with _lock:_jobs[job]={"status":"pending","progress":0}; threading.Thread(target=run,daemon=True).start(); return jsonify({"job_id":job,"status":"pending"})
@app.get("/api/backtest-status/<job>")
def backtest_status(job):
    with _lock:r=_jobs.get(job)
    return (jsonify({"error":"Unknown job"}),404) if r is None else jsonify(r)

if __name__ == "__main__": app.run(host="0.0.0.0",port=int(os.getenv("PORT",3000)),debug=False)
