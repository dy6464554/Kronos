"""Compact, compatible Flask backend for the NSE terminal."""
import json,math,os,sys,threading,uuid
import pandas as pd, plotly.graph_objects as go
from flask import Flask,jsonify,render_template,request
from flask_cors import CORS
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from data_fetcher_enhanced import *
from indicators import get_all_indicators,get_latest_summary
from signals import generate_signal
from backtester import BacktestConfig,run_backtest,result_to_dict
from model import Kronos,KronosTokenizer,KronosPredictor
app=Flask(__name__,template_folder=os.path.join(HERE,"templates")); CORS(app)
_predictor=None; _loaded=None; _df=None; _source="none"; _jobs={}; _lock=threading.Lock()
MODELS={"kronos-mini":{"name":"Kronos-mini","model_id":"NeoQuasar/Kronos-mini","tokenizer_id":"NeoQuasar/Kronos-Tokenizer-2k","context_length":2048,"params":"4.1M"},"kronos-small":{"name":"Kronos-small","model_id":"NeoQuasar/Kronos-small","tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base","context_length":512,"params":"24.7M"},"kronos-base":{"name":"Kronos-base","model_id":"NeoQuasar/Kronos-base","tokenizer_id":"NeoQuasar/Kronos-Tokenizer-base","context_length":512,"params":"102.3M"}}
def _freq(df):
    if len(df)<2:return pd.Timedelta(minutes=5)
    d=df.timestamps.diff().dropna(); d=d[d<pd.Timedelta(hours=4)]; m=d.mode(); return m.iloc[0] if len(m) else d.median()
def _chart(df,pred=None,orb=None,lookback=200):
    from plotly.subplots import make_subplots
    h=df.iloc[-min(lookback,len(df)):].reset_index(drop=True); x=h.timestamps.dt.strftime("%Y-%m-%d %H:%M").tolist(); ind=get_all_indicators(df); fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.6,.2,.2],vertical_spacing=.03,specs=[[{"secondary_y":True}],[{}],[{}]])
    fig.add_trace(go.Candlestick(x=x,open=h.open,high=h.high,low=h.low,close=h.close,name="Price"),row=1,col=1,secondary_y=False)
    for k in ("ema9","ema21","ema50"): fig.add_trace(go.Scatter(x=x,y=ind[k][-len(h):],name=k.upper()),row=1,col=1,secondary_y=False)
    if h.volume.sum(): fig.add_trace(go.Bar(x=x,y=h.volume,opacity=.25,name="Volume",showlegend=False),row=1,col=1,secondary_y=True)
    if pred is not None and not pred.empty:
        px=pd.date_range(h.timestamps.iloc[-1]+_freq(df),periods=len(pred),freq=_freq(df)).strftime("%Y-%m-%d %H:%M").tolist(); fig.add_trace(go.Candlestick(x=px,open=pred.open,high=pred.high,low=pred.low,close=pred.close,name="Kronos Forecast"),row=1,col=1,secondary_y=False)
    if orb:
        q=orb[max(orb)]; fig.add_hline(y=q["high"],line_dash="dot",line_color="#00d4ff",annotation_text="ORB H",row=1,col=1); fig.add_hline(y=q["low"],line_dash="dot",line_color="#ff6b81",annotation_text="ORB L",row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=ind["rsi"][-len(h):],name="RSI 14"),row=2,col=1); fig.add_trace(go.Scatter(x=x,y=ind["macd"][-len(h):],name="MACD"),row=3,col=1); fig.update_layout(template="plotly_dark",paper_bgcolor="#0d0f1a",plot_bgcolor="#0d0f1a",height=820,xaxis_rangeslider_visible=False)
    return json.dumps(fig,default=str)
@app.get("/")
def index(): return render_template("nse_dashboard_enhanced.html")
@app.get("/api/symbols")
def symbols(): return jsonify({"symbols":NSE_SYMBOLS,"intervals":list(INTERVALS)})
@app.get("/api/data-source-status")
def status(): return jsonify({"upstox_connected":is_upstox_connected(),"kite_connected":is_upstox_connected(),"last_source":_source})
@app.post("/api/upstox-connect")
def connect(): return jsonify({"success":is_upstox_connected(),"message":"Upstox credentials detected" if is_upstox_connected() else "Set UPSTOX_ACCESS_TOKEN in Secrets"})
@app.post("/api/kite-connect")
def legacy_connect(): return connect()
@app.post("/api/fetch-data")
def fetch():
    global _df,_source
    d=request.get_json(silent=True) or {}; sym=d.get("symbol","NSE_INDEX|Nifty 50"); iv=d.get("interval","5m")
    try:
        _df,_source=fetch_ohlcv(sym,iv,"upstox"); orb=calculate_orb(_df,int(d.get("orb_minutes",15))); last=_df.iloc[-1]; prev=_df.iloc[-2] if len(_df)>1 else last; change=float(last.close-prev.close); summary=get_latest_summary(_df)
        return jsonify({"success":True,"rows":len(_df),"start_date":_df.timestamps.min().isoformat(),"end_date":_df.timestamps.max().isoformat(),"last_price":safe_float(last.close),"change":safe_float(change),"change_pct":safe_float(change/prev.close*100 if prev.close else 0),"orb":({**orb[max(orb)],"date":max(orb)} if orb else None),"chart":_chart(_df,orb=orb),"indicators":summary,"source":"upstox","message":f"{len(_df):,} candles · {sym} @ {iv} [upstox]"})
    except Exception as e:return jsonify({"error":str(e)}),500
@app.post("/api/indicators")
def indicators():
    d=request.get_json(silent=True) or {}; rows=d.get("ohlcv",[]); frame=pd.DataFrame(rows)
    if frame.empty:return jsonify({})
    if "timestamps" not in frame: frame["timestamps"]=pd.date_range(end=pd.Timestamp.now(),periods=len(frame),freq="5min")
    return jsonify(get_all_indicators(frame))
@app.post("/api/refresh-chart")
def refresh():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    return jsonify({"success":True,"chart":_chart(_df,orb=calculate_orb(_df,15))})
@app.post("/api/signal")
def signal():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    return jsonify({"success":True,"signal":generate_signal(_df,None,calculate_orb(_df,15))})
@app.get("/api/available-models")
def models(): return jsonify({"models":MODELS,"model_available":True,"loaded_model":_loaded})
@app.get("/api/model-status")
def model_status(): return jsonify({"available":True,"loaded":_predictor is not None,"model":_loaded,"device":"cpu" if _predictor is None else str(next(_predictor.model.parameters()).device),"message":"Ready" if _predictor else "Model not loaded"})
@app.post("/api/load-model")
def load_model():
    global _predictor,_loaded
    d=request.get_json(silent=True) or {}; key=d.get("model_key","kronos-small"); c=MODELS.get(key)
    if not c:return jsonify({"error":"Unknown model"}),400
    try:
        tok=KronosTokenizer.from_pretrained(c["tokenizer_id"]); mdl=Kronos.from_pretrained(c["model_id"]); device=d.get("device","cpu"); mdl.to(device); _predictor=KronosPredictor(mdl,tok,max_context=c["context_length"]); _loaded=key; return jsonify({"success":True,"message":f"{c['name']} ready on {device}"})
    except Exception as e:return jsonify({"error":str(e)}),500
def _run(job,d):
    try:
        with _lock:_jobs[job]={"status":"running"}
        n=int(d.get("lookback",200)); p=int(d.get("pred_len",60)); ctx=_df.iloc[-n:]; ts=ctx.timestamps.reset_index(drop=True); y=pd.Series(pd.date_range(ts.iloc[-1]+_freq(_df),periods=p,freq=_freq(_df))); pred=_predictor.predict(ctx[["open","high","low","close","volume"]],ts,y,pred_len=p,T=float(d.get("temperature",1)),top_p=.9,sample_count=int(d.get("sample_count",1))).fillna(0); last=float(_df.close.iloc[-1]); final=float(pred.close.iloc[-1]); result={"status":"done","chart":_chart(_df,pred,calculate_orb(_df,15),n),"signal":generate_signal(_df,pred,calculate_orb(_df,15)),"stats":{"last_close":last,"pred_close":final,"pct_change":round((final-last)/last*100,2),"trend":"BULLISH" if final>last else "BEARISH"},"source":"upstox","message":f"Predicted {p} candles"}
        with _lock:_jobs[job]=result
    except Exception as e:
        with _lock:_jobs[job]={"status":"error","error":str(e)}
@app.post("/api/predict")
def predict():
    if _df is None:return jsonify({"error":"No data loaded"}),400
    if _predictor is None:return jsonify({"error":"Model not loaded"}),400
    job=str(uuid.uuid4()); d=request.get_json(silent=True) or {}; _jobs[job]={"status":"pending"}; threading.Thread(target=_run,args=(job,d),daemon=True).start(); return jsonify({"job_id":job,"status":"pending"})
@app.get("/api/predict-status/<job>")
def prediction_status(job):
    with _lock:r=_jobs.get(job)
    return (jsonify({"error":"Unknown job"}),404) if r is None else jsonify(r)
@app.post("/api/backtest")
def backtest():
    if _df is None or _predictor is None:return jsonify({"error":"Load data and model first"}),400
    d=request.get_json(silent=True) or {}; cfg=BacktestConfig(lookback=int(d.get("lookback",150)),pred_len=int(d.get("pred_len",20)),horizon=int(d.get("horizon",10)),step=int(d.get("step",10)),temperature=float(d.get("temperature",1)),top_p=.9,sample_count=1,position_size=float(d.get("position_size",100000)),transaction_cost=.0003); job=str(uuid.uuid4()); _jobs[job]={"status":"pending","progress":0}
    def work():
        try:
            r=result_to_dict(run_backtest(_df.copy(),_predictor,cfg,lambda p,m:_jobs[job].update(progress=round(p*100),msg=m))); _jobs[job]={"status":"done",**r}
        except Exception as e:_jobs[job]={"status":"error","error":str(e)}
    threading.Thread(target=work,daemon=True).start(); return jsonify({"job_id":job,"status":"pending"})
@app.get("/api/backtest-status/<job>")
def bt_status(job):
    with _lock:r=_jobs.get(job)
    return (jsonify({"error":"Unknown job"}),404) if r is None else jsonify(r)
if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.getenv("PORT",3000)),debug=False)
