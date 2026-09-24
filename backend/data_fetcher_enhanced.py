"""NSE OHLCV provider: yfinance primary, Upstox fallback.

Upstox only accepts 1minute/30minute/day (plus week/month) on the historical
candle endpoint. Unsupported UI intervals are fetched at the nearest supported
resolution and resampled locally.
"""
from __future__ import annotations
import math, os
from datetime import date, datetime, time, timedelta
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = time(9,15), time(15,30)
NSE_SYMBOLS = {
 "NIFTY 50":"NSE_INDEX|Nifty 50", "BANK NIFTY":"NSE_INDEX|Nifty Bank", "NIFTY IT":"NSE_INDEX|Nifty IT",
 "FIN NIFTY":"NSE_INDEX|Nifty Financial Services", "RELIANCE":"NSE_EQ|INE002A01018", "TCS":"NSE_EQ|INE467B01029",
 "HDFC BANK":"NSE_EQ|INE040A01034", "INFOSYS":"NSE_EQ|INE009A01021", "ICICI BANK":"NSE_EQ|INE090A01021",
 "KOTAK BANK":"NSE_EQ|INE237A01028", "AXIS BANK":"NSE_EQ|INE238A01034", "SBI":"NSE_EQ|INE062A01020",
 "LT":"NSE_EQ|INE018A01030", "WIPRO":"NSE_EQ|INE075A01022", "HCL TECH":"NSE_EQ|INE860A01011",
 "BAJAJ FINANCE":"NSE_EQ|INE296A01024", "MARUTI":"NSE_EQ|INE585B01010", "TITAN":"NSE_EQ|INE280A01028",
 "ASIAN PAINTS":"NSE_EQ|INE021A01026",
}
YF_SYMBOLS = {"NIFTY 50":"^NSEI", "BANK NIFTY":"^NSEBANK", "NIFTY IT":"^CNXIT", "FIN NIFTY":"NIFTY_FIN_SERVICE.NS", **{k:k+".NS" for k in ["RELIANCE","TCS","HDFC BANK","INFOSYS","ICICI BANK","KOTAK BANK","AXIS BANK","SBI","LT","WIPRO","HCL TECH","BAJAJ FINANCE","MARUTI","TITAN","ASIAN PAINTS"]}}
YF_SYMBOLS.update({"HDFC BANK":"HDFCBANK.NS","INFOSYS":"INFY.NS","ICICI BANK":"ICICIBANK.NS","KOTAK BANK":"KOTAKBANK.NS","AXIS BANK":"AXISBANK.NS","BAJAJ FINANCE":"BAJFINANCE.NS","ASIAN PAINTS":"ASIANPAINT.NS"})
INTERVALS = {"1m":"1minute", "5m":"5minute", "15m":"15minute", "30m":"30minute", "60m":"60minute", "1d":"day"}
YF_PERIOD = {"1m":"7d", "5m":"60d", "15m":"60d", "30m":"60d", "60m":"730d", "1d":"max"}

def _token():
    token=os.getenv("UPSTOX_ACCESS_TOKEN","").strip()
    if not token: raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured in AI Studio Secrets.")
    return token

def _clean(df):
    if df.empty: return df
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    required=["open","high","low","close","volume"]
    for c in required:
        if c not in df: df[c]=0.0
    if not isinstance(df.index,pd.DatetimeIndex): df.index=pd.to_datetime(df.index)
    if df.index.tz is None: df.index=df.index.tz_localize("UTC")
    df.index=df.index.tz_convert(IST).tz_localize(None)
    df=df[required].dropna().reset_index().rename(columns={"index":"timestamps","Datetime":"timestamps"})
    df=df.sort_values("timestamps").drop_duplicates("timestamps").reset_index(drop=True)
    if len(df) and df.timestamps.dt.time.iloc[0] is not None: df=df[(df.timestamps.dt.time>=MARKET_OPEN)&(df.timestamps.dt.time<=MARKET_CLOSE)].reset_index(drop=True)
    return df[["timestamps",*required]]

def _resample(df, interval):
    if interval in ("1m","30m","1d") or df.empty: return df
    rule={"5m":"5min","15m":"15min","60m":"60min"}[interval]
    x=df.set_index("timestamps").resample(rule,origin="start_day",offset="9h15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
    return x[(x.timestamps.dt.time>=MARKET_OPEN)&(x.timestamps.dt.time<=MARKET_CLOSE)].reset_index(drop=True)

def _yf(symbol, interval):
    import yfinance as yf
    name=symbol if symbol in YF_SYMBOLS else next((k for k,v in YF_SYMBOLS.items() if v==symbol),symbol)
    raw=yf.download(YF_SYMBOLS.get(name,name),period=YF_PERIOD[interval],interval=interval,progress=False,auto_adjust=True,threads=False)
    if interval=="1d":
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        raw=raw.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        raw=raw.reset_index().rename(columns={"Date":"timestamps"}); return raw[["timestamps","open","high","low","close","volume"]].dropna()
    return _clean(raw)

def _upstox_url(key, interval):
    enc=quote(key,safe="")
    if interval=="1m": return f"https://api.upstox.com/v2/historical-candle/intraday/{enc}/1minute"
    days=365*3 if interval=="1d" else 180; end=date.today(); start=end-timedelta(days=days)
    native="30minute" if interval in ("5m","15m","30m","60m") else "day"
    return f"https://api.upstox.com/v2/historical-candle/{enc}/{native}/{end:%Y-%m-%d}/{start:%Y-%m-%d}"

def _upstox(symbol, interval):
    key=symbol if "|" in symbol else NSE_SYMBOLS.get(symbol,symbol)
    req=Request(_upstox_url(key,interval),headers={"Accept":"application/json","User-Agent":"Mozilla/5.0","Authorization":"Bearer "+_token()})
    try:
        with urlopen(req,timeout=30) as r: payload=json.loads(r.read().decode())
    except HTTPError as e: raise RuntimeError(f"Upstox HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except URLError as e: raise RuntimeError(f"Upstox connection failed: {e.reason}") from e
    if payload.get("status")!="success": raise RuntimeError(str(payload.get("errors") or payload.get("message") or "Upstox request failed"))
    rows=[]
    for c in reversed(payload.get("data",{}).get("candles",[])):
        if len(c)>=6: rows.append({"timestamps":pd.to_datetime(c[0],utc=True).tz_convert(IST).tz_localize(None),"open":float(c[1]),"high":float(c[2]),"low":float(c[3]),"close":float(c[4]),"volume":float(c[5] or 0)})
    return _resample(pd.DataFrame(rows),interval)

def fetch_ohlcv(symbol, interval="5m", source="auto"):
    if interval not in INTERVALS: raise ValueError(f"Unsupported interval: {interval}")
    errors=[]
    if source in ("auto","yfinance"):
        try:
            df=_yf(symbol,interval)
            if not df.empty: return df,"yfinance"
            errors.append("yfinance returned no data")
        except Exception as e: errors.append(f"yfinance: {e}")
    if source in ("auto","upstox","kite"):
        try:
            df=_upstox(symbol,interval)
            if not df.empty: return df,"upstox"
            errors.append("upstox returned no data")
        except Exception as e: errors.append(f"upstox: {e}")
    raise RuntimeError(f"All data sources failed for {symbol} @ {interval}:\n"+"\n".join(errors))

def is_upstox_connected(): return bool(os.getenv("UPSTOX_ACCESS_TOKEN","").strip())
def set_upstox_session(*a,**k): return is_upstox_connected()
def is_kite_connected(): return is_upstox_connected()
def set_kite_session(*a,**k): return set_upstox_session(*a,**k)
def calculate_orb(df,orb_minutes=15):
    x=df.copy(); x["_date"]=x.timestamps.dt.date; x["_mins"]=x.timestamps.dt.hour*60+x.timestamps.dt.minute; out={}
    for day,g in x.groupby("_date"):
        w=g[(g._mins>=555)&(g._mins<555+orb_minutes)]
        if not w.empty: out[str(day)]={"high":float(w.high.max()),"low":float(w.low.min())}
    return out
def safe_float(v):
    try: n=float(v); return 0.0 if not math.isfinite(n) else round(n,4)
    except (TypeError,ValueError): return 0.0
def df_to_records(df): return [{"timestamp":r.timestamps.isoformat(),"open":safe_float(r.open),"high":safe_float(r.high),"low":safe_float(r.low),"close":safe_float(r.close),"volume":safe_float(r.volume)} for r in df.itertuples()]
