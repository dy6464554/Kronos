"""Upstox-only OHLCV provider for the Kronos NSE terminal."""
from __future__ import annotations
import json, math, os
from datetime import date, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd
import pytz

IST=pytz.timezone("Asia/Kolkata"); MARKET_OPEN=time(9,15); MARKET_CLOSE=time(15,30)
NSE_SYMBOLS={"NIFTY 50":"NSE_INDEX|Nifty 50","BANK NIFTY":"NSE_INDEX|Nifty Bank","NIFTY IT":"NSE_INDEX|Nifty IT","FIN NIFTY":"NSE_INDEX|Nifty Financial Services","RELIANCE":"NSE_EQ|INE002A01018","TCS":"NSE_EQ|INE467B01029","HDFC BANK":"NSE_EQ|INE040A01034","INFOSYS":"NSE_EQ|INE009A01021","ICICI BANK":"NSE_EQ|INE090A01021","KOTAK BANK":"NSE_EQ|INE237A01028","AXIS BANK":"NSE_EQ|INE238A01034","SBI":"NSE_EQ|INE062A01020","LT":"NSE_EQ|INE018A01030","WIPRO":"NSE_EQ|INE075A01022","HCL TECH":"NSE_EQ|INE860A01011","BAJAJ FINANCE":"NSE_EQ|INE296A01024","MARUTI":"NSE_EQ|INE585B01010","TITAN":"NSE_EQ|INE280A01028","ASIAN PAINTS":"NSE_EQ|INE021A01026"}
INTERVALS={"1m":"1minute","5m":"5minute","15m":"15minute","30m":"30minute","60m":"60minute","1d":"day"}; YF_PERIOD={k:"upstox" for k in INTERVALS}

def _token():
    token=os.getenv("UPSTOX_ACCESS_TOKEN","").strip()
    if not token: raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured in AI Studio Secrets.")
    return token

def _key(symbol): return symbol if "|" in symbol else NSE_SYMBOLS.get(symbol.strip(),symbol.strip())
def _request(url):
    req=Request(url,headers={"Accept":"application/json","Authorization":"Bearer "+_token()})
    try:
        with urlopen(req,timeout=30) as r: payload=json.loads(r.read().decode())
    except HTTPError as e: raise RuntimeError(f"Upstox HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from e
    except URLError as e: raise RuntimeError(f"Upstox connection failed: {e.reason}") from e
    if payload.get("status")!="success": raise RuntimeError(str(payload.get("errors") or payload.get("message") or "Upstox request failed"))
    return payload

def fetch_ohlcv(symbol,interval="5m",source="auto"):
    if interval not in INTERVALS: raise ValueError(f"Unsupported interval: {interval}")
    key=_key(symbol)
    if not key: raise ValueError(f"Unknown NSE symbol: {symbol}")
    enc=quote(key,safe="")
    if interval=="1m": url=f"https://api.upstox.com/v2/historical-candle/intraday/{enc}/1minute"
    else:
        days=365*3 if interval=="1d" else 180
        end=date.today(); start=end-timedelta(days=days)
        url=f"https://api.upstox.com/v2/historical-candle/{enc}/{INTERVALS[interval]}/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
    candles=_request(url).get("data",{}).get("candles",[]); rows=[]
    for c in reversed(candles):
        if len(c)<6: continue
        try: rows.append({"timestamps":pd.to_datetime(c[0],utc=True).tz_convert(IST).tz_localize(None),"open":float(c[1]),"high":float(c[2]),"low":float(c[3]),"close":float(c[4]),"volume":float(c[5] or 0)})
        except Exception: pass
    df=pd.DataFrame(rows).drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    if df.empty: raise RuntimeError(f"Upstox returned no candles for {symbol} @ {interval}")
    if interval!="1d": df=df[(df.timestamps.dt.time>=MARKET_OPEN)&(df.timestamps.dt.time<=MARKET_CLOSE)].reset_index(drop=True)
    return df,"upstox"

def is_upstox_connected(): return bool(os.getenv("UPSTOX_ACCESS_TOKEN","").strip())
def set_upstox_session(*a,**k): return is_upstox_connected()
def is_kite_connected(): return is_upstox_connected()
def set_kite_session(*a,**k): return set_upstox_session(*a,**k)
def calculate_orb(df,orb_minutes=15):
    w=df.copy(); w["_date"]=w.timestamps.dt.date; w["_mins"]=w.timestamps.dt.hour*60+w.timestamps.dt.minute; out={}
    for day,g in w.groupby("_date"):
        x=g[(g._mins>=555)&(g._mins<555+orb_minutes)]
        if not x.empty: out[str(day)]={"high":float(x.high.max()),"low":float(x.low.min())}
    return out
def safe_float(v):
    try: n=float(v); return 0.0 if not math.isfinite(n) else round(n,4)
    except (TypeError,ValueError): return 0.0
def df_to_records(df): return [{"timestamp":r.timestamps.isoformat(),"open":safe_float(r.open),"high":safe_float(r.high),"low":safe_float(r.low),"close":safe_float(r.close),"volume":safe_float(r.volume)} for r in df.itertuples()]
