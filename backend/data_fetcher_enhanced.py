"""Upstox-only OHLCV provider used by the terminal.

The token is read at request time from UPSTOX_ACCESS_TOKEN and is never stored
in source, logs, or a module-level variable.
"""
from __future__ import annotations
import json, math, os
from datetime import date, datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import pandas as pd

IST = "Asia/Kolkata"
MARKET_OPEN, MARKET_CLOSE = time(9, 15), time(15, 30)
NSE_SYMBOLS = {
 "NIFTY 50":"NSE_INDEX|Nifty 50", "BANK NIFTY":"NSE_INDEX|Nifty Bank", "NIFTY IT":"NSE_INDEX|Nifty IT",
 "RELIANCE":"NSE_EQ|INE002A01018", "TCS":"NSE_EQ|INE467B01018", "HDFC BANK":"NSE_EQ|INE040A01034",
 "INFOSYS":"NSE_EQ|INE009A01021", "ICICI BANK":"NSE_EQ|INE090A01021", "KOTAK BANK":"NSE_EQ|INE237A01028",
 "AXIS BANK":"NSE_EQ|INE238A01034", "SBI":"NSE_EQ|INE062A01020", "LT":"NSE_EQ|INE018A01030",
 "WIPRO":"NSE_EQ|INE075A01022", "HCL TECH":"NSE_EQ|INE860A01027", "BAJAJ FINANCE":"NSE_EQ|INE296A01024",
 "MARUTI":"NSE_EQ|INE585B01010", "TITAN":"NSE_EQ|INE280A01028", "ASIAN PAINTS":"NSE_EQ|INE021A01026"
}
INTERVALS = {"1m":"1minute", "5m":"5minute", "15m":"15minute", "30m":"30minute", "60m":"60minute", "1d":"day"}
YF_PERIOD = {key:"upstox" for key in INTERVALS}

def _token():
    token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token: raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured in AI Studio Secrets.")
    return token

def _request(url):
    req = Request(url, headers={"Accept":"application/json", "Authorization":"Bearer " + _token()})
    try:
        with urlopen(req, timeout=30) as response: payload=json.loads(response.read().decode())
    except HTTPError as exc:
        detail=exc.read().decode(errors="replace")
        raise RuntimeError(f"Upstox HTTP {exc.code}: {detail[:400]}") from exc
    except URLError as exc: raise RuntimeError(f"Upstox connection failed: {exc.reason}") from exc
    if payload.get("status") != "success": raise RuntimeError(str(payload.get("errors") or payload.get("message") or "Upstox request failed"))
    return payload

def _historical_url(key, interval, end, start):
    return f"https://api.upstox.com/v2/historical-candle/{quote(key, safe='')}/{INTERVALS[interval]}/{end:%Y-%m-%d}/{start:%Y-%m-%d}"

def fetch_ohlcv(symbol, interval="5m", source="upstox"):
    if source not in ("upstox", "auto"): raise ValueError("Only Upstox is supported as a data source.")
    key=symbol if "|" in symbol else NSE_SYMBOLS.get(symbol)
    if not key: raise ValueError(f"Unknown NSE symbol: {symbol}")
    if interval not in INTERVALS: raise ValueError(f"Unsupported interval: {interval}")
    days=365*5 if interval=="1d" else 7 if interval=="1m" else 180
    candles=list(reversed(_request(_historical_url(key, interval, date.today(), date.today()-timedelta(days=days))).get("data",{}).get("candles",[])))
    records=[]
    for c in candles:
        if len(c)<6: continue
        ts=pd.to_datetime(c[0], utc=True).tz_convert(IST).tz_localize(None)
        records.append({"timestamps":ts,"open":float(c[1]),"high":float(c[2]),"low":float(c[3]),"close":float(c[4]),"volume":float(c[5] or 0)})
    df=pd.DataFrame(records).dropna().drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    if df.empty: raise RuntimeError(f"Upstox returned no candles for {symbol} @ {interval}")
    if interval!="1d": df=df[(df.timestamps.dt.time>=MARKET_OPEN)&(df.timestamps.dt.time<=MARKET_CLOSE)].reset_index(drop=True)
    return df,"upstox"

def is_upstox_connected(): return bool(os.getenv("UPSTOX_ACCESS_TOKEN", "").strip())
def set_upstox_session(*_args, **_kwargs): return is_upstox_connected()
def is_kite_connected(): return is_upstox_connected()
def set_kite_session(*args, **kwargs): return set_upstox_session(*args, **kwargs)

def calculate_orb(df, orb_minutes=15):
    work=df.copy(); work["_date"]=work.timestamps.dt.date; work["_mins"]=work.timestamps.dt.hour*60+work.timestamps.dt.minute; start=555; result={}
    for day, group in work.groupby("_date"):
        w=group[(group._mins>=start)&(group._mins<start+orb_minutes)]
        if not w.empty: result[str(day)]={"high":float(w.high.max()),"low":float(w.low.min())}
    return result

def safe_float(value):
    try:
        n=float(value); return 0.0 if not math.isfinite(n) else round(n,4)
    except (TypeError,ValueError): return 0.0

def df_to_records(df):
    return [{"timestamp":r.timestamps.isoformat(),"open":safe_float(r.open),"high":safe_float(r.high),"low":safe_float(r.low),"close":safe_float(r.close),"volume":safe_float(r.volume)} for r in df.itertuples()]
