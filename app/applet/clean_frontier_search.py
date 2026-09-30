import os, sys, json, time
import numpy as np
import pandas as pd

DATA_PATH = "/app/applet/data/nifty50_1min_with_volume.csv"
print("Loading canonical dataset...")
df_raw = pd.read_csv(DATA_PATH)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

hours = df_raw['timestamp'].dt.hour
mins = df_raw['timestamp'].dt.minute
t_min = hours * 60 + mins

df_clean = df_raw[(t_min >= 555) & (t_min <= 929)].copy()
temp = df_clean.set_index('timestamp')
df_5m = temp.resample('5min', offset='15min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()
df_5m = df_5m[df_5m['volume'] > 0].reset_index()
n = len(df_5m)
print(f"Canonical 5m bars: {n:,}")

close = df_5m['close'].values
high = df_5m['high'].values
low = df_5m['low'].values
open_p = df_5m['open'].values
ts = df_5m['timestamp']
mod = (ts.dt.hour * 60 + ts.dt.minute).values
date_ints = (ts.dt.year * 10000 + ts.dt.month * 100 + ts.dt.day).values
ts_sec = (ts.values.astype('int64') // 10**9)

# 1. Chris Moody MACD (Fast 12, Slow 26, Signal SMA 9)
ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
macd = ema12 - ema26
signal_sma = pd.Series(macd).rolling(window=9).mean().values
hist = macd - signal_sma

hist_prev = np.zeros(n)
hist_prev[1:] = hist[:-1]
cm_aqua = (hist > 0) & (hist > hist_prev) # Accelerating bull (Aqua)
cm_red = (hist < 0) & (hist < hist_prev)   # Accelerating bear (Red)

# 2. AlgoAlpha ML Adaptive SuperTrend
tr10 = np.empty(n, dtype=np.float64)
tr10[0] = high[0] - low[0]
for i in range(1, n):
    tr10[i] = max(high[i] - low[i], max(abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
atr10 = np.empty(n, dtype=np.float64)
atr10[0] = tr10[0]
for i in range(1, n):
    atr10[i] = (atr10[i-1] * 9.0 + tr10[i]) / 10.0

assigned_centroids = np.empty(n, dtype=np.float64)
cluster_ids = np.empty(n, dtype=np.int32)
for i in range(min(100, n)):
    assigned_centroids[i] = atr10[i]
    cluster_ids[i] = 1

for i in range(99, n):
    w = atr10[i - 99 : i + 1]
    w_min = np.min(w); w_max = np.max(w)
    c_a = w_min + (w_max - w_min) * 0.75
    c_b = w_min + (w_max - w_min) * 0.50
    c_c = w_min + (w_max - w_min) * 0.25
    for _ in range(10):
        d1 = np.abs(w - c_a); d2 = np.abs(w - c_b); d3 = np.abs(w - c_c)
        m1 = (d1 < d2) & (d1 < d3)
        m2 = (d2 < d1) & (d2 < d3)
        m3 = (d3 < d1) & (d3 < d2)
        c_a = np.mean(w[m1]) if np.any(m1) else c_a
        c_b = np.mean(w[m2]) if np.any(m2) else c_b
        c_c = np.mean(w[m3]) if np.any(m3) else c_c
        
    curr_vol = atr10[i]
    d1 = abs(curr_vol - c_a); d2 = abs(curr_vol - c_b); d3 = abs(curr_vol - c_c)
    if d1 <= d2 and d1 <= d3:
        assigned_centroids[i] = c_a; cluster_ids[i] = 0
    elif d2 <= d1 and d2 <= d3:
        assigned_centroids[i] = c_b; cluster_ids[i] = 1
    else:
        assigned_centroids[i] = c_c; cluster_ids[i] = 2

def make_st(fact=3.0):
    st = np.zeros(n, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int32)
    lb = np.zeros(n, dtype=np.float64)
    ub = np.zeros(n, dtype=np.float64)
    src0 = (high[0] + low[0]) * 0.5
    lb[0] = src0 - fact * assigned_centroids[0]
    ub[0] = src0 + fact * assigned_centroids[0]
    direction[0] = 1; st[0] = ub[0]
    for i in range(1, n):
        src = (high[i] + low[i]) * 0.5
        basic_lb = src - fact * assigned_centroids[i]
        basic_ub = src + fact * assigned_centroids[i]
        prev_lb = lb[i-1]; prev_ub = ub[i-1]
        curr_lb = basic_lb if (basic_lb > prev_lb or close[i-1] < prev_lb) else prev_lb
        curr_ub = basic_ub if (basic_ub < prev_ub or close[i-1] > prev_ub) else prev_ub
        lb[i] = curr_lb; ub[i] = curr_ub
        prev_st = st[i-1]
        curr_dir = (-1 if close[i] > curr_ub else 1) if prev_st == prev_ub else (1 if close[i] < curr_lb else -1)
        direction[i] = curr_dir
        st[i] = curr_lb if curr_dir == -1 else curr_ub
    return st, direction

st3, dir3 = make_st(3.0)
st25, dir25 = make_st(2.5)

# 3. DMI 14
tr14 = np.empty(n, dtype=np.float64)
tr14[0] = high[0] - low[0]
for i in range(1, n):
    tr14[i] = max(high[i] - low[i], max(abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
atr14 = pd.Series(tr14).ewm(alpha=1/14, adjust=False).mean().values

pdm = np.zeros(n); mdm = np.zeros(n)
for i in range(1, n):
    up = high[i] - high[i-1]; down = low[i-1] - low[i]
    if up > down and up > 0: pdm[i] = up
    if down > up and down > 0: mdm[i] = down
sm_pdm = pd.Series(pdm).ewm(alpha=1/14, adjust=False).mean().values
sm_mdm = pd.Series(mdm).ewm(alpha=1/14, adjust=False).mean().values
pdi = 100.0 * (sm_pdm / np.where(atr14==0, 1.0, atr14))
mdi = 100.0 * (sm_mdm / np.where(atr14==0, 1.0, atr14))

def simulate(signals, superTrend, direction, max_hold_sec=3600, 
             stagnation_exit=True, stag_range_pts=20.0, target_pts=None):
    trades = []
    current_pos = 0
    entry_price = 0.0
    entry_time = 0
    entry_date = 0
    current_stop = 0.0
    entry_bar_idx = -1
    occupied_bar = -1
    
    for i in range(n - 1):
        if current_pos != 0 and i >= entry_bar_idx:
            should_exit = False
            exit_price = 0.0
            exit_reason = ""
            
            # 1. Profit Target check
            if target_pts is not None:
                if current_pos == 1:
                    target_p = entry_price + target_pts
                    if high[i] >= target_p:
                        should_exit = True
                        exit_price = max(open_p[i], target_p) - 0.05
                        exit_reason = "TARGET"
                elif current_pos == -1:
                    target_p = entry_price - target_pts
                    if low[i] <= target_p:
                        should_exit = True
                        exit_price = min(open_p[i], target_p) + 0.05
                        exit_reason = "TARGET"
                        
            # 2. Stop loss check (trailing ratchet)
            if not should_exit:
                if current_pos == 1:
                    if low[i] <= current_stop:
                        should_exit = True
                        exit_price = min(open_p[i], current_stop) - 0.05
                        exit_reason = "STOP_LOSS"
                elif current_pos == -1:
                    if high[i] >= current_stop:
                        should_exit = True
                        exit_price = max(open_p[i], current_stop) + 0.05
                        exit_reason = "STOP_LOSS"
                        
            # 3. EOD 15:20 check
            if not should_exit:
                if mod[i] >= 920 or (i < n - 1 and date_ints[i+1] != entry_date):
                    should_exit = True
                    exit_price = close[i] - 0.05 if current_pos == 1 else close[i] + 0.05
                    exit_reason = "EOD_1520"
                    
            # 4. 35-minute stagnation exit:
            # "exit if trails within small range for more than 35 minutes"
            # 35 min in 5m bars is 7 bars
            if not should_exit and stagnation_exit:
                bars_held = i - entry_bar_idx + 1
                if bars_held >= 7:
                    trailing_35m_range = np.max(high[i-6:i+1]) - np.min(low[i-6:i+1])
                    if trailing_35m_range <= stag_range_pts:
                        should_exit = True
                        exit_price = close[i] - 0.05 if current_pos == 1 else close[i] + 0.05
                        exit_reason = "STAGNATION_35M"
                        
            # 5. Max hold cap
            if not should_exit:
                if (ts_sec[i] - entry_time) >= max_hold_sec:
                    should_exit = True
                    exit_price = close[i] - 0.05 if current_pos == 1 else close[i] + 0.05
                    exit_reason = "MAX_HOLD"
                    
            if should_exit:
                pnl = (exit_price - entry_price) if current_pos == 1 else (entry_price - exit_price)
                trades.append({
                    'type': 'CALL' if current_pos == 1 else 'PUT',
                    'pnl': round(pnl, 2),
                    'reason': exit_reason,
                    'bars': i - entry_bar_idx + 1
                })
                current_pos = 0
                occupied_bar = i
            else:
                if current_pos == 1:
                    if direction[i] == -1 and superTrend[i] > current_stop:
                        current_stop = superTrend[i]
                elif current_pos == -1:
                    if direction[i] == 1 and superTrend[i] < current_stop:
                        current_stop = superTrend[i]
                        
        if current_pos == 0 and signals[i] != 0:
            if i != occupied_bar:
                next_mod = mod[i+1]
                next_date = date_ints[i+1]
                if next_date == date_ints[i] and next_mod < 920:
                    sig = signals[i]
                    current_pos = sig
                    entry_bar_idx = i + 1
                    entry_time = ts_sec[i+1]
                    entry_date = next_date
                    current_stop = superTrend[i]
                    entry_price = open_p[i+1] + 0.05 if sig == 1 else open_p[i+1] - 0.05
                    
    return trades

def calc(trades):
    if not trades or len(trades) < 10: return {'wr': 0, 'pf': 0, 'pts': 0, 'total': len(trades) if trades else 0, 'avg': 0, 'dd': 0}
    pnls = [t['pnl'] for t in trades]
    w = [p for p in pnls if p > 0]
    l = [abs(p) for p in pnls if p < 0]
    wr = len(w) / len(pnls) * 100.0
    pf = sum(w) / sum(l) if sum(l) > 0 else 999.0
    tp = sum(pnls)
    avg = tp / len(pnls)
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = np.max(dd) if len(dd) > 0 else 0.0
    return {'wr': round(wr, 2), 'pf': round(pf, 2), 'pts': round(tp, 2), 'total': len(trades), 'avg': round(avg, 2), 'dd': round(max_dd, 2)}

results = []
combos = [
    ("Common Filter (5 conditions)", st3, dir3, 0.90, 5.0, 0.10, False),
    ("Common Filter + CM Aqua/Red Accel", st3, dir3, 0.90, 5.0, 0.10, True),
    ("ST(2.5) + Common Filter + CM Accel", st25, dir25, 0.90, 5.0, 0.10, True),
    ("High Str (>=0.92) + DI (>=8) + CM Accel", st3, dir3, 0.92, 8.0, 0.10, True),
    ("High Str (>=0.92) + DI (>=8) + ST(2.5)", st25, dir25, 0.92, 8.0, 0.10, True),
    ("Selective DI (>=10) + Hist/ATR (>=0.15)", st3, dir3, 0.90, 10.0, 0.15, True),
]

for label, st_arr, dir_arr, min_str, min_di, min_h_atr, use_cm in combos:
    sig = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if np.isnan(signal_sma[i]): continue
        bull_shift = (dir_arr[i-1] == 1 and dir_arr[i] == -1)
        bear_shift = (dir_arr[i-1] == -1 and dir_arr[i] == 1)
        
        macd_bull = (macd[i] > 0.0 and macd[i] >= signal_sma[i])
        macd_bear = (macd[i] < 0.0 and macd[i] < signal_sma[i])
        
        if use_cm:
            macd_bull = macd_bull and cm_aqua[i]
            macd_bear = macd_bear and cm_red[i]
            
        rng = high[i] - low[i]
        if rng > 1e-4 and mod[i] < 840:
            call_strength = (close[i] - low[i]) / rng
            put_strength = (high[i] - close[i]) / rng
            cluster_ok = (cluster_ids[i] != 1)
            call_di = (pdi[i] - mdi[i]) >= min_di
            put_di = (mdi[i] - pdi[i]) >= min_di
            curr_atr = atr10[i] if atr10[i] > 0 else 1.0
            call_h_atr = (hist[i] / curr_atr) >= min_h_atr
            put_h_atr = (-hist[i] / curr_atr) >= min_h_atr
            
            if cluster_ok:
                if bull_shift and macd_bull and call_strength >= min_str and call_di and call_h_atr:
                    sig[i] = 1
                elif bear_shift and macd_bear and put_strength >= min_str and put_di and put_h_atr:
                    sig[i] = -1
                    
    raw_s = np.sum(sig != 0)
    for target in [None, 25.0, 35.0, 45.0, 60.0]:
        for stag in [False, True]:
            for stag_pts in [15.0, 25.0] if stag else [0.0]:
                for hold in [30, 45, 60, 120]:
                    trades = simulate(sig, st_arr, dir_arr, max_hold_sec=hold*60, 
                                      stagnation_exit=stag, stag_range_pts=stag_pts, 
                                      target_pts=target)
                    st = calc(trades)
                    if st['total'] >= 20:
                        results.append({
                            'config': label,
                            'target': f"{target} pts" if target else "None",
                            'stag': f"Yes ({stag_pts} pts)" if stag else "No",
                            'hold': f"{hold}m",
                            **st
                        })

results.sort(key=lambda x: (x['wr'] >= 70.0, x['wr'], x['pf']), reverse=True)

print("\n" + "="*130)
print(f"{'CONFIGURATION':<38} | {'TARGET':<8} | {'STAGNATION 35M':<16} | {'HOLD':<5} | {'WIN RATE':<9} | {'PF':<5} | {'TRADES':<6} | {'PTS':<9} | {'AVG':<6}")
print("="*130)
for r in results[:25]:
    print(f"{r['config']:<38} | {r['target']:<8} | {r['stag']:<16} | {r['hold']:<5} | {r['wr']:>7.2f}% | {r['pf']:>4.2f} | {r['total']:<6} | {r['pts']:>7.1f} pt | {r['avg']:>5.1f}")
