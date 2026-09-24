import re

with open('/app/applet/backend/nse_dashboard_enhanced.py', 'r') as f:
    content = f.read()

pred_start = content.find("def _run_prediction(")
end_pred = content.find("def predict():")

new_pred = """def _run_prediction(job_id, df, x_df, x_ts, y_ts,
                    freq, pred_len, temperature, top_p, sample_count,
                    orb_minutes, lookback, source_label, compute_bands):
    try:
        import time, json
        import pandas as pd
        import numpy as np
        global _predictor
        global MODEL_AVAILABLE
        
        with _jobs_lock: _jobs[job_id]["status"] = "running"
        
        if not MODEL_AVAILABLE or _predictor is None:
            # Fallback mock prediction when Kronos model is not loaded
            print("Kronos model not available, using simulated prediction")
            time.sleep(2)
            pred_df = x_df.iloc[-pred_len:].copy()
            pred_df["timestamps"] = y_ts
            # Simulate random walk for prediction
            last_close = x_df["close"].iloc[-1]
            volatility = x_df["close"].pct_change().std()
            if pd.isna(volatility): volatility = 0.001
            
            closes, opens, highs, lows = [], [], [], []
            curr_close = last_close
            for i in range(pred_len):
                change = curr_close * np.random.normal(0, volatility * temperature)
                curr_open = curr_close
                curr_close = curr_open + change
                curr_high = max(curr_open, curr_close) + abs(curr_close * np.random.normal(0, volatility))
                curr_low = min(curr_open, curr_close) - abs(curr_close * np.random.normal(0, volatility))
                opens.append(curr_open)
                closes.append(curr_close)
                highs.append(curr_high)
                lows.append(curr_low)
                
            pred_df["open"] = opens
            pred_df["high"] = highs
            pred_df["low"] = lows
            pred_df["close"] = closes
            
            if compute_bands:
                std_close = np.std(closes)
                pred_bands = {
                    "mean": closes,
                    "upper": (np.array(closes) + std_close).tolist(),
                    "lower": (np.array(closes) - std_close).tolist()
                }
            else:
                pred_bands = None
        else:
            if compute_bands and sample_count > 1:
                pred_df, pred_bands = _multi_sample_bands(
                    _predictor, x_df, x_ts, y_ts, pred_len, temperature, top_p, n=sample_count
                )
            else:
                pred_df = _predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts.tolist(),
                    y_timestamp=y_ts.tolist(),
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False
                )
                pred_bands = None
            
        orb_levels = calculate_orb(df, orb_minutes)
        signal = generate_signal(x_df, pred_df, orb_levels)
        
        inds = _cached_indicators or get_all_indicators(df)
        chart_json = build_chart(df, pred_df, orb_levels, lookback, source_label, indicators=inds, show_rsi=True, show_macd=True, pred_bands=pred_bands)
        
        predictions = pred_df["close"].tolist()
        stats = {
            "mean_pred": np.mean(predictions),
            "max_pred": np.max(predictions),
            "min_pred": np.min(predictions),
            "volatility": np.std(predictions)
        }
        
        res = {
            "chart": chart_json,
            "signal": signal,
            "stats": stats,
            "message": "Forecast complete" + (" (Simulated)" if not MODEL_AVAILABLE else ""),
            "source": source_label
        }
        
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = res
            
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)

"""

if pred_start != -1 and end_pred != -1:
    content = content[:pred_start] + new_pred + content[end_pred:]
    with open('/app/applet/backend/nse_dashboard_enhanced.py', 'w') as f:
        f.write(content)
