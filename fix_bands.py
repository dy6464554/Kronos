import re

with open('/app/applet/backend/nse_dashboard_enhanced.py', 'r') as f:
    content = f.read()

bands_start = content.find("def _multi_sample_bands(")
pred_start = content.find("def _run_prediction(")

if bands_start != -1 and pred_start != -1:
    new_bands = """def _multi_sample_bands(predictor, x_df, x_ts, y_ts, pred_len, temp, top_p, n=5):
    import numpy as np
    import pandas as pd
    
    samples = []
    for _ in range(n):
        pdf = predictor.predict(
            df=x_df,
            x_timestamp=x_ts.tolist(),
            y_timestamp=y_ts.tolist(),
            pred_len=pred_len,
            T=temp,
            top_p=top_p,
            sample_count=1,
            verbose=False
        )
        samples.append(pdf)
        
    closes = np.array([s["close"].values for s in samples])
    mean_close = closes.mean(axis=0)
    std_close = closes.std(axis=0)
    
    pred_df = samples[0].copy()
    pred_df["close"] = mean_close
    # For open, high, low just use mean or average for simplicity, or just average them all
    opens = np.array([s["open"].values for s in samples]).mean(axis=0)
    highs = np.array([s["high"].values for s in samples]).mean(axis=0)
    lows = np.array([s["low"].values for s in samples]).mean(axis=0)
    pred_df["open"] = opens
    pred_df["high"] = highs
    pred_df["low"] = lows
    
    pred_bands = {
        "mean": mean_close.tolist(),
        "upper": (mean_close + std_close).tolist(),
        "lower": (mean_close - std_close).tolist()
    }
    return pred_df, pred_bands

"""
    content = content[:bands_start] + new_bands + content[pred_start:]
    with open('/app/applet/backend/nse_dashboard_enhanced.py', 'w') as f:
        f.write(content)
