import numpy as np
import pandas as pd
import time

class MockParam:
    @property
    def device(self):
        return "cpu"

class Kronos:
    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        print(f"[Simulation] Loaded mock Kronos model: {model_id}")
        return cls()
    
    def to(self, device):
        pass
        
    def parameters(self):
        return iter([MockParam()])

class KronosTokenizer:
    @classmethod
    def from_pretrained(cls, tokenizer_id, *args, **kwargs):
        print(f"[Simulation] Loaded mock Kronos tokenizer: {tokenizer_id}")
        return cls()

class KronosPredictor:
    def __init__(self, model, tokenizer, max_context=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context
        
    def predict(self, df, x_timestamp, y_timestamp, pred_len=20, T=1.0, top_p=0.9, sample_count=1):
        """
        Mock prediction simulating realistic price movements.
        """
        time.sleep(1) # simulate inference time
        
        last_close = df['close'].iloc[-1]
        
        # Calculate recent volatility and drift to make it realistic
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 2 else 0.002
        drift = returns.mean() if len(returns) > 2 else 0.0
        
        volatility = max(0.0005, min(volatility, 0.02)) # cap volatility
        
        # T (temperature) scales the volatility
        scaled_vol = volatility * max(0.1, T)
        
        changes = np.random.normal(drift, scaled_vol, pred_len)
        closes = last_close * np.exp(np.cumsum(changes))
        
        # Generate OHLC based on the closes
        opens = np.roll(closes, 1)
        opens[0] = last_close
        
        # High and Low adding some random noise around open/close
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, scaled_vol/2, pred_len)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, scaled_vol/2, pred_len)))
        
        pred_df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes
        })
        
        return pred_df
