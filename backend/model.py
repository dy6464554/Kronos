"""Runtime-safe fallback model loader.

This file supports the actual Kronos model when the environment exposes a
`KRONOS_REPO_PATH` checkout; otherwise it falls back to a minimal mock so the
terminal still starts in AI Studio.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd


def _load_external_kronos():
    repo_path = os.environ.get("KRONOS_REPO_PATH") or os.path.expanduser("~/kronos_repo")
    candidate = os.path.join(repo_path, "model.py")
    if not os.path.exists(candidate):
        return None
    spec = importlib.util.spec_from_file_location("external_kronos_model", candidate)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["external_kronos_model"] = module
    spec.loader.exec_module(module)
    return module


_loaded = _load_external_kronos()
if _loaded is not None:
    Kronos = _loaded.Kronos
    KronosTokenizer = _loaded.KronosTokenizer
    KronosPredictor = _loaded.KronosPredictor
else:
    class MockParam:
        @property
        def device(self):
            return "cpu"

    class Kronos:
        @classmethod
        def from_pretrained(cls, model_id, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        @property
        def parameters(self):
            return lambda: iter([MockParam()])

    class KronosTokenizer:
        @classmethod
        def from_pretrained(cls, tokenizer_id, *args, **kwargs):
            return cls()

    class KronosPredictor:
        def __init__(self, model, tokenizer, max_context=None):
            self.model = model
            self.tokenizer = tokenizer
            self.max_context = max_context

        def predict(self, df, x_timestamp, y_timestamp, pred_len=20, T=1.0, top_p=0.9, sample_count=1):
            time.sleep(0.2)
            last_close = float(df["close"].iloc[-1])
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() if len(returns) > 2 else 0.002
            drift = returns.mean() if len(returns) > 2 else 0.0
            volatility = max(0.0005, min(volatility, 0.02))
            noise = volatility * max(0.1, T)
            changes = np.random.normal(drift, noise, pred_len)
            closes = last_close * np.exp(np.cumsum(changes))
            opens = np.roll(closes, 1)
            opens[0] = last_close
            highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, noise / 2, pred_len)))
            lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, noise / 2, pred_len)))
            return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
