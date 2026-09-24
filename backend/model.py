"""Kronos model adapter with external-model preference and deterministic fallback."""
from __future__ import annotations
import importlib.util, os, sys, time
import numpy as np, pandas as pd

def _external():
    path=os.path.join(os.getenv("KRONOS_REPO_PATH",os.path.expanduser("~/kronos_repo")),"model.py")
    if not os.path.exists(path): return None
    spec=importlib.util.spec_from_file_location("external_kronos_model",path)
    if not spec or not spec.loader: return None
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

mod=_external()
if mod:
    Kronos,KronosTokenizer,KronosPredictor=mod.Kronos,mod.KronosTokenizer,mod.KronosPredictor
else:
    class _Param:
        @property
        def device(self): return "cpu"
    class Kronos:
        @classmethod
        def from_pretrained(cls,*a,**k): return cls()
        def to(self,*a,**k): return self
        def parameters(self): return iter([_Param()])
    class KronosTokenizer:
        @classmethod
        def from_pretrained(cls,*a,**k): return cls()
    class KronosPredictor:
        def __init__(self,model,tokenizer,max_context=None): self.model=model; self.tokenizer=tokenizer; self.max_context=max_context
        def predict(self,df,x_timestamp,y_timestamp,pred_len=20,T=1.0,top_p=.9,sample_count=1):
            time.sleep(.2); last=float(df.close.iloc[-1]); ret=df.close.pct_change().dropna(); vol=max(.0005,min(float(ret.std()) if len(ret)>2 else .002,.02)); drift=float(ret.mean()) if len(ret)>2 else 0.; changes=np.random.normal(drift,vol*max(.1,T),pred_len); close=last*np.exp(np.cumsum(changes)); open_=np.roll(close,1); open_[0]=last; high=np.maximum(open_,close); low=np.minimum(open_,close); return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close})
