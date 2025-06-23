from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence
import numpy as np
import pandas as pd
from scripts.costs import cost_fn

@dataclass(frozen=True)
class Metric:
    name: str                                 
    func: Callable[..., float]                
    display: str | None = None                

REGISTRY: Dict[str, Metric] = {}

def register(metric: Metric):
    if metric.name in REGISTRY:
        raise ValueError(f"Metric '{metric.name}' already registered")
    REGISTRY[metric.name] = metric

def _safe_div(n, d):
    return n / d if d else 0.0

register(Metric("precision", lambda tp, fp, *_: _safe_div(tp, tp + fp), "Precision"))
register(Metric("recall", lambda tp, fp, tn, fn, *_: _safe_div(tp, tp + fn), "Recall"))
register(Metric("specificity", lambda tp, fp, tn, fn, *_: _safe_div(tn, tn + fp), "Specificity"))
register(Metric("f1-score", lambda tp, fp, tn, fn, *_: _safe_div(2*tp, 2*tp + fp + fn), "F1"))
register(Metric("auc", lambda tp, fp, tn, fn, *_: (_safe_div(tp, tp+fn) + _safe_div(tn, tn+fp)) / 2, "AUC‑ROC est."))

# FAR and MR need signal_time (milliseconds)
register(Metric("false alarm rate", lambda tp, fp, tn, fn, signal_time, **__:
         _safe_div(fp, signal_time / 360_000), "FAR / h"))
register(Metric("miss rate", lambda tp, fp, tn, fn, signal_time, **__:
         _safe_div(fn, signal_time / 360_000), "MR / h"))

register(Metric(
    "gain",
    lambda tp, fp, tn, fn, *_:
        cost_fn(cm=np.array([[tn, fp], [fn, tp]])) /
        ((np.sum([tp, fp, tn, fn]) or 1) * 1000),
    "Gain"
))

def compute_row(cm: np.ndarray, signal_time: float, runtime: float, delay: float) -> dict:
    """Return a dict {metric_name: value} for all registered metrics."""
    tn, fp, fn, tp = cm.ravel()
    out = {
        "runtime": runtime,
        "delay":   delay,
    }
    for m in REGISTRY.values():
        out[m.name] = m.func(tp, fp, tn, fn, signal_time)
    return out

_EXCLUDE_NUMERIC = {"seed", "fold"}

def aggregate(df: pd.DataFrame, group: str = "model") -> pd.DataFrame:
    """Return mean ± std for each metric."""
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in _EXCLUDE_NUMERIC]
    means = df.groupby(group)[num_cols].mean()
    stds  = df.groupby(group)[num_cols].std()
    aggr  = pd.DataFrame(index=means.index)
    for col in num_cols:
        aggr[col] = means[col].round(2).astype(str) + " ± " + stds[col].round(2).astype(str)
    aggr.reset_index(inplace=True)
    return aggr
