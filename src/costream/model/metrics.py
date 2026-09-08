from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# =============================================================================
# 1. Unified Cost Function
# =============================================================================

def cost_score(
    y_true: np.ndarray = None, 
    y_pred: np.ndarray = None, 
    cm: np.ndarray = None, 
    alpha: float = 2.0
) -> float:
    """
    Compute the cost-sensitive gain (negative cost).
    
    Formula: Gain = -(FP + alpha * FN)
    
    Parameters
    ----------
    y_true : array-like, optional
        True labels.
    y_pred : array-like, optional
        Predicted labels.
    cm : array-like, optional
        Confusion matrix [[TN, FP], [FN, TP]]. 
        If provided, y_true and y_pred are ignored.
    alpha : float, default=2.0
        The penalty factor for False Negatives relative to False Positives.
        (e.g., if alpha=2, a missed fall is 2x worse than a false alarm).
        
    Returns
    -------
    float
        The total gain (always <= 0). Higher is better.
    """
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Must provide either 'cm' or both 'y_true' and 'y_pred'.")
        cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Gain = -1 * FP  +  -alpha * FN
    return -(fp + alpha * fn)


# =============================================================================
# 2. Metric Registry & Definitions
# =============================================================================

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

# --- Standard Classifiers Metrics ---
# NOTE: We add **_ to all lambdas to ignore extra keyword arguments like 'signal_time'
register(Metric("precision", lambda tp, fp, tn, fn, **_: _safe_div(tp, tp + fp), "Precision"))
register(Metric("recall", lambda tp, fp, tn, fn, **_: _safe_div(tp, tp + fn), "Recall"))
register(Metric("specificity", lambda tp, fp, tn, fn, **_: _safe_div(tn, tn + fp), "Specificity"))
register(Metric("f1-score", lambda tp, fp, tn, fn, **_: _safe_div(2*tp, 2*tp + fp + fn), "F1"))
register(Metric("auc", lambda tp, fp, tn, fn, **_: (_safe_div(tp, tp+fn) + _safe_div(tn, tn+fp)) / 2, "AUC-ROC (est.)"))

# --- Streaming / Time-Based Metrics ---
# These require 'signal_time' (in ms) to be passed as a keyword argument.
# 360,000 ms = 1 hour.
register(Metric(
    "false alarm rate", 
    lambda tp, fp, tn, fn, signal_time, **__: _safe_div(fp, signal_time / 360_000), 
    "FAR / h"
))

register(Metric(
    "miss rate", 
    lambda tp, fp, tn, fn, signal_time, **__: _safe_div(fn, signal_time / 360_000), 
    "MR / h"
))

# --- Cost Metric ---
register(Metric(
    "gain",
    lambda tp, fp, tn, fn, signal_time, **kwargs: 
        cost_score(cm=np.array([[tn, fp], [fn, tp]]), alpha=kwargs.get('alpha', 2.0)) / (signal_time / 360_000),
    "Gain / h"
))


# =============================================================================
# 3. Helper Functions
# =============================================================================

def compute_metrics_from_cm(
    cm: np.ndarray, 
    signal_time: float, 
    runtime: float = 0.0, 
    delay: float = 0.0,
    **kwargs
) -> dict:
    """
    Calculate all registered metrics from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    signal_time : float
        Total duration of the signal in milliseconds (used for rates per hour).
    runtime : float
        Total inference time in seconds.
    delay : float
        Average detection delay in seconds.
    **kwargs : dict
        Additional arguments passed to metric functions (e.g., alpha).

    Returns
    -------
    dict
        Dictionary of {metric_name: value}.
    """
    tn, fp, fn, tp = cm.ravel()
    out = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "runtime": runtime,
        "delay": delay,
    }
    for m in REGISTRY.values():
        # Pass individual counts plus all kwargs
        out[m.name] = m.func(tp=tp, fp=fp, tn=tn, fn=fn, signal_time=signal_time, **kwargs)
    return out


_EXCLUDE_NUMERIC = {"seed", "fold"}

def aggregate_metrics(df: pd.DataFrame, group: str = "model") -> pd.DataFrame:
    """
    Group results by model and return Mean ± Std string representations.
    """
    if df.empty:
        return df
        
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in _EXCLUDE_NUMERIC]
    
    means = df.groupby(group)[num_cols].mean()
    stds  = df.groupby(group)[num_cols].std()
    
    aggr = pd.DataFrame(index=means.index)
    for col in num_cols:
        aggr[col] = means[col].round(2).astype(str) + " ± " + stds[col].round(2).astype(str)
    
    aggr.reset_index(inplace=True)
    return aggr