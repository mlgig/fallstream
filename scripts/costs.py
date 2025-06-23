from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional

__all__ = ["cost_fn"]

def cost_fn(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    cm: Optional[np.ndarray] = None,
    *,
    fn_factor: float = 5.0,
    tp_factor: float = 1.0,
    fp_factor: float = 1.0,
) -> float:
    """Return *gain* given either a confusion matrix or (y_true, y_pred).

    Parameters
    ----------
    y_true , y_pred
        Ground‑truth and predicted labels (ignored if *cm* is supplied).
    cm
        Confusion matrix in the form ``[[tn, fp], [fn, tp]]``. If *None*, the
        matrix is computed from *y_true* and *y_pred*.
    fn_factor , tp_factor , fp_factor
        Cost multipliers for false negatives, true positives, and false
        positives.  True negatives always have unit gain.

    Notes
    -----
    Formula mirrors the scikit‑learn cost‑sensitive learning example.
    """

    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Either cm or (y_true, y_pred) must be provided")
        cm = confusion_matrix(y_true, y_pred)

    unit_cost = 1.0
    fn_cost   = fn_factor * unit_cost
    total     = unit_cost + fn_cost
    norm_fn   = fn_cost / total
    norm_fp   = (unit_cost / total) * fp_factor

    gain_matrix = np.array([
        [ unit_cost,     -norm_fp ],   # TN, FP
        [    -norm_fn, tp_factor*unit_cost ],   # FN, TP
    ])
    return float(np.sum(cm * gain_matrix))
