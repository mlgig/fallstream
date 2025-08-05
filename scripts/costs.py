import math
import numpy as np
from sklearn.metrics import confusion_matrix

def cost_fn(
    y_true=None,
    y_pred=None,
    cm=None,
    *,
    # gains side (sums to 1)
    tp_over_tn: float = 2.0,  # “TP is 2x as valuable as TN”
    # costs side (sums to 1)
    fn_cap: float = 10.0,      
    imbalance_mode: str = "log",  # {"log", "sqrt", "ratio"}
    normalise: bool = False,    
):
    """
    Utility = (+TN * tn_gain + TP * tp_gain) - (FP * fp_cost + FN * fn_cost)

    Gains (TP+TN) are normalised to sum to 1. Costs (FP+FN) are also
    normalised to sum to 1, with FN's share growing with class imbalance.
    """
    # --- get confusion matrix
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either cm or (y_true, y_pred).")
        cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_pos, n_neg = tp + fn, tn + fp

    # gains
    tp_gain = tp_over_tn / (1.0 + tp_over_tn)
    tn_gain = 1.0 / (1.0 + tp_over_tn)

    # costs
    ratio = n_neg / max(n_pos, 1)
    if imbalance_mode == "log":
        r = max(0.0, math.log10(ratio))
    elif imbalance_mode == "sqrt":
        r = math.sqrt(ratio) - 1.0  
        r = max(0.0, r)
    elif imbalance_mode == "ratio":
        r = max(0.0, ratio - 1.0)
    else:
        raise ValueError(f"Unknown imbalance_mode={imbalance_mode}")

    r = min(r, fn_cap)  

    # normalise FP/FN to sum to 1
    fp_cost = 1.0 / (1.0 + r)
    fn_cost = r   / (1.0 + r)

    gain = (
        tn_gain * tn
        + tp_gain * tp
        - fp_cost * fp
        - fn_cost * fn
    )

    if normalise:
        gain /= (tn + fp + fn + tp) or 1

    return float(gain)