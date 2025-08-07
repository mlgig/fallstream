import math
import numpy as np
from sklearn.metrics import confusion_matrix

def cost_fn(
    y_true=None,
    y_pred=None,
    cm=None,
    *,
    # gains side (sums to 1)
    fn_factor=2.0,
    tp_factor=1.0,   
):
    # --- get confusion matrix
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either cm or (y_true, y_pred).")
        cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = cm.ravel()

    gain_matrix = np.array(
        [
            [ 0,         -1],  # -1 gain for false positives
            [-fn_factor,  0],  # -fn_factor gain for false negatives
        ]
    )
    return np.sum(cm * gain_matrix)