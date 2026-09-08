"""
Protocol A (presegmented, per-window classification) evaluation.

The conventional protocol: build the test fold's window set using literally the
same segmentation function used for training (`create_training_data`), classify
each window independently, and score with standard per-window metrics. See
CLAUDE.md's "Protocol A — presegmented" section — do not re-derive segmentation
here, and do not change window construction to make this protocol easier or
harder; that turns the comparison into a strawman.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from ..segmentation.training_segmenter import create_training_data

__all__ = ["evaluate_presegmented"]


def evaluate_presegmented(
    trained_models: Dict[str, BaseEstimator],
    test_dfs: Iterable[pd.DataFrame],
    feature_cols: Sequence[str],
    label_col: str = "label",
    *,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 1.4,
    drop_below_threshold: bool = True,
    spacing: Union[int, str] = 5,
    exclusion_tolerance: float = 20.0,
    use_post_event_data: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Classify the test fold's presegmented window set independently (Protocol A).

    `trained_models` must be the same fitted estimators used for the streaming
    (Protocol B) evaluation of this fold — this function does not fit anything,
    it only builds the test window set and scores it. All segmentation kwargs
    should match whatever was passed to `create_training_data` for the training
    set, so both fall under the same segmentation definition (CLAUDE.md
    invariant: "Frozen models across protocols").
    """
    X_test, y_test = create_training_data(
        list(test_dfs),
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=window_size,
        step=step,
        freq=freq,
        signal_thresh=signal_thresh,
        drop_below_threshold=drop_below_threshold,
        spacing=spacing,
        exclusion_tolerance=exclusion_tolerance,
        use_post_event_data=use_post_event_data,
    )

    if verbose:
        n_falls = int((y_test == 1).sum())
        print(f"  Protocol A window set: {len(y_test)} windows "
              f"({n_falls} fall / {len(y_test) - n_falls} ADL)")

    rows = []
    for name, model in trained_models.items():
        thresh = getattr(model, "threshold_", 0.5)
        probs = model.predict_proba(X_test)
        pos_probs = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()
        y_pred = (pos_probs >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        balanced_accuracy = (recall + specificity) / 2

        rows.append({
            "model": name,
            "thresh": thresh,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n_windows": int(len(y_test)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1-score": f1,
            "balanced_accuracy": balanced_accuracy,
        })

    df = pd.DataFrame(rows)
    cols = ["model"] + [c for c in df.columns if c != "model"]
    return df[cols]
