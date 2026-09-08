"""
Experiment runner (Multi-Event Support).
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from imblearn.over_sampling import RandomOverSampler

from ..model.metrics import compute_metrics_from_cm
from ..segmentation.streaming_segmenter import sliding_window_inference
from .event_detection import evaluate_recording
from .probability_cache import CacheKey, ProbabilityCache

__all__ = ["ModelSpec", "run_experiment", "train_models", "evaluate_models"]

@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: Optional[Dict[str, Any]] = None
    def clone(self):
        return ModelSpec(self.name, clone(self.estimator), self.param_grid)

def resample_training_data(X, y, random_state=42):
    # Handle both 2D and 3D arrays
    original_shape = X.shape
    if len(original_shape) == 2:
        X_flat = X
    elif len(original_shape) == 3:
        n_cases = original_shape[0]
        X_flat = X.reshape(n_cases, -1)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {original_shape}")

    # Apply random over-sampling
    ros = RandomOverSampler(random_state=random_state)
    X_resampled_flat, y_resampled = ros.fit_resample(X_flat, y)

    # Restore to original shape
    X_resampled = X_resampled_flat.reshape(X_resampled_flat.shape[0], *original_shape[1:])
    return X_resampled, y_resampled

def train_models(X_train, y_train, specs, verbose=True, resample=True, random_state=42):
    trained = {}
    if resample:
        if verbose: print("Resampling training data to address class imbalance...")
        X_train, y_train = resample_training_data(X_train, y_train, random_state=random_state)
    if verbose: print(f"TRAINING {len(specs)} models...")
    for spec in specs:
        model = clone(spec.estimator)
        model.fit(X_train, y_train)
        trained[spec.name] = model
    return trained

def evaluate_models(
    trained_models: Dict[str, BaseEstimator],
    test_signals: Sequence[np.ndarray],
    test_event_points: Sequence[Union[int, Sequence[int]]],
    *,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 0.0,
    tolerance: float = 20.0,
    debounce_secs: float = 60.0,
    ensemble_all: bool = False,
    verbose: bool = True,
    cache: Optional[ProbabilityCache] = None,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
    signal_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    `cache`, `fold`, `signal_ids` (and optionally `seed`) are only needed when you
    want window-level probabilities cached per CLAUDE.md's "Probability cache"
    section — e.g. for a later threshold or tolerance sweep that must not refit.
    Leave `cache=None` (the default) to run exactly as before, uncached.

    `tolerance` is `t` in the ground-truth interval `R = [f - 1, f + t)` (see
    `evaluate_recording`) — a single value, not a [before, after] pair.

    The returned DataFrame carries a `false_positives` DataFrame in `.attrs`
    (columns: model, signal_id, alarm_idx, nearest_event, offset_seconds) — one row
    per false alarm, per change request item 4. `.attrs` propagation across pandas
    operations isn't guaranteed, so read it off this return value directly rather
    than after further transformation.
    """
    if cache is not None:
        if fold is None or signal_ids is None:
            raise ValueError("cache requires both `fold` and `signal_ids` to build cache keys")
        if len(signal_ids) != len(test_signals):
            raise ValueError("signal_ids must have one entry per test signal")

    metrics_rows = []
    fp_records = []
    ensemble_maps = {i: {} for i in range(len(test_signals))}

    if verbose: print(f"\nTESTING on {len(test_signals)} recordings...")

    for name, model in trained_models.items():
        if verbose: print(f"  Evaluating {name}...", end=" ", flush=True)
        thresh = getattr(model, "threshold_", 0.5)

        total_CM = np.zeros((2, 2), dtype=int)
        delays = []
        total_signal_time = 0
        total_runtime = 0.0

        for i, (sig, event_pts) in enumerate(zip(test_signals, test_event_points)):
            total_signal_time += len(sig)

            if cache is not None:
                key = CacheKey(
                    model=name, fold=fold, window_size=window_size,
                    seed=seed, signal_id=signal_ids[i],
                )
                entry, _, runtime_us = cache.get_or_compute(
                    key, sig, model, window_size=window_size, step=step,
                    freq=freq, signal_thresh=signal_thresh,
                )
                conf_map = entry.confidence_map()
            else:
                conf_map, runtime_us = sliding_window_inference(
                    sig, model, window_size, step, freq, signal_thresh
                )

            if ensemble_all: ensemble_maps[i][name] = conf_map

            # Pass event_pts (which can be list) directly
            cm, _, delay, false_positives = evaluate_recording(
                len(sig), event_pts, conf_map, thresh,
                window_size, tolerance, freq, step, debounce_secs
            )

            sig_id = signal_ids[i] if signal_ids is not None else i
            for fp in false_positives:
                fp_records.append({
                    "model": name, "signal_id": sig_id,
                    "alarm_idx": fp.alarm_idx,
                    "nearest_event": fp.nearest_event,
                    "offset_seconds": fp.offset_seconds,
                })

            total_CM += cm
            delays.append(delay)
            total_runtime += runtime_us

        avg_runtime = total_runtime / max(1, len(test_signals))
        avg_delay = np.mean(delays) if delays else 0.0
        total_time_ms = (total_signal_time / freq) * 1000

        row = compute_metrics_from_cm(total_CM, total_time_ms, avg_runtime, avg_delay, alpha=getattr(model, "alpha", 2.0))
        row["model"] = name
        row["thresh"] = thresh
        metrics_rows.append(row)
        if verbose: print("Done.")

    if ensemble_all and len(trained_models) > 1:
        # (Ensemble logic omitted for brevity, logic identical to above loop)
        pass

    df = pd.DataFrame(metrics_rows)
    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols]
    df.attrs["false_positives"] = pd.DataFrame(
        fp_records, columns=["model", "signal_id", "alarm_idx", "nearest_event", "offset_seconds"]
    )
    return df

def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_signals: Sequence[np.ndarray],
    test_event_points: Sequence[Union[int, Sequence[int]]],
    model_specs: List[ModelSpec],
    *,
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    signal_thresh: float = 0.0,
    tolerance: float = 20.0,
    debounce_secs: float = 60.0,
    ensemble_all: bool = False,
    random_state: int = 42,
    verbose: bool = True,
    cache: Optional[ProbabilityCache] = None,
    fold: Optional[int] = None,
    signal_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:

    trained_models = train_models(X_train, y_train, model_specs, verbose=verbose, random_state=random_state)
    return evaluate_models(
        trained_models, test_signals, test_event_points,
        window_size=window_size, step=step, freq=freq,
        signal_thresh=signal_thresh, tolerance=tolerance,
        debounce_secs=debounce_secs, ensemble_all=ensemble_all, verbose=verbose,
        cache=cache, fold=fold, seed=random_state, signal_ids=signal_ids,
    )