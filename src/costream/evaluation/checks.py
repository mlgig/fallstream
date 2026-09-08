"""
Standalone sanity checks for the cross-protocol experiment.

These are auditors, not instrumentation — they drive the real segmentation functions
and check their actual output, rather than reimplementing the logic under test.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..segmentation.training_segmenter import create_training_data
from ..segmentation.streaming_segmenter import generate_sliding_windows

__all__ = ["assert_adl_exclusion_zone", "assert_gate_applied_identically"]


def assert_adl_exclusion_zone(
    dfs: Iterable[pd.DataFrame],
    feature_col: str,
    label_col: str,
    *,
    window_size: float,
    step: float = 1.0,
    freq: int = 100,
    tolerance: float = 20.0,
    spacing="multiphase",
) -> None:
    """Assert no ADL window `create_training_data` actually retains starts in
    `[f - (w + tolerance), f + tolerance)` for any fall `f` in its own recording.

    This is a real black-box test of `create_training_data`, not a reimplementation
    of it: `create_training_data`'s (X, y) return loses each window's source
    recording/start index, so this rebuilds a same-shape/same-fall-index copy of each
    input recording whose *signal value at every sample equals that sample's own
    index* (with the gate disabled via `signal_thresh=0`, isolating the exclusion-zone
    effect from the activity gate). A kept ADL window's first sample then reveals
    exactly which start index `create_training_data` retained it from.
    """
    encoded_dfs = []
    fall_indices_per_df = []
    for df in dfs:
        n = len(df)
        labels = df[label_col].to_numpy()
        nonzero = np.flatnonzero(labels)
        if nonzero.size == 0:
            continue
        gaps = np.diff(nonzero) > 1
        event_starts = np.insert(nonzero[1:][gaps], 0, nonzero[0])
        encoded_dfs.append(pd.DataFrame({feature_col: np.arange(n, dtype=np.float32), label_col: labels}))
        fall_indices_per_df.append(event_starts)

    violations = []
    for df, falls in zip(encoded_dfs, fall_indices_per_df):
        X, y = create_training_data(
            [df], feature_cols=[feature_col], label_col=label_col,
            window_size=window_size, step=step, freq=freq,
            signal_thresh=0.0, drop_below_threshold=True, spacing=spacing,
        )
        adl_starts = X[y == 0][:, 0].astype(int)  # first sample = encoded start index
        for f in falls:
            lo, hi = f - (window_size + tolerance) * freq, f + tolerance * freq
            bad = adl_starts[(adl_starts >= lo) & (adl_starts < hi)]
            violations.extend((int(f), int(w)) for w in bad)

    assert not violations, (
        f"{len(violations)} ADL window(s) create_training_data actually retained start "
        f"inside the exclusion zone [f-(w+{tolerance}), f+{tolerance}) for their "
        f"recording's fall — (fall_index, window_start) sample: {violations[:10]}"
    )


def assert_gate_applied_identically(
    *,
    window_size: float = 5.0,
    step: float = 1.0,
    freq: int = 100,
    threshold: float = 1.4,
    ts_len: int = 20_000,
    rng_seed: int = 0,
) -> None:
    """Assert the 1.4 g gate admits the same windows via both paths that build them:
    `create_training_data` (training / Protocol A) and `generate_sliding_windows`
    (Protocol B). There is one rule (CLAUDE.md's "The 1.4 g gate"); this is not a
    reimplementation, it drives both real functions on the same synthetic all-ADL
    signal and compares admission *counts* — a genuine black-box parity check, not
    a proxy for one.
    """
    rng = np.random.default_rng(rng_seed)
    ts = rng.uniform(0.5, 2.5, size=ts_len).astype(np.float32)
    labels = np.zeros(ts_len, dtype=np.int64)  # no events: every window is a candidate

    _, _, valid_mask, _ = generate_sliding_windows(
        ts, window_size=window_size, step=step, freq=freq, signal_thresh=threshold
    )
    n_streaming_admitted = int(valid_mask.sum())

    df = pd.DataFrame({"mag": ts, "label": labels})
    X, y = create_training_data(
        [df], feature_cols=["mag"], label_col="label",
        window_size=window_size, step=step, freq=freq,
        signal_thresh=threshold, drop_below_threshold=True,
    )
    n_training_admitted = int((y == 0).sum())

    assert n_streaming_admitted == n_training_admitted, (
        f"Gate admitted {n_streaming_admitted} windows via generate_sliding_windows "
        f"but {n_training_admitted} via create_training_data for the same signal, "
        f"window_size, step, freq, and threshold — the two gate paths have diverged."
    )
