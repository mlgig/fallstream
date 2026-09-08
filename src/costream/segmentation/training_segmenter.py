"""
Segmentation logic for creating training datasets from continuous signals.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .gate import impact_phase_gate

__all__ = ["create_training_data"]


def _window_view(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """Return a sliding window view of an array."""
    arr = np.asarray(arr)
    
    if len(arr) < win:
        if arr.ndim == 1:
            return np.empty((0, win), dtype=arr.dtype)
        else:
            return np.empty((0, win, arr.shape[1]), dtype=arr.dtype)
            
    if arr.ndim == 1:
        return sliding_window_view(arr, win, axis=0)[::step]
    else:
        v = sliding_window_view(arr, win, axis=0)[::step]
        return v.transpose(0, 2, 1)


def create_training_data(
    dfs: Iterable[pd.DataFrame],
    feature_cols: Sequence[str],
    label_col: str = "label",
    *,
    freq: int = 100,
    window_size: float = 7.0,
    step: float = 1.0,
    signal_thresh: float = 0.0,
    drop_below_threshold: bool = True,
    spacing: Union[int, str] = 5,
    event_pos: str = "fixed",
    keep_unsegmented: bool = False,
    exclusion_tolerance: float = 20.0,
    use_post_event_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) pairs. Supports MULTIPLE events per file.

    `spacing` controls ONLY where positive (event) windows are placed — it has no
    effect on the ADL gate, which always checks a window's impact phase (the 2nd
    second), never the whole window; see CLAUDE.md's "1.4 g in two roles".

    `exclusion_tolerance` sets the ADL exclusion zone to
    `[event - (window_size + exclusion_tolerance), event + exclusion_tolerance)`
    (CLAUDE.md's "Fall-free" zone) — independent of the streaming scoring tolerance,
    even though both currently default to the same value; see CLAUDE.md's "Exclusion
    zone and tolerance interval are different objects".
    """

    win_len_samples = int(window_size * freq)
    step_samples = int(step * freq)
    exclusion_before_samples = int((window_size + exclusion_tolerance) * freq)
    exclusion_after_samples = int(exclusion_tolerance * freq)

    # --- Setup Positive Window Offsets ---

    if str(spacing) == "na":
        pre_offsets = [int(window_size // 2)]
    
    elif str(spacing) == "multiphase":
        # Specific mode: Window starts exactly 1 second before the event
        pre_offsets = [1]
        
    else:
        # Integer spacing logic
        s = int(spacing)
        margin = 2 if window_size >= 4.0 else 0
        start_range = margin
        end_range = int(window_size - margin)

        if event_pos == "random":
            rng = np.random.default_rng(42)
            low = max(start_range, 0)
            high = max(end_range, low + 1)
            n_offsets = max(1, int(window_size // s))
            pre_offsets = rng.integers(low, high, n_offsets).tolist()
        else:
            if end_range <= start_range:
                pre_offsets = [int(window_size // 2)]
            else:
                pre_offsets = list(range(start_range, end_range, s)) or [int(window_size // 2)]

    X_rows, y_rows = [], []

    for df in dfs:
        # Extract signal
        signal = df[feature_cols].to_numpy(dtype=np.float32)
        if signal.shape[1] == 1:
            signal = signal.flatten()

        labels = df[label_col].to_numpy(dtype=int)
        
        # --- Find All Distinct Events (0 -> Non-Zero Transitions) ---
        # Padding ensures we catch an event if it starts at index 0
        padded_labels = np.pad(labels, (1, 0), mode='constant', constant_values=0)
        event_indices = np.where((padded_labels[:-1] == 0) & (padded_labels[1:] != 0))[0]

        # -- Unsegmented Mode --
        if keep_unsegmented:
            X_rows.append(signal)
            if len(event_indices) == 0:
                y_rows.append(-1)
            else:
                # if multiple events exist, return list, else single
                y_rows.append(event_indices if len(event_indices) > 1 else event_indices[0])
            continue

        win_list = []
        y_list = []

        # Create mask for "dirty" regions (Event + Pre-offset coverage)
        event_mask = np.zeros(len(signal), dtype=bool)

        # --- Positive Windows (Iterate over ALL events) ---
        for i, event_index in enumerate(event_indices):

            # Exclusion zone (CLAUDE.md "Fall-free"): [event - (w + tol), event + tol)
            safe_start = max(0, event_index - exclusion_before_samples)

            if not use_post_event_data:
                # Mask EVERYTHING from the start of the exclusion zone to the end,
                # effectively discarding post-fall ADL
                event_mask[safe_start:] = True

                # If we are discarding post-data, we only process the FIRST event
                # (because subsequent events are in the discarded zone)
                if i > 0:
                    break
            else:
                safe_end = min(len(signal), event_index + exclusion_after_samples)
                event_mask[safe_start:safe_end] = True

            # Extract Positive Windows
            # (We always extract positives for the current event, unless skipped above)
            for pre in pre_offsets:
                offset_samples = int(pre * freq)
                start = max(0, event_index - offset_samples)
                end = start + win_len_samples
                
                if end <= len(signal):
                    event_win = signal[start:end]
                    win_list.append(event_win)
                    y_list.append(1)
            
            # If discarding post-data, we stop finding events after the first one
            if not use_post_event_data:
                break

        # --- Negative Windows ---
        if len(signal) >= win_len_samples:
            
            # Generate ALL sliding windows
            all_windows = _window_view(signal, win_len_samples, step_samples)
            
            # Generate corresponding mask windows
            mask_windows = _window_view(event_mask, win_len_samples, step_samples)
            
            # Filter: Keep windows where mask is entirely False (Pure ADL)
            is_pure_adl = ~np.any(mask_windows, axis=1)
            view = all_windows[is_pure_adl]
            
            if view.size > 0:
                # --- Filtering Logic: the 1.4 g gate, see CLAUDE.md ---
                # Single-channel, threshold-admits case is the one actually used
                # (FARSEEING is single-channel magnitude) — shared with
                # generate_sliding_windows's gate so training and inference apply
                # literally the same check. Do not reimplement it separately.
                if view.ndim == 2 and drop_below_threshold:
                    mask = impact_phase_gate(view, freq=freq, threshold=signal_thresh)
                else:
                    # Multi-channel, or the inverse (`drop_below_threshold=False`)
                    # selection — not exercised by FARSEEING; kept as its own logic.
                    target_view = view
                    if view.shape[1] >= 2 * freq:
                        if view.ndim == 2:
                            target_view = view[:, freq : 2 * freq]
                        else:
                            target_view = view[:, freq : 2 * freq, :]

                    if target_view.ndim == 2:
                        max_vals = np.max(np.abs(target_view), axis=1)
                    else:
                        max_vals = np.max(np.abs(target_view), axis=(1, 2))

                    if drop_below_threshold:
                        mask = max_vals >= signal_thresh
                    else:
                        mask = max_vals < signal_thresh

                clean_negatives = view[mask]
                
                if clean_negatives.shape[0] > 0:
                    win_list.append(clean_negatives)
                    y_list.extend([0] * clean_negatives.shape[0])

        # -- Aggregate --
        if win_list:
            batch_list = []
            for w in win_list:
                if w.ndim == signal.ndim: 
                    batch_list.append(np.expand_dims(w, axis=0))
                else:
                    batch_list.append(w)
            
            X_rows.append(np.vstack(batch_list))
            y_rows.append(np.array(y_list, dtype=np.uint8))

    # -- Final Return --
    if keep_unsegmented:
        return np.array(X_rows, dtype=object), np.array(y_rows, dtype=object)

    if not X_rows:
        shape = (0, win_len_samples) if len(feature_cols) == 1 else (0, win_len_samples, len(feature_cols))
        return np.empty(shape, np.float32), np.empty((0,), np.uint8)

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    return X, y