"""
The 1.4 g activity gate — see CLAUDE.md's "The 1.4 g gate" section.

A single rule, used wherever windows are produced: a window is admissible only if its
maximum acceleration magnitude over the impact phase — samples `[freq, 2*freq)`, one
to two seconds from the window start — exceeds a threshold. Positional (a fixed offset
within the window) and computable without reference to any annotation.

Both `training_segmenter.create_training_data` (training-time negative-window
selection, and Protocol A's test-window construction, which calls the same function)
and `streaming_segmenter.generate_sliding_windows` (Protocol B inference) call this
same function for their single-channel case — do not reimplement it separately, or the
two paths can silently drift apart.
"""

from __future__ import annotations

import numpy as np

__all__ = ["impact_phase_gate"]


def impact_phase_gate(windows: np.ndarray, freq: int = 100, threshold: float = 1.4) -> np.ndarray:
    """Admissibility mask for single-channel windows.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, window_samples)
    freq : int
    threshold : float

    Returns
    -------
    np.ndarray of bool, shape (n_windows,) — True where the window's impact-phase
    max magnitude is at or above `threshold`.
    """
    z_start, z_end = int(1 * freq), int(2 * freq)
    zone = windows[:, z_start:z_end] if windows.shape[1] >= z_end else windows
    max_vals = np.max(np.abs(zone), axis=1)
    return max_vals >= threshold
