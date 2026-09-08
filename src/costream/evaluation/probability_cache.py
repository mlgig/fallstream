"""
On-disk cache of window-level classifier probabilities.

See CLAUDE.md's "Probability cache" section: any experiment that only varies a
decision threshold, a cost ratio, or a post-processing rule (debounce, tolerance)
must read probabilities from here rather than refitting a model or rerunning
sliding-window inference. Both are expensive; reconstructing a confidence map
from cached window probabilities via `compute_confidence_map` is not.

Cache key: `(model, fold, window_size, seed, signal_id)`. Each cell stores the
window-level probabilities plus enough metadata (`indices`, `valid_mask`,
`pad_size`, `ts_len`, and the segmentation params that produced them) to rebuild
`P(t)` for any threshold/debounce/tolerance choice without touching the model
again.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import joblib
import numpy as np

from ..segmentation.streaming_segmenter import (
    compute_confidence_map,
    generate_sliding_windows,
    predict_window_scores,
)

__all__ = ["CacheKey", "CachedEntry", "ProbabilityCache"]


@dataclass(frozen=True)
class CacheKey:
    model: str
    fold: int
    window_size: float
    seed: Optional[int]
    signal_id: str

    def relpath(self) -> Path:
        seed_part = "none" if self.seed is None else str(self.seed)
        # sanitize the one field that comes from free-form data (signal_id)
        safe_signal_id = str(self.signal_id).replace("/", "_")
        return Path(self.model) / f"fold{self.fold}" / f"w{self.window_size}" \
            / f"seed{seed_part}" / f"{safe_signal_id}.joblib"


@dataclass
class CachedEntry:
    """Window-level probabilities plus everything needed to rebuild P(t)."""

    # valid_mask is gamma(X_i) (the gate indicator, one per window); confidence_scores
    # is h(X_i) (the classifier's raw score, defined only where gamma=1). Kept as
    # separate fields rather than only their product p_i = gamma(X_i) * h(X_i) so a
    # gated-vs-ungated comparison is a one-line query against this entry rather than a
    # re-inference — see `gamma` / `h_full` below.
    valid_mask: np.ndarray
    confidence_scores: np.ndarray
    indices: Sequence[Tuple[int, int]]
    ts_len: int
    pad_size: int
    window_size: float
    step: float
    freq: int
    signal_thresh: float

    @property
    def gamma(self) -> np.ndarray:
        """gamma(X_i) for every window (gated-in windows are indices where this is True)."""
        return self.valid_mask

    def h_full(self, fill_value: float = 0.0) -> np.ndarray:
        """h(X_i) for every window, gated-out windows filled with `fill_value`.

        One array element per window (same length as `valid_mask`/`indices`), unlike
        `confidence_scores` which only has one entry per gated-in window.
        """
        h = np.full(len(self.valid_mask), fill_value, dtype=np.float32)
        h[self.valid_mask] = self.confidence_scores
        return h

    def confidence_map(self, method: str = "max") -> np.ndarray:
        """Rebuild the continuous P(t) signal from the cached window scores."""
        return compute_confidence_map(
            self.ts_len,
            list(self.indices),
            self.valid_mask,
            self.confidence_scores,
            method=method,
            pad_size=self.pad_size,
        )


class ProbabilityCache:
    """Joblib-backed cache, one file per `(model, fold, window_size, seed, signal_id)`.

    `root_dir` must be a configured artifacts directory (e.g. `results/prob_cache`),
    never a path inside the package/source tree.
    """

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, key: CacheKey) -> Path:
        return self.root_dir / key.relpath()

    def has(self, key: CacheKey) -> bool:
        return self._file(key).exists()

    def get(self, key: CacheKey) -> Optional[CachedEntry]:
        f = self._file(key)
        return joblib.load(f) if f.exists() else None

    def put(self, key: CacheKey, entry: CachedEntry) -> None:
        f = self._file(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(entry, f)

    def get_or_compute(
        self,
        key: CacheKey,
        ts: np.ndarray,
        model,
        *,
        window_size: float,
        step: float = 1.0,
        freq: int = 100,
        signal_thresh: float = 1.4,
        pad: bool = False,
    ) -> Tuple[CachedEntry, bool, float]:
        """Return `(entry, was_cached, runtime_us)`.

        On a cache hit, `runtime_us` is 0.0 (no inference happened). On a miss,
        the window/inference pipeline runs once, is stored, and `runtime_us`
        reports the wall-clock cost per output sample, same convention as
        `sliding_window_inference`.
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True, 0.0

        start = time.time()
        windows, indices, valid_mask, pad_size = generate_sliding_windows(
            ts, window_size=window_size, step=step, freq=freq,
            signal_thresh=signal_thresh, pad=pad,
        )
        valid_windows = windows[valid_mask] if len(windows) else windows
        confidence_scores = predict_window_scores(valid_windows, model).astype(np.float32)

        entry = CachedEntry(
            valid_mask=valid_mask,
            confidence_scores=confidence_scores,
            indices=indices,
            ts_len=len(ts),
            pad_size=pad_size,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=signal_thresh,
        )
        runtime_us = 1e6 * (time.time() - start) / max(1, len(ts))
        self.put(key, entry)
        return entry, False, runtime_us
