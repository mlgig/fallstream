"""
Event detection logic for streaming evaluation (Multi-Event Support).
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple, Union, Sequence

import numpy as np


class FalsePositive(NamedTuple):
    """One false-alarm record for the FP-timestamp analysis (change request item 4).

    `offset_seconds` is signed and measured from the alarm's window start to the
    nearest annotated impact *in the same recording*: negative means the alarm fired
    before that impact (the approach), positive means after (the recovery window).
    `nearest_event` is None when the recording has no annotated events at all.
    """
    alarm_idx: int
    nearest_event: Optional[int]
    offset_seconds: Optional[float]

__all__ = ["evaluate_recording", "get_high_confidence_regions", "iou", "FalsePositive"]


def iou(range_a: range, range_b: range) -> float:
    """Compute Intersection over Union (IoU) of two ranges."""
    set_a = set(range_a)
    set_b = set(range_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def get_high_confidence_regions(
    confidence_signal: np.ndarray,
    threshold: float = 0.5,
    min_interval_secs: float = 60.0,
    freq: int = 100
) -> Optional[np.ndarray]:
    """Find start indices of alarms with debouncing."""
    high_conf_indices = np.where(confidence_signal >= threshold)[0]
    
    if len(high_conf_indices) == 0:
        return None

    min_interval_samples = int(min_interval_secs * freq)
    distinct_detections = [high_conf_indices[0]]
    
    for idx in high_conf_indices[1:]:
        if idx - distinct_detections[-1] >= min_interval_samples:
            distinct_detections.append(idx)
            
    return np.array(distinct_detections)


def evaluate_recording(
    ts_len: int,
    event_points: Union[int, Sequence[int]],
    confidence_signal: np.ndarray,
    confidence_thresh: float = 0.5,
    window_size: float = 7.0,
    tolerance: float = 20.0,
    freq: int = 100,
    step: float = 1.0,
    debounce_secs: float = 60.0
) -> Tuple[np.ndarray, Optional[np.ndarray], float, List[FalsePositive]]:
    """
    Evaluate a recording against ONE OR MORE ground truth events.

    Ground-truth interval `R = [f - 1, f + t)` with tolerance `t = tolerance` seconds —
    independent of `window_size`. The pre-impact extent is fixed at the one-second
    falling phase (matching the training positive's start at `f - 1`), not widened by
    the window length. A detection window for an alarm at sample `h` is
    `d = [h, h + w)` — the fired window itself, not further widened by tolerance. See
    CLAUDE.md's Protocol B "Scoring" step; do not re-derive this independently.

    Parameters
    ----------
    event_points : int or List[int]
        Indices of the events. Use -1 or empty list if no events.
    tolerance : float
        Seconds of post-impact tolerance, `t` in `R = [f - 1, f + t)`. Single value —
        the pre-impact side is fixed at 1 second, not configurable here.

    Returns
    -------
    cm : np.ndarray
        [[TN, FP], [FN, TP]]
    high_conf : np.ndarray
        Detected alarms.
    delay : float
        Average delay of True Positives (0 if none), measured from window start.
    false_positives : list[FalsePositive]
        One record per false alarm — see `FalsePositive`.
    """

    # Normalize input to list
    if isinstance(event_points, int):
        ground_truth_events = [] if event_points == -1 else [event_points]
    else:
        # Filter out -1s if mixed in list
        ground_truth_events = [e for e in event_points if e != -1]

    # 1. Get Alarms
    high_conf = get_high_confidence_regions(
        confidence_signal,
        threshold=confidence_thresh,
        min_interval_secs=debounce_secs,
        freq=freq
    )

    # 2. Define Decision Blocks (for TN estimation)
    step_samples = int(step * freq)
    window_samples = int(window_size * freq)
    n_decision_blocks = max(1, (ts_len - window_samples) // step_samples)

    # 3. Setup Ground Truth Ranges: R = [f - 1, f + t), independent of window_size.
    tol_samples = int(tolerance * freq)
    gt_ranges = []
    for ep in ground_truth_events:
        left_bound = ep - freq
        right_bound = ep + tol_samples
        gt_ranges.append(range(int(left_bound), int(right_bound)))

    # 4. Matching Logic
    matched_gt_indices = set()
    total_delays = []
    false_positives: List[FalsePositive] = []

    if high_conf is not None:
        for alarm_idx in high_conf:
            # d = [h, h + w) — the fired window itself, no tolerance widening.
            detection_range = range(alarm_idx, alarm_idx + window_samples)

            # Check if this alarm matches ANY ground truth event
            hit_any = False
            for i, gt_range in enumerate(gt_ranges):
                if iou(detection_range, gt_range) > 0:
                    hit_any = True
                    matched_gt_indices.add(i)

                    # Calculate delay (Alarm - Event), measured from window start.
                    ep = ground_truth_events[i]
                    d = (alarm_idx - ep) / freq
                    lo, hi = -(window_size + 1), tolerance
                    assert lo < d < hi, (
                        f"TP delay {d:.3f}s outside expected ({lo:.3f}, {hi:.3f}) for "
                        f"alarm_idx={alarm_idx}, event={ep}, window_size={window_size}, "
                        f"tolerance={tolerance} — this indicates a scoring bug, not a "
                        f"value to clip."
                    )
                    total_delays.append(d)

            if not hit_any:
                false_positives.append(_nearest_event_offset(alarm_idx, ground_truth_events, freq))

    tp_count = len(matched_gt_indices)
    fp_count = len(false_positives)
    # FN = Total Events that were NEVER matched
    fn_count = len(ground_truth_events) - tp_count

    # TN = Remainder
    tn_count = max(0, n_decision_blocks - tp_count - fp_count - fn_count)

    cm = np.array([[tn_count, fp_count], [fn_count, tp_count]])
    avg_delay = float(np.mean(total_delays)) if total_delays else 0.0

    return cm, high_conf, avg_delay, false_positives


def _nearest_event_offset(alarm_idx: int, ground_truth_events: Sequence[int], freq: int) -> FalsePositive:
    """Signed offset (seconds) from `alarm_idx` to the nearest event, or None if none exist."""
    if not ground_truth_events:
        return FalsePositive(alarm_idx=alarm_idx, nearest_event=None, offset_seconds=None)
    nearest = min(ground_truth_events, key=lambda ep: abs(alarm_idx - ep))
    return FalsePositive(
        alarm_idx=alarm_idx,
        nearest_event=nearest,
        offset_seconds=(alarm_idx - nearest) / freq,
    )