from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample

FREQ_TARGET   = 100            # Hz after resampling
DEFAULT_WIN_S = 7             # seconds
DEFAULT_STEP_S = 1             # seconds

__all__ = ["get_X_y"]

def load(clip=False):
    farseeing = pd.read_pickle(r'data/farseeing.pkl').reset_index().drop(columns=['index'])
    return farseeing

def _window_view(arr: np.ndarray, win: int, step: int) -> np.ndarray:
    """Return a (n, win) view; no data copy."""
    if len(arr) < win:
        return np.empty((0, win), dtype=arr.dtype)
    return sliding_window_view(arr, win)[::step, :]

def get_X_y(
    df,
    *,
    keep_fall: bool = False,
    segment_test: bool = True,
    multiphase: bool = False,
    window_size: int = DEFAULT_WIN_S,
    fall_dur: int = 1,
    step: int = DEFAULT_STEP_S,
    thresh: float = 1.1,
    spacing: int | str = 1,
    fall_pos: str = "fixed",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised sampling function for FARSEEING dataset."""

    win_len_target   = window_size * FREQ_TARGET
    step_len  = step * FREQ_TARGET

    if multiphase:
        pre_offsets = [1]   # exactly 1 s pre‑fall
    else:
        if spacing == "na":
            pre_offsets = [window_size // 2]
        else:
            s = int(spacing)
            if fall_pos == "random":
                rng = np.random.default_rng(5)
                n = window_size // s
                pre_offsets = rng.integers(2, window_size - 2, n).tolist()
            else:  # fixed grid
                pre_offsets = list(range(2, window_size - 2, s))

    X_rows, y_rows = [], []

    for _, row in df.iterrows():
        acc  = np.asarray(row["accel_mag"], dtype=np.float32)
        freq = int(row["freq"])
        fall = int(row["fall_point"]) if not np.isnan(row["fall_point"]) else -1

        if not segment_test and keep_fall:
            # UNSEGMENTED RETURN
            # resample whole signal once 
            # if freq != FREQ_TARGET:
            #     acc  = resample(acc, int(len(acc) * FREQ_TARGET / freq))
            #     freq = FREQ_TARGET 
            X_rows.append(acc)
            y_rows.append(fall)
            continue

        # window lengths at *native* sampling rate
        orig_win  = window_size * freq
        orig_step = step       * freq

        win_list, y_list = [], []

        # add positive windows per pre_offset 
        if fall >= 0:
            for pre in pre_offsets:
                start = max(0, fall - pre * freq)
                end   = start + orig_win
                if end <= len(acc):
                    fall_win = acc[start:end]
                    if freq != FREQ_TARGET:
                        fall_win = resample(fall_win, win_len_target)
                    win_list.append(fall_win)
                    y_list.append(np.array([1], dtype=np.uint8))
            # cut margin to avoid overlap for ADL extraction
            cut_left = max(0, fall - max(pre_offsets) * freq)
            acc = acc[:cut_left]
            fall = -1  # no second fall window


        # ADL candidate windows 
        view = _window_view(acc, orig_win, orig_step)
        if view.size:
            # resample view batch if original freq ≠ 100 Hz
            if freq != FREQ_TARGET:
                view = resample(view, win_len_target, axis=1)

            if multiphase:
                main = view[:, FREQ_TARGET : 2 * FREQ_TARGET]   # 2nd second
                mask = main.max(axis=1) >= thresh
            else:
                mask = view.max(axis=1) >= thresh

            view = view[mask]
            if view.size:
                win_list.append(view)
                y_list.append(np.zeros(len(view), dtype=np.uint8))

        if win_list:
            X_rows.append(np.vstack(win_list))
            y_rows.append(np.concatenate(y_list))

    if not segment_test and keep_fall:
        X = np.array(X_rows, dtype=object)   # ragged signals
        # print([len(y) for y in y_rows])
        y = np.array(y_rows, dtype=int)
        return X, y

    X = np.vstack(X_rows) if X_rows else np.empty((0, win_len_target), np.float32)
    y = np.concatenate(y_rows) if y_rows else np.empty((0,), np.uint8)
    return X, y