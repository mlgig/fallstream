"""
Subject-wise Cross-Validation utilities.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Sequence, Dict, Optional, Union
from sklearn.model_selection import GroupKFold

# Internal package imports
from ..segmentation.training_segmenter import create_training_data
from ..data.utils import extract_streaming_data
from .tester import ModelSpec, train_models, evaluate_models
from .presegmented import evaluate_presegmented
from .probability_cache import ProbabilityCache

__all__ = ["run_subject_cv", "aggregate_cv_results"]


def run_subject_cv(
    subject_map: Dict[str, List[pd.DataFrame]],
    model_specs: List[ModelSpec],
    feature_cols: Sequence[str],
    label_col: str = "label",
    cv: int = 5,
    random_state: int = 42,
    # Segmentation Params
    window_size: float = 7.0,
    step: float = 1.0,
    freq: int = 100,
    activity_threshold: float = 1.4,
    drop_below_threshold: bool = True,
    spacing: Union[int, str] = 5,
    exclusion_tolerance: float = 20.0,
    use_post_event_data: bool = True,
    # Streaming/Eval Params
    tolerance: float = 20.0,
    debounce_secs: float = 60.0,
    verbose: bool = True,
    # Probability cache (see CLAUDE.md's "Probability cache" section). None (the
    # default) disables caching and matches prior behaviour exactly; pass a path
    # to a configured artifacts directory (e.g. "results/prob_cache") to enable it.
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Orchestrate a Subject-Wise Cross-Validation experiment.

    1. Splits subjects into K folds.
    2. For each fold:
       - Segments the training subjects into (X_train, y_train) and fits every
         model spec on it ONCE.
       - Evaluates those same frozen model instances under BOTH protocols:
         Protocol A (presegmented, `evaluate_presegmented` — builds the test
         fold's window set via the identical `create_training_data` call used
         for training) and Protocol B (streaming, `evaluate_models`). This is
         what CLAUDE.md's "Frozen models across protocols" invariant requires:
         one fit per (model, fold, window_size), both evaluations against it.
    3. Aggregates results into a single DataFrame with `fold` and `protocol`
       columns ("A" or "B").

    Parameters
    ----------
    subject_map : dict
        Mapping of {subject_id: [list of dataframes]}.
        (Usually output of costream.data.loader.load_subject with grouping).
    model_specs : list[ModelSpec]
        Models to evaluate.
    feature_cols : list[str]
        Columns to use as features.
    label_col : str
        Name of label column.
    cv : int
        Number of folds.

    Returns
    -------
    pd.DataFrame
        Combined results suitable for statistical analysis.
    """

    subjects = np.array(list(subject_map.keys()))

    # Handle case where fewer subjects than folds
    if len(subjects) < cv:
        raise ValueError(
            f"Cannot perform {cv}-fold CV with only {len(subjects)} subjects."
        )

    gkf = GroupKFold(n_splits=cv)

    # We use subjects as both X and groups for the splitter
    # (The actual data isn't split here, just the IDs)
    split_gen = gkf.split(subjects, groups=subjects)

    cache = ProbabilityCache(cache_dir) if cache_dir is not None else None

    all_results = []
    all_false_positives = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_gen, start=1):
        train_subjs = subjects[train_idx]
        test_subjs = subjects[test_idx]

        if verbose:
            print(
                f"\n=== Fold {fold_idx}/{cv} | Train Subjects: {len(train_subjs)} | Test Subjects: {len(test_subjs)} ==="
            )

        # 1. Prepare Training Data (Segmented)
        # Collect all DFs for training subjects
        train_dfs = []
        for s in train_subjs:
            train_dfs.extend(subject_map[s])

        X_train, y_train = create_training_data(
            train_dfs,
            feature_cols=feature_cols,
            label_col=label_col,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=activity_threshold,
            drop_below_threshold=drop_below_threshold,
            spacing=spacing,
            exclusion_tolerance=exclusion_tolerance,
            use_post_event_data=use_post_event_data,
        )

        if verbose:
            print(f"  Segmented Train Data: {X_train.shape}")

        # Fit every model spec ONCE on this fold's training data. Both protocols
        # below evaluate these exact same fitted instances.
        trained_models = train_models(
            X_train, y_train, model_specs, verbose=verbose, random_state=random_state
        )

        # 2a. Protocol B — streaming, over the test subjects' continuous signals
        test_signals, test_events = extract_streaming_data(
            subject_map=subject_map,
            subjects=test_subjs,
            feature_col=feature_cols[0],
            label_col=label_col,
        )

        # Signal ids for the probability cache, matching extract_streaming_data's
        # own iteration order (subject, then that subject's recordings in order) so
        # they line up 1:1 with test_signals/test_events without changing that
        # function's public contract.
        signal_ids = [
            f"{s}_{i}"
            for s in test_subjs if s in subject_map
            for i in range(len(subject_map[s]))
        ]

        streaming_results = evaluate_models(
            trained_models,
            test_signals,
            test_events,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=activity_threshold,
            tolerance=tolerance,
            debounce_secs=debounce_secs,
            verbose=False,
            cache=cache,
            fold=fold_idx,
            seed=random_state,
            signal_ids=signal_ids,
        )
        streaming_results["protocol"] = "B"

        # pd.concat below isn't guaranteed to preserve .attrs, so pull this fold's
        # false-positive records out now and accumulate them separately (change
        # request item 4).
        fold_fps = streaming_results.attrs.get("false_positives")
        if fold_fps is not None and len(fold_fps):
            fold_fps = fold_fps.copy()
            fold_fps["fold"] = fold_idx
            all_false_positives.append(fold_fps)

        # 2b. Protocol A — presegmented, over the SAME test subjects' dataframes,
        # windowed by the SAME create_training_data call used for training.
        test_dfs = [df for s in test_subjs if s in subject_map for df in subject_map[s]]
        presegmented_results = evaluate_presegmented(
            trained_models,
            test_dfs,
            feature_cols=feature_cols,
            label_col=label_col,
            window_size=window_size,
            step=step,
            freq=freq,
            signal_thresh=activity_threshold,
            drop_below_threshold=drop_below_threshold,
            spacing=spacing,
            exclusion_tolerance=exclusion_tolerance,
            use_post_event_data=use_post_event_data,
            verbose=verbose,
        )
        presegmented_results["protocol"] = "A"

        # 3. Combine both protocols for this fold
        fold_results = pd.concat([streaming_results, presegmented_results], ignore_index=True)
        fold_results["fold"] = fold_idx
        all_results.append(fold_results)

        # if verbose:
        #     # Print quick summary of this fold
        #     print("  Fold Results (Mean F1):")
        #     print(fold_results.groupby("model")["f1-score"].mean())

    # 4. Aggregate
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.attrs["false_positives"] = (
        pd.concat(all_false_positives, ignore_index=True) if all_false_positives
        else pd.DataFrame(columns=["model", "signal_id", "alarm_idx", "nearest_event", "offset_seconds", "fold"])
    )
    return final_df

def aggregate_cv_results(df: pd.DataFrame, group: Union[str, List[str], None] = None,
                         exclude_cols: list = []) -> pd.DataFrame:
    """Return mean ± std for each metric.

    Defaults to grouping by `["model", "protocol"]` when a `protocol` column is
    present. Protocol A and Protocol B metrics share column names (e.g.
    `f1-score`) but are not commensurable (per-window vs. per-event) — see
    CLAUDE.md's "Comparing them" section — so the default must never average
    them together. Pass `group` explicitly to override.
    """
    if group is None:
        group = ["model", "protocol"] if "protocol" in df.columns else "model"
    if exclude_cols==[]:
        exclude_cols = ["seed", "fold"]
    else:
        exclude_cols = exclude_cols + ["seed", "fold"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols]
    means = df.groupby(group)[num_cols].mean()
    stds  = df.groupby(group)[num_cols].std()
    aggr  = pd.DataFrame(index=means.index)
    for col in num_cols:
        aggr[col] = means[col].round(2).astype(str) + " ± " + stds[col].round(2).astype(str)
    aggr.reset_index(inplace=True)
    return aggr