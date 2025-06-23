from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Sequence
import logging
import timeit
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

logger = logging.getLogger(__name__)

SplitFn = Callable[[pd.DataFrame, "Dataset", Sequence[int]],
                   tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
RunModelsFn = Callable[..., pd.DataFrame]


@dataclass(slots=True)
class CVRunner:
    """Subject‑wise cross‑validation orchestrator.

    Parameters
    ----------
    df : pd.DataFrame
        The full table containing all samples.
    dataset : Dataset
        Dataset helper object (gives sampling rate, etc.).
    groups : str | Sequence[int]
        Subject identifiers; either a column name in *df* or a pre‑extracted
        1‑D array whose length equals ``len(df)``.
    run_models_fn : callable
        Signature should match ``run_models(X_tr, X_te, y_tr, y_te, **kw) → DataFrame``.
        Must return **one row per model**; may already aggregate model seeds.
    split_df_fn : callable
        Helper that maps *(df, dataset, test_subjects, **kw)* → ``X_tr, X_te, y_tr, y_te``.

    window_size : int, default=7
        Fixed window length in seconds.
    cv : int, default=5
        Number of outer folds (GroupKFold).
    random_state : int, default=0
    model_seeds : Sequence[int] | None, default=None
        If provided, the same list is forwarded to ``run_models_fn`` so each
        estimator can be trained with multiple random initialisations.
        If *None*, a single model instance is trained per fold.
    aggregate_seeds : bool, default=True
        Set *True* if ``run_models_fn`` already averages the per‑seed metrics
        **within the fold**. Set *False* if it returns one row per seed.
    """

    df: pd.DataFrame
    dataset: "Dataset"
    groups: str | Sequence[int]

    run_models_fn: RunModelsFn
    split_df_fn: SplitFn

    window_size: int = 7
    cv: int = 5
    random_state: int = 0
    model_seeds: Sequence[int] | None = None
    aggregate_seeds: bool = True
    kwargs: Mapping[str, object] = field(default_factory=dict)


    def _subject_ids(self) -> np.ndarray:
        if isinstance(self.groups, str):
            return self.df[self.groups].values
        return np.asarray(self.groups)

    def _splitter(self):
        return GroupKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

    def run(self, **run_kwargs) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        subj_ids = self._subject_ids()
        splitter = self._splitter()

        # fall‑back to single seed if user passed None
        seeds = self.model_seeds or (0,)

        for fold_idx, (_, test_idx) in enumerate(splitter.split(subj_ids, groups=subj_ids), start=1):
            test_subjects = np.unique(subj_ids[test_idx])
            if run_kwargs['verbose']:
                print(f"\n– Fold {fold_idx}/{self.cv}: testing on {len(test_subjects)} subjects –")

            X_tr, X_te, y_tr, y_te = self.split_df_fn(
                self.df,
                self.dataset,
                test_subjects,
                random_state=self.random_state,
                **self.kwargs
            )

            fold_df = self.run_models_fn(
                X_tr,
                X_te,
                y_tr,
                y_te,
                freq=getattr(self.dataset, "freq", 100),
                model_seeds=seeds,
                **self.kwargs,
                **run_kwargs,
            )

            # guarantee fold column exists
            fold_df = fold_df.assign(fold=fold_idx)
            rows.append(fold_df)

        metrics_df = pd.concat(rows, ignore_index=True)
        return metrics_df


def train_test_subject_split(
    subject_ids: Iterable[int],
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupShuffleSplit wrapper (unchanged)."""
    subject_ids = np.asarray(list(subject_ids))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(subject_ids, groups=subject_ids))
    return subject_ids[train_idx], subject_ids[test_idx]