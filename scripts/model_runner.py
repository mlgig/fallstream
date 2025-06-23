from __future__ import annotations
from typing import List, Sequence
import logging
import timeit
import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

from scripts.utils import (
    sliding_window_confidence,
    detect
)

from scripts.plotting import plot_detection
from scripts.metric import compute_row
from scripts.model_spec import ModelSpec
from scripts.costs import cost_fn

logger = logging.getLogger(__name__)
    
def _train_models(
    X: np.ndarray,
    y: np.ndarray,
    model_specs: List[ModelSpec],
    *,
    tune_threshold: bool = False,
    random_state: int = 0,
    **kw
) -> tuple[dict[str, BaseEstimator], dict[str, float]]:
    """Fit exactly the models in *model_specs* and optionally tune thresholds."""
    trained: dict[str, BaseEstimator] = {}
    thresholds: dict[str, float] = {}

    cost_score = make_scorer(cost_fn)

    print("TRAINING", end=" ")
    for spec in model_specs:
        est = spec.clone(seed=random_state)
        # # ensure tabular models get scaling + imputation if not already present
        # if spec.kind == "tabular" and not hasattr(est, "steps"):
        #     est = make_pipeline(
        #         StandardScaler(),
        #         SimpleImputer(strategy="mean"),
        #         est
        #     )
        est.fit(X, y)
        trained[spec.name] = est
        print(f"{spec.name}", end=". ")

        if tune_threshold:
            tuned = TunedThresholdClassifierCV(est, cv=5, scoring=cost_score).fit(X, y)
            thresholds[spec.name] = tuned.best_threshold_
            print(f"thresh={tuned.best_threshold_:.2f}", end=" ")
    print("✅")
    return trained, thresholds

def run_models(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    model_specs: List[ModelSpec],
    model_seeds: Sequence[int] = (0,),
    aggregate_seeds: bool = True,
    saved_models: str | None = None,
    save_path: str | None = None,
    freq: int = 100,
    ensemble_models: bool = True,
    ensemble_by_kind: bool = True,
    verbose: bool = True,
    **kw,
) -> pd.DataFrame:
    
    """Evaluate each model template on the provided split.

    *No seed‑level soft vote*: metrics are averaged (mean) when
    ``aggregate_seeds=True``.
    """
    # Load or train models
    if saved_models is not None:
        cache = joblib.load(saved_models)
        trained_models: dict[str, BaseEstimator] = cache["models"]
        thresholds: dict[str, float] = cache.get("thresholds", {})
        if verbose:
            print(f"Loaded cached models from {saved_models}")
        wanted = {s.name for s in model_specs}
        trained_models = {n: m for n, m in trained_models.items() if n in wanted}
        thresholds     = {n: t for n, t in thresholds.items()     if n in wanted}
        model_seeds = (None,)
        aggregate_seeds = False
    else:
        trained_models, thresholds = _train_models(X_train, y_train, model_specs, **kw)
        if save_path is not None:
            joblib.dump({"models": trained_models, "thresholds": thresholds}, save_path)
            if verbose:
                print(f"Saved models to {save_path}")

    # Evaluate models
    metrics_rows: list[dict] = []
    # (kind, Series) pairs for later kind‑level ensemble
    conf_by_model: list[tuple[str, pd.Series]] = []

    spec_lookup = {s.name: s for s in model_specs}

    w = kw.get('window_size')
    print(f"TESTING", end=" ")
    for name, base_model in trained_models.items():
        spec = spec_lookup.get(name)
        if spec is None:
            logger.warning("Model %s present in cache but not in spec list", name)
            continue
        if verbose:
            print(f"{name}", end=". ")

        seed_rows: list[dict] = []
        conf_series_list: list[pd.Series] = []

        for seed in model_seeds:
            model = base_model if seed is None else spec.clone(seed)
            if seed is not None:
                model.fit(X_train, y_train)

            CM = np.zeros((2, 2), dtype=int)
            delays: list[float] = []
            tot_time = 0.0
            signal_time = 0
            conf_map: dict[int, float] = {}

            for i, (ts, y) in enumerate(zip(X_test, y_test)):
                if len(ts) < 100000 or (120001 < len(ts) < 300000):
                    continue
                signal_time += len(ts)
                c, ave_t = sliding_window_confidence(ts, y, model, **kw)
                conf_map[i] = c
                thresh = thresholds.get(name, 0.5)
                cm, hit, delay = detect(ts, y, c, confidence_thresh=thresh, **kw)
                CM += cm
                delays.append(delay)
                tot_time += ave_t
                if kw.get("plot", False):
                    plot_detection(ts, y, c, cm, hit, ave_t, model_name=name, **kw)

            ave_t = tot_time / max(1, len(conf_map))
            row = compute_row(CM, signal_time, ave_t, np.mean(delays))
            row.update(model=name, window_size=w, seed=seed or 0)
            seed_rows.append(row)
            conf_series_list.append(pd.Series(conf_map))

        metrics_rows.extend(seed_rows)
        conf_by_model.append((spec.kind, conf_series_list[0]))

    # model‑level ensembles 
    if ensemble_models and len(conf_by_model) > 1:
        def _ensemble(label: str, series_list: List[pd.Series]):
            CM = np.zeros((2, 2), dtype=int)
            delays, signal_time = [], 0
            df_conf = pd.concat(series_list, axis=1)
            t0 = timeit.default_timer()
            for i, (ts, y) in enumerate(zip(X_test, y_test)):
                if len(ts) < 100000 or (120001 < len(ts) < 300000):
                    continue
                signal_time += len(ts)
                c = df_conf.loc[i].mean()
                cm, hit, delay = detect(ts, y, c, **kw)
                CM += cm
                delays.append(delay)
            ave_t = (timeit.default_timer() - t0) / max(1, df_conf.shape[1])
            row = compute_row(CM, signal_time, ave_t, np.mean(delays))
            row.update(model=label, window_size=w)
            metrics_rows.append(row)

        # ensemble across all models
        _ensemble("Ensemble-All", [s for _, s in conf_by_model])

        if ensemble_by_kind:
            # collect series by their kind tag
            buckets: dict[str, List[pd.Series]] = {}
            for kind, series in conf_by_model:
                buckets.setdefault(kind, []).append(series)

            # only build an ensemble if we have ≥2 models of that kind
            for kind, bucket in buckets.items():
                if len(bucket) <= 1:
                    continue
                _ensemble(f"Ensemble-{kind}", bucket)
    print("✅")
    metrics_df = pd.DataFrame(metrics_rows)
    # reorder columns to have 'model' first
    meta   = ["model", "seed", "window_size"]
    metrics_df = metrics_df[meta + [c for c in metrics_df.columns if c not in meta]]
    return metrics_df
