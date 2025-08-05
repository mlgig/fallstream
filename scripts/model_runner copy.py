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
from sklearn.metrics import make_scorer, confusion_matrix

from sklearn.model_selection import TunedThresholdClassifierCV

from scripts.utils import (
    sliding_window_confidence,
    detect
)

from scripts.plotting import plot_detection
from scripts.metric import compute_row
from scripts.model_spec import ModelSpec
from scripts.models import _set_seed
from scripts.costs import cost_fn

logger = logging.getLogger(__name__)
    
def _train_models(
    X: np.ndarray,
    y: np.ndarray,
    model_specs: List[ModelSpec],
    *,
    # tune_threshold: bool = False,
    random_state: int = 0,
    **kw
) -> tuple[dict[str, BaseEstimator], dict[str, float]]:
    """Fit exactly the models in *model_specs* and optionally tune thresholds."""
    trained: dict[str, BaseEstimator] = {}
    thresholds: dict[str, float] = {}

    cost_score = make_scorer(cost_fn)

    print(f"TRAINING (seed={random_state})", end=" ")
    for spec in model_specs:
        est = spec.clone()
        _set_seed(est, random_state)
        if not "Dummy" in spec.name:
            print(spec.name, end="")
            
        est.fit(X, y)
        # set estimator type for clf, last step in pipeline
        if hasattr(est, "steps") and est.steps:
            est.steps[-1][1]._estimator_type = "classifier"
        trained[spec.name] = est
        if not "Dummy" in spec.name:
            print(". ", end="")

        # skip if spec.kind is "baseline"
        if spec.kind == "baseline":
            continue
        
        tune = kw.get('tune_threshold', False)
        if tune:
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
        thresholds = {n: t for n, t in thresholds.items()     if n in wanted}
        model_seeds = (None,)
        aggregate_seeds = False
    
    else:
        # tune once on the first seed, reuse for the rest
        seeds = sorted(list(model_seeds)) if model_seeds else [0]
        seed0 = seeds[0]

        tm0, th0 = _train_models(
            X_train,
            y_train,
            model_specs,
            random_state=seed0,
            **kw
        )

        # Keep a seed-agnostic copy to reuse later
        base_thresholds = {name: th0.get(name, 0.5) for name in tm0.keys()}

        trained_models: dict[str, BaseEstimator] = {}
        thresholds: dict[str, float] = {}

        # Name the seed0 models with suffix _s{seed0}; keep dummies unsuffixed
        for name, est in tm0.items():
            if "Dummy" in name:
                trained_models[name] = est
                thresholds[name] = base_thresholds.get(name, 0.5)
            else:
                key = f"{name}_s{seed0}"
                trained_models[key] = est
                thresholds[key] = base_thresholds.get(name, 0.5)

        # For the remaining seeds: fit (no tuning) and reuse seed0 thresholds
        for seed in seeds[1:]:
            kw["tune_threshold"] = False
            tm, _ = _train_models(
                X_train,
                y_train,
                model_specs,
                random_state=seed,
                **kw
            )
            for name, est in tm.items():
                if "Dummy" in name:
                    # keep dummy keys seedless
                    trained_models[name] = est
                    thresholds[name] = base_thresholds.get(name, 0.5)
                else:
                    key = f"{name}_s{seed}"
                    trained_models[key] = est
                    thresholds[key] = base_thresholds.get(name, 0.5)

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

    for key, model in trained_models.items():
        if "Dummy" in key:
            name, seed = key, 0
        else:
            name, _, seed = key.partition("_s")
            seed = int(seed)
        thresh = thresholds.get(key, 0.5)
        spec = spec_lookup.get(name)
        if spec is None:
            logger.warning("Model %s present in cache but not in spec list", name)
            continue
        if verbose:
            print(f"{name}_{seed}", end=". ")

        seed_rows: list[dict] = []
        conf_series_list: list[pd.Series] = []

        CM = np.zeros((2, 2), dtype=int)
        delays: list[float] = []
        tot_time = 0.0
        signal_time = 0
        conf_map: dict[int, float] = {}

        # if spec.kind == "baseline":
        #     # constant prediction: 0 for ADL dummy, 1 for FALL dummy
        #     const_label = 1 if "Fall" in name else 0
        #     y_pred = np.full_like(y_test, const_label, dtype=int)
        #     print(y_test)

        #     CM = confusion_matrix(int(y_test>0), y_pred)
        #     ave_t        = 0.0                             
        #     delays       = [0.0]                           
        #     signal_time  = np.sum([len(ts) for ts in X_test])

        #     row = compute_row(CM, signal_time, ave_t, np.mean(delays))
        #     row.update(model=name, window_size=w, seed=seed, thresh=0.5)
        #     metrics_rows.append(row)    
        #     continue

        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 100000 or (120001 < len(ts) < 300000):
                continue
            signal_time += len(ts)
            const_label = False
            if spec.kind == "baseline":
                # constant prediction: 0 for ADL dummy, 1 for FALL dummy
                const_label = 1 if "Fall" in name else 0
            c, ave_t = sliding_window_confidence(
                ts, y, model, const_confidence=const_label, **kw)
            conf_map[i] = c
            cm, hit, delay = detect(ts, y, c, confidence_thresh=thresh,
                                    const_confidence=const_label, **kw)
            CM += cm
            delays.append(delay)
            tot_time += ave_t
            if kw.get("plot", False):
                plot_detection(ts, y, c, cm, hit, ave_t, model_name=name, **kw)

        ave_t = tot_time / max(1, len(conf_map))
        row = compute_row(CM, signal_time, ave_t, np.mean(delays))
        row.update(model=name, window_size=w, seed=seed, thresh=thresh)
        seed_rows.append(row)
        conf_series_list.append(pd.Series(conf_map))

        metrics_rows.extend(seed_rows)
        if spec.kind != "baseline":
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
