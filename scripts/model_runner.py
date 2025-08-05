from __future__ import annotations
from typing import List, Sequence
import logging
import timeit
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV

from tqdm import tqdm

from scripts.utils import sliding_window_confidence, detect
from scripts.plotting import plot_detection
from scripts.metric import compute_row
from scripts.model_spec import ModelSpec
from scripts.models import _set_seed
from scripts.costs import cost_fn

# Suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fall_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _train_models(X, y, model_specs: List[ModelSpec], *, random_state: int = 0, **kw):
    trained = {}
    thresholds = {}
    cost_score = make_scorer(cost_fn)

    logger.info(f"Training models (seed={random_state})")
    for spec in tqdm(model_specs, desc=f"Training (seed={random_state})"):
        est = spec.clone()
        _set_seed(est, random_state)
        est.fit(X, y)

        if hasattr(est, "steps") and est.steps:
            est.steps[-1][1]._estimator_type = "classifier"
        trained[spec.name] = est

        if spec.kind == "baseline":
            continue

        if kw.get("tune_threshold", False):
            tuned = TunedThresholdClassifierCV(est, cv=5, scoring=cost_score).fit(X, y)
            thresholds[spec.name] = tuned.best_threshold_
            logger.info(f"{spec.name}: threshold tuned to {tuned.best_threshold_:.2f}")
    return trained, thresholds

def run_models(X_train, X_test, y_train, y_test, *, model_specs: List[ModelSpec], model_seeds=(0,), aggregate_seeds=True, saved_models=None, save_path=None, freq=100, ensemble_models=True, ensemble_by_kind=True, verbose=True, **kw):
    if saved_models:
        cache = joblib.load(saved_models)
        trained_models = cache["models"]
        thresholds = cache.get("thresholds", {})
        logger.info(f"Loaded cached models from {saved_models}")
        wanted = {s.name for s in model_specs}
        trained_models = {n: m for n, m in trained_models.items() if n in wanted}
        thresholds = {n: t for n, t in thresholds.items() if n in wanted}
        model_seeds = (None,)
        aggregate_seeds = False
    else:
        seeds = sorted(list(model_seeds))
        seed0 = seeds[0]
        tm0, th0 = _train_models(X_train, y_train, model_specs, random_state=seed0, **kw)
        base_thresholds = {name: th0.get(name, 0.5) for name in tm0}
        trained_models = {}
        thresholds = {}

        for name, est in tm0.items():
            key = name if "Dummy" in name else f"{name}_s{seed0}"
            trained_models[key] = est
            thresholds[key] = base_thresholds.get(name, 0.5)

        for seed in seeds[1:]:
            kw["tune_threshold"] = False
            tm, _ = _train_models(X_train, y_train, model_specs, random_state=seed, **kw)
            for name, est in tm.items():
                key = name if "Dummy" in name else f"{name}_s{seed}"
                trained_models[key] = est
                thresholds[key] = base_thresholds.get(name, 0.5)

        if save_path:
            joblib.dump({"models": trained_models, "thresholds": thresholds}, save_path)
            logger.info(f"Saved models to {save_path}")

    metrics_rows = []
    conf_by_model = []
    spec_lookup = {s.name: s for s in model_specs}
    w = kw.get('window_size')

    for key, model in tqdm(trained_models.items(), desc="Evaluating models"):
        name, _, seed = key.partition("_s") if "_s" in key else (key, "", 0)
        thresh = thresholds.get(key, 0.5)
        spec = spec_lookup.get(name)
        if not spec:
            logger.warning(f"Model {name} present in cache but not in spec list")
            continue

        CM = np.zeros((2, 2), dtype=int)
        delays = []
        tot_time = 0.0
        signal_time = 0
        conf_map = {}

        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 100000 or (120001 < len(ts) < 300000):
                continue
            signal_time += len(ts)
            const_label = 1 if spec.kind == "baseline" and "Fall" in name else 0 if "ADL" in name else False
            c, ave_t = sliding_window_confidence(ts, y, model, const_confidence=const_label, **kw)
            conf_map[i] = c
            cm, hit, delay = detect(ts, y, c, confidence_thresh=thresh, const_confidence=const_label, **kw)
            CM += cm
            delays.append(delay)
            tot_time += ave_t

        ave_t = tot_time / max(1, len(conf_map))
        row = compute_row(CM, signal_time, ave_t, np.mean(delays))
        row.update(model=name, window_size=w, seed=seed, thresh=thresh)
        metrics_rows.append(row)
        if spec.kind != "baseline":
            conf_by_model.append((spec.kind, pd.Series(conf_map)))

    def _ensemble(label, series_list):
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

    if ensemble_models and len(conf_by_model) > 1:
        _ensemble("Ensemble-All", [s for _, s in conf_by_model])
        if ensemble_by_kind:
            buckets = {}
            for kind, series in conf_by_model:
                buckets.setdefault(kind, []).append(series)
            for kind, bucket in buckets.items():
                if len(bucket) > 1:
                    _ensemble(f"Ensemble-{kind}", bucket)

    logger.info("✅ Evaluation complete")
    metrics_df = pd.DataFrame(metrics_rows)
    meta = ["model", "seed", "window_size"]
    return metrics_df[meta + [c for c in metrics_df.columns if c not in meta]]