from __future__ import annotations

from typing import List, Sequence
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.deep_learning import ResNetClassifier
from sklearn.base import BaseEstimator, clone

from scripts.model_spec import ModelSpec
from scripts.baselines import AlwaysADL, AlwaysFall
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="loss",
    patience=20,
    min_delta=1e-4,
    restore_best_weights=True
)

_TEMPLATE_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="DummyADL",
        estimator=AlwaysADL,
        kind="baseline",
    ),
    ModelSpec(
        name="DummyFall",
        estimator=AlwaysFall,
        kind="baseline",
    ),
    ModelSpec(
        name="ExtraTrees",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean", 
                          missing_values=np.nan),
            ExtraTreesClassifier(
                n_estimators=150,
                max_features=0.1,
                criterion="entropy",
            ),
        ),
        kind="tabular",
    ),
    ModelSpec(
        name="MiniRocket",
        estimator=make_pipeline(
            SimpleImputer(strategy="mean", missing_values=np.nan),
            MiniRocketClassifier(n_jobs=-1)
        ),
        kind="ts",
    ),
    ModelSpec(
        name="Catch22",
        estimator=make_pipeline(
            SimpleImputer(strategy="mean", missing_values=np.nan),
            Catch22Classifier()
        ),
        kind="ts",
    ),
    ModelSpec(
        name="QUANT",
        estimator=make_pipeline(
            SimpleImputer(strategy="mean", missing_values=np.nan),
            QUANTClassifier()
        ),
        kind="ts",
    ),
    ModelSpec(
        name="ResNet",
        estimator=make_pipeline(
            SimpleImputer(strategy="mean", missing_values=np.nan),
            ResNetClassifier(
                metrics=[keras.metrics.F1Score()],
                file_path="./.keras",
                callbacks=[early_stop])
        ),
        kind="dl",
    )
]

def _set_seed(est: BaseEstimator | Pipeline, seed: int):
    """Recursively set random_state on estimators that expose the attr."""
    if hasattr(est, "random_state"):
        est.random_state = seed

    if isinstance(est, Pipeline):
        for _, step in est.steps:
            _set_seed(step, seed)


def get_model_specs(
    *,
    kind: str | None = None,
    subset: Sequence[str] | None = None,
    random_state: int | None = 0,
) -> List[ModelSpec]:
    """Return fresh *seeded* ModelSpec list.

    Parameters
    ----------
    kind : str | None
        Filter by the ``kind`` tag (e.g. "ts" or "tabular").  ``None`` = no filter.
    subset : list[str] | None
        Explicit list of ``name`` values to keep.  ``None`` = keep all.
    random_state : int | None
        If not ``None`` set this seed on every estimator that exposes a
        ``random_state`` attribute (including those inside Pipelines).
    """

    specs = _TEMPLATE_SPECS
    if kind is not None:
        specs = [s for s in specs if s.kind == kind]
    if subset is not None:
        want = set(subset)
        specs = [s for s in specs if s.name in want]

    seeded_specs: List[ModelSpec] = []
    for spec in specs:
        est_clone = clone(spec.estimator)
        if random_state is not None:
            _set_seed(est_clone, random_state)
        seeded_specs.append(
            ModelSpec(name=spec.name, estimator=est_clone, kind=spec.kind)
        )
    return seeded_specs
