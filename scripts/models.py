from __future__ import annotations

from typing import List, Sequence

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from sklearn.base import BaseEstimator, clone

from scripts.model_spec import ModelSpec


_TEMPLATE_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="LogisticCV",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            LogisticRegressionCV(cv=5, solver="newton-cg"),
        ),
        kind="tabular",
    ),
    ModelSpec(
        name="RandomForest",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            RandomForestClassifier(n_estimators=150),
        ),
        kind="tabular",
    ),
    ModelSpec(
        name="ExtraTrees",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            ExtraTreesClassifier(
                n_estimators=150,
                max_features=0.1,
                criterion="entropy",
            ),
        ),
        kind="tabular",
    ),
    ModelSpec(
        name="Rocket",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            RocketClassifier(n_jobs=-1)
        ),
        kind="ts",
    ),
    ModelSpec(
        name="Catch22",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            Catch22Classifier(n_jobs=-1)
        ),
        kind="ts",
    ),
    ModelSpec(
        name="QUANT",
        estimator=make_pipeline(
            StandardScaler(),
            SimpleImputer(strategy="mean"),
            QUANTClassifier()
        ),
        kind="ts",
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
