# scripts/model_spec.py
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.base import clone

@dataclass(slots=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    kind: str = "generic"
    def clone(self, seed=None):
        est = clone(self.estimator)
        if seed is not None and hasattr(est, "random_state"):
            est.random_state = seed
        return est
