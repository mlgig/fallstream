import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from .metrics import cost_score


class CostClassifierCV(BaseEstimator, ClassifierMixin):
    """
    Cost-sensitive ensemble classifier with optional calibration.

    This meta-estimator optimizes a weighted ensemble of base classifiers
    to maximize a cost-sensitive gain function. It supports two fusion methods:
        - "dirichlet": random search over weight vectors + threshold tuning.
        - "stacking": logistic regression meta-learner on base probabilities.

    Parameters
    ----------
    base_estimators : list of estimators
        The base classifiers to ensemble.
    alpha : float, default=2
        The cost factor for False Negatives (FN).
        Gain = -(FP + alpha * FN).
    n_dirichlet : int, default=2000
        Number of random weight vectors to try if method="dirichlet".
    n_thresholds : int, default=100
        Number of thresholds to evaluate between 0 and 1.
    cv : int, default=5
        Number of cross-validation folds for generating out-of-fold predictions.
    random_state : int, RandomState instance or None, default=None
        Controls randomness in dirichlet sampling and CV splitting.
    calibration : str, default="sigmoid"
        Method for calibration ('sigmoid' or 'isotonic') applied to base estimators.
        If None, base estimators are not calibrated.
    recall_floor : float, default=0.98
        Minimum required recall. Solutions with lower recall are rejected
        during optimization (only applies to 'dirichlet' method).
    method : {'dirichlet', 'stacking'}, default='stacking'
        The optimization strategy.
    """

    def __init__(
        self,
        base_estimators,
        alpha=2,
        n_dirichlet=2000,
        n_thresholds=100,
        cv=5,
        random_state=None,
        calibration="sigmoid",
        recall_floor=0.98,
        method="stacking",
    ):
        self.base_estimators = base_estimators
        self.alpha = alpha
        self.n_dirichlet = n_dirichlet
        self.n_thresholds = n_thresholds
        self.cv = cv
        self.random_state = random_state
        self.calibration = calibration
        self.recall_floor = recall_floor
        self.method = method

    def _recall(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return 0.0 if (tp + fn) == 0 else tp / (tp + fn)

    def _gain(self, y_true, y_pred):
        # Delegate to the unified metric function
        return cost_score(y_true, y_pred, alpha=self.alpha)

    def _wrap_estimator(self, est):
        """Clone and optionally calibrate a base estimator."""
        # Set random state if the estimator supports it
        if hasattr(est, "random_state"):
            est.random_state = self.random_state

        if self.calibration is not None and len(self.base_estimators) > 1:
            return CalibratedClassifierCV(clone(est), cv=3, method=self.calibration)
        return clone(est)

    def fit(self, X, y):
        """
        Fit the ensemble weights and threshold.
        """
        rng = np.random.RandomState(self.random_state)
        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Prepare array to store OOF probabilities
        oof_probs = np.zeros((len(y), len(self.base_estimators)))

        # Collect out-of-fold probabilities
        for train_idx, valid_idx in cv_splitter.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train = y[train_idx]

            for i, est in enumerate(self.base_estimators):
                model = self._wrap_estimator(est).fit(X_train, y_train)
                probs = model.predict_proba(X_valid)[:, 1]
                oof_probs[valid_idx, i] = probs

        # Optimization Strategy

        # FAST PATH: Single Model
        if len(self.base_estimators) == 1:
            self.weights_ = np.array([1.0])
            fused_probs = oof_probs[:, 0]

            # Simple threshold scan
            best_gain, best_tau = -np.inf, 0.5
            taus = np.linspace(0, 1, self.n_thresholds)

            for tau in taus:
                preds = (fused_probs >= tau).astype(int)
                g = self._gain(y, preds)
                if g > best_gain:
                    best_gain, best_tau = g, tau
            self.threshold_ = best_tau

        # ENSEMBLE PATH: Dirichlet Random Search
        elif self.method == "dirichlet":
            best_gain, best_w, best_tau = -np.inf, None, None
            taus = np.linspace(0, 1, self.n_thresholds)

            # Random search over weights
            weights_samples = rng.dirichlet(
                np.ones(len(self.base_estimators)), size=self.n_dirichlet
            )

            for w in weights_samples:
                fused = np.dot(oof_probs, w)
                for tau in taus:
                    preds = (fused >= tau).astype(int)

                    # Check recall constraint
                    if self.recall_floor is not None:
                        rec = self._recall(y, preds)
                        if rec < self.recall_floor:
                            continue

                    g = self._gain(y, preds)
                    if g > best_gain:
                        best_gain, best_w, best_tau = g, w, tau

            self.weights_ = (
                best_w
                if best_w is not None
                else np.ones(len(self.base_estimators)) / len(self.base_estimators)
            )
            self.threshold_ = best_tau if best_tau is not None else 0.5

        # ENSEMBLE PATH: Stacking Meta-Learner
        elif self.method == "stacking":
            meta = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=500,
                random_state=self.random_state,
                class_weight={0: 1, 1: self.alpha},
            )
            meta.fit(oof_probs, y)

            coefs = np.maximum(meta.coef_[0], 0)  # Enforce non-negative weights
            if coefs.sum() == 0:
                coefs = np.ones_like(coefs)

            self.weights_ = coefs / coefs.sum()

            # Tune threshold on the weighted probabilities
            best_gain, best_tau = -np.inf, 0.5
            taus = np.linspace(0, 1, self.n_thresholds)
            fused = np.dot(oof_probs, self.weights_)

            for tau in taus:
                preds = (fused >= tau).astype(int)
                g = self._gain(y, preds)
                if g > best_gain:
                    best_gain, best_tau = g, tau
            self.threshold_ = best_tau

        else:
            raise ValueError(
                f"Unknown method: {self.method}. Choose 'dirichlet' or 'stacking'."
            )

        # Store Optimization Curve (Gain vs Threshold) for plotting
        # calculate the gain for the SELECTED weights across ALL thresholds
        curve_taus = np.linspace(0, 1, 100)
        curve_scores = []
        final_oof_fused = np.dot(oof_probs, self.weights_)

        for t in curve_taus:
            preds = (final_oof_fused >= t).astype(int)
            curve_scores.append(self._gain(y, preds))

        self.optimization_curve_ = (curve_taus, np.array(curve_scores))

        # Refit base estimators on full training set
        self.fitted_estimators_ = [
            self._wrap_estimator(est).fit(X, y) for est in self.base_estimators
        ]

        return self

    def predict_proba(self, X):
        """Return weighted probabilities [P(y=0), P(y=1)]."""
        # Collect probs from all fitted estimators
        probs = np.column_stack(
            [est.predict_proba(X)[:, 1] for est in self.fitted_estimators_]
        )
        # Weighted sum
        fused = np.dot(probs, self.weights_)
        return np.column_stack([1 - fused, fused])

    def predict(self, X):
        """Predict class labels using the optimized threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)
