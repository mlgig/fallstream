from sklearn.dummy import DummyClassifier
import numpy as np

# Always predicts the majority class
AlwaysADL = DummyClassifier(strategy="most_frequent")
AlwaysFall = DummyClassifier(strategy="constant", constant=1)

# # Always predicts
# class AlwaysFall(BaseEstimator, ClassifierMixin):
#     def fit(self, X, y=None):
#         self.classes_ = np.array([0, 1], dtype=int)
#         return self
#     def predict(self, X):
#         return np.ones(len(X), dtype=int)
#     def predict_proba(self, X):          # 100 % probability on class 1
#         p = np.zeros((len(X), 2), float)
#         p[:, 1] = 1.0
#         return p