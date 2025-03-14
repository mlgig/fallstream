import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class WorkerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, worker_model):
        self.worker_model = worker_model

    def fit(self, X, y=None):
        self.worker_model.fit(X, y)
        return self

    def transform(self, X):
        return self.worker_model.predict_proba(X)

# class VerticalConcatenator(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass  # No initialization parameters needed

#     def fit(self, X, y=None):
#         return self  # No fitting required

#     def transform(self, X):
#         # X is a list of length 2 (outputs of two transformers)
#         # Assuming X[0] and X[1] have shape (n_samples, m)
#         return np.stack(X, axis=1)
    
# class Standardize3D(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         # Reshape to 2D for fitting
#         X_2D = X.reshape(-1, X.shape[-1])
#         self.scaler = StandardScaler().fit(X_2D)
#         return self

#     def transform(self, X):

#         X_2D = X.reshape(-1, X.shape[-1])
#         X_2D_scaled = self.scaler.transform(X_2D)
#         X_scaled = X_2D_scaled.reshape(X.shape)
#         return X_scaled
    
# class ShapePrinter(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         print("Shape:", X.shape)
#         return X


