from sklearn.model_selection import GroupKFold
import numpy as np

class ShuffledGroupKFold(GroupKFold):
    """GroupKFold with one-time random permutation of groups."""
    def __init__(self, n_splits=5, random_state=None):
        super().__init__(n_splits=n_splits)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.random_state)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)              # in-place permutation
        # map shuffled order back to the original groups array
        shuffled_order = np.argsort(np.searchsorted(unique_groups, groups))
        for train_idx, test_idx in super().split(
            X, y, groups[shuffled_order]
        ):
            yield shuffled_order[train_idx], shuffled_order[test_idx]
