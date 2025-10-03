import joblib
import timeit
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing, utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class CostClassifierCV(BaseEstimator, ClassifierMixin):
    """
    Cost-sensitive ensemble classifier with optional calibration.
    Two fusion methods:
        - "dirichlet": random search over weight vectors + threshold tuning
        - "stacking": logistic regression meta-learner on base probs
    """

    def __init__(self, base_estimators, alpha=4, n_dirichlet=2000,
                 n_thresholds=100, cv=5, random_state=None,
                 calibration="sigmoid", recall_floor=0.98,
                 method="stacking"):
        self.base_estimators = base_estimators
        self.alpha = alpha
        self.n_dirichlet = n_dirichlet
        self.n_thresholds = n_thresholds
        self.cv = cv
        self.random_state = random_state
        self.calibration = calibration
        self.recall_floor = recall_floor
        self.method = method  # "dirichlet" or "stacking"

    def _recall(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return 0.0 if (tp+fn)==0 else tp/(tp+fn)

    def _gain(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return -(fp + self.alpha * fn)

    def _wrap_estimator(self, est):
        # Set random state if possible
        if hasattr(est, "random_state"):
            est.random_state = self.random_state
        if self.calibration is not None:
            return CalibratedClassifierCV(clone(est), cv=3, method=self.calibration)
        return clone(est)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        oof_probs = [np.zeros(len(y)) for _ in self.base_estimators]

        # Collect out-of-fold probabilities for all base estimators
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train = y[train_idx]
            for i, est in enumerate(self.base_estimators):
                model = self._wrap_estimator(est).fit(X_train, y_train)
                probs = model.predict_proba(X_valid)[:, 1]
                oof_probs[i][valid_idx] = probs

        oof_probs = np.vstack(oof_probs).T  # (n_samples, n_estimators)

        # --- Fusion by method ---
        if self.method == "dirichlet":
            best_gain, best_w, best_tau = -np.inf, None, None
            taus = np.linspace(0, 1, self.n_thresholds)
            for w in rng.dirichlet(np.ones(len(self.base_estimators)), size=self.n_dirichlet):
                fused = np.dot(oof_probs, w)
                for tau in taus:
                    preds = (fused >= tau).astype(int)
                    rec = self._recall(y, preds)
                    if (self.recall_floor is not None) and (rec < self.recall_floor):
                        continue  # reject this (w, τ) operating point
                    g = self._gain(y, preds)
                    if g > best_gain:
                        best_gain, best_w, best_tau = g, w, tau
            self.weights_ = best_w
            self.threshold_ = best_tau

        elif self.method == "stacking":
            # Logistic regression on OOF probs as meta-model
            meta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500,
                                      random_state=self.random_state,
                                      class_weight={0:1, 1:self.alpha})
            meta.fit(oof_probs, y)
            coefs = np.maximum(meta.coef_[0], 0)  # force non-negativity
            if coefs.sum() == 0:
                coefs = np.ones_like(coefs)
            self.weights_ = coefs / coefs.sum()
            # Tune threshold for cost-sensitive gain
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
            raise ValueError("method must be 'dirichlet' or 'stacking'")

        # Refit base estimators on full training set
        self.fitted_estimators_ = [self._wrap_estimator(est).fit(X, y) for est in self.base_estimators]
        print(f"Chosen weights: {self.weights_}, threshold: {self.threshold_}")
        return self

    def predict_proba(self, X):
        probs = np.column_stack([est.predict_proba(X)[:, 1] for est in self.fitted_estimators_])
        fused = np.dot(probs, self.weights_)
        return np.column_stack([1 - fused, fused])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)


def cost_fn(y_true=None, y_pred=None, cm=None, **kwargs):
    """
    Thanks to https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html#sphx-glr-auto-examples-model-selection-plot-cost-sensitive-learning-py
    The count of true negatives is cm[0,0], false negatives is cm[1,0], true positives is cm[1,1] and false positives is cm[0,1].
    """
    default_kwargs = {'fn_factor': 5, 'tp_factor': 1, 'fp_factor': 1}
    kwargs = {**default_kwargs, **kwargs}
    if cm is None:
        cm = confusion_matrix(y_true, y_pred)

    fn_factor = kwargs['fn_factor']
    tp_factor = kwargs['tp_factor']
    fp_factor = kwargs['fp_factor']
    unit_cost = 1
    fn_cost = fn_factor * unit_cost
    total_cost = unit_cost + fn_cost
    norm_fn = fn_cost / total_cost
    norm_fp = (unit_cost / total_cost) * fp_factor

    gain_matrix = np.array([[unit_cost, -norm_fp], # -1 gain for false alarms
                            [-norm_fn, tp_factor*unit_cost] # -10 gain for missed falls, 1 gain for TP
                          ])
    return np.sum(cm * gain_matrix)


def train_models(X_train, y_train, **kwargs):
    default_kwargs = {'tune_threshold': False}
    kwargs = {**default_kwargs, **kwargs}
    models = get_models(**kwargs)
    trained_models = {}
    best_thresholds = {}
    cost_score = make_scorer(cost_fn)
    print(f'⏳ TRAINING', end=' ')
    for model_name, clf in models.items():
        clf = make_pipeline(
            StandardScaler(),
            SimpleImputer(missing_values=np.nan, strategy='mean'),
            clf
        )
        print(f'{model_name}', end='. ')
        clf.fit(X_train, y_train)
        trained_models[model_name] = clf
        if kwargs['tune_threshold']:
            print('Tuning threshold...')
            clf_tuned = TunedThresholdClassifierCV(
                clf, cv=5, scoring=cost_score
                ).fit(X_train, y_train)
            best_threshold = clf_tuned.best_threshold_
            best_thresholds[model_name] = best_threshold
            print(f'thresh: {best_threshold}', end=' ')
    print('✅')
    return trained_models, best_thresholds

def get_metric_row(TP, FP, TN, FN, model_name, ave_time, signal_time, **kwargs):
    auc, precision, recall, specificity, f1, far, mr = utils.compute_metrics(TP, FP, TN, FN, signal_time, **kwargs)
    metrics = {'model': model_name, 'window_size': [kwargs['window_size']],'runtime': [ave_time], 'auc': [auc], 'precision': [precision], 'recall': [recall], 'specificity': [specificity], 'f1-score': [f1], 'false alarm rate': [far], 'miss rate': [mr]}
    return pd.DataFrame(data=metrics)

def get_metric_row_dict(CM, ave_time, signal_time, delays, **kwargs):
    auc, precision, recall, specificity, f1, far, mr, gain = utils.compute_metrics(CM, signal_time, **kwargs)
    delays = np.array(delays)
    ave_delay = np.mean(delays)
    return {'runtime': ave_time, 'auc': auc, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1-score': f1, 'false alarm rate': far, 'miss rate': mr, 'delay': ave_delay, 'g': gain}

def plot_detection(ts, y, c, cm, h, ave_time, **kwargs):
    default_kwargs = {'plot_ave_conf': False}
    kwargs = {**default_kwargs, **kwargs}
    tn, fp, fn, tp = cm.ravel()
    if kwargs['plot']:
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)
    if kwargs['plot_errors'] and (fp >= 1 or fn >= 1):
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)
    if kwargs['plot_ave_conf'] and (fp >= 1 or fn >= 1):
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)


def run_models(X_train, X_test, y_train, y_test, **kwargs):
    default_kwargs = {'cm_grid': (1,5), 'confmat_name': 'confmat', 'freq': 100, 'window_size': 60, 'verbose': 1, 'plot': False, 'plot_errors': False, 'plot_ave_conf': False, 'step': 1, 'saved_models': None, 'ensemble': True}
    kwargs = {**default_kwargs, **kwargs}
    if kwargs['saved_models'] is None:
        trained_models, threshholds = train_models(
            X_train, y_train, **kwargs)
    else:
        models_and_thresholds = joblib.load(kwargs['saved_models'])
        trained_models = models_and_thresholds['models']
        threshholds = models_and_thresholds['thresholds']
        print('Loaded models')
    metrics_rows = []
    confidence_data = []

    print('🔍 TESTING', end=' ')
    for model_name, model in trained_models.items():
        print(f'{model_name}', end='')
        model_metrics = {'model': model_name, 'window_size': kwargs['window_size']}
        signal_time = 0
        TOTAL_TIME = 0
        CM = np.zeros((2,2))
        delays = []
        confidences_for_model = {'model': model_name}
        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 100000 or (len(ts) > 120001 and len(ts) < 300000):
                continue
            signal_time += len(ts)
            c, ave_time = utils.sliding_window_confidence(ts, y, model, **kwargs)
            confidences_for_model[i] = c
            thresh = threshholds[model_name] if threshholds != {} else 0.5
            cm, h, delay = utils.detect(
                ts, y, c, confidence_thresh=thresh, **kwargs)
            delays.append(delay)
            plot_detection(ts, y, c, cm, h, ave_time, model_name=model_name, **kwargs)
            CM += cm
            TOTAL_TIME += ave_time
        AVE_TIME = TOTAL_TIME / len(X_test) 
        model_metrics.update(get_metric_row_dict(CM, AVE_TIME, signal_time, delays, **kwargs)) 
        metrics_rows.append(model_metrics) 
        confidence_data.append(confidences_for_model)
        print('.', end=' ')
        confidence_df = pd.DataFrame(confidence_data)
    if kwargs['ensemble']:
        print('Ensemble', end='')
        CM = np.zeros((2,2))
        delays = []
        ensemble_metrics = {'model': 'Ensemble', 'window_size': kwargs['window_size']}
        start_time = timeit.default_timer()
        signal_time = 0
        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 120000 or len(ts) > 120001:
                continue
            signal_time += len(ts)
            c = confidence_df[i].mean()
            cm, h, delay = utils.detect(ts, y, c, **kwargs)
            delays.append(delay)
            plot_detection(ts, y, c, cm, h, ave_time, model_name=model_name, **kwargs)
            CM += cm
        ave_time = (timeit.default_timer() - start_time) / len(X_test)
        ensemble_metrics.update(get_metric_row_dict(CM, ave_time, signal_time, delays, **kwargs))
        metrics_rows.append(ensemble_metrics)
    metrics_df = pd.DataFrame(metrics_rows)
    print('. ✅')
    return metrics_df

def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def chunk_list(l, n):
    n_per_set = len(l)//n
    for i in range(1, n_per_set*n, n_per_set):
        chunk = l[i:i+n_per_set]
        if len(chunk) < n_per_set:
            break
        yield chunk

def get_dataset_name(dataset):
    names = {
        farseeing: 'FARSEEING',
        # fallalld: 'FallAllD',
        # sisfall: 'SisFall'
    }
    return names[dataset]

def cross_validate(dataset, data_subset=None, **kwargs):
    default_kwargs = {
        'model_type': None, 'models_subset': None, 'window_size': 40, 'cv': 5, 'loaded_df': None, 
        'prefall': 1, 'verbose': True, 'random_state': 0, 'multiphase': False, 
        'thresh': 1.1, 'step': 1, 'segment_test': False, 'model_seeds': [0],
    }
    kwargs = {**default_kwargs, **kwargs}
    dataset_name = get_dataset_name(dataset)
    
    if kwargs['loaded_df'] is None:
        df = dataset.load()
    else:
        df = kwargs['loaded_df']

    if data_subset is None:
        data_subset = list(df['SubjectID'].unique())

    freq = 100
    
    if kwargs['cv'] == 1:
        _, test = train_test_split(data_subset, test_size=0.3, random_state=kwargs['random_state'])
        test_sets = [test]
    else:
        rng = np.random.default_rng(kwargs['random_state'])
        rng.shuffle(data_subset)
        test_sets = list(chunk_list(data_subset, kwargs['cv']))
    
    DFS = []
    for i, test_set in enumerate(test_sets):
        print(f'\n-- fold {i+1}, testing on {len(test_set)} subjects --')
        X_train, X_test, y_train, y_test = utils.split_df(df.copy(), dataset, test_set, **kwargs)
        this_df = run_models(X_train, X_test, y_train, y_test, freq=freq, **kwargs)
        DFS.append(this_df.assign(fold=i+1))
    metrics_df = pd.concat(DFS, ignore_index=True)
    
    cols = ['model', 'window_size', 'runtime', 'auc', 'precision', 'recall',
            'specificity', 'f1-score', 'false alarm rate', 'miss rate', 'delay']   
    aggr_df = utils.aggregrate_metrics(df=metrics_df, cols=cols)
    aggr_df['Dataset'] = dataset_name
    metrics_df['Dataset'] = dataset_name
    model_type = kwargs['model_type']
    s = kwargs['window_size']
    aggr_df.to_csv(f'results/{dataset_name}{model_type}{s}.csv')
    
    return metrics_df, aggr_df

def boxplot(df, dataset, model_type, metric='f1-score', save=False, **kwargs):
    plt.figure(figsize=(5, 3), dpi=400)
    sns.boxplot(data=df.sort_values(by='model'),
                x='model', y='f1-score',
                width=0.3, **kwargs)
    plt.grid(axis='y')
    plt.xlabel('')
    sns.despine()
    plt.title(f'{model_type.capitalize()} models CV {metric}s on {dataset.upper()}')
    plt.xticks(rotation=15)
    if save:
        plt.savefig(f'figs/{dataset}_{model_type}_boxplot.eps',
                    format='eps', bbox_inches='tight')
    plt.show()