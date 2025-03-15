import copy, joblib
from email.policy import default
from re import X
import timeit
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as mcolors
import test
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing, utils
from sklearn.model_selection import KFold

import matplotlib.ticker as mticker

from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay

from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.convolution_based import RocketClassifier, HydraClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier

class WorkerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, worker_model):
        self.worker_model = worker_model

    def fit(self, X, y=None):
        self.worker_model.fit(X, y)
        return self

    def transform(self, X):
        return self.worker_model.predict_proba(X)

def get_models(**kwargs):
    
    all_models = {
        'tabular': {
            'LogisticCV': LogisticRegressionCV(cv=5, n_jobs=-1, solver='newton-cg'),
            'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_features=0.1,criterion="entropy", n_jobs=-1,random_state=0),
        },
        'ts': {
            'Rocket': RocketClassifier(random_state=0, n_jobs=-1),
            'Catch22': Catch22Classifier(random_state=0, n_jobs=-1),
            'QUANT': QUANTClassifier(random_state=0)
        }
    }

    default_kwargs = {'model_type': None, 'models_subset': None}
    kwargs = {**default_kwargs, **kwargs}

    collaposed_models = {**all_models['tabular'], **all_models['ts']}

    if kwargs['model_type'] is None: # run all models
        models = collaposed_models
    else: # the saner choice :-)
        models = all_models[kwargs['model_type']]
    if kwargs['models_subset'] is not None: # select model subset
        models = {**all_models['tabular'], **all_models['ts']}
        models = {m: collaposed_models[m] for m in kwargs['models_subset']}
    
    return models

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
        model_metrics.update(get_metric_row_dict(CM, ave_time, signal_time, delays, **kwargs)) 
        metrics_rows.append(model_metrics) 
        confidence_data.append(confidences_for_model)
        print('.', end=' ')
        confidence_df = pd.DataFrame(confidence_data)
    if kwargs['ensemble']:
        print('Ensemble', end='')
        CM = np.zeros((2,2))
        delays = []
        ensemble_metrics = {'model': 'Ensemble', 'window_size': kwargs['window_size']}
        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 120000 or len(ts) > 120001:
                continue
            c = confidence_df[i].mean()
            cm, h, delay = utils.detect(ts, y, c, **kwargs)
            delays.append(delay)
            plot_detection(ts, y, c, cm, h, ave_time, model_name=model_name, **kwargs)
            CM += cm
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

def cross_validate(dataset, **kwargs):
    default_kwargs = {
        'model_type': None, 'models_subset': None, 'window_size': 40, 'cv': 5, 'loaded_df': None, 
        'prefall': 1, 'verbose': True, 'random_state': 0, 'multiphase': False, 
        'thresh': 1.1, 'step': 1, 'segment_test': False, 'random_state': 0
    }
    kwargs = {**default_kwargs, **kwargs}
    dataset_name = get_dataset_name(dataset)
    
    if kwargs['loaded_df'] is None:
        df = dataset.load()
    else:
        df = kwargs['loaded_df']
        
    subjects = list(df['SubjectID'].unique())
    # divide subjects into cv sets
    if kwargs['cv'] == 1:
        train, test = train_test_split(subjects, test_size=0.3, random_state=kwargs['random_state'])
        test_sets = [test]
    else:
        rng = np.random.default_rng(kwargs['random_state'])
        rng.shuffle(subjects)
        test_sets = list(chunk_list(subjects, kwargs['cv']))
    
    freq = 100
    metrics_df = None
    
    for i, test_set in enumerate(test_sets):
        X_train, X_test, y_train, y_test = utils.split_df(df, dataset, test_set, **kwargs)
        if kwargs['verbose']:
            print(f'\n\n-- fold {i+1}, testing on ({len(test_set)} subjects) --')
            print(f"Train set: X: {X_train.shape}, y: {y_train.shape} ([ADLs, Falls])", np.bincount(y_train))
            print(f"Test set: X: {len(X_test)}, y: {len(y_test)}")
        
        if metrics_df is None:
            metrics_df = run_models(X_train, X_test, y_train, y_test, freq=freq, **kwargs)
            metrics_df['fold'] = i
        else:
            this_df = run_models(X_train, X_test, y_train, y_test, freq=freq, **kwargs)
            this_df['fold'] = i
            metrics_df = pd.concat([metrics_df, this_df], ignore_index=True)
    
    mean_df = metrics_df.groupby(['model']).mean().round(2)
    std_df = metrics_df.groupby(['model']).std().round(2)
    cols = ['model', 'window_size', 'runtime', 'auc', 'precision', 'recall', 'specificity', 'f1-score', 'false alarm rate', 'miss rate', 'delay']
    aggr = {c: [] for c in cols}
    
    for i in mean_df.index:
        aggr['model'].append(i)
        for col in cols[1:]:
            aggr[col].append(f'{mean_df.loc[i][col]} ± {std_df.loc[i][col]}')
    
    aggr_df = pd.DataFrame(data=aggr)
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

def get_sample_attributions(clf, X_test, y_test, c=28, normalise=True, n=2):
    y_pred = clf.predict(X_test)
    true_falls = np.logical_and(y_test==1, y_pred==1)
    false_falls = np.logical_and(y_test==0, y_pred==1)
    true_adls = np.logical_and(y_test==0, y_pred==0)
    false_adls = np.logical_and(y_test==1, y_pred==0)
    tp_exp = explain_model(clf, X_test[true_falls][:n],
                           y_test[true_falls][:n], chunks=c,
                           normalise=normalise)
    print(X_test[false_adls][:n].shape)
    fp_exp = explain_model(clf, X_test[false_falls][:n],
                           y_test[false_falls][:n], chunks=c,
                           normalise=normalise)
    tn_exp = explain_model(clf, X_test[true_adls][:n],
                            y_test[true_adls][:n], chunks=c,
                            normalise=normalise)
    fn_exp = explain_model(clf, X_test[false_adls][:n],
                            y_test[false_adls][:n], chunks=c,
                            normalise=normalise)
    tp = {'sample':X_test[true_falls][0], 'attr': tp_exp[0]}
    fp = {'sample':X_test[false_falls][0], 'attr': fp_exp[0]}
    tn = {'sample':X_test[true_adls][0], 'attr': tn_exp[0]}
    fn = {'sample':X_test[false_adls][0], 'attr': fn_exp[0]}

    return [tp, fp, tn, fn]

def scale_arr(arr):
    scaler = MinMaxScaler(feature_range=(-1,1))
    return scaler.fit_transform(arr.reshape(-1,1)).flatten()

def plot_sample_with_attributions(attr_dict):
    titles = ['True Fall', 'False Alarm', 'True ADL', 'Missed Fall']
    plt.rcParams.update({'font.size': 10})
    cmap = plt.get_cmap('coolwarm')
    attributions = copy.deepcopy(attr_dict)
    for i, (model_name, exps) in enumerate(attributions.items()):
        fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=400,
                            sharey='row', sharex='col', layout='constrained')
        fig.suptitle(model_name)
        for e, exp in enumerate(exps):
            ax = axs.flat[e]
            ax.set_title(titles[e])
            y = scale_arr(exp['sample'])
            x = np.arange(len(y))
            c = exp['attr'].flatten()
            ax.plot(c, linestyle=':', label='attribution profile', alpha=0.35)
            # Normalize the color values
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            for j in range(len(x)-1):
                ax.plot(x[j:j+2], y[j:j+2], color=cmap(norm(c[j])), linewidth=1.5, label='normalised sample' if j==0 else None)
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([i//100 for i in ticks_loc])
            # ax.grid(which='both', axis='x')     
        axs[1,1].legend()
        fig.supylabel('Attribution score')
        # Adding color bar to show the color scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = plt.axes((1.01, 0.05, 0.015, 0.92))
        plt.colorbar(sm, cax=cax)
        fig.supxlabel('Time in seconds')
        # sns.despine()
        plt.savefig(f'figs/{model_name}_explanation.pdf', bbox_inches='tight')
        plt.show()