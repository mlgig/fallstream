import copy
from email.policy import default
from re import sub
import timeit
from turtle import mode
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as mcolors
from sympy import plot
import test
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing, utils
from scripts.utils import get_freq
from sklearn.model_selection import KFold

import matplotlib.ticker as mticker

from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay

from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.convolution_based import RocketClassifier, HydraClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.deep_learning import LITETimeClassifier, ResNetClassifier


def get_models(**kwargs):
    all_models = {
        'tabular': {
            'LogisticCV': LogisticRegressionCV(cv=5, n_jobs=-1, solver='newton-cg'),
            'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_features=0.1,criterion="entropy", n_jobs=-1,random_state =0),
        },
        'ts': {
            'Rocket': RocketClassifier(random_state=0, n_jobs=-1),
            'Catch22': Catch22Classifier(random_state=0, n_jobs=-1),
            'QUANT': QUANTClassifier(random_state=0)
        }
    }

    default_kwargs = {'model_type': None, 'models_subset': None}
    kwargs = {**default_kwargs, **kwargs}

    if kwargs['model_type'] is None: # run all models
        models = {**all_models['tabular'], **all_models['ts']}
    else: # the saner choice :-)
        models = all_models[kwargs['model_type']]
    if kwargs['models_subset'] is not None: # select model subset
        models = {m: models[m] for m in kwargs['models_subset']}
    
    return models

def train_models(X_train, y_train, **kwargs):
    default_kwargs = {'tuned_threshold': False}
    kwargs = {**default_kwargs, **kwargs}
    models = get_models(**kwargs)
    trained_models = {}
    best_thresholds = {}
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
        if kwargs['tuned_threshold']:
            clf_tuned = TunedThresholdClassifierCV(
                clf, cv=5, scoring='f1',
                n_jobs=-1).fit(X_train, y_train)
            best_threshold = clf_tuned.best_threshold_
            best_thresholds[model_name] = best_threshold
    print('✅')
    return trained_models, best_thresholds

def get_metric_row(TP, FP, TN, FN, model_name, ave_time, signal_time, **kwargs):
    auc, precision, recall, specificity, f1, far, mr = utils.compute_metrics(TP, FP, TN, FN, signal_time)
    metrics = {'model': model_name, 'window_size': [kwargs['window_size']],'runtime': [ave_time], 'auc': [auc], 'precision': [precision], 'recall': [recall], 'specificity': [specificity], 'f1-score': [f1], 'false alarm rate': [far], 'miss rate': [mr]}
    return pd.DataFrame(data=metrics)

def get_metric_row_dict(TP, FP, TN, FN, ave_time, signal_time, **kwargs):
    auc, precision, recall, specificity, f1, far, mr = utils.compute_metrics(TP, FP, TN, FN, signal_time)
    return {'runtime': ave_time, 'auc': auc, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1-score': f1, 'false alarm rate': far, 'miss rate': mr}

def plot_detection(ts, y, c, tp, fp, tn, fn, h, ave_time, **kwargs):
    default_kwargs = {'plot_ave_conf': False}
    kwargs = {**default_kwargs, **kwargs}
    if kwargs['plot']:
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)
    if kwargs['plot_errors'] and (fp >= 1 or fn >= 1):
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)
    if kwargs['plot_ave_conf'] and (fp >= 1 or fn >= 1):
        utils.plot_confidence(ts, c, y, tp, fp, tn, fn, high_conf=h, ave_time=ave_time, **kwargs)


def run_models(X_train, X_test, y_train, y_test, **kwargs):
    default_kwargs = {'cm_grid': (1,5), 'confmat_name': 'confmat', 'freq': 100, 'window_size': 60, 
                      'verbose': 1, 'plot': False, 'plot_errors': False, 'plot_ave_conf': False}
    kwargs = {**default_kwargs, **kwargs}
    trained_models, threshholds = train_models(
        X_train, y_train, **kwargs)
    metrics_rows = []
    confidence_data = []

    print('🔍 TESTING', end=' ')
    for model_name, model in trained_models.items():
        print(f'{model_name}', end='')
        model_metrics = {'model': model_name, 'window_size': kwargs['window_size']}
        signal_time = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        confidences_for_model = {'model': model_name}
        for i, (ts, y) in enumerate(zip(X_test, y_test)):
            if len(ts) < 120000 or len(ts) > 120001:
                continue
            signal_time += len(ts)
            c, ave_time = utils.sliding_window_confidence(ts, y, model, **kwargs)
            confidences_for_model[i] = c
            thresh = threshholds[model_name] if threshholds != {} else 0.5
            tp, fp, tn, fn, h = utils.detect(
                ts, y, c, confidence_thresh=thresh, **kwargs)
            plot_detection(ts, y, c, tp, fp, tn, fn, h, ave_time, model_name=model_name, **kwargs) 
            TP += tp
            FP += fp
            TN += tn
            FN += fn
        model_metrics.update(get_metric_row_dict(TP, FP, TN, FN, ave_time, signal_time, **kwargs)) 
        metrics_rows.append(model_metrics) 
        confidence_data.append(confidences_for_model)
        print('.', end=' ')
        confidence_df = pd.DataFrame(confidence_data)

    print('Ensemble', end='')
    TP, FP, TN, FN = 0, 0, 0, 0
    ensemble_metrics = {'model': 'Ensemble', 'window_size': kwargs['window_size']}
    for i, (ts, y) in enumerate(zip(X_test, y_test)):
        if len(ts) < 120000 or len(ts) > 120001:
            continue
        c = confidence_df[i].mean()
        tp, fp, tn, fn, h = utils.detect(ts, y, c, **kwargs)
        plot_detection(ts, y, c, tp, fp, tn, fn, h, ave_time, model_name=model_name, **kwargs)
        TP += tp
        FP += fp
        TN += tn
        FN += fn
    ensemble_metrics.update(get_metric_row_dict(TP, FP, TN, FN, ave_time, signal_time, **kwargs))
    metrics_rows.append(ensemble_metrics)
    metrics_df = pd.DataFrame(metrics_rows)
    print('. ✅')
    return metrics_df

def plot_metrics(df, x='model', pivot='f1-score', compare='metrics', **kwargs):
    default_kwargs = {'figsize': (6,2), 'rot': 0}
    kwargs = {**default_kwargs, **kwargs}
    if compare=='metrics':
        w = max(df['window_size'])
        window_df = df[df['window_size']==w].drop(columns=['window_size', 'runtime'])
        window_df.plot(kind='bar', x='model', **kwargs)
    elif compare=='window_size':
        crosstab = df.pivot_table(pivot, ['model'], 'window_size')
        crosstab.plot(kind='bar', rot=0, **kwargs)
        plt.grid()
        plt.xlabel('')
        plt.ylabel('')
        sns.despine()
    else:
        df.plot(kind='bar', x='model', y=compare)
    plt.legend(loc=9, ncol=3, bbox_to_anchor=(0.5,1.3), title=compare) 

def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def window_from_midpoint(X, win_size):
    h = win_size//2
    mid = X.shape[1]//2
    return X[:, mid-h:mid+h]

def predict_eval(model, X_in=None, y_in=None, starttime=None, **kwargs):
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    print(f'{model_name}', end='')
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    # X_train = X_train[:, :win_size]
    # X_test = X_test[:, :win_size]
    # select half of win_size around the middle X_train and X_test
    # X_train = window_from_midpoint(X_train, win_size)
    # X_test = window_from_midpoint(X_test, win_size)

    print('.', end='')
    clf.fit(X_train, y_train)
    print('.', end='')
    if starttime is None:
        starttime = timeit.default_timer()
    if has_proba:
        probs = clf.predict_proba(X_test)[:, 1]
        train_probs = clf.predict_proba(X_train)[:, 1]
        auc_score = np.round(roc_auc_score(y_test, probs), 2)
    else:
        probs = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = np.round(auc(fpr, tpr), 2)
    y_pred = clf.predict(X_test)
    runtime = (timeit.default_timer() - starttime)/X_test.shape[0]
    runtime = np.round(runtime * 1000000) # microseconds (ms)
    print('.', end=' ')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1)
    if kwargs['verbose'] > 1:
        print(f'{model_name} AUC: {auc_score}')
        print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    return dict({'model': [model_name],
                 'window_size': kwargs['window_size'],
                 'runtime':[np.round(runtime, 2)],
                 'auc': [np.round(auc_score*100, 2)],
                 'precision':[np.round(precision*100, 2)],
                 'recall':[np.round(recall*100, 2)],
                 'specificity': [np.round(specificity*100, 2)],
                 'f1-score':[np.round(f1*100, 2)]}
                ), clf, cm

def plot_cm(cm, model_name, ax, fontsize=20, colorbar=False):
    target_names = ['ADL', 'Fall']
    plt.rcParams.update({'font.size': fontsize})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=colorbar)
    # plt.grid(False)
    plt.rcParams.update({'font.size': 10})


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
    
    freq = get_freq(dataset)
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
    cols = ['model', 'window_size', 'runtime', 'auc', 'precision', 'recall', 'specificity', 'f1-score', 'false alarm rate', 'miss rate']
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