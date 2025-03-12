import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.signal import resample
from sklearn.metrics import f1_score
import time, timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from scripts import farseeing, classifiers


def magnitude(arr):
    x, y, z = np.array(arr).T.astype('float')
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude -= min(magnitude)
    return magnitude

def visualize_samples(X_train, X_test, y_train, y_test, dataset):
    X = np.vstack(X_train, X_test)
    y = np.vstack(y_train, y_test)
    visualize_falls_adls(X, y, dataset=dataset)

def colorlist2(c1, c2, num):
    l = np.linspace(0, 1, num)
    a = np.abs(np.array(c1) - np.array(c2))
    m = np.min([c1, c2], axis=0)
    s = np.sign(np.array(c2) - np.array(c1)).astype(int)
    s[s == 0] = 1
    r = np.sqrt(np.c_[(l * a[0] + m[0])[::s[0]],
                      (l * a[1] + m[1])[::s[1]], (l * a[2] + m[2])[::s[2]]])
    return r

def color_plot(x, y):
    ynorm = (y - y.min()) / (y.max() - y.min())
    cmap = LinearSegmentedColormap.from_list(
        "", colorlist2((1, 0, 0), (0, 0, 1), 100))
    colors = [cmap(k) for k in ynorm[:-1]]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidth=2)
    lc.set_array(x)
    return lc

def plot_all_samples(X, ax, freq=100):
    X = np.squeeze(X) if X.ndim > 2 else X
    # X = np.nan_to_num(X)
    # if X.min() != -1:
    #     X = scale(X)
    ax.plot(X.T, color='lightblue', alpha=0.5)
    # lc = color_plot(x, mean_vals)
    mean_vals = X.mean(axis=0)
    # x = np.arange(len(mean_vals))
    # tiled = np.tile(mean_vals, (400,1))
    # print(tiled.min(), tiled.max())
    # print(tiled)
    # norm = plt.Normalize(-1, 1)
    # im = ax.imshow(tiled, cmap='coolwarm', alpha=0.5, norm=norm)
    # ax.imshow(tiled, cmap='coolwarm', alpha=0.5, norm=norm)
    ax.plot(mean_vals, color='blue', label='mean')
    # ax.add_collection(lc)
    # ax.autoscale()
    # plt.show()
    # return im

def visualize_falls_adls(X, y, dataset="train", save=False):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2), dpi=200,
                        sharey=True, layout='tight')
    # remove dims with size 1
    X = np.squeeze(X)
    y = np.squeeze(y)
    fallers = y.astype(bool)
    falls = X[fallers]
    adls = X[fallers == False]
    plot_all_samples(adls, ax=axs[0])
    axs[0].set_title('ADL samples')
    axs[0].set_ylabel('Accel magnitude (g)')
    
    plot_all_samples(falls, ax=axs[1])
    axs[1].set_title('Fall samples')
    axs[1].legend()
    if save:
        plt.savefig(f'figs/{dataset}_mean_samples.eps', format='eps',
                    bbox_inches='tight')
    plt.show()

def resample_to(arr, old_f, new_f=100):
    new_list = []
    old_len = arr.shape[-1]
    for sample in arr:
        resampled = resample(sample, int(new_f*(old_len/old_f)))
        new_list.append(resampled)
    new_arr = np.array(new_list)
    return new_arr

def get_freq(dataset):
    if dataset==farseeing:
        return 100
    elif dataset==fallalld:
        return 238
    else:
        return 200

def single_subject_split(dataset, **kwargs):
    default_kwargs = {'test_size': 0.3, 'random_state': 0, 'visualize': False, 'clip': False, 'new_freq': None, 'split': True, 'show_test': False, 'window_size': 60, 'segment_test': True, 'prefall': None}
    kwargs = {**default_kwargs, **kwargs}
    df = dataset.load(clip=kwargs['clip'])
    vc = df['SubjectID'].value_counts()
    more_than_3_samples = vc.where(vc>3).dropna().keys()
    df = df[df['SubjectID'].isin(more_than_3_samples)]
    subjects = df['SubjectID'].unique()
    subject_splits = {}
    for subject in subjects:
        if subject in ['00002186', '66280673', '37691185', '40865626']:
            continue
        # select the subject rows in the dataframe
        subject_df = df[df['SubjectID']==subject]
        # train test split the dataframe using sklearn
        train_set, test_set = train_test_split(subject_df, test_size=kwargs['test_size'], random_state=kwargs['random_state'])
        X_train, y_train = dataset.get_X_y(train_set, **kwargs)
        X_test, y_test = dataset.get_X_y(test_set, keep_fall=True, **kwargs)
        subject_splits[subject] = (X_train, X_test, y_train, y_test)
    return subject_splits

def split_df(df, dataset, test_set, **kwargs):
    test_df = df[df['SubjectID']==test_set[0]]
    train_df = df.drop(df[df['SubjectID']==test_set[0]].index)
    for id in test_set[1:]:
        this_df = df[df['SubjectID']==id]
        test_df = pd.concat([test_df, this_df], ignore_index=True)
        df.drop(this_df.index, inplace=True)
        df.reset_index().drop(columns=['index'], inplace=True)
    X_train, y_train = dataset.get_X_y(df, **kwargs)
    X_test, y_test = dataset.get_X_y(test_df, keep_fall=True, **kwargs)

    return X_train, X_test, y_train, y_test

def train_test_subjects_split(dataset, **kwargs):
    default_kwargs = {'test_size': 0.3, 'random_state': 0, 'visualize': False, 'clip': False, 'split': True, 'show_test': False, 'window_size': 60, 'segment_test': True, 'prefall': None, 'new_freq': 100, 'augment_data': None}
    kwargs = {**default_kwargs, **kwargs}
    df = dataset.load(clip=kwargs['clip'])
    subjects = df['SubjectID'].unique()
    if kwargs['split']==False:
        X, y = dataset.get_X_y(df, **kwargs)
        # if resample:
        #     X = resample_to(X, old_f=get_freq(dataset),
        #                     new_f=kwargs['new_freq'])
        return X, y
    else:
        train_set, test_set = train_test_split(subjects, test_size=kwargs['test_size'], random_state=kwargs['random_state'])
        if kwargs['show_test']:
            print(f'Test set -> {len(test_set)} of {len(subjects)} subjects: {test_set}.')
        test_df = df[df['SubjectID']==test_set[0]]
        X_train, X_test, y_train, y_test = split_df(df, dataset, test_set, **kwargs)
        # if resample:
        #     X_train = resample_to(X_train, old_f=get_freq(dataset), new_f=kwargs['new_freq'])
        #     X_test = resample_to(X_test, old_f=get_freq(dataset),
        #                          new_f=kwargs['new_freq'])
        print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
        ([ADLs, Falls])", np.bincount(y_train))
        if kwargs['segment_test']:
            print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
            ([ADLs, Falls])", np.bincount(y_test))
        else:
            print(f"Test set: X: {len(X_test)}, y: {len(y_test)}")
        if kwargs['visualize']:
            visualize_falls_adls(X_train, y_train)
            visualize_falls_adls(X_test, y_test, dataset="test")
        return X_train, X_test, y_train, y_test
    
def visualise_all_metrics(df, metrics, name='farseeing'):
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=400,
                            layout='constrained', sharex=True,
                            sharey='row')
    for i, metric in enumerate(metrics):
        ax = axs.flat[i]
        sns.boxplot(data=df, x='model', y=metric, width=0.5,
                    ax=ax, linewidth=1, palette='tab10')
        ax.grid()
        metric = 'runtime (ms)' if metric == 'runtime' else metric
        metric = 'FAR (per hour)' if metric == 'false alarm rate' else metric
        metric = 'MR (per hour)' if metric == 'miss rate' else metric
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    sns.despine()
    os.makedirs('figs', exist_ok=True)  
    plt.savefig(f'figs/{name}_metrics_boxplot.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def summary_visualization(dfs, x='model', name='models', title='', xlabel=None):
    dataset_names = ['FARSEEING', 'FallAllD', 'SisFall']
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1,len(dfs), figsize=(6, 4), dpi=400,
                            sharey=True, layout='tight')
    for d, df in enumerate(dfs):
        # df.sort_values(by=x, inplace=True)
        ax = axs[d] if len(dfs) > 1 else axs
        sns.boxplot(data=df, x=x, y='f1-score', width=0.5, ax=ax, linewidth=1, palette='tab10')
        ax.grid(axis='y')
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('')
        # ax.set_ylim(50, 100)
        if d != 0:
            ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()
    os.makedirs('figs', exist_ok=True)  
    plt.savefig(f'figs/{name}_summary_boxplot.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def ts_vs_tabular_summary(all_dfs):
    dataset_names = ['FARSEEING', 'FallAllD', 'SisFall']
    # add dataset names to each df
    # concatenate all results for each dataset
    farseeing_cv_df, farseeing_cv_df_ts, fallalld_cv_df, fallalld_cv_df_ts, sisfall_cv_df, sisfall_cv_df_ts = all_dfs
    farseeing_all_df = pd.concat([df.assign(
        dataset=dataset_names[0]) for df in [farseeing_cv_df.assign(type='Tabular Models'),
        farseeing_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    fallalld_all_df = pd.concat([df.assign(
        dataset=dataset_names[1]) for df in [fallalld_cv_df.assign(type='Tabular Models'),
        fallalld_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    sisfall_all_df = pd.concat([df.assign(
        dataset=dataset_names[2]) for df in [sisfall_cv_df.assign(type='Tabular Models'),
        sisfall_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    all_results_df = pd.concat([farseeing_all_df, fallalld_all_df, sisfall_all_df], ignore_index=True)
    all_results_df.to_csv('results/all_results.csv')
    all_results_df.drop(all_results_df[all_results_df['f1-score']==0].index, inplace=True)
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 3), dpi=400)
    # plt.rcParams.update({'font.size': 16})
    sns.boxplot(data=all_results_df, x='type', y='f1-score', hue='dataset', width=0.3)
    # plt.xticks(rotation=45, ha='right')
    plt.grid()
    plt.xlabel('')
    sns.despine()
    plt.savefig('figs/ts_vs_tabular_boxplot_summary.eps', format='eps', bbox_inches='tight')
    plt.show()

def cross_dataset_summary(df):
    # df = pd.concat(dfs, ignore_index=True)
    plt.rcParams.update({'font.size': 13})
    melted = df.drop(columns=['runtime', 'window_size', 'auc', 'specificity']).melt(
        id_vars=["trainset", "model"])
    
    plt.figure(figsize=(9, 3), dpi=400)
    order=['FARSEEING', 'FallAllD', 'FallAllD+', 'SisFall','SisFall+', 'All']
    # melted.replace({'FallAllD+FARSEEING':'FallAllD+',
    #                 'SisFall+FARSEEING':'SisFall+'}, inplace=True)
    sns.boxplot(melted, x='trainset', y='value', hue='variable', width=0.5, palette="tab10", order=order)
    plt.grid(axis='both')
    plt.xlabel('Training Set', labelpad=10)
    plt.ylabel('score')
    sns.despine()
    # plt.legend(loc=9, ncols=3)
    plt.savefig('figs/cross_dataset_boxplot_summary.pdf', bbox_inches='tight')
    plt.show()

def plot_window_size_ablation(window_metrics=None):
    if window_metrics is None:
        window_metrics = pd.read_csv('results/window_size_ablation.csv')
    fig, axs = plt.subplots(2,3, figsize=(9,4), dpi=(400),
                        sharex='col', layout='tight')
    titles = [f'Test time/sample ($\mu$s)', 'AUC',
            'Precision', 'Recall', 'Specificity', f'F$_1$ score']
    for i, col in enumerate(window_metrics.columns[2:]):
        ax = axs.flat[i]
        sns.lineplot(data=window_metrics, x='window_size', y=col, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(titles[i])
    axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.supxlabel('Total window size in seconds')
    plt.savefig('figs/window_size_ablation.pdf', bbox_inches='tight')
    plt.show()

def plot_confidence(ts, c, y, tp, fp, tn, fn, **kwargs):
    default_kwargs = {'high_conf': None, 'title': None}
    kwargs = {**default_kwargs, **kwargs}
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    cmap = plt.get_cmap('coolwarm')
    ax.plot(x, c, linestyle=':', label='confidence')

    # Normalize confidence values for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(c[:-1]))

    # Create a collection of line segments
    segments = [[(x[i], ts[i]), (x[i + 1], ts[i + 1])]
                for i in range(len(ts) - 1)]
    lc = mcoll.LineCollection(segments, colors=colors, linewidths=1.5)
    ax.add_collection(lc)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel('Timepoints')
    ax.set_ylabel('Acceleration (g)')
    ax.set_ylim(-0.1, max(ts)+0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label('Confidence')

    if kwargs['high_conf'] is not None:
        high_conf = kwargs['high_conf']
        for i in range(len(high_conf)):
            ax.axvspan(high_conf[i], high_conf[i] + 4000, color='gray', alpha=0.3)
    
    ax.axvline(x=y, color='red', linestyle='--', label='Fall')
    ax.legend()
    if kwargs['title'] is not None:
        title = kwargs['title']
        ax.set_title(f'Subject {title} | TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}. Time/sample: {kwargs["ave_time"]:.2f} ms. {kwargs["model_name"]}')
    else:
        ax.set_title(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}. Time/sample: {kwargs["ave_time"]:.2f} ms. {kwargs["model_name"]}')
    plt.show()

def detect(ts, fall_point, c, tolerance=20, **kwargs):
    default_kwargs = {'confidence_thresh': 0.5}
    kwargs = {**default_kwargs, **kwargs}
    """Obtain TP, FP, TN, and FN based on confidence mapping."""
    window_size = kwargs['window_size']
    step_size = kwargs['step']
    # Get high confidence regions
    high_conf = get_high_confidence_regions(ts, c, **kwargs)
    
    # n_samples = len(ts) // (kwargs['window_size']*kwargs['freq'])
    n_samples = ((len(ts) - window_size) // step_size) + 1
    TP, FP, TN, FN = 0, 0, 0, 0
    left_all = (fall_point - 100) - 100*(window_size + tolerance)
    right_all = (fall_point + 100*(window_size-1)) + 100*(tolerance)
    fall_range = np.arange(left_all, right_all)
    if high_conf is None:
        TP, FP, TN, FN = 0, 0, n_samples-1, 1
    else:
        for h in high_conf:
            detection_range = range(h, h + 100*(window_size + tolerance))
            IOU = iou(detection_range, fall_range)
            if IOU != 0:
                TP = 1
            else:
                FP += 1
        FN = 1 if TP == 0 else 0
        TN = n_samples - TP - FP - FN
    cm = np.array([[TP, FP], [FN, TN]])
    
    return cm, high_conf

def get_high_confidence_regions(ts, c, **kwargs):
    """Get high confidence regions based on threshold."""

    # Combine signal threshold and confidence threshold
    thresh = kwargs['confidence_thresh']
    high_conf = np.where((c > thresh))[0]
    # high_conf = np.where(c > kwargs['confidence_thresh'])[0]
    # signal_points_above_thresh = np.where(ts > kwargs['signal_thresh'])[0]
    # high_conf = np.intersect1d(high_conf_idx, signal_points_above_thresh, assume_unique=True)
    if len(high_conf) == 0:
        return None
    high_conf_diff = np.diff(high_conf, prepend=0)
    first_point = high_conf[0]
    # Remove points that are within a minute of each other
    the_rest = high_conf[1:][high_conf_diff[1:] >= 6000]
    if len(the_rest) == 0:
        high_conf = np.array([first_point])
    else:
        high_conf = np.array([first_point, *the_rest])
    # Final check on high confidence points
    # high_conf = [h for h in high_conf if c[h] >= max(c[:h])]
    return high_conf if len(high_conf) > 0 else None

def iou(a, b):
    """Compute intersection over union of two sets."""
    intersection = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(b)))
    return intersection / union if union > 0 else 0

def compute_metrics(cm, signal_time):
    """Compute metrics based on TP, FP, TN, FN, and signal time."""   
    # Compute metrics
    tp, fp, fn, tn = cm.ravel()
    precision = np.round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Compute AUC-ROC estimate
    fpr = 1 - specificity
    fnr = 1 - recall
    auc = (recall + (1 - fpr)) / 2

    total_pos_samples = tp + fn
    total_neg_samples = tn + fp
    hours = signal_time / 360000

    # False alarm rate per hour
    far = (fpr * total_neg_samples) / hours if hours > 0 else 0
    # Miss rate per hour
    mr = (fnr * total_pos_samples) / hours if hours > 0 else 0
    n_samples = np.sum(cm)
    gain = classifiers.cost_fn(cm=cm) / n_samples if n_samples > 0 else 0

    return auc, precision, recall, specificity, f1, far, mr, gain


def sliding_window_confidence(ts, y, model, **kwargs):
    default_kwargs = {'predict': False, 'title': None, 'label': None, 'model_name': '', 'window_size': 40, 'signal_thresh': 1.04, 'freq': 100, 'step': 1, 'method': 'max', 'pad': False} 
    kwargs = {**default_kwargs, **kwargs}
    method = kwargs['method']
    pad = kwargs['pad']
    step = kwargs['step'] * kwargs['freq']
    freq = kwargs['freq']
    window_size = int(kwargs['window_size']*freq)
    signal_thresh = kwargs['signal_thresh']
    start_time = time.time()
    ts = np.array(ts)
    n = len(ts)
    pad_size = (window_size - (n % window_size)) % window_size if pad else 0
    if pad_size:
        ts = np.pad(ts, (0, pad_size), mode='constant', constant_values=0)
        n = len(ts)
    
    conf_map = np.zeros(n, dtype=np.float32) if method == 'mean' else np.full(n, -np.inf, dtype=np.float32)
    count_map = np.zeros(n, dtype=np.uint16) if method == 'mean' else None
    
    confidence_scores = []
    indices = []
    X = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        if end > n:
            break  # Ensure we don't exceed array bounds
        window = np.array(ts[start:end]).reshape(1, -1)
        # filter out windows with max < signal_thresh
        # f1 = int(freq*kwargs['prefall']) if 'prefall' in kwargs and kwargs['prefall'] is not None else freq
        # f2 = f1 + freq
        above_thresh = np.max(window) >= signal_thresh
        indices.append((start, end, above_thresh))
        X.append(window)
    
    if X:
        confidence_scores = model.predict_proba(np.vstack(X))[:, 1]  
    
        for (start, end, above_thresh), score in zip(indices, confidence_scores):
            if method == 'max' and above_thresh:
                conf_map[start:end] = np.maximum(conf_map[start:end], score)
            elif method == 'mean' and above_thresh:
                conf_map[start:end] += score
                count_map[start:end] += 1
    
    if method == 'mean':
        nonzero_indices = count_map > 0
        conf_map[nonzero_indices] /= count_map[nonzero_indices]
    
    conf_map[conf_map == -np.inf] = 0  # Replace untouched values with 0 for max method
    conf_map[:len(ts) - pad_size] if pad_size else conf_map
    # smoothen the confidence map
    # conf_map = np.convolve(conf_map, np.ones(100)/100, mode='same')
    stop_time = time.time()
    total_time = 1000000*(stop_time - start_time)/conf_map.shape[0]

    return conf_map, total_time

def compute_confidence(ts, model):
    """Generate confidence scores for each model.
    Over the whole signal using vectorization
    """
    start_time = time.time()
    num_samples = len(ts)
    chunk_size = 4000

    # Ensure num_samples is a multiple of chunk_size
    # pad with zeros if necessary
    if num_samples % chunk_size != 0:
        pad = chunk_size - num_samples % chunk_size
        padded_ts = np.pad(ts, (0, pad), 'constant')
    else:
        padded_ts = ts

    num_chunks = len(padded_ts) // chunk_size
    padded_ts = padded_ts.reshape(num_chunks, chunk_size)
    c = model.predict_proba(padded_ts)[:, 1]
    stop_time = time.time()
    expanded_c = np.repeat(c, chunk_size)
    expanded_c = expanded_c[:num_samples]
    total_time = 1000000*(stop_time - start_time)/expanded_c.shape[0]
    return expanded_c, total_time

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