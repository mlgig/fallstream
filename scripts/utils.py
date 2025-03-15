import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.signal import resample
import time
from sklearn.model_selection import train_test_split
from scripts import farseeing, classifiers


def magnitude(arr):
    x, y, z = np.array(arr).T.astype('float')
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude -= min(magnitude)
    return magnitude

def resample_to(arr, old_f, new_f=100):
    new_list = []
    old_len = arr.shape[-1]
    for sample in arr:
        resampled = resample(sample, int(new_f*(old_len/old_f)))
        new_list.append(resampled)
    new_arr = np.array(new_list)
    return new_arr

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
    default_kwargs = {'test_size': 0.3, 'random_state': 0, 'clip': False, 'split': True, 'show_test': False, 'window_size': 60, 'segment_test': True, 'prefall': None, 'new_freq': 100, 'augment_data': None}
    kwargs = {**default_kwargs, **kwargs}
    df = dataset.load(clip=kwargs['clip'])
    subjects = df['SubjectID'].unique()
    if kwargs['split']==False:
        X, y = dataset.get_X_y(df, **kwargs)
        return X, y
    else:
        train_set, test_set = train_test_split(subjects, test_size=kwargs['test_size'], random_state=kwargs['random_state'])
        if kwargs['show_test']:
            print(f'Test set -> {len(test_set)} of {len(subjects)} subjects: {test_set}.')
        test_df = df[df['SubjectID']==test_set[0]]
        X_train, X_test, y_train, y_test = split_df(df, dataset, test_set, **kwargs)

        print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
        ([ADLs, Falls])", np.bincount(y_train))
        if kwargs['segment_test']:
            print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
            ([ADLs, Falls])", np.bincount(y_test))
        else:
            print(f"Test set: X: {len(X_test)}, y: {len(y_test)}")

        return X_train, X_test, y_train, y_test
    
def visualise_all_metrics(df, metrics, name='farseeing'):
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=400,
                            layout='tight', sharex=True,
                            sharey='row')
    for i, metric in enumerate(metrics):
        ax = axs.flat[i]
        sns.boxplot(data=df, x='model', y=metric, width=0.5,
                    ax=ax, linewidth=1, palette='tab10')
        ax.grid()
        metric = 'runtime (ms)' if metric == 'runtime' else metric
        metric = 'FAR (per hour)' if metric == 'false alarm rate' else metric
        metric = 'MR (per hour)' if metric == 'miss rate' else metric
        # add padding to title top and reduce padding below
        ax.set_title(metric, pad=2)
        # ax.set_title(metric)
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
    if y != -1:
        ax.axvline(x=y, color='red', linestyle='--', label='Fall')
    ax.legend()
    if kwargs['title'] is not None:
        title = kwargs['title']
        ax.set_title(f'Subject {title} | TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}. Time/sample: {kwargs["ave_time"]:.2f} ms. {kwargs["model_name"]}')
    else:
        ax.set_title(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}. Time/sample: {kwargs["ave_time"]:.2f} ms. {kwargs["model_name"]}')
    plt.show()

def detect(ts, fall_point, c, **kwargs):
    default_kwargs = {'confidence_thresh': 0.5, 'tolerance':20}
    kwargs = {**default_kwargs, **kwargs}
    """Obtain TP, FP, TN, and FN based on confidence mapping."""
    window_size = kwargs['window_size']
    step_size = kwargs['step']
    # Get high confidence regions
    high_conf = get_high_confidence_regions(ts, c, **kwargs)
    tolerance = kwargs['tolerance']
    # n_samples = len(ts) // (kwargs['window_size']*kwargs['freq'])
    n_samples = ((len(ts) - window_size) // step_size) + 1
    TP, FP, TN, FN = 0, 0, 0, 0
    if fall_point==-1:
        fall_range = np.arange(len(ts), len(ts) + 2)
    else:
        left_all = (fall_point - 100) - 100*(window_size + tolerance)
        right_all = (fall_point + 100*(window_size-1)) + 100*(tolerance)
        fall_range = np.arange(left_all, right_all)
    delay = 0 if fall_point==-1 else tolerance
    if high_conf is None:
        TP, FP, TN, FN = 0, 0, n_samples-1, 1
    else:
        for h in high_conf:
            detection_range = range(h, h + 100*(window_size + tolerance))
            IOU = iou(detection_range, fall_range)
            if IOU != 0:
                TP = 1
                delay = (h - fall_point) / 100
            else:
                FP += 1
        FN = 1 if TP == 0 else 0
        TN = n_samples - TP - FP - FN
    FN = 0 if fall_point==-1 else FN
    TP = 0 if fall_point==-1 else TP
    cm = np.array([[TN, FP], [FN, TP]])
    
    return cm, high_conf, delay

def get_high_confidence_regions(ts, c, **kwargs):
    """Get high confidence regions based on threshold."""

    # Combine signal threshold and confidence threshold
    thresh = kwargs['confidence_thresh']
    # high_conf = np.where((c >= thresh) | (c>=max(c)))[0]
    high_conf = np.where((c >= thresh))[0]
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

def compute_metrics(cm, signal_time, **kwargs):
    """Compute metrics based on TP, FP, TN, FN, and signal time."""   
    # Compute metrics
    tn, fp, fn, tp = cm.ravel()
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
    gain = classifiers.cost_fn(cm=cm, **kwargs) / (n_samples * 1000)

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

    stop_time = time.time()
    total_time = 1000000*(stop_time - start_time)/conf_map.shape[0]

    return conf_map, total_time