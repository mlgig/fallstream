import os
import mat73
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal import resample
from sklearn.model_selection import train_test_split

def load(clip=False):
    farseeing = pd.read_pickle(r'data/farseeing.pkl').reset_index().drop(columns=['index'])
    return farseeing

def plot_sample(df, ax):
    arr = np.array(df['accel'][1])
    # fig, ax = plt.subplots(figsize=(9,3), dpi=400)
    ax.plot(np.array(arr))
    ax.set_ylabel('Accel (g)')
    ax.set_title('FARSEEING Sample')
    ax.set_xlabel('')
    # ax.set_xticklabels([])
    rect = patches.Rectangle((59000, -1.5), 2000, 4, linewidth=1,  facecolor='CornflowerBlue', alpha=0.5, zorder=10)
    ax.add_patch(rect)
    # ax.legend(['x', 'y', 'z'])
    # sns.despine()
    # plt.savefig('figs/farseeing_signal.pdf', bbox_inches='tight')
    # plt.show()

def sample_adls(X_train, y_train, adl_samples):
    # Group falls and ADLs together to sample from ADLs
    X_train_y_train = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
    X_train_falls = X_train_y_train[X_train_y_train[:,-1]==1]
    X_train_ADLs = X_train_y_train[X_train_y_train[:,-1]==0]
    np.random.seed(5)
    rng = np.random.default_rng()
    rng.shuffle(X_train_ADLs)
    # Select <adl_samples> ADLs
    X_train_ADLs_with_labels = X_train_ADLs[:adl_samples]
    X_train_rejoined = np.concatenate([X_train_ADLs_with_labels,
                                    X_train_falls], axis=0)
    rng.shuffle(X_train_rejoined) # shuffle again
    # recover X_train and y_train
    X_train = X_train_rejoined[:,:-1]
    y_train = X_train_rejoined[:,-1].astype(int)
    return X_train, y_train

def expand_for_ts(X_train, X_test):
    X_train = np.array(X_train)[:, np.newaxis, :]
    X_test = np.array(X_test)[:, np.newaxis, :]
    return X_train, X_test

def get_X_y(df, **kwargs):
    default_kwargs = {'keep_fall': False, 'window_size': 60, 'fall_dur': 1, 'spacing': 1, 'fall_pos': 'fixed', 'multiphase': False, 'thresh': 1.1, 'step': 1}
    kwargs = {**default_kwargs, **kwargs}
    no_segmentation = kwargs['keep_fall'] and not kwargs['segment_test']
    if kwargs['multiphase']:
        prefalls = [kwargs['prefall']] if kwargs['prefall'] else [1]
    else:
        if kwargs['spacing'] == 'na':
            prefalls = [kwargs['window_size']//2]
        else:
            n_fall_samples = kwargs['window_size']//kwargs['spacing']
            if kwargs['fall_pos'] == "random":
                # let the fall position be random, between 2 and kwargs['window_size']-2, but with a random seed
                np.random.seed(5)
                prefalls = np.random.randint(2, kwargs['window_size']-2, n_fall_samples)          
            else:
                prefalls = np.arange(2, kwargs['window_size']-2, kwargs['spacing'])
    X = []
    y = []
    for i, row in df.iterrows():
        fall_point = row['fall_point']
        freq = row['freq']
        accel = row['accel_mag']
        if kwargs['keep_fall']:
            prefall_signal = accel
        else:
            # Take out the fall signal
            for prefall in prefalls:
                postfall = kwargs['window_size'] - prefall - kwargs['fall_dur']
                start = int(fall_point-(freq*prefall))
                end = int(start+(freq*kwargs['window_size']))
                fall_signal = accel[start:end]
                if freq==20:
                    # resample to 100 Hz
                    new_length = int(100*len(fall_signal)/20)
                    fall_signal = resample(fall_signal, new_length)
                X.append(fall_signal)
                y.append(1)
            # leave out the fall signal with a gap equal to the maximum prefall
            gap = int(fall_point-(freq*max(prefalls)))
            prefall_signal = accel[:gap]
            fall_point = None
        if no_segmentation:
            X.append(prefall_signal)
            y.append(fall_point)
        else:
            # Segment prefall signal if long enough
            if len(prefall_signal) >= 2*freq*kwargs['window_size']:
                cw, targets = get_adls(prefall_signal,freq,fall_point, **kwargs)
                X.extend(cw)
                y.extend(targets)
    if not no_segmentation:
        X = np.array(X)
        y = np.array(y, dtype='uint8')
    return X, y

def get_adls(ts, freq=100, fall_point=None, **kwargs):
    """
    Get candidate windows for ADLs. Loop through ts with a step of 5 seconds. Select windows with max value >= thresh.
    Any window where fall_point is within the window is labelled as a fall.
    """
    ts = np.array(ts).flatten()
    X = []
    y = []
    sample_window_size = int(freq*kwargs['window_size'])
    freq_100_length = int(100*(kwargs['window_size']))
    # resample to match signals of 100Hz if necessary
    resample_to_100Hz = freq_100_length != sample_window_size
    for j in range(0, len(ts), freq*kwargs['step']):
        potential_window = ts[j:j+sample_window_size]
        if len(potential_window) < sample_window_size:
            break
        search_window = potential_window[freq:2*freq] if kwargs['multiphase'] else potential_window
        if max(search_window) >= kwargs['thresh']:
            if resample_to_100Hz:
                potential_window = resample(potential_window, freq_100_length)
            X.append(potential_window)
            if fall_point is not None and fall_point in range(j, j+sample_window_size):
                y.append(1)
            else:
                y.append(0)
    return X, y

def get_candidate_windows(ts, freq, target, prefall,
                fall, postfall, thresh=1.08, step=60):
    ts = np.array(ts).flatten()
    X = []
    y = []
    total_duration = prefall + fall + postfall
    sample_window_size = int(freq*total_duration)
    required_length = int(freq*(total_duration))
    freq_100_length = int(100*(total_duration))
    # resample to match signals of 100Hz if necessary
    resample_to_100Hz = freq_100_length != required_length
    for j in range(0, len(ts), freq*step):
        potential_window = ts[j:j+sample_window_size]
        if len(potential_window) < required_length:
            break
        main_window = potential_window[28*freq:29*freq]
        if len(main_window) == freq*step:
            if max(main_window) >= thresh:
                selected_window = potential_window
                if resample_to_100Hz:
                    selected_window = resample(selected_window, freq_100_length)
                X.append(selected_window)
                y.append(target)
    return X, y