
import os


from datetime import timedelta
from biosppy.signals.tools import filter_signal
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, 'C:\\Users\\Mariana\\PycharmProjects\\mapme')
from read_files import read_bitalino as rb

"""
    Adapted from TAE code
"""


def get_freqs(segment, fs, max_freq = None):

    """
    Returns positive frequencies between 0 and max frequency. These frequencies don't vary with the segment values.
    """

    if not max_freq:
        max_freq = fs//2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(segment.size, time_step)
    positive_freqs = freqs[np.where(freqs > 0)]
    idx = np.argsort(positive_freqs[np.where(positive_freqs <= max_freq)])

    return freqs[idx], idx


def preprocess(signal, fs, window_seg, max_freq = 45, filter = True):
    """
    receives signal and sampling frequency and window_seg
    filters signal and calculates frequencies 
    """

    window = window_seg * fs

    if window > len(signal):
        print('Signal too short for segmentation, choose another window size')
        return [], []

    if filter:
        signal = filter_signal(signal, ftype='FIR', frequency=[3, max_freq], band='bandpass', sampling_rate=fs, order = fs//4)['signal']

    #use first window to calculate frequencies
    seg_data = signal[:window] 
    _, idx = get_freqs(seg_data, fs, max_freq)
    # remove last window from filter data if it does not reach the window size
    data_cropped = signal[:window * (len(signal)//window)]

    # calculate and stack all power spectrums, with idx, it only saves the frequency values which are between 0 and max_freq
    ps = np.vstack([(np.abs(np.fft.fft(segment))**2)[idx] for segment in data_cropped.reshape(-1, window)])
    time_ix = np.arange(0, len(data_cropped), window)

    return ps, time_ix



def clustering(X, type='Affinity', n_clusters=2):

    if type == 'Affinity':
        from sklearn.cluster import AffinityPropagation
        model = AffinityPropagation(damping=0.5)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        
    
    elif type == 'Agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters = n_clusters)
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)

    elif type == 'Birch':
        from sklearn.cluster import Birch
        model = Birch(threshold=0.01, n_clusters=n_clusters)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        
    return model, clusters, yhat


def plot_clustering(df_ecg, times, window_seg, clusters):    

    # find a random color for each cluster
    from random import randint
    import matplotlib.pyplot as plt

    colors_list = []
    for i in range(len(clusters)):
        colors_list.append('#%06X' % randint(0, 0xFFFFFF))

    # plot original data where the color of the segment is defined by the clustering algorithm
    plt.figure(figsize=(20,5))
    plt.title('Birch')

    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        for ti in time[row_ix]:
            plt.plot(df_ecg.between_time(df_ecg.index[ti].time(), (df_ecg.index[ti] + timedelta(seconds=window_seg)).time()), color = colors_list[cluster], label = str(cluster))
    #plt.legend()
    print('Number of clusters ', len(clusters))


directory = "F:\Patients_HSM\Patient107\Bitalino_24H"

bit_files = sorted(os.listdir(os.path.join(directory)))

sig = pd.read_hdf(os.path.join(directory,bit_files[0]))

sig_times = sorted(np.where(np.diff(sig.index).astype(dtype='timedelta64[s]')!=timedelta(seconds=0)))[0]

start = 0
all_ps, all_times = [], []
for st in sig_times: 
    end = int(st) + 1 
    sig_cropped = sig['ECG'].values[start:end]
    ps, time_ix = preprocess(sig_cropped, fs = 1000, window_seg=20)
    if ps != []:    
        all_ps += [ps]
        all_times += [[start, end]]
        
    start = int(st) + 1

model, clusters, yhat = clustering(np.vstack(all_ps), type='Birch', n_clusters=5)
plot_clustering(sig, all_times, 10, clusters)
print('here')
    

#preprocess(sig['ECG'].values)
