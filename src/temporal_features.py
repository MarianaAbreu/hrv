# built-in
import json
import os
from typing import Dict

# third-party
import numpy as np
import pandas as pd

# local
from biosppy import utils


def signal_temp(signal=None, FS=1, names=False):
    """Compute various metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    FS : float
        Sampling frequency
    Returns
    -------
    maxAmp : float
        Signal maximum amplitude.

    minAmp : float
        Signal minimum amplitude.

    max : float
        Signal max value.

    min : float
        Signal min value.

    dist : float
        Length of the signal.

    autocorr : float
        Signal autocorrelation.

    mean_cross : int
        Number of times the sinal crosses the zero axis, after the standarisation of the signal, hence 0 = mean.

    meanadiff : float
        Mean absolute differences.

    medadiff : float
        Median absolute differences.

    mindiff : float
        Min differences.

    maxdiff : float
        Maximum differences.

    sadiff : float
        Sum of absolute differences.

    meandiff : float
        Mean of differences.

    meddiff : float
        Median of differences.

    temp_centroid : float
        Temporal centroid.

    total_energy : float
        Total energy.

    minpeaks : int
        Number of minimum peaks.

    maxpeaks : int
        Number of maximum peaks.

    temp_dev : float
        Temporal deviation.

    counter : int
        Length of the signal.

    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.
    """
    if names:
        return ['maxAmp', 'minAmp', 'max', 'min', 'dist', 'mean_cross', 'meanadiff', 'medadiff', 'mindiff', 'maxdiff',
                'sadiff', 'meandiff', 'meddiff', 'minpeaks', 'maxpeaks', 'temp_dev']

    # check inputs
    # if signal is None or signal == []:
    #     print("Signal is empty.")

    # ensure numpy
    signal = np.array(signal)
    # assert len(signal) > 1, 'Signal size < 1'
    sig_diff = np.diff(signal)
    mean = np.mean(signal)
    maxAmp = np.max(np.abs(signal - mean))
    minAmp = np.min(np.abs(signal - mean))
    max = np.max(signal)
    min = np.min(signal)
    dist = np.sum([np.sqrt(1+df**2) for df in sig_diff])
    # autocorrelation
    # autocorr = np.correlate(signal, signal)[0]

    # zero_cross = len(np.where(np.diff(np.sign(signal)))[0])
    stand_signal = (signal-signal.mean())/signal.std()
    mean_cross = len(np.where(np.diff(np.sign(stand_signal)))[0])
    meanadiff = np.mean(np.abs(sig_diff))
    medadiff = np.median(np.abs(sig_diff))
    mindiff = np.min(np.abs(sig_diff))
    # max absolute differences
    maxdiff = np.max(np.abs(sig_diff))
    # sum of absolute differences
    sadiff = np.sum(abs(sig_diff))
    # mean of differences
    meandiff = np.mean(sig_diff)
    # median of differences
    meddiff = np.median(sig_diff)
    # total energy
    time = range(len(signal))[1:]
    time = [float(x) / FS for x in time]
    signal = signal[1:]
    # total_energy = np.sum(np.array(signal)**2)/(time[-1]-time[0])
    # number of minimum peaks
    minpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd]<0 and sig_diff[nd+1]>0)])/len(signal)
    # number of maximum peaks
    maxpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd+1]<0 and sig_diff[nd]>0)])/len(signal)
    # temporal deviation
    temp_dev = (1/np.sum(signal)) * np.sum((signal[:] - signal[1])/np.array(time))
    #counter = len(signal)

    #return utils.ReturnTuple(tuple(args), tuple(names))
    return maxAmp, minAmp, max, min, dist, mean_cross, meanadiff, medadiff, mindiff, maxdiff, sadiff, meandiff, meddiff, \
           minpeaks, maxpeaks, temp_dev
