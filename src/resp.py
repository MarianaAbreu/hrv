# -*- coding: utf-8 -*-
"""
biosppy.signals.resp
--------------------

This module provides methods to process Respiration (Resp) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np

# local
from . import tools as st
from .. import plotting, utils
from . import ecg as ecg
from scipy import interpolate

def resp(signal=None, sampling_rate=1000., show=True):
    """Process a raw Respiration signal and extract relevant signal features
    using default parameters.

    Parameters
    ----------
    signal : array
        Raw Respiration signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered Respiration signal.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Respiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=[0.1, 0.35],
                                      sampling_rate=sampling_rate)

    # compute zero crossings
    zeros, = st.zero_cross(signal=filtered, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        rate_idx = []
        rate = []
    else:
        # compute respiration rate
        rate_idx = beats[1:]
        rate = sampling_rate * (1. / np.diff(beats))

        # physiological limits
        indx = np.nonzero(rate <= 0.35)
        rate_idx = rate_idx[indx]
        rate = rate[indx]

        # smooth with moving average
        size = 3
        rate, _ = st.smoother(signal=rate,
                              kernel='boxcar',
                              size=size,
                              mirror=True)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_rate = ts[rate_idx]

    # plot
    if show:
        plotting.plot_resp(ts=ts,
                           raw=signal,
                           filtered=filtered,
                           zeros=zeros,
                           resp_rate_ts=ts_rate,
                           resp_rate=rate,
                           path=None,
                           show=True)

    # output
    args = (ts, filtered, zeros, ts_rate, rate)
    names = ('ts', 'filtered', 'zeros', 'resp_rate_ts', 'resp_rate')

    return utils.ReturnTuple(args, names)

def ecg_derived_respiration(signal=None, raw_resp = None, sampling_rate=1000., show=True):
    """Process a raw ECG signal and extract the respiration signal and relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Respiration time axis reference (seconds).
    signal : array
        Respiration derived from ECG signal.
    rpeaks : array
        R-peak location indices.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Respiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(0.3 * sampling_rate)
    ecg_filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=sampling_rate)

    # segment
    rpeaks, = ecg.hamilton_segmenter(signal=ecg_filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg.correct_rpeaks(signal=ecg_filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    #find the amplitude values of the rpeaks, based on the filtered signals and the peaks location
    ecg_peaks = [ecg_filtered[e] for e in range(len(ecg_filtered) - 1) if e in rpeaks]


    #quadratic interpolation of the peaks
    interp = interpolate.interp1d(rpeaks, ecg_peaks, kind='cubic')

    #perform the quadratic interpolation above between the first and last peak.

    #create a discrete time between the first and last peaks

    #perform the interpolation
    resp_signal = interp(np.arange(rpeaks[0],rpeaks[-1]))*[-1]

    ts_resp = np.arange((rpeaks[0] / sampling_rate), (rpeaks[-1] / sampling_rate), 1 / sampling_rate)
    signal = np.array(resp_signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    derived, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=[0.1, 0.35],
                                      sampling_rate=sampling_rate)

    # compute zero crossings
    zeros, = st.zero_cross(signal=derived, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        rate_idx = []
        rate = []
    else:
        # compute respiration rate
        rate_idx = beats[1:]
        rate = sampling_rate * (1. / np.diff(beats))

        # physiological limits
        indx = np.nonzero(rate <= 0.35)
        rate_idx = rate_idx[indx]
        rate = rate[indx]

        # smooth with moving average
        size = 3
        rate, _ = st.smoother(signal=rate,
                              kernel='boxcar',
                              size=size,
                              mirror=True)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_rate = ts[rate_idx]

    if raw_resp is not None:
        raw_resp = raw_resp[rpeaks[0]:rpeaks[-1]]
        _,raw_resp,_,ts_raw_rate, raw_rate = resp(raw_resp,show=False)
        if show:
            plotting.plot_ecg_derived_resp(ts=ts,
                                           raw=raw_resp,
                                           derived=derived,
                                           ecg=ecg_filtered[rpeaks[0]:rpeaks[-1]],
                                           zeros=zeros,
                                           resp_rate_ts=ts_rate,
                                           resp_rate=rate,
                                           raw_rate=raw_rate,
                                           raw_rate_ts=ts_raw_rate,
                                           path=None,
                                           show=True)
    else:
        if show:
            plotting.plot_ecg_derived_resp(ts=ts,
                                           raw=None,
                                           derived=derived,
                                           ecg=ecg_filtered[rpeaks[0]:rpeaks[-1]],
                                           zeros=zeros,
                                           resp_rate_ts=ts_rate,
                                           resp_rate=rate,
                                           raw_rate=None,
                                           raw_rate_ts=None,
                                           path=None,
                                           show=True)


    # plot

    # output
    args = (ts_resp, raw_resp, derived, zeros, ts_rate, rate)
    names = ('ts', 'real','filtered', 'zeros', 'resp_rate_ts', 'resp_rate')



    return utils.ReturnTuple(args, names)
