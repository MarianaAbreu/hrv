
import biosppy.signals.tools as st
import biosppy.signals.ecg as bse
import biosppy.utils as bu
import numpy as np
import os
import pandas as pd
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pickle
from processing import utils_signal_processing as usp


def ecg_derived_respiration(signal=None, rpeaks=None, sampling_rate=1000.):
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
    if rpeaks is None:
        order = int(0.3 * sampling_rate)
        ecg_filtered, _, _ = st.filter_signal(signal=signal,
                                        ftype='FIR',
                                        band='bandpass',
                                        order=order,
                                        frequency=[3, 45],
                                        sampling_rate=sampling_rate)

        # segment
        rpeaks, = bse.hamilton_segmenter(signal=ecg_filtered, sampling_rate=sampling_rate)

        # correct R-peak locations
        rpeaks, = bse.correct_rpeaks(signal=ecg_filtered,
                                rpeaks=rpeaks,
                                sampling_rate=sampling_rate,
                                tol=0.05)
    else: 
        ecg_filtered = np.array(signal)
    # find the amplitude values of the rpeaks, based on the filtered signals and the peaks location
    ecg_peaks = [ecg_filtered[e] for e in range(len(ecg_filtered) - 1) if e in rpeaks]

    # quadratic interpolation of the peaks
    interp = interp1d(rpeaks, ecg_peaks, kind='quadratic')

    # perform the quadratic interpolation above between the first and last peak.

    # create a discrete time between the first and last peaks

    # perform the interpolation
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

    if len(beats) < 3:
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
    if len(ts_resp) != len(derived):
        ts_resp = ts_resp[:len(derived)]

    # plot

    # output
    args = (ts_resp, derived, zeros, ts_rate, rate)
    names = ('ts', 'filtered', 'zeros', 'resp_rate_ts', 'resp_rate')

    return bu.ReturnTuple(args, names)


def mean_template_find(sig, sampling_rate=256):

    filt_sig = \
    st.filter_signal(sig, ftype='FIR', frequency=[3, 45], band='bandpass',
                                        order=sampling_rate // 3, sampling_rate=sampling_rate)['signal']
    rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
    rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)
    templates = bse.extract_heartbeats(filt_sig, rpeaks,
                                                       sampling_rate=sampling_rate)['templates']
    mean_template = usp.clean_outliers(templates).mean(axis=0)
    return mean_template


def choose_ecg_channel(sig, mean_template_, sampling_rate=256):

    pear_ = 0
    best = 0
    max_len = 200000 if (len(sig) > 200000) else (len(sig))
    for cl in ['ecg', 'ECG']: #sig.columns:

        filt_sig = st.filter_signal(sig[cl].values[:max_len], ftype='FIR', frequency=[3, 45], band='bandpass',
                                                       order=sampling_rate//3,sampling_rate=sampling_rate)['signal']
        rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
        rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)
        templates = bse.extract_heartbeats(filt_sig, rpeaks,
                                                                  sampling_rate=sampling_rate)['templates']
        mean_template = usp.clean_outliers(templates).mean(axis=0)
        pear_res = pearsonr(mean_template, mean_template_)[0]
        print(cl, pear_res)
        if pear_res > pear_:
            best = cl
            pear_ = float(pear_res)

        #plt.plot(mean_template, label = cl)
        #plt.legend()
    print('BEST', best)
    return best


def _ecg_correction(filt_sig, rpeaks, mean_template, min_rp, max_rp, idxmin, idxmax):
    """
    This function calculates the euclidean distance between all signal templates and the mean template.
    The signal templates with distances above the threshold are correlated point by point with the mean
        template, to find the maximum correlation using pearson correlation.
    The template is replaced by the mean template at the point of maximum correlation.
    The previous and following points are replaced by the first and last mean_template values, respectively.
    The corrected signal is returned.

    :param filt_sig: ecg signal
    :param rpeaks: r peak locations
    :param mean_template: average template with 600ms and r peak at 200ms
    :param min_rp: distance to r peak
    :param max_rp: distance from r peak to end
    :param idxmin: start of cropped template
    :param idxmax: end of cropped template
    :return: corrected signal
    """
    delta = idxmax - idxmin # cropped mean_template length
    if rpeaks[0] - min_rp < 0:
        filt_sig = np.hstack((np.zeros(min_rp-rpeaks[0]) + filt_sig[0], filt_sig))
        rpeaks += (min_rp-rpeaks[0])
    if rpeaks[-1] + max_rp > len(filt_sig):
        filt_sig = np.hstack((filt_sig,np.zeros(max_rp - (len(filt_sig)- rpeaks[-1])) + filt_sig[-1]))


    #euclidean distance between the signal around r peak and the mean template
    euc_dist = [np.linalg.norm(mean_template - filt_sig[rpi - min_rp: rpi + max_rp]) for rpi in rpeaks]
    #threshold distance
    threshold_dist = np.mean(euc_dist) + np.std(euc_dist)

    for dl in np.argwhere(euc_dist > threshold_dist).reshape(-1):
        # template to correct
        seg = filt_sig[rpeaks[dl] - min_rp: rpeaks[dl] + max_rp]
        # padding between and after
        seg_pad = np.hstack((seg[0] + np.zeros(delta), seg, seg[-1] + np.zeros(delta)))

        # pearson correlation and maximum location
        pr = np.argmax([pearsonr(mean_template[idxmin:idxmax], seg_pad[i:i + delta])[0]
                        for i in range(int(delta / 2), len(seg_pad) - delta)]) + int(delta / 2)

        # replacing by mean template
        seg_pad[:pr - idxmin] = np.zeros(pr - idxmin) + mean_template[0]
        seg_pad[pr - idxmin: pr + delta] = mean_template[:idxmax]
        if pr + idxmax < len(seg_pad):
            seg_pad[pr + idxmax:] = np.zeros(len(seg_pad) - pr - idxmax) + mean_template[-1]

        filt_sig[rpeaks[dl] - min_rp: rpeaks[dl] + max_rp] = seg_pad[delta:-delta]

    return filt_sig


def ecg_correction(sig, rpeaks = None, mean_template= None, sampling_rate=1000, show=False):

    min_rp = int(0.2 * sampling_rate)
    max_rp = int(0.4 * sampling_rate)
    idxmin = int(0.1 * sampling_rate)
    idxmax = int(0.4 * sampling_rate)
    delta = idxmax - idxmin

    filt_sig = st.filter_signal(sig, ftype='FIR', frequency=[5, 20], band='bandpass', order=150,
                                                   sampling_rate=sampling_rate)['signal']
    if rpeaks is None:
        rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
        rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)

    if mean_template is None:
        templates = bse.extract_heartbeats(filt_sig, rpeaks,
                                                              sampling_rate=sampling_rate)['templates']
        mean_template = usp.clean_outliers(templates).mean(axis=0)

    filt_sig = _ecg_correction(filt_sig, rpeaks, mean_template, min_rp, max_rp, idxmin, idxmax)

    rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
    rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)

    return filt_sig, rpeaks


def get_nn_intervals(sig=None, rpeaks=None, sampling_rate=256, resampling =False):
    """
    Get NN intervals = RR intervals
    This returns in milisseconds
    :param sig:
    :param rpeaks:
    :param sampling_rate:
    :param resampling:
    :return:
    """

    if rpeaks is None:
        filt_sig = st.filter_signal(sig, ftype='FIR', frequency=[5, 20], band='bandpass',
                                                       order=sampling_rate//3, sampling_rate=sampling_rate)['signal']
        rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
        rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)

    rpeaks = np.array(rpeaks, dtype=float)
    if len(rpeaks) < 2:
        return []
    # difference of r peaks converted to ms
    nni = (1000 * np.diff(rpeaks))/sampling_rate
    # only accept nni values within the range 300 - 1500 (200 - 40 bpm)
    nni_ts = rpeaks[1:]
    nni_idx_good = np.argwhere((nni > 300) & (nni < 1500)).reshape(-1)
    if resampling:
        nni = resample(nni, len(nni) * 4)

    return nni, nni_ts, nni_idx_good


def get_hr_rate(rpeaks, sampling_rate=256):

    hr_idx, hr = st.get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
    return hr_idx, hr

    
def get_ecg_data(sig, sampling_rate = 256, resp=False, hr=False, nni=False):
    
    columns_ecg = ['ecg']
    if resp:
        columns_ecg += ['resp']
        columns_ecg += ['rr']
    if hr:
        columns_ecg += ['rpeaks']
        columns_ecg += ['hr']
    if nni:
        columns_ecg += ['nni']
    
    ecg_df = pd.DataFrame(index=sig.index, columns=columns_ecg)
    ecg_df.is_copy = None
    
    filt_sig = st.filter_signal(sig, ftype='FIR', frequency=[5, 30], band='bandpass',
                                                       order=sampling_rate//3, sampling_rate=sampling_rate)['signal']
    
    ecg_df['ecg'] = filt_sig
    
    if hr:
        rpeaks, = bse.hamilton_segmenter(filt_sig, sampling_rate=sampling_rate)
        rpeaks, = bse.correct_rpeaks(filt_sig, rpeaks, sampling_rate, tol=0.05)
        hr_idx, hr_sig = get_hr_rate(rpeaks, sampling_rate = sampling_rate)
        ecg_df['hr'][hr_idx] = hr_sig
        ecg_df['rpeaks'][rpeaks] = ecg_df['ecg'][rpeaks]
    
    if nni:
        nn_, nn_ts, _ = get_nn_intervals(ecg_df['ecg'], sampling_rate = sampling_rate)
        # nn_range = np.arange(0, len(ecg_df)-len(ecg_df)%len(nn_), len(ecg_df)//len(nn_))
        ecg_df['nni'][nn_ts] = nn_
    
    if resp:
        if hr:
            resp_ = ecg_derived_respiration(ecg_df['ecg'], rpeaks, sampling_rate = sampling_rate)
        else:
            resp_ = ecg_derived_respiration(ecg_df['ecg'], sampling_rate = sampling_rate)
        resp_range = np.asarray((resp_['ts'])*sampling_rate, dtype='int')
        resp_rate_range = np.asarray(resp_['resp_rate_ts']*sampling_rate, dtype='int')
        ecg_df['rr'][resp_rate_range] = resp_['resp_rate']
        ecg_df['resp'][resp_range] = resp_['filtered']
    
    return ecg_df


def dissimilarity(sig, sampling_rate=256, correction = True):

    mean_template = pickle.load(open('mean_template_ecg', 'rb'))

    if correction:
        sig, rpeaks = ecg_correction(sig, sampling_rate=sampling_rate)
    else:
        rpeaks = None


def ecg_processing(sig, sampling_rate=256, correction=True, hr=False):

    if correction:
        sig, rpeaks = ecg_correction(sig, sampling_rate=sampling_rate)
    else:
        rpeaks = None

    nni = get_nn_intervals(sig, rpeaks, sampling_rate=sampling_rate)

    if hr:
        hr_idx, hr = get_hr_rate(rpeaks, sampling_rate=sampling_rate)

    return nni


if __name__ == '__main__':

    dir = 'E:\\Patients_HEM\\PAT_400\\signals'
    file = os.listdir(dir)

    sig = pd.read_hdf(dir + os.sep + file[0])['ECG'].values

    dissimilarity(sig)

#sig1 = pd.read_hdf(dir)['ECG'].values

#ecg_processing(sig1, sampling_rate=1000)