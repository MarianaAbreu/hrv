
import math

import numpy as np
import pandas as pd
from scipy.signal import welch

# from processing import utils_ecg_processing as uep


def pointecare_feats(nn):
    x1 = np.asarray(nn[:-1])
    x2 = np.asarray(nn[1:])

    # SD1 & SD2 Computation
    sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
    sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

    csi = sd2/sd1

    csv = math.log10(sd1*sd2)

    return sd1, sd2, csi, csv


def katz_fractal_dim(nn):
    """
    http://tux.uis.edu.co/geofractales/articulosinteres/PDF/waveform.pdf
    :param nn:
    :return:
    """

    d_kfd = np.max([((1-j)**2+(nn[1]-nn[j])**2)**0.5 for j in range(len(nn))])
    l_kfd = np.sum([(1+(nn[i]-nn[i+1])**2)**0.5 for i in range(len(nn)-1)])

    return np.log10(l_kfd)/np.log10(d_kfd)


def diag_det_lmax(rr):
    """Auxiliary function to RQA

    Args:
        rr (array): heaviside function of nni matrix

    Returns:
        values: determinant and long diagonal
    """
    dim = len(rr)
    assert dim == len(rr[0])
    return_grid = [[] for total in range(2 * len(rr) - 1)]
    for row in range(len(rr)):
        for col in range(len(rr[row])):
            return_grid[row + col].append(rr[col][row])
    diags = [i for i in return_grid if (0.0 not in i) & (len(i) > 2)]
    if len(diags) >= 1:
        diag_points = np.sum(np.hstack(diags))

        long_diag = np.max([len(diag) for diag in diags])
    else:
        diag_points = 0
        long_diag = 0

    return diag_points/(np.sum(rr)), long_diag


def lam_calc(rr):
    dim = len(rr)
    assert dim == len(rr[0])
    return_grid = [[] for total in range(2 * len(rr) - 1)]
    for row in range(len(rr.T)):
        idx_v = np.argwhere(np.diff(rr.T[row]) == -1).reshape(-1)
        print(idx_v)
    aaa

    return lam


def rqa(nni):
    """
    Recurrent Quantification Analysis
    rr (array): heaviside function of nni matrix
    rec = ratio between sum of rr and square of nni length
    :param nni:
    :return:
    """

    rr = np.zeros((len(nni), len(nni)))

    for i in range(len(nni)):
        for j in range(len(nni)):
            rr[i,j] = abs(nni[j]-nni[i])
    rr = np.heaviside((rr.mean()-rr), 0.5)

    rec = rr.sum()/(len(nni)**2)

    det, lmax = diag_det_lmax(rr)

    return rec, det, lmax


def time_domain(nni_df):
    """HRV time domain measures

    Args:
        nni (array): nni values
        sampling_rate (int, optional): _description_. Defaults to ms.

    Returns:
        _type_: _description_
    """
    nni = nni_df['nni']
    if 'norm_nni' in nni_df.columns:
        norm_nni = nni_df['norm_nni']
    else:
        norm_nni = nni[:]
    if nni.mean() > 1.5:
        th50 = 50
    else:
        th50 = 0.05
    # use nni to calculate nn50 and pnn50
    nni_diff = np.diff(nni)
    nntot = len(nni_diff)
    # NN50 - successive RR intervals that differ by more than 50 ms
    nn50 = len(np.argwhere(abs(nni_diff) > th50))
    # PNN50 - Percentage of successive RR intervals that differ by more than 50 ms
    pnn50 = 100*(nn50 / nntot)
    # SDNN - standard deviation of nn intervals
    sdnn = norm_nni.std()
    # RMSSD - Root mean square of successive RR interval differences
    rmssd = (np.diff(norm_nni)**2).mean() ** 0.5
    # histogram, index 0 contains heights where index 1 contains widths
    nn_hist = np.histogram(norm_nni)
    # HRV triangular - Integral of the density of the RR interval histogram divided by its height
    tri_hist = len(norm_nni) / np.max(nn_hist[0])
    # Baseline width of RR interval histogram
    tinn = norm_nni.max() - norm_nni.min()

    return rmssd, sdnn, nn50, pnn50, tri_hist, tinn


def spectral(nni, sampling_rate=1):

    frequencies, powers = welch(nni, fs=sampling_rate, scaling='density')
    very_low_freq = np.argwhere((frequencies > 0.003) & (frequencies < 0.04)).reshape(-1)
    low_freq_band = np.argwhere((frequencies > 0.04) & (frequencies < 0.15)).reshape(-1)
    high_freq_band = np.argwhere((frequencies > 0.15) & (frequencies < 0.4)).reshape(-1)

    total_pwr = np.sum(powers)

    lf_pwr = np.sum(powers[low_freq_band])/total_pwr
    hf_pwr = np.sum(powers[high_freq_band])/total_pwr

    return lf_pwr, hf_pwr, lf_pwr/hf_pwr


def get_diff(sig, window):

    diff_sig = sig.diff().abs().diff().abs()

    window_time = pd.date_range(sig.index[0], sig.index[-1], freq=str(window) + 'S')

    new_diff = pd.DataFrame([diff_sig.between_time(window_time[i].time(),
                                                   window_time[i + 1].time()).mean()
                             for i in range(len(window_time) - 1)], index=window_time[1:])

    return new_diff


def hrv_features(nni=None, sig=None, FS=256, time_=True, pointecare_=True, diff=True, sig_len=250, names=False):
    """
    names = ['rms_sd', 'sd_nn', 'mean_nn', 'nn50', 'pnn50', 'var', 'sd1',
             'sd2', 'csi', 'csv', 'rec', 'det', 'lmax']
    :param nni: norm_nni and nni
    :param time_:
    :param spectral_:
    :return:
    """
    if names:
        return ['rmssd', 'sdnn', 'mean_nn', 'nn50', 'pnn50', 'tri_hist', 'tinn', 'lf_pwr', 'hf_pwr', 'lf_hf', 'sd1',
                'sd2', 'csi', 'csv', 'kfd', 'rec', 'det', 'lmax']
    if nni is None:
        print('Here')
        # nni_sig, nni_ts, nni_good = uep.get_nn_intervals(sig=sig, sampling_rate=FS)
        # nni = pd.Series(nni_sig, index=nni_ts)
    if len(nni) < sig_len:
        print(f'This segment is too short to calculate HRV features, {len(nni)}s')
        return np.zeros(18)

    if time_:
        # extract rmssd, sdnn, nn50, pnn50, var
        rmssd, sdnn, nn50, pnn50, tri_hist, tinn = time_domain(nni)
    try:
        cols = nni.columns
    except:
        cols = []
    if 'norm_nni' in cols:
        nni = nni['norm_nni'].values
    else:
        nni = nni['nni'].values
    mean_nn = nni.mean()

    if pointecare_:
        sd1, sd2, csi, csv = pointecare_feats(nni)

    lf_pwr, hf_pwr, lf_hf = spectral(nni)
    kfd = katz_fractal_dim(nni)
    rec, det, lmax = rqa(nni)

    feats = np.hstack((rmssd, sdnn, mean_nn, nn50, pnn50, tri_hist, tinn, lf_pwr, hf_pwr, lf_hf,
                       sd1, sd2, csi, csv, kfd, rec, det, lmax))
    return feats

