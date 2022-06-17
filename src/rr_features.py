
import numpy as np
import sys

sys.path.append('C:\\Users\\Mariana\\PycharmProjects\\mapme\\code')

from processing import hrv_features, temporal_features, statistic_features


def rr_features(signal=None, siglen=20, fs=1, names=False):
    """
    Calculates features for respiratory rate, they can be useful in other time series as well
    :param signal:
    :param FS:
    :return:
    """

    if names:
        hrv_names = ['rec', 'det', 'lmax', 'sd1', 'sd2', 'csi', 'csv', 'kfd']
        temp_names = temporal_features.signal_temp(names=True)
        stat_names = statistic_features.signal_stats(names=True)
        return hrv_names + temp_names + stat_names
    if len(signal) < siglen:
        return np.zeros((33))

    rec, det, lmax = hrv_features.rqa(signal)
    sd1, sd2, csi, csv = hrv_features.pointecare_feats(signal)
    kfd = hrv_features.katz_fractal_dim(signal)

    temp_feats = temporal_features.signal_temp(signal, fs)

    stat_feats = statistic_features.signal_stats(signal, names=False)

    return np.hstack(([rec, det, lmax, sd1, sd2, csi, csv, kfd], temp_feats, stat_feats))


import os
import datetime

import pandas as pd

if __name__ == '__main__':
    directory = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'

    patient = 'PAT_312_EXAMES'

    pat_dir = os.path.join(directory, patient)

    list_ecg_files = sorted(os.listdir(os.path.join(pat_dir, 'ECG')))
    ecg_df = pd.read_parquet(os.path.join(pat_dir, 'ECG', list_ecg_files[0]))

    time_window = datetime.timedelta(minutes=5)
    time_start = ecg_df['index'][4000]
    time_end = time_start + time_window


    seg = ecg_df.loc[ecg_df['index'].between(time_start, time_end)]
    ss = seg['rr'].dropna().values

    feats_names = rr_features(names=True)
    feats = rr_features(ss)
    print('here')