import datetime
import os
import time
import warnings

import multiprocessing as mp
import numpy as np
import pandas as pd

from src import hrv_features, statistic_features, spectral_features, temporal_features, rr_features


warnings.filterwarnings('ignore')

# features_names = statistic_features.signal_stats(names=True)
# 'diff_rms_sd', 'diff_sd_nn', 'diff_mean_nn', 'diff_nn50', 'diff_pnn50', 'diff_var', 'diff_lf_pwr', 'diff_hf_pwr',
# 'diff_lf_hf', 'diff_sd1', 'diff_sd2', 'diff_csi', 'diff_csv', 'diff_kfd', 'diff_rec', 'diff_det', 'diff_lmax'


def f(signal):
    print(len(signal))
    return rr_features.rr_features(signal)
    # return hrv_features.hrv_features(nni=signal)


def get_temps(main_dir, patient, file, dict_nni =None):
    """
    Get temporal jumps for segmentation for the total file, respecting window and overlap
    :param main_dir:
    :param patient:
    :param file:
    :param dict_nni:
    :return:
    """

    ecg_df = pd.read_parquet(os.path.join(main_dir, patient, 'ECG', file), engine='fastparquet')
    if dict_nni:
        ecg_df['norm_nni'] = (ecg_df['nni'] - dict_nni['mean']) / dict_nni['std']
    last_idx = len(ecg_df) - int((fs * window).total_seconds())
    try:
        # get temporal jumps for segmentation, with the chosen overlap and window
        temps = pd.date_range(ecg_df['index'].values[0], ecg_df['index'].values[last_idx], freq=overlap)
    except:
        temps = []
        print('Not enough temps')

    return ecg_df, temps


# HEART RATE STATISTICS
if __name__ == '__main__':

    modal = 'hr'
    overlap = '30S'
    window = datetime.timedelta(minutes=5)
    fs = 1
    features_names = rr_features.rr_features(names=True)
    main_dir = 'G:\\PreEpiSeizures\\Patients_HEM\\Retrospective'
    for patient in ['PAT_413']: # sorted(os.listdir(main_dir)):
        # check duration
        start_time = time.time()
        # directory to save features, it is created if not already
        features_dir = os.path.join(main_dir, patient, 'features')
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)
        print('Processing patient ', patient)
        # list all ecg df files
        list_files = sorted(os.listdir(os.path.join(main_dir, patient, 'ECG')))
        # process differently according to modality
        file_prefix = patient + '_norm_5min_ol' + overlap + '_' + modal + '_features'
        for file in list_files:
            ft_file_name = file_prefix + str(file.split('df')[-1])
            if os.path.isfile(os.path.join(features_dir, ft_file_name)):
                continue
            print(f'Processing file {file}')
            # open file
            ecg_df, temps = get_temps(main_dir, patient, file)
            if len(temps) == 0:
                continue
            # multiprocessing for feature calculation
            with mp.Pool(mp.cpu_count()) as p:
                features = np.vstack([f(ecg_df.loc[ecg_df['index'].between(temps[i], temps[i]+window),
                                                   modal].dropna().values) for i in range(len(temps))])
            assert(len(temps) == len(features))
            index_df = np.arange(0, len(temps), 1)
            df_features = pd.DataFrame(features, index=index_df, columns=features_names)
            df_features['index'] = temps
            df_features.to_parquet(os.path.join(features_dir, ft_file_name), compression='gzip', engine='fastparquet')

        print('Total time for calculate features ', time.time()-start_time)


# RESPIRATORY RATE CALCULATION

"""
if __name__ == '__main__':

    modal = 'rr'
    overlap = '30S'
    window = datetime.timedelta(minutes=5)
    fs = 1
    features_names = rr_features.rr_features(names=True)
    main_dir = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'
    for patient in sorted(os.listdir(main_dir)):
        # check duration
        start_time = time.time()
        # directory to save features, it is created if not already
        features_dir = os.path.join(main_dir, patient, 'features')
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)
        print('Processing patient ', patient)
        # list all ecg df files
        list_files = sorted(os.listdir(os.path.join(main_dir, patient, 'ECG')))
        # process differently according to modality
        file_prefix = patient + '_norm_5min_ol' + overlap + '_' + modal + '_features'
        for file in list_files:
            ft_file_name = file_prefix + str(file.split('df')[-1])
            if os.path.isfile(os.path.join(features_dir, ft_file_name)):
                continue
            print(f'Processing file {file}')
            # open file
            ecg_df, temps = get_temps(main_dir, patient, file)
            if len(temps) == 0:
                continue
            # multiprocessing for feature calculation
            with mp.Pool(mp.cpu_count()) as p:
                features = np.vstack([f(ecg_df.loc[ecg_df['index'].between(temps[i], temps[i]+window),
                                                   modal].dropna().values) for i in range(len(temps))])
            assert(len(temps) == len(features))
            index_df = np.arange(0, len(temps), 1)
            df_features = pd.DataFrame(features, index=index_df, columns=features_names)
            df_features['index'] = temps
            df_features.to_parquet(os.path.join(features_dir, ft_file_name), compression='gzip', engine='fastparquet')

        print('Total time for calculate features ', time.time()-start_time)
"""


# HRV FEATURES FOR NN INTERVALS CALCULATION
"""
if __name__ == '__main__':

    # Import
    modal = 'rr'
    overlap = '30S'
    window = datetime.timedelta(minutes=5)
    fs = 256
    #features_names = ['rms_sd', 'sd_nn', 'mean_nn', 'nn50', 'pnn50', 'thist', 'tinn', 'lf_pwr', 'hf_pwr',
     #                 'lf_hf', 'sd1', 'sd2', 'csi', 'csv', 'kfd', 'rec', 'det', 'lmax']
    features_names = rr_features.rr_features(names=True)
    main_dir = 'F:\\PreEpiSeizures\\Patients_HEM\\Retrospective'
    for patient in sorted(os.listdir(main_dir)):
        # check duration
        start_time = time.time()
        # directory to save features, it is created if not already
        features_dir = os.path.join(main_dir, patient, 'features')
        if not os.path.isdir(features_dir):
            os.makedirs(features_dir)
        print('Processing patient ', patient)
        # list all ecg df files
        list_files = sorted(os.listdir(os.path.join(main_dir, patient, 'ECG')))
        # process differently according to modality
        if modal == 'ecg':
            dict_nni = pickle.load(open(os.path.join(main_dir, patient, 'nni_mean_std.pickle'), 'rb'))
            file_prefix = patient + '_norm_5min_ol' + overlap + '_hrv_features'
        elif modal == 'rr':
            file_prefix = patient + '_norm_5min_ol' + overlap + '_rr_features'
        for file in list_files:
            ft_file_name = file_prefix + str(file.split('df')[-1])
            if os.path.isfile(os.path.join(features_dir, ft_file_name)):
                print('features already calculated')
                continue
            print(f'Processing file {file}')
            # open file
            ecg_df, temps = get_temps(main_dir, patient, file, modal, dict_nni)
            if len(temps) == 0:
                continue

            with mp.Pool(mp.cpu_count()) as p:
                features = np.vstack([f(ecg_df.loc[ecg_df['index'].between(temps[i], temps[i]+window),
                                                   ['nni', 'norm_nni']].dropna()) for i in range(len(temps))])
            assert(len(temps) == len(features))
            index_df = np.arange(0, len(temps), 1)
            df_features = pd.DataFrame(features, index=index_df, columns=features_names)
            df_features['index'] = temps
            df_features.to_parquet(os.path.join(features_dir, ft_file_name), compression='gzip', engine='fastparquet')

        print('Total time for calculate features ', time.time()-start_time)
    """