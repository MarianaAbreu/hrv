import datetime
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle

from processing import utils_ecg_processing as uep
import matplotlib.pyplot as plt


def choose_big(sig, name, dir):

    sig2 = pd.read_hdf(os.path.join(dir, name))

    if len(sig2) >= len(sig):
        return 0
    else:
        return 1


def sig_to_dif(list):

    file = list[0]
    dir = list[1]
    sampling_rate = list[2]
    print(file)

    if 'HEM' in dir:
        sig = pd.read_hdf(os.path.join(dir, 'signals') + os.sep + file)

        date = file.split('_')[0]
        start_date = datetime.datetime.strptime(date, '%Y-%m-%d--%H-%M-%S')
        end_time = datetime.timedelta(milliseconds=len(sig) * 1000 / sampling_rate) + start_date
        mean_template_ecg = pickle.load(open('mean_template_ecg', 'rb'))
        chn = uep.choose_ecg_channel(sig, mean_template_ecg)
        sig = sig[chn].values

    nni = uep.dissimilarity(sig, sampling_rate=sampling_rate)
    index = pd.date_range(start_date, end_time, periods=len(nni))
    nni_df = pd.DataFrame(nni, columns=['nni'], index=index)

    if 'diff' + date not in os.listdir(os.path.join(dir, 'diff')):
        nni_df.to_hdf(os.path.join(dir, 'diff', 'diff_' + date), 'df', mode='w')
    else:
        if choose_big(nni_df, 'diff_'+date, os.path.join(dir, 'diff')):
            nni_df.to_hdf(os.path.join(dir, 'diff', 'diff_' + date), 'df', mode='w')
            print('diff_'+ date, ' already in folder but replaced')

    return


def sig_to_nni(list):

    file = list[0]
    dir = list[1]
    sampling_rate = list[2]
    print(file)

    if 'HEM' in dir:
        sig = pd.read_hdf(os.path.join(dir, 'signals') + os.sep + file)

        date = file.split('_')[0]
        start_date = datetime.datetime.strptime(date, '%Y-%m-%d--%H-%M-%S')
        end_time = datetime.timedelta(milliseconds=len(sig) * 1000 / sampling_rate) + start_date
        mean_template_ecg = pickle.load(open('mean_template_ecg', 'rb'))
        chn = uep.choose_ecg_channel(sig, mean_template_ecg)
        sig = sig[chn].values

    elif 'HSM' in dir:
        sig = pd.read_hdf(os.path.join(dir, 'HSM') + os.sep + file)
        date = file.split('_')[0]
        try:
            start_date = datetime.datetime.strptime(date, '%Y-%m-%d %H-%M-%S')
        except:
            start_date = datetime.datetime.strptime(date[:-3], '%Y-%m-%d %H-%M-%S')
        print('Starting ...', start_date)
        end_time = datetime.timedelta(milliseconds=len(sig) * 1000 / sampling_rate) + start_date
        sig = sig['POL  ECG-'].values

    nni = uep.ecg_processing(sig, sampling_rate=sampling_rate)
    index = pd.date_range(start_date, end_time, periods=len(nni))
    nni_df = pd.DataFrame(nni, columns=['nni'], index=index)

    if 'nni_' + date not in os.listdir(os.path.join(dir, 'nni')):
        nni_df.to_hdf(os.path.join(dir, 'nni', 'nni_' + date), 'df', mode='w')
    else:
        if choose_big(nni_df, 'nni_'+date, os.path.join(dir, 'nni')):
            nni_df.to_hdf(os.path.join(dir, 'nni', 'nni_' + date), 'df', mode='w')
            print('nni_'+date, ' already in folder but replaced')


def sig_to_ecg(list):

    file = list[0]
    dir = list[1]
    sampling_rate = list[2]
    print(file)

    if 'HEM' in dir:
        sig = pd.read_hdf(os.path.join(dir, 'signals') + os.sep + file)

        date = file.split('_')[0]
        start_date = datetime.datetime.strptime(date, '%Y-%m-%d--%H-%M-%S')
        end_time = datetime.timedelta(milliseconds=len(sig) * 1000 / sampling_rate) + start_date
        mean_template_ecg = pickle.load(open('mean_template_ecg', 'rb'))
        chn = uep.choose_ecg_channel(sig, mean_template_ecg)
        sig = sig[chn].values

    elif 'HSM' in dir:
        sig = pd.read_hdf(os.path.join(dir, 'HSM') + os.sep + file)
        date = file.split('_')[0]
        try:
            start_date = datetime.datetime.strptime(date, '%Y-%m-%d %H-%M-%S')
        except:
            start_date = datetime.datetime.strptime(date[:-3], '%Y-%m-%d %H-%M-%S')
        print('Starting ...', start_date)
        end_time = datetime.timedelta(milliseconds=len(sig) * 1000 / sampling_rate) + start_date
        sig = sig['POL  ECG-'].values

    nni = uep.ecg_processing(sig, sampling_rate=sampling_rate)
    index = pd.date_range(start_date, end_time, periods=len(nni))
    nni_df = pd.DataFrame(nni, columns=['nni'], index=index)

    if 'nni_' + date not in os.listdir(os.path.join(dir, 'nni')):
        nni_df.to_hdf(os.path.join(dir, 'nni', 'nni_' + date), 'df', mode='w')
    else:
        if choose_big(nni_df, 'nni_'+date, os.path.join(dir, 'nni')):
            nni_df.to_hdf(os.path.join(dir, 'nni', 'nni_' + date), 'df', mode='w')
            print('nni_'+date, ' already in folder but replaced')


def all_files_process(dir, save_dir='',save=True):

    list_files = os.listdir(dir)
    mean_template_ecg = pickle.load(open('mean_template_ecg', 'rb'))

    for file in list_files:

        chn = uep.choose_ecg_channel(sig, mean_template_ecg)

        sig = pd.read_hdf(dir + os.sep + file)[chn].values

        date = file.split('_')[0]
        start_date = datetime.datetime.strptime(date, '%Y-%m-%d--%H-%M-%S')
        end_time =  datetime.timedelta(milliseconds=len(sig)*1000/256)+start_date
        nni = uep.ecg_processing(sig, sampling_rate=256)
        index = pd.date_range(start_date, end_time, periods=len(nni))

        nni_df = pd.DataFrame(nni, ['nni'], index=index)
        pp
        if save:
            nni_df.to_hdf(os.path.join(save_dir,'nni_'+start_date))
    qqq
    return nni


#HEM signal to nni
if __name__ == '__main__':

    # dir = 'E:\\Patients_HEM\\PAT_358'
    for pat in ['PAT_312_EXAMES', 'PAT_326_EXAMES', 'PAT_386', 'PAT_391_EXAMES', 'PAT_400', 'PAT_358', 'PAT_352_EXAMES', 'PAT_365_EXAMES']: #os.listdir('E:\Patients_HSM\\'):

        dir = 'E:\\Patients_HEM\\' + pat

        try:
            try:
                os.mkdir(os.path.join(dir, 'nni'))
            except:
                'Path exists'

            with mp.Pool(mp.cpu_count()) as p:
                p.map(sig_to_nni,[[file, dir, 256] for file in sorted(os.listdir(os.path.join(dir, 'signals')))])
        except:
            continue


# HSM signal to nni
# if __name__ == '__main__':
#
#     # dir = 'E:\\Patients_HEM\\PAT_358'
#     for pat in ['Patient108', 'Patient109', 'Patient110']: # os.listdir('E:\Patients_HSM\\'):
#         dir = 'E:\Patients_HSM\\' + pat
#         print(pat)
#         try:
#             files = [file for file in sorted(os.listdir(os.path.join(dir, 'HSM'))) if file.startswith('20')]
#             print(files)
#         except:
#             files = []
#             continue
#
#         if files != []:
#             try:
#                 os.mkdir(os.path.join(dir, 'nni'))
#             except:
#                 'Path exists'
#
#             with mp.Pool(mp.cpu_count()) as p:
#                 p.map(sig_to_nni,[[file, dir, 1000] for file in files])
#
# #
#dir = 'E:\Patients_HSM\Patient101'

#all_files_process(os.path.join(dir, 'Baseline'),os.path.join(dir, 'nni'))
#
