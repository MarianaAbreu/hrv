####################################
#
#  MLB Project 2021
#
#  Module: Feature Selection
#  File: init
#
#  Created on May 17, 2021
#  All rights reserved to João Saraiva and Débora Albuquerque
#
####################################
import json
from _datetime import datetime

import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import os
import numpy as np

from functools import partial
from math import sqrt
from processing import utils_signal_processing as usp
from processing import hrv_features as hf
from read_files import Patient


class Features():

    def __init__(self, id, hospital, type = 'features'):
        self.dir = os.path.join('C:\\Users\\Mariana\\PycharmProjects\\mapme\\features', type, hospital)
        self.id = id
        self.hospital = hospital
        self.data = pd.DataFrame()
        self.norm_data = pd.DataFrame()


    def create_diff(self):
        feature_files = [file for file in sorted(os.listdir(self.dir)) if file.startswith(self.id)]
        new_dir = os.path.join('C:\\Users\\Mariana\\PycharmProjects\\mapme\\features', 'diff', self.hospital)
        for file in feature_files:
            diff_df = hf.get_diff(pd.read_hdf(self.dir + os.sep + file), 50)
            diff_df.to_hdf(new_dir + os.sep +file +'_diff', 'df', mode='w')



    def join_patient_features(self):
        feature_files = [file for file in sorted(os.listdir(self.dir)) if file.startswith(self.id)]

        self.data = pd.concat([pd.read_hdf(os.path.join(self.dir, file)) for file in feature_files])

    def normalise_feats(self):

        if self.data.empty:
            self.join_patient_features()

        self.norm_data = usp.normalise_feats(self.data)

    def remove_correlated(self):

        if self.norm_data.empty:
            self.normalise_feats()
        self.corr_data = usp.correlation_feats(self.norm_data)

    def rolling_average(self):

        self.norm_data

        return data

    def plot_patient_features(self):
        plt.figure(figsize=(50,50))

        self.remove_correlated()
        data = self.corr_data
        n = 1
        l = int(sqrt(len(data.columns)))
        h = int(np.round(len(data.columns)/l))
        onset = self.get_onset()

        for feature in data.columns:
            plt.subplot(l, h, n)
            plt.title(feature)
            plt.plot(data[feature])
            plt.vlines(onset, 0, 1)

            n += 1
        plt.show()

    def get_onset(self):

        try:
            report = pd.read_csv(os.path.join('E:\\Patients_' + self.hospital, 'Retrospective', self.id, 'seizure_label'))
        except:
            rep = Patient.Report(self.hospital, os.path.join('E:\\Patients_' + self.hospital, 'Retrospective', self.id), self.id)
            rep.get_report(self.hospital, self.id)
            report = pd.read_csv(
                os.path.join('E:\\Patients_' + self.hospital, 'Retrospective', self.id, 'seizure_label'))
        onset = [datetime.strptime(date, '%d-%m-%Y\n%H:%M:%S') for date in report['Date']]
        return onset

if __name__ == '__main__':

    id = 'PAT_413'
    hospital = 'HEM'

    features = Features(id, hospital)
    features.plot_patient_features()