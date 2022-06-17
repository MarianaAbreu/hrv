

import matplotlib.pyplot as plt
import tkinter as tk

from functools import partial
from math import sqrt
from processing import feature_selection


def handle_click(_options):
    print(_options)


CBC = dict()

class Table:
    """
    features: Dictionary of dictionaries of lists. First dimension corresponds to each patient, associated by its key
                Second dimension corresponds to each patient's crisis, associated to its key.
                Third dimension corresponds to the existing extracted features for that patient-crisis pair.
    """

    def __init__(self, root, features, checkbox_controls):
        # do a header for the table
        header = set()
        for feature in features.columns:
            header.add(feature)
            #if baseline:
                #for feature in state.columns:
                    #header.add('baseline'+ feature)
        self.feature_labels = list(header)

        # initialize checkbox control variables
        for p in patients:
            checkbox_controls[p] = dict()
            for f in self.feature_labels:
                checkbox_controls[p][f] = tk.IntVar(root)

        # save first column width
        width = 30

        # insert header
        self.e = tk.Entry(root, width=width, fg='black', font=('Arial', 10, 'bold'), justify='right')
        self.e.grid(row=0, column=0)
        self.e.insert(tk.END, "Patient / Crisis")
        for j in range(len(self.feature_labels)):
            self.e = tk.Entry(root, width=width, fg='black', font=('Arial', 10))
            self.e.grid(row=j+1, column=0)
            self.e.insert(tk.END, self.feature_labels[j])

        # creating body
        m = 1
        n = 0
        self.e = tk.Entry(root, width=5, fg='black', font=('Arial', 10, 'bold'))
        self.e.grid(row=n, column=m)
        self.e.insert(tk.END, '/')
        n += 1

        for patient in patients:
            for feature in self.feature_labels:
                self.e = tk.Checkbutton(root, variable=checkbox_controls[patient][feature])
                self.e.grid(row=n, column=m)
                if feature in features.columns:
                    checkbox_controls[patient][feature].set(1)
                else:
                    checkbox_controls[patient][feature].set(0)
                n += 1
        m += 1

        # create (dis)select all buttons
        n = 1
        for f in self.feature_labels:
            print(f)
            tk.Button(root, text="(De)Select All", command=partial(self.de_select_all, f)).grid(row=n, column=m+1)
            n += 1

    def de_select_all(self, f):
        print(CBC[patients[0]][f])
        if CBC[patients[0]][f].get() == 0:
            new_value = 1
            print("Selecting all", f, "features")
        else:
            new_value = 0
            print("Deselecting all", f, "features")

        for p in CBC.values():
            for c in p.values():
                c[f].set(new_value)

#def get_full_baseline(patients):
    #baseline_awake = get_baseline_from_patients(patients, "awake")
    #baseline_asleep = get_baseline_from_patients(patients, "asleep")

    #for patient in patients:
        #if baseline_awake[patient] is not None:
            #if baseline_asleep[patient] is not None:
                #baseline_awake[patient].update(baseline_asleep[patient])


    #return baseline_awake

patients = ['PAT_413']
#crises = [1,2,3]
state = "awake"

features_class = feature_selection.Features(id='PAT_413', hospital='HEM')

features_class.remove_correlated()

features = features_class.corr_data

#get_features_from_patients(patients, crises, 'asleep')
#baseline = get_full_baseline(patients)

# create root window
root = tk.Tk()
root.title("Feature Selection")

# create table
t = Table(root, features, CBC)
root.mainloop()

def convert_date_time(date_time: str):
    date, time = date_time.split(' ')
    d, m, a = date.split('/')
    return a + '-' + m + '-' + d + ' ' + time


def inspect_features(features):
    header = set()
    for feature in features.columns:
        header.add(feature)
    feature_labels = list(header)

    colors = ['#00bfc2', '#5756d6', '#fada5e', '#62d321', '#fe9b29']
    # color for each payient-crisis pair is given by: patientID - 100 + crisisID - 2

    n_features = len(feature_labels)
    n_subplots_per_side = int(sqrt(n_features))

    # get onsets
    #with open(data_path + '/patients.json') as metadata_file:
     #   metadata = json.load(metadata_file)
    onsets = features_class.get_onset()

    background_color = (0, 0, 0)
    rolling_avg = False
    if input("Do you want to do rolling average? y/n ").lower() == 'y':
        rolling_avg = True
        n_avg = int(input("Number of points around each point for rolling average: "))


    fig = plt.figure(figsize=(20, 20), facecolor=background_color)


    for i in range(n_features):
        print(i,"/",n_subplots_per_side,"/",n_features)
        ax = plt.subplot(n_subplots_per_side + 1, n_subplots_per_side + 1, i+1, facecolor=background_color)
        ax.grid(color='white', linestyle='--', linewidth=0.35)
        feature_label = feature_labels[i]
        plt.title(feature_label, color='white')

        reference_onset = None
        if feature_label in features.columns:

            feature = features[feature_label]
            feature_rolling = features[feature_label]
            time_axis = features[feature_label].index

            if not rolling_avg:
                plt.plot(time_axis, feature.values, '.', label='/', markersize=1.2,
                                 alpha=0.7)  # .values discards time axis
                plt.xlabel('time', color='white')
            else:
                for j in range(len(feature) - 2 * n_avg):
                    feature_rolling[j + n_avg] = feature.values[j:j+n_avg*2].mean()

                        # Tendency line
                plt.plot(time_axis, feature_rolling.values, '-', markersize=1.2, alpha=0.7)  # .values discards time axis

            plt.xlabel('time', color='white')

        # draw onset vertical line
        plt.vlines(x=onsets, ymin=0, ymax=1, color='red', linewidth=0.4)
        leg = plt.legend(loc='best', facecolor=background_color)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color('white')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()

inspect_features(features)
