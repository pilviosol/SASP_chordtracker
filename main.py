import os  # provides functions for interacting with the operating system
import shutil  # move files and delete folders with files
import \
    tarfile  # used to read and write tar archives (contains uncompressed byte streams of the files which it contains)
import urllib.request  # download files folder
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import IPython
import IPython.display as ipd
import librosa

import IPython  # listen to sounds on Python
# import pretty_midi # contains utility function/classes for handling MIDI data, !install only if u want to listen the predictions!

from scipy.io import wavfile
from scipy.spatial import distance_matrix
from matplotlib import colors
from itertools import product

from hmmlearn import hmm
from sklearn.metrics import f1_score
import pathlib
from pathlib import Path
import sklearn

plt.style.use('seaborn')
# from __future__ import division
# import scipy
# from scipy.signal import hamming
# from scipy.fftpack import fft

# np.genfromtxt(path_csv+name_csv, delimiter=',')


# Variables
path_lab = 'data/Beatles_lab'
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# Probabilità di avere un accordo dopo l'altro
def __calc_prob_chordpairs(chord_group):
    chord_group_count = chord_group.groupby('second_chord').size().reset_index()
    chord_group_count.columns = ['second_chord', 'count']
    total = chord_group_count['count'].sum()
    chord_group_count['transition_prob'] = chord_group_count['count']/total

    return chord_group_count


# transition calculate on the chord change not on tactus or windows -> need chromagram values
def transition_prob_matrix(firstchord, secondchord):
    sequence_chords = pd.DataFrame({'first_chord': firstchord, 'second_chord': secondchord})
    prob_matrix = sequence_chords.groupby('first_chord').apply(__calc_prob_chordpairs).reset_index()
    prob_matrix = prob_matrix.drop('level_1', axis=1)
    prob_matrix = prob_matrix.pivot(index='first_chord', columns='second_chord', values='transition_prob')
    prob_matrix = prob_matrix.fillna(0)

    return prob_matrix


# create the dictionary from CSV


# initialize the tp_matrix list
tp_matrix = []


for i,val in enumerate(chord_annotation_dic):
    first_chord = chord_annotation_dic[i]['chord'].values[:-1].tolist()
    second_chord = chord_annotation_dic[i]['chord'][1:].tolist()
    tp_matrix.append(transition_prob_matrix(first_chord, second_chord))

print('ok')
