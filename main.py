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
import csv
import json
import pickle
import IPython
import IPython.display as ipd
import librosa
import \
    re

import IPython  # listen to sounds on Python
# import pretty_midi # contains utility function/classes for handling MIDI data, !install only if u want to listen the predictions!

from scipy.io import wavfile
from scipy.spatial import distance_matrix
from matplotlib import colors
from itertools import product

#from hmmlearn import hmm
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


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
path_lab = 'data/Beatles_lab_tuned/'
path_csv = 'data/Beatles_csv/'
notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
win_size_t = 2048/44100
chroma_dic_new = []
tp_matrix = []


# ------------------------------------------------------------------------------------------
# SIMPLIFY CHORD NOTATION
# ------------------------------------------------------------------------------------------
def __simplify_chords(chords_df):  # silence
    chords_processed = chords_df['chord'].str.split(':maj')  # remove major x chords return array of array
    # containing all the chords
    chords_processed = [elem[0] for elem in chords_processed]  # further process step above to return 1 array
    # take the first element in all the N arrays (chords
    # string) to make it a list of N elements
    chords_processed = [elem.split('/')[0] for elem in chords_processed]  # remove inverted chords
    chords_processed = [elem.split('aug')[0] for elem in chords_processed]  # remove augmented chords
    chords_processed = [elem.split(':(')[0] for elem in chords_processed]  # remove added notes chords
    chords_processed = [elem.split('(')[0] for elem in chords_processed]  # remove added notes chords 2
    chords_processed = [elem.split(':sus')[0] for elem in chords_processed]  # remove sustained chords
    chords_processed = [re.split(":?\d", elem)[0] for elem in chords_processed]  # remove added note
    chords_processed = [elem.replace('dim', 'min') for elem in chords_processed]  # change diminute to minor
    chords_processed = [elem.replace('hmin', 'min') for elem in chords_processed]  # change semi-diminute to minor
    chords_processed = [re.split(":$", elem)[0] for elem in chords_processed]  # remove added notes chords

    return chords_processed


# ------------------------------------------------------------------------------------------
# READ .LAB FILES
# ------------------------------------------------------------------------------------------
def readlab(path):
    '''

    Args:
        path: path where the .lab files are

    Returns:
        dictionary: a dictionary with all songs decomposed in start, end, chord
        song_list: a list containing all song's name in the dataset

    '''
    dictionary = []
    list_name = []

    for elem in sorted(os.listdir(path)):
        song_path = f'{path}/{elem}'
        list_name.append(elem)
        chord_annotation = pd.read_csv(song_path, sep=' ', header=None)  # no header in the files and separated
                                                                         # by an empty space
                                                                         # set up the columns of the file
        chord_annotation.columns = ['start', 'end', 'chord']
        chord_annotation['chord'] = __simplify_chords(chord_annotation)
        chord_annotation.loc[chord_annotation['chord'] == 'N', 'chord'] = chord_annotation['chord'].mode()[
            0]  # replace silence by probable
                # tonal end
        dictionary.append(chord_annotation)

    return dictionary, list_name

# ------------------------------------------------------------------------------------------
# CREATE DICTIONARY WITH ALL CHROMAS
# ------------------------------------------------------------------------------------------
def readcsv_chroma(path):
    '''

    Args:
        path: path where the .csv files are

    Returns:
        dictionary: a dictionary with all songs, each one decomposed in equally spaced windows with all energies per
        note for each window

    '''
    dictionary = []

    for elem in sorted(os.listdir(path)):
        song_path = f'{path}/{elem}'
        chroma_annotation = pd.read_csv(song_path, sep=',', header=None)
        chroma_annotation = pd.DataFrame.transpose(chroma_annotation)
        chroma_annotation.columns = notes
        dictionary.append(chroma_annotation)

    return dictionary


# ------------------------------------------------------------------------------------------
# CALLING FUNCTIONS READLAB AND READCSV_CHROMA
# ------------------------------------------------------------------------------------------

# dictionary with the same form of the lab
chord_annotation_dic, song_list = readlab(path_lab)

# create a chroma dic with chromas of all songs
chroma_dic = readcsv_chroma(path_csv)


# ------------------------------------------------------------------------------------------
# PASSING CHORD TO EVERY CHROMA ROWS
# ------------------------------------------------------------------------------------------
def chord_chroma_raws(chroma, chord_annotation):
    '''

    Args:
        chroma: a dictionary with all songs, each one decomposed in equally spaced windows with all energies per
        note for each window
        chord_annotation: a dictionary with all songs decomposed in start, end, chord

    Returns:
        chroma: a dictionary with all songs, each one decomposed in equally spaced windows with all energies per
        note for each window with a new column with the chord assigned from the dataset

    '''
    chroma['chord'] = '0'
    raw = 0
    for ii in range(chroma.shape[0]):
        #print('i: ', ii)
        #print('win_size_t*np.float(ii+1): ', win_size_t*np.float(ii+1))
        #print('chord_annotation[end][raw]: ', chord_annotation['end'][raw])
        if win_size_t*np.float(ii+1) < chord_annotation['end'][raw]:
            chroma.loc[ii, 'chord'] = chord_annotation['chord'][raw]
            #print('if')
        else:
            chroma.loc[ii, 'chord'] = chord_annotation['chord'][raw]
            raw += 1
            #print('else')

    return chroma


# ------------------------------------------------------------------------------------------
# CALLING FUNCTION CHORD_CHROMA_RAWS
# ------------------------------------------------------------------------------------------

# add as last raw the name of the chord in that specific window
for idx in range(len(chord_annotation_dic)):
    print(idx)
    chroma_dic_new.append(chord_chroma_raws(chroma_dic[idx], chord_annotation_dic[idx]))

print("len(chroma_dic_new): ", len(chroma_dic_new))
#np.savetxt("data/chroma_dic_new", chroma_dic_new, delimiter=",")

'''
for index, row in chroma_dic_new[0].iterrows():
    print(row) 
'''

all_chords = []
for songs in chroma_dic_new:
    for elements in songs['chord']:
        if elements not in all_chords:
            all_chords.append(elements)

print(all_chords)

chords_dictionary = {}
for chords in all_chords:
    name = str(chords)
    chords_dictionary[name] = []

for chord in all_chords:
    print('processing chord: ', chord)
    for song in chroma_dic_new:
        for row, index in song.iterrows():
            if (index['chord'] == chord):
                chords_dictionary[chord].append(index)

# with open('data/chords_dictionary.csv', 'w') as csvfile:
#     for key in chords_dictionary.keys():
#         csvfile.write("%s, %s\n" % (key, chords_dictionary[key]))
#
# a_file = open("data/chords_dictionary.pkl", "wb")
# pickle.dump(chords_dictionary, a_file)
# a_file.close()
#
# with open("data/chords_dictionary.json", "w") as outfile:
#     json.dumps(chords_dictionary, outfile)
# print('ciao')


'''
def __get_mu_array(note_feature_vector):
    return note_feature_vector[notes].mean()

def get_mu_sigma_from_chroma(chromagram):
    
    mu_array = chromagram.groupby('chord').apply(__get_mu_array)

    states_cov_matrices = []
    for name, group in chromagram.groupby('chord'): # alphabetic order
        states_cov_matrices.append(group[notes].cov().values)
    states_cov_matrices = np.array(states_cov_matrices)

    return [mu_array, states_cov_matrices]

mu_array, states_cov_matrices = get_mu_sigma_from_chroma(chroma_dic_new) '''


# ------------------------------------------------------------------------------------------
# PROBABILITY OF HAVING ONE CHORD AFTER THE OTHER BASED ON .LAB FILES
# ------------------------------------------------------------------------------------------
def __calc_prob_chordpairs(chord_group):
    '''

    Args:
        chord_group: group of chords of a song

    Returns:
        chord_group_count: probability of having a chord after one other for all chords in a song

    '''
    chord_group_count = chord_group.groupby('second_chord').size().reset_index()
    chord_group_count.columns = ['second_chord', 'count']
    total = chord_group_count['count'].sum()
    chord_group_count['transition_prob'] = chord_group_count['count']/total

    return chord_group_count


# ------------------------------------------------------------------------------------------
# TRANSITION CALCULATED ON CHORD CHANGE FROM .LAB FILE
# ------------------------------------------------------------------------------------------
def transition_prob_matrix(firstchord, secondchord):
    '''

    Args:
        firstchord:
        secondchord:

    Returns:
        prob_matrix: matrix with probabilities of passing from a chord to another

    '''
    sequence_chords = pd.DataFrame({'first_chord': firstchord, 'second_chord': secondchord})
    prob_matrix = sequence_chords.groupby('first_chord').apply(__calc_prob_chordpairs).reset_index()
    prob_matrix = prob_matrix.drop('level_1', axis=1)
    prob_matrix = prob_matrix.pivot(index='first_chord', columns='second_chord', values='transition_prob')
    prob_matrix = prob_matrix.fillna(0)

    return prob_matrix


for i, val in enumerate(chord_annotation_dic):
    first_chord = chord_annotation_dic[i]['chord'].values[:-1].tolist()
    second_chord = chord_annotation_dic[i]['chord'][1:].tolist()
    tp_matrix.append(transition_prob_matrix(first_chord, second_chord))

print('ok')

