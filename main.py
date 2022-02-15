from extractor import *
import os  # provides functions for interacting with the operating system
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from extractor_functions import readlab, readcsv_chroma, chord_chroma_raws, get_mu_sigma_from_chroma, \
    transition_prob_matrix


notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
all_chords = ['G', 'B:min', 'E:min', 'C', 'A:min', 'F', 'D', 'F#', 'C#', 'E', 'B', 'A', 'F#:min', 'C:min', 'F:min', 'Eb', 'G:min', 'Bb', 'Ab', 'D:min', 'C#:min', 'Db', 'Bb:min', 'Eb:min', 'Gb:min', 'Gb', 'G#:min', 'G#', 'D#:min']
# import the csvs and calculate the mean
mu_dic = dict.fromkeys(all_chords)
cov_dic = dict.fromkeys(all_chords)
chromagrams_path = 'data/chromagrams'
i = int(0)

for elem in sorted(os.listdir('data/chromagrams/')):
    temp_path = f'{chromagrams_path}/{elem}'
    temp_df = pd.read_csv(temp_path)
    temp_df = temp_df.iloc[:, 1:]
    mu_array, states_cov_matrices = get_mu_sigma_from_chroma(temp_df, notes)
    mu_dic[all_chords[i]] = mu_array
    cov_dic[all_chords[i]] = states_cov_matrices
    i += 1




# ------------------------------------------------------------------------------------------
# TRANSITION CALCULATED ON CHORD CHANGE FROM .LAB FILE
# ------------------------------------------------------------------------------------------
tp_matrix = []


chord_annotation_long_dic = []
for i, val in enumerate(chord_annotation_dic):
    chord_annotation_long_dic.append(chord_annotation_dic[i])



for i, val in enumerate(chord_annotation_dic):
    first_chord = chord_annotation_dic[i]['chord'].values[:-1].tolist()
    second_chord = chord_annotation_dic[i]['chord'][1:].tolist()
    tp_matrix.append(transition_prob_matrix(first_chord, second_chord))

print('ok')

# ------------------------------------------------------------------------------------------
# INITIAL STATE MATRIX
# ------------------------------------------------------------------------------------------

in_matrix = []
for i in range(29):
    in_matrix.append(1/29)




def build_gaussian_hmm(initial_state_prob, transition_matrix, mu_array, states_cov_matrices):
    # continuous emission model
    h_markov_model = hmm.GaussianHMM(n_components=transition_matrix.shape[0], covariance_type="full")
    # initial state probability
    h_markov_model.startprob_ = initial_state_prob
    # transition matrix probability
    h_markov_model.transmat_ = transition_matrix.values
    # part of continuous emission probability - multidimensional gaussian
    # 12 dimensional mean vector
    h_markov_model.means_ = mu_array
    # array of covariance of shape [n_states, n_features, n_features]
    h_markov_model.covars_ = states_cov_matrices
    return h_markov_model

h_markov_model = build_gaussian_hmm(in_matrix, tp_matrix, mu_dic, cov_dic)

print('jhg')