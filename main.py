from extractor import *
import os
import numpy as np
import pandas as pd
import mido
from hmmlearn import hmm
from extractor_functions import readlab, readcsv_chroma, chord_chroma_raws, get_mu_sigma_from_chroma, \
    transition_prob_matrix
from utils import column_index, chords_mapper
from midi_creator import midi_chord_creator, midi_predictor

# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
print('variables')
notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
all_chords = ['A', 'A_min', 'Ab', 'B', 'B_min', 'Bb', 'Bb_min', 'C', 'C#', 'C#_min', 'C_min', 'D', 'D#_min', 'D_min', 'Db', 'E', 'E_min', 'Eb', 'Eb_min', 'F', 'F#', 'F#_min', 'F_min', 'G', 'G#', 'G#_min', 'G_min', 'Gb', 'Gb_min']
mu_dic = dict.fromkeys(all_chords)
cov_dic = dict.fromkeys(all_chords)
mu_matrix = []
cov_matrix = []
#chromagrams_path = 'data/chromagrams'
librosa_chromagrams_path = 'data/librosa_chromagrams'
i = int(0)


# ------------------------------------------------------------------------------------------
# CALCULATE MU AND SIGMA
# ------------------------------------------------------------------------------------------
print('calculate mu and sigma')
for elem in sorted(os.listdir('data/chromagrams/')):
    mu_values = []
    #temp_path = f'{chromagrams_path}/{elem}'
    temp_path = f'{librosa_chromagrams_path}/{elem}'
    temp_df = pd.read_csv(temp_path)
    temp_df = temp_df.iloc[:, 1:]
    mu_array, states_cov_matrices = get_mu_sigma_from_chroma(temp_df, notes)
    for i in np.arange(0, len(mu_array)):
        mu_values.append(mu_array[i])
    mu_matrix.append(mu_values)
    cov_matrix.append(states_cov_matrices)
    mu_dic[all_chords[i]] = mu_array
    cov_dic[all_chords[i]] = states_cov_matrices
    i += 1

# ------------------------------------------------------------------------------------------
# PLOT MU FOR EVERY CHORD
# ------------------------------------------------------------------------------------------


for i in range(len(mu_matrix)):
    f, axes = plt.subplots(1, 1)
    axes.plot(mu_matrix[i])
    plt.title(all_chords[i])
    y_pos = np.arange(len(notes))
    plt.xticks(y_pos, notes)
    plt.savefig('/Users/PilvioSol/Desktop/librosa_mu_plots2/' + all_chords[i] + '.png')
    plt.show()

# ------------------------------------------------------------------------------------------
# TRANSITION CALCULATED ON CHORD CHANGE FROM .LAB FILE
# ------------------------------------------------------------------------------------------
print('transition calculated on chord change from .lab file')
tp_matrix = []


for i, val in enumerate(chord_annotation_dic):
    first_chord = chord_annotation_dic[i]['chord'].values[:-1].tolist()
    second_chord = chord_annotation_dic[i]['chord'][1:].tolist()
    tp_matrix.append(transition_prob_matrix(first_chord, second_chord))


test = pd.DataFrame(0, index=np.arange(len(all_chords)), columns=all_chords)
test_count = pd.DataFrame(0, index=np.arange(len(all_chords)), columns=all_chords)
for k in np.arange(0, len(tp_matrix)):
    temp_tp = tp_matrix[k]
    for i in np.arange(0, len(temp_tp.columns)):
        for j in np.arange(0, len(temp_tp.index)):
            test.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] = \
                test.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] + \
                temp_tp.iloc[j,i]
            if temp_tp.iloc[j, i] > 0:
                test_count.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] = \
                    test_count.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] + 1

test = pd.DataFrame(np.matrix(test.iloc[:, :]), index=all_chords, columns=all_chords)
test_count = pd.DataFrame(np.matrix(test_count.iloc[:,:]), index=all_chords, columns=all_chords)
res_tp = test.div(test_count)
res_tp = res_tp.fillna(0)
res_tp = res_tp.div(res_tp.sum(axis=1), axis=0)
# ------------------------------------------------------------------------------------------
# INITIAL STATE MATRIX
# ------------------------------------------------------------------------------------------

print('initial state matrix')
in_matrix = []
for i in range(29):
    in_matrix.append(1/29)


# ------------------------------------------------------------------------------------------
# TRAIN GAUSSIAN HMM
# ------------------------------------------------------------------------------------------

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


h_markov_model = build_gaussian_hmm(in_matrix, res_tp, mu_matrix, cov_matrix)


# ------------------------------------------------------------------------------------------
# PREDICTION FROM CHROMA
# ------------------------------------------------------------------------------------------
'''
chroma_dic = readcsv_chroma(path_CQT_csv, notes)

chord_ix_predictions = h_markov_model.predict(chroma_dic[6])
print('HMM output predictions:')
print(chord_ix_predictions[:50])

chord_pred = []
for i in chord_ix_predictions:
    chord_pred.append(chords_mapper(i, all_chords))


# ------------------------------------------------------------------------------------------
# PREDICTION TO MIDI
# ------------------------------------------------------------------------------------------

mid_pred = mido.MidiFile()
mid_pred_trck = mido.MidiTracks()

note_dur = 192

all_chords_mid = dict.fromkeys(all_chords)
all_chords_mid = midi_chord_creator(all_chords_mid)

midi_predictor(mid_pred, mid_pred_trck, chord_pred, all_chords_mid, note_dur) #save the midi file

'''
