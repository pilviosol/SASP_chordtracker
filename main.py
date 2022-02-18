from extractor import *
import os
import numpy as np
import pandas as pd
import mido
from function import get_mu_sigma_from_chroma, transition_prob_matrix, build_gaussian_hmm
from utils import column_index, chords_mapper
from midi_creator import midi_chord_creator, midi_predictor


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
all_chords = ['A', 'A_min', 'Ab', 'B', 'B_min', 'Bb', 'Bb_min', 'C', 'C#', 'C#_min', 'C_min', 'D', 'D#_min', 'D_min',
              'Db', 'E', 'E_min', 'Eb', 'Eb_min', 'F', 'F#', 'F#_min', 'F_min', 'G', 'G#', 'G#_min', 'G_min', 'Gb',
              'Gb_min']
mu_matrix = []
cov_matrix = []
chromagrams_path = 'data/chromagrams'
librosa_chromagrams_path = 'data/librosa_chromagrams'
plot_mu_path = '/Users/PilvioSol/Desktop/librosa_mu_plots2/'
i = int(0)
tp_matrix = []


# ------------------------------------------------------------------------------------------
# CALCULATE MU AND SIGMA
# ------------------------------------------------------------------------------------------
print('calculating mu and sigma.....')
for elem in sorted(os.listdir('data/chromagrams/')):
    mu_values = []
    # temp_path = f'{chromagrams_path}/{elem}'
    temp_path = f'{librosa_chromagrams_path}/{elem}'
    temp_df = pd.read_csv(temp_path)
    temp_df = temp_df.iloc[:, 1:]
    mu_array, states_cov_matrices = get_mu_sigma_from_chroma(temp_df, notes)
    for i in np.arange(0, len(mu_array)):
        mu_values.append(mu_array[i])
    mu_matrix.append(mu_values)
    cov_matrix.append(states_cov_matrices)
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
    plt.savefig(plot_mu_path + all_chords[i] + '.png')
    # plt.show()
    plt.close()

# ------------------------------------------------------------------------------------------
# TRANSITION MATRIX FOR ALL SONGS COMBINED CALCULATION
# ------------------------------------------------------------------------------------------
print('Transition matrix for all songs combined calculation.....')
"""
The starting point is the dictionary tp_matrix that has the transition probability matrix 
for each song in the dataset
Then we have the calculation of sum_df (sum of all transition probabilities from one chord to another for every 
song) and count_df (in how many songs we have a transition from one chord to another).
The transition probability matrix is sum_df/count_df with row normalization.
"""

for i, val in enumerate(chord_annotation_dic):
    first_chord = librosa_chroma_dic_new[i]['chord'].values[:-1].tolist()
    second_chord = librosa_chroma_dic_new[i]['chord'][1:].tolist()
    tp_matrix.append(transition_prob_matrix(first_chord, second_chord))


sum_df = pd.DataFrame(0, index=np.arange(len(all_chords)), columns=all_chords)
count_df = pd.DataFrame(0, index=np.arange(len(all_chords)), columns=all_chords)


for k in np.arange(0, len(tp_matrix)):
    temp_tp = tp_matrix[k]
    for i in np.arange(0, len(temp_tp.columns)):
        for j in np.arange(0, len(temp_tp.index)):
            sum_df.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] = \
                sum_df.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] + \
                temp_tp.iloc[j, i]
            if temp_tp.iloc[j, i] > 0:
                count_df.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] = \
                    count_df.iloc[column_index(all_chords, temp_tp)[j], column_index(all_chords, temp_tp)[i]] + 1


sum_df = pd.DataFrame(np.matrix(sum_df.iloc[:, :]), index=all_chords, columns=all_chords)
count_df = pd.DataFrame(np.matrix(count_df.iloc[:, :]), index=all_chords, columns=all_chords)

# Resulting transition_probability_matrix
transition_probability_matrix = sum_df.div(count_df)
# Replacing Nan with 0s
transition_probability_matrix = transition_probability_matrix.fillna(0)
# Row-wise normalization
transition_probability_matrix = transition_probability_matrix.div(transition_probability_matrix.sum(axis=1), axis=0)

# ------------------------------------------------------------------------------------------
# INITIAL STATE MATRIX
# ------------------------------------------------------------------------------------------
print('Initial state matrix.....')
initial_state_matrix = []
for i in range(29):
    initial_state_matrix.append(1 / 29)


# ------------------------------------------------------------------------------------------
# TRAINING GAUSSIAN HMM
# ------------------------------------------------------------------------------------------
print('Training Gaussian Hmm.....')
h_markov_model = build_gaussian_hmm(initial_state_matrix, transition_probability_matrix, mu_matrix, cov_matrix)


# ------------------------------------------------------------------------------------------
# PREDICTION FROM BEATLES PRE-EXISTING CHROMAGRAMS
# ------------------------------------------------------------------------------------------
chroma_dic = readcsv_chroma(librosa_path_CQT_csv, notes)

hmm_state_predictions = h_markov_model.predict(chroma_dic[36])
print('HMM output predictions:')
print(hmm_state_predictions[:50])

chord_predictions = []
for i in hmm_state_predictions:
    chord_predictions.append(chords_mapper(i, all_chords))

print(chord_predictions)


# ------------------------------------------------------------------------------------------
# PREDICTION TO MIDI (SAVED AS data/prediction.mid)
# ------------------------------------------------------------------------------------------
mid_pred = mido.MidiFile()
mid_pred_trck = mido.MidiTrack()
note_dur = win_size_t

all_chords_dic = dict.fromkeys(all_chords)
all_chords_mid = midi_chord_creator(all_chords_dic)

# Create and save the midi file
midi_predictor(mid_pred, mid_pred_trck, chord_predictions, all_chords_mid, note_dur)

print('End of main!')
