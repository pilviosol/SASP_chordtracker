import numpy as np
import pandas as pd
import os
from hmmlearn import hmm


# ------------------------------------------------------------------------------------------
# CALCULATE MU AND SIGMA
# ------------------------------------------------------------------------------------------
def __get_mu_array(note_feature_vector, notes):
    """

    Args:
        note_feature_vector: chromagram
        notes: array with 12 notes

    Returns: mean by column (for each note)

    """
    return note_feature_vector[notes].mean()


def get_mu_sigma_from_chroma(chromagram, notes):
    """

    Args:
        chromagram: chromagram
        notes: array with 12 notes

    Returns: mu array and cov matrix array

    """
    mu = __get_mu_array(chromagram, notes)

    states_cov = chromagram.cov().values
    states_cov = np.array(states_cov)

    return [mu, states_cov]


# ------------------------------------------------------------------------------------------
# CALCULATE TRANSITION PROBABILITY MATRIX
# ------------------------------------------------------------------------------------------
def __calc_prob_chordpairs(chord_group):  # probability of having one chord after the other based on .lab files
    """

    Args:
        chord_group: group of chords of a song

    Returns:
        chord_group_count: probability of having a chord after one other for all chords in a song

    """

    chord_group_count = chord_group.groupby('second_chord').size().reset_index()
    chord_group_count.columns = ['second_chord', 'count']
    total = chord_group_count['count'].sum()
    chord_group_count['transition_prob'] = chord_group_count['count']/total

    return chord_group_count


def transition_prob_matrix(firstchord, secondchord):
    """

    Args:
        firstchord: one chord
        secondchord: another chord, to be calculated the transition from firstchord

    Returns:
        prob_matrix: matrix with probabilities of passing from a chord to another

    """

    sequence_chords = pd.DataFrame({'first_chord': firstchord, 'second_chord': secondchord})
    prob_matrix = sequence_chords.groupby('first_chord').apply(__calc_prob_chordpairs).reset_index()
    prob_matrix = prob_matrix.drop('level_1', axis=1)
    prob_matrix = prob_matrix.pivot(index='first_chord', columns='second_chord', values='transition_prob')
    prob_matrix = prob_matrix.fillna(0)

    return prob_matrix


# ------------------------------------------------------------------------------------------
# CREATE THE DICTIONARY WITH ALL THE CHROMA REGROUPED BY CHORDS
# ------------------------------------------------------------------------------------------
def calculate_chromagrams_csvs_by_chord(path):
    """

    Args:
        path: path where saved all song's csvs of chromagrams with column 'chord' appended

    Returns: all csvs of chromagrams divided by chord in the folder "data/chromagrams"

    """

    # import the new chroma dictionary csv list
    chroma_dic_path = path  # 'data/chroma_dic_new_csvs'
    chroma_dic_new_list = []
    for elem in sorted(os.listdir(chroma_dic_path)):
        temp_path = f'{chroma_dic_path}/{elem}'
        temp_df = pd.read_csv(temp_path)
        chroma_dic_new_list.append(temp_df)

    all_chords = []
    for songs in chroma_dic_new_list:
        for elements in sorted(songs['chord']):
            if elements not in all_chords:
                all_chords.append(elements)
    all_chords = sorted(all_chords)
    print(all_chords)

    chords_dictionary = {}
    for chords in all_chords:
        name = str(chords)
        chords_dictionary[name] = []

    for chord in all_chords:
        print('processing chord: ', chord)
        for song in chroma_dic_new_list:
            for row, index in song.iterrows():
                if index['chord'] == chord:
                    chords_dictionary[chord].append(index)

    for chord in all_chords:
        list_frames = []
        print('processing chord2: ', chord)
        for i in np.arange(0, len(chords_dictionary[chord])):
            list_frames.append(chords_dictionary[chord][i])
        pandas_frame = pd.DataFrame(list_frames,
                                    columns=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        # pandas_frame.to_csv('data/chromagrams/chords_dictionary_chroma_' + str(chord))
        pandas_frame.to_csv('data/librosa_chromagrams/chords_dictionary_chroma_' + str(chord))


# ------------------------------------------------------------------------------------------
# GAUSSIAN HMM MODEL DEFINITION
# ------------------------------------------------------------------------------------------
def build_gaussian_hmm(initial_state_prob, transition_matrix, mu_array, states_cov_matrices):
    """

    Args:
        initial_state_prob: initial state probability matrix
        transition_matrix: transition probability matrix
        mu_array: mean array by chroma for each song
        states_cov_matrices: cov matrix for each song

    Returns: hidden markov model object

    """
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
