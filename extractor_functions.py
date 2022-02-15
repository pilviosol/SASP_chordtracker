import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import \
    re

plt.style.use('seaborn')


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
        chord_annotation.loc[chord_annotation['chord'] == 'N', 'chord'] = chord_annotation['chord'].mode()[0]
                # replace silence by probable
                # tonal end
        dictionary.append(chord_annotation)

    return dictionary, list_name


# ------------------------------------------------------------------------------------------
# READ CQT CSV
# ------------------------------------------------------------------------------------------

def readcsv_chroma(path, notes):
    '''

    Args:
        notes: the notes name array
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
# INSERT CHORD TO THE LAST CHROMA ROW
# ------------------------------------------------------------------------------------------
def chord_chroma_raws(chroma, chord_annotation, win_size_t):
    '''

    Args:
        win_size_t: window size of the chroma
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
        # print('i: ', ii)
        # print('win_size_t*np.float(ii+1): ', win_size_t*np.float(ii+1))
        # print('chord_annotation[end][raw]: ', chord_annotation['end'][raw])
        if win_size_t*np.float(ii+1) < chord_annotation['end'][raw]:
            chroma.loc[ii, 'chord'] = chord_annotation['chord'][raw]
            # print('if')
        else:
            chroma.loc[ii, 'chord'] = chord_annotation['chord'][raw]
            raw += 1
            # print('else')

    return chroma


# ------------------------------------------------------------------------------------------
# CALCULATE MU AND SIGMA
# ------------------------------------------------------------------------------------------

def __get_mu_array(note_feature_vector, notes):
    return note_feature_vector[notes].mean()


def get_mu_sigma_from_chroma(chromagram, notes):
    mu = __get_mu_array(chromagram, notes)

    states_cov = chromagram.cov().values
    states_cov = np.array(states_cov)

    return [mu, states_cov]


# ------------------------------------------------------------------------------------------
# CALCULATE TRANSITION PROBABILITY MATRIX
# ------------------------------------------------------------------------------------------

def __calc_prob_chordpairs(chord_group): # probability of having one chord after the other based on .lab files
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
