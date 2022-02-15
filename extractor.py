import os  # provides functions for interacting with the operating system
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from extractor_functions import readlab, readcsv_chroma, chord_chroma_raws, get_mu_sigma_from_chroma, \
    transition_prob_matrix
# from hmmlearn import hmm

plt.style.use('seaborn')

# watch out to the other useful import


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
path_lab = 'data/Beatles_lab_tuned/'
path_CQT_csv = 'data/Beatles_CQT_csv/'
notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
win_size_t = 2048/44100


# ------------------------------------------------------------------------------------------
# CREATE DICTIONARY WITH ALL CHROMAS
# ------------------------------------------------------------------------------------------

# dictionary with the same form of the lab
chord_annotation_dic, song_list = readlab(path_lab)

# create a chroma dic with chromas of all songs
chroma_dic = readcsv_chroma(path_CQT_csv, notes)


# ------------------------------------------------------------------------------------------
# INSERT CHORD TO THE LAST CHROMA ROW
# ------------------------------------------------------------------------------------------

chroma_dic_new = []

# add as last raw the name of the chord in that specific window
for idx in range(len(chord_annotation_dic)):
    print(idx)
    chroma_dic_new.append(chord_chroma_raws(chroma_dic[idx], chord_annotation_dic[idx], win_size_t))


# save the new chroma dictionary as n csv files as it is a list of dataframe
for i in np.arange(0, len(chroma_dic_new)):
    chroma_dic_new[i].to_csv('data/chroma_dic_new_csvs/chroma_dic_new_ele'+ str(i), index=False)


# ------------------------------------------------------------------------------------------
# CREATE THE DICTIONARY WITH ALL THE CHROMA REGROUPED BY CHORDS
# ------------------------------------------------------------------------------------------


# import the new chroma dictionary csv list
chroma_dic_path = 'data/chroma_dic_new_csvs'
chroma_dic_new_list = []
for elem in sorted(os.listdir(chroma_dic_path)):
    temp_path = f'{chroma_dic_path}/{elem}'
    temp_df= pd.read_csv(temp_path)
    chroma_dic_new_list.append(temp_df)


all_chords = []
for songs in chroma_dic_new_list:
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
    for song in chroma_dic_new_list:
        for row, index in song.iterrows():
            if index['chord'] == chord:
                chords_dictionary[chord].append(index)


for chord in all_chords:
    list_frames = []
    print('processing chord2: ', chord)
    for i in np.arange(0, len(chords_dictionary[chord])):
        list_frames.append(chords_dictionary[chord][i])
    pandas_frame = pd.DataFrame(list_frames, columns =["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    pandas_frame.to_csv('data/chromagrams/chords_dictionary_chroma_' + str(chord))


# ------------------------------------------------------------------------------------------
# CALCULATE MU AND SIGMA
# ------------------------------------------------------------------------------------------





