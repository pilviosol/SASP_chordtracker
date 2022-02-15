#from extractor import *
import os  # provides functions for interacting with the operating system
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

print('jhg')