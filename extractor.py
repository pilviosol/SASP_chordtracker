import numpy as np
from extractor_functions import readlab, readcsv_chroma, chord_chroma_raws


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
path_lab = 'data/Beatles_lab_tuned/'
# path_CQT_csv = 'data/Beatles_CQT_csv/'
librosa_path_CQT_csv = 'data/Librosa_Beatles_CQT_csv/'
notes = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
win_size_t = 2048/44100


# ------------------------------------------------------------------------------------------
# CREATE DICTIONARY WITH ALL CHROMAS
# ------------------------------------------------------------------------------------------
# Dictionary with the same form of the lab (start, end, chord) with all songs
chord_annotation_dic, song_list = readlab(path_lab)

# Dictionary with all songs in the form of chromagram (12 energies per note for each time window)
librosa_chroma_dic = readcsv_chroma(librosa_path_CQT_csv, notes)
# chroma_dic = readcsv_chroma(path_CQT_csv, notes)


# ------------------------------------------------------------------------------------------
# INSERT CHORD TO THE LAST CHROMA ROW
# ------------------------------------------------------------------------------------------
chroma_dic_with_chords = []
librosa_chroma_dic_with_chords = []
librosa_tp_list = []

# add as last raw the name of the chord in that specific window
for idx in range(len(chord_annotation_dic)):
    print(idx)
    # chroma_dic_with_chords.append(chord_chroma_raws(chroma_dic[idx], chord_annotation_dic[idx], win_size_t))
    librosa_chroma_dic_with_chords.append(chord_chroma_raws(librosa_chroma_dic[idx], chord_annotation_dic[idx], win_size_t))


# save the new chroma dictionary as n csv files as it is a list of dataframe
for i in np.arange(0, len(librosa_chroma_dic_with_chords)):
    # chroma_dic_with_chords[i].to_csv('data/chroma_dic_new_csvs/chroma_dic_new_ele'+ str(i), index=False)
    librosa_chroma_dic_with_chords[i].to_csv('data/librosa_chroma_dic_new_csvs/chroma_dic_new_ele' + str(i), index=False)


# ------------------------------------------------------------------------------------------
# CALCULATION OF CSVS OF CHROMAGRAM GROUPED BY CHORD, SCANNING ALL SONGS (NEEDED TO CALCULATE MU AND COV)
# TO BE RUNNED ONLY ONE TIME TO GENERATE THE CSVS --> NOW COMMENTED.
# ------------------------------------------------------------------------------------------
# calculate_chromagrams_csvs_by_chord('data/chroma_dic_new_csvs')
# calculate_chromagrams_csvs_by_chord('data/librosa_chroma_dic_new_csvs')

