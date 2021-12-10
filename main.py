import os # provides functions for interacting with the operating system
import shutil # move files and delete folders with files
import tarfile # used to read and write tar archives (contains uncompressed byte streams of the files which it contains)
import urllib.request # download files folder
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import IPython
import IPython.display as ipd
import librosa, librosa.display
import re # regular expression, is a special sequence of characters that helps you match or find other strings or sets of strings
import IPython # listen to sounds on Python
# import pretty_midi # contains utility function/classes for handling MIDI data, !install only if u want to listen the predictions!

from scipy.io import wavfile
from scipy.spatial import distance_matrix
from matplotlib import colors
from itertools import product

#!pip install hmmlearn # need the installation
from hmmlearn import hmm
from sklearn.metrics import f1_score
import pathlib
from pathlib import Path
import sklearn
plt.style.use('seaborn')
#from __future__ import division
import scipy
from scipy.signal import hamming
from scipy.fftpack import fft

# np.genfromtxt(path_csv+name_csv, delimiter=',')

folder = '/content/drive/MyDrive/Colab Notebooks/Chord Detector/DataLab'

chord_annotation_dic = [] # to append mutliple dataframe without concatenating them i have to insert them as element of a list
song_list = []


# simplify the chord notation
def __simplify_chords(chords_df): # silence
    chords_processed = chords_df['chord'].str.split(':maj')                         # remove major x chords return array of array
                                                                                    # containing all the chords
    chords_processed = [elem[0] for elem in chords_processed]                       # further process step above to return 1 array
                                                                                    # take the first element in all the N arrays (chords
                                                                                    # string) to make it a list of N elements
    chords_processed = [elem.split('/')[0] for elem in chords_processed]            # remove inverted chords
    chords_processed = [elem.split('aug')[0] for elem in chords_processed]          # remove augmented chords
    chords_processed = [elem.split(':(')[0] for elem in chords_processed]           # remove added notes chords
    chords_processed = [elem.split('(')[0] for elem in chords_processed]            # remove added notes chords 2
    chords_processed = [elem.split(':sus')[0] for elem in chords_processed]         # remove sustained chords
    chords_processed = [re.split(":?\d", elem)[0] for elem in chords_processed]     # remove added note
    chords_processed = [elem.replace('dim', 'min') for elem in chords_processed]    # change diminute to minor
    chords_processed = [elem.replace('hmin', 'min') for elem in chords_processed]   # change semi-diminute to minor
    chords_processed = [re.split(":$", elem)[0] for elem in chords_processed]       # remove added notes chords
    return chords_processed


for elem in os.listdir(folder):
  song_path = f'{folder}/{elem}'
  song_list.append(elem)
  chord_annotation = pd.read_csv(song_path, sep=' ', header=None)                                       # no header in the files and separated
                                                                                                        # by a empty space
  # set up the columns of the file
  chord_annotation.columns = ['start','end','chord']
  chord_annotation['chord'] = __simplify_chords(chord_annotation)
  chord_annotation.loc[chord_annotation['chord'] == 'N', 'chord'] = chord_annotation['chord'].mode()[0] # replace silence by probable
                                                                                                        # tonal end
  chord_annotation_dic.append(chord_annotation)

print(song_list[0])
chord_annotation_dic[0]