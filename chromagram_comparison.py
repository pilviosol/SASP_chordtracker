import math

from chroma_extractor import stft_basic, F_coef, F_pitch, P, compute_Y_LF, compute_chromagram

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('/Users/PilvioSol/Desktop/00_A_Day_In_The_Life.wav', mono=True)


def extract_features(file_name):

    audio, sample_rate = librosa.load(file_name, mono=True)
    #cqt = librosa.feature.chroma_cqt(y=Hmn, sr=sample_rate)
    H = 1024
    N = 2048
    Fs = sr*2
    w = np.hanning(N)
    X = stft_basic(audio, w, H)
    Hmn, Prs = librosa.decompose.hpss(X)
    eps = np.finfo(float).eps
    Y = np.abs(Hmn)
    Y_LF, F_coef_pitch = compute_Y_LF(Y, Fs, N)
    cqt = compute_chromagram(Y_LF)
    return X, cqt


chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=1024, window='hann', win_length=2048)
stft = librosa.stft(y, hop_length=1024, window='hann')
X, chroma_clara = extract_features('/Users/PilvioSol/Desktop/00_A_Day_In_The_Life.wav')



fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(chroma_clara,
                               y_axis='chroma', x_axis='time', ax=ax[0])
ax[0].label_outer()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
plt.show()


fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
ax[0].label_outer()
img = librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
plt.show()




