import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import librosa, librosa.display
import pathlib
plt.style.use('seaborn')

import \
    re  # regular expression, is a special sequence of characters that helps you match or find other strings or sets of strings



# ------------------------------------------------------------------------------------------
# DEFINIZIONE PATH E DEFINIZIONE DIRECTORY PER CHROMAGRAMS
# ------------------------------------------------------------------------------------------


_SAMPLING_RATE = 44100

path_files = "/Users/PilvioSol/Desktop/Beatles_new_wav/"
# path_chromagrams = "/Users/PilvioSol/Desktop/progetto/codice/data/chromagrams/"
# path_files = "D:/Uni/First year/Second Semester/Sound Analysis/Project Chord detection/Beatles_wav/"
# path_chromagrams = "E:/Uni/First year/Second Semester/Sound Analysis/Project Chord detection/Chromagrams/"
path_csv = "data/Beatles_csv/"


# ------------------------------------------------------------------------------------------
# FUNZIONI PER CALCOLARE LE CHROMAGRAM
# ------------------------------------------------------------------------------------------

def stft_basic(x, w, H=8):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)

    Args:
        x: Signal to be transformed
        w: Window function
        H: Hopsize

    Returns:
        X: The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int)
    X = np.zeros((N, M + 1), dtype='complex')
    for m in range(M + 1):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win
    K = (N + 1) // 2
    X = X[:K, :]
    return X


def F_coef(k, Fs, N):
    """Computes the center frequency/ies of a Fourier coefficient

    Args:
        k: Fourier coefficient index
        Fs: Sampling rate
        N: Window size of Fourier fransform

    Returns:
        im: Frequency value(s)
    """
    return k * Fs / N


def F_pitch(p, pitch_ref=69, freq_ref=440):
    """Computes the center frequency/ies of a MIDI pitch

    Args:
        p: MIDI pitch value(s)
        pitch_ref: Reference pitch (default: 69)
        freq_ref: Frequency of reference pitch (default: 440.0)

    Returns:
        im: Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref


def P(p, Fs, N, pitch_ref=69, freq_ref=440):
    """Computes the set of frequency indices that are assigned to a given pitch

    Args:
        p: MIDI pitch value
        Fs: Sampling rate
        N: Window size of Fourier fransform
        pitch_ref: Reference pitch (default: 69)
        freq_ref:  Frequency of reference pitch (default: 440.0)

    Returns:
        im: Set of frequency indices
    """
    lower = F_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = F_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]


def compute_Y_LF(Y, Fs, N):
    """Computes a log-frequency spectrogram

    Args:
        Y: Magnitude or power spectrogram
        Fs: Sampling rate
        N: Window size of Fourier fransform
        pitch_ref: Reference pitch (default: 69)
        freq_ref: Frequency of reference pitch (default: 440.0)

    Returns:
        Y_LF: Log-frequency spectrogram
        F_coef_pitch: Pitch values
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = P(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(128)
    return Y_LF, F_coef_pitch



def compute_chromagram(Y_LF):
    """Computes a chromagram

    Args:
        Y_LF: Log-frequency spectrogram

    Returns:
        C: Chromagram
    """
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)
    return C


# ------------------------------------------------------------------------------------------
# FUNZIONE PER ESTRARRE LE FEATURES (CHROMAGRAM)
# ------------------------------------------------------------------------------------------


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, mono=True)

        #cqt = librosa.feature.chroma_cqt(y=Hmn, sr=sample_rate)
        H = 1024
        N = 2048
        Fs = _SAMPLING_RATE
        w = np.hanning(N)
        X = stft_basic(audio, w, H)
        Hmn, Prs = librosa.decompose.hpss(X)
        eps = np.finfo(float).eps
        Y = 20 * np.log10(eps + np.abs(Hmn) ** 2)
        Y_LF, F_coef_pitch = compute_Y_LF(Y, Fs, N)
        cqt = compute_chromagram(Y_LF)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return cqt







# ------------------------------------------------------------------------------------------
# CQT.CSV EXTRACTION PER TUTTI I FILE DEL DATASET
# ------------------------------------------------------------------------------------------

files_in_basepath = pathlib.Path(path_files)
songs_path = files_in_basepath.iterdir()

for song in songs_path:
    if (str(song).endswith('.wav') and song.is_file()):

        print(song)
        features = extract_features(song)
        print(features)
        name_csv = song.name[0:-4] + '_CQT.csv'
        np.savetxt(path_csv + name_csv, features, delimiter=",")

        '''
        name_cqt = song.name[0:-4] + '_CQT.png'
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(features, ref=np.max),
                                       sr=_SAMPLING_RATE, ax=ax)
        ax.set_title(name_cqt)
        # fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        fig.savefig(path_chromagrams + name_cqt)  
        '''

    else:
        print('thats not a mp3 file')


