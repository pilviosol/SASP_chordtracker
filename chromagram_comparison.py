from chroma_extractor import stft_basic, F_coef, F_pitch, P, compute_Y_LF, compute_chromagram
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
file_path = '/Users/PilvioSol/Desktop/Scala.wav'


# ------------------------------------------------------------------------------------------
# LOADING OF FILE
# ------------------------------------------------------------------------------------------
y, sr = librosa.load(file_path, mono=True)


# ------------------------------------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------------------------------------
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, mono=True)
    H = 1024
    N = 2048
    Fs = sr*2
    w = np.hanning(N)
    X = stft_basic(audio, w, H)
    Hmn, Prs = librosa.decompose.hpss(X)
    Y = np.abs(Hmn)
    Y_LF, F_coef_pitch = compute_Y_LF(Y, Fs, N)
    cqt = compute_chromagram(Y_LF)
    return X, cqt


# Librosa Chromagram from signal
librosa_chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=1024, window='hann', win_length=2048)

# Librosa STFT from signal
librosa_stft = np.abs(librosa.stft(y, hop_length=1024, window='hann'))

# STFT and Chromagram hand made
stft_handmade, chroma_handmade = extract_features(file_path)

# Librosa Chromagram from STFT
chroma_from_X = librosa.feature.chroma_stft(S=librosa_stft, sr=sr, n_fft=2048, hop_length=1024, window='hann', win_length=2048)


# ------------------------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.amplitude_to_db(chroma_handmade, ref=np.max),
                               y_axis='chroma', x_axis='time', ax=ax[0])
ax[0].label_outer()
img = librosa.display.specshow(librosa_chroma, y_axis='chroma', x_axis='time', ax=ax[1])
plt.show()


fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.amplitude_to_db(librosa_stft, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
ax[0].label_outer()
img = librosa.display.specshow(librosa.amplitude_to_db(stft_handmade, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
plt.show()

