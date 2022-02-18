from main import *
from chroma_extractor import *

"""
    QUA DOBBIAMO FARE ANCORA DELLE PROVE CON LE NOSTRE CHROMAGRAM (E ANCHE QUELLE DI LIBROSA)
"""


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
song_path = '/Users/PilvioSol/Desktop/Am_C_G_Em.wav'


# ------------------------------------------------------------------------------------------
# READ FILE AND PERFORM CHROMAGRAM EXTRACTION
# ------------------------------------------------------------------------------------------
print('READ FILE AND PERFORM CHROMAGRAM EXTRACTION.....')
cqt_test = Librosa_extract_features(song_path)
cqt_test_transpose = cqt_test.transpose()
cqt_df = pd.DataFrame(cqt_test_transpose)


# ------------------------------------------------------------------------------------------
# MAKE PREDICTIONS AND PRINT THEM
# ------------------------------------------------------------------------------------------
hmm_state_predictions = h_markov_model.predict(cqt_df)
print('HMM output predictions:')
print(hmm_state_predictions[:50])

chord_predictions = []
for i in hmm_state_predictions:
    chord_predictions.append(chords_mapper(i, all_chords))
print(chord_predictions)

