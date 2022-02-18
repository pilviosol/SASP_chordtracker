from main import *
from chroma_extractor import *

"""
    QUA DOBBIAMO FARE ANCORA DELLE PROVE CON LE NOSTRE CHROMAGRAM (E ANCHE QUELLE DI LIBROSA)
"""


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
#song_path = '/Users/PilvioSol/Desktop/Am_C_G_Em.wav'
song_path = 'data/Twist_And_Shout.wav'


# ------------------------------------------------------------------------------------------
# READ FILE AND PERFORM CHROMAGRAM EXTRACTION
# ------------------------------------------------------------------------------------------
print('READ FILE AND PERFORM CHROMAGRAM EXTRACTION.....')
# cqt_test = Librosa_extract_features(song_path)
cqt_test = extract_features(song_path)
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


# ------------------------------------------------------------------------------------------
# PREDICTION TO MIDI (SAVED AS data/prediction.mid)
# ------------------------------------------------------------------------------------------
mid_pred = mido.MidiFile()
mid_pred_trck = mido.MidiTrack()
note_dur = win_size_t

all_chords_dic = dict.fromkeys(all_chords)
all_chords_mid = midi_chord_creator(all_chords_dic)

# Create and save the midi file
midi_predictor(mid_pred, mid_pred_trck, chord_predictions, all_chords_mid, note_dur=46)

print('debug')