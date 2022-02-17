from main import *
from chroma_extractor import *

song_path = '/Users/PilvioSol/Desktop/Am_C_G_Em.wav'

# ------------------------------------------------------------------------------------------
# READ FILE AND PERFORM CHROMAGRAM EXTRACTION
# ------------------------------------------------------------------------------------------
print('READ FILE AND PERFORM CHROMAGRAM EXTRACTION.....')
cqt_test = extract_features(song_path)
cqt_test_transpose = cqt_test.transpose()
cqt_df = pd.DataFrame(cqt_test_transpose)


print('ollare')




# ------------------------------------------------------------------------------------------
# MAKE PREDICTIONS
# ------------------------------------------------------------------------------------------
chord_ix_predictions = h_markov_model.predict(cqt_df)
print('HMM output predictions:')
print(chord_ix_predictions[:50])

print('ollare')

