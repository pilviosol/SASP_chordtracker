import pandas as pd

dict_from_csv = pd.read_csv('data/chords_dictionary.csv', on_bad_lines='skip', sep=',',header=26).to_dict()
print('ollare')