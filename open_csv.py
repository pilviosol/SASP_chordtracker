import pandas as pd

dict_from_csv = pd.read_csv('data/chords_dictionary.csv', on_bad_lines='skip', header=None, index_col=0, squeeze=True, sep=',').to_dict()
