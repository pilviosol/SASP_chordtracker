import csv

with open('data/chords_dictionary.csv', mode='r') as inp:
    reader = csv.reader(inp)
    dict_from_csv = {rows[0] for rows in reader}

print('cazzi')