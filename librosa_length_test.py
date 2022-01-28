import librosa
import pathlib

path = "/Users/PilvioSol/Desktop/Beatles_new_new/"
path_lab = 'data/Beatles_lab_tuned/'

files_in_basepath = pathlib.Path(path)
songs_path = files_in_basepath.iterdir()

print('SONGS')

for song in sorted(songs_path):
    name = song.name
    #y, sr = librosa.load(song)
    #duration = librosa.get_duration(y=y, sr=sr)
    #print('Song: ', name, 'Duration', duration)
    print(name)


files_in_basepath = pathlib.Path(path_lab)
lab_path = files_in_basepath.iterdir()


print('LABS')
for file in sorted(lab_path):
    name = file.name
    #y, sr = librosa.load(song)
    #duration = librosa.get_duration(y=y, sr=sr)
    #print('Song: ', name, 'Duration', duration)
    print(name)
