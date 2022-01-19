import librosa
import pathlib

path = "/Users/PilvioSol/Desktop/Beatles_wav/"

files_in_basepath = pathlib.Path(path)
songs_path = files_in_basepath.iterdir()

for song in sorted(songs_path):
    name = song.name
    y, sr = librosa.load(song)
    duration = librosa.get_duration(y=y, sr=sr)
    print('Song: ', name, 'Duration', duration)

