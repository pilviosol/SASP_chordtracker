from os import path
from pydub import AudioSegment
import pathlib
# files
src = "/Users/PilvioSol/Desktop/Beatles_mp3/"
dst = "/Users/PilvioSol/Desktop/Beatles_wav/"

# convert wav to mp3
files_in_basepath = pathlib.Path(src)
songs_path = files_in_basepath.iterdir()

for song in songs_path:
    print(song.name)
    dst = "/Users/PilvioSol/Desktop/Beatles_wav/" + song.name[0:-4] + ".wav"
    sound = AudioSegment.from_mp3(song)
    sound.export(dst, format="wav")