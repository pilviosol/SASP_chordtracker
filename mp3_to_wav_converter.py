from pydub import AudioSegment
import pathlib


# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
src = "/Users/PilvioSol/Desktop/Beatles_new/"
dst = "/Users/PilvioSol/Desktop/Beatles_new_wav/"

# ------------------------------------------------------------------------------------------
# CONVERT MP3 TO WAV
# ------------------------------------------------------------------------------------------
files_in_basepath = pathlib.Path(src)
songs_path = files_in_basepath.iterdir()

for song in sorted(songs_path):
    print(song.name)
    dst = "/Users/PilvioSol/Desktop/Beatles_new_wav/" + song.name[0:-4] + ".wav"
    print(dst)
    sound = AudioSegment.from_mp3(song)
    sound.export(dst, format="wav")