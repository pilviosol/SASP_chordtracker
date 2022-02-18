from midi2audio import FluidSynth
from pysndfx import AudioEffectsChain
import pathlib
# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
path_soundfont = "soundfonts/"
files_in_basepath = pathlib.Path(path_soundfont)
sound_path = files_in_basepath.iterdir()


# ------------------------------------------------------------------------------------------
# EFFECT DEFINITION AND APPLICATION
# ------------------------------------------------------------------------------------------
fx = (
    AudioEffectsChain()
    # .highshelf()
    .lowpass(frequency=1100, q=0.5)
    .reverb()
    # .phaser()
    # .delay()
    # .lowshelf()
)


# ------------------------------------------------------------------------------------------
# MIDI 2 AUDIO CONVERSION
# ------------------------------------------------------------------------------------------
# FluidSynth(path_soundfont).midi_to_audio('data/prediction.mid','data/prediction.mp3')
for soundfont in sound_path:
    print(soundfont)
    FluidSynth(soundfont).midi_to_audio('data/prediction.mid', 'data/wav_outputs/prediction_' + str(soundfont.name)+ '.wav')
    fx('data/wav_outputs/prediction_' + str(soundfont.name) + '.wav', 'data/wav_outputs_effect/prediction_effect_' + str(soundfont.name)+ '.wav')
# ------------------------------------------------------------------------------------------
# EFFECT DEFINITION AND APPLICATION
# ------------------------------------------------------------------------------------------



'''
infile = 'data/prediction.mp3'
outfile = 'data/prediction_with_effect.mp3'

# Effect application
fx(infile, outfile)
'''
