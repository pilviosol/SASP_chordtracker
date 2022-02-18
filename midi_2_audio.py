from midi2audio import FluidSynth
from pysndfx import AudioEffectsChain

# ------------------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------------------
path_soundfont = "/Users/PilvioSol/Downloads/70s_sci-fi/70's Sci-fi.sf2"


# ------------------------------------------------------------------------------------------
# MIDI 2 AUDIO CONVERSION
# ------------------------------------------------------------------------------------------
FluidSynth(path_soundfont).midi_to_audio('data/test.mid','data/test.mp3')


# ------------------------------------------------------------------------------------------
# EFFECT DEFINITION AND APPLICATION
# ------------------------------------------------------------------------------------------
# Effect definition
fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .phaser()
    .delay()
    .lowshelf()
)


infile = 'data/test.mp3'
outfile = 'data/test_effect.mp3'

# Effect application
fx(infile, outfile)
