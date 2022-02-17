from midi2audio import FluidSynth
from sf2_loader import *

#FluidSynth().play_midi('data/test.mid')
FluidSynth("/Users/PilvioSol/Downloads/70s_sci-fi/70's Sci-fi.sf2").midi_to_audio('data/test.mid','data/test.mp3')
