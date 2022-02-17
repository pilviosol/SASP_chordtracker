import numpy as np
import soundfile as sf
from scipy.io import wavfile
import sounddevice as sd


sample_rate = 44100
frequency = 440
start_phase = 0.0
duration = 1
amplitude = 0.8

sample_count = int(np.floor(sample_rate * duration))

# cyclical frequency in sample^-1
omega = frequency * 2*np.pi / sample_rate

# all phases for which we want to sample our sine
phases = np.linspace(start_phase, start_phase + omega * sample_count,
                     sample_count, endpoint=False)

# our sine wave samples, generated all at once
audio = amplitude * np.sin(phases)

# now write to file
fmt, sub = 'WAV', 'PCM_16'
assert sf.check_format(fmt, sub) # to make sure we ask the correct thing beforehand
sf.write('data/wavatable.wav', audio, sample_rate, format=fmt, subtype=sub)

_, wave_table = wavfile.read('data/wavatable.wav')

# indices for the wave table values; this is just for `np.interp` to work
wave_table_period = float(len(wave_table))
wave_table_indices = np.linspace(0, wave_table_period,
                                 len(wave_table), endpoint=False)

# frequency of the wave table played at native resolution
wave_table_freq = sample_rate / wave_table_period

# start index into the wave table
start_index = start_phase * wave_table_period / 2 * np.pi

# code above you run just once at initialization of this wave table ↑
# code below is run for each audio chunk ↓

# samples of wave table per output sample
shift = frequency / wave_table_freq

# fractional indices into the wave table
indices = np.linspace(start_index, start_index + shift * sample_count,
                      sample_count, endpoint=False)

# linearly interpolated wave table sampled at our frequency
audio = np.interp(indices, wave_table_indices, wave_table,
                  period=wave_table_period)
audio *= amplitude

# at last, update `start_index` for the next chunk
start_index += shift * sample_count

# play
sd.play(audio, sample_rate)
