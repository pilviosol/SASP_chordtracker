import mido
import numpy as np


def midi_chord_creator(all_chords_mid):
    bias = 69 # A 440
    maj = np.array([0,4,7])
    minor = np.array([0,3,7])
    all_chords_mid['G'] = maj + (bias - 2)
    all_chords_mid['B_min'] = minor + (bias + 2)
    all_chords_mid['E_min'] = minor + (bias - 5)
    all_chords_mid['A_min'] = minor + (bias + 0)
    all_chords_mid['F'] = maj + (bias - 4)
    all_chords_mid['D'] = maj + (bias - 7)
    all_chords_mid['F#'] = maj + (bias - 3)
    all_chords_mid['C#'] = maj + (bias + 4)
    all_chords_mid['E'] = maj + (bias - 5)
    all_chords_mid['B'] = maj + (bias + 2)
    all_chords_mid['A'] = maj + (bias + 0)
    all_chords_mid['F#_min'] = minor + (bias - 3)
    all_chords_mid['C_min'] = minor + (bias + 3)
    all_chords_mid['F_min'] = minor + (bias - 4)
    all_chords_mid['Eb'] = maj + (bias - 6)
    all_chords_mid['G_min'] = minor + (bias - 2)
    all_chords_mid['Bb'] = maj + (bias + 1)
    all_chords_mid['Ab'] = maj + (bias - 1)
    all_chords_mid['D_min'] = minor + (bias - 7)
    all_chords_mid['C#_min'] = minor + (bias + 4)
    all_chords_mid['Db'] = maj + (bias + 4)
    all_chords_mid['Bb_min'] = minor + (bias + 1)
    all_chords_mid['Eb_min'] = minor + (bias - 6)
    all_chords_mid['Gb_min'] = minor + (bias - 3)
    all_chords_mid['Gb'] = maj + (bias - 3)
    all_chords_mid['G#_min'] = minor + (bias - 1)
    all_chords_mid['G#'] = maj + (bias - 1)
    all_chords_mid['D#_min'] = minor + (bias - 6)
    all_chords_mid['C'] = maj + (bias + 3)
    return all_chords_mid


def midi_predictor(mid_pred, mid_pred_trck, chord_pred, all_chords_mid, note_dur):

    # I need three tracks for the three voices of the triad
    mid_pred.tracks.append(mid_pred_trck)

    for elem in chord_pred:
        print(elem)
        mid_pred_trck.append(mido.Message('note_on', note=all_chords_mid[elem][0], velocity=64, time=0))
        mid_pred_trck.append(mido.Message('note_on', note=all_chords_mid[elem][1], velocity=64, time=0))
        mid_pred_trck.append(mido.Message('note_on', note=all_chords_mid[elem][2], velocity=64, time=0))
        mid_pred_trck.append(mido.Message('note_off', note=all_chords_mid[elem][0], velocity=64, time=note_dur))
        mid_pred_trck.append(mido.Message('note_off', note=all_chords_mid[elem][1], velocity=64, time=0))
        mid_pred_trck.append(mido.Message('note_off', note=all_chords_mid[elem][2], velocity=64, time=0))

    mid_pred.save('data/prediction.mid')
