import mido
import numpy as np


def midi_chord_creator(all_chords_dic):
    """

    Args:
        all_chords_dic: dictionary with all chords

    Returns: dictionary with triplets of pitch values for all chords

    """
    bias = 69  # A 440Hz
    maj = np.array([0, 4, 7])
    minor = np.array([0, 3, 7])
    all_chords_dic['G'] = maj + (bias - 2)
    all_chords_dic['B_min'] = minor + (bias + 2)
    all_chords_dic['E_min'] = minor + (bias - 5)
    all_chords_dic['A_min'] = minor + (bias + 0)
    all_chords_dic['F'] = maj + (bias - 4)
    all_chords_dic['D'] = maj + (bias - 7)
    all_chords_dic['F#'] = maj + (bias - 3)
    all_chords_dic['C#'] = maj + (bias + 4)
    all_chords_dic['E'] = maj + (bias - 5)
    all_chords_dic['B'] = maj + (bias + 2)
    all_chords_dic['A'] = maj + (bias + 0)
    all_chords_dic['F#_min'] = minor + (bias - 3)
    all_chords_dic['C_min'] = minor + (bias + 3)
    all_chords_dic['F_min'] = minor + (bias - 4)
    all_chords_dic['Eb'] = maj + (bias - 6)
    all_chords_dic['G_min'] = minor + (bias - 2)
    all_chords_dic['Bb'] = maj + (bias + 1)
    all_chords_dic['Ab'] = maj + (bias - 1)
    all_chords_dic['D_min'] = minor + (bias - 7)
    all_chords_dic['C#_min'] = minor + (bias + 4)
    all_chords_dic['Db'] = maj + (bias + 4)
    all_chords_dic['Bb_min'] = minor + (bias + 1)
    all_chords_dic['Eb_min'] = minor + (bias - 6)
    all_chords_dic['Gb_min'] = minor + (bias - 3)
    all_chords_dic['Gb'] = maj + (bias - 3)
    all_chords_dic['G#_min'] = minor + (bias - 1)
    all_chords_dic['G#'] = maj + (bias - 1)
    all_chords_dic['D#_min'] = minor + (bias - 6)
    all_chords_dic['C'] = maj + (bias + 3)
    return all_chords_dic


def midi_predictor(mid_pred, mid_pred_trck, chord_pred, all_chords_mid, note_dur):
    """

    Args:
        mid_pred: midi file created with mido
        mid_pred_trck: midi track file created with mido
        chord_pred: prediction of chords from hmm
        all_chords_mid: dictionary with triplets of pitch values for all chords
        note_dur: duration

    Returns: void, saves midi file as data/prediction.mid

    """

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
