import numpy as np
import pytest
from env.tasks._single_key_score import SingleKeyScore
from env.tasks._twinkle import TWINKLE, HOLD_SECONDS


def tick(score, angle, force=1., seconds=.1, key=39):
    angles, forces = np.zeros(88), np.zeros(88)
    angles[key], forces[key] = angle, force
    for step in range(round(seconds/.01)):
        score.advance(angles,forces,.01,step)


def test_five_complete_press_release_cycles_are_required():
    score = SingleKeyScore()
    for i in range(5):
        tick(score,.04)
        assert score.count == i and not score.success
        tick(score,0.)
        assert score.count == i+1
    assert score.success


def test_holding_a_key_never_counts_as_repeated_taps():
    score = SingleKeyScore()
    tick(score,.04,seconds=5.)
    assert score.count == 0
    tick(score,0.)
    assert score.count == 1 and not score.success


def test_displacement_without_contact_does_not_score():
    score = SingleKeyScore()
    tick(score,.04,force=0.)
    tick(score,0.)
    assert score.count == 0


def test_wrong_key_and_contact_bounce():
    score = SingleKeyScore()
    tick(score,.04,seconds=.03)
    tick(score,0.)
    assert score.count == 0
    tick(score,.04)
    tick(score,0.,seconds=.04)
    tick(score,.02)
    tick(score,0.,seconds=.06)
    assert score.count == 0
    tick(score,0.,seconds=.04)
    assert score.count == 1
    tick(score,.04,key=40)
    assert score.failure == 'wrong_key'


def test_invalid_measurements_and_configuration():
    with pytest.raises(ValueError):
        SingleKeyScore(repetitions=0)
    with pytest.raises(ValueError):
        SingleKeyScore().advance(np.full(88,np.nan),np.zeros(88),.01,0)


def test_complete_twinkle_pitch_order_and_phrase_durations():
    assert TWINKLE[:14] == (60,60,67,67,69,69,67,65,65,64,64,62,62,60)
    assert len(TWINKLE)==42 and TWINKLE[:14]==TWINKLE[-14:]
    assert [i for i,v in enumerate(HOLD_SECONDS) if v>.12]==[6,13,20,27,34,41]
    score=SingleKeyScore(key_sequence=tuple(m-21 for m in TWINKLE), note_hold_seconds=HOLD_SECONDS)
    for i,midi in enumerate(TWINKLE):
        assert score.active_key==midi-21
        tick(score,.04,seconds=HOLD_SECONDS[i],key=midi-21)
        assert score.count==i
        tick(score,0.,key=midi-21)
        assert score.count==i+1
    assert score.success
    assert [e['midi'] for e in score.events if e['kind']=='press']==list(TWINKLE)


def test_sequence_requires_correct_pitch_and_long_note_hold():
    score=SingleKeyScore(key_sequence=(39,46),note_hold_seconds=(.1,.4))
    tick(score,.04,key=39)
    tick(score,0.,key=39)
    tick(score,.04,key=46,seconds=.2)
    tick(score,0.,key=46)
    assert score.count==1 and not score.success
    tick(score,.04,key=39)
    assert score.failure=='wrong_key'
