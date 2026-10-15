import numpy as np
import pytest
from env.tasks._single_key_score import SingleKeyScore


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
