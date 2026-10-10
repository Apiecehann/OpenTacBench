import numpy as np
import pytest

from envs._tactile_memory import PianoScore, box_mesh
from envs._piano import keyboard_layout, MELODY


def test_full_piano_layout_has_standard_endpoints_and_black_key_groups():
    keys = keyboard_layout()
    assert len(keys) == 88
    assert [key.midi for key in keys] == list(range(21,109))
    assert sum(not key.black for key in keys) == 52
    assert sum(key.black for key in keys) == 36
    octave = [key for key in keys if 60 <= key.midi < 72]
    assert [key.midi%12 for key in octave if key.black] == [1,3,6,8,10]
    assert np.diff([key.position[1] for key in keys]).min() > 0
    for key in keys:
        if key.black:
            assert key.strike_position[2] == pytest.approx(.085)


def test_five_note_phrase_scores_repeated_keys_and_rejects_distant_keys():
    sequence = tuple(m-21 for m in MELODY)
    score = PianoScore(key_sequence=sequence,key_count=88)
    for key,note in zip(sequence,score.notes):
        forces = np.zeros(88)
        forces[key] = note.force_N
        for _ in range(round(note.hold_s/score.dt)):
            score.advance(forces)
        score.advance(np.zeros(88))
    assert score.success
    assert [event['key']+21 for event in score.events] == list(MELODY)
    bad = PianoScore(key_sequence=sequence,key_count=88)
    forces = np.zeros(88)
    forces[0] = 1.5
    bad.advance(forces)
    assert bad.failure == 'wrong_key'


def test_box_tet_volume_and_storage():
    points,tets = box_mesh((.1,.2,.3))
    assert points.flags.c_contiguous
    volume = np.linalg.det(np.stack([points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1)).sum()/6
    assert volume == pytest.approx(.006)


def test_piano_phrase_requires_force_duration_and_release():
    score = PianoScore()
    for i,note in enumerate(score.notes):
        forces = [0.,0.]
        forces[i] = note.force_N
        for _ in range(round(note.hold_s/score.dt)):
            score.advance(forces)
        assert score.index == i
        score.advance([0.,0.])
    assert score.success
    assert len(score.events) == 2


@pytest.mark.parametrize("forces,reason", [
    ([0.,1.5],"wrong_key"),
    ([7.,0.],"key_overload"),
    ([np.nan,0.],"invalid_key_force"),
])
def test_piano_rejects_invalid_contact(forces,reason):
    score = PianoScore()
    score.advance(forces)
    assert score.failure == reason
    assert not score.success


def test_long_force_cannot_count_as_short_note():
    score = PianoScore()
    for _ in range(12):
        score.advance([3.,0.])
    assert score.failure == "wrong_note_force"


def test_short_release_and_excessive_hold_fail():
    score = PianoScore()
    score.advance([1.5,0.])
    score.advance([0.,0.])
    assert score.failure == "note_too_short"
    score = PianoScore()
    for _ in range(60):
        score.advance([1.5,0.])
    assert score.failure == "note_too_long"


def test_interrupted_pressure_does_not_accumulate_into_a_note():
    score = PianoScore()
    for _ in range(4):
        for _ in range(10):
            score.advance([1.5,0.])
        score.advance([.7,0.])
    score.advance([0.,0.])
    assert score.failure == "note_too_short"


def test_contact_settling_is_not_release_or_valid_note_hold():
    score = PianoScore()
    for force in (.38,.19,.4,.18,.7):
        score.advance([force,0.])
    assert not score.failure and score.pressed
    assert score.stable_steps == 0 and not score.events
    # Actual release before the required force-duration still fails.
    score.advance([0.,0.])
    assert score.failure == 'note_too_short'

    score = PianoScore()
    for _ in range(24):
        score.advance([1.5,0.])
    score.advance([.19,0.])
    assert score.index == 0
    score.advance([0.,0.])
    assert score.index == 1 and not score.failure
