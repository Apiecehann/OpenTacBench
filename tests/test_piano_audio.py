import numpy as np
import pytest

from envs._piano_audio import PianoContacts, synthesize


def test_audio_tracks_actual_wrong_keys_repeats_and_release_hysteresis():
    contacts = PianoContacts()
    forces = np.zeros(88)
    for step,force in enumerate((.4,.19,1.5,.09,.5,3.,0.)):
        forces[0] = force  # A0 is not part of the intended phrase, but must sound.
        contacts.advance(step, forces)
    assert [(e['midi'],e['start_step'],e['end_step']) for e in contacts.events] == [(21,0,3),(21,4,6)]
    assert [e['peak_force_N'] for e in contacts.events] == [1.5,3.]
    assert not contacts.active


def test_audio_clock_pitch_silence_and_force_volume():
    steps = np.arange(100,701,2)
    events = [dict(midi=69,start_step=220,end_step=280,peak_force_N=1.5),
              dict(midi=69,start_step=460,end_step=520,peak_force_N=3.)]
    samples,mapped = synthesize(events,steps)
    assert len(samples) == round(len(steps)/60*48000)
    assert [e['onset_seconds'] for e in mapped] == [1.,3.]
    assert [e['release_seconds'] for e in mapped] == [1.5,3.5]
    assert np.all(samples[:48000] == 0)
    assert np.max(np.abs(samples[3*48000:])) > 1.3*np.max(np.abs(samples[48000:2*48000]))
    signal = samples[48000:72000]
    peak = np.fft.rfftfreq(len(signal),1/48000)[np.abs(np.fft.rfft(signal)).argmax()]
    assert peak == pytest.approx(440.,abs=2.)
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) < 1.


def test_end_while_held_and_duplicate_boundary_frame():
    contacts = PianoContacts()
    f = np.zeros(88); f[39] = 2.
    contacts.advance(12,f)
    contacts.finish(20)
    contacts.finish(20)
    assert len(contacts.events) == 1 and not contacts.active
    samples,mapped = synthesize(contacts.events,[10,12,14,16,18,20,20])
    assert mapped[0]['onset_seconds'] == pytest.approx(1/60)
    assert mapped[0]['release_seconds'] == pytest.approx(5/60)
    assert np.any(samples)


def test_empty_keyboard_is_silent():
    samples,mapped = synthesize([],range(0,120,2))
    assert not np.any(samples) and not mapped
