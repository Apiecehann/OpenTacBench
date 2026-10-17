import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from env.tasks._tactile_recording import TactileRecorder
from envs._piano_sequence import PianoAnnotations,make_sequence
from scripts.validate_piano_episode import validate_piano


def fixture_task():
    phrase=make_sequence('C',dynamics='light')
    labels=PianoAnnotations(phrase,embodiment='hand')
    task=SimpleNamespace(step_count=0,sequence=phrase,piano_annotations=labels,
        cfg=SimpleNamespace(physics_dt=1/240,action_repeat=8),
        tactile=SimpleNamespace(metadata=lambda:{'sensor_type':'digit'}))
    task._info=lambda:{'piano':labels.records[-1]}
    return task


def observation():
    rgb=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3)
    return {'state':np.zeros(111,np.float32),'tactile':{'index':{'rgb':rgb}}}


def test_recorder_preserves_observation_action_pairs_and_public_rgb(tmp_path):
    task=fixture_task()
    task.piano_annotations.update(0,1/240,0,np.zeros(88))
    recorder=TactileRecorder(tmp_path/'tactile.h5',task)
    action=np.array([.01,-.1,.02,.3],np.float32)
    recorder.append(observation(),action)
    task.step_count=8
    task.piano_annotations.update(8,1/240,0,np.zeros(88))
    recorder.append(observation(),action*2)
    task.step_count=14  # The final action is clipped by native termination.
    recorder.close()
    with h5py.File(tmp_path/'tactile.h5') as data:
        assert data['physics_step'][:].tolist()==[0,8]
        np.testing.assert_array_equal(data['action'][0],action)
        np.testing.assert_array_equal(data['tactile/index/rgb'][1],observation()['tactile']['index']['rgb'])
        assert data['action'].shape==(2,4)
        assert data['action_stop_physics_step'][:].tolist()==[8,14]
        assert data.attrs['terminal_physics_step']==14
        assert 'observation at t' in data.attrs['action_alignment']


def test_mixed_action_conventions_are_rejected(tmp_path):
    task=fixture_task()
    task.piano_annotations.update(0,1/240,0,np.zeros(88))
    recorder=TactileRecorder(tmp_path/'tactile.h5',task)
    recorder.append(observation(),np.zeros(4))
    with pytest.raises(ValueError,match='same action convention'):
        recorder.append(observation())
    recorder.close()


def test_positive_result_does_not_bypass_insufficient_physical_force_hold(tmp_path):
    task=fixture_task()
    recorder=TactileRecorder(tmp_path/'tactile.h5',task)
    forces=np.zeros(88);forces[39]=.35
    for step in range(25):
        task.step_count=step
        task.piano_annotations.update(step,1/240,0,forces)
        if step%8==0:recorder.append(observation(),np.zeros(4))
    recorder.close()
    task.piano_annotations.write_jsonl(tmp_path/'piano_steps.jsonl')
    (tmp_path/'result.json').write_text(json.dumps(dict(success=True,require_dynamics=True,
        piano_sequence=task.sequence.as_dict(),physics_dt_s=1/240,action_repeat=8,
        events=[dict(kind='press',midi=60,step=20),dict(kind='release',midi=60,step=24)])))
    report=validate_piano(tmp_path,'hand',require_actions=True)
    assert not report['valid']
    assert 'per-tick physical force trace lacks a requested continuous hold' in report['errors']
