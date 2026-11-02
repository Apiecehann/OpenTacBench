import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from env.tasks._tactile_recording import TactileRecorder
from envs._piano_sequence import PianoAnnotations,make_sequence
from scripts.validate_piano_episode import validate_piano,validate_rich_tool


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


def test_full_key_labels_preserve_target_and_wrong_key_without_exposing_vector(tmp_path):
    labels=PianoAnnotations(make_sequence('A0 C8',dynamics='light'),embodiment='tool',record_key_forces=True)
    forces=np.zeros(88);forces[0]=1.4;forces[39]=7.;forces[-1]=2.9
    public=labels.update(1,1/120,0,forces)
    forces[:]=0  # The offline record must own a copy of the measured vector.
    labels.update(2,1/120,1,np.arange(88,dtype=float))
    labels.update(3,1/120,2,forces)
    assert 'key_forces_N' not in public and public['actual_midi']==60
    labels.write_jsonl(tmp_path/'steps.jsonl')
    rows=[json.loads(line) for line in (tmp_path/'steps.jsonl').read_text().splitlines()]
    assert rows[0]['target_key_index']==0 and rows[0]['measured_target_force_N']==1.4
    assert rows[0]['key_forces_N'][39]==7.  # Wrong-key overload remains visible.
    assert rows[1]['target_key_index']==87 and rows[1]['measured_target_force_N']==87.
    assert rows[2]['target_key_index']==-1 and rows[2]['measured_target_force_N']==0.


def rich_fixture(tmp_path,monkeypatch):
    import scripts.validate_piano_episode as validator
    monkeypatch.setattr(validator.subprocess,'run',lambda argv,**kw:SimpleNamespace(
        stdout=json.dumps({'streams':[{'nb_read_frames':'3','avg_frame_rate':'60/1'}]})))
    (tmp_path/'video').mkdir();(tmp_path/'video/42_success.mp4').touch()
    (tmp_path/'commands').mkdir()
    (tmp_path/'commands/42.jsonl').write_text('\n'.join(json.dumps(dict(before_step=i,
        physics_step=i+1,qpos_target=[0.]*9,qvel_target=[0.]*9)) for i in range(3)))
    (tmp_path/'sensor_config.json').write_text(json.dumps(dict(command_joint_names=[f'joint{i}' for i in range(9)],
        sensors={side:dict(sensor_type='gsmini',marker_count=64,calibration_sha256={'calib':'test'})
                 for side in ('left_tactile','right_tactile')})))
    labels=PianoAnnotations(make_sequence('C',dynamics='light'),embodiment='tool',record_key_forces=True)
    force=np.zeros(88);force[39]=1.5
    for step in range(1,4):labels.update(step,1/120,0,force)
    labels.write_jsonl(tmp_path/'labels.jsonl')
    rows=[json.loads(line) for line in (tmp_path/'labels.jsonl').read_text().splitlines()]
    with h5py.File(tmp_path/'data.h5','w') as data:
        data['step']=[1,2,3];data['phase/sim_step']=[1,2,3]
        data['embodiment/joint']=np.zeros((3,9));data['embodiment/ee']=np.zeros((3,7))
        data['piano/target_midi']=[60,60,60];data.attrs['piano_recording_schema_version']=2
        for side in ('left_tactile','right_tactile'):
            depth=data.create_dataset(f'tactile/{side}/depth',data=np.full((3,240,320),28.,np.float32))
            depth.attrs['units']='mm'
            xy=np.full((3,2,64,2),100.,np.float32);xy[:,1,:,0]+=np.arange(3)[:,None]
            marker=data.create_dataset(f'tactile/{side}/marker',data=xy);marker.attrs['units']='pixels'
            data[f'tactile/{side}/rgb_marker']=[b'image']*3
    return rows,{'piano_audio':{'video_frames':3}}


def test_rich_schema_checks_64_points_and_raw_target_index(tmp_path,monkeypatch):
    rows,meta=rich_fixture(tmp_path,monkeypatch)
    with h5py.File(tmp_path/'data.h5','a') as data:
        assert validate_rich_tool(tmp_path,data,rows,meta,42)==[]
        del data['tactile/left_tactile/marker']
        field=data.create_dataset('tactile/left_tactile/marker',data=np.ones((3,2,63,2)))
        field.attrs['units']='pixels'
        assert any('(2,64,2)' in e for e in validate_rich_tool(tmp_path,data,rows,meta,42))
    rows[0]['measured_target_force_N']=0.
    with h5py.File(tmp_path/'data.h5') as data:
        assert 'target force must be key_forces_N[MIDI-21]' in validate_rich_tool(tmp_path,data,rows,meta,42)


def test_rich_schema_rejects_missing_depth_or_skipped_commands(tmp_path,monkeypatch):
    rows,meta=rich_fixture(tmp_path,monkeypatch)
    with h5py.File(tmp_path/'data.h5','a') as data:
        del data['tactile/right_tactile/depth']
        assert any('right_tactile/depth' in e for e in validate_rich_tool(tmp_path,data,rows,meta,42))
        depth=data.create_dataset('tactile/right_tactile/depth',data=np.full((3,240,320),28.))
        depth.attrs['units']='mm'
        p=tmp_path/'commands/42.jsonl';commands=p.read_text().splitlines();p.write_text('\n'.join(commands[::2]))
        assert 'issued commands do not span the recorded physics interval' in validate_rich_tool(tmp_path,data,rows,meta,42)
