import io
import hashlib
import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PIL import Image

from scripts.validate_tactile_recording import audit_recording
from scripts.publish_tactile_dataset import build_catalog,verify_published_file


def make_recording(path,raw=False):
    image=np.arange(8*6*3,dtype=np.uint8).reshape(6,8,3)
    with h5py.File(path,'w') as data:
        data['step']=[0,2]
        data['state']=np.zeros((2,4),np.float32)
        if raw:data['tactile/index/rgb']=np.stack([image,image])
        else:
            out=io.BytesIO();Image.fromarray(image).save(out,format='JPEG')
            encoded=np.frombuffer(out.getvalue(),np.uint8)
            stream=data.create_dataset('tactile/left/rgb_marker',(2,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            stream[0]=encoded;stream[1]=encoded


@pytest.mark.parametrize('raw',[False,True])
def test_decodes_all_recorded_frames(tmp_path,raw):
    path=tmp_path/'episode.h5';make_recording(path,raw)
    report=audit_recording(path)
    assert report['valid'] and report['decoded_images']==report['frames']==2
    assert report['numeric_fields']==1


def test_rejects_corrupt_compressed_image(tmp_path):
    path=tmp_path/'episode.h5';make_recording(path)
    with h5py.File(path,'a') as data:data['tactile/left/rgb_marker'][1]=np.array([0,1,2],np.uint8)
    report=audit_recording(path)
    assert not report['valid'] and report['decoded_images']==1


def test_rejects_nonfinite_state_and_wrong_frame_count(tmp_path):
    path=tmp_path/'episode.h5';make_recording(path)
    with h5py.File(path,'a') as data:
        data['state'][0,0]=np.nan
        data['short']=[0]
    report=audit_recording(path)
    assert not report['valid']
    assert 'nonfinite values: state' in report['errors']
    assert 'inconsistent frame count: short' in report['errors']


@pytest.mark.parametrize('colour',[(0,0,0),(120,30,10)])
def test_rejects_entirely_blank_raw_sensor_stream(tmp_path,colour):
    path=tmp_path/'episode.h5';make_recording(path,raw=True)
    with h5py.File(path,'a') as data:
        image=np.zeros((2,6,8,3),np.uint8);image[:]=colour
        data['tactile/index/rgb'][:]=image
    assert not audit_recording(path)['valid']


def test_catalog_includes_only_uploaded_and_fully_audited_episodes(tmp_path):
    folder=tmp_path/'tension_strap/dataset/episodes/0007'
    (folder/'hdf5').mkdir(parents=True)
    recording=folder/'hdf5/7.hdf5';make_recording(recording)
    (folder/'result.json').write_text(json.dumps(dict(dataset_accepted=True,
        validation=dict(valid=True,physical=dict(force_time_sequence=[0,1,24,2])))))
    attempt=dict(seed=7,path=str(folder),accepted=True,uploaded=True,
        upload=dict(path='tension_strap/episodes/0007',files={
            'hdf5/7.hdf5':dict(bytes=recording.stat().st_size,
                             sha256=hashlib.sha256(recording.read_bytes()).hexdigest())}))
    progress=folder.parents[1]/'progress.json'
    progress.write_text(json.dumps(dict(status='complete',collected=1,uploaded=1,attempts=[attempt])))
    _,pending=build_catalog(tmp_path)
    assert pending['tasks']['tension_strap']['published_validated_episodes']==0
    report=audit_recording(recording)
    report['file_signature']=[recording.stat().st_size,recording.stat().st_mtime_ns]
    (tmp_path/'recording_audit.json').write_text(json.dumps(dict(episodes={'tension_strap/0007':report})))
    _,ready=build_catalog(tmp_path)
    assert ready['tasks']['tension_strap']['published_validated_episodes']==1
    assert not ready['complete']  # A successful one-episode preflight is not 50.
    attempt['uploaded']=False
    progress.write_text(json.dumps(dict(attempts=[attempt])))
    _,pending=build_catalog(tmp_path)
    assert pending['tasks']['tension_strap']['published_validated_episodes']==0
    attempt['uploaded']=True
    progress.write_text(json.dumps(dict(attempts=[attempt])))
    with recording.open('ab') as stream:stream.write(b'changed after audit')
    with pytest.raises(ValueError,match='changed after its RGB audit'):build_catalog(tmp_path)


def test_catalog_readback_waits_for_new_content_and_retains_exact_hash(tmp_path,monkeypatch):
    import scripts.publish_tactile_dataset as publisher
    class IntegrityError(Exception):pass
    monkeypatch.setitem(sys.modules,'modelscope_hub.errors',SimpleNamespace(FileIntegrityError=IntegrityError))
    waits=[];calls=[]
    monkeypatch.setattr(publisher.time,'sleep',waits.append)
    source=tmp_path/'README.md';source.write_bytes(b'current verified counts')
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    class Api:
        def download_file(self,repo,kind,name,**kwargs):
            calls.append(kwargs['expected_sha256'])
            if len(calls)<3:raise IntegrityError('previous cached branch-tip file')
            path=tmp_path/'downloaded.md';path.write_bytes(source.read_bytes());return path
    verify_published_file(Api(),'owner/data',tmp_path,'README.md')
    assert calls==[expected]*3 and waits==[2,5]
    assert source.read_bytes()==b'current verified counts'
    assert not (tmp_path/'downloaded.md').exists()


def test_catalog_completes_with_independent_task_targets(tmp_path):
    targets={'tension_strap':2,'arm_shadow_piano':2,'play_piano':1}
    audits={}
    for task,target in targets.items():
        attempts=[]
        for seed in range(target):
            folder=tmp_path/task/'dataset/episodes'/f'{seed:04d}'
            folder.mkdir(parents=True)
            name='tactile.h5' if task=='arm_shadow_piano' else f'hdf5/{seed}.hdf5'
            recording=folder/name;recording.parent.mkdir(exist_ok=True)
            make_recording(recording)
            (folder/'result.json').write_text(json.dumps(dict(dataset_accepted=True,
                validation=dict(valid=True))))
            report=audit_recording(recording)
            report['file_signature']=[recording.stat().st_size,recording.stat().st_mtime_ns]
            audits[f'{task}/{seed:04d}']=report
            attempts.append(dict(seed=seed,path=str(folder),accepted=True,uploaded=True,
                upload=dict(path=f'{task}/episodes/{seed:04d}',files={name:dict(
                    bytes=recording.stat().st_size,
                    sha256=hashlib.sha256(recording.read_bytes()).hexdigest())})))
        (tmp_path/task/'dataset/progress.json').write_text(json.dumps(dict(
            status='complete',target=target,collected=target,uploaded=target,attempts=attempts)))
    (tmp_path/'recording_audit.json').write_text(json.dumps(dict(episodes=audits)))
    directory,index=build_catalog(tmp_path)
    assert index['complete'] and index['targets']==targets
    assert index['target_per_task'] is None
    assert '| `play_piano` | 1 | 1 | 1 | 1 |' in (directory/'README.md').read_text(encoding='utf-8')
    progress=tmp_path/'play_piano/dataset/progress.json'
    batch=json.loads(progress.read_text());batch['target']=2
    progress.write_text(json.dumps(batch))
    assert not build_catalog(tmp_path)[1]['complete']
    batch.update(target=1,status='running');progress.write_text(json.dumps(batch))
    assert not build_catalog(tmp_path)[1]['complete']


def test_audit_queue_reuses_unchanged_recordings_and_checks_requested_count(tmp_path,monkeypatch):
    import scripts.audit_tactile_memory as auditor
    folder=tmp_path/'tension_strap/dataset/episodes/0000'
    (folder/'hdf5').mkdir(parents=True)
    make_recording(folder/'hdf5/0.hdf5')
    (folder.parents[1]/'progress.json').write_text(json.dumps(dict(status='complete',
        collected=1,uploaded=1,attempts=[dict(seed=0,path=str(folder),accepted=True)])))
    state={'episodes':{}}
    assert not auditor.scan(tmp_path,['tension_strap'],state,50,lambda:None)
    assert state['counts']['tension_strap']==1
    assert state['decoded_images']==2
    monkeypatch.setattr(auditor,'audit_recording',lambda path:pytest.fail('unchanged file was re-audited'))
    assert auditor.scan(tmp_path,['tension_strap'],state,1,lambda:None)


def test_replacement_batch_catalog_and_audit_exclude_preserved_rgb_batch(tmp_path):
    import scripts.audit_tactile_memory as auditor
    state={'episodes':{'play_piano/2100':dict(valid=True,decoded_images=4)}}
    for directory,seed in [('dataset',2100),('dataset-rich',2200)]:
        folder=tmp_path/'play_piano'/directory/'episodes'/str(seed)
        (folder/'hdf5').mkdir(parents=True)
        recording=folder/'hdf5'/f'{seed}.hdf5';make_recording(recording)
        (folder/'result.json').write_text(json.dumps(dict(dataset_accepted=True,validation=dict(valid=True))))
        attempt=dict(seed=seed,path=str(folder),accepted=True,uploaded=True,upload=dict(
            path=f'play_piano/{directory}/episodes/{seed}',files={f'hdf5/{seed}.hdf5':dict(
                bytes=recording.stat().st_size,sha256=hashlib.sha256(recording.read_bytes()).hexdigest())}))
        (folder.parents[1]/'progress.json').write_text(json.dumps(dict(status='complete',target=1,
            collected=1,uploaded=1,attempts=[attempt])))
    assert auditor.scan(tmp_path,['play_piano'],state,1,lambda:None,play_piano_batch='dataset-rich')
    assert state['counts']=={'play_piano':1} and state['episodes']['play_piano/2100']['valid']
    (tmp_path/'recording_audit.json').write_text(json.dumps(state))
    _,catalog=build_catalog(tmp_path,play_piano_batch='dataset-rich')
    task=catalog['tasks']['play_piano']
    assert task['published_validated_episodes']==1 and task['episodes'][0]['seed']==2200
    assert task['batch_directory']=='dataset-rich'
