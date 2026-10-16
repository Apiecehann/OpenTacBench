import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from scripts.collect_tactile_memory import strap_instruction
from scripts.upload_tactile_memory import episode_files,sha256,upload_episode
from scripts.calibrate_strap_rgb import fit_extension
import scripts.collect_tactile_memory as collector


def test_force_sampling_covers_full_range_and_both_orders_reproducibly():
    samples=[strap_instruction(seed,24) for seed in range(50)]
    assert samples[0][::2]==[0,24] and samples[1][::2]==[24,0]
    assert all(0<=v<=24 for s in samples for v in s[::2])
    assert all(v>0 for s in samples for v in s[1::2])
    assert len({tuple(s) for s in samples})==50
    assert strap_instruction(12,24)==samples[12]
    assert strap_instruction(0,20)[2]==20
    interior=[i for i in range(50) if i%10 not in (0,1)]
    hidden=[np.random.default_rng(i).uniform(0,24) for i in interior]
    assert not np.allclose([samples[i][0] for i in interior],hidden)


def test_force_regression_export_restores_original_feature_units_and_prior():
    values=np.linspace(-3,5,100)
    features=np.stack([values*100,values**2/1000],axis=1)
    force=3*values+2
    prior={'weights':[0.,0.],'bias':0.,'physical_contract':{'sensor':'original'}}
    weights,bias=fit_extension(prior,features,force,1e-8)
    np.testing.assert_allclose(features@weights+bias,force,atol=1e-6)
    assert prior=={'weights':[0.,0.],'bias':0.,'physical_contract':{'sensor':'original'}}


@pytest.mark.parametrize('result',[
    {'result':'fail','validation':{'valid':True}},
    {'result':'success','validation':{'valid':False}},
    {'success':True}, {'success':True,'validation':{'valid':'yes'}},
])
def test_incomplete_or_failed_episodes_cannot_be_uploaded(tmp_path,result):
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):episode_files(tmp_path)


def test_success_claim_without_recording_cannot_be_uploaded(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps({'success':True,'validation':{'valid':True}}))
    with pytest.raises(ValueError,match='no HDF5'):episode_files(tmp_path)


def test_upload_checks_remote_manifest_and_independent_roundtrip(tmp_path,monkeypatch):
    source=tmp_path/'episode'
    source.mkdir()
    (source/'result.json').write_text(json.dumps({'success':True,'validation':{'valid':True}}))
    (source/'tactile.h5').write_bytes(b'recorded hdf5 payload')
    remote={}
    class Api:
        def upload_folder(self,repo_id,repo_type,folder_path,*,path_in_repo,allow_patterns,**kwargs):
            for pattern in allow_patterns:
                for path in Path(folder_path).glob(pattern):
                    if path.is_file():remote[path_in_repo+'/'+path.relative_to(folder_path).as_posix()]=path.read_bytes()
        def download_file(self,repo_id,repo_type,file_path,*,local_dir,expected_sha256=None,**kwargs):
            path=Path(local_dir)/Path(file_path).name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(remote[file_path])
            if expected_sha256 is not None:assert sha256(path)==expected_sha256
            return path
    monkeypatch.setitem(sys.modules,'modelscope_hub',SimpleNamespace(HubApi=Api))
    report=upload_episode(source,'owner/data','task/episodes/0000',roundtrip=True)
    assert report['roundtrip'] and len(report['files'])==2
    assert not list((source/'.upload_readback').iterdir())
    assert report['files']['tactile.h5']['sha256']==sha256(source/'tactile.h5')


@pytest.mark.skipif(sys.platform=='win32',reason='Remote batch file locking uses POSIX flock')
def test_collection_counts_only_the_requested_valid_episodes(tmp_path,monkeypatch):
    calls=[];uploads=[]
    def native(args,seed,path):
        calls.append(seed)
        return {'dataset_accepted':True}
    def upload(args,path,seed,roundtrip):
        uploads.append((seed,roundtrip))
        return {'verified':True}
    monkeypatch.setattr(collector,'fingerprint',lambda task:{})
    monkeypatch.setattr(collector.subprocess,'check_output',lambda *a,**k:'test-commit\n')
    monkeypatch.setattr(collector,'collect_one',native)
    monkeypatch.setattr(collector,'upload_one',upload)
    args=SimpleNamespace(output=tmp_path,task='tension_strap',repo_id='owner/data',
                         count=2,max_attempts=4,start_seed=100)
    collector.run(args)
    report=json.loads((tmp_path/'progress.json').read_text())
    assert report['status']=='complete' and report['collected']==report['uploaded']==2
    assert sorted(calls)==[100,101] and uploads==[(100,True),(101,False)]
