"""Publish an index/card derived from actual validated and uploaded episodes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

TASKS=('tension_strap','arm_shadow_piano','play_piano')


def verify_published_file(api,repo_id,directory,name):
    from modelscope_hub.errors import FileIntegrityError
    directory=Path(directory)
    expected=hashlib.sha256((directory/name).read_bytes()).hexdigest()
    delays=(0,2,5,10,20)
    for attempt,delay in enumerate(delays):
        if delay:time.sleep(delay)
        try:
            path=api.download_file(repo_id,'dataset',name,
                local_dir=directory/'.readback',force=True,expected_sha256=expected)
        except FileIntegrityError:
            # A replaced branch-tip file can briefly resolve to the previous
            # CDN version. Keep the exact expected hash and allow propagation.
            if attempt==len(delays)-1:raise
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Catalog readback differs')
        path.unlink()
        return


def build_catalog(root,*,play_piano_batch='dataset'):
    root=Path(root)
    audit_path=root/'recording_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    audits=audit.get('episodes',{})
    index=dict(updated_utc=datetime.now(timezone.utc).isoformat(),targets={},
        recording_audit_source_sha256=audit.get('audit_source_sha256'),tasks={})
    for task in TASKS:
        directory=play_piano_batch if task=='play_piano' else 'dataset'
        progress=root/task/directory/'progress.json'
        batch=json.loads(progress.read_text()) if progress.exists() else {}
        target=batch.get('target',50)
        if isinstance(target,bool) or not isinstance(target,int) or target<1:
            raise ValueError('Dataset target must be a positive integer: '+task)
        index['targets'][task]=target
        rows=[]
        for attempt in batch.get('attempts',[]):
            if not attempt.get('accepted') or not attempt.get('uploaded'):continue
            source=Path(attempt['path'])
            result=json.loads((source/'result.json').read_text())
            if result.get('dataset_accepted') is not True or result.get('validation',{}).get('valid') is not True:
                raise ValueError('Published episode lacks a valid physical/data verdict: '+str(source))
            validation=result['validation']
            recording=audits.get(f"{task}/{attempt['seed']:04d}",{})
            if not recording.get('valid'):continue
            h5=source/'tactile.h5' if task=='arm_shadow_piano' else source/'hdf5'/f"{attempt['seed']}.hdf5"
            info=h5.stat()
            if recording.get('file_signature')!=[info.st_size,info.st_mtime_ns]:
                raise ValueError('Recording changed after its RGB audit: '+str(h5))
            files=attempt['upload']['files']
            hdf5_name=h5.relative_to(source).as_posix()
            rows.append(dict(seed=attempt['seed'],path=attempt['upload']['path'],
                hdf5=hdf5_name,sha256=files[hdf5_name]['sha256'],bytes=files[hdf5_name]['bytes'],
                frames=recording['frames'],decoded_images=recording['decoded_images'],
                instruction=validation.get('requested_sequence',validation.get('physical',{}).get('force_time_sequence'))))
        index['tasks'][task]=dict(published_validated_episodes=len(rows),
            target=target,
            accepted=batch.get('collected',0),uploaded=batch.get('uploaded',0),
            collection_status=batch.get('status','pending'),
            batch_directory=directory,
            source_commit_at_collection=batch.get('source_commit'),
            verified_source_commit=batch.get('verified_source_commit'),
            source_sha256=batch.get('source_sha256',{}),episodes=rows)
    targets=set(index['targets'].values())
    index['target_per_task']=next(iter(targets)) if len(targets)==1 else None
    index['complete']=all(v['collection_status']=='complete' and
        v['published_validated_episodes']==v['accepted']==v['uploaded']==v['target']
        for v in index['tasks'].values())
    card=['# OpenTacBench tactile memory demonstrations','',
        'Native scripted demonstrations with physical success judging and tactile RGB.',
        'Collection is '+('complete.' if index['complete'] else '**in progress**.'),'',
        '| Task | Accepted | Uploaded | Full RGB audit passed | Target |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,task in index['tasks'].items():
        card.append(f"| `{name}` | {task['accepted']} | {task['uploaded']} | {task['published_validated_episodes']} | {task['target']} |")
    card+=['',
        'Only episodes that pass native success, data/clock validation, full RGB decoding and upload-manifest readback enter `dataset_index.json`.',
        'The first accepted HDF5 in each task also passes a complete download and SHA-256 comparison.',
        '',
        '- `tension_strap`: two stages `[F1_N, T1_s, F2_N, T2_s]`, forces 0–24 N; each duration is the minimum continuous hold within the physical ±0.5 N band while retaining the grasp. Original damage limits remain active. Instruction sampling is independent of the hidden material stream.',
        '- `arm_shadow_piano`: Franka + Shadow index finger, original DIGIT/TACTO configuration. Six random C4–B4 notes, required light/heavy force bands, holds and releases. `tactile.h5` contains RGB, state, issued actions, start/stop physics steps and aligned `piano/` labels.',
        '- `play_piano`: Franka + original TacEx GelSight Mini gripper, picked-up tapping tool. Six random C4–B4 notes, required light/heavy force bands, holds and releases. Original HDF5 layout and JPEG tactile images are retained; `piano_steps/` retains every physics tick.',
        '',
        'Each episode directory includes `result.json` and `checksums.json`; legacy tasks also include configuration and metadata. The index gives each HDF5 path, SHA-256, frame count and requested instruction.',
        'Files under `_checks/` are development or upload/readback tests and are excluded from the requested per-task totals.',
        'Recorded ground-truth diagnostics are for auditing. These are scripted demonstrations; they do not establish learned-policy performance.',
        '',
        'Native environment: Isaac Sim 5.1, Isaac Lab 2.3.2, libuipc 0.9.0, Torch 2.7 / CUDA 12.8, one NVIDIA L40S. Each episode starts a fresh native scene. Physics is 120 Hz for strap/tool and 240 Hz for the Shadow task; observations/actions are 60 Hz and 30 Hz respectively.',
        '',
        'Exact source hashes are recorded per task and per episode. `source_commit_at_collection` identifies the checkout when collection began; hashes also cover pending working-tree code. A separately verified commit, when present, identifies the saved implementation with the same runtime contents.',
        '',
        '[Source and task documentation](https://github.com/Apiecehann/OpenTacBench/tree/tactile-memory-tasks)',
        '',f"Updated UTC: {index['updated_utc']}",'']
    if play_piano_batch!='dataset':
        card+=['## Manipulation piano depth/marker recording','',
            'The active `play_piano` batch uses original Mini outputs: `tactile/{left_tactile,right_tactile}/depth` is floating camera distance in mm, and `marker` has shape `(frames,2,64,2)` in image pixels (reference/current, point, x/y). `rgb_marker`, head/wrist RGB, joint/EE state, `phase/sim_step` and requested `piano/target_midi` share the saved-frame clock.',
            '',
            '`piano_steps/<seed>.jsonl` contains privileged labels for every physics tick: all 88 measured `key_forces_N` in ascending MIDI 21–108 order, requested `target_key_index=MIDI-21`, and `measured_target_force_N`. The existing `target_force_N` is the requested force setpoint. Actual-force labels and geometry diagnostics must be excluded from policy inputs.',
            '',
            '`commands/<seed>.jsonl` preserves actual original SDK joint position/velocity targets and their before/after physics steps, including both gripper joints. `sensor_config.json` records original optical/marker/gel parameters and calibration file hashes; `result.json` records the source commit and runtime hashes. `video/<seed>_success.mp4` shows the scene and both tactile views at the native saved-frame rate with contact-triggered synthesized audio.',
            '',
            'The original 9×7 Mini grid retains its original visibility filtering and repeat-last padding to 64 output points; these are the original interface values, not 64 newly fabricated physical markers.',
            '',
            'The preceding RGB-only 30 episodes under `play_piano/episodes/2100` through `2129` remain preserved and are excluded from this replacement batch count.','']
    directory=root/'catalog';directory.mkdir(parents=True,exist_ok=True)
    (directory/'dataset_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    (directory/'README.md').write_text('\n'.join(card),encoding='utf-8')
    return directory,index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/tactile_memory'))
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--play-piano-batch',default='dataset')
    args=parser.parse_args()
    directory,index=build_catalog(args.root,play_piano_batch=args.play_piano_batch)
    if args.publish:
        from modelscope_hub import HubApi
        api=HubApi()
        api.upload_folder(args.repo_id,'dataset',directory,path_in_repo='',
            allow_patterns=['README.md','dataset_index.json'],max_workers=2,disable_tqdm=True,
            commit_message='Update validated dataset counts and index')
        for name in ('README.md','dataset_index.json'):
            verify_published_file(api,args.repo_id,directory,name)
    print(json.dumps(dict(complete=index['complete'],counts={k:v['published_validated_episodes'] for k,v in index['tasks'].items()})))


if __name__=='__main__':main()
