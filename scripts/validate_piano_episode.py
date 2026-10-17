"""Audit original piano recordings, physical note events and per-tick forces."""
import argparse
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs._piano_sequence import PianoSequence,dynamics_profile


def validate_piano(source,embodiment,seed=None,*,require_actions=False):
    source=Path(source)
    errors=[]
    if embodiment=='hand':
        result=json.loads((source/'result.json').read_text())
        sequence=result['piano_sequence']
        events=result['events']
        dt=result['physics_dt_s']
        repeat=result['action_repeat']
        episode=source/'tactile.h5'
        if not result.get('success') or result.get('failure'):
            errors.append('native physical scorer did not succeed')
        if result.get('require_dynamics') is not True:
            errors.append('force judging is required for this dataset')
        pressed=[r['midi'] for r in events if r['kind']=='press']
        released=[r['midi'] for r in events if r['kind']=='release']
        if pressed!=sequence['midi'] or released!=sequence['midi']:
            errors.append('physical press/release events differ from the requested phrase')
        event_steps=[r['step'] for r in events]
        if any(b<=a for a,b in zip(event_steps,event_steps[1:])):
            errors.append('physical events are not strictly ordered')
        annotation=source/'piano_steps.jsonl'
    elif embodiment=='tool':
        meta=json.loads((source/'metadata.json').read_text())[str(seed)]
        result=meta['piano_result']
        sequence=result['sequence']
        events=result['notes']
        dt=1/120
        repeat=2
        episode=source/'hdf5'/f'{seed}.hdf5'
        if meta.get('result')!='success' or not result.get('success') or result.get('failure'):
            errors.append('native physical scorer did not succeed')
        if result.get('require_dynamics') is not True:
            errors.append('force judging is required for this dataset')
        if [r['key']+21 for r in events]!=sequence['midi']:
            errors.append('physical key events differ from the requested phrase')
        if len(events)!=len(sequence['hold_seconds']) or any(
                r['stable_seconds']+1e-9<hold for r,hold in zip(events,sequence['hold_seconds'])):
            errors.append('physical key events did not retain the requested holds')
        annotation=source/meta['piano_annotation_path']
    else:
        raise ValueError('embodiment must be hand or tool')
    phrase=PianoSequence(tuple(sequence['midi']),tuple(sequence['dynamics']),tuple(sequence['hold_seconds']))
    targets,_,tolerances=dynamics_profile(phrase,embodiment=embodiment)
    rows=[json.loads(line) for line in annotation.read_text().splitlines() if line]
    if not rows:
        errors.append('per-physics-step annotations are missing')
    longest=np.zeros(len(phrase.midi))
    run=0
    previous_step=None
    previous_note=None
    row_by_step={}
    for row in rows:
        step=row['physics_step']
        row_by_step[step]=row
        index=row['target_index']
        if previous_step is not None and step<=previous_step:
            errors.append('annotation physics steps must increase strictly')
            break
        if not np.isclose(row['time_s'],step*dt,atol=1e-9,rtol=0):
            errors.append('annotation clock differs from physics time')
            break
        if 0<=index<len(phrase.midi):
            if row['target_midi']!=phrase.midi[index] or not np.isclose(row['target_force_N'],targets[index]):
                errors.append('annotation target differs from the requested phrase')
                break
            in_band=(row['actual_midi']==phrase.midi[index]
                and abs(row['actual_force_N']-targets[index])<=tolerances[index]+1e-9)
            run=(run+1 if previous_step is not None and step==previous_step+1
                 and index==previous_note else 1) if in_band else 0
            longest[index]=max(longest[index],run*dt)
        else:
            run=0
        previous_step,previous_note=step,index
    if np.any(longest+dt+1e-9<phrase.holds):
        errors.append('per-tick physical force trace lacks a requested continuous hold')
    decoded=0
    with h5py.File(episode,'r') as data:
        steps=data['physics_step' if embodiment=='hand' else 'step'][:]
        frames=len(steps)
        if frames<2 or np.any(np.diff(steps)<=0):
            errors.append('HDF5 clock must increase strictly')
        if embodiment=='hand':
            intervals=np.diff(steps)
            uniform=np.all(intervals==repeat)
            terminal_partial=(not require_actions and len(intervals)>0 and
                              np.all(intervals[:-1]==repeat) and 0<intervals[-1]<=repeat)
            if not (uniform or terminal_partial):errors.append('hand observations are not uniformly sampled')
            if require_actions and ('action' not in data or not data.attrs.get('action_alignment')):
                errors.append('hand data lacks observation/action alignment')
            if require_actions:
                if 'action_stop_physics_step' not in data:
                    errors.append('hand actions lack their actual stop timestamps')
                else:
                    stops=data['action_stop_physics_step'][:]
                    terminal=data.attrs.get('terminal_physics_step',-1)
                    if (np.any(stops<=steps) or np.any(stops-steps>repeat)
                            or stops[-1]!=terminal or not np.all(stops[:-1]==steps[1:])):
                        errors.append('hand action intervals do not match physical control/terminal timestamps')
            image_keys=['tactile/index/rgb']
        else:
            if str(data['phase'].attrs.get('terminal_reason'))!='success':
                errors.append('HDF5 does not end with native physical success')
            phases=data['phase/id'][:]
            policy=steps[phases==1]
            if len(policy)<2 or not np.all(np.diff(policy)==repeat):
                errors.append('tool policy observations are not uniformly sampled')
            image_keys=['tactile/left_tactile/rgb_marker','tactile/right_tactile/rgb_marker']
        def check_array(name,value):
            if not isinstance(value,h5py.Dataset):return
            if value.shape[0]!=frames:errors.append('frame count differs: '+name)
            if value.dtype.kind=='f':
                for offset in range(0,frames,64):
                    if not np.isfinite(value[offset:offset+64]).all():
                        errors.append('nonfinite values: '+name)
                        break
        data.visititems(check_array)
        for key in image_keys:
            images=data[key]
            for i in range(frames):
                if embodiment=='tool':
                    with Image.open(io.BytesIO(images[i].tobytes())) as img:
                        if img.mode!='RGB':errors.append('tactile image is not RGB')
                        img.verify()
                elif images.shape[-1]!=3 or images.dtype!=np.uint8:
                    errors.append('tactile image must be uint8 RGB')
                    break
                decoded+=1
        for i,step in enumerate(steps):
            row=row_by_step.get(int(step))
            if row is None:
                errors.append('HDF5 timestamp has no physical annotation')
                break
            for key in ('time_s','target_index','target_midi','actual_midi','target_dynamic','actual_dynamic','actual_force_N'):
                if not np.isclose(data['piano/'+key][i],row[key],atol=1e-8,rtol=0):
                    errors.append('HDF5 label is not aligned: '+key)
                    break
    return dict(valid=not errors,embodiment=embodiment,episode=str(episode),frames=frames,
        requested_sequence=sequence,force_hold_seconds=longest.tolist(),decoded_images=decoded,errors=errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--embodiment',choices=['hand','tool'],required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--require-actions',action='store_true')
    args=parser.parse_args()
    report=validate_piano(args.source,args.embodiment,args.seed,require_actions=args.require_actions)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['valid'] else 1)
