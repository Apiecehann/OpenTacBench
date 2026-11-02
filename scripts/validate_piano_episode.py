"""Audit original piano recordings, physical note events and per-tick forces."""
import argparse
import io
import json
from pathlib import Path
import sys
import subprocess

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs._piano_sequence import PianoSequence,dynamics_profile


def validate_rich_tool(source,data,rows,meta,seed):
    """Check the public streams and independent offline evidence requested for Mini."""
    errors=[]
    frames=len(data['step'])
    required=['phase/sim_step','embodiment/joint','embodiment/ee','piano/target_midi']
    for side in ('left_tactile','right_tactile'):
        required.extend(f'tactile/{side}/{kind}' for kind in ('depth','marker','rgb_marker'))
    missing=[key for key in required if key not in data]
    if missing:return ['missing rich piano streams: '+', '.join(missing)]
    if data.attrs.get('piano_recording_schema_version')!=2:
        errors.append('missing rich piano schema version')
    if not np.array_equal(data['phase/sim_step'][:],data['step'][:]):
        errors.append('public phase clock differs from observations')
    config_path=source/'sensor_config.json'
    if not config_path.exists():return errors+['missing original sensor configuration']
    schema=json.loads(config_path.read_text())
    for side in ('left_tactile','right_tactile'):
        depth=data[f'tactile/{side}/depth']
        marker=data[f'tactile/{side}/marker']
        if depth.shape!=(frames,240,320) or depth.dtype.kind!='f' or depth.attrs.get('units')!='mm':
            errors.append('expected original 240x320 floating depth in mm: '+side)
        if marker.shape!=(frames,2,64,2) or marker.dtype.kind!='f' or marker.attrs.get('units')!='pixels':
            errors.append('expected reference/current (2,64,2) pixel markers: '+side)
        else:
            xy=marker[:]
            if (not np.isfinite(xy).all() or np.any(xy<0)
                    or np.any(xy[...,0]>=320) or np.any(xy[...,1]>=240)):
                errors.append('marker coordinates lie outside the original image: '+side)
            if np.max(np.abs(xy[:,1]-xy[:,0]))<1e-6:
                errors.append('marker stream has no measured motion: '+side)
        for start in range(0,frames,32):
            d=depth[start:start+32]
            if not np.isfinite(d).all() or np.any(d<24) or np.any(d>34):
                errors.append('depth is outside original Mini clipping range: '+side)
                break
        cfg=schema.get('sensors',{}).get(side,{})
        if (cfg.get('sensor_type')!='gsmini' or cfg.get('marker_count')!=64
                or not cfg.get('calibration_sha256')):
            errors.append('original Mini calibration provenance missing: '+side)
    ticks=np.asarray([r['physics_step'] for r in rows])
    if len(ticks)<2 or np.any(np.diff(ticks)!=1):
        errors.append('full force trace must cover consecutive physics ticks')
    for row in rows:
        values=np.asarray(row.get('key_forces_N',[]),float)
        if values.shape!=(88,) or not np.isfinite(values).all() or np.any(values<0):
            errors.append('missing 88 finite nonnegative physical key forces')
            break
        key=row['target_midi']-21 if row['target_midi']>=21 else -1
        measured=float(values[key]) if key>=0 else 0.
        if row.get('target_key_index')!=key or not np.isclose(row.get('measured_target_force_N',-1),measured):
            errors.append('target force must be key_forces_N[MIDI-21]')
            break
        maximum=float(values.max())
        actual=int(values.argmax())+21 if maximum>=.1 else -1
        if row['actual_midi']!=actual or not np.isclose(row['actual_force_N'],maximum):
            errors.append('actual note/force differs from the raw 88-key trace')
            break
    if 'piano/key_forces_N' in data:
        errors.append('privileged 88-key forces must stay outside public observations')
    command_path=source/'commands'/f'{seed}.jsonl'
    if not command_path.exists():errors.append('missing actual issued command trace')
    else:
        commands=[json.loads(line) for line in command_path.read_text().splitlines() if line]
        command_steps=[r['physics_step'] for r in commands]
        n=len(schema.get('command_joint_names',[]))
        if (not commands or n!=data['embodiment/joint'].shape[1]
                or np.any(np.diff(command_steps)!=1)
                or commands[0]['before_step']>data['step'][0]
                or command_steps[-1]!=data['step'][-1]):
            errors.append('issued commands do not span the recorded physics interval')
        for row in commands:
            values=np.asarray([row.get('qpos_target',[]),row.get('qvel_target',[])],float)
            if (values.shape!=(2,n) or not np.isfinite(values).all()
                    or row['physics_step']!=row['before_step']+1):
                errors.append('invalid original SDK joint command')
                break
    video=source/'video'/f'{seed}_success.mp4'
    if not video.exists():errors.append('missing scene video')
    else:
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames,avg_frame_rate','-of','json',str(video)],
            capture_output=True,text=True,check=True)
        streams=json.loads(probe.stdout).get('streams',[])
        if (len(streams)!=1 or int(streams[0].get('nb_read_frames',0))!=frames
                or streams[0].get('avg_frame_rate')!='60/1'):
            errors.append('scene video must match saved observations at 60 fps')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],
                       capture_output=True,check=True)
        if meta.get('piano_audio',{}).get('video_frames')!=frames:
            errors.append('audio/video clock differs from saved observations')
    return errors


def validate_piano(source,embodiment,seed=None,*,require_actions=False,require_rich=False):
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
        if require_rich:
            if embodiment!='tool':raise ValueError('rich Mini schema applies to tool piano')
            errors.extend(validate_rich_tool(source,data,rows,meta,seed))
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
    parser.add_argument('--require-rich',action='store_true')
    args=parser.parse_args()
    report=validate_piano(args.source,args.embodiment,args.seed,require_actions=args.require_actions,
                          require_rich=args.require_rich)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['valid'] else 1)
