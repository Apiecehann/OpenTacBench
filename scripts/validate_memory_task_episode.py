"""Validate recorded tactile-memory tasks, including their physical verdicts."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from .validate_insert_usb_schema import datasets, validate_phase_contract
except ImportError:
    from validate_insert_usb_schema import datasets, validate_phase_contract


def validate_episode(path, expect=None):
    path = Path(path)
    seed = path.stem
    meta = json.loads((path.parent.parent/'metadata.json').read_text())[seed]
    errors = []
    images = 0
    with h5py.File(path) as data:
        context = json.loads(data.attrs['episode_context_json'])
        task = context['task']
        if task not in ('blind_pick', 'play_piano'):
            raise ValueError('Expected blind_pick or play_piano')
        errors.extend(validate_phase_contract(data, require_terminal=True, metadata=meta))
        timing = context.get('timing_contract', {})
        for name, wanted in dict(physics_dt_s=1/120, policy_action_repeat=2,
                save_frequency=2, policy_observation_dt_s=1/60, video_fps=60).items():
            if not np.isclose(timing.get(name, -1), wanted):
                errors.append('timing mismatch: '+name)
        if 'seed' in context or 'cue_kind' in context or len(data.get('actor', {})):
            errors.append('latent seed, cue, or actor poses exposed to policy')
        cameras = set(data['observation'])
        expected = {'head'} if task == 'blind_pick' else {'head', 'wrist'}
        if cameras != expected:
            errors.append('unexpected camera set: '+str(sorted(cameras)))
        if set(data['tactile']) != {'left_tactile', 'right_tactile'}:
            errors.append('both tactile sensors are required')
        if np.any(data['atom/id'][:] != 0) or any(tag != b'manipulate' for tag in data['atom/tag']):
            errors.append('privileged action tags exposed to policy')
        count = len(data['step'])
        for name, field in datasets(data).items():
            if len(field) != count:
                errors.append('inconsistent frame count: '+name)
            if field.dtype.kind == 'f':
                for start in range(0, count, 16):
                    if not np.isfinite(field[start:start+16]).all():
                        errors.append('nonfinite data: '+name)
                        break
            if name.endswith(('/rgb', '/rgb_marker')):
                nonempty = 0
                for encoded in field:
                    frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    shape = (480,640,3) if name.startswith('observation/') else (240,320,3)
                    if frame is None or frame.shape != shape:
                        errors.append('invalid image: '+name)
                        break
                    nonempty += int(np.ptp(frame) > 0)
                    images += 1
                if not nonempty:
                    errors.append('image stream is entirely empty: '+name)
        accepted = meta['result'] == 'success'
        reason = str(data['phase'].attrs['terminal_reason'])
        if accepted != (reason == 'success'):
            errors.append('collector and terminal verdict disagree')
        if expect and accepted != (expect == 'success'):
            errors.append('unexpected physical outcome')
        ticks = int(data['step'][-1])-int(data['phase'].attrs['policy_start_sim_step'])
        if ticks > (1200 if task == 'blind_pick' else 3600)+1:
            errors.append('policy timeout exceeded')
        if accepted and task == 'blind_pick':
            physical = meta.get('blind_pick_result', {})
            geometry = meta.get('blind_pick_geometry', {})
            if (not np.isclose(geometry.get('floor_top_m',0),.016,atol=1e-5)
                    or len(geometry.get('wall_tops_m',[])) != 4
                    or not np.allclose(geometry['wall_tops_m'],.197,atol=1e-5)):
                errors.append('box collision panels moved away from their visual shells')
            if not (physical.get('success') and physical.get('held_steps',0) >= 18
                    and physical.get('required_held_steps') == 18
                    and physical.get('object_min_z_m',0) > .197+.015
                    and physical.get('object_hand_distance_m',1) < .06
                    and len(physical.get('bilateral_depth_mm',[])) == 2
                    and np.all(np.asarray(physical['bilateral_depth_mm']) < 28.)):
                errors.append('object was not held clear of the rim')
        if accepted and task == 'play_piano':
            physical = meta.get('memory_result', {})
            notes = physical.get('notes', [])
            cue = meta.get('memory_ground_truth', {}).get('cue_kind')
            if not physical.get('cue_acquired') or cue not in (0,1):
                errors.append('tactile cue was not acquired')
            if [note['key']+21 for note in notes] != [64,62,60,62,64]:
                errors.append('incomplete or incorrect melody')
            for i, note in enumerate(notes if cue in (0,1) else []):
                long = (i+cue)%2
                if note['force_N'] != (3. if long else 1.5) or note['stable_seconds'] < (.6 if long else .2)-1e-9:
                    errors.append('incorrect force or duration')
    return dict(valid=not errors, task=task, seed=seed, outcome=meta['result'],
                policy_physics_ticks=ticks, frames=count, decoded_images=images, errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    parser.add_argument('--expect', choices=('success','failure'))
    args = parser.parse_args()
    reports = [validate_episode(path, args.expect) for path in args.episodes]
    print(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(row['valid'] for row in reports) else 1)
