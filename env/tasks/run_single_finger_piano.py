"""Run: python -m env.tasks.run_single_finger_piano --headless --video."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description='Fixed-palm Allegro index finger playing one piano key')
    parser.add_argument('--video', action='store_true', help='Record RGB video with contact-triggered audio')
    parser.add_argument('--output', type=Path, default=Path('data/single_finger_piano'))
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--target-midi', type=int, default=60)
    parser.add_argument('--hand-usd', help='Optional local Allegro USD; default is the Isaac Lab asset')
    parser.add_argument('--control', choices=['demo','idle','hold'], default='demo')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    import numpy as np
    from .single_finger_piano import Task, TaskCfg

    task = None
    writer = None
    video_steps = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        task = Task(TaskCfg(device=args.device, repetitions=args.repetitions,
                            target_midi=args.target_midi, hand_usd=args.hand_usd))
        if args.video:
            import imageio.v2 as imageio
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import Camera, CameraCfg
            task.camera = Camera(CameraCfg(prim_path='/World/RecordingCamera',
                width=960, height=640, data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=36.,
                                               clipping_range=(.01,10.))))
            task.sim.reset()
            import torch
            task.camera.set_world_poses_from_view(
                torch.tensor([[-.01,.48,.46]],device=task.sim.device),
                torch.tensor([[.40,task.layout[task.target].position[1],.095]],device=task.sim.device))
            writer = imageio.get_writer(args.output/'video.mp4', fps=60, codec='libx264', quality=8)
        task.reset(seed=0)
        if args.video:
            for _ in range(16):
                task.sim.render()
                task.camera.update(task.cfg.physics_dt, force_recompute=True)
        done = False
        while app.is_running() and not done:
            # Smooth 1.2 s strokes. Only the index position target is commanded.
            phase = (task.step_count*task.cfg.physics_dt) % 1.2
            if args.control == 'idle':
                amount = 0.
            elif args.control == 'hold':
                amount = min(1.,task.step_count*task.cfg.physics_dt/.35)
            elif phase < .35:
                amount = phase/.35
            elif phase < .65:
                amount = 1.
            elif phase < 1.:
                amount = 1.-(phase-.65)/.35
            else:
                amount = 0.
            _,_,terminated,truncated,info = task.step(np.array([2*amount-1.],np.float32))
            if writer is not None:
                writer.append_data(task.render())
                video_steps.append(task.step_count)
            done = terminated or truncated
        task.audio_contacts.finish(task.step_count)
        result = dict(task._info(), control=args.control, repetitions=args.repetitions,
            simulated_seconds=task.step_count*task.cfg.physics_dt,
            max_fixed_joint_error_rad=task.max_fixed_joint_error,
            max_palm_displacement_m=task.max_palm_displacement,
            max_fixed_body_displacement_m=task.max_fixed_body_displacement,
            max_target_key_depression_rad=task.max_key_depression,
            key_peak_depressions_rad={str(i+21):float(v) for i,v in enumerate(task.key_peak_angles) if v>.001},
            key_peak_forces_N={str(i+21):float(v) for i,v in enumerate(task.key_peak_forces) if v>.01},
            events=task.score.events,
            audio_events=task.audio_contacts.events,
            scoring_source='PhysX key hinge angles and measured key contact forces')
        if writer is not None:
            writer.close()
            writer = None
            from envs._piano_audio import add_piano_audio
            result['audio'] = add_piano_audio(args.output/'video.mp4', task.audio_contacts.events, video_steps)
        (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('PIANO_RESULT '+json.dumps(result),flush=True)
        if args.control == 'demo' and not result['success']:
            raise RuntimeError('Single-finger demonstration failed; see result.json')
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if task is not None:
            task.close()
        else:
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context is not None:
                context._disable_app_control_on_stop_handle = True
        app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
