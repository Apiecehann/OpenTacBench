"""Run: python -m env.tasks.run_single_finger_piano --headless --video."""
import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description='One-finger piano: Shadow Hand playing Twinkle or the original Allegro repeated-key task')
    parser.add_argument('--song', choices=['twinkle','repeated'], default='twinkle')
    parser.add_argument('--wrist',choices=['arm','rail'],default='arm',help='Shadow wrist support: Franka arm (default) or the previous slides')
    parser.add_argument('--notes', type=int, help='Optional Twinkle prefix length for a smoke run; default is all 42 notes')
    parser.add_argument('--video', action='store_true', help='Record RGB video with contact-triggered audio')
    parser.add_argument('--camera-view',choices=('wide','fingertip'),default='wide',help='Observer framing; leaves tactile optics unchanged')
    parser.add_argument('--tactile', action='store_true', help='RGB tactile sensing on the Shadow index finger only')
    parser.add_argument('--tactile-sensor',choices=('digit','gsmini'),help='DIGIT/TACTO fingertip (arm default) or the previous unscaled Mini/UIPC assembly')
    parser.add_argument('--image-width',type=int,choices=(320,640),default=640,help='Tactile RGB width; keeps the selected sensor\'s complete field of view')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--target-midi', type=int, default=60)
    parser.add_argument('--hand-usd', help='Optional local USD matching the selected hand model')
    parser.add_argument('--control', choices=['demo','idle','hold'], default='demo')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.tactile and args.song != 'twinkle':
        parser.error('--tactile currently uses the Shadow Hand / twinkle task')
    if args.video or args.tactile:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    import signal
    def stop_requested(signum, frame):
        raise KeyboardInterrupt('Simulation stop requested')
    signal.signal(signal.SIGTERM,stop_requested)
    import numpy as np
    if args.song=='twinkle':
        if args.wrist=='arm':
            from .arm_shadow_piano import Task,TaskCfg
        else:
            from .shadow_piano import Task,TaskCfg
        from ._twinkle import TWINKLE
        if args.notes is not None and not 1<=args.notes<=len(TWINKLE):
            parser.error('--notes must be between 1 and 42')
        cfg=TaskCfg(device=args.device,hand_usd=args.hand_usd,tactile=args.tactile,record_video=args.video,
                    tactile_image_width=args.image_width,tactile_sensor=args.tactile_sensor or ('digit' if args.wrist=='arm' else 'gsmini'),
                    melody=TWINKLE[:args.notes] if args.notes is not None else TWINKLE)
    else:
        from .single_finger_piano import Task, TaskCfg
        cfg=TaskCfg(device=args.device,repetitions=args.repetitions,record_video=args.video,
                    target_midi=args.target_midi,hand_usd=args.hand_usd)
    args.output=args.output or Path('data/arm_shadow_piano_digit' if args.song=='twinkle' and args.tactile and cfg.tactile_sensor=='digit' else
        'data/arm_shadow_piano' if args.song=='twinkle' and args.wrist=='arm' else
        'data/shadow_piano_calibrated' if args.tactile else
        'data/shadow_piano' if args.song=='twinkle' else 'data/single_finger_piano')

    task = None
    fps=1/(cfg.physics_dt*cfg.action_repeat)
    writer = None
    rgb_writer = None
    tactile_recorder = None
    failure = None
    video_steps = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        task = Task(cfg)
        if args.video:
            import imageio.v2 as imageio
            import torch
            task.camera.set_world_poses_from_view(
                torch.tensor([[.22,.16,.24] if args.camera_view=='fingertip' else
                    [-.58,.85,.66] if args.song=='twinkle' and args.wrist=='arm' else [-.01,.48,.46]],device=task.sim.device),
                torch.tensor([[.46 if args.camera_view=='fingertip' else .12 if args.song=='twinkle' and args.wrist=='arm' else .40,
                    task.layout[task.target].position[1],.105 if args.camera_view=='fingertip' else .095]],device=task.sim.device))
            writer = imageio.get_writer(args.output/'video.mp4', fps=fps, codec='libx264', quality=8)
            if args.tactile:
                rgb_writer=imageio.get_writer(args.output/('digit.mp4' if cfg.tactile_sensor=='digit' else 'gelsight.mp4'),fps=fps,codec='libx264',quality=8)
        initial,_=task.reset(seed=0)
        assert task.observation_space.contains(initial)
        print('PIANO_READY',flush=True)
        if args.tactile:
            from ._tactile_recording import TactileRecorder
            tactile_recorder=TactileRecorder(args.output/'tactile.h5',task)
        if args.video:
            for _ in range(16):
                task.sim.render()
                task.camera.update(task.cfg.physics_dt, force_recompute=True)
        execution_started=time.perf_counter()
        done = False
        while app.is_running() and not done:
            # The original fixed-palm task retains its one-dimensional action.
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
            action=(task.demonstration_action(args.control) if args.song=='twinkle'
                    else np.array([2*amount-1.],np.float32))
            observation,_,terminated,truncated,info = task.step(action)
            if tactile_recorder is not None:
                tactile_recorder.append(observation)
                if tactile_recorder.count % 30 == 0:
                    print(f'PIANO_TACTILE t={task.step_count*cfg.physics_dt:.2f}s '
                          f'notes={task.score.count}/{task.score.repetitions} '
                          f'indentation={task.tactile.peak_indentation*1000:.3f}mm '
                          f'RGB={observation["tactile"]["index"]["rgb"].shape}',flush=True)
            if writer is not None:
                frame=task.render()
                if args.tactile:
                    from ._tactile_recording import tactile_panel
                    rgb_writer.append_data(observation['tactile']['index']['rgb'])
                    frame=tactile_panel(frame,observation['tactile']['index'],task.tactile.display_name)
                writer.append_data(frame)
                video_steps.append(task.step_count)
            done = terminated or truncated
        task.audio_contacts.finish(task.step_count)
        result = dict(task._info(), control=args.control, song_profile=args.song,
            execution_wall_seconds=time.perf_counter()-execution_started,
            physics_dt_s=cfg.physics_dt,action_repeat=cfg.action_repeat,control_frequency_hz=fps,
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
        if args.song=='twinkle':
            result.update(expected_midi=list(cfg.melody),max_palm_rotation_rad=task.max_palm_rotation,
                          fixed_body_measurement_frame='relative to the moving palm')
        if args.tactile:
            result['tactile']=task.tactile.metadata()
            if task.tactile.physics_backend=='uipc':
                result['scoring_source']='UIPC key hinge poses and measured gel/key contact reactions'
        if writer is not None:
            writer.close()
            writer = None
            from envs._piano_audio import add_piano_audio
            result['audio'] = add_piano_audio(args.output/'video.mp4', task.audio_contacts.events, video_steps,fps=fps)
        (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('PIANO_RESULT '+json.dumps(result),flush=True)
        if args.control == 'demo' and not result['success']:
            raise RuntimeError('Single-finger demonstration failed; see result.json')
    except BaseException as error:
        import traceback
        traceback.print_exc()
        failure = error
    finally:
        if rgb_writer is not None:
            rgb_writer.close()
        if tactile_recorder is not None:
            tactile_recorder.close()
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
    if failure is not None:
        raise SystemExit(130 if isinstance(failure,KeyboardInterrupt) else 1)


if __name__ == '__main__':
    main()
