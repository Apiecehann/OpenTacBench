"""Run the isolated touch-only multi-material ball task."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balls',type=int,choices=range(2,13),default=12)
    parser.add_argument('--target',choices=('hard','soft','textured'),default='textured')
    from ._blind_ball_models import MATERIALS,DEFAULT_PROFILES
    parser.add_argument('--materials',nargs='+',choices=[m.name for m in MATERIALS],default=DEFAULT_PROFILES)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--collect-all',action='store_true',help='Retrieve every ball into a larger receiving tray')
    parser.add_argument('--video-fps',type=int,choices=(30,60),default=30)
    parser.add_argument('--control-hz',type=int,choices=(30,60),default=60,
        help='Control and tactile observation frequency; physics stays at 120 Hz')
    parser.add_argument('--control',choices=('demo','idle'),default='demo')
    parser.add_argument('--idle-frames',type=int,default=120)
    parser.add_argument('--image-width',type=int,choices=(160,320,640),default=640)
    parser.add_argument('--output',type=Path,default=Path('data/blind_pick_balls'))
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.video and args.video_fps>args.control_hz: parser.error('--video-fps must not exceed --control-hz')
    args.enable_cameras=True  # The calibrated GelSight sensors use their USD cameras.
    app=AppLauncher(args).app
    task=writer=None
    failure=None
    try:
        import numpy as np
        import torch
        from .blind_pick_balls import Task,TaskCfg
        task=Task(TaskCfg(device=args.device,ball_count=args.balls,target_class=args.target,
            material_profiles=tuple(args.materials),
            collect_all=args.collect_all,episode_seconds=900. if args.collect_all else 240.,
            action_repeat=120//args.control_hz,
            seed=args.seed,record_video=args.video,image_shape=(args.image_width*3//4,args.image_width)))
        args.output.mkdir(parents=True,exist_ok=True)
        if args.video:
            import cv2
            import imageio.v2 as imageio
            eye=[.78,-.50,1.20] if args.collect_all else [.88,-.43,.90]
            look=[.50,-.16,.09] if args.collect_all else [.50,-.08,.09]
            task.camera.set_world_poses_from_view(torch.tensor([eye],device=args.device),
                torch.tensor([look],device=args.device))
            writer=imageio.get_writer(args.output/'video.mp4',fps=args.video_fps,codec='libx264',quality=8)
        obs,_=task.reset(seed=args.seed)
        assert task.observation_space.contains(obs)
        print('BLIND_BALLS_READY '+json.dumps(task.diagnostics()),flush=True)
        frames=0
        control_steps=0
        reported_probes=0
        def record(observation,final=False):
            nonlocal frames,reported_probes,control_steps
            if not final: control_steps+=1
            if writer is not None and (final or (control_steps-1)%(args.control_hz//args.video_fps)==0):
                scene=task.render()
                cv2.rectangle(scene,(12,6),(610,42),(18,22,28),-1)
                cv2.putText(scene,f'{task._info()["collected_count"]}/{args.balls if args.collect_all else 1} collected  '
                    f'{task.step_count*task.cfg.physics_dt:.1f}s  {task.phase}',(20,32),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA)
                panel=np.full((720,480,3),(18,22,28),np.uint8)
                for i,name in enumerate(('left','right')):
                    rgb=cv2.resize(observation['tactile'][name]['rgb'],(448,336))
                    top=24+i*360
                    panel[top:top+336,16:464]=rgb
                    cv2.putText(panel,'GelSight Mini - '+name,(16,i*360+17),
                        cv2.FONT_HERSHEY_SIMPLEX,.46,(240,244,248),1,cv2.LINE_AA)
                frame=np.concatenate([scene,panel],axis=1)
                writer.append_data(frame)
                if frames==0: imageio.imwrite(args.output/'scene.png',frame)
                if frames%(args.video_fps*2)==0:
                    imageio.imwrite(Path(task.cfg.workspace).parent/'progress.png',frame)
                if final: imageio.imwrite(args.output/'final.png',frame)
                frames+=1
            if len(task.trace)>reported_probes:
                print('BLIND_BALLS_PROBE '+json.dumps(task.trace[-1]),flush=True)
                reported_probes=len(task.trace)
            if control_steps%args.control_hz==0:
                print(f'BLIND_BALLS t={task.step_count*task.cfg.physics_dt:.2f}s '
                    f'frame={frames} loads={np.round(task._loads(),3).tolist()} '
                    f'tcp={np.round(task._tcp()[0][0].cpu().numpy(),3).tolist()} '
                    f'goal={np.round(task._goal,3).tolist()} '
                    f'ball_z={np.round([task.physics.ball_points(i)[:,2].mean() for i in range(len(task.materials))],3).tolist()} '
                    f'probes={len(task.trace)} collected={len(task._collected)} phase={task.phase}',flush=True)
        task.on_frame=record
        if writer is not None:
            for _ in range(16): task.sim.render()
        if args.control=='demo': task.play_once()
        else:
            for _ in range(args.idle_frames):
                task.step(np.r_[task._goal,.04])
                if task._done: break
        if writer is not None: record(task._observation(),final=True)
        result=task.diagnostics()
        result.update(control=args.control,seed=args.seed,frames=frames,
            video_fps=args.video_fps,control_hz=args.control_hz,control_steps=control_steps,
            simulated_seconds=task.step_count*task.cfg.physics_dt)
        (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('BLIND_BALLS_RESULT '+json.dumps(result),flush=True)
        if args.control=='demo' and not result['success']: raise RuntimeError('Ball retrieval failed; inspect result.json')
    except BaseException as error:
        import traceback
        traceback.print_exc(); failure=error
    finally:
        if writer is not None: writer.close()
        if task is not None: task.close()
        app.close(wait_for_replicator=False)
    if failure is not None: raise SystemExit(1)


if __name__=='__main__': main()
