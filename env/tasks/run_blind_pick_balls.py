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
    parser.add_argument('--control',choices=('demo','idle'),default='demo')
    parser.add_argument('--idle-frames',type=int,default=120)
    parser.add_argument('--image-width',type=int,choices=(160,320,640),default=640)
    parser.add_argument('--output',type=Path,default=Path('data/blind_pick_balls'))
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
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
            seed=args.seed,record_video=args.video,image_shape=(args.image_width*3//4,args.image_width)))
        args.output.mkdir(parents=True,exist_ok=True)
        if args.video:
            import cv2
            import imageio.v2 as imageio
            task.camera.set_world_poses_from_view(torch.tensor([[.95,-.45,1.10]],device=args.device),
                torch.tensor([[.50,-.08,.09]],device=args.device))
            writer=imageio.get_writer(args.output/'video.mp4',fps=60,codec='libx264',quality=8)
        obs,_=task.reset(seed=args.seed)
        assert task.observation_space.contains(obs)
        print('BLIND_BALLS_READY '+json.dumps(task.diagnostics()),flush=True)
        frames=0
        reported_probes=0
        def record(observation):
            nonlocal frames,reported_probes
            if writer is not None:
                scene=task.render()
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
            frames+=1
            if len(task.trace)>reported_probes:
                print('BLIND_BALLS_PROBE '+json.dumps(task.trace[-1]),flush=True)
                reported_probes=len(task.trace)
            if frames%60==0:
                print(f'BLIND_BALLS t={task.step_count*task.cfg.physics_dt:.2f}s '
                    f'frame={frames} loads={np.round(task._loads(),3).tolist()} '
                    f'tcp={np.round(task._tcp()[0][0].cpu().numpy(),3).tolist()} '
                    f'goal={np.round(task._goal,3).tolist()} '
                    f'ball_z={np.round([task.physics.ball_points(i)[:,2].mean() for i in range(len(task.materials))],3).tolist()} '
                    f'probes={len(task.trace)}',flush=True)
        task.on_frame=record
        if writer is not None:
            for _ in range(16): task.sim.render()
        if args.control=='demo': task.play_once()
        else:
            for _ in range(args.idle_frames):
                task.step(np.r_[task._goal,.04])
                if task._done: break
        result=task.diagnostics()
        result.update(control=args.control,seed=args.seed,frames=frames,
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
