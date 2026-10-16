"""Native collision regressions; run with the standalone tactile interpreter.

Known ball poses are used only by this diagnostic, never by the task policy.
RGB rendering is skipped while checking the unchanged physical solvers.
"""
import argparse
import json


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',choices=('wall','shell','both'),default='both')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    args.enable_cameras=True
    app=AppLauncher(args).app
    task=None
    try:
        import numpy as np
        from env.tasks.blind_pick_balls import Task,TaskCfg
        task=Task(TaskCfg(device=args.device,ball_count=2,target_class='hard',
            material_profiles=('polished_polymer',),image_shape=(120,160),
            workspace='.cache/blind_pick_balls/collision_checks/world'))
        task._observation=lambda:None
        def report(label):
            print(label+' '+json.dumps(dict(collision=task.physics.collisions.diagnostics(),
                tcp=task._tcp()[0][0].cpu().numpy().tolist(),failure=task.failure)),flush=True)
        task.on_frame=lambda obs:report('COLLISION_PROGRESS') if task.step_count%120==0 else None
        if args.check in ('wall','both'):
            task.reset(seed=0)
            assert task._go([.51,0,.30],opening=.026)
            assert task._go([.51,0,.12],opening=.026)
            reached=task._go([.51,-.28,.12],opening=.026,limit=480)
            report('WALL_RESULT')
            assert not reached and task.failure=='motion_timeout'
            assert task.physics.collisions.max_fixture_penetration<=.001
            assert task._tcp()[0][0,1]>-.20
            print('WALL_PASS',flush=True)
        if args.check in ('shell','both'):
            task.reset(seed=0)
            task.physics.reset(np.array([[.51,0,.05],[.62,.115,.05]]))
            assert task._go([.51,-.09,.30],opening=.008)
            assert task._go([.51,-.09,.046],opening=.008)
            initial=task.physics.ball_points(0).mean(0).copy()
            assert task._go([.51,.025,.046],opening=.008,limit=360)
            report('SHELL_RESULT')
            assert task.physics.collisions.max_ball_penetration<=.001
            assert task.physics.collisions.max_reaction_force>.1
            displacement=task.physics.ball_points(0).mean(0)-initial
            assert displacement[1]>.02
            print('SHELL_PASS displacement_m='+str(displacement.tolist()),flush=True)
    finally:
        if task is not None:task.close()
        app.close(wait_for_replicator=False)


if __name__=='__main__':main()
