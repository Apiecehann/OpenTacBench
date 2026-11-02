"""Native collision/grasp regressions; run with the standalone tactile interpreter.

Known ball poses are used only by this diagnostic, never by the task policy.
RGB rendering is skipped for geometric checks; the texture grasp uses the
original tactile observations and an unchanged, independently verified score.
"""
import argparse
import json


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',choices=('wall','shell','both','aperture','tray','texture','force','all'),default='both')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    args.enable_cameras=True
    app=AppLauncher(args).app
    task=None
    try:
        import numpy as np
        from env.tasks.blind_pick_balls import Task,TaskCfg
        texture=args.check in ('texture','force')
        task=Task(TaskCfg(device=args.device,ball_count=2,target_class='textured' if texture else 'hard',
            collect_all=args.check in ('tray','texture','force','all'),action_repeat=4 if args.check in ('tray','texture','force','aperture','all') else 2,
            material_profiles=('raised_dots',) if texture else ('polished_polymer',),
            image_shape=(240,320) if texture else (120,160),
            workspace='.cache/blind_pick_balls/collision_checks/world'))
        if not texture: task._observation=lambda:None
        def report(label):
            position,quat,_=task._tcp()
            dot=float((quat*task._orientation).sum().abs().clamp(max=1.).cpu())
            print(label+' '+json.dumps(dict(collision=task.physics.collisions.diagnostics(),
                tcp=position[0].cpu().numpy().tolist(),goal=task._goal.tolist(),
                physics_step=task.step_count,orientation_error_rad=float(2*np.arccos(dot)),
                loads_N=task._loads().tolist(),opening_target_m=task._opening,
                opening_actual_m=task.robot.data.joint_pos[0,task.finger_ids].cpu().numpy().tolist(),
                load_target_N=getattr(task,'_diagnostic_load',None),phase=task.phase,failure=task.failure)),flush=True)
        task.on_frame=lambda obs:report('COLLISION_PROGRESS') if task.step_count%120==0 else None
        control_dt=task.cfg.physics_dt*task.cfg.action_repeat
        if texture:
            original_squeeze=task._squeeze
            def squeeze(load):
                task._diagnostic_load=load
                result=original_squeeze(load)
                print('FORCE_PROBE_RESULT '+json.dumps(dict(target_N=load,
                    result=None if result is None else result.tolist(),
                    loads_N=task._loads().tolist(),phase=task.phase,step=task.step_count)),flush=True)
                return result
            task._squeeze=squeeze
        if args.check=='force':
            for label,offset,z in (('centered',[0.,0.],.046),('upper',[0.,0.],.075),
                    ('offset_upper',[.014,-.008],.075)):
                task.reset(seed=0)
                task.physics.reset(np.array([[.546,-.115,.05],[.624,-.115,.05]]))
                task.phase=label
                task.on_frame=lambda obs:report('FORCE_PROGRESS') if task.step_count%120==0 else None
                for _ in range(15):task.step(np.r_[task._goal,.04])
                # Ground-truth positions isolate the controller/height in
                # this diagnostic; the actual blind policy never reads them.
                xy=task.physics.ball_points(0).mean(0)[:2]+offset
                assert task._go(np.r_[xy,.29],opening=.04)
                assert task._go(np.r_[xy,z],opening=.04)
                first=task._squeeze(.5)
                second=task._squeeze(3.) if first is not None else None
                report('FORCE_CASE_RESULT')
                print('FORCE_CASE_END '+label+' '+str(first is not None)+' '+str(second is not None),flush=True)
        if args.check=='texture':
            task.reset(seed=0)
            task.physics.reset(np.array([[.546,-.115,.05],[.624,-.115,.05]]))
            class FirstDeposit(Exception): pass
            def frame(obs):
                if task.step_count%120==0: report('TEXTURE_PROGRESS')
                if task._collected: raise FirstDeposit
            task.on_frame=frame
            try: task.play_once()
            except FirstDeposit: pass
            print('TEXTURE_RESULT '+json.dumps(task.diagnostics()),flush=True)
            # Stop this diagnostic after its first real deposit; the full
            # task's goal/scorer and the tactile controller are unchanged.
            assert 0 in task._collected and task._grasp_verified[0] and task.scores[0].success
            assert task.physics.collisions.max_ball_penetration<=.001
            assert task.physics.collisions.max_fixture_penetration<=.001
            print('TEXTURE_GRASP_PASS',flush=True)
        if args.check in ('tray','all'):
            task.reset(seed=0)
            task.physics.reset(np.array([[-.5,0.,.05],[-.5,.1,.05]]))
            from env.tasks._blind_ball_models import drop_slot
            # Include 10 mm of XY allowance for tactile grasp centering.
            for xy in (drop_slot(11)+np.array([.01,-.01]),drop_slot(0)):
                for z in (.29,.12,.29):
                    assert task._go(np.r_[xy,z],opening=.026,limit=900)
                    report('TRAY_REACH_RESULT')
            assert task.physics.collisions.max_fixture_penetration<=.001
            print('TRAY_REACH_PASS',flush=True)
        if args.check in ('aperture','all'):
            task.reset(seed=0)
            task.physics.reset(np.array([[-.5,0.,.05],[-.5,.1,.05]]))
            assert task._go([.51,0.,.29],opening=.026)
            for part in task.physics.collisions.parts:
                if 'case' in part['body'] or part['body'] in ('panda_hand','panda_link6','panda_link7'):
                    actual=task.physics.collisions.body_transforms()[part['body_id']]
                    world=part['points']@actual[:3,:3].T+actual[:3,3]
                    offset=world-task._tcp()[0][0].cpu().numpy()
                    print('APERTURE_PART '+json.dumps(dict(body=part['body'],low=offset.min(0).tolist(),
                        high=offset.max(0).tolist())),flush=True)
            for x,y in ((.390,-.115),(.390,.115),(.630,.115),(.630,-.115)):
                assert task._go([x,y,.29],opening=.04)
                assert task._go([x,y,.046],opening=.04,limit=round(10/control_dt))
                report('APERTURE_RESULT')
                assert task.physics.collisions.max_fixture_penetration<=.001
                assert task._go([x,y,.29],opening=.04)
            print('APERTURE_PASS',flush=True)
        if args.check in ('wall','both','all'):
            task.reset(seed=0)
            assert task._go([.51,0,.30],opening=.026)
            assert task._go([.51,0,.12],opening=.026)
            reached=task._go([.51,-.28,.12],opening=.026,limit=round(8/control_dt))
            report('WALL_RESULT')
            assert not reached and task.failure=='motion_timeout'
            assert task.physics.collisions.max_fixture_penetration<=.001
            assert task._tcp()[0][0,1]>-.20
            print('WALL_PASS',flush=True)
        if args.check in ('shell','both','all'):
            task.reset(seed=0)
            task.physics.reset(np.array([[.51,0,.05],[.62,.115,.05]]))
            assert task._go([.51,-.09,.30],opening=.008)
            assert task._go([.51,-.09,.046],opening=.008)
            initial=task.physics.ball_points(0).mean(0).copy()
            assert task._go([.51,.025,.046],opening=.008,limit=round(6/control_dt))
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
