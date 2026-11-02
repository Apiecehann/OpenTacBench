"""Physical controls: idle, held contact, release, reset and wrong-key rejection.

Run with the isolated tactile Python through Isaac Sim's python.sh.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('data/shadow_piano_calibrated/controls.json'))
args=parser.parse_args()
app=AppLauncher(headless=True,enable_cameras=True).app


def main():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from env.tasks.shadow_piano import Task,TaskCfg
    task=None
    try:
        task=Task(TaskCfg(tactile=True,melody=(60,67),episode_seconds=5.,
            tactile_workspace='.cache/piano_tactile/controls'))
        initial,_=task.reset(seed=0)
        assert task.observation_space.contains(initial)
        assert set(initial['tactile'])=={'index'}
        rest=initial['tactile']['index']
        rest_optical=task.tactile.sensor.get_observations(['rgb'])['rgb'].cpu().numpy().copy()
        assert set(rest)=={'rgb'}
        assert rest['rgb'].shape==(480,640,3) and rest['rgb'].dtype==np.uint8
        # Physical checks inspect the solver directly; observations expose RGB only.
        reference_nodes=task.tactile.sensor.get_init_pts().cpu().numpy().copy()
        def displacement():
            return task.tactile.sensor.get_init_pts().cpu().numpy()-reference_nodes
        for _ in range(20):
            obs,_,term,trunc,_=task.step(task.demonstration_action('idle'))
            assert not term and not trunc
        assert task.score.count==0 and len(task.score.events)==0
        assert np.max(np.abs(task._key_state()[0]))<.001
        idle_signal=float(np.linalg.norm(displacement(),axis=-1).max())
        assert idle_signal<1e-5
        print('IDLE_OK',flush=True)
        for _ in range(90):
            obs,_,term,trunc,_=task.step(task.demonstration_action('hold'))
            assert not term and not trunc
        contact=obs['tactile']['index']
        assert task.score.count==0
        assert [e['kind'] for e in task.score.events]==['press']
        # Use the unloaded numerical residual as the signal floor, rather than
        # demanding an arbitrary indentation from a particular elastic modulus.
        deformation_signal=float(np.linalg.norm(displacement(),axis=-1).max())
        assert deformation_signal>max(1e-6,20*idle_signal)
        rgb_change=float(np.abs(contact['rgb'].astype(float)-rest['rgb']).max())
        assert rgb_change>5
        optical=task.tactile.sensor.get_observations(['rgb'])['rgb'].cpu().numpy()
        optical_change=float(np.abs(optical.astype(float)-rest_optical).max())
        assert optical_change>5  # Independent of the original marker brightness noise.
        assert task.observation_space.contains(obs)
        print('HOLD_AND_TACTILE_OK',flush=True)
        for _ in range(30):
            obs,_,term,trunc,_=task.step(task.demonstration_action('idle'))
            assert not term and not trunc
        assert task.score.count==1
        assert [e['kind'] for e in task.score.events]==['press','release']
        assert len(task.audio_contacts.events)==1
        assert np.linalg.norm(displacement(),axis=-1).max()<1e-5
        print('RELEASE_OK',flush=True)
        obs,_=task.reset(seed=0)
        assert task.score.count==0 and len(task.score.events)==0
        assert np.max(np.abs(task._key_state()[0]))<1e-5
        reset_error=int(np.abs(obs['tactile']['index']['rgb'].astype(int)-rest['rgb']).max())
        # The original renderer adds independent visual noise to the ink dots.
        # Check deterministic optical RGB and physical reset independently.
        reset_optical=task.tactile.sensor.get_observations(['rgb'])['rgb'].cpu().numpy()
        reset_optical_error=int(np.abs(reset_optical.astype(int)-rest_optical).max())
        assert reset_optical_error<=1
        assert np.linalg.norm(displacement(),axis=-1).max()<1e-5
        markers=task.tactile.sensor.get_observations(['marker'])['marker'].cpu().numpy()
        assert np.linalg.norm(markers[1]-markers[0],axis=-1).max()<.5
        print('RESET_OK',flush=True)
        # Lift, move above G4, lower, then flex while C4 is the required note.
        y=task.layout[67-21].position[1]-.033
        start=float(task.hand.data.joint_pos[0,task._y])
        for i in range(24):
            obs,_,term,trunc,_=task.step(np.array([start+(y-start)*(i+1)/24,.03,0.]))
            assert not term and not trunc
        for i in range(12):
            obs,_,term,trunc,_=task.step(np.array([y,.03*(1-(i+1)/12),0.]))
            assert not term and not trunc
        wrong_info=None
        for i in range(40):
            _,_,term,trunc,info=task.step(np.array([y,0.,task.cfg.press_joint_angle*min(1,(i+1)/12)]))
            if term or trunc:
                wrong_info=info
                break
        assert wrong_info is not None and wrong_info['failure']=='wrong_key'
        print('WRONG_KEY_OK',flush=True)
        result=dict(idle=True,hold_not_repeated=True,release=True,reset=True,wrong_key=True,
            observation_contract=True,rgb_change_max_uint8=rgb_change,reset_rgb_error_uint8=reset_error,
            optical_rgb_change_max_uint8=optical_change,
            reset_optical_rgb_error_uint8=reset_optical_error,
            idle_displacement_m=idle_signal,
            loaded_displacement_m=deformation_signal,
            tactile=task.tactile.metadata())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
        print('TACTILE_CONTROLS '+json.dumps(result),flush=True)
    finally:
        if task is not None:
            task.close()


exit_code=0
try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code=1
finally:
    app.close(wait_for_replicator=False)
raise SystemExit(exit_code)
