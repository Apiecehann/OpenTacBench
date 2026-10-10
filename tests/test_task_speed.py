import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_four_force_tasks import symbol

motion_speed_scale = symbol('_force_task_utils','motion_speed_scale')
motion_ticks = symbol('_force_task_utils','motion_ticks')


def test_legacy_speed_is_the_default():
    assert motion_speed_scale({}) == 1.
    assert motion_ticks(240, 1.) == 240
    assert motion_ticks(120, 1.) == 120


@pytest.mark.parametrize("value", [0.,.5,4.,float("nan"),float("inf"),True])
def test_invalid_motion_speed_fails(value):
    with pytest.raises(ValueError):
        motion_speed_scale({"motion_speed_scale":value})


def run_empty_wrist(scale, sign):
    """Execute the real unloaded-wrist method, including its joint commands."""
    path = Path(__file__).resolve().parents[1]/'envs/bulb_tightening.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_position_empty_wrist')
    method.body = [node for node in method.body if not isinstance(node,ast.ImportFrom)]
    class Joints(np.ndarray):
        def clone(self):
            return self.copy()
    initial = np.array([[.1,.2,.3,.4,.5,.6,-.8*sign]]).view(Joints)
    task = SimpleNamespace(motion_speed_scale=scale,controller_spin_sign=sign,
        step_count=0,atom_id=0,metadata={},failure=None,commands=[],
        _read_bulb_depths=lambda:[34.,34.],
        _robot_manager=SimpleNamespace(_arm_ids=list(range(7)),robot=SimpleNamespace(
            data=SimpleNamespace(joint_pos=initial,
                soft_joint_pos_limits=np.tile([-3.,3.],(1,7,1))))))
    def execute(task, *, arm, ticks):
        assert ticks == 2
        task.commands.append(np.asarray(arm).copy())
        task.step_count += ticks
        return True,False
    scope = dict(execute_joint_target=execute,motion_ticks=motion_ticks)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    assert scope['_position_empty_wrist'](task)
    return task


@pytest.mark.parametrize('scale',[1,2,3])
@pytest.mark.parametrize('sign',[-1,1])
def test_empty_wrist_keeps_the_same_joint_path_and_endpoints(scale,sign):
    slow,fast = run_empty_wrist(1,sign),run_empty_wrist(scale,sign)
    assert fast.step_count == 120//scale
    np.testing.assert_allclose(fast.commands,slow.commands[scale-1::scale],atol=1e-12)
    assert fast.commands[-1][-1] == pytest.approx(sign*1.65)
    assert fast.metadata['empty_wrist_positions'][0]['end_step'] == fast.step_count


@pytest.mark.parametrize('module,method,target_depth', [
    ('tension_strap','_close_strap_public',27.6),
    ('bulb_tightening','_close_bulb_public',27.2),
])
def test_grasp_controllers_preserve_contact_and_only_accelerate_strap_free_travel(module,method,target_depth):
    path = Path(__file__).resolve().parents[1]/('envs/'+module+'.py')
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Task')
    node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)
    node.body = [n for n in node.body if not isinstance(n,ast.ImportFrom)]
    def run(scale):
        state = {'q':.035, 'commands':[]}
        def depths():
            d = 34. if state['q']>.025 else 28.5-(.025-state['q'])*1000
            return np.array([d,d])
        def execute(task, *, gripper, ticks):
            state['commands'].append((depths().min(),state['q']-gripper))
            state['q'] = gripper
            task.step_count += ticks
            return True,False
        tensor = SimpleNamespace(numpy=depths)
        tensor.detach = lambda:tensor
        tensor.cpu = lambda:tensor
        task = SimpleNamespace(motion_speed_scale=scale,params={},metadata={},step_count=0,
            failure=None,_accepted_result=None,
            _robot_manager=SimpleNamespace(get_gripper_qpos=lambda:state['q']),
            _tactile_manager=SimpleNamespace(get_min_depth=lambda:tensor))
        scope = dict(np=np,next_grip_qpos=symbol('_force_task_utils','next_grip_qpos'),
                     execute_joint_target=execute)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        assert scope[method](task,*([1] if module=='bulb_tightening' else []))
        assert depths().max()<target_depth
        assert all(delta<=1e-4+1e-12 for depth,delta in state['commands'] if depth<=28.5)
        return task.step_count,state['q']
    slow,fast = run(1.),run(3.)
    if module == 'bulb_tightening':
        assert fast == slow
    else:
        assert fast[0]<slow[0]
    assert fast[1] == pytest.approx(slow[1],abs=2e-5)
