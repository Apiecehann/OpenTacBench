"""Regression: physical success is not the policy's END decision."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from test_force_task_integration import Task, method, helpers, ROOT


NAMES = ('play_piano', 'blind_pick', 'tension_strap', 'bulb_tightening')


def task_for(name):
    """Run production action/scoring/END methods with a deterministic robot."""
    task = Task(force=True)
    task.cfg.policy_controls_termination = True
    task.cfg.step_lim = 500
    task.cfg.save_frequency = 0
    task.cfg.video_frequency = 0
    task.mode = 'eval'
    task.failure = '' if name in NAMES[:2] else None
    task.monitor = task._monitor_strap = task._monitor_bulb = True
    task._accepted_result = None
    task.score = NS(success=True)
    task.retrieval = NS(success=True)
    task.striker_released = True
    task._stage_scorer = NS(success=True, snapshot=lambda: {'success': True})
    task.controller = 'diagnostic_force'
    task.held_steps = 180
    task._probe_physical_complete = True
    task.thread_spec = NS(handedness=1, nominal_free_travel_m=.013)
    task.target_torque, task.torque_tolerance = .12, .012
    task.measured = dict(signed_torque_Nm=.12, torque_Nm=.12, formal_net_turns=2.,
                        formal_advance_m=.013, phase_error_m=0., center_drift_m=0.,
                        tilt_deg=0., seat_force_N=4.)
    task.work = Path('/unused')
    task.params = {}
    task.ruptured = False
    task.lit = False
    task._record = lambda: None
    task._save_metadata = lambda: None
    manager = task._robot_manager
    manager._arm_ids = list(range(7))
    manager.robot.data = NS(joint_pos=torch.zeros((1, 9)))
    manager.get_gripper_qpos = lambda: torch.tensor(.02)
    manager.set_arm = lambda value, *args, **kw: manager.commands.append(value.clone())
    manager.set_gripper = lambda value, **kw: None
    for attr in ('check_success',):
        setattr(task, attr, MethodType(method(f'envs/{name}.py', 'Task', attr), task))
    if name in NAMES[:2]:
        owner, cls = 'envs/_memory_task.py', 'MemoryTask'
    else:
        owner, cls = f'envs/{name}.py', 'Task'
        private = '_finish_strap_episode' if name == 'tension_strap' else '_finish_bulb_episode'
        finish = method(owner, cls, private)
        finish.__globals__['light_after_submission'] = lambda t: setattr(t, 'lit', True)
        setattr(task, private, MethodType(finish, task))
    for attr in ('finish_episode', 'check_early_stop'):
        setattr(task, attr, MethodType(method(owner, cls, attr), task))
    return task


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('via_env_step', [False, True])
def test_physical_success_does_not_end_or_shorten_actions(name, via_env_step):
    task = task_for(name)
    assert task.check_success()
    for _ in range(3):
        if via_env_step:
            _, _, terminated, truncated, info = task.env_step(torch.ones(8))
            assert not terminated and not truncated and not info['success']
        else:
            assert task.take_action(torch.ones(8)) == (True, False)
        assert task.phase_id == task.PHASE_POLICY
        assert task._accepted_result is None and not task.eval_success
    assert task.step_count == 6 and task.take_action_cnt == 3


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('success', [False, True])
def test_explicit_end_scores_current_state_and_is_idempotent(name, success):
    task = task_for(name)
    # A successful earlier state does not rescue a later failed state.
    task.take_action(torch.ones(8))
    if not success:
        if name == 'play_piano':
            task.failure = 'key_overload'
        elif name == 'blind_pick':
            task.retrieval.success = False  # dropped back inside the box
        elif name == 'tension_strap':
            task.failure = 'strap_ruptured'
        else:
            task.measured['torque_Nm'] = .02  # loosened before END
    assert task.finish_episode() is success
    assert task.phase_id == task.PHASE_TERMINAL
    assert task.eval_success is success
    assert task.check_early_stop() is (not success)
    assert task.lit is (success and name == 'bulb_tightening')
    before = task.step_count, len(task._robot_manager.commands)
    assert task.finish_episode() is success
    assert task.take_action(torch.zeros(8)) == (False, success)
    assert (task.step_count, len(task._robot_manager.commands)) == before
    with pytest.raises(RuntimeError, match='TERMINAL'):
        task.env_step(torch.zeros(8))


@pytest.mark.parametrize('name', NAMES)
def test_end_before_handoff_is_rejected(name):
    task = task_for(name)
    task.phase_id = task.PHASE_PRE_MOVE
    with pytest.raises(RuntimeError, match='POLICY'):
        task.finish_episode()
    assert task._accepted_result is None and not task.eval_success


def test_stationary_bulb_command_is_not_an_end_signal():
    task = task_for('bulb_tightening')
    task._reset_grip_released = True
    for _ in range(120):
        assert task.take_action(torch.zeros(8)) == (True, False)
    assert task.step_count == 240  # two seconds, beyond the old 1.5 s END
    assert task.phase_id == task.PHASE_POLICY
    assert task._accepted_result is None and not task.lit
    assert task.check_success()  # read-only external-runner compatibility
    assert task.finish_episode() is True
    assert task.step_count == 240 and task.lit


def test_physical_failure_and_step_limit_still_truncate():
    task = task_for('blind_pick')
    task.failure = 'timeout'
    _, _, terminated, truncated, _ = task.env_step(torch.zeros(8))
    assert not terminated and truncated
    task = task_for('play_piano')
    task.cfg.step_lim = 1
    _, _, terminated, truncated, _ = task.env_step(torch.zeros(8))
    assert not terminated and truncated and not task.eval_success
    assert task.finish_episode() is False  # cannot convert timeout into success


def test_only_the_requested_task_configs_opt_in():
    enabled = set()
    for path in (ROOT / 'envs').glob('*.py'):
        for cls in ast.parse(path.read_text()).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'policy_controls_termination'
                    for t in node.targets
                ) and ast.literal_eval(node.value):
                    enabled.add((path.name, cls.name))
    assert enabled == {('_memory_task.py', 'MemoryTaskCfg'),
                       ('tension_strap.py', 'TaskCfg'), ('bulb_tightening.py', 'TaskCfg')}
    users = set()
    for path in (ROOT / 'envs').glob('*.py'):
        for cls in ast.parse(path.read_text()).body:
            if isinstance(cls, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == 'MemoryTaskCfg' for b in cls.bases
            ):
                users.add(path.stem)
    assert users == {'play_piano', 'blind_pick'}


@pytest.mark.parametrize('submit', [False, True])
def test_repository_evaluator_requires_policy_end(submit):
    task = task_for('tension_strap')
    task.cfg.step_lim = 2
    task.reset = lambda **kw: None
    task.clean_cache = lambda **kw: None
    task.close = lambda: None
    task.sim = NS(stop=lambda: None)
    def evaluate(task, obs):
        task.take_action(torch.ones(8))
        if submit:
            task.finish_episode()
    policy = NS(reset=lambda: None, eval=evaluate)
    result = helpers.run_force_task_episode(task, seed=0, policy=policy)
    assert result == {'test_num': 1, 'succ_num': int(submit)}
