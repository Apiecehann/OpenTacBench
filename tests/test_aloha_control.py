"""Six-axis commands use named DOFs, even if USD import reorders fingers."""
import json

import pytest
import torch

from test_force_task_integration import Task,helpers


@pytest.mark.parametrize('live',[True,False])
@pytest.mark.parametrize('arm_count',[6,7])
def test_public_actions_select_actual_dofs_and_preserve_control_order(live,arm_count,tmp_path):
    task=Task(force=True)
    manager=task._robot_manager
    # Put one finger first and the other before the last arm DOF, unlike Panda.
    manager._arm_ids=torch.tensor([*range(1,arm_count),arm_count+1])
    manager._gripper_ids=torch.tensor([0,arm_count])
    manager.live=torch.arange(arm_count+2,dtype=torch.float32)/10
    manager.cached=manager.live+1
    task.cfg.live_action_joint_state=live
    task.cfg.public_action_trace_path=str(tmp_path/'actions.jsonl')
    task.mode='collect'
    source=manager.live if live else manager.cached
    expected=source[[*manager._arm_ids.tolist(),0]]
    target=torch.zeros(arm_count+1)
    assert task.take_action(target)==(True,False)
    first,second=manager.commands
    torch.testing.assert_close(first[0],expected[:-1]/2)
    torch.testing.assert_close(first[2],expected[-1]/2)
    torch.testing.assert_close(second[0],torch.zeros(arm_count))
    assert second[2]==0
    row=json.loads((tmp_path/'actions.jsonl').read_text())
    assert row['initial_qpos']==pytest.approx(expected.tolist())
    assert len(row['qpos'])==arm_count+1
    assert (row['before_step'],row['after_step'])==(0,2)


def test_expert_uses_the_same_six_axis_public_action(monkeypatch):
    task=Task(force=True)
    manager=task._robot_manager
    manager._arm_ids=torch.tensor([1,2,3,4,5,7])
    manager._gripper_ids=torch.tensor([0,6])
    manager.cached=torch.arange(8,dtype=torch.float32)
    received=[]
    task.take_action=lambda q,**kwargs:(received.append(q.clone()) or True,False)
    helpers.execute_joint_target(task,gripper=.02,ticks=2)
    torch.testing.assert_close(received[0],torch.tensor([1.,2.,3.,4.,5.,7.,.02]))


@pytest.mark.parametrize('width',[7,8])
def test_public_command_schedule_preserves_the_actual_action_width(tmp_path,width):
    import numpy as np
    path=tmp_path/'actions.jsonl'
    path.write_text(json.dumps(dict(before_step=0,after_step=2,requested_repeat=2,
        qpos=[1.]*width,initial_qpos=[0.]*width))+'\n')
    schedule,_=helpers.original_action_schedule([0,2,4],np.zeros((3,width)),0,path,
        terminal_step=5,terminal_joint=np.zeros(width))
    assert [r['source_step'] for r in schedule]==[2,4,5]
    assert [len(r['target']) for r in schedule]==[width]*3
    assert schedule[0]['target']==[1.]*width
