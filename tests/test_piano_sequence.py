from types import SimpleNamespace
import numpy as np
import pytest
from envs._piano_sequence import (parse_notes,make_sequence,note_name,
    PianoAnnotations,configure_piano_task)
from envs._tactile_memory import PianoScore,Note
from env.tasks._single_key_score import SingleKeyScore


def test_user_example_and_enharmonic_octaves():
    assert parse_notes('C#, C, G, A, E, F')==(61,60,67,69,64,65)
    assert parse_notes('Db4 C#4 B#3 Cb4 A0 C8')==(61,61,60,59,21,108)
    assert parse_notes([60,'C#4',67])==(60,61,67)
    assert [note_name(v) for v in (21,60,61,108)]==['A0','C4','C#4','C8']


@pytest.mark.parametrize('value',['','H4','C##4','C9','20','G#8'])
def test_invalid_note_prompts_are_rejected(value):
    with pytest.raises(ValueError): parse_notes(value)


def test_random_prompts_reproduce_notes_and_dynamics_with_seed():
    a=make_sequence(rng=np.random.default_rng(8))
    assert a==make_sequence(rng=np.random.default_rng(8))
    assert a!=make_sequence(rng=np.random.default_rng(9))
    assert len(a.midi)==6 and all(60<=m<=71 for m in a.midi)
    assert set(a.dynamics)<= {'light','heavy'}
    assert make_sequence('C C',dynamics='heavy').dynamics==('heavy','heavy')
    with pytest.raises(ValueError): make_sequence('C G',dynamics=['light','heavy','light'])
    with pytest.raises(ValueError): make_sequence('C',holds=-1)


def test_annotations_distinguish_target_actual_and_no_contact():
    labels=PianoAnnotations(make_sequence('C# C',dynamics='heavy light'),embodiment='tool')
    zero=np.zeros(88)
    idle=labels.update(0,.01,0,zero)
    assert idle['target_midi']==61 and idle['target_dynamic']==2
    assert idle['actual_dynamic']==0 and idle['actual_midi']==-1
    forces=zero.copy();forces[39]=1.5  # Wrong pitch and lighter than the target.
    row=labels.update(1,.01,0,forces)
    assert (row['target_midi'],row['actual_midi'])==(61,60)
    assert row['target_dynamic']==2 and row['actual_dynamic']==1
    assert row['actual_force_N']==1.5 and row['target_force_N']==3.
    assert row['physics_step']==1 and row['time_s']==.01
    assert labels.update(1,.01,1,zero) is row  # Repeated reads cannot change a saved tick.
    assert len(labels.records)==2
    done=labels.update(2,.01,2,zero)
    assert done['target_midi']==-1 and done['target_dynamic']==0


def test_hand_black_key_threshold_is_reported_explicitly():
    labels=PianoAnnotations(make_sequence('C# C',dynamics='light'),embodiment='hand')
    force=np.zeros(88);force[40]=.7
    row=labels.update(1,1/240,0,force)
    assert row['actual_dynamic']==1 and row['heavy_threshold_N']==.85
    assert row['target_force_N']==pytest.approx(.7)


def test_tool_pitch_only_ignores_dynamics_and_maximum_hold():
    score=PianoScore(.01,key_sequence=(0,0),key_count=2,
        notes=(Note(3.,.1),Note(1.5,.1)),require_dynamics=False)
    for _ in range(100): score.advance([.5,0.])
    assert not score.failure and not score.success and score.index==0
    score.advance([0.,0.]);assert score.index==1
    for _ in range(20): score.advance([3.,0.])
    score.advance([0.,0.])
    assert score.success
    assert score.events[0]['force_N']==.5  # Measured rather than requested force.


def test_tool_strict_dynamics_and_retry_wrong_notes():
    strict=PianoScore(.01,key_sequence=(0,),key_count=2,notes=(Note(1.5,.1),))
    for _ in range(8): strict.advance([3.,0.])
    assert strict.failure=='wrong_note_force'
    retry=PianoScore(.01,key_sequence=(0,),key_count=2,notes=(Note(1.5,.1),),
        require_dynamics=False,wrong_key_policy='retry')
    for _ in range(20): retry.advance([0.,1.5])
    assert retry.mistakes==1 and not retry.failure and retry.index==0
    retry.advance([0.,0.])
    retry.advance([0.,1.5])
    assert retry.mistakes==2 and retry.index==0
    retry.advance([0.,0.])
    for _ in range(15): retry.advance([1.5,0.])
    retry.advance([0.,0.]);assert retry.success


def test_hand_dynamics_is_optional_and_requires_release():
    angles=np.zeros(88);force=np.zeros(88);angles[39]=.04;force[39]=.6
    score=SingleKeyScore(key_sequence=(39,),force_targets=(.35,),force_tolerances=(.07,))
    for n in range(20): score.advance(angles,force,.01,n)
    assert not score.success and not score.failure
    for n in range(20,31): score.advance(np.zeros(88),np.zeros(88),.01,n)
    assert score.success
    strict=SingleKeyScore(key_sequence=(39,),force_targets=(.35,),force_tolerances=(.07,),require_dynamics=True)
    for n in range(10): strict.advance(angles,force,.01,n)
    assert strict.failure=='wrong_note_force'


def test_hand_wrong_key_retry_does_not_advance_the_sequence():
    score=SingleKeyScore(key_sequence=(39,),wrong_key_policy='retry')
    angles=np.zeros(88);angles[40]=.04
    for n in range(20): score.advance(angles,np.ones(88),.01,n)
    assert score.mistakes==1 and score.count==0 and not score.failure
    angles[40]=0.;angles[39]=.04
    for n in range(20,40): score.advance(angles,np.ones(88),.01,n)
    for n in range(40,51): score.advance(np.zeros(88),np.zeros(88),.01,n)
    assert score.success


def test_hand_partial_score_continues_after_wrong_key_and_counts_successes():
    score=SingleKeyScore(key_sequence=(39,40,41),continue_on_note_failure=True)
    wrong=np.zeros(88);wrong[42]=.04
    score.advance(wrong,np.ones(88),.01,1)
    assert score.count==0 and not score.failure
    score.advance(wrong,np.ones(88),.01,2)
    assert score.mistakes==1
    score.finish_attempt(0,3)
    assert score.count==1 and score.successful_notes==0 and not score.finished
    correct=np.zeros(88);correct[40]=.04
    for step in range(4,14): score.advance(correct,np.ones(88),.01,step)
    for step in range(14,25): score.advance(np.zeros(88),np.zeros(88),.01,step)
    score.finish_attempt(1,30)  # Successful release is already consumed.
    assert score.count==2 and score.successful_notes==1
    score.finish_attempt(2,31)  # An attempted note with no valid contact.
    assert score.finished and not score.success and score.failure=='partial_score'
    assert score.successful_notes==1
    assert [e['kind'] for e in score.events]==['miss','press','release','miss']
    assert [e['attempt_index'] for e in score.events]==[0,1,1,2]


def test_hand_partial_score_sustained_overforce_is_one_failed_attempt():
    score=SingleKeyScore(key_sequence=(39,40),force_targets=(.35,.5),
        force_tolerances=(.07,.07),require_dynamics=True,
        continue_on_note_failure=True)
    angles=np.zeros(88);forces=np.zeros(88)
    angles[39]=.04;forces[39]=.6
    for step in range(12): score.advance(angles,forces,.01,step)
    assert score.count==0 and score.successful_notes==0 and not score.failure
    score.finish_attempt(0,13)
    assert score.events[0]['reason']=='wrong_note_force'
    assert score.count==1
    angles[:]=forces[:]=0.;angles[40]=.04;forces[40]=.5
    for step in range(14,24): score.advance(angles,forces,.01,step)
    for step in range(24,35): score.advance(np.zeros(88),np.zeros(88),.01,step)
    assert score.finished and score.successful_notes==1
    assert score.failure=='partial_score'


def test_yaml_overrides_only_apply_to_piano_configs():
    other=SimpleNamespace()
    configure_piano_task(other,{'piano_notes':['C#','C']})
    assert vars(other)=={}
    piano=SimpleNamespace(piano_notes=None)
    configure_piano_task(piano,{'piano_notes':['C#','C'],'piano_require_dynamics':False})
    assert piano.piano_notes==['C#','C'] and piano.piano_require_dynamics is False


def test_tool_observation_labels_current_tick_without_advancing_score():
    # Exercise the real task methods without loading Isaac. The collector
    # reads observations before MemoryTask runs its physical scoring hook.
    import ast
    from collections import deque
    from pathlib import Path
    source=ast.parse((Path(__file__).parents[1]/'envs/play_piano.py').read_text(encoding='utf-8'))
    task_node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    methods=[n for n in task_node.body if isinstance(n,ast.FunctionDef) and n.name in
             ('_get_observations','observe_task','_sample_key_forces')]
    class Base:
        def _get_observations(self): return {}
    task_class=ast.ClassDef(name='Task',bases=[ast.Name(id='Base',ctx=ast.Load())],
                           keywords=[],body=methods,decorator_list=[])
    namespace={'Base':Base,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[task_class],type_ignores=[])),
                 'piano_task_contact_hooks','exec'),namespace)
    task=namespace['Task']()
    task.cfg=SimpleNamespace(sim=SimpleNamespace(dt=.01))
    task.monitor=True;task.step_count=1;task.key_forces=np.zeros(88)
    task._key_force_history=deque(maxlen=3);task._key_vertex_ids=np.array([39])
    task._piano_observed_step=task._piano_force_step=None
    task.striker_released=True;task.metadata={};task.failure=''
    task._robot_manager=SimpleNamespace(get_gripper_center_pose=lambda:SimpleNamespace(p=np.zeros(3)))
    task.score=PianoScore(.01,key_sequence=(39,),key_count=88,notes=(Note(1.5,.01),),require_dynamics=False)
    task.piano_annotations=PianoAnnotations(make_sequence('C4',dynamics='light'),embodiment='tool')
    contact_reads=[];audio_ticks=[]
    def contacts():
        contact_reads.append(task.step_count)
        return np.array([0]),np.array([[0.,0.,.00015 if task.step_count==1 else 0.]])
    task.uipc_sim=SimpleNamespace(get_contact_gradient=contacts)
    task.audio_contacts=SimpleNamespace(advance=lambda step,forces:audio_ticks.append(step))
    first=task._get_observations()['piano']
    assert first['actual_midi']==60 and first['actual_force_N']==pytest.approx(1.5)
    assert task.score.stable_steps==0 and not task.score.pressed
    task.observe_task();task.observe_task()
    assert task.score.stable_steps==1 and task.score.index==0
    task.step_count=2
    released=task._get_observations()['piano']
    assert released['target_index']==0 and released['actual_dynamic']==0
    task.observe_task()
    assert task.score.success and task._get_observations()['piano'] is released
    assert contact_reads==[1,2] and audio_ticks==[1,2]
