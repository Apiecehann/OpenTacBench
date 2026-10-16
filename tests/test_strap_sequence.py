import pytest
from envs._strap_sequence import StrapAcceptanceLimits, StrapLifecycle


def test_two_different_durations_and_descending_forces_use_physical_time():
    limits=StrapAcceptanceLimits.from_parameters({'force_time_sequence':[17, .25, 13, .5]})
    scorer=StrapLifecycle(0,limits)
    for step in range(1,31):scorer.advance(step,17,grasped=True)
    assert scorer.stage_index==0
    scorer.advance(31,17,grasped=True)
    assert scorer.stage_index==1 and not scorer.success
    for step in range(32,92):scorer.advance(step,13,grasped=True)
    assert not scorer.success
    scorer.advance(92,13,grasped=True)
    assert scorer.success and scorer.completed_steps==[31,92]
    assert scorer.snapshot()['hold_durations_s']==[.25,.5]


@pytest.mark.parametrize('sequence',[
    [],[12,1,18],[12,1,18,1,15,1],[-.01,1,18,1],[12,1,24.01,1],
    [12,0,18,1],[12,-1,18,1],[12,1,18,float('inf')],
    [float('nan'),1,18,1],[True,1,18,1],[12,True,18,1],
])
def test_rejects_unsupported_force_time_sequences(sequence):
    with pytest.raises((ValueError,TypeError)):
        StrapAcceptanceLimits.from_parameters({'force_time_sequence':sequence})


def test_long_user_holds_extend_the_policy_timeout():
    limits=StrapAcceptanceLimits.from_parameters({'force_time_sequence':[12,75,18,125]})
    assert limits.timeout_seconds>=230
    assert '75 seconds' in limits.instruction() and '125 seconds' in limits.instruction()


def test_zero_and_maximum_targets_preserve_physical_force_acceptance():
    limits=StrapAcceptanceLimits.from_parameters({'force_time_sequence':[0,.1,24,.2]})
    assert limits.controller_reference(0)==0 and limits.controller_reference(24)==23.75
    assert limits.targets_N==(0,24)
    scorer=StrapLifecycle(0,limits)
    for step in range(1,14):scorer.advance(step,.1,grasped=True)
    assert scorer.stage_index==1
    for step in range(14,39):scorer.advance(step,23.75,grasped=True)
    assert scorer.success
    scorer.advance(39,24.01,grasped=True,damage='overload')
    assert not scorer.success and scorer.failure=='overload'


def test_configured_physical_limit_controls_public_range():
    assert StrapAcceptanceLimits.from_parameters({'overload_N':20}).maximum_force_N==20
    with pytest.raises(ValueError):
        StrapAcceptanceLimits.from_parameters({'overload_N':20,'force_time_sequence':[0,1,21,1]})


def test_zero_tension_without_a_grip_cannot_complete_the_first_stage():
    scorer=StrapLifecycle(0,StrapAcceptanceLimits((0,12),hold_seconds=(.1,.1)))
    for step in range(1,101):scorer.advance(step,0,grasped=False)
    assert scorer.stage_index==0 and not scorer.success
    for step in range(101,114):scorer.advance(step,0,grasped=True)
    assert scorer.stage_index==1 and not scorer.success


def test_legacy_parameters_keep_the_same_two_stage_contract():
    limits=StrapAcceptanceLimits.from_parameters({'target_sequence_N':[12,18],'hold_seconds':3})
    assert limits.as_sequence()==[12,3,18,3] and limits.timeout_seconds==60


@pytest.mark.parametrize('timeout',[-1,0,float('nan')])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError):
        StrapAcceptanceLimits.from_parameters({'policy_timeout_seconds':timeout})


def test_out_of_band_sample_resets_only_the_current_hold():
    scorer=StrapLifecycle(0,StrapAcceptanceLimits((12,18),hold_seconds=(.1,.2)))
    for step in range(1,14):scorer.advance(step,12,grasped=True)
    for step in range(14,37):scorer.advance(step,18,grasped=True)
    scorer.advance(37,18.5001,grasped=True)
    for step in range(38,62):scorer.advance(step,18,grasped=True)
    assert scorer.stage_index==1 and not scorer.success
    scorer.advance(62,18,grasped=True)
    assert scorer.success
