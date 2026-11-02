"""Public two-level strap instructions and their physical-time acceptance."""
from dataclasses import dataclass
import math


SUPPORTED_FORCE_RANGE_N = (0., 24.)


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


@dataclass(frozen=True)
class StrapAcceptanceLimits:
    targets_N: tuple = (12., 18.)
    tolerance_N: float = .5
    hold_seconds: float | tuple = 3.
    timeout_seconds: float = 60.
    physics_dt: float = 1/120
    maximum_force_N: float = SUPPORTED_FORCE_RANGE_N[1]

    def __post_init__(self):
        if len(self.targets_N) != 2:
            raise ValueError('Exactly two strap force levels are required')
        targets = tuple(_finite(v, 'force') for v in self.targets_N)
        low = SUPPORTED_FORCE_RANGE_N[0]
        high = _finite(self.maximum_force_N, 'maximum force')
        if high <= low:
            raise ValueError('Maximum strap force must be positive')
        if any(not low <= v <= high for v in targets):
            raise ValueError(f'Supported strap force range is {low:g}--{high:g} N')
        holds = self.hold_seconds
        if not isinstance(holds, (tuple, list)):
            holds = (holds, holds)
        if len(holds) != 2:
            raise ValueError('Exactly two hold durations are required')
        holds = tuple(_finite(v, 'hold duration') for v in holds)
        if any(v <= 0 for v in holds):
            raise ValueError('Hold durations must be positive seconds')
        dt = _finite(self.physics_dt, 'physics timestep')
        timeout = _finite(self.timeout_seconds, 'timeout')
        if dt <= 0 or timeout <= sum(holds):
            raise ValueError('Positive timestep and timeout longer than both holds are required')
        if _finite(self.tolerance_N, 'tolerance') != .5:
            raise ValueError('The physical force tolerance remains 0.5 N')
        object.__setattr__(self, 'targets_N', targets)
        object.__setattr__(self, 'hold_seconds', holds)
        object.__setattr__(self, 'physics_dt', dt)
        object.__setattr__(self, 'timeout_seconds', timeout)
        object.__setattr__(self, 'maximum_force_N', high)

    @classmethod
    def from_parameters(cls, parameters, physics_dt=1/120):
        sequence = parameters.get('force_time_sequence')
        if sequence is None:
            targets = parameters.get('target_sequence_N', (12., 18.))
            holds = parameters.get('hold_seconds', 3.)
        else:
            if not isinstance(sequence, (tuple, list)) or len(sequence) != 4:
                raise ValueError('force_time_sequence must be [force_N, seconds, force_N, seconds]')
            targets = (sequence[0], sequence[2])
            holds = (sequence[1], sequence[3])
        durations = holds if isinstance(holds, (tuple, list)) else (holds, holds)
        configured_timeout = _finite(parameters.get('policy_timeout_seconds', 60.), 'timeout')
        if configured_timeout <= 0:
            raise ValueError('Policy timeout must be positive seconds')
        timeout = max(configured_timeout, sum(_finite(v, 'hold duration') for v in durations)+30.)
        return cls(tuple(targets), .5, holds, timeout, physics_dt,
                   parameters.get('overload_N', SUPPORTED_FORCE_RANGE_N[1]))

    def as_sequence(self):
        return [self.targets_N[0], self.hold_seconds[0], self.targets_N[1], self.hold_seconds[1]]

    def controller_reference(self,target_N):
        # At the damage boundary, center the controller inside the intersection
        # of the original acceptance band and the original safe force interval.
        # The requested target and independent physical scorer remain unchanged.
        return min(float(target_N),max(0.,self.maximum_force_N-self.tolerance_N/2))

    def instruction(self):
        f, t = self.targets_N, self.hold_seconds
        return (f'Pull the elastic strap to {f[0]:g} newtons for {t[0]:g} seconds, '
                f'then {f[1]:g} newtons for {t[1]:g} seconds; remain within '
                f'{self.tolerance_N:g} newtons of each target.')


class StrapLifecycle:
    def __init__(self, start_step, limits=None):
        self.limits = limits or StrapAcceptanceLimits()
        self.start_step = int(start_step)
        self.last_step = self.start_step
        self.stage_index = 0
        self.stable_since = None
        self.completed_steps = []
        self.failure = ''
        self.success = False

    def advance(self, step, tension_N, *, grasped, damage=''):
        step = int(step)
        if step < self.last_step:
            raise ValueError('Physical time moved backwards')
        if step == self.last_step:
            return
        consecutive = step == self.last_step+1
        self.last_step = step
        if self.failure:
            return
        if damage:
            self.failure = str(damage)
            self.success = False
            return
        if not math.isfinite(float(tension_N)):
            self.failure = 'invalid_tension'
            return
        if self.success:
            return
        if not consecutive:
            self.stable_since = None
        target = self.limits.targets_N[self.stage_index]
        if grasped and abs(float(tension_N)-target) <= self.limits.tolerance_N:
            if self.stable_since is None:
                self.stable_since = step
            duration = self.limits.hold_seconds[self.stage_index]
            if (step-self.stable_since)*self.limits.physics_dt >= duration-1e-12:
                self.completed_steps.append(step)
                self.stage_index += 1
                self.stable_since = None
                self.success = self.stage_index == 2
        else:
            self.stable_since = None
        if not self.success and (step-self.start_step)*self.limits.physics_dt >= self.limits.timeout_seconds-1e-12:
            self.failure = 'timeout'

    def snapshot(self):
        return dict(contract='strap_two_stage_v2',
            target_sequence_N=list(self.limits.targets_N),
            force_time_sequence=self.limits.as_sequence(),
            supported_force_range_N=[0., self.limits.maximum_force_N],
            tolerance_N=self.limits.tolerance_N,
            hold_seconds_per_stage=list(self.limits.hold_seconds),
            hold_durations_s=list(self.limits.hold_seconds),
            timeout_seconds=self.limits.timeout_seconds,
            physics_dt_s=self.limits.physics_dt,
            stage_index=self.stage_index, completed_steps=self.completed_steps.copy(),
            current_continuous_hold_seconds=(0. if self.stable_since is None else
                (self.last_step-self.stable_since)*self.limits.physics_dt),
            policy_elapsed_seconds=(self.last_step-self.start_step)*self.limits.physics_dt,
            success=self.success, failure=self.failure)
