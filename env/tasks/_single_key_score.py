"""Simulation-independent scoring of physical key press/release sequences."""
from dataclasses import dataclass, field
import math


@dataclass
class SingleKeyScore:
    target: int = 39  # MIDI C4 minus A0
    repetitions: int = 5
    press_angle: float = .025  # radians of downward key rotation
    release_angle: float = .008
    hold_seconds: float = .08
    release_seconds: float = .10
    key_sequence: tuple = ()
    note_hold_seconds: tuple = ()
    force_targets: tuple = ()
    force_tolerances: tuple = ()
    require_dynamics: bool = False
    wrong_key_policy: str = 'fail'
    mistakes: int = 0
    count: int = 0
    failure: str = ""
    events: list = field(default_factory=list)
    _held: float = 0.
    _released: float = 0.
    _armed: bool = False
    _wrong_contact: bool = False
    _overheld: float = 0.

    def __post_init__(self):
        if self.key_sequence:
            self.repetitions = len(self.key_sequence)
        else:
            self.key_sequence = (self.target,)*self.repetitions
        if (not 0 <= self.target < 88 or self.repetitions < 1
                or not 0 <= self.release_angle < self.press_angle
                or self.hold_seconds <= 0 or self.release_seconds <= 0
                or any(not isinstance(k,int) or not 0 <= k < 88 for k in self.key_sequence)
                or (self.note_hold_seconds and (len(self.note_hold_seconds) != self.repetitions
                    or any(not math.isfinite(t) or t <= 0 for t in self.note_hold_seconds)))):
            raise ValueError('Invalid repeated-key scoring configuration')
        if self.wrong_key_policy not in ('fail','retry'):
            raise ValueError('wrong_key_policy must be fail or retry')
        if self.require_dynamics and (len(self.force_targets)!=self.repetitions
                or len(self.force_tolerances)!=self.repetitions
                or any(not math.isfinite(v) or v<=0 for v in (*self.force_targets,*self.force_tolerances))):
            raise ValueError('Dynamics scoring needs one positive force target and tolerance per note')

    @property
    def success(self):
        return self.count >= self.repetitions and not self.failure

    @property
    def active_key(self):
        return self.key_sequence[min(self.count,self.repetitions-1)]

    def advance(self, angles, forces, dt, step):
        if (len(angles) != 88 or len(forces) != 88 or not math.isfinite(dt) or dt <= 0
                or not all(math.isfinite(x) for x in (*angles, *forces))):
            raise ValueError('Scoring requires 88 finite key angles/forces and a positive timestep')
        if self.failure or self.success:
            return
        key = self.active_key
        if any(angle >= self.press_angle for i, angle in enumerate(angles) if i != key):
            if not self._wrong_contact: self.mistakes+=1
            self._wrong_contact=True
            if self.wrong_key_policy=='fail': self.failure = 'wrong_key'
            self._held=self._released=0.
            self._armed=False
            self._overheld=0.
            return
        self._wrong_contact=False
        down = angles[key] >= self.press_angle and forces[key] >= .1
        if self.require_dynamics:
            target,tolerance=self.force_targets[self.count],self.force_tolerances[self.count]
            self._overheld=self._overheld+dt if forces[key]>target+tolerance else 0.
            if self._overheld>.05:
                self.failure='wrong_note_force'
                return
            down=down and abs(forces[key]-target)<=tolerance
        up = angles[key] <= self.release_angle
        if not self._armed:
            self._held = self._held + dt if down else 0.
            hold = self.note_hold_seconds[self.count] if self.note_hold_seconds else self.hold_seconds
            if self._held + 1e-10 >= hold:
                self._armed = True
                self._released = 0.
                self.events.append(dict(kind='press', step=step, midi=key+21, force_N=float(forces[key])))
        elif up:
            self._released += dt
            if self._released + 1e-10 >= self.release_seconds:
                self.count += 1
                self.events.append(dict(kind='release', step=step, midi=key+21, repetition=self.count))
                self._armed = False
                self._held = self._released = 0.
                self._overheld=0.
        else:
            self._released = 0.
