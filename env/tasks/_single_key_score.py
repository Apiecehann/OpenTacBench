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
    continue_on_note_failure: bool = False
    mistakes: int = 0
    count: int = 0
    successful_notes: int = 0
    failure: str = ""
    events: list = field(default_factory=list)
    _held: float = 0.
    _released: float = 0.
    _armed: bool = False
    _wrong_contact: bool = False
    _overheld: float = 0.
    _pending_note_failure: str = ""

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
        return self.finished and self.successful_notes == self.repetitions and not self.failure

    @property
    def finished(self):
        return self.count >= self.repetitions

    @property
    def active_key(self):
        return self.key_sequence[min(self.count,self.repetitions-1)]

    def finish_attempt(self, index, step):
        """Consume one released note attempt without revealing its result to control."""
        if not self.continue_on_note_failure:
            raise RuntimeError('Per-note completion requires continue_on_note_failure')
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<self.repetitions:
            raise ValueError('Invalid piano attempt index')
        if index == self.count-1:
            return  # A successful release already advanced this attempt.
        if index != self.count or self.failure:
            raise RuntimeError('Piano attempts must finish in sequence')
        reason=self._pending_note_failure or 'incomplete_note'
        self.events.append(dict(kind='miss',step=step,midi=self.active_key+21,
                                attempt_index=index,reason=reason))
        self.count+=1
        self._held=self._released=self._overheld=0.
        self._armed=self._wrong_contact=False
        self._pending_note_failure=''
        self._finish_sequence()

    def _finish_sequence(self):
        if self.finished and self.successful_notes != self.repetitions:
            self.failure='partial_score'

    def advance(self, angles, forces, dt, step):
        if (len(angles) != 88 or len(forces) != 88 or not math.isfinite(dt) or dt <= 0
                or not all(math.isfinite(x) for x in (*angles, *forces))):
            raise ValueError('Scoring requires 88 finite key angles/forces and a positive timestep')
        if self.failure or self.finished or self._pending_note_failure:
            return
        key = self.active_key
        if any(angle >= self.press_angle for i, angle in enumerate(angles) if i != key):
            if not self._wrong_contact: self.mistakes+=1
            self._wrong_contact=True
            if self.wrong_key_policy=='fail':
                if self.continue_on_note_failure: self._pending_note_failure='wrong_key'
                else: self.failure = 'wrong_key'
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
                if self.continue_on_note_failure: self._pending_note_failure='wrong_note_force'
                else: self.failure='wrong_note_force'
                return
            down=down and abs(forces[key]-target)<=tolerance
        up = angles[key] <= self.release_angle
        if not self._armed:
            self._held = self._held + dt if down else 0.
            hold = self.note_hold_seconds[self.count] if self.note_hold_seconds else self.hold_seconds
            if self._held + 1e-10 >= hold:
                self._armed = True
                self._released = 0.
                event=dict(kind='press',step=step,midi=key+21,force_N=float(forces[key]))
                if self.continue_on_note_failure: event['attempt_index']=self.count
                self.events.append(event)
        elif up:
            self._released += dt
            if self._released + 1e-10 >= self.release_seconds:
                self.count += 1
                self.successful_notes += 1
                event=dict(kind='release',step=step,midi=key+21,repetition=self.count)
                if self.continue_on_note_failure: event['attempt_index']=self.count-1
                self.events.append(event)
                self._armed = False
                self._held = self._released = 0.
                self._overheld=0.
                self._finish_sequence()
        else:
            self._released = 0.
