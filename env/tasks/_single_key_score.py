"""Simulation-independent scoring of repeated physical key press/release cycles."""
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
    count: int = 0
    failure: str = ""
    events: list = field(default_factory=list)
    _held: float = 0.
    _released: float = 0.
    _armed: bool = False

    def __post_init__(self):
        if (not 0 <= self.target < 88 or self.repetitions < 1
                or not 0 <= self.release_angle < self.press_angle
                or self.hold_seconds <= 0 or self.release_seconds <= 0):
            raise ValueError('Invalid repeated-key scoring configuration')

    @property
    def success(self):
        return self.count >= self.repetitions and not self.failure

    def advance(self, angles, forces, dt, step):
        if (len(angles) != 88 or len(forces) != 88 or not math.isfinite(dt) or dt <= 0
                or not all(math.isfinite(x) for x in (*angles, *forces))):
            raise ValueError('Scoring requires 88 finite key angles/forces and a positive timestep')
        if self.failure:
            return
        if any(angle >= self.press_angle for i, angle in enumerate(angles) if i != self.target):
            self.failure = 'wrong_key'
            return
        if self.success:
            return
        down = angles[self.target] >= self.press_angle and forces[self.target] >= .1
        up = angles[self.target] <= self.release_angle
        if not self._armed:
            self._held = self._held + dt if down else 0.
            if self._held + 1e-10 >= self.hold_seconds:
                self._armed = True
                self._released = 0.
                self.events.append(dict(kind='press', step=step, force_N=float(forces[self.target])))
        elif up:
            self._released += dt
            if self._released + 1e-10 >= self.release_seconds:
                self.count += 1
                self.events.append(dict(kind='release', step=step, repetition=self.count))
                self._armed = False
                self._held = self._released = 0.
        else:
            self._released = 0.
