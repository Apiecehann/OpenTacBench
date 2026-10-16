"""Simulation-independent geometry and physical piano scoring."""
from dataclasses import dataclass
import math

import numpy as np


def box_mesh(size, cells=(2, 2, 2)):
    """Positive-volume tetrahedra; origin at the middle of the bottom face."""
    axes = [np.linspace(-size[i] / 2, size[i] / 2, cells[i] + 1) for i in range(2)]
    axes.append(np.linspace(0, size[2], cells[2] + 1))
    points = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    ids = np.arange(len(points)).reshape(*(n + 1 for n in cells))
    tets = []
    for i, j, k in np.ndindex(*cells):
        a, b, c, d = ids[i,j,k], ids[i+1,j,k], ids[i,j+1,k], ids[i+1,j+1,k]
        e, f, g, h = ids[i,j,k+1], ids[i+1,j,k+1], ids[i,j+1,k+1], ids[i+1,j+1,k+1]
        tets.extend([(a,b,d,h),(a,d,c,h),(a,c,g,h),(a,g,e,h),(a,e,f,h),(a,f,b,h)])
    tets = np.asarray(tets, dtype=np.int32)
    det = np.linalg.det(np.stack([points[tets[:, i]] - points[tets[:, 0]] for i in (1,2,3)], axis=-1))
    negative = det < 0
    tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
    # libuipc's native tetmesh binding expects row-major point storage.
    return np.ascontiguousarray(points), tets


@dataclass(frozen=True)
class Note:
    force_N: float
    hold_s: float
    tolerance_N: float = .45
    release_slack_s: float = .18


SHORT_NOTE = Note(1.5, .20)
LONG_NOTE = Note(3.0, .60)


class PianoScore:
    """Score physical key loads at every physics tick, including release."""
    def __init__(self, dt=1/120, *, key_sequence=(0,1), key_count=2,
                 notes=None,require_dynamics=True,wrong_key_policy='fail'):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid piano clock")
        self.key_sequence = tuple(key_sequence)
        self.key_count = int(key_count)
        if not self.key_sequence or any(not 0 <= k < key_count for k in self.key_sequence):
            raise ValueError('invalid piano key sequence')
        self.notes = tuple(notes) if notes is not None else tuple((SHORT_NOTE,LONG_NOTE)[i%2] for i in range(len(self.key_sequence)))
        if len(self.notes)!=len(self.key_sequence) or wrong_key_policy not in ('fail','retry'):
            raise ValueError('Invalid piano note specifications or wrong-key policy')
        self.require_dynamics=require_dynamics
        self.wrong_key_policy=wrong_key_policy
        self.mistakes=0
        self._wrong_contact=False
        self.peak_force=0.
        self.dt = dt
        self.index = 0
        self.stable_steps = 0
        self.hold_steps = 0
        self.over_steps = 0
        self.pressed = False
        self.failure = ""
        self.events = []

    @property
    def success(self):
        return self.index == len(self.notes) and not self.failure

    def advance(self, forces):
        values = np.asarray(forces, dtype=float)
        if values.shape != (self.key_count,) or not np.isfinite(values).all():
            self.failure = "invalid_key_force"
            return
        if self.failure or self.success:
            return
        if values.max() > 6.0:
            self.failure = "key_overload"
            return
        if values.max() < .3 and not self.pressed:
            self._wrong_contact=False
            return
        target = self.key_sequence[self.index]
        if np.any(np.delete(values,target) > (.6 if self.require_dynamics else .3)):
            if not self._wrong_contact: self.mistakes+=1
            self._wrong_contact=True
            if self.wrong_key_policy=='fail': self.failure = "wrong_key"
            self.stable_steps=self.hold_steps=self.over_steps=0
            self.pressed=False
            self.peak_force=0.
            return
        self._wrong_contact=False
        note = self.notes[self.index]
        force = values[target]
        # Contact onset and release need separate thresholds. A small initial
        # impact can settle from 0.38 N to 0.19 N without losing contact; using
        # 0.3 N for both edges incorrectly scored that settling as a release.
        # This does not contribute any time to the required note-force band.
        if force >= .3 or (self.pressed and force >= .1):
            self.pressed = True
            self.peak_force=max(self.peak_force,float(force))
            self.over_steps = self.over_steps + 1 if force > note.force_N + note.tolerance_N else 0
            if self.require_dynamics and self.over_steps * self.dt > .05:
                self.failure = "wrong_note_force"
            if (abs(force-note.force_N) <= note.tolerance_N if self.require_dynamics else force>=.3):
                self.stable_steps += 1
            elif self.hold_steps == 0:
                self.stable_steps = 0
            if self.stable_steps * self.dt >= note.hold_s - 1e-9:
                self.hold_steps += 1
                if self.require_dynamics and self.hold_steps * self.dt > note.release_slack_s:
                    self.failure = "note_too_long"
        elif self.pressed:
            if self.stable_steps * self.dt < note.hold_s - 1e-9:
                if self.wrong_key_policy=='fail': self.failure = "note_too_short"
                self.stable_steps=self.hold_steps=self.over_steps=0
                self.pressed=False
                self.peak_force=0.
                return
            self.events.append(dict(key=target, force_N=self.peak_force,target_force_N=note.force_N,
                                    stable_seconds=self.stable_steps*self.dt))
            self.index += 1
            self.stable_steps = self.hold_steps = self.over_steps = 0
            self.pressed = False
            self.peak_force=0.
