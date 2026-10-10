"""Simulation-independent contracts for the two tactile recall tasks."""
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


def cue_mesh(kind):
    """Equal envelopes, with one or two raised tactile ribs on both jaw faces."""
    if kind not in (0, 1):
        raise ValueError("cue kind must be 0 or 1")
    points, tets = box_mesh((.030, .030, .050), (2, 2, 20))
    z = points[:, 2]
    centers = (.027,) if kind == 0 else (.021, .033)
    relief = sum(.0015 * np.maximum(0., 1 - np.abs(z-c) / .004) for c in centers)
    for axis in (0, 1):
        points[:, axis] += np.sign(points[:, axis]) * relief
    return points, tets


@dataclass
class CueMemory:
    """The cue must be touched, then released before an answer can count."""
    touch_steps: int = 12
    gap_steps: int = 12
    touched: int = 0
    released: int = 0
    acquired: bool = False

    def advance(self, contact):
        if not self.acquired:
            self.touched = self.touched + 1 if contact else 0
            self.acquired = self.touched >= self.touch_steps
        if self.acquired:
            self.released = 0 if contact else self.released + 1

    @property
    def ready(self):
        return self.acquired and self.released >= self.gap_steps


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
    def __init__(self, cue_kind, dt=1/120, *, key_sequence=(0,1), key_count=2):
        if cue_kind not in (0, 1) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid piano cue or clock")
        self.key_sequence = tuple(key_sequence)
        self.key_count = int(key_count)
        if not self.key_sequence or any(not 0 <= k < key_count for k in self.key_sequence):
            raise ValueError('invalid piano key sequence')
        self.notes = tuple((SHORT_NOTE,LONG_NOTE)[(i+cue_kind)%2] for i in range(len(self.key_sequence)))
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

    def advance(self, forces, *, memory_ready):
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
            return
        if not memory_ready:
            self.failure = "cue_not_remembered"
            return
        target = self.key_sequence[self.index]
        if np.any(np.delete(values,target) > .6):
            self.failure = "wrong_key"
            return
        note = self.notes[self.index]
        force = values[target]
        # Contact onset and release need separate thresholds. A small initial
        # impact can settle from 0.38 N to 0.19 N without losing contact; using
        # 0.3 N for both edges incorrectly scored that settling as a release.
        # This does not contribute any time to the required note-force band.
        if force >= .3 or (self.pressed and force >= .1):
            self.pressed = True
            self.over_steps = self.over_steps + 1 if force > note.force_N + note.tolerance_N else 0
            if self.over_steps * self.dt > .05:
                self.failure = "wrong_note_force"
            if abs(force-note.force_N) <= note.tolerance_N:
                self.stable_steps += 1
            elif self.hold_steps == 0:
                self.stable_steps = 0
            if self.stable_steps * self.dt >= note.hold_s - 1e-9:
                self.hold_steps += 1
                if self.hold_steps * self.dt > note.release_slack_s:
                    self.failure = "note_too_long"
        elif self.pressed:
            if self.stable_steps * self.dt < note.hold_s - 1e-9:
                self.failure = "note_too_short"
                return
            self.events.append(dict(key=target, force_N=note.force_N,
                                    stable_seconds=self.stable_steps*self.dt))
            self.index += 1
            self.stable_steps = self.hold_steps = self.over_steps = 0
            self.pressed = False
