"""Shared note prompts and time-aligned dynamics labels for the two piano tasks."""
from dataclasses import dataclass
import json
import math
import re
import numpy as np

PITCHES=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
DYNAMICS={'idle':0,'light':1,'heavy':2}


def parse_notes(value):
    """Accept MIDI integers or scientific pitches; omitted octaves mean octave 4."""
    if value is None: return None
    tokens=re.split(r'[,，\s]+',value.strip()) if isinstance(value,str) else list(value)
    if not tokens or any(str(v).strip()=='' for v in tokens):
        raise ValueError('The piano sequence must contain at least one note')
    result=[]
    for token in tokens:
        text=str(token).strip()
        if re.fullmatch(r'\d+',text) and not isinstance(token,bool):
            midi=int(text)
        else:
            match=re.fullmatch(r'([A-Ga-g])([#b♯♭]?)(-?\d+)?',text)
            if match is None: raise ValueError(f'Invalid piano note: {token!r}')
            letter,accidental,octave=match.groups()
            midi=12*(int(octave) + 1 if octave is not None else 5)+PITCHES.index(letter.upper())
            midi+=1 if accidental in ('#','♯') else -1 if accidental in ('b','♭') else 0
        if not 21<=midi<=108: raise ValueError(f'Note outside the 88-key piano: {token!r}')
        result.append(midi)
    return tuple(result)


def note_name(midi):
    return f'{PITCHES[int(midi)%12]}{int(midi)//12-1}'


@dataclass(frozen=True)
class PianoSequence:
    midi: tuple
    dynamics: tuple
    holds: tuple

    def as_dict(self):
        return dict(midi=list(self.midi),notes=[note_name(m) for m in self.midi],
                    dynamics=list(self.dynamics),hold_seconds=list(self.holds))

    def instruction(self):
        notes=', '.join(f'{note_name(m)} ({d})' for m,d in zip(self.midi,self.dynamics))
        return f'Play and release these piano notes in order: {notes}.'


def make_sequence(notes=None, *, count=6, dynamics=None, holds=None, rng=None, pool=tuple(range(60,72))):
    rng=np.random.default_rng() if rng is None else rng
    midi=parse_notes(notes)
    if midi is None:
        if isinstance(count,bool) or int(count)!=count or not 1<=count<=128:
            raise ValueError('Random piano note count must be an integer from 1 to 128')
        midi=tuple(int(v) for v in rng.choice(pool,size=int(count)))
    if dynamics is None:
        dynamics=tuple(str(v) for v in rng.choice(('light','heavy'),size=len(midi)))
    else:
        if isinstance(dynamics,str): dynamics=re.split(r'[,，\s]+',dynamics.strip())
        dynamics=tuple(str(v).lower() for v in dynamics)
        if len(dynamics)==1: dynamics=dynamics*len(midi)
        if len(dynamics)!=len(midi) or any(v not in ('light','heavy') for v in dynamics):
            raise ValueError('Provide light/heavy once or once per piano note')
    if holds is None: holds=(.12,)*len(midi)
    elif np.isscalar(holds): holds=(float(holds),)*len(midi)
    holds=tuple(float(v) for v in holds)
    if len(holds)!=len(midi) or any(not math.isfinite(v) or v<=0 for v in holds):
        raise ValueError('Provide one positive hold duration per piano note')
    return PianoSequence(midi,dynamics,holds)


def dynamics_profile(sequence, *, embodiment):
    """Task force bands in N, not MIDI velocity or a hardware calibration."""
    light,heavy,tolerance=(1.5,3.,.45) if embodiment=='tool' else (.35,.50,.07)
    scale=np.ones(len(sequence.midi))
    if embodiment=='hand':
        # The shorter black key needs roughly twice the normal force for
        # the same return-spring torque. Keep this explicit in the labels.
        scale=np.array([2. if m%12 in (1,3,6,8,10) else 1. for m in sequence.midi])
    targets=np.array([light if d=='light' else heavy for d in sequence.dynamics])*scale
    return targets,scale*(light+heavy)/2,scale*tolerance


class PianoAnnotations:
    """One record per actual physics step; no-contact is distinct from light."""
    def __init__(self,sequence,*,embodiment,record_key_forces=False):
        self.sequence=sequence
        self.targets,self.thresholds,self.tolerances=dynamics_profile(sequence,embodiment=embodiment)
        self.embodiment=embodiment
        self.records=[]
        # Full force vectors are privileged offline labels, never observations.
        self.key_forces_records=[] if record_key_forces else None

    def update(self,step,dt,index,forces):
        if self.records and self.records[-1]['physics_step']==step: return self.records[-1]
        values=np.asarray(forces,float)
        if values.shape!=(88,) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Piano labels require 88 finite, nonnegative measured forces')
        i=min(index,len(self.sequence.midi)-1)
        actual=int(values.argmax())
        force=float(values[actual])
        contact=force>=.1
        # Classify the key actually touched, even if it is a wrong note.
        if self.embodiment=='tool': threshold=2.25
        else: threshold=.85 if (actual+21)%12 in (1,3,6,8,10) else .425
        complete=index>=len(self.sequence.midi)
        row=dict(physics_step=int(step),time_s=float(step*dt),target_index=int(index),
            target_midi=-1 if complete else int(self.sequence.midi[i]),
            target_dynamic=0 if complete else DYNAMICS[self.sequence.dynamics[i]],
            target_force_N=0. if complete else float(self.targets[i]),
            actual_midi=actual+21 if contact else -1,
            actual_dynamic=(2 if force>=threshold else 1) if contact else 0,
            actual_force_N=force,heavy_threshold_N=float(threshold),contact=bool(contact))
        self.records.append(row)
        if self.key_forces_records is not None:
            self.key_forces_records.append(values.copy())
        return row

    def write_jsonl(self,path):
        with open(path,'w',encoding='utf-8') as stream:
            for i,row in enumerate(self.records):
                if self.key_forces_records is not None:
                    key=row['target_midi']-21 if row['target_midi']>=21 else -1
                    values=self.key_forces_records[i]
                    row=dict(row,key_forces_N=values.tolist(),target_key_index=key,
                             measured_target_force_N=float(values[key]) if key>=0 else 0.)
                stream.write(json.dumps(row)+'\n')


def configure_piano_task(cfg,config):
    """Piano-only YAML overrides, used by existing collection/evaluation entries."""
    if not hasattr(cfg,'piano_notes'): return
    for name in ('piano_notes','piano_dynamics','piano_note_count','piano_require_dynamics',
                 'piano_wrong_key_policy','piano_hold_seconds'):
        if name in config: setattr(cfg,name,config[name])


def draw_piano_label(frame,row):
    """Observer overlay; callers keep the original tactile RGB pixels intact."""
    import cv2
    result=np.array(frame,copy=True)
    names={0:'idle',1:'light',2:'heavy'}
    target=note_name(row['target_midi']) if row['target_midi']>=0 else 'done'
    actual=note_name(row['actual_midi']) if row['actual_midi']>=0 else '-'
    text=(f"step {row['physics_step']} | {row['time_s']:.2f}s | goal {target} {names[row['target_dynamic']]}"
          f" | actual {actual} {names[row['actual_dynamic']]} {row['actual_force_N']:.2f}N")
    cv2.rectangle(result,(0,0),(min(result.shape[1],1100),33),(14,18,24),-1)
    cv2.putText(result,text,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,240,245),1,cv2.LINE_AA)
    return result
