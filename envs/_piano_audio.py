"""Contact-triggered synthesized piano audio for recorded simulation videos.

This is a presentation layer, not a string/acoustic physics model. It never
reads the intended melody or the scorer: every sounding key comes from contact.
"""
from pathlib import Path
import subprocess
import wave

import numpy as np


class PianoContacts:
    def __init__(self):
        self.active = {}
        self.events = []

    def advance(self, step, forces):
        forces = np.asarray(forces, float)
        if forces.shape != (88,) or not np.isfinite(forces).all():
            raise ValueError('Audio requires 88 finite physical key forces')
        for key, force in enumerate(forces):
            event = self.active.get(key)
            if event is None and force >= .3:
                self.active[key] = dict(midi=key+21, start_step=int(step),
                                        peak_force_N=float(force))
            elif event is not None:
                event['peak_force_N'] = max(event['peak_force_N'], float(force))
                if force < .1:
                    event['end_step'] = int(step)
                    self.events.append(self.active.pop(key))

    def finish(self, step):
        for event in self.active.values():
            self.events.append(dict(event, end_step=int(step)))
        self.active.clear()


def synthesize(events, video_steps, *, fps=60., sample_rate=48000):
    """Map physical contacts onto actual recorded frames; preserve gaps/holds."""
    steps = np.asarray(video_steps, float)
    if (steps.ndim != 1 or not len(steps) or not np.isfinite(steps).all()
            or np.any(np.diff(steps) < 0) or not np.isfinite(fps) or fps <= 0
            or sample_rate < 8000):
        raise ValueError('Invalid video/audio clock')
    unique, indices = np.unique(steps, return_index=True)
    duration = len(steps)/fps
    samples = np.zeros(int(np.ceil(duration*sample_rate)), dtype=np.float64)
    mapped = []
    for event in events:
        midi, force = int(event['midi']), float(event['peak_force_N'])
        begin, end = float(event['start_step']), float(event['end_step'])
        if not 21 <= midi <= 108 or not np.isfinite([force, begin, end]).all() or force < 0 or end < begin:
            raise ValueError('Invalid contact event')
        if end < steps[0] or begin > steps[-1]:
            continue
        on, off = np.interp([begin,end], unique, indices/fps)
        off = max(on+1/sample_rate, off)
        start = int(round(on*sample_rate))
        stop = min(len(samples), int(np.ceil((off+.35)*sample_rate)))
        t = np.arange(stop-start)/sample_rate
        frequency = 440.*2.**((midi-69)/12.)
        # Decaying string partials, slight detuning, and a soft hammer attack.
        # Louder contacts brighten the upper partials and increase amplitude.
        strength = float(np.clip(force/3., .02, 1.7))
        tone = np.zeros_like(t)
        for partial in range(1,9):
            f = frequency*partial*(1.+.000035*partial**2)
            if f >= sample_rate*.48:
                break
            gain = partial**(-1.7+.25*min(strength,1.))
            decay = np.exp(-t/(2.8/(1.+.32*(partial-1))))
            tone += gain*decay*(np.sin(2*np.pi*f*t)+.12*np.sin(2*np.pi*f*1.0007*t))/1.12
        envelope = (1.-np.exp(-t/.003))*np.exp(-np.maximum(t-(off-on),0.)/.065)
        samples[start:stop] += .18*strength**.7*tone*envelope
        mapped.append(dict(event, onset_seconds=float(on), release_seconds=float(off),
                           frequency_hz=float(frequency)))
    # A fixed soft limiter handles accidental chords without normalizing each
    # note independently (which would erase the intended force-volume contrast).
    return np.tanh(samples).astype(np.float32), mapped


def add_piano_audio(video_path, events, video_steps, *, fps=60.):
    video_path = Path(video_path)
    sample_rate = 48000
    samples, mapped = synthesize(events, video_steps, fps=fps, sample_rate=sample_rate)
    wav_path = video_path.with_suffix('.wav')
    with wave.open(str(wav_path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.rint(samples*32767).astype('<i2').tobytes())
    muxed = video_path.with_name(video_path.stem+'.with-audio.mp4')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(video_path),'-i',str(wav_path),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
        '-shortest','-movflags','+faststart',str(muxed)], check=True)
    muxed.replace(video_path)
    return dict(source='physical key contact onset/release and peak force',
                synthesis='decaying detuned partials; not acoustic physics',
                sample_rate_hz=sample_rate, duration_seconds=len(samples)/sample_rate,
                video_fps=float(fps), video_frames=len(video_steps), events=mapped,
                wav_path=str(wav_path))
