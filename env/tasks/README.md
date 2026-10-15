# Single-finger piano

`single_finger_piano.Task` fixes an Allegro hand's palm and every joint except
`index_joint_1`. All other finger joints are welded. One scalar action
bends/extends the index finger over C4.
The 88-key keyboard has physical hinges, return springs and contact sensors.
Success requires five presses with at least 80 ms of key depression/contact,
each followed by at least 100 ms of release. Holding a key does not count as
multiple presses. Pressing another key fails the episode.

From the repository root, use the Python environment provided by Isaac Lab
(validated with Isaac Sim 5.1 and Isaac Lab 2.3.2):

```bash
python -m env.tasks.run_single_finger_piano --headless --video
```

With a standalone Isaac Sim installation, replace `python` with its `python.sh`.
Omit `--headless` to open the viewer. Results are written to
`data/single_finger_piano/result.json`; `--video` also writes `video.mp4` and
`video.wav`. Audio reuses the existing piano synthesizer and follows measured
contacts, including unintended keys. It is synthesized audio, not a string model.
The default hand asset downloads from NVIDIA's asset server on first launch;
`--hand-usd /path/to/allegro_hand_instanceable.usd` selects a local copy.

The Gymnasium interface is `reset()` and `step(action)`. The action has shape
`(1,)`: -1 releases, +1 flexes the index joint. Observations contain the joint
position/velocity, 88 measured key angles, completed-cycle fraction and whether
a press is waiting for release. Reward is one per completed press/release cycle.
The episode ends on success, a wrong key, fixed-hand motion, or the 15 s timeout.

This task has its own launcher because the existing `envs/` collector and
TacEx tactile observations assume a Franka arm with a parallel gripper. It does
not export GelSight/Xense/Neote fingertip images or use that collector's action
format. Existing tasks and collection commands retain their current interfaces.

```bash
# Physics controls: idle must score zero; holding must score at most one cycle.
python -m env.tasks.run_single_finger_piano --headless --control idle --output data/single_finger_piano/idle
python -m env.tasks.run_single_finger_piano --headless --control hold --output data/single_finger_piano/hold
python -m pytest -q tests/test_single_finger_piano.py tests/test_piano_audio.py
```
