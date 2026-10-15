# Single-finger piano: Shadow Hand and Twinkle Twinkle Little Star

The default task is `shadow_piano.Task`: a five-finger Shadow Hand plays the
complete 42-note melody of **Twinkle Twinkle Little Star**, in C major. Only the
index finger flexes. All other finger and wrist joints are welded; two driven
prismatic joints lift and translate the hand between keys. The hand follows
continuous PhysX motion. The task never teleports the root to the next key.

Run from the repository root in the Isaac Lab Python environment:

```bash
python -m env.tasks.run_single_finger_piano --headless --video
```

Use the standalone Isaac Sim installation's `python.sh` in place of `python`
when appropriate. The implementation targets Isaac Sim 5.1 / Isaac Lab 2.3.2.
Omit `--headless` to open the viewer. The default outputs are
`data/shadow_piano/video.mp4`, `video.wav` and `result.json`.
`--notes 3 --output data/shadow_piano/smoke` plays only C4 C4 G4 for a smoke run;
omit `--notes` for the entire song.

The Shadow task's Gymnasium action is `(rail_y_m, lift_z_m, index_flexion_rad)`.
Observations contain the three joint positions and velocities, all 88 key
angles, completed-note fraction, press/release state and next pitch. Each note
must physically depress its key with measured contact force and then release
it. Wrong keys fail the episode; repeated pitches require separate releases.
Phrase-ending notes require longer holds. Sound comes from measured key
contacts and uses the shared piano synthesizer; no melody audio is prerecorded.

The hand uses a dark forearm, satin metal finger links, a pale palm shell and
dark fingertip pads. Its articulated/collision geometry comes from Isaac Lab's
Shadow Hand asset. Materials change the appearance; motion and scoring use the
same physical model. The asset downloads from NVIDIA on first launch.
`--hand-usd` may select a local copy of the same hand model.

## Original fixed-palm task

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
python -m env.tasks.run_single_finger_piano --song repeated --headless --video
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
python -m env.tasks.run_single_finger_piano --song repeated --headless --control idle --output data/single_finger_piano/idle
python -m env.tasks.run_single_finger_piano --song repeated --headless --control hold --output data/single_finger_piano/hold
python -m pytest -q tests/test_single_finger_piano.py tests/test_piano_audio.py
```
