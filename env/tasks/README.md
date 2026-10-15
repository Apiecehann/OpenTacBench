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

## Index-only calibrated GelSight Mini

The optional `--tactile` backend mounts the repository's original, unscaled
GelSight Mini assembly on the Shadow index distal link. It calls
`envs.sensors.tactile.create_gelsight_mini_cfg` and `VisualTactileSensor`
directly. Camera optics, gel tetrahedra, attachments, FEM marker motion and
Taxim calibration come from that existing implementation. The task changes
only the assembly's rigid mounting pose and keyboard mechanics.

For the standalone Isaac Sim 5.1 / Isaac Lab 2.3.2 environment, install build
tools (`cmake >= 3.26`, `ninja-build`, `build-essential`, `autoconf`, `automake`,
`libtool`, `pkg-config`, `git`, `curl`, `zip`, `unzip`) and a CUDA 12.8 toolkit.
From the repository root:

```bash
bash scripts/install_piano_tactile.sh
bash scripts/run_piano_tactile.sh --headless --video
```

`ISAAC_SIM_ROOT` selects the standalone Isaac installation (default `/isaac-sim`).
The installer creates a separate environment under `.cache/piano_tactile` and
links it as `.venv-tactile`; `PIANO_TACTILE_ROOT` selects another build/data
location. It preserves Isaac's Python/PyTorch installation and installs the
matching Python 3.11 / PyTorch 2.7 CUDA 12.8 scatter wheel. Building libuipc
requires more time and disk space than installing the Python wrappers alone.
For restricted networks, optional `PIANO_GITHUB_BASE` and
`PIANO_GITHUB_DOWNLOAD_PREFIX` select Git and archive transports respectively;
archive checksums remain enforced.

The tactile task keeps 240 Hz physics and 60 Hz control/recording. The backing
follows the index distal link. Gel and reachable piano keys are solved together
in UIPC, using affine rigid keys, rear revolute hinges, 1 N m/rad return springs,
0.025 N m s/rad damping and mechanical travel stops. The visible Isaac keyboard
mirrors measured UIPC hinge angles; PhysX drives the hand. Contact energy
gradients include `dt²`, so forces in newtons are `-gradient / dt²`. Scoring still
requires real key depression, sustained contact and a separate release.

In this variant the rail Y bounds cover the selected melody plus 8 mm at each
end. The finite-width pad can also reach adjacent wrong notes, which are
included in the solver and rejected by scoring. The fixed wrist X position
places the fingertip ahead of the black-key region, so its contact domain is
the front white keys. Other keys retain their visual positions. The sensor
retains the original gel dimensions, material settings and
calibrated pixel scale. The backing follows the index through the original
UIPC attachment callback; reaction forces do not feed back into PhysX motors.
Position commands are limited to 1 m/s lateral travel, 0.375 m/s lift and
8 rad/s index flexion to smooth contact transitions;
`TaskCfg.tactile_action_rates` configures these limits.
A 40 ms command filter also smooths abrupt changes; configure it with
`TaskCfg.tactile_command_filter_seconds`.
The tactile demonstrator uses 1.2 s note spacing (2.4 s at phrase endings) so
filtered presses meet the same physical hold/release requirements.

The installer uses the repository's modified source. Its headless compatibility
fixes make UI registration optional, allow reuse of the separately built native
library and disable an unfinished shell registration that conflicts with the
revolute joint's model ID. The zlib 1.3.1 build is pinned to its upstream Git
commit because its old archive endpoint is unavailable on some networks.

`reset()` and `step()` return `{"state": ..., "tactile": {"index": ...}}` when
enabled. `state` retains the usual 97 values. The index dictionary contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `rgb` | 480 × 640 × 3, uint8 | GelSight Mini Taxim RGB with FEM-advected markers |

The complete original camera frame is returned without cropping or overlays.
Surface depth and displacement remain internal to the FEM
and optical simulation; tactile observations and recordings expose only RGB.
The default output directory is `data/shadow_piano_calibrated`: `tactile.h5`
stores synchronized timestamps, state and `tactile/index/rgb`; `--video`
records `video.mp4` with the scene, complete RGB image and contact audio, plus
`gelsight.mp4` with just the untouched 640 × 480 tactile frames (silent).
The composite title sits outside the image, fixing the earlier title bar
that covered the top 27 rows of the displayed RGB. No depth or deformation
panels are included. The original non-tactile command retains its flat
observation format.

To check unloaded contact, a sustained press, release, reset, wrong-key
rejection and the index observation contract:

```bash
PYTHONEXE="$PWD/.venv-tactile/bin/python" /isaac-sim/python.sh scripts/check_piano_tactile.py
```

The original-API implementation passed all 42 press/release cycles and produced
42 contact-driven audio notes, with 3472 synchronized RGB/video frames over
57.86 simulated seconds. The controls verify idle, sustained hold, release,
physical reset and wrong-key rejection. Unmarked optical RGB changes under
contact and resets exactly; the original marker brightness noise remains enabled.
The earlier compact-pad recordings remain in `data/shadow_piano_rgb`; the
calibrated run is in `data/shadow_piano_calibrated/full`.

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
not use that collector's action format. The original Allegro variant has no
tactile images; the Shadow variant supports the optional index sensor above.
Existing tasks and collection commands retain their current interfaces.

```bash
# Physics controls: idle must score zero; holding must score at most one cycle.
python -m env.tasks.run_single_finger_piano --song repeated --headless --control idle --output data/single_finger_piano/idle
python -m env.tasks.run_single_finger_piano --song repeated --headless --control hold --output data/single_finger_piano/hold
python -m pytest -q tests/test_single_finger_piano.py tests/test_piano_audio.py
```
