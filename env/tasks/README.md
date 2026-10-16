# Piano note sequences: tapping tool and dexterous hand

The default task is `arm_shadow_piano.Task`: a Franka Panda carries a five-finger
Shadow Hand and plays a supplied sequence, or six random C4–B4 notes when no
sequence is supplied. **Twinkle Twinkle Little Star** remains available with
`--song twinkle`. Only the
index finger flexes. All other hand joints are welded; seven driven Panda
joints lift and translate the wrist through Isaac Lab's differential IK
controller. The hand follows continuous PhysX motion. The task never teleports
the root to the next key.

Run from the repository root in the Isaac Lab Python environment:

```bash
python -m env.tasks.run_single_finger_piano --headless --video
```

Use the standalone Isaac Sim installation's `python.sh` in place of `python`
when appropriate. The implementation targets Isaac Sim 5.1 / Isaac Lab 2.3.2.
Omit `--headless` to open the viewer. The default outputs are
`data/arm_shadow_piano/video.mp4`, `video.wav` and `result.json`.
`--notes 3 --seed 0` requests three reproducible random notes.
`--song twinkle --notes 3` plays C4 C4 G4; `--song twinkle` plays all 42 notes.
`--wrist rail` selects the previous standalone Shadow Hand on Y/Z slides.

The arm action is `(wrist_x_offset_m, wrist_y_m, lift_z_m, index_flexion_rad)`.
The X offset is relative to the reset palm and reaches the black-key row.
The rail retains `(wrist_y_m, lift_z_m, index_flexion_rad)`.
The arm variant has 111 state values, including all seven arm joint positions
and velocities. The rail variant retains its 97 values. Both include wrist/finger
positions and velocities, all 88 key
angles, completed-note fraction, press/release state and next pitch. Each note
must physically depress its key with measured contact force and then release
it. Wrong keys fail by default; `--wrong-key-policy retry` permits another
attempt at the current note. Repeated pitches require separate releases.
The Twinkle preset retains longer phrase-ending holds. Sound comes from measured key
contacts and uses the shared piano synthesizer; no melody audio is prerecorded.
The hand connects to `panda_link7` by a fixed joint at the Panda flange. Initial
arm geometry and joint angles agree through Isaac Sim's Lula solver. Playing
uses motor targets; only reset writes joint state directly. The arm's optional
tactile backend defaults to DIGIT; the earlier GelSight Mini remains available.
The arm demo waits for the wrist to settle before bending the index, verifies
each physical press, releases the finger, then lifts and moves to the next key.
The index motor uses 6 N m/rad stiffness, 0.20 N m s/rad damping and the existing
0.9 N m effort limit, with the original 0.50 rad tactile command. Each sensor
retains its original dimensions and optical settings. Musical scoring uses the
same thresholds. IK increments are scaled together to preserve
their Cartesian direction when motor rates are limited.
The arm demonstrator uses measured force to adjust wrist height toward the
requested dynamic, bounded to 6 mm down and 4 mm up. Policies still command
wrist/finger targets directly; scoring uses the measured result.

The hand uses a dark forearm, satin metal finger links, a pale palm shell and
dark fingertip pads. Its articulated/collision geometry comes from Isaac Lab's
Shadow Hand asset. Materials change the appearance; motion and scoring use the
same physical model. The asset downloads from NVIDIA on first launch.
`--hand-usd` may select a local copy of the same hand model.

## Shared prompts, scoring and timestep labels

These additions apply to `envs/play_piano.py` (Franka and tapping tool) and
`env/tasks/arm_shadow_piano.py` (Franka and Shadow index).

```bash
# The user's example, with default pitch-order and force-band scoring:
bash scripts/run_piano_tactile.sh --headless --video --image-width 320 --sequence "C# C G A E F" --dynamics "light heavy light heavy light heavy"
# Explicitly disable force-band scoring, retaining per-tick dynamics labels:
bash scripts/run_piano_tactile.sh --headless --video --image-width 320 --sequence "C#4 C4 G4 A4 E4 F4" --dynamics "light heavy light heavy light heavy" --no-judge-dynamics
# No prompt: six random notes; a seed makes the prompt reproducible:
bash scripts/run_piano_tactile.sh --headless --video --image-width 320 --seed 0
```

Notes accept names with sharps/flats or MIDI integers. An omitted octave means
4, so `C#` is C#4. For the tool task set `piano_notes`, `piano_dynamics`,
`piano_note_count`, `piano_require_dynamics`, `piano_wrong_key_policy` and
`piano_hold_seconds` in
[`task_config/play_piano.yml`](../../task_config/play_piano.yml), then use the
existing collector/evaluator. The shared wrapper selects the gripper collector
with a leading `gripper` argument and accepts an optional YAML path:

```bash
bash scripts/run_piano_tactile.sh gripper --headless --episode_num 1
bash scripts/run_piano_tactile.sh gripper --config path/to/piano.yml --headless
```

The gripper also needs the benchmark's cuRobo motion-planning dependency
(see [installation](../../docs/Installation.md)). Its default YAML saves only
`rgb_marker` tactile images; depth remains internal to adaptive grasping.
All serial/parallel collection/evaluation entries
apply these piano-only settings. Both tasks also accept
`reset(options={"notes": ["C#4", "C4"], "dynamics": ["light", "heavy"]})`.

Default success requires pitches in order, physical contact in the requested
force band, a minimum hold and separate releases. Both task configs enable
dynamics judging by default. Use `--no-judge-dynamics` for the hand or
`piano_require_dynamics: false` for the gripper to disable force-band scoring;
target/actual dynamics remain annotated. `--judge-dynamics` still explicitly
enables it. Duration is independent of dynamics; `--hold-seconds`
or `piano_hold_seconds` changes it. The tool task additionally requires tool
acquisition and rejects physical overload. Its policy still submits END through
the existing interface.

Every physics step has `physics_step`, `time_s`, `target_index`, `target_midi`,
`target_dynamic`, `target_force_N`, `actual_midi`, `actual_dynamic`,
`actual_force_N`, `heavy_threshold_N` and `contact`. Codes are **0 idle,
1 light, 2 heavy**; no-contact pitch is -1. Targets refer to the note being
evaluated during that tick, including its release tick. Actual labels use
measured force on the key actually touched, including wrong keys.

The hand exposes labels in Gym `info["piano"]`, per-physics-step
`piano_steps.jsonl`, and a `piano/` group aligned with recorded `tactile.h5`
frames. The tool adds the same fields under observation `piano`, saves them in
normal HDF5, and writes `piano_steps/<seed>.jsonl` under its collection root.
Episode context stores the complete prompt. Observer videos display target and
actual labels; raw tactile RGB is preserved.
The tool task uses `decimation: 1` so each annotated step is one 120 Hz physics
tick. The hand records every 240 Hz physics tick and samples tactile frames at
its control rate (30 Hz for the arm default).

These are task force bands, not hardware calibration or MIDI velocity. The
tool targets 1.5 N / 3 N with ±0.45 N tolerance. The hand targets 0.35 N / 0.50 N
with ±0.07 N tolerance on white keys; black-key values are doubled for the
shorter spring lever. Actual light/heavy thresholds are 2.25 N for the tool,
0.425 N for a white key under the hand and 0.85 N for a black key.
Black-key hand sequences use the arm with DIGIT or without tactile sensing.
The legacy Mini/UIPC and rail variants retain white-key operation. The arm
rejects notes outside its lateral workspace before running.

## Integrated DIGIT index tip

The arm's `--tactile` mode uses the original Meta TACTO renderer and DIGIT
configuration. The scene keeps the Shadow Hand's distal shell and a dark
sensing surface; the external white sensor housing is omitted. This is an
integrated visual representation. The original DIGIT gel geometry, camera,
lighting, reference image and optical noise are retained without scaling.
The mounting angle faces the gel toward the key at the nominal pressed pose,
placing contact inside the original camera's field of view.

Install the Isaac tactile environment below first, then run from the repository root:

```bash
bash scripts/install_piano_digit.sh
bash scripts/run_piano_tactile.sh --headless --rendering_mode performance --image-width 320 --video
# Inspect the index tip up close:
bash scripts/run_piano_tactile.sh --headless --rendering_mode performance --image-width 320 --notes 3 --camera-view fingertip --video --output data/arm_shadow_piano_digit/preview
```

TACTO runs in its own pinned Python environment under
`.cache/piano_tactile/digit`, keeping its NumPy/OpenGL dependencies separate from
Isaac and TacEx. The installer verifies checksums for the original CAD and
reference image. The renderer is imported directly from the installed TACTO
distribution, without its unused PyBullet/URDF scene loader.

PhysX drives the keys and rigid sensing surface. Native filtered contact
forces require the sensing surface itself to touch each key; the index's
other links cannot press the keys. TACTO uses those real poses and forces
with its original force-to-appearance model. This backend does not simulate
FEM gel deformation. The Mini option below retains the original UIPC/FEM backend.

Outputs are `data/arm_shadow_piano_digit/video.mp4`, `digit.mp4`, `tactile.h5`
and `result.json`. The observation contract remains 111 state values plus
`tactile/index/rgb`, with one index sensor and no depth/deformation outputs.
`--camera-view fingertip` changes only the observer camera.

The six-note C#4 C4 G4 A4 E4 F4 demonstration passed with strict dynamics,
black/white key travel and six matching physical audio events: 5942 physics
ticks, 743 aligned tactile frames and 24.758 simulated seconds. The default
random seed-0 run also passed all six notes, including consecutive D#4 strokes:
A#4 G4 F#4 D#4 D#4 C4, 4829 ticks and 604 tactile frames. Every saved label was
checked against the full physics trace and measured-force classification.
Recordings are in `data/piano_sequences/hand_strict` and `hand_random`.
The gripper/tool run passed the same six-note prompt on seed 7: 1598 policy
physics ticks, 821 saved frames and six matching contact-driven audio events.
The standard episode validator decoded 4926 RGB frames and passed all timing,
terminal-verdict and measured-force-label checks. Its video, HDF5, metadata and
per-tick JSONL are in `data/piano_sequences/tool`.
The regular collector also passed seeds 8 and 9 in one process, including
reset and two separately released C4 strokes with strict light/heavy judging:
690/680 policy ticks and 368/362 saved frames. Both RGB-only episodes passed
the standard validator and complete JSONL/HDF5/audio alignment checks. These
outputs are in `data/piano_sequences/tool_repeat`.
The earlier Twinkle reference recording
remains in `data/arm_shadow_piano_digit/full`.

## Original index-only calibrated GelSight Mini

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
bash scripts/run_piano_tactile.sh --tactile-sensor gsmini --headless --video
# Use the original factory's standard 320 x 240 full-field resolution:
bash scripts/run_piano_tactile.sh --tactile-sensor gsmini --headless --video --image-width 320
# Validated faster rendering profile, with the same physics and sensor API:
bash scripts/run_piano_tactile.sh --tactile-sensor gsmini --headless --rendering_mode performance --video --image-width 320
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

Both piano supports keep 240 Hz physics. The arm uses 30 Hz control/recording;
the rail retains 60 Hz. Audio and RGB recording use the corresponding frame
clock, without changing playback speed. The backing
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
The default arm output directory is `data/arm_shadow_piano`; the rail tactile
variant uses `data/shadow_piano_calibrated`. In each, `tactile.h5`
stores synchronized timestamps, state and `tactile/index/rgb`; `--video`
records `video.mp4` with the scene, complete RGB image and contact audio, plus
`gelsight.mp4` with just the untouched tactile frames at the selected resolution
(640 × 480 by default, silent).
The composite title sits outside the image, fixing the earlier title bar
that covered the top 27 rows of the displayed RGB. No depth or deformation
panels are included. The original non-tactile command retains its flat
observation format.

To check unloaded contact, a sustained press, release, reset, wrong-key
rejection and the index observation contract:

```bash
PYTHONEXE="$PWD/.venv-tactile/bin/python" /isaac-sim/python.sh scripts/check_piano_tactile.py
```

The arm implementation passed all 42 physical press/release cycles with 42
matching contact-driven audio notes in the headless performance profile at
320 × 240. It recorded 3742 synchronized frames over 124.73 simulated seconds
in `data/arm_shadow_piano/full`. The H5 contains 111 state values and only
`tactile/index/rgb`. All seven arm joints moved, inactive hand bodies remained
within 0.2 micrometres in palm coordinates, and the largest wrist trim was
0.50 mm. This is one complete reference demonstration, not a success-rate study.

The earlier original-API rail implementation passed all 42 press/release cycles and produced
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

## Blind material ball retrieval

`blind_pick_balls.py` is an independent Gymnasium task. It keeps the existing
`envs/blind_pick.py` unchanged and runs in the standalone tactile environment
described above, with a Franka Panda and two GelSight Mini pads.

```bash
bash scripts/run_blind_pick_balls.sh --headless --video
# Twelve balls with faster rendering:
bash scripts/run_blind_pick_balls.sh --headless --rendering_mode performance --image-width 320 --video
# Nine balls, retrieve a soft ball:
bash scripts/run_blind_pick_balls.sh --headless --video --balls 9 --target soft
# Ten balls, retrieve a hard ball, with only two material profiles:
bash scripts/run_blind_pick_balls.sh --headless --video --balls 10 --target hard --materials polished_polymer rubber
```

The default scene contains twelve balls, four per category, using three profiles:
polished polymer, rubber and raised dots. Each sphere has a 30 mm base radius
(previously 24 mm); dots extend up to another 2.5 mm. The box footprint is
36 × 44 cm (previously 40 × 50 cm), with the original 19 cm rim height.
`--balls 2` through `--balls 12` select the count; 9–12 are useful cluttered
settings. Jittered, randomly assigned cells keep the larger balls separated.
The initial cluster stays inside the aperture needed by the wide Panda palm;
the 19 cm walls are retained. The unused wrist RGB camera housing is disabled
in this isolated task; both original GelSight Mini assemblies are retained.
`--materials` selects a subset of the following synthetic profiles, while
balancing their categories:

| Category | Profiles | Physical difference |
| --- | --- | --- |
| `hard` | Polished polymer, hard resin | Smooth spheres, Young's modulus 3 or 6 MPa |
| `soft` | Rubber, foam | Smooth deformable spheres, Young's modulus 60 or 25 kPa |
| `textured` | Raised dots, raised ribs | Actual surface relief, Young's modulus 1 MPa |

Every ball is a volumetric FEM mesh, including its raised surface. The seed
selects profiles when the task is created; resets randomize continuous,
non-overlapping positions inside the box.
`--target hard`, `--target soft` or `--target textured` chooses which category
counts as success. At least one selected material must belong to that category.
Python callers can use `TaskCfg(target_class='soft', ball_count=9)` or change
the goal on reset with `task.reset(options={'target_class': 'hard'})`, provided
that category exists in the already-created scene. Any ball of that category
can satisfy the task; it must still be lifted clear of the rim and released
fully inside the bowl. A wrong category fails the episode.

The target category is public. The policy observation contains 25 robot/TCP
state values, the three-way target category and `tactile/left/rgb` and
`tactile/right/rgb`. No scene camera, object pose, material label or seed is
returned to the policy. The observer camera is used only for video.

Both standalone tasks and the other benchmark tasks import their background
and table defaults from `envs/_scene.py`: `assets/scene/base5.exr`,
`plate01.usda` and its original `texture01` material. The ball-task table is
scaled by 1.35 in both horizontal axes, including its collision geometry.
The piano table is scaled to support its wider keyboard and arm. Sparse checkouts must include these
scene assets and `gridroom_curved.usd` before launch.

The action is `(tcp_x_m, tcp_y_m, tcp_z_m, jaw_position_m)`. The scripted
demonstrator searches the known workspace, localizes through its own pad
contacts and probes incremental squeeze compliance and fine contact relief.
It then lifts the selected ball clear of the rim and releases it into the
bowl. The two tactile contact centroids compensate for an off-center grasp
when choosing the drop position, without reading the ball pose. Scoring
independently checks the actual ball's category, a sustained
bilateral lift and a released ball inside the bowl; a wrong material fails.

PhysX drives the arm; UIPC solves the pads, balls and container together.
The container meshes also have static PhysX collision shapes, so the arm,
fingers and sensor cases are blocked by the walls. The static PhysX shapes
use a 1 mm rest separation and 2 mm contact margin to accommodate convex
cooking and contact tolerances under sustained controller pressure.
The robot asset's original
convex-hull collision meshes are mirrored as native UIPC affine bodies;
pad and rigid-shell contact forces and torques are returned to PhysX through
its original wrench composer. The mirrors are initialized after the robot
reaches its starting joint pose and follow the original attachment callback
order. The existing sensor attachment reset places both gels at that pose
before saving frame zero, avoiding a large initial sweep of the thin pads.
No tactile optical or calibration parameters are changed.
`result.json/collision` reports mirror error, reaction force and maximum
measured penetration of ball vertices into rigid hulls and of robot hulls
into container fixtures. Penetration above 1 mm invalidates the episode,
including an otherwise successful release. These checks supplement the
physical collision solvers. The task reuses `create_gelsight_mini_cfg` and `VisualTactileSensor`
from `envs/sensors/tactile.py`, the original Franka GelSight Mini camera optics,
gel tetrahedra and attachments, bundled Taxim calibration and FEM marker motion.
The calibrated pixel scale is unchanged; the RGB image includes its moving
black markers. No custom pad dimensions or orthographic height projection are
used. The task registers the original sensors and `UipcObject` instances
with `UipcInteractiveScene`, keeps the original physics callbacks, and calls
`UipcSim.update_render_meshes()` for rendering. It does not copy cameras,
create alternate gel surfaces, or override optical parameters.
Small stationary sphere contacts use UIPC's existing configuration fields
`eps_velocity=.001` and `newton.velocity_tol=.0001`. Coarse contact settings
caused creep and loss of the grasp during slow transfers. These task physics
settings do not change the sensor model or optical calibration.
The scripted expert uses internal contact forces and physical contact
relief for search/probing; those measurements are not returned in observations.
The wrapper also exposes Isaac's bundled Torch libraries for the IK solver.
Search uses a narrower jaw opening before centering: the fully open Mini
gripper can straddle a 6 cm ball without touching it.
The expert counts contact gradients only on the original Mini sensing-face
vertices with loads directed into the backing. In clutter, a neighboring ball
can push the back or edge of a pad; those contacts do not establish a grasp.
Squeeze probes regulate their requested load while settling, rather than
freezing the first threshold-crossing jaw position as a soft ball relaxes.
The final grasp approach transitions to force probing when both sensing
faces are loaded and the TCP is within 6 mm of its target. Other moves keep
the usual position tolerance. This accommodates contact deflection from
the physical reaction wrenches without accepting wall or back-shell loads
as a grasp; the independent lift, release and penetration checks still apply.

The shared UIPC renderer converts world-space FEM vertices to the mesh's
current local frame before writing Fabric points. This prevents the arm's
transform from being applied twice, which made gel surfaces appear as small
floating dark pieces in older recordings. A native pose-change check compared
both rendered gel surfaces with their physical vertices (maximum error below
0.1 micrometres) and reproduced the floating piece by restoring the old
coordinate write. Robot, sensor mounting and optical calibration are retained.

Outputs are `data/blind_pick_balls/video.mp4`, `scene.png` and `result.json`. The video has
the observer scene and complete left/right tactile RGB images, with titles
outside the images. Ground-truth labels and positions in `result.json` are
private diagnostics for validating episodes, separate from observations.
`--control idle --idle-frames 120` checks the no-action counterfactual.

Native collision regressions are available separately from the blind policy:

```bash
# With the standalone tactile Python environment, from the repository root:
python scripts/check_blind_ball_collisions.py --headless --check both
python -m pytest -q tests/test_blind_ball_collision.py tests/test_blind_pick_balls.py tests/test_uipc_vertex_reset.py
```

The native checks deliberately command the gripper through a wall, which
must block it, and push a hard ball with the sensor's back shell. They use
known poses only for diagnostic placement and skip RGB rendering while
checking physical collision and wrench feedback.

The native wall test commanded the TCP to y = -0.28 m while below the rim;
the palm was blocked near y = -0.104 m, with 0.0694 mm maximum measured
robot/fixture penetration. The back-shell check pushed a hard ball 95.9 mm
without sensing-face loads or measured ball-vertex penetration into a rigid
hull. These are controlled regression trajectories, separate from retrieval.

The collision-coupled twelve-ball, seed-0 demonstration with `--target soft`
passed at 320 × 240 in the headless performance profile: 5689 physics steps,
2845 frames and 47.41 simulated seconds. It identified a rubber ball with
0.474 mm/N squeeze compliance and 17.6 micrometres of fine surface relief,
held about 2.7 N on each sensing face, verified the lift and released the ball
1.43 mm from the bowl centre. The other eleven balls remained in the box.
Measured ball-vertex/rigid-hull and robot/fixture penetration stayed at zero;
the maximum collider-following error was 0.00190 mm. The complete H.264
recording has 2845 frames at 60 fps (47.42 s), including both tactile RGB views.
The recording, scene preview and diagnostics are in
`data/blind_pick_balls/collision_demo`. The earlier
`data/blind_pick_balls/twelve_balls_demo` recording predates rigid collision coupling.
An earlier native nine-ball idle check switched among all three target categories,
validated their public goal vectors and instructions, and scored zero for
each. It also checked the unchanged 19 cm rim and the scaled table geometry.
The 66 focused CPU checks include layout clearance, material selection,
wrong-category rejection, full-ball placement, sensing-face contact filtering,
drive deadband/relaxation, convex collision geometry, penetration rejection,
bilateral contact transitions and shared UIPC reset/render regressions. These
checks do not establish retrieval success for every layout and target.

The earlier six-ball, seed-0 textured-target demonstration (24 mm ball radius,
six profiles and the larger box) passed in the
headless performance profile at 320 × 240: blind search, tactile classification,
verified lift and released placement completed in 64.45 simulated seconds
(3867 frames). The raised-rib ball settled 0.41 mm from the bowl centre, and
the other five balls remained in the box. Its recording and diagnostics are
in `data/blind_pick_balls/six_balls_demo`. A six-ball idle run also preserved
the original 640 × 480 observation contract and scored zero. Layout sampling
tests now cover counts 2–12, all target categories and material selection;
these demonstrations do not establish retrieval
success across every count, layout and material profile.
The historical six-ball clip predates the shared gel-render transform fix;
use the current twelve-ball recording to review the sensor mounting.

The earlier three-ball, seed-0 textured-target demo completed the blind search,
verified lift and released placement in 45.47 simulated seconds (2728 video
frames). The rubber and polymer distractors remained inside the box. Its
recording and diagnostics are in `data/blind_pick_balls/original_api_demo`.
A separate controlled probe identified rubber as soft and polished polymer as
hard, with measured squeeze compliances of 0.464 and 0.120 mm/N respectively;
this probe used known positions to validate classification, rather than blind
retrieval. A two-ball idle run scored zero and preserved the RGB observation
contract. These checks do not establish success across all layouts and profiles.
