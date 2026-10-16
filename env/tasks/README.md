# Single-finger piano: Shadow Hand and Twinkle Twinkle Little Star

The default task is `arm_shadow_piano.Task`: a Franka Panda carries a five-finger
Shadow Hand and plays the complete 42-note melody of **Twinkle Twinkle Little
Star**, in C major. Only the
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
`--notes 3 --output data/arm_shadow_piano/smoke` plays only C4 C4 G4 for a smoke run;
omit `--notes` for the entire song.
`--wrist rail` selects the previous standalone Shadow Hand on Y/Z slides.

The Shadow task's Gymnasium action is `(wrist_y_m, lift_z_m, index_flexion_rad)`.
The arm variant has 111 state values, including all seven arm joint positions
and velocities. The rail variant retains its 97 values. Both include wrist/finger
positions and velocities, all 88 key
angles, completed-note fraction, press/release state and next pitch. Each note
must physically depress its key with measured contact force and then release
it. Wrong keys fail the episode; repeated pitches require separate releases.
Phrase-ending notes require longer holds. Sound comes from measured key
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
A wrist trim bounded to 1.5 mm uses measured key travel to maintain the same
press criterion across arm poses; it never changes the scoring thresholds.

The hand uses a dark forearm, satin metal finger links, a pale palm shell and
dark fingertip pads. Its articulated/collision geometry comes from Isaac Lab's
Shadow Hand asset. Materials change the appearance; motion and scoring use the
same physical model. The asset downloads from NVIDIA on first launch.
`--hand-usd` may select a local copy of the same hand model.

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

The 320 × 240 headless demonstration passed all 42 notes and produced 42
matching contact-triggered sounds in 124.37 simulated seconds, with 3731 RGB
frames at 30 Hz. Its recordings are in `data/arm_shadow_piano_digit/full`;
the three-note close-up is in `data/arm_shadow_piano_digit/preview`.
Native contact-point inspection confirmed contact near the gel centre after
correcting the mount angle. Recorded RGB responds to those contacts, and the
640 × 480 renderer output also passed a contact/no-contact check.
The retained Mini/UIPC backend passed a separate one-note regression.

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
# Reference six-ball demonstration with faster rendering:
bash scripts/run_blind_pick_balls.sh --headless --rendering_mode performance --image-width 320 --video
# Increase clutter to nine balls:
bash scripts/run_blind_pick_balls.sh --headless --video --balls 9
# Two balls, another target category and another random layout:
bash scripts/run_blind_pick_balls.sh --headless --video --balls 2 --target soft --seed 3
```

The default scene contains six balls, two per category, using all six profiles.
`--balls 2` through `--balls 9` select the count, with balanced categories and
randomized, non-overlapping positions. The catalogue
has six synthetic material profiles:

| Category | Profiles | Physical difference |
| --- | --- | --- |
| `hard` | Polished polymer, hard resin | Smooth spheres, Young's modulus 3 or 6 MPa |
| `soft` | Rubber, foam | Smooth deformable spheres, Young's modulus 60 or 25 kPa |
| `textured` | Raised dots, raised ribs | Actual surface relief, Young's modulus 1 MPa |

Every ball is a volumetric FEM mesh, including its raised surface. The seed
selects profiles when the task is created; resets randomize continuous,
non-overlapping positions inside the box.
The target category is public. The policy observation contains 25 robot/TCP
state values, the three-way target category and `tactile/left/rgb` and
`tactile/right/rgb`. No scene camera, object pose, material label or seed is
returned to the policy. The observer camera is used only for video.

Both standalone tasks and the other benchmark tasks import their background
and table defaults from `envs/_scene.py`: `assets/scene/base5.exr`,
`plate01.usda` and its original `texture01` material. The piano table is scaled
to support its wider keyboard and arm. Sparse checkouts must include these
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
Gel backing poses follow the fingers, with no gel-force feedback to robot
drives. The task reuses `create_gelsight_mini_cfg` and `VisualTactileSensor`
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

Outputs are `data/blind_pick_balls/video.mp4` and `result.json`. The video has
the observer scene and complete left/right tactile RGB images, with titles
outside the images. Ground-truth labels and positions in `result.json` are
private diagnostics for validating episodes, separate from observations.
`--control idle --idle-frames 120` checks the no-action counterfactual.

The default six-ball, seed-0 textured-target demonstration passed in the
headless performance profile at 320 × 240: blind search, tactile classification,
verified lift and released placement completed in 64.45 simulated seconds
(3867 frames). The raised-rib ball settled 0.41 mm from the bowl centre, and
the other five balls remained in the box. Its recording and diagnostics are
in `data/blind_pick_balls/six_balls_demo`. A six-ball idle run also preserved
the original 640 × 480 observation contract and scored zero. Layout sampling
tests cover counts 2–9; these demonstrations do not establish retrieval
success across every count, layout and material profile.

The earlier three-ball, seed-0 textured-target demo completed the blind search,
verified lift and released placement in 45.47 simulated seconds (2728 video
frames). The rubber and polymer distractors remained inside the box. Its
recording and diagnostics are in `data/blind_pick_balls/original_api_demo`.
A separate controlled probe identified rubber as soft and polished polymer as
hard, with measured squeeze compliances of 0.464 and 0.120 mm/N respectively;
this probe used known positions to validate classification, rather than blind
retrieval. A two-ball idle run scored zero and preserved the RGB observation
contract. These checks do not establish success across all layouts and profiles.
