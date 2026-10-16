# Task Creation Guide

Task creation covers asset generation, tetrahedral meshing, point annotation, initial placement, scripted actions, success checks, and early-stop checks. A new task usually corresponds to two classes in `envs/<task_name>.py`: `TaskCfg` and `Task`. The `Task` class inherits from `BaseTask` and implements the following methods as needed:

- `create_actors`: Create all assets used by this task.
- `_reset_actors`: Randomize asset states on every `reset`.
- `pre_move`: Preparation actions before data collection starts.
- `_play_once`: The actual task action used for data collection or demonstration.
- `check_success`: Success condition.
- `check_early_stop`: Early failure condition used during evaluation.

## Asset Generation

Assets should first be modeled in Blender, CAD software, or another modeling tool, then exported to a format supported by `scripts/convert.py`, such as `obj`, `stl`, `fbx`, or `glb`. Before exporting, use the following conventions:

- Use meters as the unit, or verify the export scale in the modeling software to avoid incorrect simulation sizes.
- Use Z-up coordinates, and place the asset origin at a position that is convenient for later placement, grasping, and success checks.
- Merge meshes that do not need independent control. The conversion script also enables `merge_all_meshes`.
- Surface meshes should be closed, have consistent normals, and avoid obvious self-intersections. Otherwise, tetrahedral meshing can fail. Blender's 3D Print Toolbox can be useful for making a mesh manifold.
- File names should be valid USD identifiers. English letters, numbers, and underscores are recommended.

Converted `.usd` files are usually placed under `assets/objects/` and referenced by file name in task code, for example:

```python
self.block = self._actor_manager.add_from_usd_file(
    name="blue_block",
    asset_path="BlueBlock.usd",
    pose=Pose([0.7, 0.0, 0.005], [1, 0, 0, 0]),
)
```

## Tetrahedral Meshing

`scripts/convert.py` converts regular meshes to USD and writes tetrahedral mesh attributes required by UIPC soft-body simulation onto the USD Mesh prim:

- `tet_points`
- `tet_indices`
- `tet_surf_points`
- `tet_surf_indices`

These attributes are saved with the `.usd` file and are used directly when `Actor.from_usd_file` loads the asset. Basic usage:

```bash
python scripts/convert.py \
  --input assets/objects/ipt/MyObject.obj \
  --output assets/objects/MyObject.usd \
  --show
```

`--input` can be a single file or a directory. Directory mode batch-converts `obj`, `stl`, `fbx`, and `glb` files inside the directory. `--output` can be a target USD file or an output directory. Common option:

- `--show`: Visualize the generated tetrahedral mesh with trimesh, so you can check for missing surfaces, flipped surfaces, or meshes that are too coarse or too dense.

There are two meshing backends:

- `tetgen`: The current default backend. The script first cleans the surface with `pymeshfix`, then calls `tetgen.TetGen(...).tetrahedralize()` to generate tetrahedra, and extracts the surface mesh from `pyvista`.
- `MeshGenerator`: The built-in backend, using `MeshGenerator` and `TetMeshCfg` from `scripts/utils/mesh_gen.py`. This branch is used when `gen_tet(..., backend=...)` receives a backend name other than `tetgen`. Mesh quality and density can be adjusted through `stop_quality`, `max_its`, `edge_length_r`, and `epsilon_r`.

If `--show` reveals poor mesh quality, switch backends or tune the built-in backend's `TetMeshCfg` parameters.

## Point Annotation

Points in a task are local coordinate frames with poses. They describe grasp points, target points, functional points, and other task-specific references. `Actor` supports four point types:

- `contact`: Grasp points. `atom.grasp_actor` selects grasp poses from this type by default.
- `target`: Target points, often used as intermediate references during task planning.
- `functional`: Functional points, representing the actual interaction part of an object, such as a hammer head, insertion tip, or bottle mouth.
- `orientation`: Orientation points, used to describe object direction.

When registering a point, pass a `Pose` in the current world frame. `Actor.register_point` automatically converts it into the asset's local frame before saving it:

```python
grasp_pose = construct_grasp_pose(
    target_pose.p,
    [0, 0, 1],
    [1, 0, 0],
)
grasp_idx = self.prism.register_point(grasp_pose, type="contact")
```

When using a point, `Actor.get_point` converts the saved local point back into the world frame according to the object's current pose:

```python
world_pose = self.prism.get_point(type="contact", idx=grasp_idx)
world_matrix = self.prism.get_point(type="contact", idx=grasp_idx, ret="matrix")
```

You can also preconfigure `contact_points`, `functional_points`, and similar matrix lists in `ActorCfg`. Most tasks dynamically register points in `pre_move` or `_play_once` according to the randomized object pose. Dynamic registration is useful because random offsets, grasp heights, and functional point poses can be written directly into the current episode.

Common `Pose` helper methods:

- `add_bias(vec, coord="local")`: Translate along local or world coordinates.
- `add_rotation(euler, coord="local")`: Apply an additional Euler-angle rotation.
- `add_offset(noise)`: Add a random pose offset.
- `rebase(to_coord=pose)`: Express the pose in another coordinate frame. This is commonly used in success checks.
- `to_transformation_matrix()`: Convert to a 4x4 transformation matrix.

## Initial Placement

`create_actors` is called only once during the environment lifecycle. It creates every asset that may be used by the task. Since dynamic asset loading at runtime is not currently supported, every object that may appear in any branch must be created here. Assets that are not used immediately can be placed far away from the workspace and moved into the task area later in `_reset_actors`.

Example:

```python
def create_actors(self):
    self.slot = self._actor_manager.add_from_usd_file(
        name="slot",
        asset_path="TestTubeHoleSlot.usd",
        pose=Pose([0.6, 0.0, 0.002], [1, 0, 0, 0]),
        density=1e5,
    )
    self.prism = self._actor_manager.add_from_usd_file(
        name="prism",
        asset_path="TestTube.usd",
        pose=Pose([0.4, 0.0, 0.005], [1, 0, 0, 0]),
        density=10,
    )
```

`_reset_actors` is called on every `reset`. It is used to randomize positions, poses, category choices, and episode-level variables. If `_reset_actors` is not implemented, assets keep the initial poses specified in `create_actors`. A common pattern is to use `create_noise` to generate a random offset, then write the pose back with `set_pose`:

```python
def _reset_actors(self):
    block_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi / 3])
    block_pose = Pose([0.7, 0.0, 0.005], [1, 0, 0, 0]).add_offset(block_noise)
    self.block.set_pose(block_pose)
```

If multiple randomized objects must not intersect or be too close, sample in a loop until the distance constraint is satisfied:

```python
while True:
    hammer_noise = self.create_noise([0.02, 0.05, 0.0], [0, 0, np.pi / 3])
    hammer_pose = base_hammer_pose.add_offset(hammer_noise)
    if np.linalg.norm(hammer_pose.p - block_pose.p) > 0.1:
        break
self.hammer.set_pose(hammer_pose)
```

`_reset_actors` can also record episode state needed later, such as the selected category, target object, target pose, or labels written to `metadata`.

## Scripted Actions

Task actions are mainly written in `pre_move` and `_play_once`.

`pre_move` is the preparation stage before data collection. It is not saved as collected data, but it also runs during evaluation. It is suitable for stabilizing the scene, opening the gripper, pre-grasping an object, moving an object to the initial interaction pose, and similar setup actions.

`_play_once` is the actual data-collection or demonstration stage. In collection mode, observation saving starts from this method.

Actions are executed through `self.move(...)` and `self.delay(...)`. Common atomic actions are available under `self.atom`:

- `self.atom.grasp_actor(actor, contact_point_id=..., pre_dis=..., dis=..., is_close=True)`: Move to a grasp point and optionally close the gripper.
- `self.atom.place_actor(actor, target_pose, functional_point_id=..., pre_dis=..., dis=..., is_open=True, constrain="align")`: Place an object, or a functional point on the object, at a target pose.
- `self.atom.move_by_displacement(x=..., y=..., z=..., xyz_coord="world" | "local" | Pose, rpy=...)`: Translate or rotate relative to the current end-effector pose.
- `self.atom.move_to_pose(target_pose)`: Move the end effector directly to a target pose.
- `self.atom.open_gripper(pos=1.0)` / `self.atom.close_gripper(pos=0.0)`: Open or close the gripper.
- `self.atom.back_to_origin()`: Return the robot to its initial pose.

Example flow:

```python
def pre_move(self):
    self.delay(10)
    self.move(self.atom.open_gripper(0.5))

    target_pose = self.prism.get_pose().add_bias([0.0, 0.0, 0.04])
    grasp_pose = construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])
    grasp_idx = self.prism.register_point(grasp_pose, type="contact")

    self.move(self.atom.grasp_actor(
        self.prism,
        contact_point_id=grasp_idx,
        pre_dis=0.04,
        dis=0.0,
    ))
    self.move(self.atom.move_by_displacement(z=0.05))

def _play_once(self):
    self.move(self.atom.place_actor(
        self.prism,
        target_pose=self.target_pose,
        pre_dis=0.04,
        dis=0.0,
        is_open=False,
    ))
    self.delay(20, is_save=False)
```

Common `self.move` parameters:

- `tag`: Label the current action segment. The value is saved to the observation's `atom` field.
- `is_save`: Whether to save observations for this action segment.
- `delay`: Whether to automatically delay several steps after each action.
- `constraint_pose`: Constraint mask passed to the motion planner, for example to restrict specific degrees of freedom. A common value `[1, 1, 1, 0, 0, 0]` constrains rotation and only plans translation. `[1, 1, 1, 1, 1, 0]` only allows the gripper to move along its local z axis, which is the gripper approach direction, and is often used for insertion tasks. The stricter the constraint, the more likely planning is to fail.
- `time_dilation_factor`: Adjust the execution speed of the planned action.
- `gripper_depth_threshold`: Override the tactile depth threshold used by adaptive grasping.

If `self.move` planning fails, `self.plan_success` is set to `False`, and later actions stop executing.

## Success and Early-Stop Checks

`check_success` determines whether the task has succeeded. In collection mode, it usually runs only once after `_play_once` finishes. In evaluation mode, it runs after every policy action. Common checks include:

- Whether the target object's position error relative to the target pose is below a threshold.
- Whether the target object's orientation axis is aligned with the expected direction.
- Whether the object is still in the gripper, or has been inserted, placed, or stabilized.
- Whether necessary `metadata` has been recorded for post-hoc failure analysis.

Example:

```python
def check_success(self):
    rel_pose = self.prism.get_pose().rebase(self.target_pose)
    self.metadata["rel_pose"] = rel_pose.tolist()
    return (
        np.all(np.abs(rel_pose.p) < np.array([0.02, 0.02, 0.01]))
        and np.dot(rel_pose.to_transformation_matrix()[:3, 2], [0, 0, 1]) > 0.965
    )
```

To check whether a functional point has reached the target, use `get_point` to retrieve the current world pose of the functional point, then rebase it to the target pose:

```python
functional_pose = self.hammer.get_point("functional", self.function_idx)
rel_pose = functional_pose.rebase(self.place_pose)
return np.all(np.abs(rel_pose.p) < [0.02, 0.02, 0.02])
```

`check_early_stop` runs step-by-step only in evaluation mode. It does not run in collection mode. It ends obviously failed trajectories early to save evaluation resources. It is suitable for detecting:

- The object has slipped out of the gripper.
- Tactile depth or contact state is clearly abnormal.
- The object has moved too far away from the target area.
- The pose has become impossible to recover to a successful state.

Example:

```python
def check_early_stop(self):
    prism_inhand_pose = self.prism.get_pose().rebase(
        self._robot_manager.get_gripper_center_pose()
    )
    inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
    if inhand_bias > 0.04:
        self.metadata["early_stop"] = True
        self.metadata["inhand_bias"] = float(inhand_bias)
        return True
    return False
```

Keep both `check_success` and `check_early_stop` lightweight. They should only read the current simulation state and return a boolean value; they should not execute actions that modify the simulation state.

## Open-box tactile search (`blind_pick`)

This task contains one 5 by 5 by 6 cm rounded block at a continuously randomized position and yaw inside an opaque, open-top box, measuring 32 by 38 cm with its rim 19.7 cm above the table. The whole top is open: there is no lid, curtain, or flexible entrance. The arm enters from above, searches by touch, grasps the block, and withdraws it over the rim. There is no reference object or texture-matching stage.

Only the external head camera is instantiated; the wrist visual camera is removed from the sensor configuration and observations. The head camera sits at a low external angle, so the box walls conceal the initial object placement. Tactile cameras and proprioception remain available. Actor poses and the episode seed are excluded from policy observations/context. Sampled poses are retained only in episode metadata for diagnostics and scoring. Review videos combine the head view and both tactile images. High-angle geometry previews are for human inspection, not policy input.

The scripted demonstrator follows a continuous serpentine sweep without reading the sampled object pose, and remembers which locations it has already visited. Physical contact forces and contact positions on its own tactile pads localize the touched face, including edge contacts outside the tactile camera's optical window. The hand retreats, clears the object, and returns to the remembered location. During closure, moving the hand toward the first loaded pad keeps that jaw approximately stationary while the other jaw approaches.

Opposing contact patches give a remembered centre and, for a diagonal pinch, an estimate of the square block's face direction. The hand unloads before recentering or rotating, then grasps the faces. This avoids dragging the block with loaded pads or letting a corner pinch rotate out of the tactile area. Tactile feedback maintains preload throughout withdrawal and pauses ascent if a pad unloads. Missed grasps retreat above the object before resuming the scan. These control decisions use touch and robot state; object pose appears only in reset, diagnostics, and scoring.

Success requires the entire object to clear the rim by 15 mm while held near the gripper with bilateral contact for 18 consecutive physics ticks. Lifting inside the box, throwing the object, or dropping it does not pass. At 120 Hz the hold is 0.15 s; observations/video are recorded at 60 Hz. The search has a 1200-physics-tick budget, not a claimed average episode length.

Use the regular collection entry point with task/config `blind_pick`:

```bash
bash collect_data.sh blind_pick blind_pick 0
```

On fresh-process seeds 0, 1, and 2, complete retrieval passed in 841, 1148, and 873 physics ticks. The mean was 954 total ticks, 918 policy ticks, and 458.3 saved policy observations. Total recorded frames, including preparation and terminal observations, were 404/557/420 (mean 460.3). Each object cleared the rim by about 48 mm and maintained bilateral contact for 46 ticks. Geometry, decoded observations, and terminal physical gates were checked in the collected episodes. The regular collector also completed seeds 3 and 4 in one process, including reset between episodes, in 873 and 831 ticks.

## Piano articulation (`play_piano`)

The standalone [arm and Shadow Hand task](../env/tasks/arm_shadow_piano.py)
and the tapping-tool task both accept an ordered note prompt. Without one,
they sample six C4–B4 notes, including black keys; an explicit seed reproduces
the sequence. Pitch names accept sharps/flats and optional octaves (default 4).
The hand's `--song twinkle` preset retains all 42 notes. The arm moves in X/Y/Z
to reach both rows; the older rail and fixed-palm variants remain available.
See [prompts, interfaces and timestep labels](../env/tasks/README.md).

The scene is an upright piano with 88 contactable keys in the standard 52-white/36-black arrangement, a keybed, cabinet, music desk, and decorative pedals. The arm and tactile gripper pick up a tapping tool and play the requested sequence. Keys use stiff FEM bodies with compliant rear flexures and exactly pinned rear faces. The tool has a rounded dark teal grip, brass ferrule and cap, satin steel shaft, and ivory-coloured compliant tip. Its detailed visible surface is also its deforming FEM contact surface, with a 26 mm grip, 7.8 mm tip, and 108 mm overall length. Static box panels and piano supports are exactly fixed. The tool task solves key motion and fingertip reactions through UIPC contact; the hand's DIGIT backend uses PhysX contacts and the original TACTO renderer.

Default success checks the requested pitch order, contact in the target force band, minimum hold and separate release. Both task configs enable dynamics judging by default. Set `piano_require_dynamics: false` for the tool or pass `--no-judge-dynamics` for the hand to disable force-band scoring while retaining the timestep labels. The tool targets 1.5 ± 0.45 N for light and 3.0 ± 0.45 N for heavy; hold duration is separate, defaulting to 0.12 s. `piano_wrong_key_policy` selects immediate failure or retry at the current note. The tool retains its overload check and acquisition requirement. Configure the prompt through `piano_notes` and related settings in `task_config/play_piano.yml` or through reset options.

Both tasks record separate target and actual dynamics every physics step (0 idle, 1 light, 2 heavy), with note indices, measured force and a simulation timestamp. The tool exposes numeric observation fields under `piano`; the hand uses Gym `info["piano"]`. Recorded HDF5 frames carry aligned `piano/` fields, and JSONL traces retain every physics tick. Video annotations affect the observer view, preserving raw tactile RGB. These fields and overrides apply only to the two piano tasks.

Contact onset is 0.3 N and release is below 0.1 N. This hysteresis prevents the initial contact transient from being mistaken for release; forces between these thresholds do not count toward a valid note-force hold.

Both head and wrist cameras show the scene without a cue mask. Policy observations omit actor poses, the seed, and privileged action tags. The scripted demonstrator uses physical key forces and geometry; its success is not a learned-policy memory ablation. Contact descent remains capped at 75 micrometres per 60 Hz update. Its position servo uses the median of the last three 120 Hz force samples, preventing a brief landing impulse from commanding an immediate lift. The scorer and overload detector still use every raw force sample. Self-contact is disabled within the intact stylus because its fine fillets are smaller than the contact barrier; contact with the jaws, keys and table remains active.

Recorded videos include a synthesized piano soundtrack and a companion WAV. All 88 physical key contacts can trigger sound, including wrong keys; the renderer does not read the intended melody or successful-note labels. Contact onset/release determines timing, the actual peak contact force controls amplitude/brightness, and the physical key index determines pitch. Physical event steps are mapped to actual recorded frames, including the policy boundary. Sound uses decaying, slightly detuned partials and a release envelope; it is an audible presentation of the contact events, not a simulation of strings or room acoustics. Audio is generated at episode finalization, so headless collection produces playable sound without requiring an audio device.

The earlier fixed five-note tool recording passed seeds 0 and 1 in the regular collector, including reset, in 1334 and 1356 physics ticks. Those recordings remain reference results for the original phrase. The configurable hand task passed the C#4/C4/G4/A4/E4/F4 prompt with strict dynamics (5942 ticks, 743 tactile frames), and the default random seed-0 six-note prompt (4829 ticks, 604 frames). Per-tick labels, HDF5 alignment, measured-force classes and physical audio pitches were checked. The configurable tool task passed the same six-note prompt on seed 7, with 1598 policy ticks, 821 saved frames and six matching physical audio events. The standard episode validator passed after decoding all 4926 saved RGB images; outputs are in `data/piano_sequences/tool`. The regular collector also passed seeds 8 and 9 in the same process, including reset, with strict light/heavy judging on two separately released C4 strokes: 690/680 policy ticks and 368/362 saved frames. Both episodes passed RGB-only image/schema validation and JSONL/HDF5/audio alignment checks (`data/piano_sequences/tool_repeat`). These are scripted demonstrations, not trained-policy results.

The Isaac Sim 5.1 / Isaac Lab 2.3.2 gripper path reuses the original Mini
factory, optical calibration, FEM pad and adaptive-grasp API. Reset synchronizes
the gel vertices and original attachment targets with the robot pose before the
first solve. FEM writes use UIPC's `backend_fem_vertex_offset`, because native
storage is grouped by geometry dimension/constitution and can differ from Python
creation order. Rendering converts UIPC world vertices into each mesh's current
Fabric local frame after `SimulationContext.forward()` and updates local bounds.
This avoids the extra transform that PhysX applies when refreshing a cached
`worldMatrix`. The original gel visibility flag is retained: the optical camera
measures contacting objects through the gel. The observer-camera adapter also
preserves scale values while normalizing float-authored scales to Vec3d for
Isaac Lab's pose reader. The shared wrapper's `gripper` selector calls the regular
collector, whose app arguments are now registered before parsing.

Development runs exposed an early release after a one-tick landing impulse and a reset timeout with a tighter global Newton tolerance. The final controller uses the three-sample median described above while retaining the original 0.01 m/s solver tolerance and all raw scoring gates. Short-note landing transients can briefly exceed their 1.5 N holding target; neither scoring nor the sound renderer replaces these measurements with the requested target force.

Both new tasks export one actual terminal observation with the physical verdict. Their `phase/policy_step` records elapsed physics ticks since handoff; the public action cadence is two physics ticks per action. Piano review frames contain two 640 by 480 camera views and two 240 by 240 tactile views; blind search contains only the head view and tactile views.

Configure downstream camera lists to match each task: `blind_pick` has `head` only, plus the two tactile streams. Validate successful raw episodes with `python scripts/validate_memory_task_episode.py path/to/hdf5/0.hdf5 --expect success`. This checks cadence, terminal scoring, image decoding, camera sets, and absence of privileged labels in policy fields.

```bash
bash collect_data.sh play_piano play_piano 0
```

## Motion speed and episode length

`tension_strap` and `bulb_tightening` accept `task_parameters.motion_speed_scale` from 1 to 3. The code default is 1, preserving legacy timing. This parameter changes motion timing while retaining the 120 Hz physics clock, 60 Hz policy/recording cadence, force/torque acceptance thresholds, and required holds. Strap motion accelerates free gripper closure and tapers back to its original controller near contact and the force target. Bulb motion accelerates unloaded wrist return, unloaded recentering, free closure after the direction probe, and terminal opening/lift. The initial grip and reversible tactile probe retain their original timing. A separate bulb `thread_speed_scale` (1--3, default 1) accelerates free thread travel while retaining the 60 Hz command cadence and half-turn endpoint. Sensed seating load switches back to the original slow increments. The final stationary submission and physical hold durations remain unchanged. The standard bulb YAML selects motion scale 3 and thread scale 2.

Bulb default sampling now targets nominal 1.55--2.88 turns, leaving margin for actual seat/probe compliance; physical success requires 1.5--3 measured formal turns. Previously sampled seeds already within the new nominal range retain their geometry, including comparison seeds 0, 1 and 2. Out-of-range default samples are redrawn; explicit `nominal_turns` must be within 1.5--3. Direction and lead estimation still use tactile image responses rather than the hidden thread specification.

The strap accepts exactly two force/time stages through
`task_parameters.force_time_sequence: [F1_N, T1_s, F2_N, T2_s]`, or
`--force-sequence F1_N T1_s F2_N T2_s` on `scripts/collect_data.py`.
The instruction interval is 0 N through `overload_N` (24 N with the current
band configuration); durations are independently chosen
positive seconds. For example, `--force-sequence 13 1.5 17 4` requests
13 N for 1.5 seconds followed by 17 N for 4 seconds. Descending levels are
also accepted. The default remains `[12, 3, 18, 3]`. Each stage requires
continuous measured physical tension within 0.5 N of its target; an excursion
restarts that stage's hold. Zero means near-zero tension while retaining the
grip. The unchanged overload check additionally rejects tension above 24 N.
For a requested 24 N, the controller aims at 23.75 N, inside the intersection
of the original tolerance and overload limit; the physical target remains 24 N.
Both requested targets and controller references are recorded separately.
The RGB controller uses the corresponding minimum hold duration plus 0.30
seconds of settling margin. The original sensor, mesh, optical calibration
and feature/prediction interfaces are retained. The old 12/18 N force
regressor underestimates unloading tension over the expanded range. The
new frozen `tension_strap_full_range.json` extends its force coefficients
using separate native RGB/physical-force probes; the original regressor
remains available. Regression inputs contain image-derived features only.
Native image-feedback trials passed `[0, .75, 24, 1.25]`,
`[24, 1, 0, .75]` and `[4, 1.25, 20, .75]`, with the original force,
grip, strain, travel and rupture checks enabled.

The policy timeout is at least the sum of both requested holds plus 30 seconds
(60 seconds by default); the frame/action and worker wall-time budgets grow with it. Metadata,
instructions, per-step physical diagnostics and HDF5 validation use the same
declared sequence. The previous `target_sequence_N` plus scalar/list
`hold_seconds` configuration remains readable. Original v1 recordings retain
their original default contract when validated.

Validated strap datasets can be collected and uploaded from the repository
root after installing `requirements-dataset.txt` in the upload environment,
logging in with the official ModelScope Hub client and selecting an existing
writable dataset:

```bash
python scripts/collect_tactile_memory.py --task tension_strap --count 50 --start-seed 100 --repo-id OWNER/DATASET
```

The native interpreter defaults to `.venv-tactile/bin/python`;
`--native-python` and `--isaac-root` select other installed paths. The
collector freezes source hashes, starts a fresh native scene per physics
seed and counts successful, validated episodes. Prompt forces and hold
times use a separate random stream from hidden material parameters.
The collector uses one native process at a time and overlaps uploads with
the next episode.

Uploads contain the original HDF5, instruction/configuration, physical
verdict, checksum manifest and issued-joint-command audit. The strap's
`work/<seed>/trace.jsonl` includes privileged physical force diagnostics
for auditing. The first accepted HDF5 is downloaded and hash-checked;
each subsequent upload includes remote manifest readback. Completed
episodes can be resumed with the same source hashes and output path;
interrupted attempts are preserved for inspection.

Audit collected recordings and publish a dataset index with:

```bash
python scripts/audit_tactile_memory.py --tasks tension_strap
python scripts/publish_tactile_dataset.py --repo-id OWNER/DATASET --publish
```

The auditor decodes every RGB frame and checks frame counts, numeric
finiteness and nonempty image streams. It reuses an audit only while the
recording size, modification timestamp and audit implementation remain
unchanged. Add `--watch` to audit alongside collection until the selected
tasks each have 50 validated and uploaded episodes. Without `--tasks`, it
watches the strap and both piano datasets. The catalog includes only
episodes that pass collection validation, upload readback and this image
audit; `_checks/` upload tests do not count toward the requested totals.
Publishing checks both remote index files against their exact local hashes,
allowing brief branch-cache propagation after updates.

Bulb submission still requires 180 stationary physics ticks, while physical
success requires at least 132 stable ticks; changing `policy_hold_steps`
alone does not shorten the stationary submission gate.

The standard `tension_strap.yml` profile selects scale 3. On paired seeds 0, 1, and 2, both scale 1 and scale 3 succeeded in all three episodes. Total physics ticks were 2833/2870/2911 before and 1458/1490/1594 after: the mean fell from 2871.3 to 1514 (47.3%). Mean policy ticks fell from 2807.3 to 1450, and saved policy frames from 1403 to 724. Total recorded frames are 711/727/779 (mean 739). Both force stages still require three seconds of sustained physical acceptance. Comparing recorded end-effector pose and jaw width after normalized path-length alignment gave maximum position differences of 0.186/0.066/0.083 mm and orientation differences of 0.100/0.038/0.021 degrees. This is a three-seed regression result, not a large-scale success-rate estimate.

The current bulb profile passed seeds 0, 1, and 2 in 2557/3520/2780 physics ticks, with 1.648/2.862/2.359 actual formal turns. The preceding unloaded-motion-only profile passed the same seeds in 3169/4541/3974 ticks. Mean total ticks fell from 3894.7 to 2952.3 (24.2%), and policy ticks from 3630.7 to 2688.3 (26.0%). Mean saved policy frames fell from 1814.7 to 1343.3. Total recorded frames are 1256/1737/1367, averaging 1453.3 versus 1924.7 previously (24.5% fewer). All three episodes passed the unchanged torque, seating, damage and hold gates and the stricter actual-turn gate; all three 60 fps videos fully decoded and matched HDF5 frame counts.

The motion is not an identical geometric replay. Seeds 0 and 1 retain their complete four/seven wrist-positioning sequences. After per-stage pose-arc-length alignment, maximum position differences are 0.578/1.772 mm and orientation differences 1.826/7.802 degrees. Whole pose-and-jaw-curve alignment gives 0.627/1.729 mm, 1.826/13.830 degrees and 1.508/8.270 mm full-jaw-width differences. Seed 2 uses five wrist-positioning cycles instead of six: faster free travel leaves enough of the existing segment budget to finish seating before another regrasp. Its complete sequence therefore differs, and whole-curve alignment across the changed cycles is not a geometric deviation bound. Wrist counts include repositioning after the direction probe.

These bulb episodes still exceed the requested 500--900 recorded-frame range. The demonstrated reduction retains the initial tactile probe, half-turn motion segments, and stationary hold. In seed 1 alone, initial closure takes 364 physics ticks (182 data intervals), the reversible probe 348 ticks (174 intervals), stationary END 180 ticks (90 intervals), and terminal review 190 ticks (95 intervals), in addition to approach, repeated regrasp and rotation. No recording-rate reduction, shortened hold, or playback speedup is used to claim an efficiency gain.

For historical context, the original pre-speedup bulb profile succeeded on seeds 0 and 2 in 3755/4717 ticks; seed 1 failed direction probing. The preceding unloaded-motion-only profile averaged 3571.5 ticks on that common-success pair, 15.7% below the original 4236-tick mean. Current comparisons above instead use the same three successful seeds on both sides. Probe estimates varied between runs, so these few episodes do not establish a success-rate improvement.

A development trial that also accelerated the initial grasp failed to resolve thread direction at 850 physics ticks. Its estimated lead was 3.378 mm, below the unchanged 3.5 mm validity bound. Probe estimates varied even in earlier runs, so this observation alone does not establish that grip speed caused the failure. The final profile conservatively preserves the original initial grasp and probe instead of lowering the validity bound.

For comparison, the unchanged `insert_USB` baseline succeeded on two of three attempted seeds (0 and 1 succeeded; 2 failed). Successful episodes averaged 493 total physics ticks, 435 policy ticks, 217 saved policy observations, and 224 total recorded frames. All measured runs used 120 Hz physics and recording every two ticks. These finite-seed measurements describe scripted demonstrations, not trained-policy success rates.

Measure collected episodes rather than `step_lim` or video length:

```bash
python scripts/summarize_episode_steps.py data/path/to/runs
```

The output reports attempts and successes, successful-episode means and sample counts, total physics ticks, preparation ticks, policy ticks, terminal review ticks, saved policy frames, and `saved_total_frames` counted directly from HDF5 rows. Terminal reveal movements after the bulb's physical verdict are excluded from policy timing. Saved frames are not physics ticks; changing recording frequency alone does not accelerate execution.

The measurements above were collected on 2026-10-10 with Isaac Sim 4.5, Isaac Lab 2.1.1, libuipc 0.9.0, and an RTX 4090. The related regression suite passed 150 tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_tactile_memory.py tests/test_piano_audio.py tests/test_blind_box.py tests/test_task_speed.py \
  tests/test_episode_steps.py tests/test_four_force_tasks.py tests/test_force_task_integration.py -q
```
