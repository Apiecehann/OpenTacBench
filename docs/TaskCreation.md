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

On fresh-process seeds 0, 1, and 2, complete retrieval passed in 841, 1148, and 873 physics ticks. The mean was 954 total ticks, 918 policy ticks, and 458.3 saved policy observations. Each object cleared the rim by about 48 mm and maintained bilateral contact for 46 ticks. Geometry, decoded observations, and terminal physical gates were checked in the collected episodes. The regular collector also completed seeds 3 and 4 in one process, including reset between episodes, in 873 and 831 ticks.

## Piano articulation recall (`play_piano`)

The scene is an upright piano with 88 contactable keys in the standard 52-white/36-black arrangement, a keybed, cabinet, music desk, and decorative pedals. The existing arm and tactile gripper pick up a slender tapping tool and play E4 D4 C4 D4 E4. Keys use stiff FEM bodies with compliant rear flexures and exactly pinned rear faces; the tool has a compliant tip. Static box panels and piano supports are also exactly fixed so collision geometry stays aligned with the visual shells under gravity. Both key motion and fingertip reactions are simulated through UIPC contact.

Before playing, the gripper touches a cue with one or two raised ribs, then releases it. One rib selects a short-light first note; two ribs select a long-firm first note. Subsequent notes alternate. Short notes require 1.5 ± 0.45 N for 0.20 s; long notes require 3.0 ± 0.45 N for 0.60 s. Scoring uses physical contact loads at every 120 Hz physics tick, including wrong-key presses, overload, early release, and late release. The cue must first be touched continuously for 12 ticks, then released for at least 12 ticks.

Contact onset is 0.3 N and release is below 0.1 N. This hysteresis prevents the initial contact transient from being mistaken for release; forces between these thresholds do not count toward a valid note-force hold.

The cue's fixed world volume is masked in both visual cameras, including depth if requested. Tactile views retain its physical relief. Policy observations omit actor poses, latent cue labels, seed, and identifying action tags. The scripted demonstrator is privileged; its success is not a learned-policy memory ablation. Human review videos use unmasked camera views, while recorded policy observations remain masked.

Contact descent is capped at 75 micrometres per 60 Hz update: larger first-contact increments produced an impact overload on one seed even when the preceding force sample was small. The original force thresholds remain enforced. Fresh-process seeds 0, 1, and 2 passed the complete phrase in 1936, 1831, and 1934 physics ticks, covering both cue kinds. Means were 1900.3 total ticks, 1819.7 policy ticks, and 909 saved policy observations. In seed 0, short-note key travel was about 1.1 mm and long-note travel about 2.6 mm.

The regular collector also passed seeds 5 and 6 consecutively in one process, in 1966 and 1829 ticks. This checks reset with the cue changing from two ribs to one, including both complete five-note phrases.

Both new tasks export one actual terminal observation with the physical verdict. Their `phase/policy_step` records elapsed physics ticks since handoff; the public action cadence is two physics ticks per action. Piano review frames contain two 640 by 480 camera views and two 240 by 240 tactile views; blind search contains only the head view and tactile views.

Configure downstream camera lists to match each task: `blind_pick` has `head` only, plus the two tactile streams. Validate successful raw episodes with `python scripts/validate_memory_task_episode.py path/to/hdf5/0.hdf5 --expect success`. This checks cadence, terminal scoring, image decoding, camera sets, and absence of privileged labels in policy fields.

```bash
bash collect_data.sh play_piano play_piano 0
```

## Motion speed and episode length

`tension_strap` and `bulb_tightening` accept `task_parameters.motion_speed_scale` from 1 to 3. The code default is 1, preserving legacy timing. This parameter changes motion timing while retaining the 120 Hz physics clock, 60 Hz policy/recording cadence, acceptance thresholds, and required force/torque holds. Strap motion accelerates free gripper closure and tapers back to its original controller near contact and the force target. Bulb motion accelerates only the unloaded wrist return and terminal opening/lift; approach, grasp, probing, thread travel, seating, and torque holds retain their original timing. Retiming contact motion can change how many regrasp cycles the feedback controller needs, so it is excluded from the bulb speed profile.

The standard `tension_strap.yml` profile selects scale 3. On paired seeds 0, 1, and 2, both scale 1 and scale 3 succeeded in all three episodes. Total physics ticks were 2833/2870/2911 before and 1458/1490/1594 after: the mean fell from 2871.3 to 1514 (47.3%). Mean policy ticks fell from 2807.3 to 1450, and saved policy frames from 1403 to 724. Both force stages still require three seconds of sustained physical acceptance. Comparing recorded end-effector pose and jaw width after normalized path-length alignment gave maximum position differences of 0.186/0.066/0.083 mm and orientation differences of 0.100/0.038/0.021 degrees. This is a three-seed regression result, not a large-scale success-rate estimate.

The standard `bulb_tightening.yml` profile also selects scale 3, restricted to unloaded motion. On common-success seeds 0 and 2, total physics ticks fell from 3755/4717 to 3169/3974: means of 4236 and 3571.5 (15.7% fewer). Mean policy ticks fell from 3712 to 3307.5 (10.9%), and saved policy frames from 1855.5 to 1653. Terminal opening/lift/review fell from 450 to 190 ticks; this is excluded from the policy reduction. Both pairs retained their full stage sequence, including four and six wrist-return/regrasp cycles respectively. After normalized pose-and-jaw path-length alignment, maximum position differences were 0.172/0.424 mm, orientation differences 0.606/4.515 degrees, and full jaw-width differences 0.528/3.634 mm. Physical torque, seating, damage, and hold gates were unchanged and passed in both pairs.

Across all attempted seeds 0, 1, and 2, the original bulb profile completed two of three episodes; seed 1 failed direction probing. The new profile completed all three, with seed 1 taking 4541 ticks. Its complete three-seed means were 3894.7 total ticks, 3630.7 policy ticks, and 1814.7 policy frames. The probing controller was not modified, and its estimate varied between runs; these few episodes do not establish a success-rate improvement. Speed comparisons above use only the common-success pair, rather than comparing means from different success cohorts.

For comparison, the unchanged `insert_USB` baseline succeeded on two of three attempted seeds (0 and 1 succeeded; 2 failed). Successful episodes averaged 493 total physics ticks, 435 policy ticks, and 217 saved policy observations. All measured runs used 120 Hz physics and recording every two ticks. These finite-seed measurements describe scripted demonstrations, not trained-policy success rates.

Measure collected episodes rather than `step_lim` or video length:

```bash
python scripts/summarize_episode_steps.py data/path/to/runs
```

The output reports attempts and successes, successful-episode means and sample counts, total physics ticks, preparation ticks, policy ticks, terminal review ticks, and saved policy frames. Terminal reveal movements after the bulb's physical verdict are excluded from policy timing. Saved frames are not physics ticks; changing recording frequency alone does not accelerate execution.

The measurements above were collected on 2026-10-10 with Isaac Sim 4.5, Isaac Lab 2.1.1, libuipc 0.9.0, and an RTX 4090. The related regression suite passed 137 tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_tactile_memory.py tests/test_blind_box.py tests/test_task_speed.py \
  tests/test_episode_steps.py tests/test_four_force_tasks.py tests/test_force_task_integration.py -q
```
