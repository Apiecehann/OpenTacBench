# ALOHA 2 arm source

The XML, included control files and six arm STL meshes come unmodified from
[Google DeepMind MuJoCo Menagerie's ALOHA model](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/aloha),
commit `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`.
The model describes ALOHA 2's ViperX 300S six-axis arms and uses the included
[BSD-3-Clause license](LICENSE), copyright Trossen Robotics.

`envs/robot/aloha_assets.py` extracts the left six-axis arm, removes its native
gripper and unused visual sites, and derives the single-arm MJCF and planner
URDF. The source XML here is an input to that derivation, not a standalone
installation of the complete bimanual Menagerie asset package.

Isaac Lab's original `MjcfConverter` builds the arm USD. The complete original
TacEx GelSight Mini gripper is mounted by a fixed joint, preserving internal
body, gel, camera and joint-relative transforms. Sensor configuration and
optical calibration are retained. Generated assets live in `.cache/aloha_gsmini`.

The derived assembly adds a 35 mm long, 20 mm radius mounting flange along
the original wrist's local X axis, with the Mini rotated 90 degrees about Y.
Its visible and collision geometry are present in the native USD and planner
description. This clears the Mini palm's rear collision surface from the
original wrist while keeping native self collision enabled. The benchmark
home pose is raised to leave room for the unchanged Mini fingertip offset.

The benchmark uses a stiff position servo like its original Franka carrier;
the ALOHA source joint ranges, axes, inertias and effort limits are retained.
This is a simulated ALOHA arm with a Mini gripper, not a hardware validation
of the original ALOHA system-identification controller.
