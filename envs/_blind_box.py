"""Geometry and retrieval contract for the open-top tactile search box."""
from dataclasses import dataclass

import numpy as np

from ._tactile_memory import box_mesh


CENTER = np.array([.52, 0.])
FLOOR_Z = .016
RIM_Z = .197
OBJECT_SIZE = (.050, .050, .060)


def sample_object_pose(rng):
    """Continuous random placement, with clearance for the object and jaws."""
    xy = CENTER + rng.uniform([-.028, -.052], [.028, .052])
    yaw = float(rng.uniform(-np.pi, np.pi))
    return np.r_[xy, FLOOR_Z + .001], np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])


def object_mesh():
    points, tets = box_mesh(OBJECT_SIZE, (4, 4, 4))
    # Round all edges of the block, retaining a positive-volume solid mesh.
    half = np.array(OBJECT_SIZE)/2
    centered = points - [0., 0., half[2]]
    core = np.clip(centered, -half+.006, half-.006)
    offset = centered-core
    lengths = np.linalg.norm(offset, axis=1, keepdims=True)
    centered = core + offset * np.minimum(1., .006/np.maximum(lengths, 1e-12))
    return centered + [0., 0., half[2]], tets


def scan_path():
    """One nonrepeating serpentine sweep, independent of the sampled object."""
    points = []
    for row, y in enumerate((-.045, 0., .045)):
        xs = np.linspace(.445, .600, 63)
        if row % 2:
            xs = xs[::-1]
        if points:
            points.extend([np.array([xs[0],v,.047])
                           for v in np.linspace(points[-1][1],y,19)[1:]])
        points.extend(np.array([x,y,.047]) for x in xs)
    return points


def grasp_yaw_delta(contact_points, jaw_axis):
    """Align square faces from opposing corner contacts, without object pose.

    A long contact chord identifies a diagonal pinch. The nearest face normal
    is approximately 45 degrees from that chord. Broad face contacts have a
    shorter chord and need no rotation.
    """
    chord = np.asarray(contact_points)[1,:2]-np.asarray(contact_points)[0,:2]
    if np.linalg.norm(chord) < .061:
        return 0.
    diagonal = np.arctan2(chord[1],chord[0])
    current = np.arctan2(jaw_axis[1],jaw_axis[0])
    candidates = diagonal+np.array([-.25,.25])*np.pi
    delta = (candidates-current+np.pi/2)%np.pi-np.pi/2
    chosen = float(delta[np.argmin(np.abs(delta))])
    return chosen if abs(chosen) > np.deg2rad(8.) else 0.


@dataclass
class Retrieval:
    """Success needs the entire object clear of the rim and a sustained grasp."""
    held_steps: int = 0
    required_steps: int = 18

    def advance(self, object_vertices, gripper_center, bilateral_contact):
        vertices = np.asarray(object_vertices)
        center = np.asarray(gripper_center)
        valid = (vertices.ndim == 2 and vertices.shape[1:] == (3,)
                 and len(vertices) > 0 and np.isfinite(vertices).all()
                 and np.isfinite(center).all())
        held = (valid and bilateral_contact and vertices[:, 2].min() > RIM_Z+.015
                and np.linalg.norm(vertices.mean(axis=0)-center) < .06)
        self.held_steps = self.held_steps+1 if held else 0

    @property
    def success(self):
        return self.held_steps >= self.required_steps


def build_case(task):
    """An entirely open top, solid collision walls, and rounded outer trim."""
    import omni.usd
    from pxr import UsdGeom
    from ._force_task_utils import _material, _rounded_box

    # Sizes use a bottom-centred origin, matching the tetrahedral box helper.
    parts = [
        ('floor', (.320, .380, .010), (.52, 0., .006)),
        ('front', (.012, .380, .180), (.366, 0., .017)),
        ('back', (.012, .380, .180), (.674, 0., .017)),
        ('left', (.296, .012, .180), (.52, -.184, .017)),
        ('right', (.296, .012, .180), (.52, .184, .017)),
    ]
    task.case_actors = []
    contacts = task.uipc_sim.scene.contact_tabular()
    shell = contacts.create('blind_box_shell')
    resistance = task.cfg.uipc_sim.contact.default_contact_resistance*1e9
    contacts.insert(shell, shell, friction_rate=.4, resistance=resistance, enable=False)
    for name, size, position in parts:
        actor = task.add_box('box_'+name, size, position, fixed=True, color=(.045,.09,.13))
        actor.set_physics_mesh_visible(False)
        for mesh in actor.uipc_meshes:
            shell.apply_to(mesh)
        task.case_actors.append(actor)

    stage = omni.usd.get_context().get_stage()
    for env in task.scene.env_prim_paths:
        root = env+'/blind_box_case'
        UsdGeom.Xform.Define(stage, root)
        blue = _material(stage, root+'/Looks/Case', (.055,.12,.17), .40, .12)
        charcoal = _material(stage, root+'/Looks/Charcoal', (.018,.024,.030), .85)
        metal = _material(stage, root+'/Looks/Metal', (.38,.42,.46), .28, .8)
        for name, size, position in parts:
            center = np.asarray(position)+[0,0,size[2]/2]
            _rounded_box(stage, root+'/'+name, tuple(center), size, .005, blue)
        _rounded_box(stage, root+'/liner', (.52,0,.0161), (.292,.352,.0002), .006, charcoal)
        for name, center, size in [
            ('rim_front', (.366,0,.195), (.013,.380,.004)),
            ('rim_back', (.674,0,.195), (.013,.380,.004)),
            ('rim_left', (.52,-.184,.195), (.296,.013,.004)),
            ('rim_right', (.52,.184,.195), (.296,.013,.004)),
        ]:
            _rounded_box(stage, root+'/'+name, center, size, .001, charcoal)
        for side in (-1,1):
            label = 'left' if side < 0 else 'right'
            # Flush side grips stay clear of the open working volume.
            _rounded_box(stage,root+'/handle_'+label,(.52,side*.1905,.133),(.095,.003,.029),.008,metal)
            _rounded_box(stage,root+'/handle_inset_'+label,(.52,side*.1925,.133),(.071,.002,.014),.005,charcoal)
        for i, (x,y) in enumerate(((.389,-.157),(.389,.157),(.651,-.157),(.651,.157))):
            _rounded_box(stage,root+'/foot_'+str(i),(x,y,.004),(.030,.030,.008),.006,charcoal)
