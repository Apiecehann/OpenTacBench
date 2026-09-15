"""Shared force-task timing, contact/image measurements, presentation and terminal recording."""
from __future__ import annotations

import cv2
import hashlib
import json
import math
import numpy as np
import os
import pickle
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


# Tension strap geometry

def tetra_faces(tets):
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]])
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse] == 1]

def write_tet_asset(path, points, tets, faces, color=(.12, .45, .72)):
    from pxr import Sdf, Usd, UsdGeom, Vt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/Object')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    mesh = UsdGeom.Mesh.Define(stage, '/Object/body')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([color])
    for name in ('tet_points', 'tet_surf_points'):
        mesh.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Double3Array).Set(
            Vt.Vec3dArray.FromNumpy(np.asarray(points, np.float64)))
    mesh.GetPrim().CreateAttribute('tet_indices', Sdf.ValueTypeNames.IntArray).Set(np.asarray(tets).ravel().tolist())
    mesh.GetPrim().CreateAttribute('tet_surf_indices', Sdf.ValueTypeNames.IntArray).Set(np.asarray(faces).ravel().tolist())
    stage.GetRootLayer().Export(str(path))
    return str(path)


# Chip fracture

FRAGMENT_SEAM_SCALE = np.array([0.96, 0.96, 1.0])

def capture_geometry_state(simulation):
    return [
        (np.asarray(body.geo_slot_list[0].geometry().positions().view()).copy(),
         np.asarray(body.geo_slot_list[0].geometry().transforms().view()).copy())
        for body in simulation.uipc_objects
    ]


# Tactile first utils

def estimate_affine_deformation(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map Q[i] ~ F @ P[i] + t (column convention).

    Solves [P | 1] @ M = Q (Nx4 @ 4x3) via lstsq and returns F = M[:3, :].T
    (3, 3) and t = M[3, :] (3,), the affine deformation gradient and
    translation. The fragile chip is a single affine element, so this recovers
    the true F to machine precision. Raises ValueError on mismatched / too-few
    (< 4) / non-finite points.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have identical shapes")
    n, dim = P.shape
    if dim != 3:
        raise ValueError("points must be 3D (N, 3)")
    if n < 4:
        raise ValueError("at least 4 points are required for an affine fit")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
        raise ValueError("P and Q must be finite")
    Pbar = np.hstack([P, np.ones((n, 1))])
    M, *_ = np.linalg.lstsq(Pbar, Q, rcond=None)  # (4, 3)
    return M[:3, :].T, M[3, :]


# Chip fracture

def replace_squeezed_chip(simulation, chip, fragments, standby_pose, previous_state, dt, *, seam_scale=FRAGMENT_SEAM_SCALE):
    from uipc import builtin, view
    from uipc.core import Engine, World
    import torch

    seam_scale = np.asarray(seam_scale, dtype=float)
    if seam_scale.shape != (3,) or np.any(seam_scale <= 0) or np.any(seam_scale > 1):
        raise ValueError("fragment seam_scale must contain three values in (0, 1]")
    parent_linear, parent_translation = estimate_affine_deformation(chip.origin_surf_pts, chip.vertices)
    current_state = capture_geometry_state(simulation)
    parent_index = next(index for index, body in enumerate(simulation.uipc_objects) if body is chip)
    parent_velocity = ((current_state[parent_index][1] - previous_state[parent_index][1]) / dt).reshape(4, 4)
    fragment_speeds = []
    fragment_ids = {id(fragment) for fragment in fragments}
    replacements = {}
    for fragment in fragments:
        center = fragment.origin_surf_pts.mean(axis=0)
        translation = parent_translation + parent_linear @ (center * (1.0 - seam_scale))
        transform = np.eye(4)
        transform[:3, :3] = parent_linear
        transform[:3, 3] = translation - parent_linear @ fragment.init_pose.p
        replacements[id(fragment)] = transform
    replacements[id(chip)] = standby_pose.to_transformation_matrix() @ np.linalg.inv(
        chip.init_pose.to_transformation_matrix()
    )
    temporary_velocities = []
    for index, body in enumerate(simulation.uipc_objects):
        geometry = body.geo_slot_list[0].geometry()
        affine = geometry.meta().find(builtin.backend_abd_body_offset) is not None
        collection = geometry.instances() if affine else geometry.vertices()
        current = current_state[index][1 if affine else 0]
        previous = previous_state[index][1 if affine else 0]
        velocity = (current - previous) / dt
        if id(body) in replacements:
            velocity = np.zeros_like(current)
            if body is not chip:
                center = body.origin_surf_pts.mean(axis=0)
                derivative = parent_velocity.copy()
                derivative[:3, 3] += parent_velocity[:3, :3] @ (
                    chip.init_pose.p + center * (1.0 - seam_scale) - body.init_pose.p)
                velocity[:] = derivative
                center_velocity = derivative[:3, :3] @ (body.init_pose.p + center * seam_scale) + derivative[:3, 3]
                fragment_speeds.append(float(np.linalg.norm(center_velocity)))
            transform = replacements[id(body)]
            view(geometry.transforms())[:] = transform
            view(geometry.instances().find(builtin.aim_transform))[:] = transform
            view(geometry.instances().find(builtin.is_constrained))[:] = int(body is chip)
            body.next_status = None
            body.next_mat = None
            body.next_pts = None
            body.next_mask = None
            if body is chip:
                body.next_status = 'set'
                body.next_mat = transform
            else:
                for slot in simulation.scene.geometries().find(body.geo_slot_list[0].id()):
                    fragment_geometry = slot.geometry()
                    points = np.asarray(fragment_geometry.positions().view())
                    view(fragment_geometry.positions())[:] = (
                        (points.reshape(-1, 3) - body.init_pose.p) * seam_scale
                        + body.init_pose.p
                    ).reshape(points.shape)
                    volume = fragment_geometry.instances().find('volume')
                    if volume is not None:
                        view(volume)[:] = np.asarray(volume.view()) * np.prod(seam_scale)
                anchor = torch.as_tensor(body.init_pose.p, device=body.init_vertex_pos.device)
                factor = torch.as_tensor(seam_scale, device=body.init_vertex_pos.device)
                body.init_vertex_pos = (body.init_vertex_pos - anchor) * factor + anchor
                body.origin_surf_pts = body.origin_surf_pts * seam_scale
        slot = collection.find('velocity')
        if slot is None:
            slot = collection.create('velocity', np.zeros((4, 4)) if affine else np.zeros((3, 1)))
            original_velocity = None
        else:
            original_velocity = np.asarray(slot.view()).copy()
        view(slot)[:] = velocity
        temporary_velocities.append((collection, slot, original_velocity))
    engine = Engine(simulation.cfg.device, simulation.cfg.workspace)
    world = World(engine)
    world.init(simulation.scene)
    if not world.is_valid():
        raise RuntimeError("fragment handoff rejected: rebuilt collision world is invalid")
    world.retrieve()
    simulation.world = world
    simulation.engine = engine
    simulation._contact_grad_cache = None
    for collection, slot, original in temporary_velocities:
        if original is None:
            collection.destroy('velocity')
        else:
            view(slot)[:] = original
    offsets = [0]
    for body in simulation.uipc_objects:
        geometry = body.geo_slot_list[0].geometry()
        offsets.append(int(geometry.meta().find(builtin.global_vertex_offset).view()[0]))
        body.global_system_id = len(offsets) - 1
        if getattr(body, '_data', None) is not None:
            body._data.update(dt)
    simulation._system_vertex_offsets['uipc::backend::cuda::GlobalVertexManager'] = offsets
    preserved = []
    for index, body in enumerate(simulation.uipc_objects):
        if id(body) in fragment_ids or body is chip:
            continue
        geometry = body.geo_slot_list[0].geometry()
        preserved.append(float(np.max(np.abs(np.asarray(geometry.positions().view()) - current_state[index][0]))))
    return {'mode': 'free_fragment_replacement', 'seam_scale': seam_scale.tolist(),
            'other_geometry_max_position_change_m': max(preserved, default=0.0),
            'fragment_velocity_mode': 'inherited_parent_affine_velocity_no_added_kick',
            'fragment_initial_speed_max_m_s': max(fragment_speeds, default=0.0),
            'seam_volume_ratio': float(np.prod(seam_scale))}


# Chip scene

def set_actor_visible(stage, actor, visible):
    for path in actor.cfg.visual_prim_paths:
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


# Bulb grasp control

def next_grip_qpos(depths_mm, current_qpos, target_depth_mm=27.2,
                   minimum_step_m=5e-6, maximum_step_m=1e-4):
    depths=np.asarray(depths_mm,dtype=float).reshape(-1)
    if depths.size!=2 or not np.all(np.isfinite(depths)) or not np.isfinite(current_qpos):
        raise ValueError("Bilateral finite tactile depths and jaw position are required")
    if not (0<minimum_step_m<=maximum_step_m and np.isfinite(target_depth_mm)):
        raise ValueError("Invalid bounded gripper step")
    if np.all(depths<target_depth_mm):
        return None
    # Continue toward the weaker pad. An already-loaded pad near its target
    # must not shrink all motion to nanometres for seconds.
    remaining=float(np.maximum(depths-target_depth_mm,0).max())
    step=float(np.clip(remaining*.0005,minimum_step_m,maximum_step_m))
    return max(0.,float(current_qpos)-step)


# Force task io

def image_features(reference, current):
    """Two-image Lucas-Kanade flow of dark marker centers, no simulator coordinates."""
    result = []
    details = {}
    for name in sorted(reference):
        a = cv2.cvtColor(reference[name], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(current[name], cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(a, 180, .02, 8, mask=np.uint8(a < 130)*255,
                                       blockSize=5)
        if points is None or len(points) < 16:
            raise ValueError('insufficient image features')
        moved, valid, error = cv2.calcOpticalFlowPyrLK(a, b, points, None,
            winSize=(21,21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None, winSize=(21,21), maxLevel=2)
        mask = valid.ravel().astype(bool) & valid_back.ravel().astype(bool)
        mask &= np.linalg.norm(back[:,0]-points[:,0],axis=1) < .3
        flow = (moved[:,0]-points[:,0])[mask]
        coords = points[:,0][mask]
        if len(flow) < 16:
            raise ValueError('insufficient matched image features')
        # Whole-pad and four-region vector means; preserves signed shear.
        values = list(np.median(flow,axis=0))
        for xsign,ysign in [(0,0),(0,1),(1,0),(1,1)]:
            sel=(coords[:,0] >= a.shape[1]/2)==bool(xsign)
            sel &= (coords[:,1] >= a.shape[0]/2)==bool(ysign)
            values.extend(np.median(flow[sel],axis=0) if sel.any() else [0.,0.])
        result.extend(values)
        details[name]={'matched':int(len(flow)), 'p90_px':float(np.percentile(np.linalg.norm(flow,axis=1),90)),
                       'median_xy_px':np.median(flow,axis=0).tolist()}
    return np.asarray(result,float), details

def rgb_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * (255 if image.max() <= 1 else 1), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)

def read_rgb(task):
    if task.last_render != task.step_count:
        task._update_render()
    obs = task._tactile_manager.get_observations(['rgb_marker'])
    return {name: rgb_array(val['rgb_marker']) for name, val in obs.items()}


# Force task scene

def _material(stage, path, color, roughness=.5, metallic=0.):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


# Chip review export

def record_terminal_observation(task, reason):
    last = None
    last_path = None
    if task.save_count:
        last_path = task.tmp_save_dir / f"{task.save_count - 1}.pkl"
        with last_path.open("rb") as file:
            last = pickle.load(file)
    last_step = None if last is None else int(last["step"])
    if last_step is not None and last_step > task.step_count:
        raise ValueError("Last saved observation is ahead of physical time")
    if not reason:
        raise ValueError("A terminal export requires an explicit reason")
    if task.cfg.video_frequency > 0 and task.cfg.video_frequency != task.cfg.save_frequency:
        raise ValueError("Review export requires matching data/video cadence")
    task._set_phase(task.PHASE_TERMINAL, terminal_reason=reason)
    task._update_render()
    observation = task._get_observations()
    appended = last_step is None or last_step < task.step_count
    if not appended:
        # Replace the final sample at the same physical instant. Never invent
        # a duplicate step or claim a terminal frame without saving one.
        old_phase = int(last["phase"]["id"])
        task.phase_saved_counts[old_phase] -= 1
        task.save_count -= 1
        previous_phase = None
        if task.save_count:
            with (task.tmp_save_dir / f"{task.save_count - 1}.pkl").open("rb") as file:
                previous_phase = int(pickle.load(file)["phase"]["id"])
        observation["phase"]["is_boundary"] = int(previous_phase != task.PHASE_TERMINAL)
        if old_phase == task.PHASE_POLICY and task.policy_start_saved_index == task.save_count:
            task.policy_start_saved_index = None
    task.save_observations(observation)
    if appended and task.cfg.video_frequency > 0:
        task.video_handler.write(task.get_frame_shot(observation))
    task.metadata["review_terminal_observation"] = {
        "physical_step": int(task.step_count),
        "previous_saved_step": last_step,
        "appended": bool(appended),
        "replaced_same_step": not appended,
        "physics_advanced": False,
        "plan_success_value": bool(task.plan_success),
        "reason": reason,
    }


# Final task contract

ACTION_REPEAT = 2

PHYSICS_HZ = 120

POLICY_HZ = 60

def configure_final_task(cfg, parameters, *, max_policy_seconds):
    if not math.isclose(float(cfg.sim.dt), 1.0 / PHYSICS_HZ, abs_tol=1e-12):
        raise ValueError('Final tasks require120Hz physics and60Hz policy timing')
    cfg.decimation = 1
    cfg.save_frequency = ACTION_REPEAT
    cfg.render_frequency = ACTION_REPEAT
    cfg.video_frequency = ACTION_REPEAT
    cfg.uniform_policy_recording = True
    cfg.policy_action_repeat = ACTION_REPEAT
    cfg.final_acceptance_contract = True
    # Observations remain compatible with Insert_USB; actual commands are a
    # separate audit sidecar because observed qpos is not the issued target.
    from pathlib import Path
    workspace=parameters.get("workspace")
    cfg.public_action_trace_path=str(Path(workspace)/"public_actions_{seed}.jsonl") if workspace else None
    cfg.absolute_joint_zero_velocity_targets = True
    live_state=parameters.get("live_action_joint_state",True)
    if not isinstance(live_state,bool):
        raise ValueError("live_action_joint_state must be a boolean task parameter")
    cfg.live_action_joint_state=live_state
    cfg.cache_actor_surfaces = bool(parameters.get("cache_actor_surfaces", False))
    cfg.reuse_same_step_tactile_depth = bool(parameters.get("reuse_same_step_tactile_depth", False))
    cfg.eval_start_delay_steps = 0
    cfg.final_policy_timeout_seconds = float(max_policy_seconds)
    cfg.step_lim = int(round(max_policy_seconds * POLICY_HZ))
    cfg.max_save_frames = max(int(cfg.max_save_frames), cfg.step_lim + 1200)

def execute_joint_target(task, arm=None, gripper=None, *, ticks=ACTION_REPEAT):
    """The expert uses exactly the public position-action path."""
    import torch
    manager = task._robot_manager
    q = manager.get_observations(['joint'])['joint'][:8].clone()
    if arm is not None:
        q[:7] = torch.as_tensor(arm, dtype=q.dtype, device=q.device).reshape(-1)
    if gripper is not None:
        q[7] = float(gripper)
    executed, success = task.take_action(q, action_type='qpos', force=True, action_repeat=int(ticks))
    return bool(executed and not task.check_early_stop() and task.plan_success), bool(success)


# Force task io

def world_contact_vertices(actor):
    """World points in contact-gradient indexing, including ABD instance motion."""
    from uipc import builtin
    points=np.asarray(actor.vertex_positions).reshape(-1,3)
    if getattr(actor,'actor_type',None)=='affine_body':
        transforms=actor.geo_slot_list[0].geometry().instances().find(builtin.transform)
        matrices=np.asarray(transforms.view()).reshape(-1,4,4)
        if len(matrices)!=1:
            raise ValueError('Force-task contact accounting expects one actor instance')
        points=points@matrices[0,:3,:3].T+matrices[0,:3,3]
    return points

def actor_contact_resultant(task,actor,mask,center=None):
    """Contact resultant on selected object vertices, excludes pad/backing self-contact."""
    from uipc import builtin
    idx,grad=task.uipc_sim.get_contact_gradient()
    idx=np.asarray(idx).reshape(-1)
    grad=np.asarray(grad).reshape(-1,3)
    start=int(actor.geo_slot_list[0].geometry().meta().find(builtin.global_vertex_offset).view()[0])
    points=world_contact_vertices(actor)
    selected=(idx>=start)&(idx<start+len(points))
    local=(idx[selected]-start).astype(int)
    forces=np.zeros_like(points)
    np.add.at(forces,local,-grad[selected]/task.uipc_sim.cfg.dt**2)
    center=np.zeros(3) if center is None else np.asarray(center)
    mask=np.asarray(mask,bool)
    return {'force_N':forces[mask].sum(axis=0).tolist(),
            'torque_Nm':np.cross(points[mask]-center,forces[mask]).sum(axis=0).tolist()}

def flow_rgb_features(reference,current):
    """Flow plus coarse photometric shape; still exactly four raw RGB images."""
    flow,details=image_features(reference,current)
    shape=[]
    for name in sorted(reference):
        a=cv2.GaussianBlur(reference[name].astype(np.float32),(0,0),4.)
        b=cv2.GaussianBlur(current[name].astype(np.float32),(0,0),4.)
        shape.extend(cv2.resize(b-a,(6,4),interpolation=cv2.INTER_AREA).ravel())
    return np.r_[flow,np.asarray(shape,float)],details

def predict_calibrated(model,features):
    x=np.asarray(features,float)
    if model.get('feature_transform','linear')=='linear_square':
        x=np.concatenate([x,x*x])
    weights=np.asarray(model['weights'],float)
    if x.shape!=weights.shape:
        raise ValueError('RGB calibration feature contract mismatch')
    return float(x@weights+model['bias'])

def task_parameters(*, require_nonnegative_seed=True):
    path = os.environ.get('VITAFORGE_TASK_CONFIG')
    if not path:
        return {}
    import yaml
    config=yaml.safe_load(Path(path).read_text())
    parameters=dict(config.get('task_parameters', {}))
    parameters.setdefault('physics_seed',int(config.get('start_seed',0)))
    if require_nonnegative_seed and parameters['physics_seed']<0:
        raise ValueError('Force tasks require an explicit nonnegative physics/start seed')
    return parameters

def tracked_flow_rgb_features(reference,current,tracker):
    """Persistent raw-image flow with the same coarse photometric features."""
    flow,details=tracker.update(current)
    shape=[]
    for name in sorted(reference):
        a=cv2.GaussianBlur(reference[name].astype(np.float32),(0,0),4.)
        b=cv2.GaussianBlur(current[name].astype(np.float32),(0,0),4.)
        shape.extend(cv2.resize(b-a,(6,4),interpolation=cv2.INTER_AREA).ravel())
    return np.r_[flow,np.asarray(shape,float)],details


# Force task parameters

def sample_physics(task, seed):
    rng=np.random.default_rng(int(seed))
    if task=='tension_strap':
        return {'modulus_mpa':float(rng.uniform(.25,.40))}
    if task=='bulb_tightening':
        return {'initial_depth_m':float(rng.uniform(.019,.021)),
                'seat_top_m':float(rng.uniform(.021,.0226)),
                'seat_modulus_mpa':float(rng.uniform(.08,.20))}
    raise ValueError(f'Unknown force task: {task}')


# Force task probe

def configure_gel(tactiles, parameters):
    modulus=parameters.get('gel_modulus_mpa')
    if modulus is None:
        return
    modulus=float(modulus)
    if not np.isfinite(modulus) or not .025<=modulus<=.20:
        raise ValueError("Diagnostic gel modulus must be 0.025--0.20 MPa")
    for cfg in tactiles:
        cfg.gelpad_cfg.constitution_cfg.youngs_modulus=modulus


# Force task scene

def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

def _cylinder(stage,path,center,radius,height,mat):
    mesh=UsdGeom.Cylinder.Define(stage,path)
    mesh.CreateRadiusAttr(radius)
    mesh.CreateHeightAttr(height)
    mesh.CreateAxisAttr('Z')
    mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(mesh.GetPrim(),mat)

def _rounded_box(stage, path, center, size, radius, mat):
    width, depth, height = size
    radius = min(radius, width*.45, depth*.45)
    bevel = min(height*.22, radius*.45, .0015)
    vertices = []
    for z, inset in [(-height/2, bevel), (-height/2+bevel, 0),
                      (height/2-bevel, 0), (height/2, bevel)]:
        rr = max(.0001, radius-inset)
        for cx,cy,angle in [(width/2-radius,depth/2-radius,0),
                           (-width/2+radius,depth/2-radius,90),
                           (-width/2+radius,-depth/2+radius,180),
                           (width/2-radius,-depth/2+radius,270)]:
            for a in np.linspace(angle, angle+90, 7):
                radians = math.radians(float(a))
                vertices.append((cx+rr*math.cos(radians), cy+rr*math.sin(radians), z))
    n = 28
    faces = [list(range(n-1,-1,-1)), list(range(3*n,4*n))]
    for ring in range(3):
        for i in range(n):
            j = (i+1)%n
            faces.append([ring*n+i,ring*n+j,(ring+1)*n+j,(ring+1)*n+i])
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,np.float32)))
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(mesh.GetPrim(),mat)
    return mesh

class ForceTaskScene:
    def __init__(self,task,kind):
        self.task=task
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for env_path in task.scene.env_prim_paths:
            root=env_path+'/force_task_presentation'
            UsdGeom.Xform.Define(stage,root)
            metal=_material(stage,root+'/Looks/Metal',
                            (.46,.49,.53) if kind=='strap' else (.33,.38,.42),
                            .48 if kind=='strap' else .28, .65 if kind=='strap' else .8)
            edge=_material(stage,root+'/Looks/DarkMetal',
                           (.055,.060,.068) if kind=='strap' else (.06,.085,.10),
                           .75 if kind=='strap' else .42, .15 if kind=='strap' else .55)
            rubber=_material(stage,root+'/Looks/Elastomer',
                             (.035,.12,.36) if kind=='strap' else (.025,.38,.32),
                             .78 if kind=='strap' else .56)
            grip=_material(stage,root+'/Looks/Grip',(.045,.15,.145),.67)
            # Both tasks inherit the base table and environment used by the chip task.
            if kind=='strap':
                _rounded_box(stage,root+'/anchor_base',(.55,0.,.006),(.096,.082,.010),.009,metal)
                for side in [-1,1]:
                    _rounded_box(stage,root+('/anchor_jaw_front' if side<0 else '/anchor_jaw_back'),(.55,side*.025,.020),
                                 (.057,.014,.024),.003,edge)
                for i,(x,y) in enumerate([(-.035,-.028),(.035,-.028),(-.035,.028),(.035,.028)]):
                    _cylinder(stage,root+f'/bolt_{i}',(.55+x,y,.012),.004,.004,edge)
                    _rounded_box(stage,root+f'/bolt_slot_{i}',(.55+x,y,.0141),
                                 (.0047,.0011,.0003),.0002,metal)
                actor=stage.GetPrimAtPath(env_path+'/elastic_strap')
                if actor.IsValid():
                    for prim in Usd.PrimRange(actor):
                        if prim.IsA(UsdGeom.Mesh):
                            mesh=UsdGeom.Mesh(prim)
                            _bind(prim,rubber)
                            if task.params.get('geometry') == 'bundle':
                                self.color_strap(mesh)
                                reader=UsdShade.Shader.Define(stage,root+'/Looks/Elastomer/Color')
                                reader.CreateIdAttr('UsdPrimvarReader_float3')
                                reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
                                reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
                                shader=UsdShade.Shader(stage.GetPrimAtPath(root+'/Looks/Elastomer/Shader'))
                                shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(),'result')
            else:
                _rounded_box(stage,root+'/socket_mount',(.55,0.,.001),(.125,.125,.002),.014,metal)


    def color_strap(self, mesh, rest_points=None):
        task=self.task
        points=(np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
                if rest_points is None else np.asarray(rest_points))
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int).reshape(-1,3)
        centers=points[faces].mean(axis=1)-task.base
        palette=np.asarray([[50,126,207],[253,230,64],[253,77,54],[55,58,62]],dtype=float)/255.
        palette=np.where(palette<=.04045,palette/12.92,((palette+.055)/1.055)**2.4)
        which=np.full(len(faces),3,dtype=int)
        band=(centers[:,2]>.01601)&(centers[:,2]<.13199)
        band_centers=np.asarray(task.params.get('band_centers_m',[-.031,0.,.031]))
        which[band]=np.abs(centers[band,0,None]-band_centers[None,:]).argmin(axis=1)
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(palette[which].astype(np.float32)))
        mesh.GetDisplayColorPrimvar().SetInterpolation('uniform')


# Force task tracking

class MarkerFlowTracker:
    def __init__(self, reference):
        self.reference={n:cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for n,im in reference.items()}
        self.points={}
        self.current={}
        for name,gray in self.reference.items():
            pts=cv2.goodFeaturesToTrack(gray,180,.02,8,mask=np.uint8(gray<130)*255,blockSize=5)
            if pts is None or len(pts)<16:
                raise ValueError("Insufficient raw marker features")
            self.points[name]=pts
            self.current[name]=pts.copy()
        self.updates=0

    def update(self, images):
        values=[];details={}
        for name in sorted(self.reference):
            a=self.reference[name];b=cv2.cvtColor(images[name],cv2.COLOR_RGB2GRAY)
            p=self.points[name]
            # Finer local search starts at the last correspondence, preserving
            # dot identity even when total travel exceeds half the grid spacing.
            q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,self.current[name].copy(),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(21,21),
                maxLevel=2 if self.updates==0 else 0,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
            back,valid,_=cv2.calcOpticalFlowPyrLK(b,a,q,p.copy(),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(21,21),maxLevel=0,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
            mask=ok.ravel().astype(bool)&valid.ravel().astype(bool)
            mask &= np.linalg.norm(back[:,0]-p[:,0],axis=1)<.3
            mask &= np.isfinite(q[:,0]).all(axis=1)
            if self.updates:
                mask &= np.linalg.norm(q[:,0]-self.current[name][:,0],axis=1)<4.
            self.current[name][mask]=q[mask]
            flow=(q[:,0]-p[:,0])[mask];coords=p[:,0][mask]
            if len(flow)<16:
                raise ValueError("Insufficient temporally consistent image correspondences")
            f=list(np.median(flow,axis=0))
            for xs,ys in [(0,0),(0,1),(1,0),(1,1)]:
                sel=((coords[:,0]>=a.shape[1]/2)==bool(xs))&((coords[:,1]>=a.shape[0]/2)==bool(ys))
                f.extend(np.median(flow[sel],axis=0) if sel.any() else [0.,0.])
            values.extend(f)
            details[name]={'matched':int(len(flow)),'p90_px':float(np.percentile(np.linalg.norm(flow,axis=1),90)),
                           'median_xy_px':np.median(flow,axis=0).tolist()}
        self.updates+=1
        return np.asarray(values,float),details


# Tension strap geometry

class ElasticResultant:
    """Exact gradient of the installed StableNeoHookean energy in metres/Pascals."""
    def __init__(self, rest, tets, mu, lame, energy_volume_scale=6.):
        self.rest = np.asarray(rest, float)
        self.tets = np.asarray(tets, int)
        dm = np.stack([self.rest[self.tets[:, i]] - self.rest[self.tets[:, 0]]
                       for i in (1,2,3)], axis=-1)
        self.volume = np.linalg.det(dm) / 6.
        if np.any(self.volume <= 0):
            raise ValueError('nonpositive tetrahedron volume')
        # Installed libuipc 0.9.0 FEM backend uses det(Dm), NOT det(Dm)/6,
        # in its elastic energy. Keep geometric volume separately for mass.
        self.energy_volume = self.volume * float(energy_volume_scale)
        self.dm_inv = np.linalg.inv(dm)
        self.mu = np.broadcast_to(np.asarray(mu, float), (len(tets),))
        self.lame = np.broadcast_to(np.asarray(lame, float), (len(tets),))

    def measure(self, points, upper_mask, center=None):
        pts = np.asarray(points, float)
        ds = np.stack([pts[self.tets[:, i]] - pts[self.tets[:, 0]] for i in (1,2,3)], axis=-1)
        f = ds @ self.dm_inv
        j = np.linalg.det(f)
        if not np.all(np.isfinite(f)) or np.any(j <= 0):
            raise ValueError('invalid deformed tetrahedra')
        cof = j[:, None, None] * np.linalg.inv(f).transpose(0,2,1)
        p = self.mu[:,None,None] * f + (self.lame*(j-1.)-self.mu)[:,None,None]*cof
        g = self.energy_volume[:,None,None] * (p @ self.dm_inv.transpose(0,2,1))
        local = np.concatenate([-g.sum(axis=2)[:,:,None], g], axis=2).transpose(0,2,1)
        selected = np.asarray(upper_mask, bool)[self.tets]
        force = -(local * selected[:,:,None]).sum(axis=(0,1))
        center = np.zeros(3) if center is None else np.asarray(center)
        torque = -np.cross(pts[self.tets]-center, local*selected[:,:,None]).sum(axis=(0,1))
        stretch = np.linalg.svd(f, compute_uv=False)
        return {'elastic_force_N': force, 'elastic_torque_Nm': torque, 'maximum_stretch': float(stretch.max()),
                'minimum_jacobian': float(j.min())}


# Final task contract

FINAL_TASKS = frozenset(("grasp_fragile_chip","bulb_tightening","tension_strap","wipe_vase"))

def resolve_policy_action_repeat(task_name, training_config):
    repeat=training_config.get("action_repeat", ACTION_REPEAT if task_name in FINAL_TASKS else 1)
    if isinstance(repeat,bool) or not isinstance(repeat,int) or repeat<1:
        raise ValueError("Policy action_repeat must be a positive integer")
    if task_name in FINAL_TASKS and repeat!=ACTION_REPEAT:
        raise ValueError("FinalAcceptance tasks use 60Hz data and action_repeat=2; an older checkpoint cadence requires separate validation")
    return repeat

def timing_context(task):
    cfg = task.cfg
    physical_dt = float(cfg.sim.dt)
    environment_dt = physical_dt * int(cfg.decimation)
    repeat = int(getattr(cfg, 'policy_action_repeat', 1))
    result = {
        'physics_dt_s': physical_dt,
        'decimation': int(cfg.decimation),
        'save_frequency': int(cfg.save_frequency),
        'policy_observation_dt_s': environment_dt * int(cfg.save_frequency),
        'policy_action_repeat': repeat,
        'policy_action_dt_s': environment_dt * repeat,
        'video_frequency': int(cfg.video_frequency),
        'video_fps': float(task.video_handler.fps),
        'joint_position_units': 'arm radians, fingers meters',
        'joint_action_semantics': 'absolute qpos, first8 of recorded9 joints',
        'joint_action_start_state': 'live_physx' if getattr(cfg,'live_action_joint_state',False) else 'observation_cache',
        'quaternion_order': 'wxyz',
        'position_units': 'meters',
        'pose_frame': 'world',
        'zero_velocity_targets': bool(getattr(cfg, 'absolute_joint_zero_velocity_targets', False)),
        'actor_surface_cache_enabled': bool(getattr(cfg, 'cache_actor_surfaces', False)),
        'same_step_tactile_depth_reuse': bool(getattr(cfg, 'reuse_same_step_tactile_depth', False)),
    }
    tactiles = getattr(getattr(cfg, 'robot', None), 'tactiles', [])
    result['gel_modulus_mpa'] = {
        str(t.name): float(t.gelpad_cfg.constitution_cfg.youngs_modulus)
        for t in tactiles if getattr(t, 'gelpad_cfg', None) is not None
    }
    return result


# Public action recording

def append_public_action(task, target, initial, before_step, phase_id, requested_repeat):
    cfg=task.cfg
    path=getattr(cfg,"public_action_trace_path",None)
    if not path or task.mode!="collect" or phase_id!=task.PHASE_POLICY:
        return
    after=int(task.step_count)
    if after<=before_step:return
    def array(value):
        if hasattr(value,"detach"):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float).reshape(-1)
    q=array(target);q0=array(initial)
    if q.shape!=(8,) or q0.shape!=(8,) or not np.isfinite(np.r_[q,q0]).all():
        raise ValueError("Public action evidence must contain eight finite joint targets")
    row=dict(before_step=int(before_step),after_step=after,
             requested_repeat=int(requested_repeat),qpos=q.tolist(),initial_qpos=q0.tolist())
    path=Path(str(path).format(seed=int(getattr(cfg,"seed",0))))
    path.parent.mkdir(parents=True,exist_ok=True)
    if getattr(task,"_public_action_hash_path",None)!=str(path):
        if path.exists() and path.stat().st_size:
            raise FileExistsError("Preserve existing original-action evidence; use a new output")
        task._public_action_hash_path=str(path)
        task._public_action_hasher=hashlib.sha256()
        task._public_action_count=0
    encoded=(json.dumps(row,separators=(",",":"))+"\n").encode()
    with path.open("ab") as stream:stream.write(encoded)
    task._public_action_hasher.update(encoded);task._public_action_count+=1
    task.metadata["public_action_trace"]=dict(path=str(path),
        sha256=task._public_action_hasher.hexdigest(),calls=task._public_action_count,last_physical_step=after,
        meaning="original public absolute qpos targets; observation HDF remains unchanged",
        zero_velocity_targets=True)

def original_action_schedule(steps,joints,start,command_path,*,terminal_step=None,terminal_joint=None):
    """Merge original public calls with observation targets only in untraced setup gaps.

    The full schedule is resolved before simulation; no contact, force, material
    or success feedback enters command generation. Each source segment is named.
    """
    steps=np.asarray(steps,dtype=int);joints=np.asarray(joints,dtype=float)
    if joints.shape!=(len(steps),8) or not np.isfinite(joints).all():
        raise ValueError("Recorded observations must contain finite eight-joint rows")
    if len(steps)<2 or np.any(np.diff(steps)<=0):raise ValueError("Invalid observation clock")
    path=Path(command_path);raw=path.read_bytes()
    commands=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if not commands:raise ValueError("Original public action trace is empty")
    cursor=int(start);index=0;result=[]
    def add(end,target,kind,repeat=None):
        nonlocal cursor
        target=np.asarray(target,dtype=float)
        if target.shape!=(8,) or not np.isfinite(target).all():raise ValueError("Invalid original target")
        ticks=int(end-cursor) if repeat is None else int(repeat)
        if not 1<=ticks<=40:raise ValueError("Action evidence has an invalid interval")
        if end<=cursor or end-cursor>ticks:raise ValueError("Action evidence clock is inconsistent")
        result.append(dict(before_source_step=cursor,source_step=int(end),
            requested_physics_ticks=ticks,target=target.tolist(),source=kind))
        cursor=int(end)
    for row in commands:
        begin=int(row["before_step"]);end=int(row["after_step"]);repeat=int(row["requested_repeat"])
        if begin<int(start):raise ValueError("Action trace contains preparation outside POLICY")
        if begin<cursor:raise ValueError("Original public calls overlap")
        while index<len(steps) and steps[index]<=begin:
            if steps[index]>cursor:add(int(steps[index]),joints[index],"recorded_setup_joint_state")
            index+=1
        if begin>cursor:
            if begin-cursor>2:raise ValueError("Missing setup boundary observation")
            add(begin,row["initial_qpos"],"recorded_precommand_joint_state")
        add(end,row["qpos"],"original_public_qpos",repeat)
        while index<len(steps) and steps[index]<=cursor:index+=1
    while index<len(steps):
        if steps[index]>cursor:add(int(steps[index]),joints[index],"recorded_setup_joint_state")
        index+=1
    if terminal_step is not None and int(terminal_step)>cursor:
        add(int(terminal_step),terminal_joint,"recorded_terminal_joint_state")
    return result,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
        original_public_calls=len(commands),
        recorded_state_gap_calls=sum(row["source"]!="original_public_qpos" for row in result),
        uses_feedback_corrections=False)

