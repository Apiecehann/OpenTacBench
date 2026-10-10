"""Search an open-top box by touch and retrieve one randomly placed object."""
import numpy as np
from transforms3d.quaternions import mat2quat, axangle2quat, qmult

from ._base_task import BaseTask, Pose, configclass
from ._memory_task import MemoryTask, MemoryTaskCfg
from ._blind_box import RIM_Z, Retrieval, build_case, grasp_yaw_delta, object_mesh, sample_object_pose, scan_path

TASK_INSTRUCTION = ("Reach into the open box, scan by touch to find the object, "
                    "then grasp it and lift it out of the box.")


def _box_cameras():
    cameras = [camera for camera in MemoryTaskCfg().cameras if camera.name == 'head']
    eye, aim = np.array([.10,-.82,.34]), np.array([.52,0.,.12])
    z = (eye-aim)/np.linalg.norm(eye-aim)
    x = np.cross([0.,0.,1.],z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    cameras[0].offset.pos = tuple(eye)
    cameras[0].offset.rot = tuple(mat2quat(np.stack([x,y,z],axis=1)))
    cameras[0].offset.convention = 'opengl'
    cameras[0].width, cameras[0].height = 640, 480
    cameras[0].spawn.focal_length = 2.5
    cameras[0].spawn.horizontal_aperture = 2.4
    return cameras


@configclass
class TaskCfg(MemoryTaskCfg):
    step_lim = 600  # Public 60 Hz actions; 1200 physics ticks.
    memory_timeout_seconds = 10.0
    cameras = _box_cameras()
    video_size = (880, 480)


class Task(MemoryTask):
    def _setup_base_scene(self):
        # Start above the rim, before UIPC initializes collision geometry.
        joints = self.cfg.robot.robot.init_state.joint_pos
        joints.update(panda_joint2=-.4, panda_joint4=-2.2, panda_joint6=1.8)
        super()._setup_base_scene()

    def create_actors(self):
        build_case(self)
        self.object = self.add_mesh('hidden_object', *object_mesh(), (.52,0,.017),
                                    color=(.72,.33,.065))
        self.retrieval = Retrieval()

    def _reset_actors(self):
        self.monitor = False
        self.failure = ''
        self.retrieval = Retrieval()
        position, rotation = sample_object_pose(self.rng)
        self.object.set_pose(Pose(position, rotation))
        # Ground truth is used only for reset and scoring, never search decisions.
        self.metadata['blind_pick_ground_truth'] = dict(position=position.tolist(),
            rotation=rotation.tolist(), object_count=1)
        self.metadata['blind_pick_search'] = []
        self.metadata['blind_pick_trace'] = []

    def _release_reset_constraints(self):
        self.object.remove_animate(force=True)

    def pre_move(self):
        self.monitor = True
        self.metadata['blind_pick_geometry'] = dict(
            floor_top_m=float(self.case_actors[0].vertices[:,2].max()),
            wall_tops_m=[float(actor.vertices[:,2].max()) for actor in self.case_actors[1:]],
            object_start_min_z_m=float(self.object.vertices[:,2].min()))

    def _get_observations(self):
        # Keep the external scene visible: the enclosure is an actual occluder.
        obs = BaseTask._get_observations(self)
        if self.policy_start_step is not None:
            obs['phase']['policy_step'] = self.step_count-self.policy_start_step
        obs['actor'] = {}
        obs['atom'] = {'id': 0, 'tag': 'manipulate'}
        return obs

    def _step(self, is_save=True):
        before = getattr(self, 'step_count', 0)
        BaseTask._step(self, is_save)
        if self.policy_start_step is None or self.step_count == before:
            return
        vertices = self.object.vertices
        center = self._robot_manager.get_gripper_center_pose().p
        # Below the rim, retrieval cannot pass regardless of contact. Avoid an
        # extra tactile render on those odd ticks; search still samples at every
        # control update, and recording retains its normal 60 Hz cadence.
        bilateral = False
        if vertices[:,2].min() > RIM_Z+.015 or self.step_count % 12 == 0:
            depth = self._tactile_manager.get_min_depth().detach().cpu().numpy()
            bilateral = bool(np.isfinite(depth).all() and np.all(depth < 28.0))
        self.retrieval.advance(vertices, center, bilateral)
        if self.step_count % 12 == 0:
            contacts = self._pad_contacts()
            self.metadata['blind_pick_trace'].append(dict(step=self.step_count,
                object_center=vertices.mean(axis=0).tolist(),gripper_center=center.tolist(),
                depth_mm=depth.tolist(),opening=self._robot_manager.get_gripper_qpos(),
                object_rotation=self.object.get_pose().q.tolist(),
                contacts=[None if item is None else dict(force_N=item[0],patch_m=item[1].tolist(),
                    direction=item[2].tolist()) for item in contacts]))
        if self.policy_start_step is not None:
            if (self.step_count-self.policy_start_step)*self.cfg.sim.dt > self.cfg.memory_timeout_seconds:
                self.failure = self.failure or 'search_timeout'
        if self.failure:
            self.metadata['blind_pick_failure'] = self.failure
            self.plan_success = False

    def _close_on_contact(self):
        from ._force_task_utils import next_grip_qpos
        manager = self._robot_manager
        sensors = list(self._tactile_manager.tactiles.values())
        axis = sensors[1].get_attach_pose().p-sensors[0].get_attach_pose().p
        axis[2] = 0.
        axis /= np.linalg.norm(axis)
        alignments = 0
        oriented = False
        for _ in range(140):
            depths = self._tactile_manager.get_min_depth().detach().cpu().numpy()
            current = manager.get_gripper_qpos()
            center = manager.get_gripper_center_pose().p
            contacts = self._pad_contacts()
            touched = (depths < 28.2) | np.array([item is not None for item in contacts])
            unilateral = touched.sum() == 1
            if len(contacts) == 2 and all(item is not None for item in contacts):
                midpoint = (contacts[0][1]+contacts[1][1])*.5
                error = midpoint[:2]-center[:2]
                yaw_delta = 0. if oriented else grasp_yaw_delta(
                    np.array([contacts[0][1],contacts[1][1]]),axis)
                if np.linalg.norm(error) > .003 or yaw_delta:
                    if alignments >= 3 or np.linalg.norm(error) > .030:
                        return False
                    # Loaded pads stick to the block: sliding the closed hand
                    # drags the block instead of centring it. Remember the two
                    # contact locations, unload, then reposition and reclose.
                    target_center = np.r_[midpoint[:2],center[2]]
                    self.metadata.setdefault('blind_pick_alignment',[]).append(dict(
                        step=self.step_count,center=center.tolist(),
                        contact_midpoint=midpoint.tolist(),target=target_center.tolist(),
                        yaw_delta_deg=float(np.rad2deg(yaw_delta))))
                    count = max(1,int(np.ceil((.039-current)/.001)))
                    for opening in np.linspace(current,.039,count+1)[1:]:
                        manager.set_gripper(float(opening),force=True)
                        self._step(); self._step()
                        if not self.plan_success:
                            return False
                    self._step(); self._step()
                    if yaw_delta:
                        if not self._line_to(np.r_[center[:2],.115],spacing=.003):
                            return False
                        rotation = qmult(axangle2quat([0.,0.,1.],yaw_delta),
                                         manager.get_gripper_center_pose().q)
                        pose = Pose(np.r_[midpoint[:2],.115],rotation)
                        self.move(self.atom.move_to_pose(manager.gripper_center_to_ee(pose)),
                                  tag='manipulate',delay=False,time_dilation_factor=.8)
                        if not self.plan_success or not self._line_to(target_center,spacing=.003):
                            return False
                        axis = sensors[1].get_attach_pose().p-sensors[0].get_attach_pose().p
                        axis[2] = 0.
                        axis /= np.linalg.norm(axis)
                        oriented = True
                    elif not self._line_to(target_center):
                        return False
                    alignments += 1
                    continue
            target = next_grip_qpos(depths, current, target_depth_mm=27.9,
                maximum_step_m=.0003 if unilateral else (.0004 if not touched.any() else .0001))
            if target is None:
                return .008 < current < .0385
            if target >= current or not self.plan_success:
                return False
            manager.set_gripper(target, force=True)
            shift = 0.
            if unilateral:
                # Move the hand while closing so the loaded jaw stays still.
                # The opposite jaw approaches at twice the single-jaw speed,
                # without dragging the object sideways across the box floor.
                direction = 1. if touched[1] else -1.
                shift = direction*(current-target)
            if shift:
                if not self.servo_center(center+shift*axis):
                    return False
            else:
                self._step(); self._step()
        return False

    def _pad_contacts(self):
        # Use contact forces and positions on our own tactile pads. Edge
        # contacts are physical contacts too, even before the optical depth
        # camera sees the object inside its smaller field of view.
        indices, gradient = self.uipc_sim.get_contact_gradient()
        candidates = []
        for sensor in self._tactile_manager.tactiles.values():
            pad = sensor.gelpad
            points = pad.vertex_positions
            start = int(pad.geo_slot_list[0].geometry().meta().find('global_vertex_offset').view()[0])
            mask = (indices >= start) & (indices < start+len(points))
            inward = gradient[mask] / self.cfg.sim.dt**2
            weights = np.linalg.norm(inward, axis=1)
            total = float(weights.sum())
            normal = inward.sum(axis=0)[:2]
            if total < .08 or np.linalg.norm(normal) < .05:
                candidates.append(None)
                continue
            patch = np.average(points[indices[mask]-start], axis=0, weights=weights)
            direction = normal/np.linalg.norm(normal)
            candidates.append((total, patch, direction))
        return candidates

    def _locate_contact(self):
        # Known object size, unknown pose: start half a width behind the
        # touched face and finish centring through bilateral touch.
        contacts = [item for item in self._pad_contacts() if item is not None]
        if not contacts:
            return None
        load, patch, direction = max(contacts,key=lambda item:item[0])
        return load, patch[:2]+.025*direction, direction, patch

    def _lift_object(self):
        from ._force_task_utils import next_grip_qpos
        manager = self._robot_manager
        position = manager.get_gripper_center_pose().p.copy()
        goal_z = RIM_Z+.080
        held_updates = 0
        for _ in range(190):
            depths = self._tactile_manager.get_min_depth().detach().cpu().numpy()
            if not np.isfinite(depths).all() or np.min(depths) > 29.:
                return False
            target = next_grip_qpos(depths,manager.get_gripper_qpos(),
                target_depth_mm=27.8,maximum_step_m=.00015)
            if target is not None:
                if target < .008:
                    return False
                manager.set_gripper(target,force=True)
            # Preserve the tactile preload as the pads relax under vertical
            # shear. Pause ascent if either pad unloads; never lift open-loop.
            if np.max(depths) < 28.05:
                position[2] = min(goal_z,position[2]+.003)
            if not self.servo_center(position):
                return False
            held_updates = held_updates+1 if position[2] == goal_z else 0
            if held_updates >= 12 and self.check_success():
                return True
        return False

    def _play_once(self):
        # Sweep until tactile contact, then close. No object pose, identity or
        # camera feature enters this search. The path index remembers free space.
        path = scan_path()
        self.move(self.atom.open_gripper(1.), tag='manipulate', delay=False)
        self.move_center(np.r_[path[0][:2], RIM_Z+.060])
        if not self._line_to(path[0], spacing=.003):
            return
        found = False
        attempted = []
        recovering = False
        for index, point in enumerate(path):
            if self.failure or not self.plan_success:
                break
            if recovering:
                if any(np.linalg.norm(point[:2]-old) < .02 for old in attempted):
                    continue
                if not self._line_to(np.r_[point[:2],.115], spacing=.003):
                    break
                recovering = False
            if not self._line_to(point, spacing=.003):
                break
            contact_patch = self._locate_contact()
            if contact_patch is None:
                continue
            if any(np.linalg.norm(point[:2]-old) < .02 for old in attempted):
                continue
            attempted.append(point[:2].copy())
            load, xy, inward, patch = contact_patch
            depth = self._tactile_manager.get_min_depth().detach().cpu().numpy()
            center = self._robot_manager.get_gripper_center_pose().p
            # Release first contact, clear the object, then return to the
            # remembered location. No further blind shove along the scan row.
            retreat = center-np.r_[.003*inward,0.]
            grasp = np.r_[xy,.047]
            if not (self._line_to(retreat) and
                    self._line_to(np.r_[retreat[:2],.115], spacing=.003) and
                    self._line_to(np.r_[xy,.115], spacing=.003) and
                    self._line_to(grasp, spacing=.003)):
                break
            contact = self._close_on_contact()
            self.metadata['blind_pick_search'].append(dict(scan_index=index,
                position=grasp.tolist(),bilateral=contact,first_contact_depth_mm=depth.tolist(),
                contact_force_N=load,contact_patch_m=patch.tolist(),inward_direction=inward.tolist(),
                grasp_depth_mm=self._tactile_manager.get_min_depth().detach().cpu().tolist(),
                opening=self._robot_manager.get_gripper_qpos(),step=self.step_count))
            if contact:
                found = self._lift_object()
                break
            self.move(self.atom.open_gripper(1.), tag='manipulate', delay=False)
            if len(attempted) >= 6:
                break
            # Resume from above after a missed grasp. Moving straight back to
            # the next low scan point can shove the displaced object sideways.
            center = self._robot_manager.get_gripper_center_pose().p
            if not self._line_to(np.r_[center[:2],.115], spacing=.003):
                break
            recovering = True
        if not found:
            self.failure = self.failure or 'object_not_retrieved'
        vertices = self.object.vertices
        center = self._robot_manager.get_gripper_center_pose().p
        self.metadata['blind_pick_result'] = dict(success=self.check_success(),failure=self.failure,
            probes=len(self.metadata['blind_pick_search']),
            held_steps=self.retrieval.held_steps,required_held_steps=self.retrieval.required_steps,
            object_min_z_m=float(vertices[:,2].min()),rim_z_m=RIM_Z,
            object_hand_distance_m=float(np.linalg.norm(vertices.mean(axis=0)-center)),
            bilateral_depth_mm=self._tactile_manager.get_min_depth().detach().cpu().tolist(),
            policy_steps=self.step_count-self.policy_start_step)

    def check_success(self):
        return bool(self.retrieval.success and not self.failure and self.plan_success)
