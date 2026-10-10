"""Shared physical cue acquisition for the compact tactile memory tasks."""
from pathlib import Path
from itertools import product

import numpy as np

from ._base_task import BaseTask, BaseTaskCfg, Pose, UipcObjectCfg, configclass, torch
from ._force_task_utils import tetra_faces, write_tet_asset
from ._tactile_memory import CueMemory, box_mesh, cue_mesh


@configclass
class MemoryTaskCfg(BaseTaskCfg):
    step_lim = 1200
    max_save_frames = 1400
    save_frequency = 2
    render_frequency = 2
    video_frequency = 2
    save_pre_move = True
    uniform_policy_recording = True
    final_acceptance_contract = True
    policy_action_repeat = 2
    absolute_joint_zero_velocity_targets = True
    live_action_joint_state = True
    eval_start_delay_steps = 0
    memory_timeout_seconds = 20.0


class MemoryTask(BaseTask):
    cue_position = (.38, -.12, .005)

    def __init__(self, cfg, **kwargs):
        self.failure = ""
        self.cue_kind = 0
        self.memory = CueMemory()
        self.monitor = False
        self.work = Path(cfg.save_dir).resolve() / "work" / "geometry"
        self.work.mkdir(parents=True, exist_ok=True)
        cfg.seed = None
        cfg.obs_data_type = dict(cfg.obs_data_type)
        cfg.obs_data_type.pop("actor", None)
        for camera in cfg.cameras:
            camera.update_latest_camera_pose = True
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, **kwargs)

    def add_mesh(self, name, points, tets, position, *, fixed=False, modulus=None, color=(.8,.8,.75)):
        path = write_tet_asset(self.work / (name + ".usda"), points, tets, tetra_faces(tets), color)
        material = None if modulus is None else UipcObjectCfg.StableNeoHookeanCfg(
            youngs_modulus=modulus, poisson_rate=.40)
        return self._actor_manager.add_from_usd_file(
            name=name, asset_path=path, pose=Pose(position, [1.,0.,0.,0.]),
            constitution_cfg=material, density=700., keep_constrained=fixed)

    def add_box(self, name, size, position, **kwargs):
        actor = self.add_mesh(name, *box_mesh(size), position, **kwargs)
        if kwargs.get('fixed'):
            from uipc import view
            # keep_constrained only preserves an existing reset animation;
            # static fixtures have none. Pin their rigid transforms explicitly
            # so collision panels stay aligned with the visual shells.
            for mesh in actor.uipc_meshes:
                view(mesh.instances().find('is_fixed'))[:] = 1
        return actor

    def create_cues(self):
        self.cues = [self.add_mesh(f"cue_{i}", *cue_mesh(i),
                     self.cue_position if i == 0 else (.38,-1.,.005), fixed=True)
                     for i in range(2)]
        self.cues[1].set_physics_mesh_visible(False)
        # These are alternative versions of ONE reference. During reset they
        # exchange the display and parking positions; they must pass each other.
        contacts = self.uipc_sim.scene.contact_tabular()
        variants = contacts.create('reference_variants')
        contacts.insert(variants,variants,friction_rate=0.,
            resistance=self.cfg.uipc_sim.contact.default_contact_resistance*1e9,enable=False)
        for cue in self.cues:
            for mesh in cue.uipc_meshes:
                variants.apply_to(mesh)

    def reset_cue(self):
        self.monitor = False
        self.failure = ""
        self.memory = CueMemory()
        self.cue_kind = int(self.rng.integers(2))
        for i, cue in enumerate(self.cues):
            cue.set_pose(Pose(self.cue_position if i == self.cue_kind else (.38,-1.,.005), [1.,0.,0.,0.]))
            cue.set_physics_mesh_visible(i == self.cue_kind)
        self.cue = self.cues[self.cue_kind]
        self.metadata["memory_ground_truth"] = {"cue_kind":self.cue_kind,
            "expert":"privileged scripted demonstrator; latent labels are absent from policy observations",
            "cue":"one or two physical ribs, followed by release"}

    def pre_move(self):
        # Cue acquisition belongs to POLICY, so training cannot trim the memory away.
        self.monitor = True

    def get_episode_context(self):
        context = super().get_episode_context()
        context.pop('seed', None)
        return context

    def play_once(self):
        super().play_once()
        success = self.check_success()
        reason = 'success' if success else (self.failure or 'execution_failure')
        self.metadata['terminal_reason'] = reason
        self.metadata['task_physical_verdict_step'] = self.step_count
        self.metadata['success_diagnostics'] = dict(success=success,
            terminal_reason=reason, physical_step=self.step_count)
        self._set_phase(self.PHASE_TERMINAL, terminal_reason=reason)
        if self.cfg.save_frequency > 0 and self.mode == 'collect':
            from ._force_task_utils import record_terminal_observation
            record_terminal_observation(self, reason)
        self._save_metadata()

    def _get_observations(self):
        obs = super()._get_observations()
        if self.policy_start_step is not None:
            obs['phase']['policy_step'] = self.step_count-self.policy_start_step
        obs["actor"] = {}
        # Project a fixed world volume around the cue into BOTH cameras. This
        # masks RGB and depth without removing the tactile deformation image.
        for name, data in obs["observation"].items():
            cam = self._camera_manager.cameras[name]
            position = cam.data.pos_w[0].detach().cpu().numpy()
            quat = cam.data.quat_w_ros[0].detach().cpu().numpy()
            camera_to_world = Pose(position, quat).to_transformation_matrix()
            corner_offsets = np.array(list(product((-.03,.03),(-.03,.03),(-.01,.07))))
            corners = np.asarray(self.cue_position) + corner_offsets
            local = (corners-position) @ camera_to_world[:3,:3]
            if np.all(local[:,2] <= .001):
                continue  # A cue entirely behind this camera cannot be seen.
            if np.any(local[:,2] <= .001):
                # Conservative at a near-plane crossing; never expose the cue.
                data = {key: torch.zeros_like(value) for key,value in data.items()}
            else:
                intrinsic = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
                pixels = local @ intrinsic.T
                pixels = pixels[:,:2] / pixels[:,2:]
                data = {key: value.clone() for key,value in data.items()}
                for value in data.values():
                    h,w = value.shape[:2]
                    lo = np.floor(pixels.min(axis=0)-6).astype(int)
                    hi = np.ceil(pixels.max(axis=0)+6).astype(int)
                    x0,x1 = np.clip([lo[0],hi[0]],0,w)
                    y0,y1 = np.clip([lo[1],hi[1]],0,h)
                    value[y0:y1,x0:x1] = 0
            obs["observation"][name] = data
        # Script tags must not reveal selected actors or the latent cue.
        obs["atom"] = {"id": 0, "tag": "manipulate"}
        return obs

    def get_frame_shot(self, obs):
        # Human review is sighted; policy RGB/depth remain masked in the data.
        from torchvision.transforms.functional import resize
        def tile(value, size):
            value = value.clone()
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            if value.shape[-1] == 1:
                value = value.expand(-1,-1,3)
            return resize(value[:,:,:3].permute(2,0,1), size).permute(1,2,0)
        cameras = self._camera_manager.get_observations(['rgb'])
        views = [tile(data['rgb'], (480,640)) for data in cameras.values()]
        key = self.cfg.tactile_video_key
        left = tile(obs['tactile']['left_tactile'][key], (240,240))
        right = tile(obs['tactile']['right_tactile'][key], (240,240))
        return torch.cat((*views,torch.cat((left,right),dim=0)),dim=1)

    def _step(self, is_save=True):
        before = getattr(self, "step_count", 0)
        super()._step(is_save)
        if not self.monitor or self.step_count == before:
            return
        center = self._robot_manager.get_gripper_center_pose().p
        near = np.linalg.norm(center-(np.asarray(self.cue_position)+[0,0,.027])) < .022
        contact = False
        if near:
            depths = self._tactile_manager.get_min_depth().detach().cpu().numpy()
            contact = bool(np.all(np.isfinite(depths)) and np.all(depths < 28.1))
        self.memory.advance(contact)
        if self.policy_start_step is not None:
            if (self.step_count-self.policy_start_step)*self.cfg.sim.dt > self.cfg.memory_timeout_seconds:
                self.failure = self.failure or "timeout"
        self.observe_task()
        if self.failure:
            self.metadata['memory_failure'] = self.failure
            self.plan_success = False

    def observe_task(self):
        pass

    def check_early_stop(self):
        return bool(self.failure)

    def move_center(self, position):
        pose = Pose(position, self._robot_manager.get_gripper_center_pose().q)
        return self.move(self.atom.move_to_pose(self._robot_manager.gripper_center_to_ee(pose)),
                         tag="manipulate", delay=False, time_dilation_factor=.8)

    def _line_to(self, target, spacing=.002):
        start = self._robot_manager.get_gripper_center_pose().p.copy()
        count = max(1, int(np.ceil(np.linalg.norm(target-start)/spacing)))
        for fraction in np.linspace(0., 1., count+1)[1:]:
            if not self.servo_center(start+(target-start)*fraction):
                return False
        return True

    def touch_cue(self):
        position = np.asarray(self.cue_position)+[0,0,.027]
        self.move(self.atom.open_gripper(.7), tag="manipulate", delay=False)
        self.move_center(position+[0,0,.055])
        self.move_center(position)
        self.move(self.atom.close_gripper(0., depth_threshold=27.8), tag="manipulate", delay=False)
        self.delay(16, is_save=True)
        self.move(self.atom.open_gripper(.7), tag="manipulate", delay=False)
        self.move_center(position+[0,0,.060])
        self.delay(12, is_save=True)
        if not self.memory.ready:
            self.failure = "cue_contact_missing"
        self.metadata['cue_result'] = dict(acquired=self.memory.acquired,
            released_steps=self.memory.released,failure=self.failure,
            actual_pose=self.cue.get_pose().p.tolist())
        return self.plan_success and not self.failure

    def servo_center(self, position):
        manager = self._robot_manager
        pose = Pose(position, manager.get_gripper_center_pose().q)
        target = manager.gripper_center_to_ee(pose)
        pos, quat = manager.get_ee_pose_tensor()
        joints = manager.robot.data.joint_pos[:, manager._arm_ids]
        command = torch.tensor(np.r_[target.p,target.q],dtype=torch.float32,device=self.device).reshape(1,7)
        manager._ik_controller.set_command(command)
        goal = manager._ik_controller.compute(pos,quat,manager.jacobian_b[:,:,manager._arm_ids],joints)
        if not bool(torch.isfinite(goal).all()) or float(torch.abs(goal-joints).max()) > .08:
            self.failure = "ik_discontinuity"
            return False
        limits = manager.robot.data.soft_joint_pos_limits[:,manager._arm_ids]
        if bool(torch.any(goal < limits[...,0])) or bool(torch.any(goal > limits[...,1])):
            self.failure = "joint_limit"
            return False
        manager.set_arm(goal[0], torch.zeros_like(goal[0]), force=True)
        self._step()
        self._step()
        if self.last_render != self.step_count:
            manager.robot.update(dt=self.cfg.sim.dt)
        return self.plan_success and not self.failure
