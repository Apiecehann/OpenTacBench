"""Shared geometry, observation and motion helpers for compact tactile tasks."""
from pathlib import Path

import numpy as np

from ._base_task import BaseTask, BaseTaskCfg, Pose, UipcObjectCfg, configclass, torch
from ._force_task_utils import tetra_faces, write_tet_asset
from ._tactile_memory import box_mesh


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
    policy_controls_termination = True
    policy_action_repeat = 2
    absolute_joint_zero_velocity_targets = True
    live_action_joint_state = True
    eval_start_delay_steps = 0
    memory_timeout_seconds = 20.0


class MemoryTask(BaseTask):
    def __init__(self, cfg, **kwargs):
        self.failure = ""
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


    def pre_move(self):
        self.monitor = True

    def get_episode_context(self):
        context = super().get_episode_context()
        context.pop('seed', None)
        return context

    def play_once(self):
        super().play_once()
        self.finish_episode()

    def finish_episode(self):
        """Submit the policy's END and score the state at that instant."""
        if self.phase_id == self.PHASE_TERMINAL:
            return bool(self.eval_success)
        if self.phase_id != self.PHASE_POLICY:
            raise RuntimeError('finish_episode() requires the POLICY handoff')
        success = self.check_success()
        self.eval_success = bool(success)
        self.monitor = False
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
        return self.eval_success

    def _get_observations(self):
        obs = super()._get_observations()
        if self.policy_start_step is not None:
            obs['phase']['policy_step'] = self.step_count-self.policy_start_step
        obs["actor"] = {}
        # Keep neutral action tags in the policy data.
        obs["atom"] = {"id": 0, "tag": "manipulate"}
        return obs

    def get_frame_shot(self, obs):
        # Compose the available scene cameras and the two tactile views.
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
        return bool(self.failure or (self.phase_id == self.PHASE_TERMINAL and not self.eval_success))

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
