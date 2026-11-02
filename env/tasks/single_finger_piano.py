"""Fixed Allegro palm, one movable index joint, repeated physical C4 presses.

Import after Isaac Lab's AppLauncher. This Gymnasium task uses PhysX rather than
the arm/gripper-specific TacEx task base; select --song repeated in the launcher.
"""
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_assets import ALLEGRO_HAND_CFG

from envs._piano import keyboard_layout, build_cabinet
from envs._piano_audio import PianoContacts
from ._single_key_score import SingleKeyScore

TASK_INSTRUCTION = 'Keep the palm and other fingers fixed. Press and release C4 five times with the index finger.'


@dataclass
class TaskCfg:
    device: str = 'cuda:0'
    target_midi: int = 60
    repetitions: int = 5
    episode_seconds: float = 15.
    physics_dt: float = 1/240
    action_repeat: int = 4
    press_joint_angle: float = .45
    hand_position: tuple = (.20, -.053, .128)
    # Local +Z points along the keys; positive index flexion points downward.
    hand_rotation: tuple = (.27059805, .65328148, .27059805, .65328148)
    hand_usd: str | None = None
    record_video: bool = False


class Task(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, cfg=None, render_mode=None):
        self.cfg = cfg or TaskCfg()
        cfg = self.cfg
        if (not 21 <= cfg.target_midi <= 108 or cfg.target_midi % 12 in (1,3,6,8,10)
                or cfg.repetitions < 1 or cfg.physics_dt <= 0 or cfg.action_repeat < 1
                or cfg.episode_seconds <= 0 or not 0 < cfg.press_joint_angle <= .55):
            raise ValueError('Choose a white target key, positive timing/counts and a joint target in (0, .55]')
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        # Index position/velocity, 88 key depressions, progress and press state.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (92,), np.float32)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=cfg.physics_dt, device=cfg.device, render_interval=cfg.action_repeat,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6)))
        self.stage = omni.usd.get_context().get_stage()
        self.layout = keyboard_layout()
        self.target = cfg.target_midi - 21
        self.step_count = 0
        self._done = False
        self._create_keyboard()
        self._create_hand()
        self._create_tactile()
        if cfg.record_video:
            from isaaclab.sensors import Camera, CameraCfg
            self.camera = Camera(CameraCfg(prim_path='/World/RecordingCamera',
                width=960, height=640, data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=36.,
                                               clipping_range=(.01,10.))))
        self.key_contacts = ContactSensor(self._key_contact_cfg())
        self.sim.set_camera_view((-.10,-.60,.53), (.44, self.layout[self.target].position[1], .09))
        sim_utils.DomeLightCfg(intensity=2200.).func('/World/Light', sim_utils.DomeLightCfg(intensity=2200.))
        self.sim.reset()
        self.hand.update(cfg.physics_dt)
        self.keys.update(cfg.physics_dt)
        self._configure_hand_state()
        self._fixed_body_positions = self.hand.data.body_pos_w[:,self._fixed_bodies].clone()
        self._rest = self.hand.data.default_joint_pos.clone()
        self._root = self.hand.data.root_state_w.clone()
        # Sensors/articulations can enumerate matching prims in a different order.
        key_paths = self.keys.root_physx_view.prim_paths
        contact_paths = self.key_contacts.body_physx_view.prim_paths
        self._key_order = [next(i for i,p in enumerate(key_paths) if f'Key_{m}' in p.split('/'))
                           for m in range(21,109)]
        self._contact_order = [next(i for i,p in enumerate(contact_paths) if f'Key_{m}' in p.split('/'))
                               for m in range(21,109)]
        self.score = self._new_score()
        self.audio_contacts = PianoContacts()

    def _create_tactile(self):
        """Create optional sensors before the first simulation reset."""
        pass

    def _key_contact_cfg(self):
        return ContactSensorCfg(prim_path='/World/Keyboard/Key_.*/Body',update_period=0.,history_length=1)

    def _configure_hand_state(self):
        self._index = self.hand.find_joints('index_joint_1')[0][0]
        self._fixed = [i for i in range(self.hand.num_joints) if i != self._index]
        self._fixed_bodies = [i for i,name in enumerate(self.hand.body_names)
                              if name not in ('index_link_1','index_link_2','index_link_3')]

    def _new_score(self):
        return SingleKeyScore(self.target, self.cfg.repetitions)

    def _box(self, path, center, size, color, **kwargs):
        shape = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, roughness=.38), **kwargs)
        shape.func(path, shape, translation=center)

    def _create_keyboard(self):
        UsdGeom.Xform.Define(self.stage, '/World/Keyboard')
        for key in self.layout:
            path = f'/World/Keyboard/Key_{key.midi}'
            UsdGeom.Xform.Define(self.stage, path)
            UsdPhysics.ArticulationRootAPI.Apply(self.stage.GetPrimAtPath(path))
            PhysxSchema.PhysxArticulationAPI.Apply(self.stage.GetPrimAtPath(path))
            center = (key.position[0], key.position[1], key.position[2]+key.size[2]/2)
            pivot = (center[0]+key.size[0]/2, center[1], center[2])
            rigid = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=.2)
            self._box(path+'/Root', pivot, (.002,.002,.002), (.02,.02,.02),
                      rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=.01))
            self._box(path+'/Body', center, key.size,
                      (.012,.014,.018) if key.black else (.91,.89,.82),
                      rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=.02),
                      collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
                      activate_contact_sensors=True)
            fixed = UsdPhysics.FixedJoint.Define(self.stage, path+'/Mount')
            fixed.CreateBody1Rel().SetTargets([path+'/Root'])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
            hinge = UsdPhysics.RevoluteJoint.Define(self.stage, path+'/hinge')
            hinge.CreateBody0Rel().SetTargets([path+'/Root'])
            hinge.CreateBody1Rel().SetTargets([path+'/Body'])
            hinge.CreateLocalPos1Attr(Gf.Vec3f(key.size[0]/2,0,0))
            hinge.CreateAxisAttr('Y')
            hinge.CreateLowerLimitAttr(-4.)
            hinge.CreateUpperLimitAttr(0.)
            # PhysX needs an authored drive before Isaac Lab can set its gains.
            drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateTargetPositionAttr(0.)
            drive.CreateStiffnessAttr(1.)
            drive.CreateDampingAttr(.025)
            drive.CreateMaxForceAttr(1.)
        self.keys = Articulation(ArticulationCfg(prim_path='/World/Keyboard/Key_.*', spawn=None,
            actuators={'return_springs': ImplicitActuatorCfg(joint_names_expr=['hinge'],
                stiffness=1., damping=.025, effort_limit_sim=1.)}))
        # The standard layout's black keys overlap the white-key rear volumes.
        # Keys contact the hand, but must not push each other away from rest.
        group = UsdPhysics.CollisionGroup.Define(self.stage, '/World/KeyContactGroup')
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(['/World/Keyboard'])
        group.CreateFilteredGroupsRel().SetTargets(['/World/KeyContactGroup'])
        self._box('/World/Table', (.40,0.,-.025), (1.30,1.70,.05), (.13,.09,.055),
                  collision_props=sim_utils.CollisionPropertiesCfg())
        # Reuse the existing piano cabinet; its keybed leaves travel below the keys.
        from types import SimpleNamespace
        build_cabinet(SimpleNamespace(scene=SimpleNamespace(env_prim_paths=['/World'])))
        self._box('/World/PalmStand', (.20,-.053,.040), (.075,.09,.08), (.06,.07,.08))

    def _create_hand(self):
        cfg = ALLEGRO_HAND_CFG.replace(prim_path='/World/Hand')
        if self.cfg.hand_usd:
            cfg.spawn.usd_path = self.cfg.hand_usd
        cfg.spawn.articulation_props.fix_root_link = True
        cfg.spawn.articulation_props.enabled_self_collisions = False
        cfg.spawn.rigid_props.max_depenetration_velocity = .2
        position = np.array(self.cfg.hand_position)
        position[1] += self.layout[self.target].position[1]
        cfg.init_state.pos = tuple(position)
        cfg.init_state.rot = self.cfg.hand_rotation
        lock_pose = {'^(?!thumb_joint_0).*': 0., 'thumb_joint_0': .28}
        cfg.init_state.joint_pos = {'index_joint_1': 0.}
        cfg.actuators['fingers'].stiffness = 4.
        cfg.actuators['fingers'].damping = .15
        self.hand = Articulation(cfg)
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand')):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        sim_utils.modify_collision_properties('/World/Hand',
            sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.))
        # Replace inactive revolute joints with welds. Equal revolute limits can
        # still drift under contact loads in the iterative constraint solver.
        import re
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand')):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                name = prim.GetName()
                joint = UsdPhysics.RevoluteJoint(prim)
                if name == 'index_joint_1':
                    joint.GetLowerLimitAttr().Set(0.)
                    joint.GetUpperLimitAttr().Set(math.degrees(.55))
                else:
                    value = next(v for expr,v in lock_pose.items() if re.fullmatch(expr,name))
                    old = joint.GetLocalRot0Attr().Get()
                    axis = {'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}[joint.GetAxisAttr().Get()]
                    rotation = Gf.Quatd(old)*Gf.Rotation(axis,math.degrees(value)).GetQuat()
                    UsdPhysics.FixedJoint.Define(self.stage, prim.GetPath()).GetLocalRot0Attr().Set(Gf.Quatf(rotation))

    def _physics_step(self, target):
        self.hand.set_joint_position_target(target)
        self.keys.set_joint_position_target(torch.zeros_like(self.keys.data.joint_pos))
        self.hand.write_data_to_sim()
        self.keys.write_data_to_sim()
        self.sim.step(render=False)
        self.hand.update(self.cfg.physics_dt)
        self.keys.update(self.cfg.physics_dt)
        self.key_contacts.update(self.cfg.physics_dt)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.hand.write_joint_state_to_sim(self._rest, torch.zeros_like(self._rest))
        zero = torch.zeros_like(self.keys.data.joint_pos)
        self.keys.write_joint_state_to_sim(zero, zero)
        self.hand.reset()
        self.keys.reset()
        self.key_contacts.reset()
        for _ in range(120):
            self._physics_step(self._rest)
        self.step_count = 0
        self.score = self._new_score()
        self.audio_contacts = PianoContacts()
        self._done = False
        self.max_fixed_joint_error = self.max_palm_displacement = 0.
        self.max_fixed_body_displacement = 0.
        self.max_key_depression = 0.
        self.key_peak_angles = np.zeros(88)
        self.key_peak_forces = np.zeros(88)
        return self._observation(), self._info()

    def _key_state(self):
        angles = -self.keys.data.joint_pos[self._key_order,0].detach().cpu().numpy()
        forces = np.abs(self.key_contacts.data.net_forces_w.reshape(-1,3)[self._contact_order,2].detach().cpu().numpy())
        return angles, forces

    def _observation(self):
        angles, _ = self._key_state()
        return np.r_[float(self.hand.data.joint_pos[0,self._index]),
                     float(self.hand.data.joint_vel[0,self._index]), angles,
                     self.score.count/self.cfg.repetitions, float(self.score._armed)].astype(np.float32)

    def _info(self):
        return dict(success=self.score.success, failure=self.score.failure,
                    presses=self.score.successful_notes, notes_attempted=self.score.count,
                    target_midi=self.cfg.target_midi, physics_step=self.step_count,
                    terminal_reason=('success' if self.score.success else self.score.failure
                        or ('timeout' if self.step_count*self.cfg.physics_dt >= self.cfg.episode_seconds else '')))

    def _action_target(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Action must contain one finite index-finger command')
        target = self._rest.clone()
        target[0,self._index] = (float(np.clip(action[0],-1.,1.))+1.)*.5*self.cfg.press_joint_angle
        return target

    def _check_hand_motion(self):
        error = (float(torch.abs(self.hand.data.joint_pos[:,self._fixed]-self._rest[:,self._fixed]).max())
                 if self._fixed else 0.)
        fixed_displacement = float(torch.linalg.vector_norm(
            self.hand.data.body_pos_w[:,self._fixed_bodies]-self._fixed_body_positions,dim=-1).max())
        self.max_fixed_body_displacement = max(self.max_fixed_body_displacement,fixed_displacement)
        displacement = float(torch.linalg.vector_norm(self.hand.data.root_pos_w-self._root[:,:3]))
        self.max_fixed_joint_error = max(self.max_fixed_joint_error,error)
        self.max_palm_displacement = max(self.max_palm_displacement,displacement)
        if error > .005 or displacement > .001 or fixed_displacement > .001:
            self.score.failure = 'fixed_hand_moved'

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset() before stepping a completed episode')
        target = self._action_target(action)
        before = self.score.count
        for _ in range(self.cfg.action_repeat):
            self._physics_step(target)
            self.step_count += 1
            angles, forces = self._key_state()
            self.key_peak_angles = np.maximum(self.key_peak_angles, angles)
            self.key_peak_forces = np.maximum(self.key_peak_forces, forces)
            self.max_key_depression = max(self.max_key_depression, float(angles[self.score.active_key]))
            self._check_hand_motion()
            if hasattr(self,'piano_annotations'):
                self.piano_annotations.update(self.step_count,self.cfg.physics_dt,self.score.count,forces)
            self.score.advance(angles, forces, self.cfg.physics_dt, self.step_count)
            self.audio_contacts.advance(self.step_count, self._audio_forces(angles, forces))
            if self.score.finished or self.score.failure:
                break
        self._before_render()
        self.sim.render()
        terminated = self.score.finished or bool(self.score.failure)
        truncated = self.step_count*self.cfg.physics_dt >= self.cfg.episode_seconds and not terminated
        self._done = terminated or truncated
        return self._observation(), float(self.score.count-before), terminated, truncated, self._info()

    def _before_render(self):
        """Allow derived tasks to synchronize external simulated geometry."""
        pass

    def _audio_forces(self, angles, forces):
        return forces

    def render(self):
        # The CLI attaches an RTX camera only when recording is requested.
        if not hasattr(self, 'camera'):
            raise RuntimeError('Attach a camera or run the CLI with --video')
        self.camera.update(self.cfg.physics_dt*self.cfg.action_repeat, force_recompute=True)
        return self.camera.data.output['rgb'][0,:,:,:3].detach().cpu().numpy()

    def close(self):
        # Isaac Lab 2.3 keeps rendering on STOP by default. Disable that viewer
        # callback when intentionally shutting down this standalone environment.
        self.sim._disable_app_control_on_stop_handle = True
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
