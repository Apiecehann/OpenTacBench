"""Franka Panda with a physically mounted Shadow Hand plays one-finger piano."""
from dataclasses import dataclass
import math
import numpy as np
import torch
import gymnasium as gym
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController,DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat,skew_symmetric_matrix
from .shadow_piano import Task as ShadowTask,TaskCfg as ShadowCfg


@dataclass
class TaskCfg(ShadowCfg):
    arm_position: tuple=(-.45,-.09175,0.)
    episode_seconds: float=180.
    action_repeat: int=8
    tactile_sensor: str='digit'


class Task(ShadowTask):
    metadata=dict(ShadowTask.metadata,render_fps=30)
    def __init__(self,cfg=None,render_mode=None):
        self.max_arm_tracking_error=0.
        self.max_press_trim=0.
        super().__init__(cfg or TaskCfg(),render_mode)
        self.action_space.low[1]=-.0015
        state=gym.spaces.Box(-np.inf,np.inf,(111,),np.float32)
        if self.tactile is None:
            self.observation_space=state
        else:
            self.observation_space.spaces['state']=state

    def _hand_cfg(self):
        cfg=super()._hand_cfg()
        cfg.init_state.pos=tuple(np.asarray(self.cfg.hand_position)+[0.,-.09175,.03])
        cfg.init_state.joint_pos={'panda_joint[1-7]':0.,'robot0_FFJ2':0.}
        cfg.actuators.pop('slides')
        cfg.actuators['finger'].stiffness=6.
        cfg.actuators['finger'].damping=.20
        cfg.actuators['shoulder']=ImplicitActuatorCfg(joint_names_expr=['panda_joint[1-4]'],
            stiffness=8000.,damping=200.,effort_limit_sim=87.,velocity_limit_sim=2.)
        cfg.actuators['forearm']=ImplicitActuatorCfg(joint_names_expr=['panda_joint[5-7]'],
            stiffness=8000.,damping=200.,effort_limit_sim=12.,velocity_limit_sim=2.)
        return cfg

    def _mount_wrist(self):
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.robot_motion.motion_generation')
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
        from transforms3d.quaternions import mat2quat
        solver=LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config('Franka'))
        cache=UsdGeom.XformCache()
        root=self.stage.GetPrimAtPath('/World/Hand')
        hand_world=np.array(cache.GetLocalToWorldTransform(root)).T
        mount_world=np.array(cache.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath('/World/Hand/robot0_hand_mount'))).T
        base=np.asarray(self.cfg.arm_position)
        joints,ok=solver.compute_inverse_kinematics('panda_link8',mount_world[:3,3]-base,
            mat2quat(mount_world[:3,:3]),warm_start=np.array([0.,-.569,0.,-2.81,0.,2.241,.741]),
            position_tolerance=1.e-5,orientation_tolerance=1.e-3)
        if not ok: raise RuntimeError('The piano wrist pose is outside the Franka arm workspace')
        self.hand.cfg.init_state.joint_pos={**{f'panda_joint{i+1}':float(q) for i,q in enumerate(joints)},'robot0_FFJ2':0.}
        arm_cfg=FRANKA_PANDA_HIGH_PD_CFG.copy()
        arm_cfg.spawn.articulation_props.fix_root_link=True
        arm_cfg.spawn.articulation_props.enabled_self_collisions=False
        arm_cfg.spawn.articulation_props.solver_position_iteration_count=32
        # Express the existing Panda asset in the Shadow root's frame.
        arm_world=np.eye(4); arm_world[:3,3]=base
        arm_local=np.linalg.solve(hand_world,arm_world)
        arm_cfg.spawn.func('/World/Hand/Arm',arm_cfg.spawn,
            translation=tuple(arm_local[:3,3]),orientation=tuple(mat2quat(arm_local[:3,:3])))
        for prim in list(Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand/Arm'))):
            if not prim.IsValid(): continue
            if prim.GetName() in ('panda_hand','panda_leftfinger','panda_rightfinger'):
                prim.SetActive(False)
            elif prim.IsA(UsdPhysics.Joint):
                bodies=UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()+UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                if any(str(p).rsplit('/',1)[-1] in ('panda_hand','panda_leftfinger','panda_rightfinger') for p in bodies):
                    prim.SetActive(False)
        position,rotation=solver.compute_forward_kinematics('panda_link7',joints)
        relative=np.eye(4)
        relative[:3,3]=rotation.T@(mount_world[:3,3]-base-position)
        relative[:3,:3]=rotation.T@mount_world[:3,:3]
        joint=UsdPhysics.FixedJoint.Define(self.stage,'/World/Hand/ArmHandMount')
        joint.CreateBody0Rel().SetTargets(['/World/Hand/Arm/panda_link7'])
        joint.CreateBody1Rel().SetTargets(['/World/Hand/robot0_hand_mount'])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*relative[:3,3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Matrix4d(relative.T.copy()).ExtractRotationQuat()))
        UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/PalmStand')).MakeInvisible()

    def _create_keyboard(self):
        super()._create_keyboard()
        from envs._scene import scene_plate_cfg
        self.stage.RemovePrim('/World/Table')
        plate=scene_plate_cfg('/World/Table')
        plate.spawn.scale=(2.2,2.1,1.)
        plate.spawn.func(plate.prim_path,plate.spawn,translation=(.1,0.,-.0025))

    def _configure_hand_state(self):
        # Initialize the arm before capturing hand frames and before UIPC setup.
        # Isaac's first reset loads USD joint states, which can still be zero.
        initial=self.hand.data.default_joint_pos
        self.hand.write_joint_state_to_sim(initial,torch.zeros_like(initial))
        self.hand.set_joint_position_target(initial)
        self.hand.write_data_to_sim()
        self.sim.forward()
        self.hand.update(self.cfg.physics_dt)
        self._index=self.hand.find_joints('robot0_FFJ2')[0][0]
        self._arm_ids=self.hand.find_joints('panda_joint[1-7]')[0]
        self._palm=self.hand.find_bodies('robot0_palm')[0][0]
        self._fixed=[]
        self._fixed_bodies=[i for i,n in enumerate(self.hand.body_names) if n.startswith('robot0_')
            and n not in ('robot0_ffproximal','robot0_ffmiddle','robot0_ffdistal')]
        palm=self.hand.data.body_pos_w[:,self._palm]
        rotation=matrix_from_quat(self.hand.data.body_quat_w[:,self._palm])
        self._relative_fixed=((self.hand.data.body_pos_w[:,self._fixed_bodies]-palm[:,None])@rotation).clone()
        self._initial_palm=palm.clone()
        self._palm_quat=self.hand.data.body_quat_w[:,self._palm].clone()
        self._arm_start=self.hand.data.joint_pos[:,self._arm_ids].clone()
        self._arm_joint_peak=torch.zeros_like(self._arm_start)
        self._ik=DifferentialIKController(DifferentialIKControllerCfg(
            command_type='pose',use_relative_mode=False,ik_method='dls'),num_envs=1,device=self.cfg.device)
        # Both scene families use the same HDR environment configuration.
        from envs._scene import scene_light_cfg,resolve_scene_background_texture
        light=scene_light_cfg('/World/Light')
        self.stage.RemovePrim('/World/Light')
        light.spawn.texture_file=str(resolve_scene_background_texture('base5'))
        light.spawn.func(light.prim_path,light.spawn)

    def reset(self,**kwargs):
        self.max_arm_tracking_error=0.
        self.max_press_trim=0.
        self._press_trim=0.
        self._arm_joint_peak.zero_()
        self._ik.reset()
        result=super().reset(**kwargs)
        self._demo_stage='travel'
        self._stage_start=0.
        self._travel_from=float(self._wrist_state()[0])
        return result

    def _wrist_state(self):
        displacement=(self.hand.data.body_pos_w[0,self._palm]-self._initial_palm[0]).cpu().numpy()
        return np.array([-.09175+displacement[1],.03+displacement[2],
            float(self.hand.data.joint_pos[0,self._index])])

    def _action_target(self,action):
        return self._filter_action(action)

    def demonstration_action(self,control='demo'):
        if control!='demo': return super().demonstration_action(control)
        now=self.step_count*self.cfg.physics_dt
        elapsed=now-self._stage_start
        stage=self._demo_stage
        y=self.layout[self.cfg.melody[self._demo_note]-21].position[1]-.033
        actual=self._wrist_state()
        def smooth(value):
            v=float(np.clip(value,0.,1.)); return v*v*(3-2*v)
        def enter(next_stage):
            self._demo_stage=next_stage; self._stage_start=now
        if elapsed>3.5:
            self.score.failure='wrist_tracking_timeout' if stage!='press' else 'press_timeout'
        if stage=='travel':
            duration=max(.08,abs(y-self._travel_from)/.10)
            command=self._travel_from+(y-self._travel_from)*smooth(elapsed/duration)
            if elapsed>=duration and abs(actual[0]-y)<.0008 and abs(actual[1]-.03)<.001:
                self._press_trim=0.
                enter('lower')
            return np.array([command,.03,0.])
        if stage=='lower':
            z=.03*(1-smooth(elapsed/.50))
            if elapsed>=.65 and abs(actual[1])<.0005 and abs(actual[0]-y)<.001:
                enter('press')
            return np.array([y,z,0.])
        if stage=='press':
            amount=self.cfg.press_joint_angle*smooth(elapsed/.45)
            if elapsed>.55 and not self.score._armed:
                angle=float(self._key_state()[0][self.score.active_key])
                dt=self.cfg.physics_dt*self.cfg.action_repeat
                descent=min(.002,.20*max(0.,.032-angle))*dt
                self._press_trim=max(-.0015,self._press_trim-descent)
                self.max_press_trim=max(self.max_press_trim,abs(self._press_trim))
            hold=2. if self._demo_note%7==6 else 1.
            if self.score._armed and elapsed>=hold: enter('release')
            return np.array([y,self._press_trim,amount])
        if stage=='release':
            amount=self.cfg.press_joint_angle*(1-smooth(elapsed/.30))
            if elapsed>=.35: enter('lift')
            return np.array([y,self._press_trim,amount])
        z=self._press_trim+(.03-self._press_trim)*smooth(elapsed/.50)
        if elapsed>=.50 and self.score.count>self._demo_note and abs(actual[1]-.03)<.001:
            self._demo_note=self.score.count
            self._travel_from=float(actual[0])
            enter('travel')
        return np.array([y,z,0.])

    def _physics_step(self,target):
        # Reset supplies joint positions; runtime actions specify Y/Z/index.
        if torch.is_tensor(target):
            return super()._physics_step(target)
        position=self.hand.data.body_pos_w[:,self._palm]
        quat=self.hand.data.body_quat_w[:,self._palm]
        shift=position-self.hand.data.body_com_pos_w[:,self._palm]
        jac=self.hand.root_physx_view.get_jacobians()[:,self._palm-1,:,self._arm_ids].clone()
        jac[:,:3]-=skew_symmetric_matrix(shift)@jac[:,3:]
        desired=self._initial_palm.clone()
        desired[0,1]+=float(target[0]+.09175)
        desired[0,2]+=float(target[1]-.03)
        self._ik.set_command(torch.cat((desired,self._palm_quat),dim=-1))
        joints=self.hand.data.joint_pos[:,self._arm_ids]
        q_arm=self._ik.compute(position,quat,jac,joints)
        limits=self.hand.data.soft_joint_pos_limits[:,self._arm_ids]
        # Scale the complete IK increment together. Clipping each joint
        # independently changes its Cartesian direction and rotates the wrist.
        delta=q_arm-joints
        rate=torch.clamp(.008/delta.abs().amax(dim=-1,keepdim=True).clamp_min(1.e-12),max=1.)
        q_arm=torch.clip(joints+rate*delta,limits[...,0],limits[...,1])
        q=self._rest.clone()
        q[:,self._arm_ids]=q_arm
        q[0,self._index]=float(target[2])
        self.max_arm_tracking_error=max(self.max_arm_tracking_error,float(torch.linalg.vector_norm(position-desired)))
        super()._physics_step(q)
        self._arm_joint_peak=torch.maximum(self._arm_joint_peak,
            (self.hand.data.joint_pos[:,self._arm_ids]-self._arm_start).abs())

    def _check_hand_motion(self):
        # Arm tracking can rotate the wrist slightly. Test the inactive fingers
        # in palm coordinates, so rigid wrist rotation is not mistaken for flexion.
        palm=self.hand.data.body_pos_w[:,self._palm]
        quat=self.hand.data.body_quat_w[:,self._palm]
        relative=(self.hand.data.body_pos_w[:,self._fixed_bodies]-palm[:,None])@matrix_from_quat(quat)
        error=float(torch.linalg.vector_norm(relative-self._relative_fixed,dim=-1).max())
        angle=2*math.acos(min(1.,float(torch.abs((quat*self._palm_quat).sum()))))
        self.max_fixed_body_displacement=max(self.max_fixed_body_displacement,error)
        self.max_palm_displacement=max(self.max_palm_displacement,float(torch.linalg.vector_norm(palm-self._initial_palm)))
        self.max_palm_rotation=max(self.max_palm_rotation,angle)
        if error>.001: self.score.failure='inactive_hand_moved'
        if angle>.05: self.score.failure='wrist_orientation_lost'

    def _observation(self):
        angles,_=self._key_state()
        # Preserve the existing musical task state and include arm proprioception.
        palm_v=self.hand.data.body_lin_vel_w[0,self._palm]
        wrist=self._wrist_state()
        state=np.r_[wrist,[float(palm_v[1]),float(palm_v[2]),float(self.hand.data.joint_vel[0,self._index])],
            angles,self.score.count/self.score.repetitions,float(self.score._armed),self.score.active_key/87,
            self.hand.data.joint_pos[0,self._arm_ids].cpu().numpy(),
            self.hand.data.joint_vel[0,self._arm_ids].cpu().numpy()].astype(np.float32)
        return state if self.tactile is None else dict(state=state,tactile=dict(index=self.tactile.observation()))

    def _info(self):
        result=super()._info()
        result.update(arm_model='Franka Panda',wrist_motion='7 driven arm joints with differential IK',
            max_arm_tracking_error_m=self.max_arm_tracking_error,
            max_press_trim_m=self.max_press_trim,
            arm_joint_travel_rad=self._arm_joint_peak[0].cpu().tolist())
        return result
