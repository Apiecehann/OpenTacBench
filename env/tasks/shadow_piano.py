"""One Shadow Hand index finger plays Twinkle on physical piano keys.

The wrist rides on driven Y/Z slides. Other hand joints are welded, so lateral
travel and lifting are continuous PhysX motion rather than root teleportation.
"""
from dataclasses import dataclass
import math
import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab_assets import SHADOW_HAND_CFG

from .single_finger_piano import Task as PianoTask, TaskCfg as PianoCfg
from ._single_key_score import SingleKeyScore
from ._twinkle import TWINKLE, HOLD_SECONDS

TASK_INSTRUCTION = 'Play the complete Twinkle Twinkle Little Star with the index finger; move and lift the wrist between keys.'


@dataclass
class TaskCfg(PianoCfg):
    episode_seconds: float = 90.
    press_joint_angle: float = .55
    hand_position: tuple = (-.011, 0., .110)
    hand_rotation: tuple = (0., 2**-.5, -2**-.5, 0.)
    melody: tuple = TWINKLE


class Task(PianoTask):
    def __init__(self, cfg=None, render_mode=None):
        super().__init__(cfg or TaskCfg(), render_mode)
        self.action_space = gym.spaces.Box(np.array([-.16,0.,0.],np.float32),
            np.array([.10,.05,self.cfg.press_joint_angle],np.float32))
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(97,),np.float32)

    def _new_score(self):
        holds = HOLD_SECONDS if self.cfg.melody == TWINKLE else tuple(.12 for _ in self.cfg.melody)
        return SingleKeyScore(key_sequence=tuple(m-21 for m in self.cfg.melody), note_hold_seconds=holds)

    def _create_hand(self):
        cfg = SHADOW_HAND_CFG.replace(prim_path='/World/Hand')
        if self.cfg.hand_usd:
            cfg.spawn.usd_path = self.cfg.hand_usd
        cfg.spawn.articulation_props.fix_root_link = False
        cfg.spawn.articulation_props.enabled_self_collisions = False
        cfg.spawn.fixed_tendons_props = None
        cfg.spawn.rigid_props.max_depenetration_velocity = .2
        cfg.spawn.rigid_props.kinematic_enabled = False
        cfg.init_state.pos, cfg.init_state.rot = self.cfg.hand_position,self.cfg.hand_rotation
        cfg.init_state.joint_pos = {'rail_y':-.09175,'rail_z':.03,'robot0_FFJ2':0.}
        cfg.actuators = {
            'finger':ImplicitActuatorCfg(joint_names_expr=['robot0_FFJ2'],stiffness=4.,damping=.15,effort_limit_sim=.9),
            'slides':ImplicitActuatorCfg(joint_names_expr=['rail_.*'],stiffness=20000.,damping=250.,effort_limit_sim=300.)}
        self.hand = Articulation(cfg)
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand')):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in list(Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand'))):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if prim.IsA(UsdPhysics.FixedJoint):
                fixed=UsdPhysics.FixedJoint(prim)
                if not fixed.GetBody0Rel().GetTargets() or not fixed.GetBody1Rel().GetTargets():
                    # Removing a referenced prim's local spec reveals its
                    # original enabled anchor again. Deactivate the prim with
                    # a stronger USD opinion before adding the rail anchor.
                    prim.SetActive(False)
                    continue
            for schema in list(prim.GetAppliedSchemas()):
                if schema.startswith('PhysxTendonAxisRootAPI:'):
                    prim.RemoveAPI(PhysxSchema.PhysxTendonAxisRootAPI,schema.split(':',1)[1])
                elif schema.startswith('PhysxTendonAxisAPI:'):
                    prim.RemoveAPI(PhysxSchema.PhysxTendonAxisAPI,schema.split(':',1)[1])
            if prim.IsA(UsdPhysics.RevoluteJoint):
                if prim.GetName() != 'robot0_FFJ2':
                    UsdPhysics.FixedJoint.Define(self.stage,prim.GetPath())
                else:
                    joint=UsdPhysics.RevoluteJoint(prim)
                    joint.GetLowerLimitAttr().Set(0.)
                    joint.GetUpperLimitAttr().Set(math.degrees(.65))
        root=self.stage.GetPrimAtPath('/World/Hand')
        rigid=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        for name in ('RailBase','YCarriage','ZCarriage'):
            self._box('/World/Hand/'+name,(0,0,0),(.01,.01,.01),(.06,.07,.09),
                rigid_props=rigid,mass_props=sim_utils.MassPropertiesCfg(mass=.2))
            UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Hand/'+name)).MakeInvisible()
        anchor=UsdPhysics.FixedJoint.Define(self.stage,'/World/Hand/Anchor')
        anchor.CreateBody1Rel().SetTargets(['/World/Hand/RailBase'])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(*self.cfg.hand_position))
        rotation=Gf.Quatf(self.cfg.hand_rotation[0],Gf.Vec3f(*self.cfg.hand_rotation[1:]))
        anchor.CreateLocalRot0Attr(rotation)
        # Mark the world-to-rail joint explicitly as the articulation root.
        # On a parent Xform PhysX can otherwise choose the asset's old mount
        # body as the fixed root and keep the slides on the stationary branch.
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        articulation=PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim())
        articulation.CreateEnabledSelfCollisionsAttr(False)
        articulation.CreateSolverPositionIterationCountAttr(16)
        articulation.CreateSolverVelocityIterationCountAttr(2)
        for name,axis,parent,child,limits in (
                ('rail_y','Y','RailBase','YCarriage',(-.16,.10)),
                ('rail_z','Z','YCarriage','ZCarriage',(0.,.05))):
            joint=UsdPhysics.PrismaticJoint.Define(self.stage,'/World/Hand/'+name)
            joint.CreateBody0Rel().SetTargets(['/World/Hand/'+parent])
            joint.CreateBody1Rel().SetTargets(['/World/Hand/'+child])
            joint.CreateAxisAttr(axis)
            joint.CreateLowerLimitAttr(limits[0])
            joint.CreateUpperLimitAttr(limits[1])
            joint.CreateLocalRot0Attr(rotation.GetConjugate())
            joint.CreateLocalRot1Attr(rotation.GetConjugate())
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(20000.)
            drive.CreateDampingAttr(250.)
            drive.CreateMaxForceAttr(300.)
        mount=UsdPhysics.FixedJoint.Define(self.stage,'/World/Hand/CarriageMount')
        mount.CreateBody0Rel().SetTargets(['/World/Hand/ZCarriage'])
        mount.CreateBody1Rel().SetTargets(['/World/Hand/robot0_hand_mount'])
        # The USD mount body has its own local orientation. Preserve it when
        # attaching the asset to the slide; equating the two body frames would
        # rotate the entire hand despite an unchanged model-root quaternion.
        relative=UsdGeom.XformCache().ComputeRelativeTransform(
            self.stage.GetPrimAtPath('/World/Hand/robot0_hand_mount'),root)[0]
        mount.CreateLocalPos0Attr(Gf.Vec3f(relative.ExtractTranslation()))
        mount.CreateLocalRot0Attr(Gf.Quatf(relative.ExtractRotationQuat()))
        sim_utils.modify_collision_properties('/World/Hand',
            sim_utils.CollisionPropertiesCfg(contact_offset=.0005,rest_offset=0.))
        UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/PalmStand')).MakeInvisible()
        self._box('/World/WristRail',(-.011,0.,.040),(.09,.36,.08),(.035,.045,.060))
        self._finish_hand()

    def _finish_hand(self):
        from envs._force_task_utils import _material
        looks={
            'shell':_material(self.stage,'/World/Hand/Looks/Shell',(.74,.79,.83),.28,.45),
            'metal':_material(self.stage,'/World/Hand/Looks/Metal',(.35,.43,.50),.25,.75),
            'pad':_material(self.stage,'/World/Hand/Looks/Pad',(.035,.045,.055),.82),
            'base':_material(self.stage,'/World/Hand/Looks/Base',(.025,.035,.050),.32,.55)}
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand')):
            path=str(prim.GetPath()).lower()
            if prim.IsA(UsdGeom.Mesh) and 'collisions' not in path:
                kind=('base' if 'forearm' in path or 'mount' in path
                      else 'pad' if 'distal' in path else 'shell' if 'palm' in path else 'metal')
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[kind],bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _configure_hand_state(self):
        self._index=self.hand.find_joints('robot0_FFJ2')[0][0]
        self._y=self.hand.find_joints('rail_y')[0][0]
        self._z=self.hand.find_joints('rail_z')[0][0]
        self._palm=self.hand.find_bodies('robot0_palm')[0][0]
        self._fixed=[]
        self._fixed_bodies=[i for i,n in enumerate(self.hand.body_names) if n.startswith('robot0_')
            and n not in ('robot0_ffproximal','robot0_ffmiddle','robot0_ffdistal')]
        self._relative_fixed=(self.hand.data.body_pos_w[:,self._fixed_bodies]
                              -self.hand.data.body_pos_w[:,self._palm,None]).clone()
        self._initial_palm=self.hand.data.body_pos_w[:,self._palm].clone()
        self._palm_quat=self.hand.data.body_quat_w[:,self._palm].clone()

    def reset(self, **kwargs):
        result=super().reset(**kwargs)
        self._demo_note=0
        self._demo_start=0.
        self._demo_y_start=float(self.hand.data.joint_pos[0,self._y])
        self.max_palm_rotation=0.
        return result

    def _action_target(self, action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all():
            raise ValueError('Action must contain finite rail Y (m), lift Z (m), and index flexion (rad)')
        a=np.clip(a,self.action_space.low,self.action_space.high)
        q=self._rest.clone()
        q[0,[self._y,self._z,self._index]]=torch.as_tensor(a,dtype=q.dtype,device=q.device)
        return q

    def _check_hand_motion(self):
        palm=self.hand.data.body_pos_w[:,self._palm]
        relative=self.hand.data.body_pos_w[:,self._fixed_bodies]-palm[:,None]
        error=float(torch.linalg.vector_norm(relative-self._relative_fixed,dim=-1).max())
        dot=float(torch.abs((self.hand.data.body_quat_w[:,self._palm]*self._palm_quat).sum()))
        angle=2*math.acos(min(1.,dot))
        self.max_fixed_body_displacement=max(self.max_fixed_body_displacement,error)
        self.max_palm_displacement=max(self.max_palm_displacement,float(torch.linalg.vector_norm(palm-self._initial_palm)))
        self.max_palm_rotation=max(self.max_palm_rotation,angle)
        if error>.001 or angle>.005:
            self.score.failure='inactive_hand_moved'

    def _observation(self):
        ids=[self._y,self._z,self._index]
        angles,_=self._key_state()
        return np.r_[self.hand.data.joint_pos[0,ids].cpu().numpy(),
            self.hand.data.joint_vel[0,ids].cpu().numpy(),angles,
            self.score.count/self.score.repetitions,float(self.score._armed),self.score.active_key/87].astype(np.float32)

    def _info(self):
        result=super()._info()
        result.update(song='Twinkle Twinkle Little Star',notes_completed=self.score.count,
            total_notes=self.score.repetitions,hand_model='Shadow Hand',single_finger='index',
            wrist_motion='driven Y/Z slides',next_midi=None if self.score.success else self.score.active_key+21)
        return result

    def demonstration_action(self, control='demo'):
        if control=='idle':
            return np.array([-.09175,.03,0.])
        if control=='hold':
            return np.array([-.09175,0.,self.cfg.press_joint_angle])
        now=self.step_count*self.cfg.physics_dt
        old_note=self._demo_note
        duration=2. if old_note%7==6 else 1.
        if self.score.count!=old_note:
            if now-self._demo_start<duration:
                return np.array([self.layout[self.cfg.melody[old_note]-21].position[1]-.033,.03,0.])
            self._demo_note=self.score.count
            self._demo_start=now
            self._demo_y_start=float(self.hand.data.joint_pos[0,self._y])
        elapsed=now-self._demo_start
        y=self.layout[self.score.active_key].position[1]-.033
        def smooth(value):
            v=float(np.clip(value,0.,1.))
            return v*v*(3-2*v)
        if elapsed<.35:
            return np.array([self._demo_y_start+(y-self._demo_y_start)*smooth(elapsed/.25),.03,0.])
        if elapsed<.47:
            return np.array([y,.03*(1-smooth((elapsed-.35)/.12)),0.])
        if elapsed<.59:
            return np.array([y,0.,self.cfg.press_joint_angle*smooth((elapsed-.47)/.12)])
        release_at=1.74 if self._demo_note%7==6 else .74
        if elapsed<release_at:
            return np.array([y,0.,self.cfg.press_joint_angle])
        v=smooth((elapsed-release_at)/.14)
        if elapsed>release_at+.6 and self.score.count==self._demo_note:
            self.score.failure='note_timeout'
        return np.array([y,.03*v,self.cfg.press_joint_angle*(1-v)])
