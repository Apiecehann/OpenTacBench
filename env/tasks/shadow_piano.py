"""One Shadow Hand index finger plays a note sequence on physical piano keys.

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
from envs._piano_sequence import make_sequence,PianoAnnotations

TASK_INSTRUCTION = 'Play the requested notes with the index finger; move and lift the wrist between keys.'


@dataclass
class TaskCfg(PianoCfg):
    episode_seconds: float = 90.
    press_joint_angle: float = .55
    hand_position: tuple = (-.011, 0., .110)
    hand_rotation: tuple = (0., 2**-.5, -2**-.5, 0.)
    melody: tuple = TWINKLE
    tactile: bool = False
    tactile_workspace: str = '.cache/piano_tactile/world'
    tactile_image_width: int = 640
    tactile_sensor: str = 'gsmini'
    tactile_action_rates: tuple = (1., .375, 8.)  # Y m/s, Z m/s, index rad/s
    tactile_command_filter_seconds: float = .04
    piano_notes: object = None
    piano_dynamics: object = None
    piano_note_count: int = 6
    piano_require_dynamics: bool = False
    piano_wrong_key_policy: str = 'fail'
    piano_hold_seconds: object = None
    piano_seed: int | None = None

    def __post_init__(self):
        if self.tactile_sensor not in ('gsmini','digit'):
            raise ValueError('Choose the original gsmini or digit tactile backend')
        if self.tactile_image_width not in (320,640):
            raise ValueError('Tactile image width must be 320 or 640')
        if self.tactile:
            rates=np.asarray(self.tactile_action_rates,dtype=float)
            if (rates.shape not in ((3,),(4,)) or not np.isfinite(rates).all() or (rates<=0).any()
                    or not math.isfinite(self.tactile_command_filter_seconds)
                    or self.tactile_command_filter_seconds<=0):
                raise ValueError('Tactile command rates and filter time must be finite and positive')
        if self.tactile and self.press_joint_angle == .55:
            self.press_joint_angle = .50
        if self.tactile and self.hand_position == (-.011, 0., .110):
            # Offset the wrist for the full Mini housing at the pressed pose.
            self.hand_position = (.003, 0., .1355)
            if self.tactile_sensor=='digit':
                # Preserve the front-key position with the shorter fingertip.
                # Lower the wrist 2.5 mm for a complete physical key stroke
                # with the face-aligned gel, including the audio threshold.
                self.hand_position = (-.00946499, 0., .11018281)


class Task(PianoTask):
    def __init__(self, cfg=None, render_mode=None):
        self.tactile = None
        self._last_action = None
        cfg=cfg or TaskCfg()
        self._sequence_request=cfg.piano_notes if cfg.piano_notes is not None else cfg.melody
        self._set_sequence(cfg,np.random.default_rng(cfg.piano_seed))
        super().__init__(cfg, render_mode)
        y_limits=(-.16,.10)
        if self.cfg.tactile:
            workspace=set(self.cfg.melody)|{60,62,64,65,67,69,71}
            centers=[self.layout[m-21].position[1]-.033 for m in workspace]
            y_limits=(min(centers)-.008,max(centers)+.008)
        self.action_space = gym.spaces.Box(np.array([y_limits[0],0.,0.],np.float32),
            np.array([y_limits[1],.05,self.cfg.press_joint_angle],np.float32))
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(97,),np.float32)
        if self.cfg.tactile:
            self.tactile.setup()
            h,w = self.tactile.image_shape
            self.observation_space = gym.spaces.Dict(state=self.observation_space,
                tactile=gym.spaces.Dict(index=gym.spaces.Dict(
                    rgb=gym.spaces.Box(0,255,(h,w,3),np.uint8))))

    def _create_tactile(self):
        if self.cfg.tactile:
            if self.cfg.tactile_sensor=='digit':
                from ._digit_tactile import DigitTactile
                self.tactile=DigitTactile(self)
            else:
                from ._index_tactile import IndexTactile
                self.tactile = IndexTactile(self)

    def _set_sequence(self,cfg,rng,options=None):
        options=options or {}
        notes=options.get('notes',self._sequence_request)
        holds=options.get('hold_seconds',cfg.piano_hold_seconds)
        if holds is None and notes is not None and tuple(notes)==TWINKLE: holds=HOLD_SECONDS
        allow_black=hasattr(cfg,'arm_position') and (not cfg.tactile or cfg.tactile_sensor=='digit')
        pool=tuple(m for m in range(60,72) if allow_black or m%12 not in (1,3,6,8,10))
        self.sequence=make_sequence(notes,count=cfg.piano_note_count,
            dynamics=options.get('dynamics',cfg.piano_dynamics),holds=holds,rng=rng,pool=pool)
        if any(m%12 in (1,3,6,8,10) for m in self.sequence.midi):
            if not hasattr(cfg,'arm_position') or (cfg.tactile and cfg.tactile_sensor!='digit'):
                raise ValueError('Black-key sequences require the arm task with DIGIT or without tactile sensing')
        if hasattr(cfg,'arm_position'):
            from envs._piano import keyboard_layout
            layout=keyboard_layout()
            if any(not -.32<=layout[m-21].position[1]-.033<=.26 for m in self.sequence.midi):
                raise ValueError('The requested notes exceed the arm task lateral workspace')
        cfg.melody=self.sequence.midi
        self.piano_annotations=PianoAnnotations(self.sequence,embodiment='hand')

    def _new_score(self):
        labels=self.piano_annotations
        return SingleKeyScore(key_sequence=tuple(m-21 for m in self.sequence.midi),
            note_hold_seconds=self.sequence.holds,require_dynamics=self.cfg.piano_require_dynamics,
            force_targets=tuple(labels.targets),force_tolerances=tuple(labels.tolerances),
            wrong_key_policy=self.cfg.piano_wrong_key_policy)

    def _key_contact_cfg(self):
        cfg=super()._key_contact_cfg()
        if self.cfg.tactile and self.cfg.tactile_sensor=='digit':
            cfg.filter_prim_paths_expr=['/World/Hand/Digit']
        return cfg

    def _create_keyboard(self):
        super()._create_keyboard()
        if self.cfg.tactile and self.cfg.tactile_sensor=='gsmini':
            # UIPC owns gel/key contact and the hinged key dynamics. These
            # articulations mirror measured angles for scene rendering only.
            for key in self.layout:
                path=f'/World/Keyboard/Key_{key.midi}'
                UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(path+'/Body')).CreateCollisionEnabledAttr(False)
                joint=UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(path+'/hinge'))
                joint.GetLowerLimitAttr().Set(-6.)
                joint.GetUpperLimitAttr().Set(2.)

    def _hand_cfg(self):
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
        return cfg

    def _create_hand(self):
        self.hand = Articulation(self._hand_cfg())
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
        self._mount_wrist()
        self._finish_hand()
        if self.cfg.tactile:
            # The sensing surface supplies index contact; avoid duplicates.
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand/robot0_ffdistal')):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    def _mount_wrist(self):
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

    def _finish_hand(self):
        from envs._force_task_utils import _material
        looks={
            'shell':_material(self.stage,'/World/Hand/Looks/Shell',(.74,.79,.83),.28,.45),
            'metal':_material(self.stage,'/World/Hand/Looks/Metal',(.35,.43,.50),.25,.75),
            'pad':_material(self.stage,'/World/Hand/Looks/Pad',(.035,.045,.055),.82),
            'base':_material(self.stage,'/World/Hand/Looks/Base',(.025,.035,.050),.32,.55)}
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Hand')):
            path=str(prim.GetPath()).lower()
            if path.startswith('/world/hand/arm/'): continue
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
        seed=kwargs.get('seed',self.cfg.piano_seed)
        self._sequence_seed=seed
        self._set_sequence(self.cfg,np.random.default_rng(seed),kwargs.get('options'))
        if self.tactile is not None and self.tactile.physics_backend=='uipc':
            supported={self.layout[i].midi for i in self.tactile.key_indices}
            if not set(self.sequence.midi)<=supported:
                raise ValueError('The new notes exceed this Mini keyboard; construct a task with the wider note range')
        self._last_action=None
        self._audio_active=np.zeros(88,bool)
        self._audio_peaks=np.zeros(88)
        if self.tactile is not None:
            self.tactile.reset()
        result=super().reset(**kwargs)
        self._demo_note=0
        self._demo_start=0.
        self._demo_y_start=float(self._wrist_state()[0])
        self.max_palm_rotation=0.
        if self.tactile is not None:
            self.tactile.reset_reference()
            self.tactile.update_visual()
            return self._observation(), self._info()
        return result

    def _physics_step(self, target):
        super()._physics_step(target)
        if self.tactile is not None:
            self.tactile.step()

    def _key_state(self):
        angles,forces=super()._key_state()
        if self.tactile is not None and self.tactile.physics_backend=='uipc':
            angles=self.tactile.keyboard.angles.copy()
            forces=np.abs(self.tactile.key_forces[:,2])
        elif self.tactile is not None:
            forces=self.tactile.key_forces()
        return angles,forces

    def _audio_forces(self, angles, forces):
        if self.tactile is None:
            return forces
        # A piano hammer fires once per key stroke. Use measured key travel
        # for the latch and actual contact peaks for loudness.
        self._audio_peaks=np.maximum(self._audio_peaks,forces)
        self._audio_active|=angles>=self.score.press_angle
        up=angles<=self.score.release_angle
        self._audio_active[up]=False
        signal=np.where(self._audio_active,self._audio_peaks,0.)
        self._audio_peaks[up]=0.
        return signal

    def _before_render(self):
        if self.tactile is not None:
            self.tactile.update_visual()

    def close(self):
        if self.tactile is not None:
            self.tactile.close()
        super().close()

    def _filter_action(self, action):
        a=np.asarray(action,dtype=float)
        if a.shape!=self.action_space.shape or not np.isfinite(a).all():
            raise ValueError(f'Action must contain {self.action_space.shape[0]} finite wrist/finger commands')
        a=np.clip(a,self.action_space.low,self.action_space.high)
        if self.cfg.tactile:
            # Position targets can otherwise command an impulsive lift/flex
            # combination that collapses a millimetre-scale soft gel.
            previous=(self._last_action if self._last_action is not None else
                self._action_state())
            alpha=1-math.exp(-self.cfg.physics_dt*self.cfg.action_repeat/self.cfg.tactile_command_filter_seconds)
            a=previous+alpha*(a-previous)
            delta=np.asarray(self.cfg.tactile_action_rates)*self.cfg.physics_dt*self.cfg.action_repeat
            a=np.clip(a,previous-delta,previous+delta)
            self._last_action=a.copy()
        return a

    def _action_state(self):
        return self._wrist_state()

    def _action_target(self, action):
        a=self._filter_action(action)
        q=self._rest.clone()
        q[0,[self._y,self._z,self._index]]=torch.as_tensor(a,dtype=q.dtype,device=q.device)
        return q

    def _wrist_state(self):
        return self.hand.data.joint_pos[0,[self._y,self._z,self._index]].cpu().numpy()

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
        state = np.r_[self.hand.data.joint_pos[0,ids].cpu().numpy(),
            self.hand.data.joint_vel[0,ids].cpu().numpy(),angles,
            self.score.count/self.score.repetitions,float(self.score._armed),self.score.active_key/87].astype(np.float32)
        if self.tactile is None:
            return state
        return dict(state=state,tactile=dict(index=self.tactile.observation()))

    def _info(self):
        result=super()._info()
        result.update(song='Twinkle Twinkle Little Star' if self.sequence.midi==TWINKLE else 'Custom piano sequence',notes_completed=self.score.count,
            total_notes=self.score.repetitions,hand_model='Shadow Hand',single_finger='index',
            wrist_motion='driven Y/Z slides',next_midi=None if self.score.success else self.score.active_key+21)
        result['piano_sequence']=self.sequence.as_dict()
        result['piano_instruction']=self.sequence.instruction()
        result['require_dynamics']=self.cfg.piano_require_dynamics
        result['piano_seed']=getattr(self,'_sequence_seed',self.cfg.piano_seed)
        result['piano_dynamic_codes']=dict(idle=0,light=1,heavy=2)
        result['wrong_key_policy']=self.cfg.piano_wrong_key_policy
        result['mistakes']=self.score.mistakes
        result['piano']=self.piano_annotations.records[-1] if self.piano_annotations.records else self.piano_annotations.update(
            self.step_count,self.cfg.physics_dt,self.score.count,np.zeros(88))
        return result

    def demonstration_action(self, control='demo'):
        if control=='idle':
            return np.array([-.09175,.03,0.])
        if control=='hold':
            return np.array([-.09175,0.,self.cfg.press_joint_angle])
        now=self.step_count*self.cfg.physics_dt
        old_note=self._demo_note
        duration=((2.4 if old_note%7==6 else 1.2) if self.cfg.tactile
                  else (2. if old_note%7==6 else 1.))
        if self.score.count!=old_note:
            if now-self._demo_start<duration:
                return np.array([self.layout[self.cfg.melody[old_note]-21].position[1]-.033,.03,0.])
            self._demo_note=self.score.count
            self._demo_start=now
            self._demo_y_start=float(self._wrist_state()[0])
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
        release_at=((2.10 if self._demo_note%7==6 else .90) if self.cfg.tactile
                    else (1.74 if self._demo_note%7==6 else .74))
        if elapsed<release_at:
            return np.array([y,0.,self.cfg.press_joint_angle])
        v=smooth((elapsed-release_at)/.14)
        if elapsed>release_at+.6 and self.score.count==self._demo_note:
            self.score.failure='note_timeout'
        return np.array([y,.03*v,self.cfg.press_joint_angle*(1-v)])
