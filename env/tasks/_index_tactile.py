"""Mount an unscaled GelSight Mini on the index and use the original sensor API.

Only keyboard mechanics and mounting live here. Cameras, calibration, FEM gel,
marker motion and image generation belong to envs.sensors.tactile / TacEx.
"""
from pathlib import Path
import numpy as np
import torch
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_uipc import UipcSim,UipcSimCfg
from tacex.simulation_approaches.fem_based.sim.gelpad_info import SURFACE_FACES
from envs.sensors.tactile import VisualTactileSensor,create_gelsight_mini_cfg
from ._uipc_keyboard import UipcKeyboard


class IndexTactile:
    image_shape=(480,640)

    def __init__(self,task):
        self.task=task
        self.dt=task.cfg.physics_dt
        self.key_forces=np.zeros((88,3))
        self.peak_indentation=0.
        self.uipc=UipcSim(UipcSimCfg(dt=self.dt,workspace=task.cfg.tactile_workspace,
            gravity=(0.,0.,0.),ground_height=-1.,
            contact=UipcSimCfg.Contact(d_hat=.0001),
            newton=UipcSimCfg.Newton(velocity_tol=.001,max_iter=64)))
        self._mount_mini()
        cfg=create_gelsight_mini_cfg('/World/Hand/GelSightMini',
            '/World/IndexGel','GelSightMini',name='index',
            resolution=(640,480),data_type=['tactile_rgb','marker_rgb','marker_motion'])
        self.sensor=VisualTactileSensor('index',cfg,task.hand,None,self.uipc)
        self.scene=self.uipc.scene
        contacts=self.scene.contact_tabular()
        contacts.default_model(.6,1.e9,enable=False)
        gel_contact,key_contact=contacts.create('index_gel'),contacts.create('keys')
        contacts.insert(gel_contact,key_contact,.6,1.e9)
        gel_contact.apply_to(self.sensor.gelpad.uipc_meshes[0])
        low,high=min(task.cfg.melody)-2,max(task.cfg.melody)+2
        self.key_indices=[i for i,k in enumerate(task.layout) if not k.black and low<=k.midi<=high]
        self.keyboard=UipcKeyboard(self.scene,task.layout,self.key_indices,self.dt,key_contact)

    def _mount_mini(self):
        # Reuse the original complete assembly. Change its rigid mounting pose
        # only; retain camera intrinsics/extrinsics, gel mesh and unit scale.
        asset=str(Path(TACEX_ASSETS_DATA_DIR)/'Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_attached.usda')
        source=Usd.Stage.Open(asset)
        root=source.GetDefaultPrim().GetPath()
        case=source.GetPrimAtPath(root.AppendChild('gelsight_mini_case_left'))
        gel=source.GetPrimAtPath(root.AppendChild('gelpad_left'))
        cache=UsdGeom.XformCache()
        case_world=np.array(cache.GetLocalToWorldTransform(case)).T
        gel_world=np.array(cache.GetLocalToWorldTransform(gel)).T
        camera_world=np.array(cache.GetLocalToWorldTransform(case.GetChild('Camera'))).T
        mesh=gel.GetChild('mesh')
        points=np.asarray(mesh.GetAttribute('tet_points').Get(),float)
        front=np.unique(np.asarray(SURFACE_FACES['gsmini']).ravel())
        camera_points=(np.c_[points[front],np.ones(len(front))]@
            (np.linalg.solve(camera_world,gel_world)).T)[:,:3]
        self.gel_camera_size=np.ptp(camera_points,axis=0)
        # Camera USD +X along finger; +Y across finger; -Z out of the gel.
        distal_camera=np.eye(4)
        distal_camera[:3,:3]=[[0,1,0],[0,0,1],[1,0,0]]
        camera_front=(camera_points.min(0)+camera_points.max(0))/2
        # Keep the full-size camera housing outside the opaque finger mesh.
        distal_camera[:3,3]=np.array([0.,-.038,.019])-distal_camera[:3,:3]@camera_front
        distal_case=distal_camera@np.linalg.inv(np.linalg.solve(case_world,camera_world))
        distal=self.task.stage.GetPrimAtPath('/World/Hand/robot0_ffdistal')
        distal_world=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(distal)).T
        hand_world=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
            self.task.stage.GetPrimAtPath('/World/Hand'))).T
        for target,source_prim,transform in (
            ('/World/Hand/GelSightMini',case,np.linalg.solve(hand_world,distal_world@distal_case)),
            ('/World/IndexGel',gel,distal_world@distal_case@np.linalg.solve(case_world,gel_world))):
            prim=self.task.stage.DefinePrim(target)
            prim.GetReferences().AddReference(asset,str(source_prim.GetPath()))
            xform=UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            mount=Gf.Transform(Gf.Matrix4d(transform.T.copy()))
            xform.AddTranslateOp().Set(mount.GetTranslation())
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(mount.GetRotation().GetQuat()))
            xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(mount.GetScale()))
        joint=UsdPhysics.FixedJoint.Define(self.task.stage,'/World/Hand/IndexMiniMount')
        joint.CreateBody0Rel().SetTargets(['/World/Hand/robot0_ffdistal'])
        joint.CreateBody1Rel().SetTargets(['/World/Hand/GelSightMini'])
        mount=Gf.Transform(Gf.Matrix4d(distal_case.T.copy()))
        joint.CreateLocalPos0Attr(Gf.Vec3f(mount.GetTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(mount.GetRotation().GetQuat()))

    def setup(self):
        self.uipc.setup_sim()
        self.world=self.uipc.world
        self.sensor.setup()
        self.uipc.save_frame()
        offsets=[int(slot.geometry().meta().find('global_vertex_offset').view()[0]) for slot in self.keyboard.slots]
        self._key_node_map=np.full(max(offsets)+8,-1,dtype=np.int64)
        for i,start in enumerate(offsets): self._key_node_map[start:start+8]=np.arange(8)+i*8
        self.gel_slot=self.sensor.gelpad.geo_slot_list[0]
        self._gel_offset=int(self.gel_slot.geometry().meta().find('global_vertex_offset').view()[0])

    def reset(self):
        if not self.world.recover(0): raise RuntimeError('Could not restore the initial UIPC scene')
        self.world.retrieve()
        self.keyboard.reset()
        self.key_forces.fill(0.)
        self.peak_indentation=0.

    def _mirror_keys(self):
        task=self.task
        q=torch.zeros_like(task.keys.data.joint_pos)
        velocity=torch.zeros_like(q)
        q[task._key_order,0]=-torch.as_tensor(self.keyboard.angles,device=q.device,dtype=q.dtype)
        velocity[task._key_order,0]=-torch.as_tensor(self.keyboard.velocities,device=q.device,dtype=q.dtype)
        task.keys.write_joint_state_to_sim(q,velocity)
        task.sim.forward()
        task.keys.update(0.)

    def step(self):
        matrices=self.keyboard.update()
        nodal_force=np.zeros((len(self.key_indices)*8,3))
        gel_force=np.zeros(3)
        global_idx,gradient=self.uipc.get_contact_gradient()
        if len(global_idx):
            force=-gradient/self.dt**2
            in_range=(global_idx>=0)&(global_idx<len(self._key_node_map))
            idx=self._key_node_map[global_idx[in_range]]
            valid=idx>=0
            np.add.at(nodal_force,idx[valid],force[in_range][valid])
            on_gel=(global_idx>=self._gel_offset)&(global_idx<self._gel_offset+self.sensor.gelpad._vertex_count)
            gel_force=force[on_gel].sum(0)
        nodal_force=nodal_force.reshape(-1,8,3)
        self.key_forces.fill(0.)
        world=self.sensor.gelpad.vertex_positions
        touching=[]
        for i,key_id in enumerate(self.key_indices):
            key=self.task.layout[key_id]
            local=(world-matrices[i,:3,3])@matrices[i,:3,:3]
            near_top=(np.abs(local[:,2]-key.size[2]/2)<.00011)
            in_key=(np.abs(local[:,:2])<=np.array(key.size[:2])/2+.00011).all(1)
            if (near_top&in_key).any():
                touching.append(key_id)
                self.key_forces[key_id]=nodal_force[i].sum(0)
        if len(touching)==1:
            # With one touched key, action/reaction gives its exact gel load,
            # excluding mechanical-stop impacts from scoring and sound.
            self.key_forces[touching[0]]=-gel_force
        if not np.isfinite(self.key_forces).all():
            raise RuntimeError('Non-finite UIPC contact force')
        self._mirror_keys()

    def observation(self):
        self.sensor.update(self.dt*self.task.cfg.action_repeat,force_recompute=True)
        output=self.sensor.get_observations(['rgb_marker'])
        self.peak_indentation=max(self.peak_indentation,float(self.sensor.sensor.indentation_depth.max())/1000)
        return {'rgb':output['rgb_marker'].cpu().numpy()}

    def update_visual(self):
        self.uipc.update_render_meshes()

    def metadata(self):
        optical=self.sensor.sensor.optical_simulator
        surface=self.sensor.sensor.marker_motion_simulator.marker_motion_sim.get_surface_vertices_world().cpu().numpy()
        return dict(finger='index',link='robot0_ffdistal',sensor_count=1,
            backend='envs.sensors.tactile.VisualTactileSensor / bundled TacEx UIPC',
            calibration='original GelSight Mini; unscaled USD assembly; original camera and FEM marker model',
            pixel_size_mm=float(optical._taxim.sensor_params.pixmm),
            gel_camera_size_m=self.gel_camera_size.tolist(),
            gel_world_bounds_m=[surface.min(0).tolist(),surface.max(0).tolist()],
            image_shape=list(self.image_shape),observation_fields=['rgb'],
            image_dtype='uint8',colour_order='RGB',peak_indentation_m=self.peak_indentation,
            simulated_key_midi=[i+21 for i in self.key_indices],
            max_key_hinge_error_m=self.keyboard.max_hinge_error,
            max_key_affine_shape_error=self.keyboard.max_shape_error)

    def close(self):
        self.keyboard._animate=None
