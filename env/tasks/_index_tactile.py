"""A compact index pad: bundled libuipc FEM and TacEx's calibrated Taxim optics.

The gel backing follows the distal link. Gel and hinged keys are solved
together in UIPC; measured key angles are mirrored into the visible keyboard.
No other finger has a gel, optical renderer or tactile observation.
"""
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from pxr import UsdGeom, Gf, Vt
from uipc import Logger, builtin, view
from uipc.core import Engine, World, Scene, ContactSystemFeature
from uipc.geometry import Geometry
from uipc.constitution import StableNeoHookean, ElasticModuli, SoftPositionConstraint
from isaaclab.utils.math import matrix_from_quat
from tacex.simulation_approaches.gpu_taxim.sim import Taxim
from tacex_assets import TACEX_ASSETS_DATA_DIR
from envs._tactile_memory import box_mesh
from envs._force_task_utils import tetra_faces
from ._uipc_keyboard import UipcKeyboard,volume_mesh


class IndexTactile:
    image_shape = (480, 640)
    size = (.016, .012, .003)
    cells = (8, 6, 2)
    camera_distance = .020
    origin = np.array([0., -.006, .019])
    # Columns: u toward fingertip, v across finger, n outward from backing.
    sensor_to_distal = np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])

    def __init__(self, task):
        self.task = task
        self.dt = task.cfg.physics_dt
        self.body_id = task.hand.find_bodies('robot0_ffdistal')[0][0]
        self.key_forces = np.zeros((88,3))
        self._cache = None
        self.peak_indentation = 0.
        self.peak_displacement = 0.
        self.min_volume_ratio = 1.
        self.solve_seconds = 0.
        self.workspace = Path(task.cfg.tactile_workspace)
        self.workspace.mkdir(parents=True,exist_ok=True)
        Logger.set_level(Logger.Error)
        config = Scene.default_config()
        config['dt'] = self.dt
        config['gravity'] = [[0.],[0.],[0.]]
        config['contact']['d_hat'] = .0001
        config['newton']['velocity_tol'] = .001
        config['newton']['max_iter'] = 64
        config['sanity_check'] = {'enable':True,'mode':'quiet'}
        self.scene = Scene(config)
        contacts = self.scene.contact_tabular()
        contacts.default_model(.6,1.e9,enable=False)
        gel_contact, key_contact = contacts.create('index_gel'),contacts.create('keys')
        contacts.insert(gel_contact,key_contact,.6,1.e9)
        snk, constraint = StableNeoHookean(),SoftPositionConstraint()
        self.scene.constitution_tabular().insert(snk)
        self.scene.constitution_tabular().insert(constraint)
        self.rest_sensor,self.tets = box_mesh(self.size,self.cells)
        self.rest_distal = self.rest_sensor@self.sensor_to_distal.T+self.origin
        self.back = np.flatnonzero(np.isclose(self.rest_sensor[:,2],0.))
        self.front = np.flatnonzero(np.isclose(self.rest_sensor[:,2],self.size[2]))
        self.gel_aim = self._gel_targets()
        gel = volume_mesh(self.gel_aim,self.tets)
        snk.apply_to(gel,ElasticModuli.youngs_poisson(task.cfg.tactile_youngs_modulus_Pa,.45),mass_density=1100.)
        constraint.apply_to(gel,1.e7)
        gel_contact.apply_to(gel)
        gel_obj = self.scene.objects().create('index_gel')
        self.gel_slot,_ = gel_obj.geometries().create(gel)
        self._callbacks = []
        def animate_gel(info):
            geo=info.geo_slots()[0].geometry()
            view(geo.vertices().find(builtin.is_constrained))[self.back]=1
            view(geo.vertices().find(builtin.aim_position))[self.back]=self.gel_aim[self.back,:,None]
        self._callbacks.append(animate_gel)
        self.scene.animator().insert(gel_obj,animate_gel)
        # Fixed wrist X places the gel ahead of the black keys. Include every
        # white key the bounded lateral action can reach, also wrong pitches.
        reach_low=task.action_space.low[0]+.033-self.size[1]/2-.002
        reach_high=task.action_space.high[0]+.033+self.size[1]/2+.002
        self.key_indices=[i for i,key in enumerate(task.layout)
            if not key.black and key.position[1]+key.size[1]/2>=reach_low
            and key.position[1]-key.size[1]/2<=reach_high]
        self.keyboard=UipcKeyboard(self.scene,task.layout,self.key_indices,self.dt,key_contact)
        self.engine=Engine('cuda',str(self.workspace))
        self.world=World(self.engine)
        self.world.init(self.scene)
        self.world.retrieve()
        self.world.dump()
        offsets=[int(slot.geometry().meta().find(builtin.global_vertex_offset).view()[0]) for slot in self.keyboard.slots]
        self._key_node_map=np.full(max(offsets)+8,-1,dtype=np.int64)
        for i,start in enumerate(offsets):
            self._key_node_map[start:start+8]=np.arange(8)+i*8
        self._gel_offset=int(self.gel_slot.geometry().meta().find(builtin.global_vertex_offset).view()[0])
        self._rest_volumes=self._volumes(self.rest_sensor)
        self._create_visuals()
        self._create_optics()

    def _volumes(self,points):
        vertices=points[self.tets]
        return np.linalg.det(np.stack([vertices[:,i]-vertices[:,0] for i in (1,2,3)],axis=-1))

    def _body_pose(self):
        hand=self.task.hand.data
        position=hand.body_pos_w[0,self.body_id].cpu().numpy()
        rotation=matrix_from_quat(hand.body_quat_w[0,self.body_id]).cpu().numpy()
        return position,rotation

    def _gel_targets(self):
        position,rotation=self._body_pose()
        return self.rest_distal@rotation.T+position

    def _create_visuals(self):
        mesh=UsdGeom.Mesh.Define(self.task.stage,'/World/IndexTactile/Gel')
        faces=tetra_faces(self.tets)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(self.gel_aim.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([(.12,.55,.68)])
        self.visual_mesh=mesh
        housing=UsdGeom.Cube.Define(self.task.stage,'/World/Hand/robot0_ffdistal/TactileHousing')
        housing.CreateSizeAttr(1.)
        housing.AddTranslateOp().Set(Gf.Vec3d(0.,-.004,.019))
        housing.AddScaleOp().Set(Gf.Vec3d(.014,.006,.018))
        housing.CreateDisplayColorAttr([(.10,.15,.19)])

    def _create_optics(self):
        calibration=Path(TACEX_ASSETS_DATA_DIR)/'Sensors/GelSight_Mini/calibs/640x480'
        self.taxim=Taxim(calib_folder=calibration,backend='torch',device=self.task.sim.device)
        self.taxim.sensor_params.pixmm=self.size[0]*1000/self.image_shape[1]
        self.surface_rest=self.rest_sensor[self.front].reshape(self.cells[0]+1,self.cells[1]+1,3)
        self.u,self.v=self.surface_rest[:,0,0],self.surface_rest[0,:,1]
        # Sample pixel centres across the entire pad, without the old 7% crop.
        self.pixel_u=self.u[0]+(np.arange(self.image_shape[1])+.5)*np.ptp(self.u)/self.image_shape[1]
        self.pixel_v=self.v[0]+(np.arange(self.image_shape[0])+.5)*np.ptp(self.v)/self.image_shape[0]
        self.pixels=np.stack(np.meshgrid(self.pixel_u,self.pixel_v),-1)
        mu=np.arange(self.u[0]+.0015,self.u[-1],.002)
        mv=np.arange(self.v[0]+.0015,self.v[-1],.002)
        self.marker_uv=np.stack(np.meshgrid(mu,mv,indexing='ij'),-1).reshape(-1,2)

    def reset(self):
        if not self.world.recover(0):
            raise RuntimeError('UIPC could not restore the initial sensor state')
        self.world.retrieve()
        self.key_forces.fill(0.)
        self.keyboard.reset()
        self._cache=None
        self.peak_indentation=self.peak_displacement=self.solve_seconds=0.
        self.min_volume_ratio=1.

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
        self.gel_aim=self._gel_targets()
        start=perf_counter()
        self.world.advance()
        self.world.retrieve()
        self.solve_seconds+=perf_counter()-start
        matrices=self.keyboard.update()
        export=Geometry()
        self.world.features().find(ContactSystemFeature.FeatureName).contact_gradient(export)
        nodal_force=np.zeros((len(self.key_indices)*8,3))
        gel_force=np.zeros(3)
        attrs=export.instances()
        if attrs.size():
            global_idx=np.asarray(attrs.find('i').view()).reshape(-1)
            # Native contact gradients include dt^2; divide to obtain newtons.
            force=-np.asarray(attrs.find('grad').view()).reshape(-1,3)/(self.dt**2)
            in_range=(global_idx>=0)&(global_idx<len(self._key_node_map))
            idx=self._key_node_map[global_idx[in_range]]
            valid=idx>=0
            np.add.at(nodal_force,idx[valid],force[in_range][valid])
            on_gel=(global_idx>=self._gel_offset)&(global_idx<self._gel_offset+len(self.rest_sensor))
            gel_force=force[on_gel].sum(0)
        nodal_force=nodal_force.reshape(-1,8,3)
        self.key_forces.fill(0.)
        world=np.asarray(self.gel_slot.geometry().positions().view()).reshape(-1,3)
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
        self._cache=None
        self._mirror_keys()

    def observation(self):
        if self._cache is not None:
            return self._cache
        world=np.asarray(self.gel_slot.geometry().positions().view()).reshape(-1,3)
        ratio=float((self._volumes(world)/self._rest_volumes).min())
        self.min_volume_ratio=min(self.min_volume_ratio,ratio)
        if ratio<.05:
            raise RuntimeError(f'Gel tetrahedra collapsed or inverted (minimum volume ratio {ratio:.4f})')
        position,rotation=self._body_pose()
        sensor=((world-position)@rotation-self.origin)@self.sensor_to_distal
        surface=sensor[self.front]
        rest=self.rest_sensor[self.front]
        fields=np.c_[surface[:,2],surface-rest]
        image=LinearNDInterpolator(Delaunay(surface[:,:2]),fields)(self.pixels)
        valid=np.isfinite(image[...,0])
        depth=self.camera_distance+np.where(valid,image[...,0]-self.size[2],0.)
        deformation=np.where(valid[...,None],image[...,1:],0.)
        indentation=np.maximum(0.,self.camera_distance-depth)
        self.peak_indentation=max(self.peak_indentation,float(indentation.max()))
        self.peak_displacement=max(self.peak_displacement,float(np.linalg.norm(deformation,axis=-1).max()))
        height=torch.tensor(-indentation[None]*1000,dtype=torch.float32,device=self.task.sim.device)
        with torch.inference_mode():
            optical=self.taxim.render_direct(height,with_shadow=False)
        rgb=np.rint(optical[0].movedim(0,-1).cpu().numpy()*255).astype(np.uint8)
        markers=RegularGridInterpolator((self.u,self.v),sensor[self.front].reshape(*self.surface_rest.shape))(self.marker_uv)
        rgb_marker=rgb.copy()
        rows=(markers[:,1]-self.v[0])/np.ptp(self.v)*self.image_shape[0]-.5
        cols=(markers[:,0]-self.u[0])/np.ptp(self.u)*self.image_shape[1]-.5
        ink=np.zeros(self.image_shape)
        sigma=.8*self.image_shape[1]/160
        for row,col in zip(rows,cols):
            # A local six-sigma window avoids full-frame work for each marker.
            r0=max(0,int(np.floor(row-6*sigma)))
            r1=min(self.image_shape[0],int(np.ceil(row+6*sigma))+1)
            c0=max(0,int(np.floor(col-6*sigma)))
            c1=min(self.image_shape[1],int(np.ceil(col+6*sigma))+1)
            if r1<=r0 or c1<=c0:
                continue
            rr,cc=np.ogrid[r0:r1,c0:c1]
            ink[r0:r1,c0:c1]=np.maximum(ink[r0:r1,c0:c1],
                np.exp(-((rr-row)**2+(cc-col)**2)/(2*sigma**2)))
        rgb_marker=np.rint(rgb_marker*(1.-.85*ink[...,None])).astype(np.uint8)
        self._cache=dict(rgb=rgb_marker)
        return self._cache

    def update_visual(self):
        world=np.asarray(self.gel_slot.geometry().positions().view()).reshape(-1,3)
        self.visual_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(world.astype(np.float32)))

    def metadata(self):
        return dict(finger='index',link='robot0_ffdistal',sensor_count=1,
            backend='bundled libuipc 0.9 FEM + TacEx Taxim',
            pad_size_m=list(self.size),image_shape=list(self.image_shape),
            coordinates='u toward fingertip; v across finger; n outward',
            observation_fields=['rgb'],image_dtype='uint8',colour_order='RGB',
            image_field_of_view='entire 16 x 12 mm pad, pixel-centre sampling, no crop',
            camera_model='orthographic, 20 mm unloaded depth',
            calibration='GelSight Mini optical calibration transferred to a compact virtual pad',
            coupling='prescribed backing; gel and hinged keys solved together in UIPC; key angles mirrored to USD',
            peak_indentation_m=self.peak_indentation,peak_displacement_m=self.peak_displacement,
            max_key_hinge_error_m=self.keyboard.max_hinge_error,
            max_key_affine_shape_error=self.keyboard.max_shape_error,
            min_tet_volume_ratio=self.min_volume_ratio,gel_nodes=len(self.rest_sensor),
            gel_tetrahedra=len(self.tets),youngs_modulus_Pa=self.task.cfg.tactile_youngs_modulus_Pa,poisson_ratio=.45,
            simulated_key_midi=[i+21 for i in self.key_indices],
            command_slew_limits=dict(rail_y_m_per_s=self.task.cfg.tactile_action_rates[0],
                lift_z_m_per_s=self.task.cfg.tactile_action_rates[1],index_rad_per_s=self.task.cfg.tactile_action_rates[2]),
            command_filter_seconds=self.task.cfg.tactile_command_filter_seconds,
            uipc_solve_wall_seconds=self.solve_seconds)

    def close(self):
        self._callbacks.clear()
        self.keyboard._animate=None
        self._cache=None
        self.taxim=None
        self.world=None
        self.engine=None
        self.scene=None
        self.task=None
