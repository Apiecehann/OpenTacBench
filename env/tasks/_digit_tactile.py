"""Original DIGIT optics and gel with an integrated index-tip appearance.

PhysX supplies the actual key poses and contact forces. TACTO's original rigid
contact/force-to-appearance model is used; this backend does not claim FEM gel
deformation or GelSight Mini calibration.
"""
import json
import os
from pathlib import Path
import select
import subprocess
import numpy as np
from pxr import Gf,UsdGeom,UsdPhysics
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import matrix_from_quat


class DigitTactile:
    display_name='DIGIT - index'
    physics_backend='physx'

    def __init__(self,task):
        self.task=task
        self.image_shape=(task.cfg.tactile_image_width*3//4,task.cfg.tactile_image_width)
        self.peak_indentation=0.
        repo=Path(__file__).resolve().parents[2]
        root=repo/'.cache/piano_tactile/digit'
        python=root/'venv/bin/python'
        if not python.is_file():
            raise RuntimeError('Install the original DIGIT backend with scripts/install_piano_digit.sh')
        environment=dict(os.environ)
        for name in ('PYTHONPATH','PYTHONHOME'): environment.pop(name,None)
        environment.update(PYTHONNOUSERSITE='1',PYOPENGL_PLATFORM='egl')
        self._log=(root/'renderer.log').open('w')
        self._process=subprocess.Popen([str(python),str(Path(__file__).with_name('_digit_renderer.py')),
            str(root),str(task.cfg.tactile_image_width)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self._log,env=environment)
        self._geometry=self._receive()
        self._mount()
        self._send(dict(command='setup',keys={str(k.midi):k.size for k in task.layout}))
        self._receive()

    def _send(self,value):
        self._process.stdin.write(json.dumps(value).encode()+b'\n')
        self._process.stdin.flush()

    def _receive(self):
        if not select.select([self._process.stdout],[],[],30)[0]:
            raise RuntimeError('DIGIT renderer timed out; inspect .cache/piano_tactile/digit/renderer.log')
        line=self._process.stdout.readline()
        if not line: raise RuntimeError('DIGIT renderer stopped; inspect .cache/piano_tactile/digit/renderer.log')
        return json.loads(line)

    def _mount(self):
        from envs._force_task_utils import _material
        from pxr import UsdShade
        stage=self.task.stage
        distal=stage.GetPrimAtPath('/World/Hand/robot0_ffdistal')
        # Face the original gel toward the key at the nominal pressed joint
        # angle. An untilted mount touches only the gel's far edge, outside
        # the original camera's field of view. Keep the apex at the same
        # distal point; neither the mesh nor optical calibration is scaled.
        self.distal_sensor=np.eye(4)
        self.distal_sensor[:3,:3]=np.array([[0,1,0],[-1,0,0],[0,0,1]])@Rotation.from_euler('y',-self.task.cfg.press_joint_angle).as_matrix()
        self.distal_sensor[:3,3]=np.array([0.,-.012,.019])-self.distal_sensor[:3,:3]@np.array([.022,0.,.015])
        cache=UsdGeom.XformCache()
        hand_world=np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Hand'))).T
        distal_world=np.array(cache.GetLocalToWorldTransform(distal)).T
        transform=np.linalg.solve(hand_world,distal_world@self.distal_sensor)
        body=UsdGeom.Xform.Define(stage,'/World/Hand/Digit')
        body.AddTransformOp().Set(Gf.Matrix4d(transform.T.copy()))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.1)  # Original DIGIT URDF.
        # Keep the hand's original distal shell and expose only the dark gel.
        # The external white CAD housing is deliberately absent from the scene;
        # the original TACTO optical frame and physical sensing mesh stay intact.
        mesh=UsdGeom.Mesh.Define(stage,'/World/Hand/Digit/gel')
        vertices=self._geometry['gel_vertices']; faces=np.asarray(self._geometry['gel_faces'])
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        material=_material(stage,'/World/Hand/Digit/Looks/Gel',(.035,.045,.055),.82)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        joint=UsdPhysics.FixedJoint.Define(stage,'/World/Hand/IndexDigitMount')
        joint.CreateBody0Rel().SetTargets([distal.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        pose=Gf.Transform(Gf.Matrix4d(self.distal_sensor.T.copy()))
        joint.CreateLocalPos0Attr(Gf.Vec3f(pose.GetTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(pose.GetRotation().GetQuat()))
        # Keep the bending finger's other links from pressing keys before its
        # sensing surface. Other hand/key collision rules are unchanged.
        group=UsdPhysics.CollisionGroup.Define(stage,'/World/IndexNonSensingContactGroup')
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([
            '/World/Hand/robot0_ffproximal','/World/Hand/robot0_ffmiddle','/World/Hand/robot0_ffdistal'])
        group.CreateFilteredGroupsRel().SetTargets(['/World/KeyContactGroup'])

    def setup(self):
        self._body=self.task.hand.find_bodies('Digit')[0][0]
        self._key_body=self.task.keys.find_bodies('Body')[0][0]

    def reset(self): self.peak_indentation=0.
    def reset_reference(self): pass  # TACTO uses its fixed original background.
    def update_visual(self): pass  # PhysX already updates the mounted geometry.
    def step(self): pass

    def key_forces(self):
        data=self.task.key_contacts.data.force_matrix_w
        if data is None: raise RuntimeError('DIGIT requires native filtered key-contact forces')
        return np.abs(data.reshape(-1,3)[self.task._contact_order,2].cpu().numpy())

    def observation(self):
        task=self.task
        position=task.hand.data.body_pos_w[0,self._body].cpu().numpy()
        rotation=matrix_from_quat(task.hand.data.body_quat_w[0,self._body]).cpu().numpy()
        forces=self.key_forces()
        objects={}
        for i,key in enumerate(task.layout):
            index=task._key_order[i]
            p=task.keys.data.body_pos_w[index,self._key_body].cpu().numpy()
            r=matrix_from_quat(task.keys.data.body_quat_w[index,self._key_body]).cpu().numpy()
            objects[str(key.midi)]=[p.tolist(),Rotation.from_matrix(r).as_euler('xyz').tolist(),float(forces[i])]
        self._send(dict(command='render',sensor=[position.tolist(),Rotation.from_matrix(rotation).as_euler('xyz').tolist()],objects=objects))
        header=self._receive()
        data=bytearray()
        while len(data)<header['bytes']:
            part=self._process.stdout.read(header['bytes']-len(data))
            if not part: raise RuntimeError('Incomplete DIGIT RGB frame')
            data.extend(part)
        self.peak_indentation=max(self.peak_indentation,header['indentation_m'])
        return dict(rgb=np.frombuffer(data,np.uint8).reshape(*self.image_shape,3).copy())

    def metadata(self):
        return dict(finger='index',link='robot0_ffdistal',sensor_count=1,sensor_model='DIGIT',
            backend='Meta TACTO 0.0.3 Renderer; original DIGIT config, CAD and reference image',
            calibration='Original TACTO DIGIT camera, lighting, gel geometry, noise and real background',
            contact_model='PhysX rigid key/gel contact; original TACTO force-to-appearance model; no FEM gel',
            scene_representation='Original Shadow distal shell and dark DIGIT gel; external CAD housing omitted',
            case_size_m=self._geometry['case_size_m'],image_shape=list(self.image_shape),
            observation_fields=['rgb'],image_dtype='uint8',colour_order='RGB',peak_indentation_m=self.peak_indentation)

    def close(self):
        process=getattr(self,'_process',None)
        if process is not None and process.poll() is None:
            self._send(dict(command='close'))
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=3)
        if hasattr(self,'_log'): self._log.close()
