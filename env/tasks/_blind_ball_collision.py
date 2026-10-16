"""Original robot convex colliders shared with UIPC, and reaction wrenches.

The USD/PhysX geometry remains the visible robot. These native UIPC affine
bodies use the asset's existing convexHull shapes and are appended after the
rendered UipcObjects, so they do not change tactile optics or render offsets.
"""
import numpy as np
import torch
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
from isaaclab.utils.math import matrix_from_quat,convert_quat
from uipc import builtin,view
from uipc.constitution import AffineBodyConstitution,SoftTransformConstraint
from ._uipc_keyboard import volume_mesh
from ._blind_ball_models import convex_collision_mesh,convex_box_penetration


def enable_fixture_collision(prim):
    """The same static mesh must block the PhysX arm and the UIPC balls."""
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
            collision=PhysxSchema.PhysxCollisionAPI.Apply(child)
            # Small positive separation absorbs PhysX convex cooking/contact
            # tolerances even when a position controller keeps pushing a wall.
            collision.CreateContactOffsetAttr(.002)
            collision.CreateRestOffsetAttr(.001)


class RobotBallCollisions:
    def __init__(self,task,uipc,gel_contact,fixture_contact):
        self.task,self.uipc=task,uipc
        self.parts=[]
        self.fixtures=[]
        self.max_proxy_error=0.
        self.max_ball_penetration=0.
        self.max_fixture_penetration=0.
        self.max_reaction_force=0.
        contacts=uipc.scene.contact_tabular()
        contact=contacts.create('robot_rigid_colliders')
        contacts.insert(contact,contact,.5,1.e9,enable=False)
        contacts.insert(contact,gel_contact,.5,1.e9,enable=False)
        # Static fixtures are handled by PhysX for the robot, by UIPC for balls.
        contacts.insert(contact,fixture_contact,.5,1.e9,enable=False)
        contacts.insert(contact,contacts.default_element(),.5,1.e9)
        abd,drive=AffineBodyConstitution(),SoftTransformConstraint()
        uipc.scene.constitution_tabular().insert(abd)
        uipc.scene.constitution_tabular().insert(drive)
        self.obj=None
        cache=UsdGeom.XformCache()
        root=task.stage.GetPrimAtPath(task.robot.cfg.prim_path)
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull':
                raise ValueError('Expected the original robot convexHull collider: '+str(prim.GetPath()))
            body=prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body:raise ValueError('Robot collider has no rigid body: '+str(prim.GetPath()))
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            transform=np.asarray(cache.GetLocalToWorldTransform(prim))@np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)))
            points=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
            points,tets,planes,edges=convex_collision_mesh(points)
            mesh=volume_mesh(points,tets)
            view(mesh.transforms())[0]=np.asarray(cache.GetLocalToWorldTransform(body)).T
            abd.apply_to(mesh,1.e8,mass_density=1000.)
            drive.apply_to(mesh,np.array([1000.,1000.]))
            contact.apply_to(mesh)
            self.parts.append(dict(path=str(prim.GetPath()),body=body.GetName(),mesh=mesh,
                points=points,planes=planes,edges=edges))
        def animate(info):
            for part,slot in zip(self.parts,info.geo_slots()):
                geo=slot.geometry()
                view(geo.instances().find(builtin.is_constrained))[:]=1
                view(geo.instances().find(builtin.aim_transform))[0]=self.targets[part['body_id']]
        self._animate=animate

    def body_transforms(self):
        poses=self.task.robot.root_physx_view.get_link_transforms()[0]
        matrices=np.tile(np.eye(4),(len(poses),1,1))
        matrices[:,:3,:3]=matrix_from_quat(convert_quat(poses[:,3:7],to='wxyz')).cpu().numpy()
        matrices[:,:3,3]=poses[:,:3].cpu().numpy()
        return matrices

    def initialize(self):
        transforms=self.body_transforms()
        self.targets=transforms
        # UipcObject creates its scene slots on PLAY, inside sim.reset(). Add
        # these non-rendered shapes afterwards to preserve SceneIO surface order.
        self.obj=self.uipc.scene.objects().create('original_robot_colliders')
        for part in self.parts:
            part['body_id']=self.task.robot.find_bodies(part['body'])[0][0]
            view(part['mesh'].transforms())[0]=transforms[part['body_id']]
            view(part['mesh'].instances().find(builtin.aim_transform))[0]=transforms[part['body_id']]
            part['slot'],_=self.obj.geometries().create(part['mesh'])
        self.uipc.scene.animator().insert(self.obj,self._animate)
        # Follow the original UipcIsaacAttachments lifecycle: read PhysX before
        # entering UIPC, and only copy CPU targets inside its animation callback.
        self.task.sim.add_physics_callback('ball_robot_collider_poses',self.update_targets)
        self.reset()

    def update_targets(self,dt=0.):
        self.targets=self.body_transforms()

    def reset(self):
        self.task.robot.permanent_wrench_composer.reset()
        self.max_proxy_error=self.max_ball_penetration=self.max_fixture_penetration=self.max_reaction_force=0.

    def update(self,indices,forces,sensors,balls):
        robot=self.task.robot
        transforms=self.body_transforms()
        com=robot.data.body_com_pos_w[0].cpu().numpy()
        loads=np.zeros((robot.num_bodies,3));torques=np.zeros_like(loads)
        def accumulate(body,slot,positions):
            start=int(slot.geometry().meta().find(builtin.global_vertex_offset).view()[0])
            mask=(indices>=start)&(indices<start+len(positions))
            f=forces[mask]
            loads[body]+=f.sum(0)
            torques[body]+=np.cross(positions[indices[mask]-start]-com[body],f).sum(0)
        for sensor in sensors:
            accumulate(sensor.attachment.rigid_body_id[0],sensor.gelpad.geo_slot_list[0],sensor.gelpad.vertex_positions)
        ball_points=[ball.vertex_positions for ball in balls]
        for part in self.parts:
            geo=part['slot'].geometry()
            transform=np.asarray(geo.transforms().view())[0]
            world=part['points']@transform[:3,:3].T+transform[:3,3]
            body=part['body_id'];actual=transforms[body]
            physical=part['points']@actual[:3,:3].T+actual[:3,3]
            self.max_proxy_error=max(self.max_proxy_error,float(np.linalg.norm(world-physical,axis=1).max()))
            accumulate(body,part['slot'],world)
            low,high=physical.min(0),physical.max(0)
            for p in ball_points:
                if np.any(p.max(0)<low) or np.any(p.min(0)>high):continue
                local=(p-actual[:3,3])@actual[:3,:3]
                signed=(local@part['planes'][:,:3].T+part['planes'][:,3]).max(1)
                self.max_ball_penetration=max(self.max_ball_penetration,float(max(0.,-signed.min())))
            for name,center,half,rotation in self.fixtures:
                reach=np.abs(rotation)@half
                if np.any(high<center-reach) or np.any(low>center+reach):continue
                depth=convex_box_penetration(physical,part['planes'][:,:3]@actual[:3,:3].T,
                    part['edges']@actual[:3,:3].T,center,half,rotation)
                self.max_fixture_penetration=max(self.max_fixture_penetration,depth)
        self.max_reaction_force=max(self.max_reaction_force,float(np.linalg.norm(loads,axis=1).max()))
        composer=robot.permanent_wrench_composer
        composer.reset()
        composer.set_forces_and_torques(forces=torch.as_tensor(loads[None],dtype=torch.float32,device=robot.device),
            torques=torch.as_tensor(torques[None],dtype=torch.float32,device=robot.device),is_global=True)

    def diagnostics(self):
        return dict(robot_convex_colliders=len(self.parts),physx_fixture_colliders=len(self.fixtures),
            max_proxy_error_m=self.max_proxy_error,max_ball_rigid_penetration_m=self.max_ball_penetration,
            max_robot_fixture_penetration_m=self.max_fixture_penetration,max_reaction_force_N=self.max_reaction_force)
