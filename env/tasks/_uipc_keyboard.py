"""UIPC piano keys with rigid bodies, rear hinges, return springs and stops."""
import numpy as np
from uipc import builtin,view
from uipc.geometry import tetmesh,linemesh,label_surface,label_triangle_orient
from uipc.constitution import AffineBodyConstitution,SoftTransformConstraint,AffineBodyRevoluteJoint
from envs._tactile_memory import box_mesh


def y_transform(angle,position):
    c,s=np.cos(angle),np.sin(angle)
    matrix=np.eye(4)
    matrix[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    matrix[:3,3]=position
    return matrix


def volume_mesh(points,tets):
    mesh=tetmesh(np.ascontiguousarray(points),np.ascontiguousarray(tets))
    label_surface(mesh)
    label_triangle_orient(mesh)
    return mesh


class UipcKeyboard:
    stiffness=1.0  # N m / rad
    damping=.025  # N m s / rad
    mass=.020     # kg per key

    def __init__(self,scene,layout,indices,dt,key_contact):
        self.dt=dt
        self.indices=indices
        self.local=[]
        self.centers=[]
        self.pivots=[]
        self.slots=[]
        self.angles=np.zeros(88)
        self.velocities=np.zeros(88)
        self.max_hinge_error=0.
        self.max_shape_error=0.
        abd, spring, hinge=AffineBodyConstitution(),SoftTransformConstraint(),AffineBodyRevoluteJoint()
        for constitution in (abd,spring,hinge):
            scene.constitution_tabular().insert(constitution)
        stop_contact=scene.contact_tabular().create('key_stops')
        scene.contact_tabular().insert(key_contact,stop_contact,.2,1e9)
        # Two mechanical stop pads per reachable white key. The upper pad sits
        # behind the fingertip and ahead of the black-key region.
        parts,tets=[],[]
        for key_id in indices:
            key=layout[key_id]
            x=key.position[0]-key.size[0]/2+.05
            pivot_x=key.position[0]+key.size[0]/2
            top=key.position[2]+key.size[2]
            lower_top=key.position[2]-(pivot_x-x)*np.sin(np.deg2rad(4.))
            for bottom in (top+.0002,lower_top-.01):
                p,t=box_mesh((.008,.015,.010),(1,1,1))
                p+=np.array([x,key.position[1],bottom])
                tets.append(t+len(parts)*8)
                parts.append(p)
        fixture=volume_mesh(np.concatenate(parts),np.concatenate(tets))
        abd.apply_to(fixture,1e8,mass_density=1e6)
        view(fixture.instances().find(builtin.is_fixed))[:]=1
        stop_contact.apply_to(fixture)
        fixture_slot,_=scene.objects().create('piano_fixture').geometries().create(fixture)
        keys_obj=scene.objects().create('piano_keys')
        hinge_vertices=[]
        pairs=[]
        for key_id in indices:
            key=layout[key_id]
            center=np.array(key.position)+np.array([0,0,key.size[2]/2])
            pivot=center+np.array([key.size[0]/2,0,0])
            p,t=box_mesh(key.size,(1,1,1))
            p[:,2]-=key.size[2]/2
            mesh=volume_mesh(p,t)
            view(mesh.transforms())[0]=y_transform(0,center)
            abd.apply_to(mesh,1e8,mass_density=self.mass/np.prod(key.size))
            key_contact.apply_to(mesh)
            inertia=self.mass*(key.size[0]**2+key.size[2]**2)/12
            spring.apply_to(mesh,np.array([0.,self.stiffness*dt*dt/inertia]))
            slot,_=keys_obj.geometries().create(mesh)
            self.slots.append(slot)
            self.local.append(p)
            self.centers.append(center)
            self.pivots.append(pivot)
            hinge_vertices.extend([pivot+np.array([0,-.01,0]),pivot+np.array([0,.01,0])])
            pairs.append((fixture_slot,slot))
        self.local=np.asarray(self.local)
        self.centers=np.asarray(self.centers)
        self.pivots=np.asarray(self.pivots)
        axes=linemesh(np.asarray(hinge_vertices),np.arange(2*len(indices),dtype=np.int32).reshape(-1,2))
        hinge.apply_to(axes,pairs,100.)
        scene.objects().create('key_hinges').geometries().create(axes)
        def animate_springs(info):
            for i,slot in enumerate(info.geo_slots()):
                geo=slot.geometry()
                view(geo.instances().find(builtin.is_constrained))[:]=1
                # Equivalent viscous torque using the previous angular velocity.
                goal=self.damping/self.stiffness*self.velocities[indices[i]]
                view(geo.instances().find(builtin.aim_transform))[0]=y_transform(goal,self.centers[i])
        self._animate=animate_springs
        scene.animator().insert(keys_obj,animate_springs)

    def reset(self):
        self.angles.fill(0.)
        self.velocities.fill(0.)
        self.max_hinge_error=self.max_shape_error=0.

    def update(self):
        transforms=np.stack([np.asarray(slot.geometry().transforms().view())[0] for slot in self.slots])
        angles=-np.arctan2(transforms[:,0,2],transforms[:,0,0])
        self.velocities[self.indices]=(angles-self.angles[self.indices])/self.dt
        self.angles[self.indices]=angles
        local_pivots=np.array([[p[:,0].max(),0,0] for p in self.local])
        pivots=np.einsum('nj,nij->ni',local_pivots,transforms[:,:3,:3])+transforms[:,:3,3]
        self.max_hinge_error=max(self.max_hinge_error,float(np.linalg.norm(pivots-self.pivots,axis=-1).max()))
        rotation=transforms[:,:3,:3]
        self.max_shape_error=max(self.max_shape_error,float(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3)).max()))
        self.transforms=transforms
        self.vertices=np.einsum('nkj,nij->nki',self.local,rotation)+transforms[:,None,:3,3]
        return transforms
