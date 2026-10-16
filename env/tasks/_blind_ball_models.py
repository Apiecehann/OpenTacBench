"""Ball geometry, material catalogue and scoring, without simulator imports."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull


CLASSES=('hard','soft','textured')
FLOOR_Z=.016
RIM_Z=.19
BOWL_CENTER=np.array([.49,-.34])
BOWL_RADIUS=.075
BALL_RADIUS=.030
BOX_CENTER=np.array([.51,0.])
BOX_SIZE=np.array([.36,.48])
WALL_THICKNESS=.012
DEFAULT_PROFILES=('polished_polymer','rubber','raised_dots')
TRAY_CENTER=np.array([.40,-.44])
TRAY_SIZE=np.array([.40,.30])


def drop_slot(index):
    """Public receiving-tray grid; independent of hidden source-ball poses."""
    if index not in range(12): raise ValueError('Tray has twelve drop slots')
    return TRAY_CENTER+np.array([(index%4-1.5)*.085,(1-index//4)*.085])


@dataclass(frozen=True)
class BallMaterial:
    name: str
    kind: str
    youngs_modulus: float
    texture: str
    color: tuple
    radius: float=BALL_RADIUS
    density: float=700.


MATERIALS=(
    BallMaterial('polished_polymer','hard',3.e6,'smooth',(.045,.38,.72)),
    BallMaterial('hard_resin','hard',6.e6,'smooth',(.12,.30,.55)),
    BallMaterial('rubber','soft',6.e4,'smooth',(.86,.26,.12)),
    BallMaterial('foam','soft',2.5e4,'smooth',(.90,.48,.22)),
    BallMaterial('raised_dots','textured',1.e6,'bumps',(.92,.66,.12)),
    BallMaterial('raised_ribs','textured',1.e6,'ribs',(.28,.65,.24)),
)


@lru_cache
def _sphere_surface(subdivisions):
    golden=(1+np.sqrt(5))/2
    vertices=[]
    for a in (-1,1):
        for b in (-golden,golden):
            vertices.extend([(0,a,b),(a,b,0),(b,0,a)])
    vertices=np.asarray(vertices,dtype=float)
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    faces=ConvexHull(vertices).simplices
    vertices=list(vertices)
    for _ in range(subdivisions):
        edges={}
        def midpoint(a,b):
            edge=tuple(sorted((int(a),int(b))))
            if edge not in edges:
                point=vertices[a]+vertices[b]
                edges[edge]=len(vertices)
                vertices.append(point/np.linalg.norm(point))
            return edges[edge]
        next_faces=[]
        for a,b,c in faces:
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            next_faces.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
        faces=np.asarray(next_faces,dtype=np.int32)
    return np.asarray(vertices),faces


def ball_mesh(material,subdivisions=3):
    """Star tetrahedral mesh with a watertight textured icosphere boundary."""
    direction,faces=_sphere_surface(subdivisions)
    relief=np.zeros(len(direction))
    if material.texture=='bumps':
        n=42
        z=1-2*(np.arange(n)+.5)/n
        phi=np.arange(n)*np.pi*(3-np.sqrt(5))
        axes=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
        relief=.0025*np.exp((direction@axes.T-1)/.012).max(axis=1)
    elif material.texture=='ribs':
        longitude=np.arctan2(direction[:,1],direction[:,0])
        relief=.0018*(.5+.5*np.cos(12*longitude))*np.maximum(0,1-direction[:,2]**2)
    elif material.texture!='smooth':
        raise ValueError('Unknown ball texture: '+material.texture)
    points=np.vstack([np.zeros(3),direction*(material.radius+relief)[:,None]])
    tets=np.c_[np.zeros(len(faces),dtype=np.int32),faces+1]
    volume6=np.linalg.det(np.stack([points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1))
    negative=volume6<0
    tets[negative,1],tets[negative,2]=tets[negative,2].copy(),tets[negative,1].copy()
    if not np.all(np.abs(volume6)>1e-15):
        raise ValueError('Ball mesh contains inverted tetrahedra')
    return np.ascontiguousarray(points),tets


def sample_scene(rng,count=12,target='textured',profiles=DEFAULT_PROFILES,collect_all=False):
    if count not in range(2,13) or target not in CLASSES:
        raise ValueError('Use 2 to 12 balls and a known target class')
    catalogue={m.name:m for m in MATERIALS}
    if not profiles or len(set(profiles))!=len(profiles) or any(p not in catalogue for p in profiles):
        raise ValueError('Select distinct known material profiles')
    selected=[catalogue[p] for p in profiles]
    available=[k for k in CLASSES if any(m.kind==k for m in selected)]
    if collect_all: target=available[0]
    if target not in available: raise ValueError('Target class must be included in the selected materials')
    others=[name for name in available if name!=target]
    kinds=([target,*list(rng.choice(others,count-1,replace=False))] if count<len(available)
        else available*(count//len(available))+list(rng.choice(available,count%len(available),replace=False)))
    rng.shuffle(kinds)
    materials=[]
    pools={}
    for kind in kinds:
        if not pools.get(kind): pools[kind]=list(rng.permutation([i for i,m in enumerate(selected) if m.kind==kind]))
        materials.append(selected[int(pools[kind].pop())])
    # Randomized cells guarantee clearance even with twelve larger FEM balls.
    # Rows leave room for the original Mini back shells, as well as the balls.
    # The controller only knows the workspace rows, never occupied cells/poses.
    cells=np.array([(.390+.078*j,y) for y in (-.115,0.,.115) for j in range(4)])
    xy=cells[rng.permutation(len(cells))[:count]]+rng.uniform(-.001,.001,(count,2))
    centers=np.c_[xy,[FLOOR_Z+m.radius+.004 for m in materials]]
    return materials,centers


def scan_points(count=12):
    """Known-workspace raster; never receives a ball pose or material label."""
    rows=[]
    for i,y in enumerate((-.115,0.,.115)):
        xs=np.arange(.390,.631,.005)
        if i%2: xs=xs[::-1]
        rows.extend(np.array([x,y,FLOOR_Z+BALL_RADIUS]) for x in xs)
    return rows


def surface_roughness(height,threshold=1.e-6):
    """RMS fine relief after removing a smooth quadratic contact envelope."""
    height=np.asarray(height,dtype=float)
    yy,xx=np.meshgrid(np.linspace(-1,1,height.shape[0]),np.linspace(-1,1,height.shape[1]),indexing='ij')
    mask=height>max(threshold,float(height.max())*.15)
    if mask.sum()<12: return 0.
    x,y=xx[mask],yy[mask]
    design=np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]
    residual=height[mask]-design@np.linalg.lstsq(design,height[mask],rcond=None)[0]
    return float(np.sqrt(np.mean(residual**2)))


def classify_touch(compliance_m_per_N,roughness_m):
    """Physical probe features only; thresholds are in SI units."""
    if not np.isfinite([compliance_m_per_N,roughness_m]).all():
        raise ValueError('Non-finite tactile probe features')
    if roughness_m>1.8e-5: return 'textured'
    return 'soft' if compliance_m_per_N>3.5e-4 else 'hard'


def sphere_center_xy(contact_point,force,radius=BALL_RADIUS):
    """Normalize the full outward force before projecting sphere geometry to XY."""
    point,force=np.asarray(contact_point,dtype=float),np.asarray(force,dtype=float)
    magnitude=np.linalg.norm(force)
    if point.shape!=(3,) or force.shape!=(3,) or not np.isfinite([*point,*force,radius]).all() or magnitude<1e-8 or radius<=0:
        raise ValueError('Expected a finite contact point, nonzero force and positive radius')
    return (point-radius*force/magnitude)[:2]


def pad_front_contacts(indices,forces,vertex_offset,surface_vertices,backing_axis):
    """Select loads on the original sensing face, directed into its backing."""
    return (np.isin(np.asarray(indices)-vertex_offset,surface_vertices)
        & (np.asarray(forces)@np.asarray(backing_axis)>0.))


def convex_collision_mesh(points):
    """Match an asset's authored convexHull collider with a positive-volume mesh."""
    points=np.unique(np.round(np.asarray(points,dtype=float),9),axis=0)
    hull=ConvexHull(points)
    vertices=points[hull.vertices]
    remap=np.full(len(points),-1,dtype=int)
    remap[hull.vertices]=np.arange(len(vertices))
    faces=remap[hull.simplices]
    mesh=np.vstack([vertices.mean(0),vertices])
    tets=np.c_[np.zeros(len(faces),dtype=np.int32),faces+1]
    volume=np.linalg.det(np.stack([mesh[tets[:,i]]-mesh[tets[:,0]] for i in (1,2,3)],axis=-1))
    negative=volume<0
    tets[negative,1],tets[negative,2]=tets[negative,2].copy(),tets[negative,1].copy()
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    return mesh,np.ascontiguousarray(tets,dtype=np.int32),hull.equations,vertices[edges[:,1]]-vertices[edges[:,0]]


def convex_box_penetration(vertices,normals,edges,center,half_size,rotation=None):
    """Separating-axis penetration in metres; zero for disjoint convex solids."""
    rotation=np.eye(3) if rotation is None else np.asarray(rotation)
    vertices=(np.asarray(vertices)-center)@rotation
    normals=np.asarray(normals)@rotation
    edges=np.asarray(edges)@rotation
    axes=np.vstack([np.eye(3),normals,np.cross(edges[:,None,:],np.eye(3)[None,:,:]).reshape(-1,3)])
    lengths=np.linalg.norm(axes,axis=1)
    axes=axes[lengths>1.e-10]/lengths[lengths>1.e-10,None]
    projection=vertices@axes.T
    radius=np.abs(axes)@half_size
    overlap=np.minimum(projection.max(0)+radius,radius-projection.min(0))
    return float(max(0.,overlap.min()))


@dataclass
class PickScore:
    stable_steps: int=0
    success: bool=False
    failure: str=''

    def advance(self,points,opening,target_kind,actual_kind,grasp_verified,dt,collect_all=False):
        points=np.asarray(points)
        extent=np.linalg.norm(points[:,:2]-BOWL_CENTER,axis=1).max()
        in_bowl=(extent<BOWL_RADIUS
            and points[:,2].min()>=FLOOR_Z-.003 and points[:,2].max()<.10)
        if collect_all:
            in_bowl=(np.all(np.abs(points[:,:2]-TRAY_CENTER)<TRAY_SIZE/2-WALL_THICKNESS)
                and points[:,2].min()>=FLOOR_Z-.003 and points[:,2].max()<.15)
        released=opening>.036
        if in_bowl and released:
            if not collect_all and actual_kind!=target_kind:
                self.failure='wrong_material'
            if not grasp_verified:
                self.failure='unverified_grasp'
            self.stable_steps+=1
        else:
            self.stable_steps=0
        self.success=bool(self.stable_steps*dt>=.25 and not self.failure)
