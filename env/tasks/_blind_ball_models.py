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


@dataclass(frozen=True)
class BallMaterial:
    name: str
    kind: str
    youngs_modulus: float
    texture: str
    color: tuple
    radius: float=.024
    density: float=700.


MATERIALS=(
    BallMaterial('polished_polymer','hard',3.e6,'smooth',(.32,.18,.55)),
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


def sample_scene(rng,count=3,target='textured'):
    if count not in (2,3) or target not in CLASSES:
        raise ValueError('Use 2 or 3 balls and a known target class')
    others=[name for name in CLASSES if name!=target]
    kinds=[target,*list(rng.choice(others,count-1,replace=False))]
    rng.shuffle(kinds)
    materials=[MATERIALS[int(rng.choice([i for i,m in enumerate(MATERIALS) if m.kind==kind]))] for kind in kinds]
    centers=[]
    for material in materials:
        for _ in range(1000):
            xy=rng.uniform([.445,-.12],[.575,.12])
            if all(np.linalg.norm(xy-p[:2])>.078 for p in centers):
                centers.append(np.r_[xy,FLOOR_Z+material.radius+.004])
                break
        else:
            raise RuntimeError('Could not sample separated balls')
    return materials,np.asarray(centers)


def scan_points():
    """Known-workspace raster; never receives a ball pose or material label."""
    rows=[]
    for i,y in enumerate(np.arange(-.08,.081,.04)):
        xs=np.arange(.405,.616,.005)
        if i%2: xs=xs[::-1]
        rows.extend(np.array([x,y,FLOOR_Z+.025]) for x in xs)
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


@dataclass
class PickScore:
    stable_steps: int=0
    success: bool=False
    failure: str=''

    def advance(self,points,opening,target_kind,actual_kind,grasp_verified,dt):
        points=np.asarray(points)
        center=points.mean(axis=0)
        in_bowl=(np.linalg.norm(center[:2]-BOWL_CENTER)<BOWL_RADIUS-.028
            and points[:,2].min()>=FLOOR_Z-.003 and points[:,2].max()<.10)
        released=opening>.036
        if in_bowl and released:
            if actual_kind!=target_kind:
                self.failure='wrong_material'
            if not grasp_verified:
                self.failure='unverified_grasp'
            self.stable_steps+=1
        else:
            self.stable_steps=0
        self.success=bool(self.stable_steps*dt>=.25 and not self.failure)
