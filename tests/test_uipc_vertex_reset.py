"""A reset must touch its own FEM geometry after backend storage reordering."""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _object_type():
    path=Path(__file__).parents[1]/'third_party/TacEx/source/tacex_uipc/tacex_uipc/objects/uipc_object.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UipcObject')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='write_vertex_positions_to_sim')
    minimal=ast.ClassDef(name='UipcObject',bases=[],keywords=[],body=[method],decorator_list=[])
    namespace={'builtin':SimpleNamespace(global_vertex_offset='global_vertex_offset')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[minimal],type_ignores=[])),
                 str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace['UipcObject']


class Positions:
    def __init__(self,array): self.array=np.asarray(array)
    def cpu(self): return self
    def numpy(self): return self.array


class ReorderedWorld:
    def __init__(self):
        # Three shell vertices precede four volume vertices in FEM storage.
        # Four rigid-body vertices precede both in the global vertex manager.
        self.fem=np.arange(21,dtype=float).reshape(7,3)
        self.global_vertices=np.vstack((np.full((4,3),-1.),self.fem.copy()))

    def write_vertex_pos_to_sim(self,points,global_offset,local_offset,count,system):
        assert system=='uipc::backend::cuda::FiniteElementMethod'
        self.fem[local_offset:local_offset+count]=points
        self.global_vertices[global_offset:global_offset+count]=points


def _make_object(world,*,local_id,backend_offset,global_offset,count):
    def attribute(value): return SimpleNamespace(view=lambda:np.array([value]))
    attrs={'global_vertex_offset':attribute(global_offset)}
    if backend_offset is not None:attrs['backend_fem_vertex_offset']=attribute(backend_offset)
    geometry=SimpleNamespace(meta=lambda:SimpleNamespace(find=attrs.get))
    obj=_object_type()()
    obj.geo_slot_list=[SimpleNamespace(geometry=lambda:geometry)]
    obj._system_name='uipc::backend::cuda::FiniteElementMethod'
    obj.local_system_id=local_id;obj._vertex_count=count
    # Python created the volume first, then the shell: opposite to backend order.
    obj._uipc_sim=obj.uipc_sim=SimpleNamespace(world=world,_system_vertex_offsets={obj._system_name:[0,4,7]})
    return obj


@pytest.mark.parametrize('local_id,backend_offset,global_offset,count',[(1,3,7,4),(2,0,4,3)])
def test_reset_preserves_other_geometry(local_id,backend_offset,global_offset,count):
    world=ReorderedWorld();before=world.fem.copy();global_before=world.global_vertices.copy()
    obj=_make_object(world,local_id=local_id,backend_offset=backend_offset,global_offset=global_offset,count=count)
    points=np.full((count,3),42.)
    obj.write_vertex_positions_to_sim(Positions(points))
    expected=before.copy();expected[backend_offset:backend_offset+count]=points
    global_expected=global_before.copy();global_expected[global_offset:global_offset+count]=points
    np.testing.assert_array_equal(world.fem,expected)
    np.testing.assert_array_equal(world.global_vertices,global_expected)


def test_reset_before_world_initialization_is_rejected():
    world=ReorderedWorld()
    obj=_make_object(world,local_id=1,backend_offset=None,global_offset=7,count=4)
    with pytest.raises(RuntimeError,match='initialized UIPC world'):
        obj.write_vertex_positions_to_sim(Positions(np.zeros((4,3))))


def _render_namespace(**dependencies):
    path=Path(__file__).parents[1]/'third_party/TacEx/source/tacex_uipc/tacex_uipc/sim/uipc_sim.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UipcSim')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_render_meshes')
    resolver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_render_world_matrix')
    namespace={'np':np,**dependencies}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[resolver,method],type_ignores=[])),str(path),'exec'),namespace)
    return namespace


def test_rendered_vertices_follow_world_geometry_across_parent_pose_changes():
    class Mesh:
        def __init__(self): self.matrix=np.eye(4);self.points=None;self.extent=None
        def GetPath(self): return '/mesh'
        def GetAttribute(self,name):
            assert name=='points'
            return SimpleNamespace(Set=lambda points:setattr(self,'points',np.array(points)))
        def GetExtentAttr(self):
            return SimpleNamespace(Set=lambda points:setattr(self,'extent',np.array(points)))
        def GetFabricHierarchyWorldMatrixAttr(self):
            return SimpleNamespace(Get=lambda:self.matrix)
    meshes=[Mesh(),Mesh()]
    rt=SimpleNamespace(Rt=SimpleNamespace(Xformable=lambda prim:prim),
                       UsdGeom=SimpleNamespace(Mesh=lambda prim:prim),
                       Vt=SimpleNamespace(Vec3fArray=np.array))
    namespace=_render_namespace(usdrt=rt)
    points=np.array([[.3,.1,.2],[.4,-.1,.3],[.5,.2,.1],[.6,-.2,.4]])
    offsets=[np.array([.001,0.,0.]),np.zeros(3)]
    sim=SimpleNamespace(_fabric_meshes=meshes,_surf_vertex_offsets=[0,2,4],
        _fabric_mesh_offsets=offsets,_line_renderers=[],
        sio=SimpleNamespace(simplicial_surface=lambda dim:SimpleNamespace(
            positions=lambda:SimpleNamespace(view=lambda:points))))
    expected=[points[:2]+offsets[0],points[2:]]
    def flush_physics():
        # PhysX replaces the cached transform before each rendered frame.
        for i,mesh in enumerate(meshes):
            matrix=np.eye(4)
            matrix[:3,:3]=np.array([[0.,2.,0.],[-3.,0.,0.],[0.,0.,4.]])
            matrix[3,:3]=[.1+i,.2,-.3]
            mesh.matrix=matrix
    def render():
        flush_physics()
        for mesh,world in zip(meshes,expected):
            shown=np.c_[mesh.points,np.ones(len(mesh.points))]@mesh.matrix
            np.testing.assert_allclose(shown[:,:3],world,atol=1e-12)
            assert (mesh.points>=mesh.extent[0]-1e-12).all()
            assert (mesh.points<=mesh.extent[1]+1e-12).all()
    sim.isaac_sim=SimpleNamespace(forward=flush_physics,render=render)
    update=namespace['update_render_meshes']
    update(sim)
    # Also verify that a previous frame's transform cannot leak into this one.
    for mesh in meshes: mesh.matrix=np.eye(4)*2
    update(sim)


class TransformAttribute:
    def __init__(self,value=None,valid=False): self.value=value;self.valid=valid
    def __bool__(self): return self.valid
    def Get(self):
        assert self.valid, 'Do not read an invalid Fabric attribute'
        return self.value


class TransformPrim:
    def __init__(self,path,stage,*,parent=None,local=None,world=None,reset=False):
        self.path=path;self.stage=stage;self.parent=parent
        self.local=np.eye(4) if local is None else local
        self.world=TransformAttribute(world,world is not None)
        self.fabric_local=TransformAttribute();self.legacy_local=TransformAttribute()
        self.reset=reset;self.points=None;self.extent=None
        stage[path]=self
    def GetPath(self): return self.path
    def GetStage(self): return SimpleNamespace(GetPrimAtPath=self.stage.get)
    def GetParent(self): return self.parent
    def IsPseudoRoot(self): return False
    def GetLocalTransformation(self): return self.local
    def GetResetXformStack(self): return self.reset
    def GetFabricHierarchyWorldMatrixAttr(self): return self.world
    def GetFabricHierarchyLocalMatrixAttr(self): return self.fabric_local
    def GetLocalMatrixAttr(self): return self.legacy_local
    def HasWorldXform(self): return False
    def GetAttribute(self,name):
        assert name=='points'
        return SimpleNamespace(Set=lambda value:setattr(self,'points',np.array(value)))
    def GetExtentAttr(self):
        return SimpleNamespace(Set=lambda value:setattr(self,'extent',np.array(value)))


def _fallback_namespace(stage):
    rt=SimpleNamespace(Rt=SimpleNamespace(Xformable=lambda prim:prim),
        UsdGeom=SimpleNamespace(Mesh=lambda prim:prim),Vt=SimpleNamespace(Vec3fArray=np.array))
    usd_stage=SimpleNamespace(GetPrimAtPath=stage.get)
    omni=SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:usd_stage)))
    return _render_namespace(usdrt=rt,omni=omni,UsdGeom=SimpleNamespace(Xformable=lambda prim:prim))


@pytest.mark.parametrize('attribute_exists',[False,True])
def test_missing_world_matrix_keeps_live_parent_rotation_scale_and_translation(attribute_exists):
    stage={}
    # USD's stale parent pose deliberately differs from the current Fabric pose.
    stale=np.eye(4);stale[3,:3]=[100.,200.,300.]
    live=np.array([[0.,2.,0.,0.],[-3.,0.,0.,0.],[0.,0.,4.,0.],[.1,.2,-.3,1.]])
    parent=TransformPrim('/robot',stage,local=stale,world=live)
    local=np.eye(4);local[3,:3]=[.05,.1,.02]
    mesh=TransformPrim('/robot/mesh',stage,parent=parent,local=local)
    mesh.world=TransformAttribute(None,valid=attribute_exists)
    rest=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.02,.03]])
    sim=SimpleNamespace(isaac_sim=None,_fabric_meshes=[mesh],_fabric_mesh_offsets=[np.zeros(3)],
        _surf_vertex_offsets=[0,3],_line_renderers=[])
    update=_fallback_namespace(stage)['update_render_meshes']
    for translation in ([.1,.2,-.3],[.8,-.4,.6]):
        parent.world.value[3,:3]=translation
        points=(np.c_[rest,np.ones(3)]@local@parent.world.value)[:,:3]
        sim.sio=SimpleNamespace(simplicial_surface=lambda dim:SimpleNamespace(
            positions=lambda:SimpleNamespace(view=lambda:points)))
        update(sim)
        np.testing.assert_allclose(mesh.points,rest,atol=1e-12)
        assert mesh.world.value is None, 'The fallback must not author a stale world cache'
        np.testing.assert_allclose(mesh.extent,[rest.min(0),rest.max(0)],atol=1e-12)


@pytest.mark.parametrize('local_source',['usd','fabric','legacy'])
def test_fallback_uses_current_local_transform_and_respects_reset_stack(local_source):
    stage={}
    parent=TransformPrim('/parent',stage,world=np.eye(4)*5)
    local=np.eye(4);local[3,:3]=[.55,.1,.005]
    mesh=TransformPrim('/parent/mesh',stage,parent=parent,reset=True)
    if local_source=='usd':mesh.local=local
    elif local_source=='fabric':mesh.fabric_local=TransformAttribute(local,True)
    else:mesh.legacy_local=TransformAttribute(local,True)
    resolve=_fallback_namespace(stage)['_render_world_matrix']
    np.testing.assert_allclose(resolve(mesh),local)
    local[3,0]=.75
    np.testing.assert_allclose(resolve(mesh),local)


@pytest.mark.parametrize('matrix',[np.ones(3),np.full((4,4),np.nan)])
def test_corrupt_world_matrix_reports_affected_mesh(matrix):
    stage={};mesh=TransformPrim('/bad_mesh',stage,world=matrix)
    with pytest.raises(RuntimeError,match='/bad_mesh.*finite 4x4'):
        _fallback_namespace(stage)['_render_world_matrix'](mesh)


def test_missing_backing_prim_is_not_silently_replaced_by_identity():
    stage={};mesh=TransformPrim('/removed',stage)
    stage.clear()
    with pytest.raises(RuntimeError,match='/removed.*USD prim is missing'):
        _fallback_namespace(stage)['_render_world_matrix'](mesh)


def test_singular_transform_reports_mesh_before_writing_points():
    stage={};matrix=np.eye(4);matrix[0,0]=0.
    mesh=TransformPrim('/collapsed',stage,world=matrix)
    sim=SimpleNamespace(isaac_sim=None,_fabric_meshes=[mesh],_fabric_mesh_offsets=[],
        _surf_vertex_offsets=[0,3],_line_renderers=[],
        sio=SimpleNamespace(simplicial_surface=lambda dim:SimpleNamespace(
            positions=lambda:SimpleNamespace(view=lambda:np.ones((3,3))))))
    with pytest.raises(RuntimeError,match='Non-invertible render transform for /collapsed'):
        _fallback_namespace(stage)['update_render_meshes'](sim)
    assert mesh.points is None
