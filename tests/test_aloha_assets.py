import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from envs.robot.aloha_assets import ARM_JOINTS,HOME_QPOS,MINI_MOUNT_LENGTH_M,arm_mjcf,arm_urdf,assemble_mini_arm,planner_config


def rpy_matrix(values):
    x,y,z=values
    cx,sx=math.cos(x),math.sin(x)
    cy,sy=math.cos(y),math.sin(y)
    cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([
        [cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def urdf_forward(root,values):
    by_child={j.find('child').get('link'):j for j in root.findall('joint')}
    chain=[];child='panda_hand'
    while child in by_child:
        joint=by_child[child];chain.append(joint)
        child=joint.find('parent').get('link')
    pose=np.eye(4)
    for joint in reversed(chain):
        origin=joint.find('origin')
        t=np.eye(4);t[:3,3]=np.fromstring(origin.get('xyz'),sep=' ')
        t[:3,:3]=rpy_matrix(np.fromstring(origin.get('rpy'),sep=' '))
        pose=pose@t
        if joint.get('type')=='revolute':
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            angle=values[joint.get('name')]
            cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            t=np.eye(4);t[:3,:3]=np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
            pose=pose@t
    return pose


@pytest.fixture
def description(tmp_path):
    xml=tmp_path/'aloha_arm.xml'
    arm_mjcf(xml)
    return xml,arm_urdf(xml,tmp_path/'aloha.urdf')


def test_model_uses_six_original_joints_and_no_franka_arm(description):
    xml,root=description
    joints=[j for j in root.findall('joint') if j.get('type')=='revolute']
    assert tuple(j.get('name') for j in joints)==ARM_JOINTS
    assert not any('panda_joint' in j.get('name') for j in root.findall('joint'))
    assert [j.find('axis').get('xyz') for j in joints]==[
        '0 0 1','0 1 0','0 1 0','1 0 0','0 1 0','1 0 0']
    for joint,value in zip(joints,HOME_QPOS):
        limit=joint.find('limit')
        assert float(limit.get('lower'))<value<float(limit.get('upper'))
    model=ET.parse(xml).getroot()
    for mesh in model.findall('./asset/mesh'):
        assert (Path(model.find('compiler').get('meshdir'))/mesh.get('file')).is_file()


def test_zero_pose_matches_original_link_lengths_and_mini_mount(description):
    _,root=description
    pose=urdf_forward(root,dict.fromkeys(ARM_JOINTS,0.))
    np.testing.assert_allclose(pose[:3,3],[.429294+MINI_MOUNT_LENGTH_M,0,.42705],atol=1e-9)
    np.testing.assert_allclose(pose[:3,:3],[[0,0,1],[0,1,0],[-1,0,0]],atol=1e-9)
    home=urdf_forward(root,dict(zip(ARM_JOINTS,HOME_QPOS)))
    assert .43<home[0,3]<.49 and .38<home[2,3]<.42
    np.testing.assert_allclose(home[:3,2],[0,0,-1],atol=1e-9)


def test_kinematics_matches_the_original_mujoco_model(description):
    mujoco=pytest.importorskip('mujoco')
    repo=Path(__file__).resolve().parents[1]
    original=repo/'.cache/aloha_menagerie/aloha.xml'
    if not original.is_file():pytest.skip('Original complete Menagerie cache is not installed')
    model=mujoco.MjModel.from_xml_path(str(original))
    data=mujoco.MjData(model)
    _,root=description
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left/base_link')
    gripper=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left/gripper_link')
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'left/'+j.removeprefix('aloha_')) for j in ARM_JOINTS]
    rng=np.random.default_rng(91)
    for values in [np.zeros(6),np.array(HOME_QPOS),*rng.uniform(-.6,.6,(12,6))]:
        for joint,value in zip(ids,values):data.qpos[model.jnt_qposadr[joint]]=value
        mujoco.mj_forward(model,data)
        expected=np.eye(4)
        expected[:3,3]=data.xpos[gripper]-data.xpos[base]+data.xmat[gripper].reshape(3,3)@np.array([MINI_MOUNT_LENGTH_M,0,0])
        expected[:3,:3]=data.xmat[gripper].reshape(3,3)@rpy_matrix((0,math.pi/2,0))
        np.testing.assert_allclose(urdf_forward(root,dict(zip(ARM_JOINTS,values))),expected,atol=1e-9)


def test_planner_has_six_axes_and_collision_geometry(description,tmp_path):
    import yaml
    _,root=description
    # The FK fixture needs no vendor Mini meshes. Use a separate test box for
    # the planning-only mesh-loader check; native builds export the real USD.
    mesh=tmp_path/'hand.obj'
    mesh.write_text('\n'.join(f'v {x} {y} {z}' for x in (-.03,.03) for y in (-.04,.04) for z in (0.,.05)))
    root.find("link[@name='panda_hand']/collision/geometry/mesh").set('filename',str(mesh))
    ET.ElementTree(root).write(tmp_path/'aloha.urdf')
    planner=planner_config(tmp_path/'aloha.urdf',tmp_path)
    cfg=yaml.safe_load(planner.read_text())['robot_cfg']['kinematics']
    assert cfg['cspace']['joint_names']==list(ARM_JOINTS)
    assert len(cfg['cspace']['retract_config'])==6
    spheres=yaml.safe_load(Path(cfg['collision_spheres']).read_text())['collision_spheres']
    assert set(spheres)==set(cfg['collision_link_names'])
    assert {'aloha_shoulder_link','aloha_wrist_link','aloha_gripper_link','panda_hand'}<=set(spheres)
    assert 'aloha_base_link' not in spheres
    assert not any('finger' in k or 'case' in k for k in spheres)
    assert all(s['radius']>0 and np.isfinite(s['center']).all() for rows in spheres.values() for s in rows)


def test_original_mini_geometry_and_relative_calibration_survive_mount(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    repo=Path(__file__).resolve().parents[1]
    mini=repo/'third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist_attached.usda'
    if not mini.exists():pytest.skip('Original complete Mini assembly is not installed')
    arm=Usd.Stage.CreateNew(str(tmp_path/'arm.usda'))
    base=UsdGeom.Xform.Define(arm,'/Arm').GetPrim();arm.SetDefaultPrim(base)
    gripper=UsdGeom.Xform.Define(arm,'/Arm/aloha_gripper_link')
    gripper.AddTranslateOp().Set(Gf.Vec3d(.429294,0,.42705))
    UsdPhysics.RigidBodyAPI.Apply(gripper.GetPrim())
    arm.GetRootLayer().Save()
    result=assemble_mini_arm(tmp_path/'arm.usda',mini,tmp_path/'combined.usda')
    original=Usd.Stage.Open(str(mini));combined=Usd.Stage.Open(str(result))
    cache0=UsdGeom.XformCache();cache1=UsdGeom.XformCache()
    def pose(stage,cache,name):
        return np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(name))).T
    h0=pose(original,cache0,'/panda/panda_hand')
    h1=pose(combined,cache1,'/Robot/panda_hand')
    np.testing.assert_allclose(h1[:3,3],[.429294+MINI_MOUNT_LENGTH_M,0,.42705],atol=1e-9)
    np.testing.assert_allclose(h1[:3,:3],[[0,0,1],[0,1,0],[-1,0,0]],atol=1e-9)
    for name in ['panda_leftfinger','panda_rightfinger','gelsight_mini_case_left',
                 'gelsight_mini_case_right','gelpad_left','gelpad_right','WristCamera']:
        np.testing.assert_allclose(np.linalg.inv(h0)@pose(original,cache0,'/panda/'+name),
            np.linalg.inv(h1)@pose(combined,cache1,'/Robot/'+name),atol=1e-9)
        for p in Usd.PrimRange(original.GetPrimAtPath('/panda/'+name),Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh):continue
            target=combined.GetPrimAtPath(str(p.GetPath()).replace('/panda/','/Robot/',1))
            assert target
            np.testing.assert_array_equal(UsdGeom.Mesh(p).GetPointsAttr().Get(),UsdGeom.Mesh(target).GetPointsAttr().Get())
    assert len([p for p in combined.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)])==1
    assert not any(combined.GetPrimAtPath(f'/Robot/panda_link{i}') for i in range(9))
    assert not any(p.IsInstance() for p in combined.Traverse())
    flange=UsdGeom.Cylinder(combined.GetPrimAtPath('/Robot/Arm/aloha_gripper_link/mini_mount_flange'))
    assert flange and flange.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert flange.GetAxisAttr().Get()=='X'
    assert flange.GetHeightAttr().Get()==MINI_MOUNT_LENGTH_M
    for p in combined.Traverse():
        if p.IsA(UsdPhysics.Joint):
            for rel in [UsdPhysics.Joint(p).GetBody0Rel(),UsdPhysics.Joint(p).GetBody1Rel()]:
                for path in rel.GetTargets():assert combined.GetPrimAtPath(path),str(path)
