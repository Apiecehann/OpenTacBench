"""Derive an ALOHA ViperX 300S arm from the pinned original Menagerie model.

Robot meshes and kinematics retain the original BSD-3-Clause model. Sensor
geometry, attachments and optical configuration come from the TacEx Mini.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SOURCE_COMMIT='62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35'
ARM_JOINTS=tuple('aloha_'+s for s in ('waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate'))
HOME_QPOS=(0.,.45,-.9,0.,math.pi/2+.45,0.)
MINI_MOUNT_LENGTH_M=.035
MINI_MOUNT_RADIUS_M=.020
ASSETS=Path(__file__).resolve().parents[2]/'assets/embodiments/aloha'


def _numbers(text):
    return np.fromstring(text,sep=' ')


def _text(values):
    return ' '.join(f'{float(v):.12g}' for v in values)


def _rotation(quaternion):
    w,x,y,z=np.asarray(quaternion,float)/np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def arm_mjcf(output,source=ASSETS/'aloha.xml'):
    """Keep the original six-joint arm; its gripper is replaced by the Mini."""
    source=Path(source)
    root=ET.parse(source).getroot()
    for include in list(root.findall('include')):
        included=ET.parse(source.parent/include.get('file')).getroot()
        root.remove(include)
        root.extend(deepcopy(list(included)))
    world=root.find('worldbody')
    body=deepcopy(world.find("body[@name='left/base_link']"))
    if body is None:raise ValueError('Pinned ALOHA arm is missing')
    gripper=body.find(".//body[@name='left/gripper_link']")
    gripper.remove(gripper.find("body[@name='left/gripper_base']"))
    for link in body.iter('body'):
        for site in list(link.findall('site')):link.remove(site)
    body.set('pos','0 0 0')
    world.clear();world.append(body)
    for tag in ('contact','equality','tendon','sensor','keyframe'):
        for child in root.findall(tag):root.remove(child)
    actuators=root.find('actuator')
    if actuators is not None:
        for actuator in list(actuators):
            if actuator.get('joint') not in {'left/'+j.removeprefix('aloha_') for j in ARM_JOINTS}:
                actuators.remove(actuator)
    for element in root.iter():
        for key,value in list(element.attrib.items()):
            if value.startswith('left/'):element.set(key,'aloha_'+value[5:])
    used={geom.get('mesh') for geom in body.iter('geom') if geom.get('mesh')}
    for mesh in list(root.find('asset').findall('mesh')):
        if Path(mesh.get('file')).stem not in used:root.find('asset').remove(mesh)
    root.find('compiler').set('meshdir',str((source.parent/'meshes').resolve()))
    root.set('model','aloha_vx300s')
    actual=tuple(j.get('name') for j in body.iter('joint'))
    if actual!=ARM_JOINTS:raise ValueError('ALOHA model must have exactly six original arm joints')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    return root


def _origin(parent,position,rotation=(0,0,0)):
    ET.SubElement(parent,'origin',xyz=_text(position),rpy=_text(rotation))


def arm_urdf(mjcf,output,franka_urdf=None,mini_usd=None):
    """Use the same six-joint geometry for Curobo and the native USD importer."""
    root=ET.parse(mjcf).getroot()
    robot=ET.Element('robot',name='aloha_vx300s_gsmini')
    assets={Path(m.get('file')).stem:m for m in root.findall('./asset/mesh')}
    meshdir=Path(root.find('compiler').get('meshdir'))
    defaults=root.find("./default/default[@class='vx300s']")
    joint_defaults=dict(defaults.find('joint').attrib)
    def add_body(body,parent=None):
        name=body.get('name')
        link=ET.SubElement(robot,'link',name=name)
        if body.get('quat') or body.get('euler'):raise ValueError('Unexpected rotated arm body in pinned ALOHA model')
        inertia=body.find('inertial')
        if inertia is not None:
            i=ET.SubElement(link,'inertial')
            _origin(i,_numbers(inertia.get('pos','0 0 0')))
            ET.SubElement(i,'mass',value=inertia.get('mass'))
            rotation=_rotation(_numbers(inertia.get('quat','1 0 0 0')))
            matrix=rotation@np.diag(_numbers(inertia.get('diaginertia')))@rotation.T
            ET.SubElement(i,'inertia',**{key:f'{matrix[a,b]:.12g}' for key,a,b in
                [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})
        for geom in body.findall('geom'):
            if geom.get('class') not in ('visual','collision'):continue
            item=ET.SubElement(link,geom.get('class'))
            rotation=_rotation(_numbers(geom.get('quat','1 0 0 0')))
            rpy=(math.atan2(rotation[2,1],rotation[2,2]),
                 math.asin(float(np.clip(-rotation[2,0],-1,1))),
                 math.atan2(rotation[1,0],rotation[0,0]))
            _origin(item,_numbers(geom.get('pos','0 0 0')),rpy)
            shape=ET.SubElement(item,'geometry')
            mesh=assets[geom.get('mesh')]
            ET.SubElement(shape,'mesh',filename=str(meshdir/mesh.get('file')),scale=mesh.get('scale','1 1 1'))
        if parent is not None:
            original=body.find('joint')
            settings=joint_defaults|defaults.find(f"default[@class='{original.get('class')}']/joint").attrib|original.attrib
            joint=ET.SubElement(robot,'joint',name=original.get('name'),type='revolute')
            ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=name)
            _origin(joint,_numbers(body.get('pos','0 0 0')))
            ET.SubElement(joint,'axis',xyz=settings['axis'])
            low,high=_numbers(settings['range'])
            forces=_numbers(settings.get('actuatorfrcrange','-35 35'))
            ET.SubElement(joint,'limit',lower=str(low),upper=str(high),effort=str(max(abs(forces))),velocity='2')
        for child in body.findall('body'):add_body(child,name)
    add_body(root.find('./worldbody/body'))
    # The Mini palm extends 26 mm behind its frame. A physical flange keeps
    # it clear of the original wrist without disabling self collisions.
    flange=robot.find("link[@name='aloha_gripper_link']")
    for kind in ('visual','collision'):
        item=ET.SubElement(flange,kind)
        _origin(item,(MINI_MOUNT_LENGTH_M/2,0,0),(0,math.pi/2,0))
        ET.SubElement(ET.SubElement(item,'geometry'),'cylinder',
            radius=str(MINI_MOUNT_RADIUS_M),length=str(MINI_MOUNT_LENGTH_M))
    franka_urdf=Path(franka_urdf or ASSETS.parent/'franka/panda.urdf')
    panda=ET.parse(franka_urdf).getroot()
    mini_links={'panda_hand','panda_leftfinger','panda_rightfinger'}
    mini_meshes=mini_planner_meshes(mini_usd,Path(output).parent/'mini_meshes') if mini_usd else None
    for link in panda.findall('link'):
        if link.get('name') not in mini_links:continue
        link=deepcopy(link)
        for kind in ('visual','collision'):
            for geometry in link.findall(kind):
                mesh=geometry.find('geometry/mesh')
                if mesh is None:continue
                if mini_meshes:
                    mesh.set('filename',str(mini_meshes[link.get('name')][kind]))
                    mesh.set('scale','1 1 1')
                    origin=geometry.find('origin')
                    if origin is not None:geometry.remove(origin)
                    _origin(geometry,(0,0,0))
                else:
                    path=Path(mesh.get('filename'))
                    if not path.is_absolute():mesh.set('filename',str((franka_urdf.parent/path).resolve()))
        robot.append(link)
    for joint in panda.findall('joint'):
        if joint.get('name') in ('panda_finger_joint1','panda_finger_joint2'):robot.append(deepcopy(joint))
    mount=ET.SubElement(robot,'joint',name='aloha_gsmini_mount',type='fixed')
    ET.SubElement(mount,'parent',link='aloha_gripper_link')
    ET.SubElement(mount,'child',link='panda_hand')
    _origin(mount,(MINI_MOUNT_LENGTH_M,0,0),(0,math.pi/2,0))
    ET.indent(robot)
    ET.ElementTree(robot).write(output,encoding='utf-8',xml_declaration=True)
    return robot


def mini_planner_meshes(mini_usd,directory):
    """Export existing Mini body meshes for Curobo's URDF asset loader.

    The original USD remains the native physical asset. This avoids missing
    legacy Panda mesh files and retains the actual original Mini body frames.
    """
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(Path(mini_usd).resolve()))
    root=stage.GetDefaultPrim().GetPath()
    cache=UsdGeom.XformCache()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result={}
    for link in ('panda_hand','panda_leftfinger','panda_rightfinger'):
        body=stage.GetPrimAtPath(root.AppendChild(link))
        body_world=np.array(cache.GetLocalToWorldTransform(body)).T
        result[link]={}
        for kind in ('visual','collision'):
            pieces=[];offset=0
            part=stage.GetPrimAtPath(body.GetPath().AppendChild(kind+'s'))
            for prim in Usd.PrimRange(part,Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh):continue
                mesh=UsdGeom.Mesh(prim)
                local=np.linalg.solve(body_world,np.array(cache.GetLocalToWorldTransform(prim)).T)
                points=np.asarray(mesh.GetPointsAttr().Get(),float)
                points=points@local[:3,:3].T+local[:3,3]
                pieces.extend('v '+_text(p) for p in points)
                indices=np.array(mesh.GetFaceVertexIndicesAttr().Get(),int)
                cursor=0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    face=indices[cursor:cursor+count]+offset+1;cursor+=count
                    pieces.append('f '+' '.join(str(int(i)) for i in face))
                offset+=len(points)
            if not offset:raise ValueError(f'Original Mini {link} has no {kind} mesh')
            path=(directory/f'{link}_{kind}.obj').resolve()
            path.write_text('\n'.join(pieces)+'\n')
            result[link][kind]=path
    return result


def assemble_mini_arm(arm_usd, mini_usd, output):
    """Rigidly mount the complete original Mini assembly on the original arm.

    Reference composition keeps the Mini's geometry, materials and attachment
    paths. All Mini bodies receive one common transform, so joint anchors and
    camera/gel relative poses retain their original calibration.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/Robot').GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root.GetReferences().AddReference(str(Path(mini_usd).resolve()))
    # The source robot root has an authored display rotation. The new arm base
    # defines the world frame; keep only the Mini's internal relative geometry.
    UsdGeom.Xformable(root).ClearXformOpOrder()
    old_links={f'panda_link{i}' for i in range(9)}|{'root_joint'}
    for child in list(root.GetChildren()):
        if child.GetName() in old_links:child.SetActive(False)
    # Instance proxies cannot receive the transform and material overrides that
    # the original UIPC sensor setup needs. Flatten only after de-instancing.
    while True:
        instances=[p for p in stage.Traverse() if p.IsInstance()]
        if not instances:break
        for prim in instances:prim.SetInstanceable(False)
    stage=Usd.Stage.Open(stage.Flatten())
    root=stage.GetPrimAtPath('/Robot')
    for name in old_links:stage.RemovePrim('/Robot/'+name)
    parts=list(root.GetChildren())
    transforms=UsdGeom.XformCache()
    old_hand=np.array(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath('/Robot/panda_hand'))).T
    old_world={p.GetPath():np.array(transforms.GetLocalToWorldTransform(p)).T
               for p in parts if p.IsA(UsdGeom.Xformable)}
    arm=UsdGeom.Xform.Define(stage,'/Robot/Arm').GetPrim()
    arm.GetReferences().AddReference(str(Path(arm_usd).resolve()))
    for prim in Usd.PrimRange(arm):
        if prim.IsA(UsdPhysics.Scene):prim.SetActive(False)
    grippers=[p for p in Usd.PrimRange(arm) if p.GetName()=='aloha_gripper_link']
    if len(grippers)!=1:raise ValueError('Imported ALOHA arm has no unique gripper link')
    gripper=grippers[0]
    transforms.Clear()
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    mount[0,3]=MINI_MOUNT_LENGTH_M
    target=np.array(transforms.GetLocalToWorldTransform(gripper)).T@mount
    common=target@np.linalg.inv(old_hand)
    for path,pose in old_world.items():
        xform=UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d((common@pose).T.tolist()))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    flange=UsdGeom.Cylinder.Define(stage,gripper.GetPath().AppendChild('mini_mount_flange'))
    flange.CreateAxisAttr('X')
    flange.CreateHeightAttr(MINI_MOUNT_LENGTH_M)
    flange.CreateRadiusAttr(MINI_MOUNT_RADIUS_M)
    flange.AddTranslateOp().Set(Gf.Vec3d(MINI_MOUNT_LENGTH_M/2,0,0))
    flange.CreateDisplayColorAttr().Set([Gf.Vec3f(.13,.15,.16)])
    UsdPhysics.CollisionAPI.Apply(flange.GetPrim())
    joint=UsdPhysics.FixedJoint.Define(stage,'/Robot/aloha_gsmini_mount')
    joint.CreateBody0Rel().SetTargets([gripper.GetPath()])
    joint.CreateBody1Rel().SetTargets([Sdf.Path('/Robot/panda_hand')])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(MINI_MOUNT_LENGTH_M,0,0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.sqrt(.5),0,math.sqrt(.5),0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    UsdPhysics.ArticulationRootAPI.Apply(root)
    stage.GetRootLayer().Export(str(output))
    return Path(output)


def planner_config(urdf, directory):
    """Six-axis Curobo config; no unused Franka joint or padded action."""
    import yaml
    directory=Path(directory)
    description=ET.parse(urdf).getroot()
    links=[l.get('name') for l in description.findall('link')
           if (l.get('name').startswith('aloha_') and l.get('name')!='aloha_base_link')
           or l.get('name')=='panda_hand']
    spheres={}
    for link in links:
        geometry=description.find(f"link[@name='{link}']/collision")
        if geometry is None:continue
        mesh=geometry.find('geometry/mesh')
        if mesh is None:
            cylinder=geometry.find('geometry/cylinder')
            if cylinder is None:raise ValueError('Unsupported ALOHA collision primitive')
            angle=np.arange(32)*2*math.pi/32
            radius=float(cylinder.get('radius'));length=float(cylinder.get('length'))
            points=np.array([(radius*math.cos(a),radius*math.sin(a),z)
                for a in angle for z in (-length/2,length/2)])
        else:
            path=Path(mesh.get('filename'))
            if path.suffix=='.obj':
                points=np.array([np.fromstring(line[2:],sep=' ') for line in path.read_text().splitlines() if line.startswith('v ')])
            else:points=stl_vertices(path)
            points=points*_numbers(mesh.get('scale','1 1 1'))
        from scipy.spatial.transform import Rotation
        origin=geometry.find('origin')
        attrs={} if origin is None else origin.attrib
        rotation=Rotation.from_euler('xyz',_numbers(attrs.get('rpy','0 0 0'))).as_matrix()
        points=points@rotation.T+_numbers(attrs.get('xyz','0 0 0'))
        middle=points.mean(axis=0)
        _,axes=np.linalg.eigh((points-middle).T@(points-middle))
        axis=axes[:,-1]
        projected=(points-middle)@axis
        centers=middle+np.linspace(projected.min(),projected.max(),
            max(2,int(np.ceil(np.ptp(projected)/.03))+1))[:,None]*axis
        distance=np.linalg.norm(points[:,None]-centers[None],axis=2)
        nearest=distance.argmin(axis=1)
        spheres[link]=[{'center':c.tolist(),'radius':float(distance[nearest==i,i].max()+.008)}
            for i,c in enumerate(centers) if np.any(nearest==i)]
    sphere_path=directory/'collision_aloha.yml'
    sphere_path.write_text(yaml.safe_dump({'collision_spheres':spheres},sort_keys=False))
    collision_links=list(spheres)
    # As in the original Panda planner, the fixed base resting on the table
    # and the fingers/cases used for intentional grasp contact are excluded.
    # Their native physical collisions remain enabled. Check the moving arm
    # and palm, ignoring adjacent links that share a joint/mount.
    ignore={link:collision_links[i+1:i+3] for i,link in enumerate(collision_links)}
    cfg={'robot_cfg':{'kinematics':{
        'use_usd_kinematics':False,'urdf_path':str(Path(urdf).resolve()),
        'asset_root_path':None,'base_link':'aloha_base_link','ee_link':'panda_hand',
        'collision_link_names':collision_links,'collision_spheres':str(sphere_path.resolve()),
        'collision_sphere_buffer':.004,'use_global_cumul':True,
        'self_collision_ignore':ignore,'self_collision_buffer':{k:0. for k in collision_links},
        'mesh_link_names':collision_links,
        'cspace':{'joint_names':list(ARM_JOINTS),'retract_config':list(HOME_QPOS),
                  'null_space_weight':[1]*6,'cspace_distance_weight':[1]*6,
                  'max_acceleration':15.,'max_jerk':500.}}},
        'planner':{'frame_bias':[0.,0.,0.]}}
    path=directory/'curobo.yml'
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return path


def stl_vertices(path):
    """Read the pinned binary STL mesh without a geometry-package dependency."""
    import struct
    raw=Path(path).read_bytes()
    count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+50*count:raise ValueError('Expected the original binary ALOHA STL')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    return np.frombuffer(raw,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)


def ensure_aloha_assets(mini_usd):
    """Build once through Isaac Lab's original MJCF importer (requires App)."""
    import hashlib
    import inspect
    import json
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from pxr import Usd

    repo=ASSETS.parents[2]
    directory=repo/'.cache/aloha_gsmini'
    directory.mkdir(parents=True,exist_ok=True)
    reference=Usd.Stage.CreateInMemory()
    reference.DefinePrim('/Mini').GetReferences().AddReference(str(Path(mini_usd).resolve()))
    files={Path(__file__),Path(mini_usd),Path(inspect.getfile(MjcfConverter)),
           *[p for p in ASSETS.rglob('*') if p.is_file()],
           *[Path(layer.realPath) for layer in reference.GetUsedLayers() if layer.realPath]}
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.resolve()).encode()+b'\0')
        digest.update(path.read_bytes())
    signature=digest.hexdigest()
    marker=directory/'build.json'
    usd=directory/'aloha_gsmini.usda'
    outputs=[usd,directory/'curobo.yml',directory/'collision_aloha.yml',directory/'aloha.urdf',
             directory/'arm/aloha_arm.usd',
             *[directory/'mini_meshes'/f'{body}_{kind}.obj' for body in
               ('panda_hand','panda_leftfinger','panda_rightfinger') for kind in ('visual','collision')]]
    if marker.is_file() and all(p.is_file() for p in outputs) and json.loads(marker.read_text()).get('sha256')==signature:
        return usd,directory/'curobo.yml'
    xml=directory/'aloha_arm.xml'
    arm_mjcf(xml)
    urdf=directory/'aloha.urdf'
    arm_urdf(xml,urdf,mini_usd=mini_usd)
    planner=planner_config(urdf,directory)
    # Isaac Sim 5.1's MJCF mesh importer prefixes even an absolute meshdir
    # with the XML directory. Use the equivalent relative path at this SDK
    # boundary, after exporting the planner's absolute URDF mesh paths.
    native_xml=ET.parse(xml)
    native_xml.getroot().find('compiler').set('meshdir',
        os.path.relpath((ASSETS/'meshes').resolve(),directory.resolve()))
    native_xml.write(xml,encoding='utf-8',xml_declaration=True)
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.mjcf')
    converted=MjcfConverter(MjcfConverterCfg(
        asset_path=str(xml),usd_dir=str(directory/'arm'),usd_file_name='aloha_arm.usd',
        fix_base=True,make_instanceable=False,import_inertia_tensor=True,
        self_collision=True,import_sites=False,force_usd_conversion=True))
    assemble_mini_arm(converted.usd_path,mini_usd,usd)
    marker.write_text(json.dumps({'sha256':signature,'source_commit':SOURCE_COMMIT},indent=2))
    return usd,planner


def with_aloha_arm(original):
    """Keep the original Mini sensor configuration and change its carrier only."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    result=original.copy()
    usd,planner=ensure_aloha_assets(result.robot.spawn.usd_path)
    urdf=ET.parse(usd.parent/'aloha.urdf').getroot()
    efforts={j.get('name'):float(j.find('limit').get('effort'))
             for j in urdf.findall('joint') if j.get('name') in ARM_JOINTS}
    result.robot=result.robot.replace(
        spawn=result.robot.spawn.replace(usd_path=str(usd)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={
            **dict(zip(ARM_JOINTS,HOME_QPOS)),'panda_finger_joint.*':.02}),
        actuators={
            'aloha_arm':ImplicitActuatorCfg(joint_names_expr=list(ARM_JOINTS),
                effort_limit_sim=efforts,velocity_limit_sim=2.,stiffness=400.,damping=80.),
            'panda_hand':result.robot.actuators['panda_hand'].copy()})
    result.robot_type='aloha_vx300s'
    result.planner_yaml=str(planner)
    return result
