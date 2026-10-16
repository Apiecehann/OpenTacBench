import ast
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from env.tasks._blind_ball_models import (
    MATERIALS,CLASSES,FLOOR_Z,BOWL_CENTER,BALL_RADIUS,BOX_SIZE,BOX_CENTER,
    WALL_THICKNESS,DEFAULT_PROFILES,PickScore,TRAY_CENTER,TRAY_SIZE,drop_slot,
    ball_mesh,sample_scene,scan_points,surface_roughness,classify_touch,sphere_center_xy,pad_front_contacts,
)


def test_catalogue_varies_real_stiffness_and_surface_geometry():
    assert {m.kind for m in MATERIALS}==set(CLASSES)
    assert all(sum(m.kind==kind for m in MATERIALS)>=2 for kind in CLASSES)
    assert min(m.youngs_modulus for m in MATERIALS if m.kind=='hard')>30*max(
        m.youngs_modulus for m in MATERIALS if m.kind=='soft')
    for material in MATERIALS:
        p,t=ball_mesh(material)
        volume=np.linalg.det(np.stack([p[t[:,i]]-p[t[:,0]] for i in (1,2,3)],axis=-1))
        assert (volume>0).all()
        radius=np.linalg.norm(p[1:],axis=-1)
        assert np.ptp(radius)>1e-3 if material.kind=='textured' else np.ptp(radius)<1e-12
        faces=np.concatenate([t[:,[0,1,2]],t[:,[0,1,3]],t[:,[0,2,3]],t[:,[1,2,3]]])
        _,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
        assert (counts<=2).all()


@pytest.mark.parametrize('count',range(2,13))
@pytest.mark.parametrize('target',CLASSES)
def test_random_layout_has_target_and_clearance(count,target):
    samples=[]
    for seed in range(30):
        materials,p=sample_scene(np.random.default_rng(seed),count,target)
        assert target in {m.kind for m in materials}
        assert len({m.kind for m in materials})==min(count,3)
        assert {m.name for m in materials}<=set(DEFAULT_PROFILES)
        assert p.shape==(count,3)
        distances=np.linalg.norm(p[:,None,:2]-p[None,:,:2],axis=-1)+np.eye(count)
        assert distances.min()>.066
        # Includes the largest possible raised relief, independent of mesh sampling.
        radius=np.array([m.radius+.0025 for m in materials])
        inner=BOX_SIZE/2-WALL_THICKNESS
        assert (np.abs(p[:,:2]-BOX_CENTER)+radius[:,None]<inner).all()
        assert ((distances-np.add.outer(radius,radius))>.001).all()
        # Keep every initial ball centre within the wide palm's box aperture.
        # Measured palm half-width is 103.7 mm with the original Mini asset.
        assert (np.abs(p[:,1])+.104<inner[1]-.003).all()
        samples.append(p[0,:2])
    assert len(np.unique(samples,axis=0))==30


def test_entire_search_raster_leaves_clearance_for_original_palm_and_wrist():
    # Bounds measured from original native colliders, including link 7.
    xy=np.array(scan_points())[:,:2]
    assert (np.abs(xy-BOX_CENTER)+[.0445,.104]<BOX_SIZE/2-WALL_THICKNESS-.002).all()


def test_probe_features_distinguish_smooth_soft_and_real_relief():
    x,y=np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    smooth=.0015-.0004*(x*x+y*y)
    rough=smooth+.0002*np.cos(20*x)*np.cos(18*y)
    assert surface_roughness(smooth)<1e-12
    assert surface_roughness(rough)>6e-5
    assert classify_touch(.0001,surface_roughness(smooth))=='hard'
    assert classify_touch(.002,surface_roughness(smooth))=='soft'
    assert classify_touch(.0001,surface_roughness(rough))=='textured'


def test_ball_center_estimate_handles_top_and_side_contacts():
    center=np.array([.52,-.08,.04])
    for normal in ([0.,0.,1.],[.1,0.,1.],[1.,2.,3.],[-1.,0.,0.],[0.,-1.,0.]):
        normal=np.asarray(normal); normal=normal/np.linalg.norm(normal)
        point=center+BALL_RADIUS*normal
        for load in (.08,.5,3.):
            np.testing.assert_allclose(sphere_center_xy(point,load*normal),center[:2],atol=1e-12)


@pytest.mark.parametrize('axis',([0,-1,0],[0,1,0],[1,0,0]))
def test_probe_ignores_back_face_edge_and_other_objects(axis):
    axis=np.array(axis,dtype=float)
    indices=np.array([100,101,102,103,200])
    forces=np.array([axis,-axis,axis,np.zeros(3),axis])
    # A backside push on a front boundary vertex must also be rejected.
    mask=pad_front_contacts(indices,forces,100,[0,1,3],axis)
    np.testing.assert_array_equal(mask,[True,False,False,False,False])


@pytest.mark.parametrize('stiffness,expected',[(800.,'soft'),(20000.,'hard')])
@pytest.mark.parametrize('hz',(30,60))
def test_squeeze_reaches_both_loads_with_drive_deadband_and_relaxation(stiffness,expected,hz):
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_squeeze')
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    class Plant:
        def __init__(self):
            self._opening=.04; self._goal=np.zeros(3); self._done=False
            self.cfg=SimpleNamespace(physics_dt=1/120,action_repeat=120//hz)
            self.finger_ids=[0,1]; self.force=0.; self.age=0.
            self.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=np.full((1,2),.04)))
        def _loads(self): return np.full(2,self.force)
        def step(self,action):
            self._opening=float(action[3])
            measured=self.robot.data.joint_pos[0,0]
            error=self._opening-measured
            # Independent actuator model with 25 micrometre static deadband.
            measured+=(1-.5**(60/hz))*np.sign(error)*max(0.,abs(error)-.000025)
            self.robot.data.joint_pos[:]=measured
            self.age+=1/hz if measured<.026 else 0.
            self.force=max(0.,.026-measured)*stiffness*(.7+.3*np.exp(-self.age/.1))
    plant=Plant()
    first=namespace['_squeeze'](plant,.5)
    assert first is not None
    assert .425<=plant.force<=.9
    second=namespace['_squeeze'](plant,3.)
    assert second is not None
    assert 2.55<=plant.force<=5.4
    compliance=2*(first[0]-second[0])/(second[1]-first[1])
    assert classify_touch(compliance,0.)==expected


def test_scoring_requires_correct_ball_verified_lift_and_release():
    material=next(m for m in MATERIALS if m.kind=='textured')
    p,_=ball_mesh(material)
    p+=np.r_[BOWL_CENTER,FLOOR_Z+BALL_RADIUS+.0005]
    for q,kind,verified in ((.02,'textured',True),(.04,'hard',True),(.04,'textured',False)):
        score=PickScore()
        for _ in range(40): score.advance(p,q,'textured',kind,verified,1/120)
        assert not score.success
    score=PickScore()
    for _ in range(29): score.advance(p,.04,'textured','textured',True,1/120)
    assert not score.success
    score.advance(p,.04,'textured','textured',True,1/120)
    assert score.success


def test_material_selection_and_target_validation():
    for count in (2,9,12):
        materials,_=sample_scene(np.random.default_rng(2),count,'soft',('rubber','polished_polymer'))
        assert {m.kind for m in materials}=={'soft','hard'}
        assert abs(sum(m.kind=='soft' for m in materials)-count/2)<=.5
    materials,_=sample_scene(np.random.default_rng(2),12,'textured',tuple(m.name for m in MATERIALS))
    assert len({m.name for m in materials})==6
    for profiles in ((),('unknown',),('rubber','rubber'),('rubber',)):
        with pytest.raises(ValueError): sample_scene(np.random.default_rng(0),12,'textured',profiles)
    for count in (1,13):
        with pytest.raises(ValueError): sample_scene(np.random.default_rng(0),count)


@pytest.mark.parametrize('target',CLASSES)
def test_target_selection_changes_scoring_and_requires_whole_ball_inside_bowl(target):
    material=next(m for m in MATERIALS if m.kind==target)
    p,_=ball_mesh(material)
    p+=np.r_[BOWL_CENTER,FLOOR_Z+BALL_RADIUS+.0005]
    correct,wrong,edge=PickScore(),PickScore(),PickScore()
    for _ in range(40):
        correct.advance(p,.04,target,target,True,1/120)
        wrong.advance(p,.04,next(k for k in CLASSES if k!=target),target,True,1/120)
        edge.advance(p+np.array([.06,0,0]),.04,target,target,True,1/120)
    assert correct.success
    assert wrong.failure=='wrong_material' and not wrong.success
    assert not edge.success


def test_touch_controller_has_no_actor_state_or_camera_reads():
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    task=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    controller={'play_once','_go','_locate','_squeeze','_probe'}
    forbidden={'ball_points','materials','_initial_centers','render','camera','scores','diagnostics'}
    for method in task.body:
        if isinstance(method,ast.FunctionDef) and method.name in controller:
            assert not {n.attr for n in ast.walk(method) if isinstance(n,ast.Attribute)}&forbidden


def test_collection_accepts_every_material_but_requires_real_lifts_and_current_containment():
    scores=[]
    for i,material in enumerate(MATERIALS):
        p,_=ball_mesh(material)
        p+=np.r_[drop_slot(i),FLOOR_Z+BALL_RADIUS+.0005]
        score=PickScore()
        for _ in range(40):
            score.advance(p,.04,'textured',material.kind,True,1/120,collect_all=True)
        assert score.success and not score.failure
        # A completed deposit ceases to satisfy the goal if it rolls out again.
        score.advance(p+np.array([1.,0.,0.]),.04,'textured',material.kind,True,1/120,collect_all=True)
        assert not score.success
        unverified=PickScore()
        unverified.advance(p,.04,'textured',material.kind,False,1/120,collect_all=True)
        assert unverified.failure=='unverified_grasp' and not unverified.success
        scores.append(score)
    slots=np.array([drop_slot(i) for i in range(12)])
    assert np.all(np.abs(slots-TRAY_CENTER)+.0325<TRAY_SIZE/2-WALL_THICKNESS)
    distance=np.linalg.norm(slots[:,None]-slots[None,:],axis=-1)+np.eye(12)
    assert distance.min()>.08


def test_collection_requires_all_distinct_balls_simultaneously_delivered():
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_success')
    namespace={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    scores=[SimpleNamespace(success=False) for _ in range(12)]
    task=SimpleNamespace(scores=scores,cfg=SimpleNamespace(collect_all=True),failure='')
    for score in scores[:-1]: score.success=True
    assert not namespace['check_success'](task)
    scores[-1].success=True
    assert namespace['check_success'](task)
    task.failure='rigid_interpenetration'
    assert not namespace['check_success'](task)


def test_collection_material_selection_does_not_require_an_unused_target_category():
    materials,_=sample_scene(np.random.default_rng(0),12,'textured',('rubber',),collect_all=True)
    assert len(materials)==12 and {m.kind for m in materials}=={'soft'}


@pytest.mark.parametrize('ball_depth,wall_depth,success',[(0.,0.,True),(.0011,0.,False),(0.,.0011,False)])
def test_rigid_penetration_invalidates_an_otherwise_successful_episode(ball_depth,wall_depth,success):
    from types import SimpleNamespace,MethodType
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    task=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    methods=[n for n in task.body if isinstance(n,ast.FunctionDef) and n.name in ('_score','check_success')]
    namespace={'np':np}
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),namespace)
    class Array(np.ndarray):
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    array=lambda a:np.asarray(a).view(Array)
    env=SimpleNamespace(physics=SimpleNamespace(force=np.zeros((2,3)),collisions=SimpleNamespace(
        max_ball_penetration=ball_depth,max_fixture_penetration=wall_depth)),failure='',materials=[],
        _tcp=lambda:(array(np.zeros((1,3))),None,None),finger_ids=[0,1],step_count=1,
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=array(np.zeros((1,2))))),
        scores=[SimpleNamespace(success=True)],cfg=SimpleNamespace(physics_dt=1/120,episode_seconds=240,collect_all=False))
    env.check_success=MethodType(namespace['check_success'],env)
    namespace['_score'](env)
    assert env.check_success()==success
    assert env.failure==('' if success else 'rigid_interpenetration')


@pytest.mark.parametrize('distance,loads,tolerance,expected',[
    (.004,(1.,1.),.006,True),(.008,(1.,1.),.006,False),
    (.004,(1.,0.),.006,False),(.004,(0.,0.),.006,False),(.004,(1.,1.),None,False),
])
def test_loaded_approach_requires_bilateral_face_contact_near_its_target(distance,loads,tolerance,expected):
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    task=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in task.body if isinstance(n,ast.FunctionDef) and n.name=='_go')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    class Array(np.ndarray):
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    actual=np.array([[distance,0.,0.]]).view(Array)
    quat=np.array([[1.,0.,0.,0.]]).view(Array)
    env=SimpleNamespace(_done=False,_opening=.026,_goal=np.zeros(3),failure='',
        _orientation=quat,_loads=lambda:np.array(loads),_tcp=lambda:(actual,quat,None),step=lambda action:None)
    assert namespace['_go'](env,np.zeros(3),opening=.026,limit=3,contact_tolerance=tolerance)==expected
    if expected:np.testing.assert_allclose(env._goal,actual[0])
    else:assert env.failure=='motion_timeout'


@pytest.mark.parametrize('angle,expected',[(0.,True),(np.pi/6,False)])
def test_pose_move_waits_for_orientation_as_well_as_position(angle,expected):
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_go')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    class Array(np.ndarray):
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    actual=np.zeros((1,3)).view(Array)
    quat=np.array([[np.cos(angle/2),np.sin(angle/2),0.,0.]]).view(Array)
    target=np.array([[1.,0.,0.,0.]]).view(Array)
    env=SimpleNamespace(_done=False,_opening=.026,_goal=np.zeros(3),failure='',
        _orientation=target,_loads=lambda:np.zeros(2),_tcp=lambda:(actual,quat,None),step=lambda a:None)
    assert namespace['_go'](env,np.zeros(3),opening=.026,limit=3)==expected


def test_contact_seek_can_start_force_centering_from_one_sensing_face():
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_go')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    class Array(np.ndarray):
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    actual=np.array([[0.,0.,.02]]).view(Array)
    quat=np.array([[1.,0.,0.,0.]]).view(Array)
    env=SimpleNamespace(_done=False,_opening=.026,_goal=np.zeros(3),failure='',
        _orientation=quat,_loads=lambda:np.array([0.,.2]),_tcp=lambda:(actual,quat,None),step=lambda a:None)
    assert namespace['_go'](env,np.zeros(3),opening=.026,sense=True,limit=1)
    np.testing.assert_allclose(env._goal,actual[0])


@pytest.mark.parametrize('hz',(30,60))
def test_cartesian_reference_stays_near_a_slow_robot_at_either_control_rate(hz):
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    namespace={'np':np,'RIM_Z':.19}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    class Array(np.ndarray):
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    class SlowRobot:
        def __init__(self):
            self.cfg=SimpleNamespace(physics_dt=1/120,action_repeat=120//hz,
                approach_speed=.09,travel_speed=.16,carry_speed=.09,
                acceleration=.6,max_tracking_error_m=.04)
            self.position=np.array([[.5,0.,.3]]).view(Array)
            self._goal=self.position[0].copy();self._velocity=np.zeros(3)
            self._opening=.04;self._done=False;self._carrying=False
            self.action_space=SimpleNamespace(low=np.array([.2,-.56,.032,.008]),
                high=np.array([.7,.17,.65,.04]))
        def _tcp(self): return self.position,None,None
        def _physics_step(self):
            # Independent slow actuator: one-second first-order response.
            self.position+=(1-np.exp(-self.cfg.physics_dt))*(self._goal-self.position)
        def _observation(self): return {}
        def _info(self): return {}
        def check_success(self): return False
    plant=SlowRobot()
    for _ in range(hz*4):
        namespace['step'](plant,np.array([.5,-.3,.3,.026]))
        assert np.linalg.norm(plant._goal-plant.position[0])<=.040001
    assert plant.position[0,1]<-.10
