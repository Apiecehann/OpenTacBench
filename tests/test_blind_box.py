import numpy as np
import ast
from pathlib import Path
from types import SimpleNamespace

from envs._blind_box import RIM_Z, Retrieval, grasp_yaw_delta, object_mesh, sample_object_pose, scan_path


def test_random_poses_fit_inside_the_box_and_are_not_discrete_slots():
    rng = np.random.default_rng(23)
    positions = np.array([sample_object_pose(rng)[0] for _ in range(100)])
    assert len(np.unique(positions[:, 0])) == 100
    assert np.all((positions[:, 0] > .49) & (positions[:, 0] < .55))
    assert np.max(np.abs(positions[:, 1])) < .055
    for _ in range(20):
        _, quaternion = sample_object_pose(rng)
        np.testing.assert_allclose(np.linalg.norm(quaternion), 1.)


def test_rounded_object_is_a_positive_volume_solid():
    points, tets = object_mesh()
    volume6 = np.linalg.det(np.stack([points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1))
    assert np.all(volume6 > 0)


def test_external_head_view_is_blocked_by_open_box_walls():
    # Rays to the extreme upper corners of every allowed block placement must
    # hit an opaque wall below its top; this does not rely on a black image mask.
    eye = np.array([.10, -.82, .34])
    for x in (.492-.035, .548+.035):
        for y in (-.052-.035, .052+.035):
            target = np.array([x,y,.078])
            ray = target-eye
            entry_t = max((.36-eye[0])/ray[0], (-.19-eye[1])/ray[1])
            wall_hit = eye+entry_t*ray
            assert .017 < wall_hit[2] < .197


def test_scan_is_continuous_and_large_block_fits_the_gripper():
    points = np.array(scan_path())
    assert np.max(np.linalg.norm(np.diff(points,axis=0),axis=1)) < .003
    block, _ = object_mesh()
    for yaw in np.linspace(-np.pi,np.pi,37):
        transverse = block[:,0]*np.sin(yaw) + block[:,1]*np.cos(yaw)
        assert np.ptp(transverse) < .078  # Actual Panda maximum jaw travel.


def test_lift_inside_box_throw_and_drop_cannot_count_as_retrieval():
    points, _ = object_mesh()
    score = Retrieval()
    for _ in range(25):
        score.advance(points+[.52,0,.09], [.52,0,.11], True)
    assert not score.success
    for _ in range(25):
        score.advance(points+[.52,0,RIM_Z+.02], [.52,0,RIM_Z+.04], False)
    assert not score.success
    for _ in range(17):
        score.advance(points+[.52,0,RIM_Z+.02], [.52,0,RIM_Z+.04], True)
    assert not score.success
    score.advance(points+[.52,0,.017], [.52,0,RIM_Z+.04], True)
    assert score.held_steps == 0
    for _ in range(18):
        score.advance(points+[.52,0,RIM_Z+.02], [.52,0,RIM_Z+.04], True)
    assert score.success


def test_opposing_corner_contacts_recover_a_nearby_square_face_direction():
    assert grasp_yaw_delta([[0,-.025,0],[0,.025,0]],[0,1,0]) == 0.
    for angle in (-45,-34,-20,20,34,45):
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta),-np.sin(theta)],
                             [np.sin(theta),np.cos(theta)]])
        corners = np.array([[x,y] for x in (-.019,.019) for y in (-.019,.019)])@rotation.T
        top = corners[corners[:,1].argmax()]+[0.,.006]
        yaw = np.pi/2+grasp_yaw_delta([-top,top],[0,1,0])
        error = (yaw-theta+np.pi/4)%(np.pi/2)-np.pi/4
        assert abs(np.rad2deg(error)) < 5.


def test_search_localizes_from_own_pad_contact_without_object_state():
    path = Path(__file__).resolve().parents[1]/'envs/blind_pick.py'
    task_class = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node,ast.ClassDef) and node.name == 'Task')
    methods = [node for node in task_class.body
               if isinstance(node,ast.FunctionDef) and node.name in ('_locate_contact','_pad_contacts')]
    scope = {'np':np}
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
    points = np.array([[.50,.02,.04],[.50,.024,.045]])
    geometry = SimpleNamespace(meta=lambda:SimpleNamespace(
        find=lambda name:SimpleNamespace(view=lambda:np.array([10]))))
    pad = SimpleNamespace(vertex_positions=points,
                          geo_slot_list=[SimpleNamespace(geometry=lambda:geometry)])
    dt = 1/120
    gradient = np.array([[-.1,0,0],[-.1,0,0],[100.,0,0]])*dt**2
    task = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(dt=dt)),
        uipc_sim=SimpleNamespace(get_contact_gradient=lambda:(np.array([10,11,900]),gradient)),
        _tactile_manager=SimpleNamespace(tactiles={'left':SimpleNamespace(gelpad=pad)}))
    task._pad_contacts = lambda:scope['_pad_contacts'](task)
    load, center, direction, patch = scope['_locate_contact'](task)
    np.testing.assert_allclose(load,.2)
    np.testing.assert_allclose(center,[.475,.022])
    np.testing.assert_allclose(direction,[-1.,0.])
    np.testing.assert_allclose(patch,[.50,.022,.0425])
    gradient[:2] = 0.
    assert scope['_locate_contact'](task) is None  # Other actors' loads cannot trigger search.


def test_grasp_keeps_first_loaded_jaw_still_until_bilateral_contact():
    from test_four_force_tasks import symbol
    path = Path(__file__).resolve().parents[1]/'envs/blind_pick.py'
    task_class = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node,ast.ClassDef) and node.name == 'Task')
    method = next(node for node in task_class.body
                  if isinstance(node,ast.FunctionDef) and node.name == '_close_on_contact')
    method.body = [node for node in method.body if not isinstance(node,ast.ImportFrom)]
    scope = dict(np=np,grasp_yaw_delta=grasp_yaw_delta,
                 next_grip_qpos=symbol('_force_task_utils','next_grip_qpos'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    state = dict(center=.005,q=.028,steps=0)
    def depths():
        penetration = np.array([state['center']-state['q']+.025,
                                .025-state['center']-state['q']])
        return np.where(penetration>=0,28.3-1000*penetration,34.)
    tensor = SimpleNamespace(numpy=depths)
    tensor.detach = lambda:tensor
    tensor.cpu = lambda:tensor
    loaded_positions = []
    def set_gripper(value, **kwargs):
        if (depths()<28.2).sum() == 1:
            loaded_positions.append(state['center']-state['q'])
        state['q'] = value
    def servo(position):
        state['center'] = position[1]
        state['steps'] += 2
        return True
    manager = SimpleNamespace(get_gripper_qpos=lambda:state['q'],set_gripper=set_gripper,
        get_gripper_center_pose=lambda:SimpleNamespace(p=np.array([0.,state['center'],.05])))
    sensors = {str(side):SimpleNamespace(get_attach_pose=lambda side=side:
        SimpleNamespace(p=np.array([0.,state['center']+side*state['q'],.05]))) for side in (-1,1)}
    task = SimpleNamespace(_robot_manager=manager,plan_success=True,servo_center=servo,
        _pad_contacts=lambda:[None,None],
        _tactile_manager=SimpleNamespace(get_min_depth=lambda:tensor,tactiles=sensors),
        _step=lambda:state.update(steps=state['steps']+1))
    assert scope['_close_on_contact'](task)
    assert len(loaded_positions)>2
    np.testing.assert_allclose(loaded_positions,loaded_positions[0],atol=1e-12)
    assert np.all(depths()<27.9)
    assert state['steps']<100


def test_grasp_remembers_contact_centre_and_unloads_before_repositioning():
    from test_four_force_tasks import symbol
    path = Path(__file__).resolve().parents[1]/'envs/blind_pick.py'
    task_class = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node,ast.ClassDef) and node.name == 'Task')
    method = next(node for node in task_class.body
                  if isinstance(node,ast.FunctionDef) and node.name == '_close_on_contact')
    method.body = [node for node in method.body if not isinstance(node,ast.ImportFrom)]
    scope = dict(np=np,grasp_yaw_delta=grasp_yaw_delta,
                 next_grip_qpos=symbol('_force_task_utils','next_grip_qpos'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    state = dict(center=np.array([.010,0.,.05]),q=.0245,steps=0)
    tensor = SimpleNamespace(numpy=lambda:np.full(2,
        34. if state['q'] > .025 else 28.3-1000*(.025-state['q'])))
    tensor.detach = lambda:tensor
    tensor.cpu = lambda:tensor
    def reposition(position):
        assert state['q'] == .039  # No loaded sliding along the pad.
        state['center'] = position
        state['steps'] += 2
        return True
    sensors = {str(side):SimpleNamespace(get_attach_pose=lambda side=side:
        SimpleNamespace(p=np.array([.010,side*.025,.05]))) for side in (-1,1)}
    task = SimpleNamespace(plan_success=True,_line_to=reposition,metadata={},step_count=0,
        _step=lambda:state.update(steps=state['steps']+1),
        _pad_contacts=lambda:[(1.,np.array([0.,side*.025,.05]),None)
            if state['q'] <= .025 else None for side in (-1,1)],
        _tactile_manager=SimpleNamespace(tactiles=sensors,get_min_depth=lambda:tensor),
        _robot_manager=SimpleNamespace(get_gripper_qpos=lambda:state['q'],
            set_gripper=lambda value,**kwargs:state.update(q=value),
            get_gripper_center_pose=lambda:SimpleNamespace(p=state['center'])))
    assert scope['_close_on_contact'](task)
    assert abs(state['center'][0]) < .003
    assert state['q'] < .025
    assert 0 < state['steps'] < 140
    assert len(task.metadata['blind_pick_alignment']) == 1


def test_lift_compensates_tactile_unloading_instead_of_dropping_the_object():
    from test_four_force_tasks import symbol
    path = Path(__file__).resolve().parents[1]/'envs/blind_pick.py'
    task_class = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node,ast.ClassDef) and node.name == 'Task')
    method = next(node for node in task_class.body
                  if isinstance(node,ast.FunctionDef) and node.name == '_lift_object')
    method.body = [node for node in method.body if not isinstance(node,ast.ImportFrom)]
    scope = dict(np=np,RIM_Z=RIM_Z,next_grip_qpos=symbol('_force_task_utils','next_grip_qpos'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    state = dict(center=np.array([.52,0.,.047]),q=.029,depth=27.85,steps=0)
    tensor = SimpleNamespace(numpy=lambda:np.array([state['depth'],state['depth']-.04]))
    tensor.detach = lambda:tensor
    tensor.cpu = lambda:tensor
    def close(value,**kwargs):
        state['depth'] += 2000*(value-state['q'])
        state['q'] = value
    def servo(position):
        state['center'] = position.copy()
        state['steps'] += 2
        state['depth'] += .02  # Progressive gel unloading under vertical shear.
        assert state['depth'] < 28.05
        return True
    task = SimpleNamespace(servo_center=servo,
        check_success=lambda:state['center'][2] == RIM_Z+.080 and state['depth']<28.,
        _tactile_manager=SimpleNamespace(get_min_depth=lambda:tensor),
        _robot_manager=SimpleNamespace(get_gripper_qpos=lambda:state['q'],set_gripper=close,
            get_gripper_center_pose=lambda:SimpleNamespace(p=state['center'])))
    assert scope['_lift_object'](task)
    assert 0 < .029-state['q'] < .002
    assert state['steps'] < 220
