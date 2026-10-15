import ast
from pathlib import Path
import numpy as np
import pytest
from env.tasks._blind_ball_models import (
    MATERIALS,CLASSES,FLOOR_Z,BOWL_CENTER,PickScore,
    ball_mesh,sample_scene,surface_roughness,classify_touch,
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


@pytest.mark.parametrize('count',[2,3])
@pytest.mark.parametrize('target',CLASSES)
def test_random_layout_has_target_and_clearance(count,target):
    samples=[]
    for seed in range(30):
        materials,p=sample_scene(np.random.default_rng(seed),count,target)
        assert target in {m.kind for m in materials}
        assert len({m.kind for m in materials})==count
        assert p.shape==(count,3)
        distances=np.linalg.norm(p[:,None,:2]-p[None,:,:2],axis=-1)+np.eye(count)
        assert distances.min()>.078
        assert (p[:,0]>.44).all() and (p[:,0]<.58).all()
        assert (np.abs(p[:,1])<.13).all()
        samples.append(p[0,:2])
    assert len(np.unique(samples,axis=0))==30


def test_probe_features_distinguish_smooth_soft_and_real_relief():
    x,y=np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    smooth=.0015-.0004*(x*x+y*y)
    rough=smooth+.0002*np.cos(20*x)*np.cos(18*y)
    assert surface_roughness(smooth)<1e-12
    assert surface_roughness(rough)>6e-5
    assert classify_touch(.0001,surface_roughness(smooth))=='hard'
    assert classify_touch(.002,surface_roughness(smooth))=='soft'
    assert classify_touch(.0001,surface_roughness(rough))=='textured'


def test_scoring_requires_correct_ball_verified_lift_and_release():
    material=next(m for m in MATERIALS if m.kind=='textured')
    p,_=ball_mesh(material)
    p+=np.r_[BOWL_CENTER,FLOOR_Z+.025]
    for q,kind,verified in ((.02,'textured',True),(.04,'hard',True),(.04,'textured',False)):
        score=PickScore()
        for _ in range(40): score.advance(p,q,'textured',kind,verified,1/120)
        assert not score.success
    score=PickScore()
    for _ in range(29): score.advance(p,.04,'textured','textured',True,1/120)
    assert not score.success
    score.advance(p,.04,'textured','textured',True,1/120)
    assert score.success


def test_touch_controller_has_no_actor_state_or_camera_reads():
    path=Path(__file__).resolve().parents[1]/'env/tasks/blind_pick_balls.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    task=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    controller={'play_once','_go','_locate','_squeeze','_probe'}
    forbidden={'ball_points','materials','_initial_centers','render','camera','scores','diagnostics'}
    for method in task.body:
        if isinstance(method,ast.FunctionDef) and method.name in controller:
            assert not {n.attr for n in ast.walk(method) if isinstance(n,ast.Attribute)}&forbidden
