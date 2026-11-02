"""Geometry invariants for the original convex-collider bridge."""
import numpy as np
import pytest
from env.tasks._blind_ball_models import convex_collision_mesh,convex_box_penetration


def cube(half=(.01,.01,.01)):
    return np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half


def test_convex_mesh_matches_asset_envelope_volume_and_orientation():
    corners=cube((.016,.014,.024))+[.001,-.002,.012]
    # Original CAD collision meshes duplicate positions across triangle seams.
    original=np.r_[corners,corners,corners.mean(0)[None]]
    points,tets,planes,edges=convex_collision_mesh(original)
    volume6=np.linalg.det(np.stack([points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1))
    assert (volume6>0).all()
    assert volume6.sum()/6==pytest.approx(.032*.028*.048)
    np.testing.assert_allclose(points.min(0),corners.min(0))
    np.testing.assert_allclose(points.max(0),corners.max(0))
    assert (original@planes[:,:3].T+planes[:,3]<=1.e-12).all()
    assert np.isfinite(edges).all()


@pytest.mark.parametrize('offset,expected',[(0.,.02),(.018,.002),(.02,0.),(.021,0.)])
def test_convex_box_depth_has_metric_scale_and_no_contact_false_positive(offset,expected):
    points,_,planes,edges=convex_collision_mesh(cube())
    result=convex_box_penetration(points,planes[:,:3],edges,np.array([offset,0,0]),np.full(3,.01))
    assert result==pytest.approx(expected,abs=1.e-12)


def test_thin_wall_crossing_is_detected_when_no_robot_vertex_is_inside_wall():
    points,_,planes,edges=convex_collision_mesh(cube())
    # A narrow wall crosses the cube away from both its corners and fan centre.
    assert not ((points[:,0]>.003)&(points[:,0]<.005)).any()
    depth=convex_box_penetration(points,planes[:,:3],edges,np.array([.004,0,0]),np.array([.001,.1,.1]))
    assert depth==pytest.approx(.007)


def test_rotated_box_test_is_invariant_under_world_frame_change():
    points,_,planes,edges=convex_collision_mesh(cube((.02,.01,.015)))
    angle=.73
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    center=np.array([.028,0,0]);half=np.array([.01,.02,.03])
    expected=convex_box_penetration(points,planes[:,:3],edges,center,half)
    shift=np.array([.4,-.2,.3])
    actual=convex_box_penetration(points@rotation.T+shift,planes[:,:3]@rotation.T,edges@rotation.T,
        center@rotation.T+shift,half,rotation)
    assert actual==pytest.approx(expected,abs=1.e-12)
