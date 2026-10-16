"""Independent geometry checks for partial cylindrical feature extraction."""
from dataclasses import replace
import numpy as np
import pytest
from src.cylinder_features import discover_cylinders, export_patches
from src.ctc05_experiment import evaluate
from src.synthetic_scan import sample_step


def cylinder_points(radius=13.7, center=(17.,-8.), seed=6):
    rng=np.random.default_rng(seed)
    a=rng.uniform(-np.pi,np.pi,9000); z=rng.uniform(4,55,9000)
    p=np.column_stack((center[0]+radius*np.cos(a),center[1]+radius*np.sin(a),z))
    p+=rng.normal(0,.008,p.shape)
    return p


def test_unseen_cylinder_translation_and_open_export(tmp_path):
    import cadquery as cq
    features=discover_cylinders(cylinder_points(),tolerance_mm=.07)
    assert len(features)==1
    f=features[0]
    assert abs(f.radius_mm-13.7)<.005
    assert np.linalg.norm(np.array(f.center_xy_mm)-[17,-8])<.005
    path=tmp_path/'patches.step'; export_patches(features,path)
    shape=cq.importers.importStep(str(path)).val()
    assert len(shape.Solids())==0
    assert len(shape.Faces())==1
    ref=[{'center_xy_mm':[17,-8],'radius_mm':13.7}]
    assert abs(evaluate(features,ref)[0]['diameter_error_mm'])<.01
    wrong=evaluate([replace(f,radius_mm=f.radius_mm+1)],ref)
    assert wrong[0]['diameter_error_mm']>1.99


def test_planar_cloud_does_not_become_cylinder():
    rng=np.random.default_rng(3)
    p=np.column_stack((rng.uniform(-30,30,(5000,2)),np.zeros(5000)))
    assert discover_cylinders(p)==[]


@pytest.mark.parametrize('tol',[0,-1,float('nan')])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        discover_cylinders(cylinder_points(),tolerance_mm=tol)


@pytest.mark.parametrize('linear,angular',[(0,.1),(.01,0),(.01,float('nan'))])
def test_invalid_sampling_resolution(tmp_path,linear,angular):
    with pytest.raises(ValueError,match='tessellation'):
        sample_step('unused.step',tmp_path,linear_tolerance_mm=linear,angular_tolerance_rad=angular)
