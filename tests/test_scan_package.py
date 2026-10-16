import json
import numpy as np
import pytest
from src.scan_package import generate, projected_samples
from src.cad_evaluation import evaluate


def test_projected_scan_exact_surfaces_and_portable_input(tmp_path):
    import cadquery as cq
    import trimesh
    step=tmp_path/'source.step'
    cq.exporters.export(cq.Workplane('XY').circle(13.7).extrude(27.3),str(step))
    points,quality=projected_samples(step,1500,min_per_face=50)
    radius=np.linalg.norm(points[:,:2],axis=1)
    distance=np.minimum.reduce((abs(radius-13.7),abs(points[:,2]),abs(points[:,2]-27.3)))
    assert distance.max()<1e-7
    assert len(quality['face_coverage'])==3
    scan,q=generate(step,tmp_path/'package',1500,noise_mm=0,min_per_face=50)
    assert q['noise_displacement_p95_mm']==0
    input_dir=tmp_path/'package/input'
    assert {p.name for p in input_dir.iterdir()}=={'scan_points_mm.npy','scan.xyz','scan.ply','README.txt'}
    ply=trimesh.load(input_dir/'scan.ply')
    assert np.allclose(np.asarray(ply.vertices),scan)
    assert np.allclose(np.loadtxt(input_dir/'scan.xyz'),scan)
    assert (tmp_path/'package/evaluation/scan_quality.json').exists()


def test_held_out_evaluator_accepts_identity_rejects_wrong_geometry(tmp_path):
    import cadquery as cq
    reference=tmp_path/'reference.step';wrong=tmp_path/'wrong.step'
    cq.exporters.export(cq.Workplane('XY').circle(13.7).extrude(27.3),str(reference))
    cq.exporters.export(cq.Workplane('XY').circle(14.2).extrude(27.3),str(wrong))
    spec={'checks':[]}
    result=evaluate(reference,reference,spec,tmp_path/'same.json')
    assert result['accepted']
    assert result['surface_comparison']['sampled_max_mm']<1e-6
    result=evaluate(wrong,reference,spec,tmp_path/'wrong.json')
    assert not result['accepted']
    assert abs(result['cylinder_comparison']['matches'][0]['diameter_error_mm']-1)<1e-6
    assert result['surface_comparison']['sampled_max_mm']>.4


@pytest.mark.parametrize('noise',[-1,float('nan')])
def test_invalid_noise_rejected_before_reading_source(tmp_path,noise):
    with pytest.raises(ValueError,match='noise'):
        generate('missing.step',tmp_path,noise_mm=noise)


def test_complex_reference_volume_uses_accurate_integration():
    # The reference STEP contains a loose auxiliary face outside its closed solid.
    # Including it in mass properties overestimates physical volume by ~1.06%.
    from pathlib import Path
    from src.cad_evaluation import inventory
    source=Path(__file__).resolve().parents[1]/'data/nist/ctc05/reference.step'
    result=inventory(source)
    assert result['ignored_auxiliary_faces']==1
    assert result['face_count']==208
    assert abs(result['volume_mm3']-12694921.72)<2
