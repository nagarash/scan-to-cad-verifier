"""Integration checks use independent CAD construction and actual STEP round trips."""
import json

import cadquery as cq
import numpy as np
import pytest
import trimesh

from src.dataset import build_part
from src.reconstruct import UnsupportedScan, fit_plate, reconstruct
from src.verification import verify_steps


def sample(part, count=12000, noise=.05, seed=42):
    vertices, triangles = part.val().tessellate(.01, .05)
    mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in vertices], faces=triangles, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return points + np.random.default_rng(seed).normal(0, noise, points.shape)


@pytest.mark.parametrize('name,extent,count,diameter', [
    ('four_hole_plate', [60, 40, 6], 4, 6),
    ('two_hole_plate', [55, 30, 5], 2, 8),
])
@pytest.mark.parametrize('noise,seed', [(0., 7), (.05, 42), (.05, 123)])
def test_reference_reconstruction(tmp_path, name, extent, count, diameter, noise, seed):
    reference = build_part(name)
    scan = tmp_path/'scan.npy'
    np.save(scan, sample(reference, noise=noise, seed=seed))
    # Reconstruction runs before reference export and receives only the scan path.
    report = reconstruct(scan, tmp_path/'result')
    params = report['parameters']
    assert [params['length_mm'], params['width_mm'], params['thickness_mm']] == pytest.approx(extent, abs=.03)
    assert len(params['holes']) == count
    assert all(abs(h['diameter_mm']-diameter) < .04 for h in params['holes'])
    step = tmp_path/'reference.step'
    cq.exporters.export(reference, str(step))
    measured = verify_steps(step, tmp_path/'result/reconstructed.step')
    assert measured['passed'], measured
    assert measured['hole_f1'] == 1
    assert measured['solid_iou'] > .99
    assert json.loads((tmp_path/'result/reconstruction.json').read_text()) == report


def test_unseen_dimensions_and_translation(tmp_path):
    # Different sizes, radii, asymmetrical layout, and nonzero world coordinates.
    reference = cq.Workplane('XY').box(73, 47, 7, centered=(True, True, False))
    for x, y, diameter in [(-21, -9, 7), (17, 12, 10), (4, -11, 5)]:
        reference = reference.cut(cq.Workplane('XY').center(x, y).circle(diameter/2).extrude(7))
    reference = reference.translate((103, -61, 14))
    scan = tmp_path/'scan.npy'
    np.save(scan, sample(reference, count=24000, seed=91))
    report = reconstruct(scan, tmp_path/'out')
    assert report['parameters']['center_xy_mm'] == pytest.approx([103, -61], abs=.02)
    cq.exporters.export(reference, str(tmp_path/'ref.step'))
    assert verify_steps(tmp_path/'ref.step', tmp_path/'out/reconstructed.step')['passed']


@pytest.mark.parametrize('name', ['slotted_plate', 'raised_boss', 'l_bracket'])
def test_rejects_unsupported_shapes(name):
    with pytest.raises(UnsupportedScan):
        fit_plate(sample(build_part(name)))


def test_rejects_rotated_plate():
    part = build_part('four_hole_plate').rotate((0, 0, 0), (0, 0, 1), 17)
    with pytest.raises(UnsupportedScan):
        fit_plate(sample(part))


@pytest.mark.parametrize('points', [np.empty((0, 3)), np.zeros((500, 2)),
                                    np.full((500, 3), np.nan), np.full((500, 3), np.inf)])
def test_invalid_cloud(points):
    with pytest.raises(ValueError, match='finite Nx3'):
        fit_plate(points)


def test_missing_cylinder_evidence():
    points = sample(build_part('four_hole_plate'), noise=0.)
    # Preserve only top and bottom faces; visible holes alone do not measure cylinders.
    points = points[(points[:, 2] < 1e-8) | (points[:, 2] > 6-1e-8)]
    with pytest.raises(UnsupportedScan):
        fit_plate(points)


def test_verification_detects_missing_hole_and_dimension_error(tmp_path):
    reference = build_part('two_hole_plate')
    wrong = (cq.Workplane('XY').box(56, 30, 5, centered=(True, True, False))
             .faces('>Z').workplane().pushPoints([(-17, 0)]).hole(9))
    for name, part in [('ref', reference), ('wrong', wrong)]:
        cq.exporters.export(part, str(tmp_path/f'{name}.step'))
    report = verify_steps(tmp_path/'ref.step', tmp_path/'wrong.step')
    assert not report['passed']
    assert report['missing_holes'] == 1
    assert report['extra_holes'] == 0
    assert report['extent_absolute_errors_mm'][0] == pytest.approx(1.)
    assert report['hole_matches'][0]['diameter_error_mm'] == pytest.approx(1.)
    assert report['hole_recall'] == .5
    assert report['symmetric_volume_difference_fraction'] > .01


def test_identical_step_verification(tmp_path):
    cq.exporters.export(build_part('four_hole_plate'), str(tmp_path/'part.step'))
    report = verify_steps(tmp_path/'part.step', tmp_path/'part.step')
    assert report['passed']
    assert report['solid_iou'] == pytest.approx(1.)
    assert report['symmetric_volume_difference_fraction'] == pytest.approx(0.)
