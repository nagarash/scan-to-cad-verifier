from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import cadquery as cq
import numpy as np
import pytest

from src import nist_experiment
from src.drawing import validate_transcription, verify_sizes
from src.reconstruct import UnsupportedScan
from src.ring import Ring, build_ring, fit_ring, read_ring_step, sample_ring_surface, verify_ring_steps
from src.synthetic_scan import sample_step


SOURCE = nist_experiment.DEFAULT_SOURCE
SPEC = json.loads((SOURCE/'drawing_checks.json').read_text())


def proposal():
    return {'units': 'mm', 'dimensions': deepcopy(SPEC['dimensions']), 'uncertainties': []}


def test_nist_sources_and_independent_cad_dimensions():
    provenance = nist_experiment.verify_sources(SOURCE)
    assert provenance['origin'].startswith('NIST')
    _, ring = read_ring_step(SOURCE/'reference.step')
    assert ring.dimensions() == pytest.approx({'outer_diameter_mm': 63, 'inner_diameter_mm': 32,
                                              'rim_radius_mm': 1.5, 'web_thickness_mm': 1.925})
    # Keep the drawing's nominal distinct from the actual STEP plane separation.
    assert SPEC['dimensions']['web_thickness_mm']['nominal'] == 1.93


@pytest.mark.parametrize('field,value', [('nominal', 99.), ('tolerance', 0.), ('page', 2)])
def test_wrong_transcription_is_rejected(field, value):
    p = proposal()
    p['dimensions']['outer_diameter_mm'][field] = value
    assert not validate_transcription(p, SPEC)['accepted']


def test_wrong_units_and_uncertainty_are_not_silently_accepted():
    p = proposal()
    p['units'] = 'inch'
    assert 'source_units' in validate_transcription(p, SPEC)['discrepancies']
    p = proposal()
    p['uncertainties'] = ['Cannot read the rim radius']
    assert not validate_transcription(p, SPEC)['accepted']


def test_drawing_checks_detect_actual_size_error():
    values = {k: v['nominal'] for k, v in SPEC['dimensions'].items()}
    assert verify_sizes(values, SPEC)['passed']
    values['rim_radius_mm'] += .1
    checked = verify_sizes(values, SPEC)
    assert not checked['passed']
    failed = [v['dimension'] for v in checked['checks'] if not v['passed']]
    assert failed == ['rim_radius_mm']
    assert checked['not_evaluated'] == SPEC['not_evaluated']


def test_synthetic_sampling_is_reproducible_with_recorded_corruption(tmp_path):
    a, meta = sample_step(SOURCE/'reference.step', tmp_path/'a', 3000, .01, .02, .1, 9)
    b, second = sample_step(SOURCE/'reference.step', tmp_path/'b', 3000, .01, .02, .1, 9)
    assert np.array_equal(a, b)
    assert meta == second
    assert meta['retained_points'] == len(a)
    assert meta['outlier_count'] == round(len(a)*.02)
    assert len(a) < 3000
    assert 'not a physical scanner' in meta['kind']


@pytest.mark.parametrize('kwargs', [{'n_points': 10}, {'noise_sigma_mm': -1},
                                   {'outlier_fraction': 1}, {'dropout_fraction': float('nan')}])
def test_invalid_scan_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        sample_step(SOURCE/'reference.step', tmp_path, **kwargs)


def test_non_ring_step_uses_mesh_sampling_fallback(tmp_path):
    step = tmp_path/'box.step'
    cq.exporters.export(cq.Workplane('XY').box(10, 6, 2), str(step))
    points, meta = sample_step(step, tmp_path/'scan', n_points=1000, noise_sigma_mm=0)
    assert points.shape == (1000, 3)
    assert np.isfinite(points).all()
    assert 'tessellation' in meta['sampling_method']
    assert np.all(np.max(np.abs(points), axis=0) <= np.array([5, 3, 1])+1e-6)


def test_overlapping_rims_are_rejected():
    with pytest.raises(UnsupportedScan, match='overlap'):
        build_ring(Ring([0., 0.], 10., 20., -1., 0., .2, 2.))


@pytest.fixture(scope='module')
def corrected_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp('nist_loop')
    report = nist_experiment.run(folder, seed=42)
    return folder, report


def test_real_nist_loop_corrects_outlier_failure_without_nominal_snapping(corrected_run):
    folder, r = corrected_run
    assert r['accepted'] and r['outcome'] == 'accepted_after_refit'
    first, last = r['iterations']
    assert not first['drawing_verification']['passed']
    assert {'outer_diameter_mm', 'rim_radius_mm'} <= set(first['failed_dimensions'])
    assert last['drawing_verification']['passed']
    assert last['evidence_coverage_passed']
    assert last['parameters']['outer_diameter_mm'] != 63.
    assert abs(last['parameters']['outer_diameter_mm']-63.) < .01
    assert last['held_out_step_geometry']['sampled_p95_surface_distance_mm'] < .01
    assert (folder/'iteration_0/reconstructed.step').exists()
    assert Path(r['selected_step']).exists()
    assert not r['measurement_uses_drawing_dimensions']
    assert 'solid_iou' not in last['held_out_step_geometry']


def test_identical_step_surface_verification_has_no_sampling_floor():
    r = verify_ring_steps(SOURCE/'reference.step', SOURCE/'reference.step', n_samples=1000)
    assert max(r['cad_feature_absolute_errors_mm'].values()) == 0.
    assert r['sampled_max_surface_distance_mm'] < .002


def test_unseen_annular_dimensions_and_translation():
    original = Ring([12., -7.], 28., 55., -2., 1., 1.4, 1.25)
    part = build_ring(original)
    assert part.val().isValid()
    points, area = sample_ring_surface(original, 20000, 18)
    assert area == pytest.approx(part.val().Area())
    points += np.random.default_rng(18).normal(0, .005, points.shape)
    fitted = fit_ring(points, robust=True)
    assert fitted.dimensions() == pytest.approx(original.dimensions(), abs=.015)
    assert fitted.center_xy_mm == pytest.approx(original.center_xy_mm, abs=.01)
    assert fitted.explained_fraction > .99


def test_exported_cad_feature_error_detects_wrong_radius(tmp_path):
    _, ref = read_ring_step(SOURCE/'reference.step')
    candidate = build_ring(replace(ref, rim_radius_mm=ref.rim_radius_mm+.1))
    step = tmp_path/'wrong.step'
    cq.exporters.export(candidate, str(step))
    r = verify_ring_steps(SOURCE/'reference.step', step, n_samples=1000)
    assert r['cad_feature_absolute_errors_mm']['rim_radius_mm'] == pytest.approx(.1)
    assert r['sampled_max_surface_distance_mm'] > .05


def test_replay_bad_drawing_blocks_scan_generation(tmp_path, monkeypatch):
    bad = proposal()
    bad['units'] = 'inch'
    path = tmp_path/'bad.json'
    path.write_text(json.dumps({'proposal': bad}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not generate a scan after a failed transcription')
    monkeypatch.setattr(nist_experiment, 'sample_step', forbidden)
    result = nist_experiment.run(tmp_path/'out', drawing_response=path)
    assert not result['accepted']
    assert not result['iterations']
    assert result['transcription_review']['discrepancies'] == ['source_units']


@pytest.mark.skipif(shutil.which('pdftoppm') is None, reason='Optional Poppler drawing renderer is not installed')
def test_bounded_drawing_retry_has_no_expected_numeric_feedback(tmp_path, monkeypatch):
    calls = []
    bad = proposal()
    for v in bad['dimensions'].values(): v['tolerance'] = 0.
    def interpret(*args, **kwargs):
        calls.append(kwargs)
        return {'proposal': bad if len(calls) == 1 else proposal(), 'test_double': True}
    monkeypatch.setattr(nist_experiment, 'interpret_drawing', interpret)
    # Exercise real PDF rendering/cropping, but do not repeat geometry fitting here.
    def refuse(*args, **kwargs): raise UnsupportedScan('Test stops at geometry boundary')
    monkeypatch.setattr(nist_experiment, 'fit_ring', refuse)
    result = nist_experiment.run(tmp_path, live_drawing=True, n_points=3000)
    assert len(calls) == 2
    assert not result['drawing_attempts'][0]['review']['accepted']
    assert result['drawing_attempts'][1]['review']['accepted']
    assert len(calls[1]['crops']) == 3
    assert '63' not in calls[1]['feedback'] and '0.5' not in calls[1]['feedback']
    from PIL import Image
    # Regression guard for a mistakenly blank web-thickness crop.
    with Image.open(calls[1]['crops'][2][2]) as image:
        assert np.mean(np.array(image) < 230) > .03


@pytest.mark.parametrize('points', [np.zeros((20, 3)), np.zeros((1000, 2)), np.full((1000, 3), np.nan)])
def test_invalid_ring_scan(points):
    with pytest.raises(ValueError): fit_ring(points)
