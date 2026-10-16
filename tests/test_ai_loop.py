import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from src import ai_loop, model
from src.geometry import make_bracket


def proposal(**changes):
    value = {'part_family': 'rectangular_plate_with_round_holes', 'visible_hole_count': 4,
             'hole_shape': 'circular', 'symmetry_axes': ['x_midplane', 'y_midplane'],
             'hypotheses': ['A repeated mounting-hole arrangement'],
             'uncertainties': ['Images alone do not establish through-hole depth']}
    return value | changes


def parameters():
    return {'center_xy_mm': [0, 0], 'holes': [
        {'center_xy_mm': [x, y], 'diameter_mm': 6}
        for x, y in [(-20, -10), (-20, 10), (20, -10), (20, 10)]]}


@pytest.mark.parametrize('changes', [{'visible_hole_count': '4'}, {'visible_hole_count': True},
                                    {'visible_hole_count': -1}, {'length_mm': 60},
                                    {'symmetry_axes': ['z']}, {'part_family': 'bracket'}])
def test_schema_rejects_invalid_or_dimensional_proposals(changes):
    with pytest.raises(ValidationError):
        model.ScanProposal.model_validate(proposal(**changes))


def test_agreement_gate():
    result = ai_loop.validate_proposal(proposal(), parameters())
    assert result['status'] == 'accepted'
    assert all(c['passed'] for c in result['checks'])


def test_wrong_hole_count_is_rejected():
    result = ai_loop.validate_proposal(proposal(visible_hole_count=3), parameters())
    assert result['status'] == 'rejected'
    assert not next(c for c in result['checks'] if c['claim'] == 'hole_count')['passed']


def test_incorrect_symmetry_is_rejected():
    p = parameters()
    p['holes'][0]['center_xy_mm'][0] += 2
    assert ai_loop.validate_proposal(proposal(), p)['status'] == 'rejected'


def test_uncertainty_abstains():
    assert ai_loop.validate_proposal(proposal(visible_hole_count=None), parameters())['status'] == 'abstained'


@pytest.fixture(scope='module')
def scan_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp('ai_scan')
    make_bracket(folder, n_points=12000, noise_sigma_mm=.05)
    return folder/'scan_points_mm.npy', folder/'ground_truth.step'


def test_offline_replay_and_step_comparison(tmp_path, scan_case):
    scan, ref = scan_case
    saved = tmp_path/'proposal.json'
    saved.write_text(json.dumps(proposal()))
    report = ai_loop.run(scan, tmp_path/'out', reference_path=ref, proposal_file=saved)
    assert report['mode'] == 'offline_replay'
    assert not report['model_response_received']
    assert report['gate']['status'] == 'accepted'
    assert report['comparison']['baseline']['passed']
    assert report['comparison']['ai_gated'] == report['comparison']['baseline']
    assert not report['comparison']['geometry_changed']
    assert Path(report['selected_step']).is_file()
    with Image.open(report['model_input_image']) as image:
        assert image.size == (1400, 1120)


def test_model_error_preserves_baseline(tmp_path, scan_case, monkeypatch):
    def failing(*args):
        raise RuntimeError('sensitive provider error body')
    monkeypatch.setattr(ai_loop, 'interpret_scan_image', failing)
    report = ai_loop.run(scan_case[0], tmp_path)
    assert report['selected_step'] is None
    assert report['gate']['status'] == 'no_valid_proposal'
    assert report['baseline']['status'] == 'reconstructed'
    assert not report['model_response_received']
    assert 'sensitive provider' not in (tmp_path/'report.json').read_text()


def test_model_interpretation_precedes_measurements(tmp_path, scan_case, monkeypatch):
    events = []
    real_reconstruct = ai_loop.reconstruct
    def fake_interpret(image, config):
        events.append('interpret')
        assert Path(image).name == 'scan_views.png'
        return {'proposal': proposal(), 'call': {'test_double': True}}
    def measured(*args):
        events.append('measure')
        return real_reconstruct(*args)
    monkeypatch.setattr(ai_loop, 'interpret_scan_image', fake_interpret)
    monkeypatch.setattr(ai_loop, 'reconstruct', measured)
    report = ai_loop.run(scan_case[0], tmp_path, reference_path=scan_case[1])
    assert events == ['interpret', 'measure']
    assert report['gate']['status'] == 'accepted'
    assert report['ground_truth_sent_to_model'] is False
    assert report['model_response_received']


@pytest.mark.parametrize('changes,status', [({'visible_hole_count': 3}, 'rejected'),
                                          ({'visible_hole_count': None}, 'abstained'),
                                          ({'part_family': 'other'}, 'rejected')])
def test_disagreement_selects_no_cad(tmp_path, scan_case, monkeypatch, changes, status):
    monkeypatch.setattr(ai_loop, 'interpret_scan_image', lambda *args: {'proposal': proposal(**changes)})
    report = ai_loop.run(scan_case[0], tmp_path)
    assert report['gate']['status'] == status
    assert report['selected_step'] is None
    assert (tmp_path/'baseline/reconstructed.step').exists()


def test_invalid_response_is_saved_without_acceptance(tmp_path, scan_case, monkeypatch):
    def invalid(*args):
        raise model.InvalidModelResponse('Invalid JSON', 'broken JSON', {'response_id': 'test-id'})
    monkeypatch.setattr(ai_loop, 'interpret_scan_image', invalid)
    report = ai_loop.run(scan_case[0], tmp_path)
    assert report['ai']['status'] == 'invalid_response'
    assert report['ai']['raw_response'] == 'broken JSON'
    assert report['model_response_received']
    assert report['selected_step'] is None


def test_unsupported_geometry_cannot_be_overridden(tmp_path, scan_case, monkeypatch):
    monkeypatch.setattr(ai_loop, 'interpret_scan_image', lambda *args: {'proposal': proposal()})
    def unsupported(*args):
        raise ai_loop.UnsupportedScan('No supported cylinders')
    monkeypatch.setattr(ai_loop, 'reconstruct', unsupported)
    report = ai_loop.run(scan_case[0], tmp_path)
    assert report['gate']['status'] == 'geometry_rejected'
    assert report['selected_step'] is None


@pytest.mark.parametrize('content,finish,valid', [
    (json.dumps(proposal()), 'stop', True),
    ('not JSON', 'stop', False),
    (json.dumps(proposal(visible_hole_count='4')), 'stop', False),
    (json.dumps(proposal()), 'length', False),
])
def test_hosted_request_and_response_contract(tmp_path, monkeypatch, content, finish, valid):
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model='test-model', id='test-id', usage=None,
                               choices=[SimpleNamespace(finish_reason=finish,
                                                        message=SimpleNamespace(content=content))])
    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(model, 'OpenAI', FakeClient)
    monkeypatch.setattr(model, 'load_dotenv', lambda *args: None)
    monkeypatch.setenv('TEST_MODEL_KEY', 'test-key')
    config = tmp_path/'config.yaml'
    config.write_text('model:\n  name: test-model\n  base_url: https://example.invalid\n  api_key_env: TEST_MODEL_KEY\n')
    image = tmp_path/'anonymous.png'
    Image.new('RGB', (10, 10)).save(image)
    if valid:
        result = model.interpret_scan_image(image, config)
        assert result['proposal'] == proposal()
        assert result['call']['returned_model'] == 'test-model'
    else:
        with pytest.raises(model.InvalidModelResponse) as caught:
            model.interpret_scan_image(image, config)
        assert caught.value.raw_response == content
    assert captured['max_tokens'] == 1200
    assert captured['response_format']['type'] == 'json_schema'
    payload = json.dumps(captured['messages'])
    assert 'data:image/png;base64,' in payload
    assert 'reference.step' not in payload and 'test-key' not in payload


def test_no_key_fails_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr(model, 'load_dotenv', lambda *args: None)
    monkeypatch.delenv('TEST_ABSENT_KEY', raising=False)
    config = tmp_path/'config.yaml'
    config.write_text('model:\n  api_key_env: TEST_ABSENT_KEY\n')
    with pytest.raises(RuntimeError, match='TEST_ABSENT_KEY'):
        model.interpret_scan_image(tmp_path/'missing.png', config)
