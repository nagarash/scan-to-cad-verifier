"""Single-pass AI interpretation -> geometric validation -> CAD selection.

This milestone tests interpretation and gating, not AI dimensional correction.
The deterministic baseline is preserved even when the model fails or abstains.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from openai import OpenAIError
from scipy.optimize import linear_sum_assignment

from src.model import InvalidModelResponse, ScanProposal, interpret_scan_image
from src.reconstruct import UnsupportedScan, reconstruct
from src.scan_views import render_scan
from src.verification import verify_steps


def validate_proposal(proposal, parameters, relation_tolerance_mm=.1):
    """Validate only against scan-derived geometry; no reference CAD is accepted."""
    proposal = ScanProposal.model_validate(proposal).model_dump()
    if not np.isfinite(relation_tolerance_mm) or relation_tolerance_mm <= 0:
        raise ValueError('Relation tolerance must be positive and finite')
    checks = []
    uncertain = (proposal['part_family'] == 'uncertain' or proposal['hole_shape'] == 'uncertain'
                 or proposal['visible_hole_count'] is None)
    checks.append({'claim': 'supported_part_family',
                   'passed': proposal['part_family'] == 'rectangular_plate_with_round_holes'})
    checks.append({'claim': 'circular_holes', 'passed': proposal['hole_shape'] == 'circular'})
    holes = parameters['holes']
    checks.append({'claim': 'hole_count', 'proposed': proposal['visible_hole_count'],
                   'measured': len(holes), 'passed': proposal['visible_hole_count'] == len(holes)})
    center = np.array(parameters['center_xy_mm'])
    centers = np.array([h['center_xy_mm'] for h in holes])
    diameters = np.array([h['diameter_mm'] for h in holes])
    for axis in proposal['symmetry_axes']:
        index = {'x_midplane': 0, 'y_midplane': 1}[axis]
        reflected = centers.copy()
        reflected[:, index] = 2*center[index]-centers[:, index]
        distances = np.linalg.norm(reflected[:, None]-centers[None, :], axis=2)
        diameter_errors = np.abs(diameters[:, None]-diameters[None, :])
        compatible = (distances <= relation_tolerance_mm) & (diameter_errors <= relation_tolerance_mm)
        rows, cols = linear_sum_assignment(np.where(compatible, distances, 1e9))
        checks.append({'claim': axis, 'passed': bool(compatible[rows, cols].all()),
                       'max_center_error_mm': float(distances[rows, cols].max()),
                       'max_diameter_error_mm': float(diameter_errors[rows, cols].max())})
    status = 'abstained' if uncertain else ('accepted' if all(c['passed'] for c in checks) else 'rejected')
    return {'status': status, 'checks': checks, 'relation_tolerance_mm': relation_tolerance_mm,
            'note': 'Text hypotheses and uncertainty explanations are recorded, not executed or validated.'}


def run(scan_path, output_dir='outputs/ai_loop', config_path='config/settings.yaml',
        reference_path=None, proposal_file=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    points = np.load(scan_path, allow_pickle=False)
    image = render_scan(points, root/'scan_views.png')
    report = {'mode': 'offline_replay' if proposal_file else 'live_model',
              'model_response_received': False,
              'scan_sha256': hashlib.sha256(Path(scan_path).read_bytes()).hexdigest(),
              'model_input_image': str(image), 'selected_step': None,
              'scope': 'AI interpretation and CAD acceptance gate; dimensions are unchanged',
              'ground_truth_sent_to_model': False}
    # Interpretation happens before fitting, so even measured hole counts cannot leak.
    try:
        if proposal_file:
            proposal = ScanProposal.model_validate_json(Path(proposal_file).read_text()).model_dump()
            report['ai'] = {'proposal': proposal, 'source': str(proposal_file),
                            'note': 'Offline supplied proposal; this is not a live model result.'}
        else:
            report['ai'] = interpret_scan_image(image, config_path)
            report['model_response_received'] = True
    except InvalidModelResponse as exc:
        report['model_response_received'] = exc.metadata is not None
        report['ai'] = {'status': 'invalid_response', 'error': str(exc),
                        'raw_response': exc.raw_response, 'call': exc.metadata}
    except (OpenAIError, RuntimeError, ValueError, OSError) as exc:
        # Provider errors can contain arbitrary response bodies; keep secrets out of reports.
        report['ai'] = {'status': 'model_error', 'error_type': type(exc).__name__,
                        'error': 'Interpretation failed; check configuration, credentials, network, or proposal schema.'}
    try:
        baseline = reconstruct(scan_path, root/'baseline')
    except UnsupportedScan as exc:
        report['baseline'] = {'status': 'unsupported', 'error': str(exc)}
        report['gate'] = {'status': 'geometry_rejected', 'note': 'AI cannot override unsupported geometry.'}
    else:
        report['baseline'] = {'status': 'reconstructed', **baseline}
        if 'proposal' in report['ai']:
            report['gate'] = validate_proposal(report['ai']['proposal'], baseline['parameters'])
        else:
            report['gate'] = {'status': 'no_valid_proposal'}
        step = root/'baseline/reconstructed.step'
        if report['gate']['status'] == 'accepted':
            report['selected_step'] = str(step)
        if reference_path:
            baseline_verification = verify_steps(reference_path, step)
            report['comparison'] = {'baseline': baseline_verification,
                                    'ai_gated': baseline_verification if report['selected_step'] else None,
                                    'geometry_changed': False,
                                    'note': 'Accepted geometry equals baseline; no accuracy improvement is claimed.'}
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan')
    parser.add_argument('--output-dir', default='outputs/ai_loop')
    parser.add_argument('--config', default='config/settings.yaml')
    parser.add_argument('--reference', help='Optional held-out STEP, used only after reconstruction')
    parser.add_argument('--proposal-file', help='Offline JSON replay; makes no model call')
    args = parser.parse_args()
    report = run(args.scan, args.output_dir, args.config, args.reference, args.proposal_file)
    print(f'Mode: {report["mode"]}; gate: {report["gate"]["status"]}; {args.output_dir}/report.json')
    if report['selected_step'] is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
