"""NIST FTC-11: synthetic scan -> size verification -> bounded robust refit.

The source STEP is used for simulation and held-out evaluation only. Drawing
values decide pass/fail; no nominal size is passed into the geometry fitter.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.drawing import (DrawingProposal, callout_crops, interpret_drawing, render_drawing,
                         validate_transcription, verify_sizes)
from src.reconstruct import UnsupportedScan
from src.ring import export_ring, fit_ring, verify_ring_steps
from src.scan_views import render_scan
from src.synthetic_scan import sample_step


DEFAULT_SOURCE = Path(__file__).resolve().parents[1]/'data/nist/ftc11'


def verify_sources(directory):
    root = Path(directory)
    provenance = json.loads((root/'provenance.json').read_text())
    for name, details in provenance['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != details['sha256']:
            raise ValueError(f'NIST source checksum mismatch: {name}')
    return provenance


def run(output_dir='outputs/nist/ftc11', source_dir=DEFAULT_SOURCE, seed=42,
        n_points=30000, noise_mm=.01, outliers=.02, dropout=0., live_drawing=False,
        drawing_response=None, config_path='config/settings.yaml'):
    source, out = Path(source_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = verify_sources(source)
    reviewed = json.loads((source/'drawing_checks.json').read_text())
    report = {'part_id': reviewed['part_id'], 'source_provenance': provenance,
              'mode': 'live_drawing_model' if live_drawing else ('drawing_response_replay' if drawing_response else 'reviewed_drawing'),
              'drawing_spec': reviewed, 'accepted': False, 'selected_step': None, 'iterations': [],
              'measurement_uses_drawing_dimensions': False,
              'scope': 'Known aligned annular topology; four selected size checks; synthetic data only',
              'correction_policy': 'At most one soft-L1 robust refit after failed checks; no nominal-size snapping'}
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    # Resolve drawing evidence before sampling; model receives only the PDF images.
    if live_drawing or drawing_response:
        if live_drawing:
            pages = render_drawing(source/'drawing.pdf', out/'drawing_pages')
            if drawing_response:
                response = json.loads(Path(drawing_response).read_text())
                report['resumed_drawing_response'] = str(drawing_response)
            else:
                try:
                    response = interpret_drawing(pages, config_path)
                except Exception as exc:
                    # Persist a diagnostic without arbitrary provider bodies or credentials.
                    report['drawing_model'] = {'status': 'error', 'error_type': type(exc).__name__}
                    report['blocked_reason'] = 'No valid drawing transcription; scan experiment was not run'
                    return save()
            (out/'drawing_response.json').write_text(json.dumps(response, indent=2)+'\n')
        else:
            response = json.loads(Path(drawing_response).read_text())
        report['drawing_model'] = response
        review = validate_transcription(response['proposal'], reviewed)
        report['drawing_attempts'] = [{'response': response, 'review': review}]
        if live_drawing and not review['accepted']:
            crops = callout_crops(pages, out/'drawing_callouts')
            feedback = ('The first transcription failed independent review in these fields: '
                        + ', '.join(review['discrepancies'])
                        + '. Re-read the full pages and enlarged callouts. Copy the printed +/- tolerances; '
                        'do not default them to zero. Trace the section-view extension lines when interpreting '
                        'web thickness. Return a complete replacement JSON transcription, with uncertainty '
                        'if the selected values remain illegible.')
            try:
                response = interpret_drawing(pages, config_path, crops=crops, feedback=feedback)
            except Exception as exc:
                report['drawing_retry_error_type'] = type(exc).__name__
            else:
                review = validate_transcription(response['proposal'], reviewed)
                report['drawing_attempts'].append({'response': response, 'review': review})
                report['drawing_model'] = response
                (out/'drawing_response_corrected.json').write_text(json.dumps(response, indent=2)+'\n')
        report['transcription_review'] = review
        if not review['accepted']:
            report['blocked_reason'] = 'Drawing transcription disagrees with reviewed evidence; no correction allowed'
            return save()
        proposed = DrawingProposal.model_validate(response['proposal']).model_dump()
        # Retain the reviewed not-evaluated list and source; use accepted transcribed sizes.
        reviewed = reviewed | {'units': proposed['units'], 'dimensions': proposed['dimensions']}
        report['drawing_spec'] = reviewed
    scan, scan_metadata = sample_step(source/'reference.step', out/'scan', n_points, noise_mm,
                                      outliers, dropout, seed)
    report['scan'] = scan_metadata
    render_scan(scan, out/'scan_views.png')
    for number, robust in enumerate([False, True]):
        attempt = {'iteration': number, 'method': 'soft_l1_robust_refit' if robust else 'ordinary_least_squares'}
        try:
            # Important boundary: scan and fitting method only; no reference/spec.
            ring = fit_ring(scan, robust=robust)
            step = export_ring(ring, out/f'iteration_{number}')
            checks = verify_sizes(ring.dimensions(), reviewed)
            attempt.update(parameters=asdict(ring), drawing_verification=checks,
                           evidence_coverage_passed=ring.explained_fraction >= .95,
                           reconstructed_step=str(step))
            attempt['accepted'] = checks['passed'] and attempt['evidence_coverage_passed']
            # Evaluation is recorded after candidate creation, never used to tune its dimensions.
            attempt['held_out_step_geometry'] = verify_ring_steps(source/'reference.step', step)
            attempt['failed_dimensions'] = [c['dimension'] for c in checks['checks'] if not c['passed']]
        except UnsupportedScan as exc:
            attempt.update(accepted=False, error=str(exc))
        report['iterations'].append(attempt)
        if attempt['accepted']:
            report.update(accepted=True, selected_step=str(step))
            break
    report['outcome'] = ('accepted_after_refit' if report['accepted'] and len(report['iterations']) > 1
                         else 'accepted_initial_fit' if report['accepted'] else 'rejected_after_bounded_refit')
    return save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default='outputs/nist/ftc11')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--points', type=int, default=30000)
    p.add_argument('--noise-mm', type=float, default=.01)
    p.add_argument('--outliers', type=float, default=.02)
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--live-drawing', action='store_true')
    p.add_argument('--drawing-response', help='Replay a saved response; with --live-drawing, resume it and allow one corrective call')
    p.add_argument('--config', default='config/settings.yaml')
    a = p.parse_args()
    r = run(a.output_dir, seed=a.seed, n_points=a.points, noise_mm=a.noise_mm,
            outliers=a.outliers, dropout=a.dropout, live_drawing=a.live_drawing,
            drawing_response=a.drawing_response, config_path=a.config)
    print(f'{r.get("outcome", r.get("blocked_reason"))}; {a.output_dir}/report.json')
    if not r['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
