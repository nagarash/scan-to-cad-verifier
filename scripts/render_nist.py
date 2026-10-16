"""Render recorded NIST loop evidence without any model calls."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.ring import Ring, read_ring_step


def profile(ring):
    inner, outer, radius = ring.inner_diameter_mm/2, ring.outer_diameter_mm/2, ring.rim_radius_mm
    angle = np.arcsin((ring.web_top_z_mm-ring.shoulder_z_mm)/radius)
    outer_angles = np.linspace(0, np.pi-angle, 200)
    inner_angles = np.linspace(angle, np.pi, 200)
    xy = [(inner, ring.bottom_z_mm), (outer, ring.bottom_z_mm), (outer, ring.shoulder_z_mm)]
    xy.extend(zip(outer-radius+radius*np.cos(outer_angles), ring.shoulder_z_mm+radius*np.sin(outer_angles)))
    xy.extend(zip(inner+radius+radius*np.cos(inner_angles), ring.shoulder_z_mm+radius*np.sin(inner_angles)))
    xy.append((inner, ring.bottom_z_mm))
    return np.array(xy)


def main():
    paths = ['outputs/nist/ftc11/report.json', 'outputs/nist/ftc11_seed43/report.json',
             'outputs/nist/ftc11_seed44/report.json']
    reports = [json.loads((ROOT/p).read_text()) for p in paths]
    report = reports[0]
    first, last = report['iterations']
    before, after = Ring(**first['parameters']), Ring(**last['parameters'])
    _, reference = read_ring_step(ROOT/'data/nist/ftc11/reference.step')
    points = np.load(ROOT/'outputs/nist/ftc11/scan/scan_points_mm.npy')
    radial = np.linalg.norm(points[:, :2]-after.center_xy_mm, axis=1)
    fig, (ax, checks) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.4, 1]})
    fig.patch.set_facecolor('white')
    ink, muted, orange, green = '#15243b', '#65758b', '#d97732', '#0d9488'
    ax.scatter(radial, points[:, 2], color='#9ba9b9', s=.5, alpha=.35, rasterized=True)
    for ring, color, label, linestyle in [(before, orange, 'Initial least squares', '-'),
                                          (after, green, 'Robust refit', '-'),
                                          (reference, ink, 'Held-out STEP', '--')]:
        xy = profile(ring)
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.7, label=label, linestyle=linestyle)
    ax.set_xlim(14, 34)
    ax.set_ylim(-2, 2.2)
    ax.set_xlabel('Radius from fitted axis (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Cross-section: fit improves against held-out CAD', loc='left', color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='lower center', ncol=3)
    labels = ['Outside diameter', 'Inside diameter', 'Rim radius', 'Web thickness']
    for attempt, offset, color, label in [(first, -.1, orange, 'Initial'), (last, .1, green, 'Refit')]:
        ratios = [c['absolute_error_mm']/c['tolerance_mm'] for c in attempt['drawing_verification']['checks']]
        checks.scatter(ratios, np.arange(4)+offset, color=color, label=label, s=50, zorder=3)
    checks.axvline(1., color='#ba4741', linestyle='--', linewidth=1)
    checks.set_xscale('log')
    checks.set_xlim(.001, 15)
    checks.set_yticks(range(4), labels)
    checks.invert_yaxis()
    checks.set_xlabel('Absolute error / drawing tolerance (log scale)')
    checks.set_title('Below 1 = selected size check passes', loc='left', fontsize=11, color=ink)
    checks.grid(axis='x', color='#e2e8f0', zorder=0)
    checks.legend(frameon=False, fontsize=9)
    for panel in (ax, checks):
        panel.spines[['top', 'right']].set_visible(False)
        panel.spines[['left', 'bottom']].set_color('#cbd5e1')
        panel.tick_params(colors=muted)
    fig.text(.06, .95, 'NIST FTC-11: verify → refit → recheck', fontsize=19, color=ink, weight='bold')
    fig.text(.06, .88, '30,000 synthetic samples · Gaussian noise σ = 0.01 mm · 2% outliers · seed 42', color=muted, fontsize=11)
    fig.text(.06, .035, 'Four selected size checks only. Deterministic correction uses reviewed drawing values for pass/fail, never for fitting.',
             fontsize=9, color=muted)
    fig.subplots_adjust(left=.06, right=.98, top=.76, bottom=.18, wspace=.45)
    assets = ROOT/'docs/assets'
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets/'nist-loop.png', dpi=150)
    plt.close(fig)
    live = json.loads((ROOT/'outputs/nist/ftc11_live_corrected/report.json').read_text())
    snapshot = {'source': report['source_provenance'], 'scope': report['scope'],
                'drawing_spec_source': 'Visually reviewed transcription; not the rejected model output',
                'drawing_dimensions': report['drawing_spec']['dimensions'],
                'not_evaluated': report['drawing_spec']['not_evaluated'],
                'cases': [{'seed': r['scan']['seed'], 'scan': r['scan'], 'outcome': r['outcome'],
                            'iterations': [{k: v for k, v in a.items() if k != 'reconstructed_step'}
                                           for a in r['iterations']]} for r in reports],
                'live_drawing_attempts': [{'model': a['response']['call']['returned_model'],
                                           'proposal': a['response']['proposal'], 'review': a['review'],
                                           'latency_seconds': a['response']['call']['latency_seconds'],
                                           'reported_cost_usd': a['response']['call']['usage'].get('cost')}
                                          for a in live['drawing_attempts']]}
    (assets/'nist-results.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print('Wrote docs/assets/nist-loop.png and docs/assets/nist-results.json')


if __name__ == '__main__':
    main()
