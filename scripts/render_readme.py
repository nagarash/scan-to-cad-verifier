"""Build README figures from existing local benchmark and live-model artifacts.

Run from the repository root: .venv/bin/python scripts/render_readme.py
Requires outputs/plate_benchmark and the three recorded live AI runs.
Makes no hosted model requests. Generated assets are kept outside outputs/.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

ASSETS = ROOT/'docs/assets'
NAVY, INK, MUTED = '#0b1426', '#15243b', '#65758b'
CYAN, BLUE, ORANGE = '#5de0cf', '#5d9fff', '#f5ae60'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})


def load(path):
    return json.loads((ROOT/path).read_text())


def solid_axis(fig, position):
    ax = fig.add_subplot(1, 3, position, projection='3d')
    ax.set_facecolor(NAVY)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-2, 10)
    ax.set_box_aspect((70, 50, 12), zoom=1.2)
    ax.set_proj_type('ortho')
    ax.view_init(elev=32, azim=-58)
    ax.set_axis_off()
    return ax


def outline(ax, parameters, color, linestyle, label):
    cx, cy = parameters['center_xy_mm']
    length, width, _ = parameters['extents_mm']
    ax.add_patch(Rectangle((cx-length/2, cy-width/2), length, width, fill=False,
                           edgecolor=color, linewidth=2, linestyle=linestyle, label=label))
    for hole in parameters['holes']:
        ax.add_patch(Circle(hole['center_xy_mm'], hole['diameter_mm']/2, fill=False,
                            edgecolor=color, linewidth=2, linestyle=linestyle))


def hero(benchmark):
    case = benchmark['results'][0]
    folder = ROOT/'outputs/plate_benchmark'/f'{case["part"]}_seed_{case["seed"]}'
    points = np.load(folder/'scan_points_mm.npy')
    verification = case['verification']
    fig = plt.figure(figsize=(15, 5.6), facecolor=NAVY)
    fig.text(.045, .91, 'SCAN → SOLID → EVIDENCE', color='white', fontsize=24, weight='bold')
    fig.text(.045, .845, 'A measured baseline for AI-assisted reverse engineering', color='#aabbd2', fontsize=12)
    ax = solid_axis(fig, 1)
    ax.scatter(*points.T, s=.9, color=CYAN, alpha=.9, linewidths=0)
    ax = fig.add_subplot(1, 3, 2)
    ax.set_facecolor(NAVY)
    # Exact top footprint from the independently reimported STEP measurements.
    # A 2D view avoids painter-order artifacts in triangulated CAD surfaces.
    measured = verification['reconstructed']
    cx, cy = measured['center_xy_mm']
    length, width, _ = measured['extents_mm']
    ax.add_patch(Rectangle((cx-length/2, cy-width/2), length, width,
                           facecolor=BLUE, edgecolor='#a8cbff', linewidth=1.5))
    for hole in measured['holes']:
        ax.add_patch(Circle(hole['center_xy_mm'], hole['diameter_mm']/2,
                            facecolor=NAVY, edgecolor='#a8cbff', linewidth=1.5))
    ax.set_xlim(-35, 35)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax = fig.add_subplot(1, 3, 3)
    ax.set_facecolor(NAVY)
    outline(ax, verification['reference'], CYAN, '-', 'Reference STEP')
    outline(ax, verification['reconstructed'], ORANGE, '--', 'Reconstructed STEP')
    ax.set_aspect('equal')
    ax.set_xlim(-35, 35)
    ax.set_ylim(-25, 25)
    ax.set_axis_off()
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.08), ncol=2,
              frameon=False, fontsize=8.5, labelcolor='white')
    for x, title, caption in [(.045, '01  Noisy scan', '12,000 points · noise σ = 0.05 mm'),
                              (.365, '02  Reconstructed CAD', 'STEP top view · measured box + circular cuts'),
                              (.695, '03  Independent verification', f'Solid IoU {verification["solid_iou"]:.5f} · outlines overlap')]:
        fig.text(x, .2, title, color='white', fontsize=13, weight='bold')
        fig.text(x, .145, caption, color='#aabbd2', fontsize=10)
    fig.text(.045, .045, 'Actual four-hole benchmark case · synthetic scan · aligned plate model · no AI dimension correction',
             color='#8197b5', fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.78, bottom=.27, wspace=.08)
    fig.savefig(ASSETS/'reconstruction.png', dpi=150, facecolor=NAVY)
    plt.close(fig)


def errors(benchmark):
    metrics = []
    for name in ('four_hole_plate', 'two_hole_plate'):
        cases = [c['verification'] for c in benchmark['results'] if c['part'] == name]
        metrics.append([max(max(v['extent_absolute_errors_mm']) for v in cases),
                        max(h['diameter_error_mm'] for v in cases for h in v['hole_matches']),
                        max(h['center_error_mm'] for v in cases for h in v['hole_matches'])])
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor='white')
    x = np.arange(3)
    for offset, row, label, color in [(-.19, metrics[0], 'Four-hole plate', '#2777c7'),
                                      (.19, metrics[1], 'Two-hole plate', '#0d9488')]:
        bars = ax.bar(x+offset, row, .34, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in row], padding=5, fontsize=10, color=INK)
    ax.set_xticks(x, ['Plate extents', 'Hole diameters', 'Hole centers'])
    ax.set_ylabel('Maximum absolute error (mm)', color=INK)
    ax.set_ylim(0, .016)
    ax.grid(axis='y', color='#e2e8f0', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cbd5e1')
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc='upper right')
    fig.text(.08, .94, 'Small errors, measured against independent STEP solids', fontsize=16, weight='bold', color=INK)
    fig.text(.08, .875, 'Maximum across three seeds per design · lower is better · benchmark acceptance limit: 0.1 mm',
             fontsize=10, color=MUTED)
    fig.text(.08, .035, 'Six synthetic scans · 12,000 points each · Gaussian noise σ = 0.05 mm · not manufacturing qualification',
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.79)
    fig.savefig(ASSETS/'benchmark.png', dpi=150)
    plt.close(fig)


def ai_checks(summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    names = ['four_hole_plate', 'two_hole_plate', 'slotted_plate']
    titles = ['Four-hole plate', 'Two-hole plate', 'Slotted plate']
    for ax, case, name, title in zip(axes, summary['cases'], names, titles):
        points = np.load(ROOT/f'data/generated/{name}/scan_points_mm.npy')
        ax.scatter(points[:, 0], points[:, 1], s=.7, color=INK, linewidths=0)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=13, weight='bold', pad=14)
    fig.text(.04, .94, 'Three live Qwen calls. Geometry checks every proposal.', fontsize=18, weight='bold', color=INK)
    labels = [('4 circular holes · both symmetries', 'ACCEPTED', '#0d9488'),
              ('2 circular holes · missed Y symmetry', 'ACCEPTED', '#0d9488'),
              ('Model incorrectly called this circular', 'GEOMETRY REJECTED', '#c05621')]
    for x, (description, status, color) in zip([.04, .365, .69], labels):
        fig.text(x, .22, description, color=MUTED, fontsize=10)
        fig.text(x, .15, status, color=color, fontsize=12, weight='bold')
    fig.text(.04, .045, 'Actual recorded interpretations · no dimensions supplied to the model · three examples do not establish general accuracy',
             color=MUTED, fontsize=9)
    fig.subplots_adjust(left=.03, right=.98, top=.78, bottom=.3, wspace=.15)
    fig.savefig(ASSETS/'ai-validation.png', dpi=150)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    benchmark = load('outputs/plate_benchmark/report.json')
    summary = load('outputs/ai_live_summary.json')
    hero(benchmark)
    errors(benchmark)
    ai_checks(summary)
    # Publish only selected evidence fields; never copy credentials or full local reports.
    snapshot = {'source': 'Recorded local synthetic benchmark and three live model calls',
                'benchmark': {k: benchmark[k] for k in ('environment', 'fit_settings', 'passed_cases', 'total_cases')},
                'benchmark_cases': [{'part': c['part'], 'seed': c['seed'], 'noise_sigma_mm': c['noise_sigma_mm'],
                                      'point_count': c['point_count'], 'verification': c['verification']}
                                     for c in benchmark['results']],
                'ai_model': summary['model'], 'ai_total_cost_usd': summary['total_reported_cost_usd'],
                'ai_cases': [{k: v for k, v in c.items() if k not in ('report', 'proposal')}
                             | {'proposal': {k: c['proposal'][k] for k in
                                            ('part_family', 'visible_hole_count', 'hole_shape', 'symmetry_axes')}}
                            for c in summary['cases']]}
    (ASSETS/'results.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print('Wrote three README figures and docs/assets/results.json')


if __name__ == '__main__':
    main()
