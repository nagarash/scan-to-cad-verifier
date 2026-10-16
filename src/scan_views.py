"""Render anonymous scan-only views for the vision model, without reference CAD."""
import os
from pathlib import Path

import numpy as np


def render_scan(points, output_path):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 500 or not np.isfinite(p).all():
        raise ValueError('Expected at least 500 finite Nx3 points')
    # Keep Matplotlib's cache local; avoid importing a GUI backend in tests/CLI.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    for pos, axes, title in [(1, (0, 1), 'Top: X–Y'), (3, (0, 2), 'Side: X–Z'),
                             (4, (1, 2), 'Side: Y–Z')]:
        ax = fig.add_subplot(2, 2, pos)
        ax.scatter(p[:, axes[0]], p[:, axes[1]], s=.6, color='#164e63', linewidths=0, rasterized=True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('XYZ'[axes[0]])
        ax.set_ylabel('XYZ'[axes[1]])
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(*p.T, s=.4, color='#164e63', linewidths=0, depthshade=True)
    ax.set_box_aspect(np.maximum(np.ptp(p, axis=0), 1e-6))
    ax.view_init(elev=25, azim=-55)
    ax.set_title('Oblique: X–Y–Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.suptitle('Views of one scanned part — point samples only')
    fig.tight_layout()
    try:
        fig.savefig(target, dpi=140)
    finally:
        plt.close(fig)
    return target
