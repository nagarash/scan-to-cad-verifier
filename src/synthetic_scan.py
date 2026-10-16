"""Sample arbitrary STEP surfaces with explicit, reproducible synthetic corruption."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sample_step(step_path, output_dir, n_points=30000, noise_sigma_mm=.01,
                outlier_fraction=0., dropout_fraction=0., seed=42,
                linear_tolerance_mm=.02, angular_tolerance_rad=.1):
    import cadquery as cq
    import trimesh
    if (type(n_points) is not int or n_points < 500
            or not np.isfinite([noise_sigma_mm, outlier_fraction, dropout_fraction]).all()
            or noise_sigma_mm < 0 or not 0 <= outlier_fraction < .2 or not 0 <= dropout_fraction < .9):
        raise ValueError('Invalid point count, noise, outlier fraction, or dropout fraction')
    if (not np.isfinite([linear_tolerance_mm, angular_tolerance_rad]).all()
            or linear_tolerance_mm <= 0 or not 0 < angular_tolerance_rad <= np.pi):
        raise ValueError('Invalid tessellation tolerances')
    source = Path(step_path)
    shape = cq.importers.importStep(str(source))
    if len(shape.solids().vals()) != 1 or not shape.val().isValid():
        raise ValueError('Expected one valid STEP solid')
    # Exact area-weighted sampling is available for this known analytic family.
    # Other STEP geometries fall back to an explicitly recorded tessellation.
    from src.ring import read_ring_step, sample_ring_surface
    try:
        solid, ring = read_ring_step(source)
    except ValueError:
        ring = None
    if ring is not None:
        reference, area = sample_ring_surface(ring, n_points, seed)
        if not np.isclose(area, solid.Area(), rtol=1e-6, atol=1e-6):
            raise ValueError('Unexpected ring surface trimming')
        method = 'exact area-weighted analytic ring surfaces from STEP'
        linear, angular = None, None
    else:
        linear, angular = linear_tolerance_mm, angular_tolerance_rad
        vertices, faces = shape.val().tessellate(linear, angular)
        mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in vertices], faces=faces, process=False)
        reference, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
        method = 'area-weighted surface sampling of STEP tessellation'
    rng = np.random.default_rng(seed)
    keep = rng.random(n_points) >= dropout_fraction
    scan = reference[keep].copy()
    scan += rng.normal(0, noise_sigma_mm, scan.shape)
    count = int(round(len(scan)*outlier_fraction))
    if count:
        box = shape.val().BoundingBox()
        low = np.array([box.xmin, box.ymin, box.zmin])
        high = np.array([box.xmax, box.ymax, box.zmax])
        margin = (high-low)*.1
        indices = rng.choice(len(scan), count, replace=False)
        scan[indices] = rng.uniform(low-margin, high+margin, (count, 3))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'scan_points_mm.npy', scan)
    np.save(out/'reference_points_mm.npy', reference)
    metadata = {'kind': 'synthetic full-surface samples, not a physical scanner simulation',
                'source_step_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'units': 'mm', 'unit_handling': 'OpenCascade STEP importer converts source units to mm',
                'requested_points': n_points, 'retained_points': len(scan), 'seed': seed,
                'noise_sigma_mm': noise_sigma_mm, 'outlier_fraction': outlier_fraction,
                'outlier_count': count, 'dropout_fraction': dropout_fraction,
                'sampling_method': method,
                'linear_tessellation_tolerance_mm': linear, 'angular_tessellation_tolerance_rad': angular,
                'limitations': ['Uniform surface sampling; no ray visibility, occlusion, or scanner calibration',
                                'Random dropout is not a model of missing views',
                                'Outliers replace random samples inside an expanded bounding box']}
    (out/'scan_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return scan, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step')
    p.add_argument('--output-dir', default='outputs/synthetic_scan')
    p.add_argument('--points', type=int, default=30000)
    p.add_argument('--noise-mm', type=float, default=.01)
    p.add_argument('--outliers', type=float, default=0.)
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    _, meta = sample_step(a.step, a.output_dir, a.points, a.noise_mm, a.outliers, a.dropout, a.seed)
    print(f'Wrote {meta["retained_points"]} synthetic scan points to {a.output_dir}')


if __name__ == '__main__':
    main()
