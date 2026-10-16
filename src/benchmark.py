"""Reproducible deterministic reconstruction benchmark, with held-out STEP verification."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.dataset import build_part
from src.reconstruct import UnsupportedScan, reconstruct
from src.verification import verify_steps


def run(output_dir='outputs/plate_benchmark', seeds=(42, 43, 44), noise_mm=.05, n_points=12000):
    import cadquery as cq
    import trimesh

    if not seeds or n_points < 500 or not np.isfinite(noise_mm) or noise_mm < 0:
        raise ValueError('Supply seeds, at least 500 points, and finite nonnegative noise')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ('four_hole_plate', 'two_hole_plate'):
        for seed in seeds:
            folder = root/f'{name}_seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            reference = build_part(name)
            cq.exporters.export(reference, str(folder/'reference.step'))
            cq.exporters.export(reference, str(folder/'reference.stl'))
            mesh = trimesh.load_mesh(folder/'reference.stl', process=False)
            points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
            points += np.random.default_rng(seed).normal(0, noise_mm, points.shape)
            scan = folder/'scan_points_mm.npy'
            np.save(scan, points)
            result = {'part': name, 'seed': seed, 'noise_sigma_mm': noise_mm,
                      'point_count': n_points, 'scan': str(scan)}
            started = time.perf_counter()
            try:
                reconstruction = reconstruct(scan, folder)
            except UnsupportedScan as exc:
                result.update(passed=False, error=str(exc))
            else:
                result['reconstruction_seconds'] = time.perf_counter()-started
                result['parameters'] = reconstruction['parameters']
                result['verification'] = verify_steps(folder/'reference.step', folder/'reconstructed.step')
                result['passed'] = result['verification']['passed']
                (folder/'verification.json').write_text(json.dumps(result['verification'], indent=2)+'\n')
            results.append(result)
    report = {'method': 'deterministic reconstruction; no AI',
              'scope': 'Complete aligned rectangular plates with circular through-holes; synthetic scans only',
              'environment': {'python': platform.python_version(),
                              **{name: version(name) for name in ('numpy', 'scipy', 'cadquery', 'cadquery-ocp', 'trimesh')}},
              'fit_settings': {'distance_tolerance_mm': .2, 'cluster_radius_mm': 1.5},
              'passed': all(r['passed'] for r in results),
              'passed_cases': sum(r['passed'] for r in results), 'total_cases': len(results),
              'results': results}
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/plate_benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--noise-mm', type=float, default=.05)
    parser.add_argument('--points', type=int, default=12000)
    args = parser.parse_args()
    report = run(args.output_dir, args.seeds, args.noise_mm, args.points)
    print(f'{report["passed_cases"]}/{report["total_cases"]} cases passed; {args.output_dir}/report.json')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
