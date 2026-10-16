"""Deterministic reconstruction of aligned, fully scanned plates with round holes.

Input is only an Nx3 point cloud in millimeters. No reference CAD, nominal
dimensions, hole count, or part ID is used during fitting. This is a constrained
baseline, not a general segmentation or design-intent recovery algorithm.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class UnsupportedScan(ValueError):
    """The scan does not provide sufficient support for this model family."""


@dataclass
class Hole:
    center_xy_mm: list[float]
    diameter_mm: float
    radial_rms_mm: float
    support_points: int


@dataclass
class Plate:
    center_xy_mm: list[float]
    bottom_z_mm: float
    length_mm: float
    width_mm: float
    thickness_mm: float
    holes: list[Hole]
    fit_rms_mm: float
    explained_fraction: float
    distance_tolerance_mm: float


def _plane_pair(values, tolerance):
    """Find dense outer planar modes instead of noisy bounding-box extrema."""
    low, high = np.quantile(values, [0.001, 0.999])
    if high - low < 8 * tolerance:
        raise UnsupportedScan('Insufficient thickness or extent for the fitting tolerance')
    estimates = []
    for lo, hi in [(low, low + .2 * (high-low)),
                   (high - .2 * (high-low), high)]:
        bins = max(4, int(np.ceil((hi-lo) / tolerance)))
        counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
        idx = np.argmax(counts)
        peak = (edges[idx] + edges[idx+1]) / 2
        for _ in range(4):
            support = values[np.abs(values-peak) <= tolerance]
            peak = float(np.median(support))
        if len(support) < max(25, .005 * len(values)):
            raise UnsupportedScan('Outer plane has insufficient point support')
        estimates.append(peak)
    return estimates


def _clusters(points, radius):
    tree = cKDTree(points)
    remaining = set(range(len(points)))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = [start], [start]
        while pending:
            for index in tree.query_ball_point(points[pending.pop()], radius):
                if index in remaining:
                    remaining.remove(index)
                    component.append(index)
                    pending.append(index)
        yield np.asarray(component)


def fit_plate(points, tolerance_mm=.2, cluster_radius_mm=1.5):
    """Fit six axis-aligned planes and circular, Z-directed through-holes.

    Requires full surface coverage, Gaussian-like noise substantially below the
    tolerance, separated holes, and no appreciable rotation, wear, or outliers.
    Tolerance and clustering radius are explicit algorithm settings, not measured
    manufacturing tolerances. Unsupported or insufficient evidence raises.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 500 or not np.isfinite(p).all():
        raise ValueError('Expected at least 500 finite Nx3 points in millimeters')
    if not np.isfinite([tolerance_mm, cluster_radius_mm]).all() or min(tolerance_mm, cluster_radius_mm) <= 0:
        raise ValueError('Fitting tolerance and cluster radius must be positive and finite')
    bounds = np.array([_plane_pair(p[:, axis], tolerance_mm) for axis in range(3)])
    extents = bounds[:, 1] - bounds[:, 0]
    if extents[2] >= min(extents[:2]) / 2:
        raise UnsupportedScan('Expected a thin plate with thickness along Z')
    distances = np.min(np.abs(p[:, :, None] - bounds[None, :, :]), axis=2)
    outside = np.maximum(bounds[:, 0]-p, 0) + np.maximum(p-bounds[:, 1], 0)
    bounded_planes = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        bounded_planes.append(np.sqrt(distances[:, axis]**2 + np.sum(outside[:, other]**2, axis=1)))
    planar_distance = np.min(bounded_planes, axis=0)
    # Exclude surface intersections where top-face points can contaminate circles.
    inside = np.all(p > bounds[:, 0] + 1.5*tolerance_mm, axis=1)
    inside &= np.all(p < bounds[:, 1] - 1.5*tolerance_mm, axis=1)
    candidates = p[inside]
    if len(candidates) < 30:
        raise UnsupportedScan('No supported circular through-hole walls found')
    holes = []
    for indices in _clusters(candidates[:, :2], cluster_radius_mm):
        wall = candidates[indices]
        if len(wall) < 30:
            continue
        xy = wall[:, :2]
        # Algebraic initialization followed by geometric radial least squares.
        design = np.column_stack((2*xy, np.ones(len(xy))))
        solution = np.linalg.lstsq(design, np.sum(xy*xy, axis=1), rcond=None)[0]
        center = solution[:2]
        radius = np.sqrt(max(0., solution[2] + center @ center))
        fit = least_squares(lambda v: np.linalg.norm(xy-v[:2], axis=1)-v[2],
                            [*center, radius])
        center, radius = fit.x[:2], float(fit.x[2])
        rms = float(np.sqrt(np.mean(fit.fun**2)))
        angles = np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0])
        coverage = len(np.unique(np.floor((angles+np.pi) / (2*np.pi) * 12).astype(int) % 12))
        if (radius <= 2*tolerance_mm or rms > tolerance_mm/2 or coverage < 11
                or np.ptp(wall[:, 2]) < .65*extents[2]
                or np.any(center-radius <= bounds[:2, 0])
                or np.any(center+radius >= bounds[:2, 1])):
            raise UnsupportedScan('Interior surface is not a sufficiently supported round through-hole')
        holes.append(Hole(center.tolist(), 2*radius, rms, len(wall)))
    if not holes:
        raise UnsupportedScan('No supported circular through-hole walls found')
    holes.sort(key=lambda h: tuple(h.center_xy_mm))
    for i, hole in enumerate(holes):
        for other in holes[i+1:]:
            if np.linalg.norm(np.array(hole.center_xy_mm)-other.center_xy_mm) <= (hole.diameter_mm+other.diameter_mm)/2:
                raise UnsupportedScan('Overlapping hole candidates')
    residual = planar_distance.copy()
    for hole in holes:
        radial = np.abs(np.linalg.norm(p[:, :2]-hole.center_xy_mm, axis=1)-hole.diameter_mm/2)
        residual = np.minimum(residual, np.hypot(radial, outside[:, 2]))
    fraction = float(np.mean(residual <= tolerance_mm))
    if fraction < .98:
        raise UnsupportedScan(f'Plate model explains only {fraction:.1%} of scan points')
    return Plate(bounds[:2].mean(axis=1).tolist(), float(bounds[2, 0]),
                 *map(float, extents), holes, float(np.sqrt(np.mean(residual**2))),
                 fraction, tolerance_mm)


def build_plate(plate):
    """Create editable box-and-hole CAD operations from measured parameters."""
    import cadquery as cq
    cx, cy = plate.center_xy_mm
    part = cq.Workplane('XY').box(plate.length_mm, plate.width_mm, plate.thickness_mm,
                                 centered=(True, True, False))
    for hole in plate.holes:
        # Workplane stays centered on the box rather than a changing face centroid.
        cutter = (cq.Workplane('XY').center(hole.center_xy_mm[0]-cx, hole.center_xy_mm[1]-cy)
                  .circle(hole.diameter_mm/2).extrude(plate.thickness_mm))
        part = part.cut(cutter)
    return part.translate((cx, cy, plate.bottom_z_mm))


def reconstruct(scan_path, output_dir, tolerance_mm=.2, cluster_radius_mm=1.5):
    import cadquery as cq
    plate = fit_plate(np.load(scan_path, allow_pickle=False), tolerance_mm, cluster_radius_mm)
    part = build_plate(plate)
    if len(part.solids().vals()) != 1 or not part.val().isValid():
        raise UnsupportedScan('Reconstruction did not produce one valid solid')
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(part, str(target/'reconstructed.step'))
    cq.exporters.export(part, str(target/'reconstructed.stl'))
    report = {'method': 'deterministic aligned rectangular plate with circular through-holes',
              'units': 'mm', 'parameters': asdict(plate),
              'limitations': ['Complete surface coverage and XYZ alignment required',
                              'No wear recovery, arbitrary pose, slots, bosses, or blind holes',
                              'Fit residual is not an uncertainty bound or tolerance certification']}
    (target/'reconstruction.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan', help='Nx3 .npy cloud in millimeters')
    parser.add_argument('--output-dir', default='outputs/reconstruction')
    parser.add_argument('--tolerance-mm', type=float, default=.2)
    parser.add_argument('--cluster-radius-mm', type=float, default=1.5)
    args = parser.parse_args()
    reconstruct(args.scan, args.output_dir, args.tolerance_mm, args.cluster_radius_mm)
    print(f'Wrote reconstruction to {args.output_dir}')


if __name__ == '__main__':
    main()
