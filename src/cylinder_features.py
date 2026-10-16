"""Scan-only discovery of Z-aligned cylindrical patches, not solid reconstruction.

Uses local PCA normals and circle RANSAC. No CAD, nominal dimensions, drawing,
feature count, or part-specific region is accepted by the fitter.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares


@dataclass
class CylinderPatch:
    center_xy_mm: list[float]
    radius_mm: float
    z_low_mm: float
    z_high_mm: float
    support_points: int
    radial_rms_mm: float
    angular_bins: int


def discover_cylinders(points, tolerance_mm=.15, seed=42, max_features=20):
    p = np.asarray(points, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 500 or not np.isfinite(p).all()
            or not np.isfinite(tolerance_mm) or tolerance_mm <= 0):
        raise ValueError('Expected finite Nx3 scan and positive tolerance')
    rng = np.random.default_rng(seed)
    # Local covariance normals reject horizontal planar samples and most outliers.
    _, indices = cKDTree(p).query(p, k=20, workers=-1)
    neighbors = p[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
    normals = vectors[:, :, 0]
    eligible = (np.abs(normals[:, 2]) < .18) & (values[:, 0] < .08*values[:, 1])
    remaining = np.flatnonzero(eligible)
    features = []
    for _ in range(max_features):
        if len(remaining) < 100:
            break
        trial = rng.choice(remaining, min(8000, len(remaining)), replace=False)
        xy, ns = p[trial, :2], normals[trial, :2]
        best, best_score = None, 0
        # Normals point along the radius; two normal lines propose a circle center.
        for _ in range(450):
            a, b = rng.choice(len(trial), 2, replace=False)
            matrix = np.column_stack((ns[a], -ns[b]))
            if abs(np.linalg.det(matrix)) < .2:
                continue
            lengths = np.linalg.solve(matrix, xy[b]-xy[a])
            center = xy[a]+lengths[0]*ns[a]
            radius = np.linalg.norm(xy[a]-center)
            if radius < 2*tolerance_mm:
                continue
            delta = xy-center
            r = np.linalg.norm(delta, axis=1)
            mask = (abs(r-radius) < tolerance_mm) & (abs(np.sum(delta*ns, axis=1))/np.maximum(r, 1e-9) > .96)
            score = int(mask.sum())
            if score > best_score:
                best, best_score = (center, radius), score
        if best is None or best_score < 55:
            break
        center, radius = best
        def membership(c, r):
            delta = p[remaining, :2]-c
            rr = np.linalg.norm(delta, axis=1)
            return (abs(rr-r) < tolerance_mm) & (abs(np.sum(delta*normals[remaining, :2], axis=1))/np.maximum(rr, 1e-9) > .96)
        for _ in range(3):
            ids = remaining[membership(center, radius)]
            fit = least_squares(lambda v: np.linalg.norm(p[ids, :2]-v[:2], axis=1)-v[2],
                                [*center, radius], loss='soft_l1', f_scale=tolerance_mm/3)
            center, radius = fit.x[:2], fit.x[2]
        mask = membership(center, radius)
        ids = remaining[mask]
        remaining = remaining[~mask]
        angles = np.arctan2(p[ids, 1]-center[1], p[ids, 0]-center[0])
        bins = len(np.unique(np.floor((angles+np.pi)/(2*np.pi)*24).astype(int).clip(0,23)))
        lo, hi = np.quantile(p[ids, 2], [.01, .99])
        # A patch needs circumferential and axial support; endpoints remain estimates.
        if bins < 18 or hi-lo < 5*tolerance_mm or len(ids) < 100:
            continue
        residual = np.linalg.norm(p[ids, :2]-center, axis=1)-radius
        features.append(CylinderPatch(center.tolist(), float(radius), float(lo), float(hi),
                                      len(ids), float(np.sqrt(np.mean(residual**2))), bins))
    return features


def export_patches(features, path):
    """Export independent cylindrical faces. Deliberately does not imply a solid."""
    import cadquery as cq
    faces = []
    for f in features:
        cylinder = cq.Solid.makeCylinder(f.radius_mm, f.z_high_mm-f.z_low_mm,
                                       cq.Vector(*f.center_xy_mm, f.z_low_mm))
        faces.extend(face for face in cylinder.Faces() if face.geomType() == 'CYLINDER')
    if not faces:
        raise ValueError('No supported cylindrical patches')
    cq.exporters.export(cq.Compound.makeCompound(faces), str(path))
