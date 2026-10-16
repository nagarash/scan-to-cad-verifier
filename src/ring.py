"""Scan-only fitting of an aligned annulus with two raised circular-section rims.

This constrained family matches NIST FTC-11 topology. No drawing dimensions,
reference STEP faces, or sampling labels are accepted by the fitting functions.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.reconstruct import UnsupportedScan


@dataclass
class Ring:
    center_xy_mm: list[float]
    inner_diameter_mm: float
    outer_diameter_mm: float
    bottom_z_mm: float
    shoulder_z_mm: float
    web_top_z_mm: float
    rim_radius_mm: float
    median_surface_residual_mm: float = 0.
    explained_fraction: float = 0.

    def dimensions(self):
        return {'outer_diameter_mm': self.outer_diameter_mm,
                'inner_diameter_mm': self.inner_diameter_mm,
                'rim_radius_mm': self.rim_radius_mm,
                'web_thickness_mm': self.web_top_z_mm-self.bottom_z_mm}


def _planes(z, tolerance):
    low, high = np.quantile(z, [.001, .999])
    bins = max(20, int(np.ceil((high-low)/tolerance)))
    if bins > 20000 or high-low < tolerance*8:
        raise UnsupportedScan('Insufficient extent or unsupported scan scale')
    counts, edges = np.histogram(z, bins=bins+4, range=(low-2*tolerance, high+2*tolerance))
    smoothed = gaussian_filter1d(counts.astype(float), 1)
    peaks, _ = find_peaks(smoothed)
    selected = []
    for index in sorted(peaks, key=lambda i: smoothed[i], reverse=True):
        value = (edges[index]+edges[index+1])/2
        if all(abs(value-other) > 4*tolerance for other in selected):
            for _ in range(3):
                support = z[np.abs(z-value) < 2*tolerance]
                value = float(np.median(support))
            if len(support) < .08*len(z):
                continue
            selected.append(value)
        if len(selected) == 2:
            return sorted(selected)
    raise UnsupportedScan('Two well-supported parallel annular planes are required')


def _profile(ring):
    inner, outer, radius = ring.inner_diameter_mm/2, ring.outer_diameter_mm/2, ring.rim_radius_mm
    height = ring.web_top_z_mm-ring.shoulder_z_mm
    if (not 0 < inner < outer or radius <= 0 or not 0 < height < radius
            or ring.bottom_z_mm >= ring.shoulder_z_mm or 2*radius >= outer-inner):
        raise UnsupportedScan('Inconsistent annular rim profile')
    chord = np.sqrt(radius**2-height**2)
    if inner+radius+chord >= outer-radius-chord:
        raise UnsupportedScan('Raised rims overlap; no supported central web remains')
    return inner, outer, radius, height, chord


def surface_distances(points, ring):
    inner, outer, radius, height, chord = _profile(ring)
    radial = np.linalg.norm(points[:, :2]-ring.center_xy_mm, axis=1)
    z = points[:, 2]
    def segment_plane(level, lo, hi):
        return np.hypot(z-level, np.maximum(lo-radial, 0)+np.maximum(radial-hi, 0))
    outside_z = np.maximum(ring.bottom_z_mm-z, 0)+np.maximum(z-ring.shoulder_z_mm, 0)
    distances = [segment_plane(ring.bottom_z_mm, inner, outer),
                 segment_plane(ring.web_top_z_mm, inner+radius+chord, outer-radius-chord),
                 np.hypot(radial-inner, outside_z), np.hypot(radial-outer, outside_z)]
    for center, start, end in [(outer-radius, 0., np.pi-np.arcsin(height/radius)),
                                (inner+radius, np.arcsin(height/radius), np.pi)]:
        angle = np.clip(np.arctan2(z-ring.shoulder_z_mm, radial-center), start, end)
        distances.append(np.hypot(radial-center-radius*np.cos(angle),
                                   z-ring.shoulder_z_mm-radius*np.sin(angle)))
    return np.min(distances, axis=0)


def fit_ring(points, robust=False, plane_tolerance_mm=.05):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 1000 or not np.isfinite(p).all():
        raise ValueError('Expected at least 1000 finite Nx3 scan points')
    if not np.isfinite(plane_tolerance_mm) or plane_tolerance_mm <= 0:
        raise ValueError('Plane tolerance must be positive and finite')
    bottom, web = _planes(p[:, 2], plane_tolerance_mm)
    low_z, high_z = np.quantile(p[:, 2], [.01, .99])
    span = high_z-low_z
    walls = p[(p[:, 2] > bottom+.15*span) & (p[:, 2] < web-.15*span)]
    if len(walls) < 100:
        raise UnsupportedScan('Insufficient lower cylindrical-wall evidence')
    low, high = np.quantile(p[:, :2], [.005, .995], axis=0)
    center = (low+high)/2
    r = np.linalg.norm(walls[:, :2]-center, axis=1)
    split = np.mean(np.quantile(r, [.1, .9]))
    groups = r > split
    if min(np.sum(groups), np.sum(~groups)) < 40:
        raise UnsupportedScan('Two distinct cylinder walls are required')
    loss = 'soft_l1' if robust else 'linear'
    fitted = least_squares(lambda v: np.linalg.norm(walls[:, :2]-v[:2], axis=1)
                           - np.where(groups, v[3], v[2]),
                           [*center, np.median(r[~groups]), np.median(r[groups])],
                           loss=loss, f_scale=.015, max_nfev=150)
    cx, cy, inner, outer = fitted.x
    if not fitted.success or not 0 < inner < outer:
        raise UnsupportedScan('Cylinder fitting did not converge')
    for group in [groups, ~groups]:
        angles = np.arctan2(walls[group, 1]-cy, walls[group, 0]-cx)
        if len(np.unique(np.floor((angles+np.pi)/(2*np.pi)*24).astype(int) % 24)) < 23:
            raise UnsupportedScan('Full circumferential wall coverage is required')
    radial = np.linalg.norm(p[:, :2]-[cx, cy], axis=1)
    upper = p[:, 2] > web+2*plane_tolerance_mm
    if np.sum(upper) < 100:
        raise UnsupportedScan('Insufficient raised-rim evidence')
    rr, zz = radial[upper], p[upper, 2]
    is_outer = rr > (inner+outer)/2
    def rim_residual(v):
        radius, shoulder = v
        centers = np.where(is_outer, outer-radius, inner+radius)
        return np.hypot(rr-centers, zz-shoulder)-radius
    initial_radius = min(span/2, (outer-inner)/3)
    bead = least_squares(rim_residual, [initial_radius, high_z-initial_radius],
                         bounds=([plane_tolerance_mm, bottom+plane_tolerance_mm],
                                 [(outer-inner)/2-.01, web-.001]),
                         loss=loss, f_scale=.015, max_nfev=150)
    if not bead.success:
        raise UnsupportedScan('Rim fitting did not converge')
    result = Ring([float(cx), float(cy)], 2*float(inner), 2*float(outer), bottom,
                  float(bead.x[1]), web, float(bead.x[0]))
    distances = surface_distances(p, result)
    result.median_surface_residual_mm = float(np.median(distances))
    result.explained_fraction = float(np.mean(distances <= plane_tolerance_mm))
    return result


def build_ring(ring):
    import cadquery as cq
    inner, outer, radius, height, chord = _profile(ring)
    shoulder, web, bottom = ring.shoulder_z_mm, ring.web_top_z_mm, ring.bottom_z_mm
    profile = (cq.Workplane('XZ').moveTo(inner, bottom).lineTo(outer, bottom).lineTo(outer, shoulder)
               .threePointArc((outer-radius, shoulder+radius), (outer-radius-chord, web))
               .lineTo(inner+radius+chord, web)
               .threePointArc((inner+radius, shoulder+radius), (inner, shoulder)).close())
    return profile.revolve(360, (0, 0), (0, 1)).translate((*ring.center_xy_mm, 0))


def export_ring(ring, output_dir):
    import cadquery as cq
    part = build_ring(ring)
    if len(part.solids().vals()) != 1 or not part.val().isValid():
        raise UnsupportedScan('Expected one valid reconstructed solid')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(part, str(out/'reconstructed.step'))
    cq.exporters.export(part, str(out/'reconstructed.stl'))
    (out/'parameters.json').write_text(json.dumps(asdict(ring), indent=2)+'\n')
    return out/'reconstructed.step'


def read_ring_step(path):
    """Independent extraction of the exported CAD's analytic surface parameters."""
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    solids = cq.importers.importStep(str(path)).solids().vals()
    if len(solids) != 1 or not solids[0].isValid():
        raise ValueError('Expected one valid ring STEP solid')
    solid = solids[0]
    cylinders, planes, tori = [], [], []
    centers = []
    for face in solid.Faces():
        adaptor = BRepAdaptor_Surface(face.wrapped)
        kind = face.geomType()
        if kind == 'CYLINDER':
            c = adaptor.Cylinder()
            if abs(c.Axis().Direction().Z()) < 1-1e-7:
                raise ValueError('Ring STEP must be aligned with Z')
            cylinders.append(c.Radius())
            centers.append([c.Location().X(), c.Location().Y()])
        elif kind == 'PLANE':
            if abs(face.normalAt().z) < 1-1e-7:
                raise ValueError('Ring planes must be normal to Z')
            planes.append(face.Center().z)
        elif kind == 'TORUS':
            t = adaptor.Torus()
            if abs(t.Axis().Direction().Z()) < 1-1e-7:
                raise ValueError('Ring tori must be aligned with Z')
            tori.append((t.MajorRadius(), t.MinorRadius(), t.Location().Z()))
            centers.append([t.Location().X(), t.Location().Y()])
        else:
            raise ValueError(f'Unsupported ring STEP face: {kind}')
    if list(map(len, [cylinders, planes, tori])) != [2, 2, 2]:
        raise ValueError('Expected two cylinders, two planes, and two tori')
    if not np.allclose(centers, centers[0], atol=1e-6, rtol=0):
        raise ValueError('Ring STEP surfaces must be coaxial')
    if not np.isclose(tori[0][1], tori[1][1], atol=1e-6, rtol=0) or not np.isclose(tori[0][2], tori[1][2], atol=1e-6, rtol=0):
        raise ValueError('Ring rims must share radius and shoulder height')
    inner, outer = sorted(cylinders)
    bottom, web = sorted(planes)
    radius, shoulder = tori[0][1:]
    if not np.allclose(sorted(t[0] for t in tori), [inner+radius, outer-radius], atol=1e-6, rtol=0):
        raise ValueError('Unexpected torus locations')
    ring = Ring(centers[0], 2*inner, 2*outer, bottom, shoulder, web, radius)
    _profile(ring)
    return solid, ring


def verify_ring_steps(reference_path, predicted_path, n_samples=6000, seed=101):
    """Compare CAD features and sample-to-analytic-surface distances in both directions.

    Avoid Boolean overlap for these nearly coincident trimmed tori: OCCT can
    produce zero intersections or impossible IoU > 1 at different tolerances.
    Sampling uses area-weighted analytic surfaces; the reported maximum is not a
    full Hausdorff bound. This also avoids costly trimmed-torus tessellation.
    """
    if type(n_samples) is not int or n_samples < 100:
        raise ValueError('Use at least 100 evaluation samples')
    reference, ref = read_ring_step(reference_path)
    predicted, pred = read_ring_step(predicted_path)
    distances = []
    for shape, source, target, offset in [(reference, ref, pred, 0), (predicted, pred, ref, 1)]:
        points, area = sample_ring_surface(source, n_samples, seed+offset)
        if not np.isclose(area, shape.Area(), rtol=1e-6, atol=1e-6):
            raise ValueError('STEP surface trimming does not match the supported ring family')
        distances.append(surface_distances(points, target))
    all_distances = np.concatenate(distances)
    errors = {name: abs(ref.dimensions()[name]-pred.dimensions()[name]) for name in ref.dimensions()}
    errors.update(center_xy_mm=float(np.linalg.norm(np.array(ref.center_xy_mm)-pred.center_xy_mm)),
                  bottom_z_mm=abs(ref.bottom_z_mm-pred.bottom_z_mm),
                  shoulder_z_mm=abs(ref.shoulder_z_mm-pred.shoulder_z_mm))
    return {'reference_cad_dimensions_mm': ref.dimensions(), 'candidate_cad_dimensions_mm': pred.dimensions(),
            'cad_feature_absolute_errors_mm': errors,
            'symmetric_mean_surface_distance_mm': float(np.mean(all_distances)),
            'sampled_p95_surface_distance_mm': float(np.percentile(all_distances, 95)),
            'sampled_max_surface_distance_mm': float(np.max(all_distances)),
            'samples_per_direction': n_samples, 'evaluation_seed': seed,
            'surface_sampling': 'Area-weighted analytic surfaces extracted from each STEP; no evaluation tessellation',
            'note': 'Independent STEP analytic-feature and sampled-surface comparison; no Boolean IoU or full GD&T claim'}


def sample_ring_surface(ring, count, seed):
    """Area-uniform samples on the supported analytic CAD surfaces, for evaluation."""
    inner, outer, radius, height, chord = _profile(ring)
    rng = np.random.default_rng(seed)
    low_web, high_web = inner+radius+chord, outer-radius-chord
    inner_start, outer_end = np.arcsin(height/radius), np.pi-np.arcsin(height/radius)
    def torus_area(center, start, end):
        return 2*np.pi*radius*(center*(end-start)+radius*(np.sin(end)-np.sin(start)))
    areas = np.array([np.pi*(outer**2-inner**2), np.pi*(high_web**2-low_web**2),
                      2*np.pi*inner*(ring.shoulder_z_mm-ring.bottom_z_mm),
                      2*np.pi*outer*(ring.shoulder_z_mm-ring.bottom_z_mm),
                      torus_area(outer-radius, 0, outer_end),
                      torus_area(inner+radius, inner_start, np.pi)])
    choices = rng.choice(6, count, p=areas/areas.sum())
    radial, z = np.empty(count), np.empty(count)
    for kind in range(6):
        selected = np.flatnonzero(choices == kind)
        n = len(selected)
        if kind in (0, 1):
            lo, hi, level = ((inner, outer, ring.bottom_z_mm) if kind == 0
                              else (low_web, high_web, ring.web_top_z_mm))
            radial[selected] = np.sqrt(rng.uniform(lo**2, hi**2, n))
            z[selected] = level
        elif kind in (2, 3):
            radial[selected] = inner if kind == 2 else outer
            z[selected] = rng.uniform(ring.bottom_z_mm, ring.shoulder_z_mm, n)
        else:
            center, start, end = ((outer-radius, 0, outer_end) if kind == 4
                                  else (inner+radius, inner_start, np.pi))
            angles = []
            while sum(len(a) for a in angles) < n:
                proposed = rng.uniform(start, end, max(n*2, 10))
                accept = rng.random(len(proposed)) < (center+radius*np.cos(proposed))/(center+radius)
                angles.append(proposed[accept])
            theta = np.concatenate(angles)[:n] if n else np.array([])
            radial[selected] = center+radius*np.cos(theta)
            z[selected] = ring.shoulder_z_mm+radius*np.sin(theta)
    azimuth = rng.uniform(-np.pi, np.pi, count)
    points = np.column_stack((ring.center_xy_mm[0]+radial*np.cos(azimuth),
                              ring.center_xy_mm[1]+radial*np.sin(azimuth), z))
    return points, float(areas.sum())
