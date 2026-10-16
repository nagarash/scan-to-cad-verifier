"""Synthetic bracket ground truth and point sampling. CADQuery required."""
from pathlib import Path
import numpy as np


def make_bracket(output_dir='outputs', n_points=10000, noise_sigma_mm=0.0, seed=42):
    import cadquery as cq
    import trimesh
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    # 60 x 40 x 6 mm plate, 4 through holes (diameter 6 mm)
    part = (cq.Workplane('XY').box(60, 40, 6, centered=(True, True, False))
            .faces('>Z').workplane().pushPoints([(-20,-10),(-20,10),(20,-10),(20,10)]).hole(6))
    cq.exporters.export(part, str(out/'ground_truth.step'))
    cq.exporters.export(part, str(out/'ground_truth.stl'))
    mesh = trimesh.load_mesh(out/'ground_truth.stl', process=False)
    rng = np.random.default_rng(seed)
    # Surface sampling; deterministic RNG for reproducible baseline
    points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    if noise_sigma_mm:
        points += rng.normal(0, noise_sigma_mm, points.shape)
    np.save(out/'scan_points_mm.npy', points)
    return out


def evaluate_points(reference_path, predicted_path):
    """Symmetric nearest-neighbor surface discrepancy (not CAD feature accuracy)."""
    from scipy.spatial import cKDTree
    a, b = np.load(reference_path), np.load(predicted_path)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3 or not len(a) or not len(b):
        raise ValueError('Expected nonempty Nx3 point clouds')
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    return {'symmetric_chamfer_mm': float((d_ab.mean()+d_ba.mean())/2),
            'p95_nearest_neighbor_mm':float(np.percentile(np.r_[d_ab,d_ba],95)),
            'hausdorff_sampled_mm':float(max(d_ab.max(),d_ba.max()))}
