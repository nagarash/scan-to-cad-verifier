"""Deterministic geometric observations, not exact analytic feature recognition."""
import numpy as np
from scipy.spatial import cKDTree

def summarize_points(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<10: raise ValueError('Expected Nx3 points')
    bounds=np.ptp(p,axis=0)
    center=np.mean(p,axis=0)
    cov=np.cov((p-center).T)
    eigenvalues=np.linalg.eigvalsh(cov)
    return {'point_count':int(len(p)),'axis_aligned_extents_mm':[float(x) for x in bounds],
            'centroid_mm':[float(x) for x in center], 'covariance_eigenvalues_mm2':[float(x) for x in eigenvalues],
            'warning':'Bounding boxes are scan-frame dependent; holes/planes not yet detected or measured.'}
