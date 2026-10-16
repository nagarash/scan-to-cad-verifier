"""Evaluate exported, aligned plate STEP solids against independent ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_plate_step(path):
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    part = cq.importers.importStep(str(path))
    solids = part.solids().vals()
    if len(solids) != 1 or not solids[0].isValid():
        raise ValueError('Expected one valid STEP solid')
    solid = solids[0]
    box = solid.BoundingBox()
    holes = []
    planes = 0
    for face in solid.Faces():
        kind = face.geomType()
        if kind == 'PLANE':
            normal = np.abs(face.normalAt().toTuple())
            if not np.isclose(normal.max(), 1., atol=1e-7):
                raise ValueError('STEP verification requires XYZ-aligned plate faces')
            planes += 1
        elif kind == 'CYLINDER':
            cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
            axis = cylinder.Axis()
            if abs(axis.Direction().Z()) < 1-1e-7:
                raise ValueError('Expected Z-aligned hole cylinders')
            radius = cylinder.Radius()
            bounds = face.BoundingBox()
            if (abs(bounds.zmin-box.zmin) > 1e-5 or abs(bounds.zmax-box.zmax) > 1e-5
                    or not np.isclose(face.Area(), 2*np.pi*radius*box.zlen, rtol=1e-5)):
                raise ValueError('Expected complete circular through-hole faces')
            holes.append({'center_xy_mm': [axis.Location().X(), axis.Location().Y()],
                          'diameter_mm': 2*radius})
        else:
            raise ValueError(f'Unsupported STEP surface: {kind}')
    if planes != 6:
        raise ValueError('Expected six outer planar plate faces')
    holes.sort(key=lambda h: tuple(h['center_xy_mm']))
    return solid, {'extents_mm': [box.xlen, box.ylen, box.zlen],
                   'center_xy_mm': [(box.xmin+box.xmax)/2, (box.ymin+box.ymax)/2],
                   'bottom_z_mm': box.zmin, 'holes': holes, 'volume_mm3': solid.Volume()}


def verify_steps(reference_path, predicted_path, dimension_tolerance_mm=.1,
                 max_volume_difference_fraction=.01, hole_match_distance_mm=1.):
    """Read both STEP files anew; compare dimensions, features, and Boolean volume.

    Hole correspondence uses a gated global assignment with dummy unmatched
    entries. A solid-volume difference measures geometry, not manufacturing
    qualification. No ground-truth parameters enter the reconstruction.
    """
    settings = [dimension_tolerance_mm, max_volume_difference_fraction, hole_match_distance_mm]
    if not np.isfinite(settings).all() or min(settings) <= 0:
        raise ValueError('Verification thresholds must be positive and finite')
    reference, ref = read_plate_step(reference_path)
    predicted, pred = read_plate_step(predicted_path)
    extent_errors = np.abs(np.array(ref['extents_mm'])-pred['extents_mm'])
    position_error = float(np.linalg.norm(np.array(ref['center_xy_mm'])-pred['center_xy_mm']))
    bottom_error = abs(ref['bottom_z_mm']-pred['bottom_z_mm'])
    a, b = ref['holes'], pred['holes']
    matches = []
    if a and b:
        distances = np.linalg.norm(np.array([h['center_xy_mm'] for h in a])[:, None, :]
                                   - np.array([h['center_xy_mm'] for h in b])[None, :, :], axis=2)
        # Dummy rows/columns allow unmatched features without sacrificing valid pairs.
        cost = np.full((len(a)+len(b), len(a)+len(b)), hole_match_distance_mm*2)
        cost[:len(a), :len(b)] = np.where(distances <= hole_match_distance_mm, distances, 1e9)
        cost[len(a):, len(b):] = 0
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows, columns):
            if i < len(a) and j < len(b) and distances[i, j] <= hole_match_distance_mm:
                matches.append({'reference_index': int(i), 'predicted_index': int(j),
                                'center_error_mm': float(distances[i, j]),
                                'diameter_error_mm': abs(a[i]['diameter_mm']-b[j]['diameter_mm'])})
    tp = len(matches)
    precision = tp/len(b) if b else (1. if not a else 0.)
    recall = tp/len(a) if a else 1.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    missing_volume = reference.cut(predicted).Volume()
    excess_volume = predicted.cut(reference).Volume()
    difference = (missing_volume+excess_volume)/ref['volume_mm3']
    iou = reference.intersect(predicted).Volume() / reference.fuse(predicted).Volume()
    max_error = max([*extent_errors, position_error, bottom_error,
                     *[m[key] for m in matches for key in ('center_error_mm', 'diameter_error_mm')]])
    return {'reference': ref, 'reconstructed': pred,
            'extent_absolute_errors_mm': extent_errors.tolist(),
            'plate_center_error_mm': position_error, 'bottom_z_error_mm': bottom_error,
            'hole_matches': matches, 'hole_precision': precision, 'hole_recall': recall, 'hole_f1': f1,
            'missing_holes': len(a)-tp, 'extra_holes': len(b)-tp,
            'missing_volume_mm3': missing_volume, 'excess_volume_mm3': excess_volume,
            'symmetric_volume_difference_fraction': difference, 'solid_iou': iou,
            'thresholds': {'dimension_tolerance_mm': dimension_tolerance_mm,
                           'max_volume_difference_fraction': max_volume_difference_fraction,
                           'hole_match_distance_mm': hole_match_distance_mm},
            'passed': bool(tp == len(a) == len(b) and max_error <= dimension_tolerance_mm
                           and difference <= max_volume_difference_fraction)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('predicted')
    parser.add_argument('--output', default='outputs/verification.json')
    args = parser.parse_args()
    report = verify_steps(args.reference, args.predicted)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {path}; passed={report["passed"]}')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
