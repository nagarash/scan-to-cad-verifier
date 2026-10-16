"""CTC-05 scan and cylindrical-feature benchmark; incomplete reconstruction rejected."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from src.cylinder_features import discover_cylinders, export_patches
from src.nist_experiment import verify_sources
from src.scan_views import render_scan
from src.synthetic_scan import sample_step

SOURCE = Path(__file__).resolve().parents[1]/'data/nist/ctc05'


def reference_cylinders(path):
    """Held-out evaluation only. Deduplicate split cylinder faces."""
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    solid = cq.importers.importStep(str(path)).val()
    cylinders = {}
    for face in solid.Faces():
        if face.geomType() != 'CYLINDER':
            continue
        c = BRepAdaptor_Surface(face.wrapped).Cylinder()
        if abs(c.Axis().Direction().Z()) < .999:
            continue
        x,y,_ = c.Location().Coord()
        key = tuple(round(v, 5) for v in (x,y,c.Radius()))
        cylinders[key] = {'center_xy_mm':[x,y], 'radius_mm':c.Radius()}
    return list(cylinders.values()), len(solid.Faces())


def evaluate(features, reference):
    results = []
    for feature in features:
        target = min(reference, key=lambda r: np.linalg.norm(np.array(r['center_xy_mm'])-feature.center_xy_mm)+abs(r['radius_mm']-feature.radius_mm))
        results.append({'measured':asdict(feature), 'held_out_reference':target,
                        'diameter_error_mm':2*(feature.radius_mm-target['radius_mm']),
                        'center_error_mm':float(np.linalg.norm(np.array(target['center_xy_mm'])-feature.center_xy_mm))})
    return results


def run(output_dir='outputs/nist/ctc05', seeds=(42,43,44), n_points=120000):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    provenance = verify_sources(SOURCE)
    spec = json.loads((SOURCE/'drawing_checks.json').read_text())
    runs = []
    for linear, angular, label in [(.02,.1,'coarse'),(.002,.03,'fine')]:
        for seed in seeds:
            directory = out/f'{label}_seed{seed}'
            scan, metadata = sample_step(SOURCE/'reference.step', directory/'scan', n_points,
                                         .03,.02,0.,seed,linear,angular)
            # Critical boundary: only scan + fixed algorithm settings enter discovery.
            patches = discover_cylinders(scan, seed=seed)
            if patches:
                export_patches(patches, directory/'cylindrical_patches.step')
            # Ground truth is opened only after candidate extraction/export.
            reference, faces = reference_cylinders(SOURCE/'reference.step')
            comparison = evaluate(patches, reference)
            # Feature association here uses held-out CAD, so this is not autonomous
            # drawing-to-scan correspondence. The fitter never sees these values.
            bore = [x for x in comparison if abs(x['held_out_reference']['radius_mm']-127) < 1e-5
                    and np.linalg.norm(x['held_out_reference']['center_xy_mm']) < 1e-5]
            check = spec['checks'][0]
            diameter = 2*bore[0]['measured']['radius_mm'] if len(bore)==1 else None
            passed = diameter is not None and abs(diameter-check['nominal_mm']) <= check['tolerance_mm']
            record = {'resolution':label,'seed':seed,'scan':metadata,'features':comparison,
                      'bore_check':{'measured_diameter_mm':diameter,'nominal_mm':check['nominal_mm'],
                                    'tolerance_mm':check['tolerance_mm'],'passed':passed,
                                    'association':'held-out CAD-assisted feature association'},
                      'full_reconstruction_accepted':False}
            (directory/'report.json').write_text(json.dumps(record,indent=2)+'\n')
            runs.append(record)
            if seed == seeds[0] and label == 'fine':
                render_scan(scan, out/'scan_views.png')
    report = {'part_id':provenance['id'],'provenance':provenance,'drawing_spec':spec,
              'reference_face_count':faces,'runs':runs,'full_reconstruction_accepted':False,
              'outcome':'partial_cylindrical_feature_recovery_only',
              'measurement_uses_reference_or_drawing':False,
              'resolution_sweep':'Simulation fidelity audit, not an autonomous reconstruction correction',
              'limitations':['Aligned Z cylinders only; no general pose recovery',
                'Independent open cylinder faces exported, not a closed solid',
                'Axial bounds are sample quantiles, not fitted shoulder dimensions',
                'Holes, flats, counterbores, blends, cavity closure, and GD&T not fully reconstructed',
                'Drawing feature association uses held-out CAD; no autonomous correspondence claim',
                'No AI interpretation or repair is executed by this benchmark',
                'Surface sampling includes hidden surfaces; no physical scanner occlusion model']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/nist/ctc05')
    a=p.parse_args(); r=run(a.output_dir)
    for x in r['runs']:
        print(x['resolution'],x['seed'],len(x['features']),x['bore_check'])
    print('Full reconstruction REJECTED: partial cylindrical patches only')

if __name__=='__main__':
    main()
