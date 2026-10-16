"""Separated scan-only reconstruction and held-out evaluation commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


def reconstruct(scan_path,output_dir,tolerance_mm=.15):
    # No ground-truth imports or paths in this reconstruction boundary.
    from src.cylinder_features import discover_cylinders,export_patches
    scan=np.load(scan_path,allow_pickle=False)
    features=discover_cylinders(scan,tolerance_mm=tolerance_mm)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    candidate=out/'candidate.step'
    if features:export_patches(features,candidate)
    report={'input_scan':str(scan_path),'point_count':len(scan),'features':[asdict(f) for f in features],
            'candidate_step':str(candidate) if features else None,'complete_solid':False,
            'reference_access_required':False,'drawing_access_required':False,
            'method':'scan-only local normals + cylinder RANSAC',
            'limitation':'Open cylinder-face baseline; missing topology remains unresolved'}
    (out/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('reconstruct');r.add_argument('--scan',required=True);r.add_argument('--output-dir',required=True)
    r.add_argument('--backend',choices=['cad-recode','cylinder'],default='cad-recode')
    r.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    r.add_argument('--model-dir',default='outputs/models/cad-recode-v1.5')
    r.add_argument('--tokenizer-dir',default='outputs/models/qwen2-tokenizer')
    r.add_argument('--tolerance-mm',type=float,default=.15)
    e=sub.add_parser('evaluate');e.add_argument('--candidate',required=True)
    e.add_argument('--reference',default='data/nist/ctc05/reference.step')
    e.add_argument('--drawing-checks',default='data/nist/ctc05/drawing_checks.json')
    e.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='reconstruct':
        if a.backend=='cad-recode':
            from src.cad_recode import run
            result=run(a.scan,a.output_dir,a.model_dir,a.tokenizer_dir,a.device)
            print(result['outcome'],result.get('error',''))
            if result['outcome']=='failed':raise SystemExit(1)
        else:
            result=reconstruct(a.scan,a.output_dir,a.tolerance_mm)
            print(f"Recovered {len(result['features'])} cylindrical patches; complete solid: False")
    else:
        from src.cad_evaluation import evaluate
        result=evaluate(a.candidate,a.reference,json.loads(Path(a.drawing_checks).read_text()),a.output)
        print(f"Matched {len(result['cylinder_comparison']['matches'])} cylindrical features; accepted: {result['accepted']}")
        for reason in result['rejection_reasons']:print(reason)

if __name__=='__main__':main()
