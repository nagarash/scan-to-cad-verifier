"""Reproducible baseline and optional AI annotation; no false reconstruction claims."""
import argparse,json
from pathlib import Path
import numpy as np
from src.dataset import PARTS
from src.features import summarize_points
from src.geometry import evaluate_points

def run(data_dir='data/generated',output='outputs/ablation.json'):
    root=Path(data_dir); results=[]
    for name in PARTS:
        folder=root/name
        scan=folder/'scan_points_mm.npy'; ref=folder/'reference_points_mm.npy'
        metrics=evaluate_points(ref,scan)
        results.append({'part':name,'baseline':'scan vs ground-truth sampled surface (NOT reconstructed CAD)',
                        'geometry_metrics':metrics,'observations':summarize_points(np.load(scan)),
                        'ai_interpretation':None,'ai_improved_cad':None})
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'status':'baseline only; AI CAD generation and ablation not yet implemented','results':results},indent=2))
    print(f'Wrote {target}')
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data/generated');p.add_argument('--output',default='outputs/ablation.json');a=p.parse_args();run(a.data_dir,a.output)
