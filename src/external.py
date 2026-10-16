"""Register user-supplied STEP ground truth without asserting its license."""
import argparse,json,shutil
from pathlib import Path

def register(step_path,identifier,source_url,license_note,directory='data/external'):
    source=Path(step_path)
    if not source.is_file() or source.suffix.lower() not in ('.step','.stp'): raise ValueError('Supply an existing .step or .stp file')
    if not identifier.replace('_','').replace('-','').isalnum(): raise ValueError('Use an alphanumeric ID')
    dest=Path(directory)/identifier;dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest/'reference.step')
    (dest/'metadata.json').write_text(json.dumps({'id':identifier,'source_url':source_url,'license_note':license_note,'license_verified':False,'units':'verify manually'},indent=2))
    print(f'Registered {dest}; license and dimensions must be verified independently')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step');p.add_argument('--id',required=True);p.add_argument('--source-url',required=True);p.add_argument('--license-note',required=True);a=p.parse_args();register(a.step,a.id,a.source_url,a.license_note)
