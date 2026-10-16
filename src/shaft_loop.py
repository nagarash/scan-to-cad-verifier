"""One live-model strategy decision and one scan-fitted repair, held-out evaluation."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict


class RepairDecision(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    action:Literal['build_scan_fitted_shaft_envelope','abstain']
    rationale:str
    uncertainties:list[str]


def decide(image,measured,failed_checks,config_path):
    from dotenv import load_dotenv
    from openai import OpenAI
    import yaml
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    config=yaml.safe_load(Path(config_path).read_text())['model']
    key=os.getenv(config.get('api_key_env','OPENROUTER_API_KEY'))
    if not key:raise RuntimeError('Model key missing')
    prompt=('Choose one bounded reconstruction repair for a synthetic scan generated from public NIST geometry. '
            'The initial candidate consists of open cylindrical faces. The only implemented repair '
            'build_scan_fitted_shaft_envelope fits closed flange/hub/shaft cylinders, a rear bore and '
            'an approximate rectangular upper body using measured scan cylinders and planar levels. '
            'It does not recover mounting holes, blends or tapered body geometry. A subset of four coaxial cylinders is sufficient; off-axis holes do not invalidate that subset. Choose it only if '
            'the scan supports a coaxial shaft/flange family. Otherwise abstain. This action is an '
            'approximate hypothesis, not a claim of full reconstruction. Do not invent dimensions. '
            'Return schema JSON. Reference CAD and drawing are unavailable to you.')
    payload={'measured_cylinders':measured,'failed_checks':failed_checks}
    encoded=base64.b64encode(Path(image).read_bytes()).decode('ascii')
    started=time.perf_counter()
    with OpenAI(api_key=key,base_url=config['base_url'],timeout=90,max_retries=0) as client:
        response=client.chat.completions.create(model=config['name'],temperature=0,max_tokens=1000,
          response_format={'type':'json_schema','json_schema':{'name':'shaft_repair','strict':True,'schema':RepairDecision.model_json_schema()}},
          messages=[{'role':'system','content':prompt},{'role':'user','content':[
           {'type':'text','text':json.dumps(payload)},
           {'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}}]}])
    if not response.choices or response.choices[0].finish_reason!='stop':raise ValueError('Incomplete model response')
    raw=response.choices[0].message.content
    decision=RepairDecision.model_validate_json(raw)
    return {'decision':decision.model_dump(),'raw_response':raw,
            'call':{'requested_model':config['name'],'returned_model':response.model,
                    'response_id':response.id,'latency_seconds':time.perf_counter()-started,
                    'usage':response.usage.model_dump() if response.usage else None,
                    'prompt':prompt,'payload':payload,'image_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest()}}


def run(scan_path,image_path,output_dir,reference,drawing,config_path='config/settings.yaml',decision_file=None,previous_decision=None):
    from src.reverse_engineering import reconstruct
    from src.cad_evaluation import evaluate
    from src.cylinder_features import CylinderPatch
    from src.shaft_candidate import build
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scan_hash=hashlib.sha256(Path(scan_path).read_bytes()).hexdigest()
    initial=reconstruct(scan_path,out/'attempt_0')
    spec=json.loads(Path(drawing).read_text())
    before=evaluate(initial['candidate_step'],reference,spec,out/'attempt_0/evaluation.json')
    report={'scan_sha256':scan_hash,'reference_access':'held-out evaluator only',
            'maximum_repair_attempts':1,'initial':before,'repair':None,'final':before,
            'model_role':'bounded strategy selection; numeric fitting and CAD construction deterministic',
            'accepted':False,'mode':'decision_replay' if decision_file else 'live_model'}
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
    try:
        if decision_file:response=json.loads(Path(decision_file).read_text())
        else:
            evidence=initial['features']
            if previous_decision:
                prior=json.loads(Path(previous_decision).read_text())
                if RepairDecision.model_validate(prior['decision']).action!='abstain':
                    raise ValueError('Clarification only applies to prior abstention')
                report['prior_model_decision']=prior
                evidence={'coaxial_group_measured_from_scan':[f for f in evidence if np.linalg.norm(f['center_xy_mm'])<.2],
                          'remaining_off_axis_patches':[f for f in evidence if np.linalg.norm(f['center_xy_mm'])>=.2]}
            response=decide(image_path,evidence,before['rejection_reasons'],config_path)
        decision=RepairDecision.model_validate(response['decision'])
        (out/'repair_decision.json').write_text(json.dumps(response,indent=2)+'\n')
        report['model']=response
    except Exception as exc:
        report['outcome']='model_failed_baseline_retained';report['error_type']=type(exc).__name__;return save()
    if decision.action=='abstain':
        report['outcome']='model_abstained';return save()
    try:
        # No reference path, drawing, or nominal dimensions passed into repair.
        proposal=build(np.load(scan_path,allow_pickle=False),[CylinderPatch(**f) for f in initial['features']],out/'attempt_1')
        (out/'attempt_1/reconstruction.json').write_text(json.dumps(proposal,indent=2)+'\n')
        after=evaluate(proposal['candidate_step'],reference,spec,out/'attempt_1/evaluation.json')
        report.update(repair=proposal,final=after,accepted=after['accepted'])
        report['outcome']='accepted_after_repair' if after['accepted'] else 'repaired_candidate_rejected'
    except Exception as exc:
        report['outcome']='repair_failed_baseline_retained';report['error_type']=type(exc).__name__
    if hashlib.sha256(Path(scan_path).read_bytes()).hexdigest()!=scan_hash:raise RuntimeError('Scan changed')
    return save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',default='outputs/reverse_engineering/ctc05/input/scan_points_mm.npy')
    p.add_argument('--image',default='outputs/reverse_engineering/ctc05/scan_views.png')
    p.add_argument('--output-dir',default='outputs/reverse_engineering/ctc05/live_loop')
    p.add_argument('--reference',default='data/nist/ctc05/reference.step')
    p.add_argument('--drawing',default='data/nist/ctc05/drawing_checks.json')
    p.add_argument('--config',default='config/settings.yaml')
    p.add_argument('--decision-file',help='Offline replay; no live call')
    p.add_argument('--previous-decision',help='Clarify one saved abstention with scan-only coaxial grouping')
    a=p.parse_args();r=run(a.scan,a.image,a.output_dir,a.reference,a.drawing,a.config,a.decision_file,a.previous_decision)
    print(r['outcome'])

if __name__=='__main__':main()
