"""Direct fixed-scan -> CAD-Recode v1.5 -> generated CadQuery -> held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def prepare_points(points,seed=42):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<256 or not np.isfinite(p).all():
        raise ValueError('Expected at least 256 finite XYZ points')
    low,high=p.min(axis=0),p.max(axis=0)
    center=(low+high)/2;scale=float(np.max(high-low)/2)
    if scale<=1e-9:raise ValueError('Degenerate scan')
    normalized=(p-center)/scale
    rng=np.random.default_rng(seed)
    candidates=normalized[rng.choice(len(p),min(8192,len(p)),replace=False)]
    distances=np.full(len(candidates),np.inf);ids=[];current=0
    for _ in range(256):
        ids.append(current)
        distances=np.minimum(distances,np.sum((candidates-candidates[current])**2,axis=1))
        distances[ids]=-1
        current=int(np.argmax(distances))
    return candidates[ids].astype(np.float32),{'center_mm':center.tolist(),'scale_mm':scale,
        'model_output_coordinate_scale':100,'input_count':len(p),'model_point_count':256,
        'candidate_point_count':len(candidates),'seed':seed,'normalization':'bbox center, longest extent maps to [-1,1]'}


def infer(points,model_dir,tokenizer_dir,device='auto',max_tokens=768):
    import torch
    from transformers import AutoTokenizer
    from third_party.cad_recode.model import CADRecode
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype=torch.float32 if device=='cpu' else torch.float16
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,
                                            pad_token='<|im_end|>',padding_side='left')
    model=CADRecode.from_pretrained(model_dir,torch_dtype=dtype,
                                   attn_implementation='eager',local_files_only=True).eval().to(device)
    # Keep the Fourier projection in FP32, as in the upstream architecture.
    model.point_encoder.float()
    input_ids=[tokenizer.pad_token_id]*len(points)+[tokenizer('<|im_start|>')['input_ids'][0]]
    attention=[-1]*len(points)+[1]
    started=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(input_ids=torch.tensor([input_ids],device=device),
                attention_mask=torch.tensor([attention],device=device),
                point_cloud=torch.tensor(points[None],device=device),max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,do_sample=False)
    new=output[0,len(input_ids):]
    raw=tokenizer.decode(new,skip_special_tokens=False)
    code=raw.split('<|endoftext|>')[0].split('<|im_end|>')[0].strip()
    return code,{'device':device,'dtype':str(dtype),'latency_seconds':time.perf_counter()-started,
                 'generated_tokens':len(new),'max_new_tokens':max_tokens,'raw_generation':raw,
                 'finished_with_eos':int(new[-1]) in (tokenizer.eos_token_id,151643,151645) if len(new) else False,
                 'torch_version':torch.__version__}


def run(scan_path,output_dir,model_dir='outputs/models/cad-recode-v1.5',tokenizer_dir='outputs/models/qwen2-tokenizer',
        device='auto',reference=None,drawing=None,program_file=None):
    from src.cad_program import execute_program
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scan=np.load(scan_path,allow_pickle=False);points,transform=prepare_points(scan)
    np.save(out/'model_points_normalized.npy',points)
    (out/'transform.json').write_text(json.dumps(transform,indent=2)+'\n')
    report={'backend':'CAD-Recode v1.5','mode':'program_replay' if program_file else 'local_model',
            'scan_sha256':hashlib.sha256(Path(scan_path).read_bytes()).hexdigest(),
            'model_provenance':{},
            'transform':transform,'reference_used_for_generation':False,'accepted':False}
    for path in [Path(model_dir),Path(tokenizer_dir)]:
        metadata_path=path.parent/(path.name+'-revision.json')
        if metadata_path.exists():report['model_provenance'][path.name]=json.loads(metadata_path.read_text())
    try:
        if program_file:code=Path(program_file).read_text()
        else:
            code,metadata=infer(points,model_dir,tokenizer_dir,device)
            report['inference']=metadata
            (out/'generation.json').write_text(json.dumps(metadata,indent=2)+'\n')
            if not metadata['finished_with_eos']:
                (out/'generated.py').write_text(code+'\n');raise ValueError('Generation reached token limit without EOS')
        (out/'generated.py').write_text(code+'\n')
        report['execution']=execute_program(code,transform,out)
        report['outcome']='candidate_generated'
        if reference:
            from src.cad_evaluation import evaluate
            spec=json.loads(Path(drawing).read_text()) if drawing else {'checks':[]}
            report['evaluation']=evaluate(report['execution']['candidate_step'],reference,spec,out/'evaluation.json')
            report['accepted']=report['evaluation']['accepted']
            report['outcome']='accepted' if report['accepted'] else 'candidate_rejected'
    except Exception as exc:
        report.update(outcome='failed',error_type=type(exc).__name__,error=str(exc)[:500])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',default='outputs/reverse_engineering/ctc05/input/scan_points_mm.npy')
    p.add_argument('--output-dir',default='outputs/reverse_engineering/ctc05/cad_recode')
    p.add_argument('--model-dir',default='outputs/models/cad-recode-v1.5')
    p.add_argument('--tokenizer-dir',default='outputs/models/qwen2-tokenizer')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--reference',default='data/nist/ctc05/reference.step')
    p.add_argument('--drawing',default='data/nist/ctc05/drawing_checks.json')
    p.add_argument('--program-file',help='Offline execution replay; no model')
    a=p.parse_args();r=run(a.scan,a.output_dir,a.model_dir,a.tokenizer_dir,a.device,a.reference,a.drawing,a.program_file)
    print(r['outcome'],r.get('error',''))
    if r['outcome']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
