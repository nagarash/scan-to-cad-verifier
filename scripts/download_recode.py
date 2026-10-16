"""Download pinned public inference weights; no scan data leaves the machine."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]/'outputs/models'
ASSETS=[('filapro/cad-recode-v1.5','765e8cc315a1a77bd8c69ccd0b403bad20ce35e8','cad-recode-v1.5',
         ['config.json','generation_config.json','*.safetensors','README.md']),
        ('Qwen/Qwen2-1.5B','8a16abf2848eda07cc5253dec660bf1ce007ad7a','qwen2-tokenizer',
         ['tokenizer*','vocab.json','merges.txt','config.json'])]
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    for repo,revision,name,patterns in ASSETS:
        snapshot_download(repo,revision=revision,local_dir=ROOT/name,allow_patterns=patterns)
        (ROOT/(name+'-revision.json')).write_text(json.dumps({'repo':repo,'revision':revision},indent=2)+'\n')
        print(repo,revision)
