import json
import numpy as np
import pytest
from src.cad_recode import prepare_points,run
from src.cad_program import validate_program,execute_program


def test_normalization_fps_reproducible_and_inverse():
    rng=np.random.default_rng(3)
    p=rng.uniform([10,-20,5],[50,60,15],(1000,3))
    a,t=prepare_points(p);b,u=prepare_points(p)
    assert a.shape==(256,3) and np.array_equal(a,b) and t==u
    recovered=a*t['scale_mm']+t['center_mm']
    from scipy.spatial import cKDTree
    assert cKDTree(p).query(recovered)[0].max()<1e-5
    assert abs(a).max()<=1.000001
    assert len(np.unique(a,axis=0))==256


@pytest.mark.parametrize('points',[np.zeros((255,3)),np.zeros((500,3)),np.full((500,3),np.nan)])
def test_invalid_cloud(points):
    with pytest.raises(ValueError):prepare_points(points)


@pytest.mark.parametrize('code',[
 "import os\nr=os.system('pwd')", "r=open('/tmp/a')", "r=cq.importers.importStep('reference.step')",
 "r=cq.Workplane().__class__", "while True: pass", "r=getattr(cq,'importers')", "x=cq.Workplane()"])
def test_unsafe_or_unsupported_program_rejected(code):
    with pytest.raises((ValueError,SyntaxError)):validate_program(code)


def test_generated_program_restores_mm_scale_and_position(tmp_path):
    import cadquery as cq
    code="import cadquery as cq\nr=cq.Workplane('XY').box(100,100,100)"
    execution=execute_program(code,{'scale_mm':40,'center_mm':[12,-5,7]},tmp_path)
    shape=cq.importers.importStep(execution['candidate_step']).val()
    box=shape.BoundingBox()
    assert np.allclose([box.xlen,box.ylen,box.zlen],[40,40,40])
    assert np.allclose(shape.Center().toTuple(),[12,-5,7])
    assert execution['valid'] and execution['solid_count']==1


def test_program_replay_separate_from_live_model(tmp_path):
    scan=tmp_path/'scan.npy';code=tmp_path/'program.py'
    rng=np.random.default_rng(3);np.save(scan,rng.uniform(-20,20,(500,3)))
    code.write_text("import cadquery as cq\nr=cq.Workplane('XY').box(100,100,100)")
    report=run(scan,tmp_path/'out',program_file=code)
    assert report['mode']=='program_replay' and report['outcome']=='candidate_generated'
    assert 'inference' not in report
    assert not report['reference_used_for_generation']


def test_point_encoder_cpu_and_cached_generation():
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from third_party.cad_recode.model import CADRecode
    from transformers import Qwen2Config
    previous=torch.get_default_dtype()
    model=CADRecode(Qwen2Config(vocab_size=100,hidden_size=32,intermediate_size=64,
                               num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)).eval()
    assert torch.get_default_dtype()==previous
    mask=torch.tensor([[-1,-1,1]])
    with torch.inference_mode():
        output=model.generate(input_ids=torch.tensor([[1,1,2]]),attention_mask=mask,
                     point_cloud=torch.randn(1,2,3),max_new_tokens=3,pad_token_id=1,do_sample=False)
    assert output.shape==(1,6)
    assert torch.equal(mask,torch.tensor([[-1,-1,1]]))
