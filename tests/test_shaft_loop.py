import json
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError
from src.cylinder_features import CylinderPatch
from src.shaft_candidate import build
from src.shaft_loop import RepairDecision, run


def scan_fixture():
    rng=np.random.default_rng(41)
    planes=[np.column_stack((rng.uniform(-3,3,(1800,2)),np.full(1800,z)))
            for z in [0.,5.,20.,30.,50.,80.]]
    angle=rng.uniform(-np.pi,np.pi,2000)
    flange=np.column_stack((11*np.cos(angle),11*np.sin(angle),rng.uniform(0,5,2000)))
    body=np.column_stack((rng.uniform(-3,3,(2000,2)),rng.uniform(30,50,2000)))
    return np.concatenate([*planes,flange,body])


def features():
    return [CylinderPatch([0.,0.],r,lo,hi,1000,.002,24)
            for r,lo,hi in [(2,51,79),(4,.5,19),(7,6,29),(11,1,4)]]


def test_unseen_shaft_size_builds_valid_closed_cad(tmp_path):
    import cadquery as cq
    result=build(scan_fixture(),features(),tmp_path)
    solid=cq.importers.importStep(result['candidate_step']).val()
    assert solid.isValid() and len(solid.Solids())==1
    assert abs(solid.BoundingBox().zlen-80)<.001
    assert result['flange_top_mm']==5
    assert result['bore_end_mm']==20
    assert result['hub_top_mm']==30
    assert result['shaft_start_mm']==50
    assert result['unresolved']


def test_unsupported_topology_rejected(tmp_path):
    with pytest.raises(ValueError,match='four'):
        build(scan_fixture(),features()[:3],tmp_path)


@pytest.mark.parametrize('action',['execute_python','read_reference','shell'])
def test_model_cannot_select_arbitrary_execution(action):
    with pytest.raises(ValidationError):
        RepairDecision(action=action,rationale='x',uncertainties=[])


def test_bounded_loop_keeps_reference_out_of_model_and_repair(tmp_path,monkeypatch):
    import src.reverse_engineering as reconstruction
    import src.cad_evaluation as evaluator
    import src.shaft_candidate as builder
    import src.shaft_loop as loop
    scan=tmp_path/'scan.npy';np.save(scan,scan_fixture())
    spec=tmp_path/'drawing.json';spec.write_text('{"checks": []}')
    calls=[]
    monkeypatch.setattr(reconstruction,'reconstruct',lambda *a: {'candidate_step':'initial.step','features':[vars(f) for f in features()]})
    def evaluate(*args):
        calls.append(args)
        return {'accepted':False,'rejection_reasons':['Incomplete topology'],'private_reference_value':12345}
    monkeypatch.setattr(evaluator,'evaluate',evaluate)
    def decide(image,measured,failed,config):
        assert failed==['Incomplete topology']
        assert '12345' not in json.dumps(measured)
        return {'decision':{'action':'build_scan_fitted_shaft_envelope','rationale':'Scan supports this family','uncertainties':['Upper body unresolved']}}
    monkeypatch.setattr(loop,'decide',decide)
    repairs=[]
    def repair(points,measured,output):
        repairs.append((points,measured));Path(output).mkdir(parents=True)
        return {'candidate_step':'repaired.step'}
    monkeypatch.setattr(builder,'build',repair)
    report=run(scan,'image.png',tmp_path/'loop','held_out.step',spec)
    assert len(repairs)==1 and len(calls)==2
    assert report['outcome']=='repaired_candidate_rejected'
    assert not report['accepted']


def test_model_failure_retains_baseline(tmp_path,monkeypatch):
    import src.reverse_engineering as reconstruction
    import src.cad_evaluation as evaluator
    import src.shaft_loop as loop
    scan=tmp_path/'scan.npy';np.save(scan,scan_fixture())
    spec=tmp_path/'drawing.json';spec.write_text('{"checks": []}')
    monkeypatch.setattr(reconstruction,'reconstruct',lambda *a: {'candidate_step':'initial.step','features':[]})
    monkeypatch.setattr(evaluator,'evaluate',lambda *a: {'accepted':False,'rejection_reasons':['Incomplete']})
    def fail(*a):raise RuntimeError('Provider failed')
    monkeypatch.setattr(loop,'decide',fail)
    result=run(scan,'image',tmp_path/'loop','reference',spec)
    assert result['outcome']=='model_failed_baseline_retained'
    assert result['final']==result['initial']
