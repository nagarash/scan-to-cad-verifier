import numpy as np
from src.geometry import evaluate_points


def test_identical_points(tmp_path):
    pts=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    a=tmp_path/'a.npy'; b=tmp_path/'b.npy'
    np.save(a,pts); np.save(b,pts)
    result=evaluate_points(a,b)
    assert result['symmetric_chamfer_mm']==0
