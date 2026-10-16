import numpy as np
from src.features import summarize_points

def test_extents():
    result=summarize_points(np.array([[0.,0,0],[1,2,3]]*5))
    assert result['axis_aligned_extents_mm']==[1.,2.,3.]
