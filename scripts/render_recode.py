"""Render actual model input and generated candidate with held-out evidence."""
from pathlib import Path
import os,json
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','outputs/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
root=Path(__file__).resolve().parents[1];out=root/'outputs/reverse_engineering/ctc05/cad_recode'
r=json.loads((out/'report.json').read_text())
scan=np.load(root/'outputs/reverse_engineering/ctc05/input/scan_points_mm.npy')
rng=np.random.default_rng(42);scan=scan[rng.choice(len(scan),50000,replace=False)]
t=r['transform'];model=np.load(out/'model_points_normalized.npy')*t['scale_mm']+t['center_mm']
mesh=trimesh.load(out/'candidate.stl');candidate,_=trimesh.sample.sample_surface(mesh,50000,seed=42)
lo=np.minimum(scan.min(axis=0),candidate.min(axis=0));hi=np.maximum(scan.max(axis=0),candidate.max(axis=0))
fig=plt.figure(figsize=(14,5),layout='constrained')
for i,(p,title,color,size) in enumerate([(scan,'Fixed scan (600,000 points)','#164e63',.3),(model,'Actual model input (256 points)','#0891b2',8),(candidate,'Generated CAD (2 solids; rejected)','#ea580c',.3)]):
 ax=fig.add_subplot(1,3,i+1,projection='3d');ax.scatter(*p.T,s=size,color=color,linewidths=0,depthshade=False)
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=title)
 ax.set_box_aspect(hi-lo);ax.view_init(elev=23,azim=-55)
 ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
 ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('')
fig.suptitle('CAD-Recode v1.5: direct point-cloud-to-program test',fontsize=15)
fig.savefig(out/'comparison.png',dpi=150);fig.savefig(root/'docs/assets/cad-recode.png',dpi=150)
plt.close(fig)
(root/'docs/assets/cad-recode-results.json').write_text(json.dumps(r,indent=2)+'\n')
