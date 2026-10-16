"""Render actual fixed scan and model-selected CAD candidate, with recorded evidence."""
import os
from pathlib import Path
import json
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','outputs/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh

root=Path(__file__).resolve().parents[1]
out=root/'outputs/reverse_engineering/ctc05/live_loop_clarified'
r=json.loads((out/'report.json').read_text())
scan=np.load(root/'outputs/reverse_engineering/ctc05/input/scan_points_mm.npy')
rng=np.random.default_rng(42);scan=scan[rng.choice(len(scan),60000,replace=False)]
mesh=trimesh.load(out/'attempt_1/candidate.stl')
points,_=trimesh.sample.sample_surface(mesh,60000,seed=42)
fig=plt.figure(figsize=(12,5.5),layout='constrained')
low=np.minimum(scan.min(axis=0),points.min(axis=0));high=np.maximum(scan.max(axis=0),points.max(axis=0))
for i,(p,title,color) in enumerate([(scan,'Fixed scan input','#164e63'),(points,'Model-selected closed CAD approximation','#ea580c')]):
 ax=fig.add_subplot(1,2,i+1,projection='3d')
 ax.scatter(*p.T,s=.25,color=color,linewidths=0,alpha=.65)
 ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),title=title)
 ax.set_box_aspect(high-low);ax.view_init(elev=23,azim=-55)
 ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
 ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
fig.suptitle('One bounded repair: closed solid created, full reconstruction still rejected',fontsize=14)
fig.savefig(out/'comparison.png',dpi=150)
fig.savefig(root/'docs/assets/shaft-loop.png',dpi=150)
plt.close(fig)
summary={key:r[key] for key in ['outcome','scan_sha256','initial','final','repair','model_role','accepted','model','prior_model_decision']}
(root/'docs/assets/shaft-loop-results.json').write_text(json.dumps(summary,indent=2)+'\n')
