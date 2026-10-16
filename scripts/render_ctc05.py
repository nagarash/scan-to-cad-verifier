"""Render checked-in CTC-05 evidence from the recorded local benchmark."""
import json
from pathlib import Path
import shutil
import os
os.environ.setdefault('MPLCONFIGDIR','outputs/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
r=json.loads((root/'outputs/nist/ctc05/report.json').read_text())
assets=root/'docs/assets'
shutil.copyfile(root/'outputs/nist/ctc05/scan_views.png',assets/'ctc05-scan.png')
(assets/'ctc05-results.json').write_text(json.dumps(r,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
for label,color in [('coarse','#ea580c'),('fine','#0891b2')]:
 runs=[x for x in r['runs'] if x['resolution']==label]
 ax.plot([x['seed'] for x in runs], [abs(x['bore_check']['measured_diameter_mm']-254) for x in runs],
         'o-',label=f'{label.capitalize()} scan mesh',color=color,lw=2)
ax.axhline(.0254,color='#475569',ls='--',label='Drawing tolerance: 0.0254 mm')
ax.set(xlabel='Random seed',ylabel='Absolute rear-bore diameter error (mm)',
       title='CTC-05: scan mesh resolution changes the size-check result',xticks=[42,43,44],ylim=(0,.065))
ax.legend();ax.grid(alpha=.15)
fig.savefig(assets/'ctc05-results.png',dpi=160)
plt.close(fig)
