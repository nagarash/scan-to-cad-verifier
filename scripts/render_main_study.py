"""Promote recorded scan-only CTC-05 study to portable README evidence. No model calls."""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','outputs/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/reconstruction_scan'
ASSETS=ROOT/'docs/assets'

def main():
    evaluation_path=STUDY/'held_out_evaluation.json'
    evaluation=(json.loads(evaluation_path.read_text()) if evaluation_path.exists() else
                json.loads((ASSETS/'main-study-results.json').read_text())['evaluation'])
    scan=(np.load(STUDY/'points.npy') if (STUDY/'points.npy').exists() else
          np.asarray(trimesh.load(ROOT/'data/nist/ctc05/scan.ply').vertices))
    rng=np.random.default_rng(42);scan=scan[rng.choice(len(scan),100000,replace=False)]
    mesh_path=STUDY/'reconstructed.stl'
    mesh=trimesh.load(mesh_path if mesh_path.exists() else ROOT/'examples/ctc05_scan_only/reconstructed.stl')
    cad,_=trimesh.sample.sample_surface(mesh,100000,seed=42)
    lo=np.minimum(scan.min(axis=0),cad.min(axis=0));hi=np.maximum(scan.max(axis=0),cad.max(axis=0))
    fig=plt.figure(figsize=(15,5.6),facecolor='white')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for position,points,title in [([.015,.12,.29,.70],scan,'1. Scan input'),
                                   ([.695,.12,.29,.70],None,'3. CAD solid')]:
        ax=fig.add_axes(position,projection='3d')
        if points is not None:
            ax.scatter(*points.T,s=.25,color='#164e63',linewidths=0,depthshade=False)
        else:
            # Render the surface mesh exported from the reconstructed STEP solid.
            light=np.array([.3,-.7,.65]);light/=np.linalg.norm(light)
            intensity=.40+.60*np.clip(mesh.face_normals@light,0,1)
            color=np.array([.17,.43,.69])
            colors=np.column_stack((intensity[:,None]*color,np.ones(len(mesh.faces))))
            collection=Poly3DCollection(mesh.triangles,facecolors=colors,edgecolors='none',linewidths=0)
            ax.add_collection3d(collection)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=title)
        ax.set_box_aspect(hi-lo);ax.view_init(elev=23,azim=-55)
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
        ax.set_axis_off()
    code_ax=fig.add_axes([.36,.19,.28,.58]);code_ax.set_axis_off()
    from matplotlib.patches import FancyBboxPatch
    code_ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=.015',
                                    facecolor='#0f172a',edgecolor='#334155',transform=code_ax.transAxes,clip_on=False))
    code_ax.text(.5,1.17,'2. CadQuery code',ha='center',va='center',fontsize=13,transform=code_ax.transAxes)
    excerpt="""import cadquery as cq

FLANGE_R = 279.4
FLANGE_T = 25.4
SHAFT_R = 31.75

def build():
    base = cyl(FLANGE_R, FLANGE_T)
    # Measured profiles, bores,
    # counterbores and blends...
    return part"""
    code_ax.text(.045,.91,excerpt,ha='left',va='top',fontfamily='monospace',fontsize=9.5,
                 color='#e2e8f0',linespacing=1.6,transform=code_ax.transAxes)
    fig.text(.16,.105,'600,000 XYZ points · millimeters',ha='center',color='#475569',fontsize=11)
    fig.text(.50,.105,'Editable construction program · excerpt',ha='center',color='#475569',fontsize=11)
    fig.text(.84,.105,'STEP (.step) · one valid closed B-rep solid',ha='center',color='#475569',fontsize=11)
    fig.text(.16,.045,'Synthetic sampling · NumPy + OpenCascade',ha='center',color='#0f172a',fontsize=10,fontweight='bold')
    fig.text(.50,.045,'OpenAI Astra in Codex + measurement tools',ha='center',color='#0f172a',fontsize=10,fontweight='bold')
    fig.text(.84,.045,'CadQuery 2.8 / OpenCascade · build + STEP export',ha='center',color='#0f172a',fontsize=10,fontweight='bold')
    from matplotlib.patches import FancyArrowPatch
    for x in [.315,.655]:
        fig.add_artist(FancyArrowPatch((x,.49),(x+.03,.49),transform=fig.transFigure,
                       arrowstyle='-|>',mutation_scale=20,color='#64748b',linewidth=2))
    fig.suptitle('Scan → CadQuery program → CAD solid',fontsize=19,y=.96)
    fig.savefig(ASSETS/'main-study.png',dpi=160);plt.close(fig)
    summary={'study':'NIST CTC-05 scan-only agent reconstruction',
        'generation_provenance':'OpenAI Astra in Codex, attributed by the user; exact checkpoint/model ID, effort, and complete session trace not recorded',
        'stage_tools':{'scan_input':'Synthetic NIST surface sampling: NumPy, trimesh, CadQuery/OpenCascade',
                       'scan_to_program':'OpenAI Astra in Codex with Python geometric measurement tools (user attribution)',
                       'program_to_solid':'CadQuery 2.8 / OpenCascade geometry construction and STEP export'},
        'scan':{'point_count':600000,'units':'mm','noise_sigma_mm':.005,
                'kind':'synthetic complete-surface samples, not a physical scanner simulation',
                'known_issue':'Input included one loose auxiliary STEP face; evaluator uses closed physical solid only'},
        'generation_boundary':'Scan and scan-derived tools only, according to study notes and supplied scripts',
        'evaluation':evaluation,
        'study_scope':'Single scan-only reconstruction case study; not a model ranking',
        'source_program':'examples/ctc05_scan_only/reconstruct.py',
        'program_changes':'Only CLI/output-directory routing changed from supplied program; build geometry preserved'}
    (ASSETS/'main-study-results.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
