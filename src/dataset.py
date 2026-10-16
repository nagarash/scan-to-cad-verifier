"""Five reproducible procedural CAD references. These are NOT downloaded Onshape parts."""
from pathlib import Path
import json
import numpy as np

PARTS = ['four_hole_plate','two_hole_plate','slotted_plate','raised_boss','l_bracket']

def build_part(name):
    import cadquery as cq
    if name == 'four_hole_plate':
        return cq.Workplane('XY').box(60,40,6,centered=(True,True,False)).faces('>Z').workplane().pushPoints([(-20,-10),(-20,10),(20,-10),(20,10)]).hole(6)
    if name == 'two_hole_plate':
        return cq.Workplane('XY').box(55,30,5,centered=(True,True,False)).faces('>Z').workplane().pushPoints([(-17,0),(17,0)]).hole(8)
    if name == 'slotted_plate':
        return cq.Workplane('XY').box(70,35,5,centered=(True,True,False)).faces('>Z').workplane().slot2D(28,8).cutThruAll()
    if name == 'raised_boss':
        return cq.Workplane('XY').box(50,50,5,centered=(True,True,False)).faces('>Z').workplane().circle(12).extrude(10).faces('>Z').workplane().hole(8)
    if name == 'l_bracket':
        base=cq.Workplane('XY').box(60,35,5,centered=(True,True,False))
        wall=cq.Workplane('XY').box(60,5,25,centered=(True,True,False)).translate((0,15,5))
        return base.union(wall)
    raise ValueError(f'Unknown part: {name}')

def make_dataset(directory='data/generated', n_points=12000, noise_sigma_mm=0.05, seed=42):
    import cadquery as cq
    import trimesh
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for idx,name in enumerate(PARTS):
        folder=out/name;folder.mkdir(exist_ok=True)
        part=build_part(name)
        cq.exporters.export(part,str(folder/'reference.step'))
        cq.exporters.export(part,str(folder/'reference.stl'))
        mesh=trimesh.load_mesh(folder/'reference.stl',process=False)
        reference,_=trimesh.sample.sample_surface(mesh,n_points,seed=seed+idx)
        scan=reference+np.random.default_rng(seed+idx).normal(0,noise_sigma_mm,reference.shape)
        np.save(folder/'reference_points_mm.npy',reference)
        np.save(folder/'scan_points_mm.npy',scan)
        meta={'id':name,'origin':'procedurally generated in this repository','license':'project-generated sample; no third-party CAD source','units':'mm','noise_sigma_mm':noise_sigma_mm,'seed':seed+idx}
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
        manifest.append(meta)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return out

if __name__=='__main__':
    print(make_dataset())
