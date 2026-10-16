"""Editable reconstruction measured exclusively from input/scan.ply. Units: mm.
See reconstruction_notes.md for approximations and unresolved geometry.
Run with CadQuery 2.8 and trimesh: python reconstruct.py
"""
from pathlib import Path
import math
import cadquery as cq
OUT=Path(__file__).resolve().parents[2]/'outputs/reconstruction_scan'
# Measured dimensions; nominal rounding is an explicit modeling assumption.
FLANGE_R=279.4
FLAT_X=263.525
FLANGE_T=25.4
HUB_R=152.4
HUB_TOP=127.0
BORE_R=127.0
BORE_DEPTH=101.6
EDGE_R=6.35
SHAFT_R=31.75
SHAFT_TOP=482.6
BOLT_CENTERS=[(192.474,111.125),(111.113,192.481),(0,222.25),(-111.113,192.481),(-192.474,111.125),(-192.474,-111.125),(-111.113,-192.481),(0,-222.25),(111.113,-192.481),(192.474,-111.125)]
def box(x,y,z,cx=0,cy=0,z0=0):
 return cq.Workplane('XY').box(x,y,z,centered=(True,True,False)).translate((cx,cy,z0))
def cyl(r,h,z=0,x=0,y=0):
 return cq.Workplane('XY').circle(r).extrude(h).translate((x,y,z))
def fillet_edges(obj,predicate,r):
 edges=[e for e in obj.val().Edges() if predicate(e)]
 if not edges: raise RuntimeError('No matching fillet edges')
 return obj.newObject(edges).fillet(r)
ENGRAVING = [{'holes': [],
  'outer': [[37.6389, 253.1924],
            [35.9041, 251.5855],
            [36.5093, 250.0673],
            [46.7357, 250.0699],
            [46.7506, 246.9403],
            [34.7963, 246.953],
            [32.434, 248.3573],
            [31.7479, 250.2936],
            [32.434, 252.5494],
            [42.7117, 262.0774],
            [42.2028, 263.5908],
            [31.7534, 263.5955],
            [31.738, 266.685],
            [43.9602, 266.7085],
            [45.7603, 266.1344],
            [47.1032, 264.1536],
            [46.7114, 261.7499]]},
 {'holes': [],
  'outer': [[35.825263452378486, 216.24541983104584],
            [35.8287, 216.241],
            [42.5971, 216.2366],
            [43.5753, 215.4353],
            [43.6074, 209.8344],
            [46.7451, 209.8245],
            [46.7515, 218.1699],
            [46.7531, 230.0227],
            [45.557, 229.3953],
            [44.3005, 230.2848],
            [45.4742, 231.5308],
            [46.2697, 231.5345],
            [46.7538, 230.5937],
            [46.6003, 236.6244],
            [45.5699, 238.4931],
            [45.2572, 238.8101],
            [43.5139, 239.7954],
            [41.5217, 239.9383],
            [39.5445, 239.1309],
            [38.2657, 237.7293],
            [37.704, 236.1443],
            [37.6759, 233.9988],
            [37.6692, 226.0897],
            [36.8977, 224.9077],
            [35.7642, 224.8426],
            [34.8981, 225.8189],
            [34.8478, 232.2759],
            [34.8572, 240.68],
            [35.243, 241.5174],
            [46.7206, 241.7883],
            [46.7508, 244.9066],
            [43.6254, 244.9222],
            [43.62346616483993, 244.91982071420574],
            [35.7816, 244.9243],
            [34.0234, 244.4608],
            [32.7072, 243.4024],
            [31.9186, 241.9229],
            [31.7334, 236.4876],
            [31.7544, 225.609],
            [32.6623, 226.5727],
            [34.0187, 225.4946],
            [33.2879, 224.4339],
            [31.7653, 225.4512],
            [31.9447, 224.7026],
            [33.2271, 222.8189],
            [35.1827, 221.7923],
            [37.4543, 221.8282],
            [39.5335, 223.0419],
            [40.7702, 225.5394],
            [40.8213, 235.4818],
            [41.6543, 236.4526],
            [42.7655, 236.4252],
            [43.5821, 235.2315],
            [43.5887, 220.1821],
            [42.7151, 219.342],
            [31.7528, 219.334],
            [31.7561, 216.2337]]}]

def build():
 base=cyl(FLANGE_R,FLANGE_T).intersect(box(2*FLAT_X,600,FLANGE_T))
 hub=cyl(HUB_R,HUB_TOP-FLANGE_T,FLANGE_T).edges('>Z').chamfer(6.35)
 part=base.union(hub)
 # Raised arms: measured cylindrical crowns, flat crest, and vertical sides.
 crown=cq.Workplane('YZ',origin=(-FLAT_X,0,-146.05)).circle(203.2).extrude(2*FLAT_X)
 arms=box(2*FLAT_X,139.7,54.61-25.4,z0=25.4).intersect(crown)
 part=part.union(arms)
 # Concave base transitions; select the seams in the z=25.4 plane.
 part=fillet_edges(part,lambda e:abs(e.Center().z-25.4)<.01 and e.BoundingBox().zlen<.01 and abs(e.Center().x)<260 and e.Length()<500,6.35)
 part=fillet_edges(part,lambda e: e.Center().z>45 and e.Center().z<55 and abs(e.Center().x)>130 and abs(e.Center().x)<155,6.35)
 # Side-open U recesses in arm tops, with four blind drilled holes.
 for sign in [-1,1]:
  slot=cyl(19.05,20,45.72,sign*203.2).union(box(100,38.1,20,cx=sign*253.2,z0=45.72))
  part=part.cut(slot)
  for xx in [203.2,234.95]:
   part=part.cut(cyl(3.96875,45.72-15.24,15.24,sign*xx))
   tip=cq.Solid.makeCone(0,3.96875,2.3847,cq.Vector(sign*xx,0,12.8553))
   part=part.cut(tip)
 # T-shaped upper support: curved front and inclined rear web.
 slope=math.tan(math.radians(12.5))
 def rear(z):return -50.8+(z-279.4)*slope
 thick_y=50.8/math.cos(math.radians(12.5))
 back0,back1=rear(127),rear(279.4)
 web=cq.Workplane('YZ',origin=(-88.9,0,0)).polyline([(back0,127),(back1,279.4),(back1+thick_y,279.4),(back0+thick_y,127)]).close().extrude(177.8)
 main=cyl(127,152.4,127,y=-50.8).intersect(box(127,300,152.4,cy=80,z0=127))
 # Trim the main section to the same inclined rear plane as the web.
 keep=cq.Workplane('YZ',origin=(-100,0,0)).polyline([(back0,127),(back1,279.4),(100,279.4),(100,127)]).close().extrude(200)
 main=main.intersect(keep)
 part=part.union(main.union(web))
 part=fillet_edges(part,lambda e:abs(e.Center().z-127)<.01 and e.BoundingBox().zlen<.01 and e.Length()<400,6.35)
 part=part.union(cyl(SHAFT_R,SHAFT_TOP-279.4,279.4).edges('>Z').chamfer(6.35))
 part=fillet_edges(part,lambda e:abs(e.Center().z-279.4)<.01 and e.geomType()=='CIRCLE' and abs(e.radius()-31.75)<.01,6.35)
 for y,z in [(-48.7109,171.45),(-40.9682,206.375),(-33.2255,241.3)]:
  for sign in [-1,1]:
   part=part.cut(cq.Solid.makeCylinder(5.3581,30.48,cq.Vector(sign*88.9,y,z),cq.Vector(-sign,0,0)))
   part=part.cut(cq.Solid.makeCone(5.3581,0,3.2195,cq.Vector(sign*58.42,y,z),cq.Vector(-sign,0,0)))
 # Bottom blind cavity with measured 45-degree lead-in.
 part=part.cut(cyl(BORE_R,BORE_DEPTH))
 part=part.cut(cq.Solid.makeCone(136.525,127,9.525))
 for x,y in BOLT_CENTERS:
  part=part.cut(cyl(9.525,25.4,0,x,y)).cut(cyl(19.05,12.7,12.7,x,y))
 # Approximate measured polygonal engraving, generated solely from scan points.
 for poly in ENGRAVING:
  cutter=cq.Workplane('YZ',origin=(61.468,0,0)).polyline(poly['outer']).close()
  part=part.cut(cutter.extrude(2.1))
 return part
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=OUT)
 OUT=parser.parse_args().output_dir.resolve()
 OUT.mkdir(parents=True,exist_ok=True)
 model=build()
 assert model.val().isValid(), 'Invalid solid'
 assert len(model.solids().vals())==1, 'Expected one connected solid'
 cq.exporters.export(model,str(OUT/'reconstructed.step'))
 cq.exporters.export(model,str(OUT/'reconstructed.stl'),tolerance=.08,angularTolerance=.08)
 # Remove zero-area/duplicate tessellation facets at coincident CAD seams.
 import trimesh
 mesh=trimesh.load(str(OUT/'reconstructed.stl'))
 mesh.merge_vertices(digits_vertex=7)
 mesh.update_faces(mesh.nondegenerate_faces())
 mesh.update_faces(mesh.unique_faces())
 assert mesh.is_watertight, 'STL must be watertight'
 mesh.export(str(OUT/'reconstructed.stl'))
 print('Valid solids:',len(model.solids().vals()),'Volume mm3:',model.val().Volume())
