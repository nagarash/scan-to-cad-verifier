"""Constrained scan-fitted shaft envelope. Approximate design hypothesis, not exact CAD."""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks


def plane_modes(values, bin_mm=.05, minimum=1000):
    counts,edges=np.histogram(values,bins=np.arange(values.min()-.1,values.max()+.15,bin_mm))
    indices,_=find_peaks(counts,height=minimum,distance=5)
    modes=[]
    for i in indices:
        value=(edges[i]+edges[i+1])/2
        for _ in range(3):
            support=values[abs(values-value)<bin_mm]
            value=float(np.median(support))
        modes.append({'value_mm':value,'points':len(support)})
    return sorted(modes,key=lambda x:x['value_mm'])


def build(scan,features,output_dir):
    """Known aligned shaft family; inputs contain scan measurements only."""
    import cadquery as cq
    p=np.asarray(scan)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Invalid scan')
    main=[f for f in features if np.linalg.norm(f.center_xy_mm)<.2]
    if len(main)!=4:raise ValueError('Expected four supported coaxial cylinder patches')
    shaft,bore,hub,flange=sorted(main,key=lambda f:f.radius_mm)
    cx,cy=np.median([f.center_xy_mm for f in main],axis=0)
    modes=plane_modes(p[:,2]);z=np.array([m['value_mm'] for m in modes])
    if len(z)<6:raise ValueError('Insufficient supported planar levels')
    bottom,top=float(z.min()),float(z.max())
    def near(value):return float(z[np.argmin(abs(z-value))])
    flange_top=near(flange.z_high_mm)
    hub_top=near(hub.z_high_mm)
    shaft_start=near(shaft.z_low_mm)
    bore_end=near(bore.z_high_mm)
    if not bottom<flange_top<bore_end<hub_top<shaft_start<top:
        raise ValueError('Planar/cylindrical topology hypothesis unsupported')
    def cylinder(radius,low,high):
        return cq.Solid.makeCylinder(radius,high-low,cq.Vector(cx,cy,low))
    # Sample bounds remove noise tails. They are not snapped to reference sizes.
    flange_points=p[(p[:,2]>bottom+.1)&(p[:,2]<flange_top-.1)]
    if len(flange_points)<100:raise ValueError('Insufficient flange surface support')
    xmin,xmax=np.quantile(flange_points[:,0],[.001,.999])
    clipping=cq.Solid.makeBox(xmax-xmin,2*flange.radius_mm+2,flange_top-bottom,
                             cq.Vector(xmin,cy-flange.radius_mm-1,bottom))
    base=cylinder(flange.radius_mm,bottom,flange_top).intersect(clipping)
    base=base.fuse(cylinder(hub.radius_mm,flange_top,hub_top))
    # A rectangular upper body is an explicit approximation of unresolved flats.
    body=p[(p[:,2]>hub_top+.5)&(p[:,2]<shaft_start-.5)]
    if len(body)<100:raise ValueError('Insufficient upper-body surface support')
    lo,hi=np.quantile(body[:,:2],[.001,.999],axis=0)
    block=cq.Solid.makeBox(hi[0]-lo[0],hi[1]-lo[1],shaft_start-hub_top,
                           cq.Vector(*lo,hub_top))
    base=base.fuse(block).fuse(cylinder(shaft.radius_mm,shaft_start-.01,top))
    base=base.cut(cylinder(bore.radius_mm,bottom-.01,bore_end)).clean()
    if not base.isValid() or len(base.Solids())!=1:raise ValueError('Closure failed')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cq.exporters.export(base,str(out/'candidate.step'))
    cq.exporters.export(base,str(out/'candidate.stl'))
    return {'candidate_step':str(out/'candidate.step'),'valid_solid':True,
            'plane_modes':modes,'bottom_mm':bottom,'top_mm':top,
            'flange_top_mm':flange_top,'hub_top_mm':hub_top,'shaft_start_mm':shaft_start,
            'bore_end_mm':bore_end,'body_bounds_xy_mm':[lo.tolist(),hi.tolist()],
            'hypotheses':['Four coaxial cylinders interpreted as shaft, bore, hub, flange',
                          'Upper body approximated by a rectangular extrusion',
                          'Rim, hub and shaft shoulders approximated as sharp transitions'],
            'unresolved':['Counterbores and mounting holes','Cross-axis holes','Blends and chamfers',
                          'Upper-body taper and curved profile','Small slots and modeled lettering']}
