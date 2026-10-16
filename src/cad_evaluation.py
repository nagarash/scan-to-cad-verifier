"""Held-out STEP feature/dimension evaluator, independent of scan reconstruction."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def inventory(path):
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    imported=cq.importers.importStep(str(path)).val()
    solids=imported.Solids()
    shape=cq.Compound.makeCompound(solids) if solids else imported
    ignored_faces=len(imported.Faces())-len(shape.Faces())
    box=shape.BoundingBox()
    cylinders={}; counts={}
    for face in shape.Faces():
        kind=face.geomType();counts[kind]=counts.get(kind,0)+1
        if kind!='CYLINDER':continue
        c=BRepAdaptor_Surface(face.wrapped).Cylinder()
        axis=np.array(c.Axis().Direction().Coord());axis*=1 if axis[np.argmax(abs(axis))]>0 else -1
        origin=np.array(c.Location().Coord());origin-=axis*np.dot(origin,axis)
        key=tuple(round(x,5) for x in [*origin,*axis,c.Radius()])
        cylinders[key]={'axis':axis.tolist(),'axis_origin_mm':origin.tolist(),'radius_mm':c.Radius()}
    return {'valid':shape.isValid(),'solid_count':len(shape.Solids()),'face_count':len(shape.Faces()),
            'ignored_auxiliary_faces':ignored_faces,
            'surface_counts':counts,'extent_mm':[box.xlen,box.ylen,box.zlen],
            'volume_mm3':shape.Volume(tol=1e-7) if shape.Solids() else None,
            'volume_method':'OpenCascade adaptive integration, relative tolerance 1e-7',
            'cylinders':list(cylinders.values())}


def match_cylinders(candidate,reference,max_center_error_mm=1.,max_radius_error_mm=1.):
    a,b=candidate['cylinders'],reference['cylinders']
    pairs=[]
    if a and b:
        costs=np.empty((len(a),len(b)))
        for i,x in enumerate(a):
            for j,y in enumerate(b):
                axis=abs(np.dot(x['axis'],y['axis']))
                costs[i,j]=(np.linalg.norm(np.array(x['axis_origin_mm'])-y['axis_origin_mm'])
                            +abs(x['radius_mm']-y['radius_mm']) if axis>.9999 else 1e9)
        rows,cols=linear_sum_assignment(costs)
        for i,j in zip(rows,cols):
            x,y=a[i],b[j]
            center=float(np.linalg.norm(np.array(x['axis_origin_mm'])-y['axis_origin_mm']))
            radius=x['radius_mm']-y['radius_mm']
            if costs[i,j]<1e9 and center<=max_center_error_mm and abs(radius)<=max_radius_error_mm:
                pairs.append({'candidate_index':int(i),'reference_index':int(j),'diameter_mm':2*x['radius_mm'],
                              'reference_diameter_mm':2*y['radius_mm'],'diameter_error_mm':2*radius,
                              'axis_offset_mm':center})
    return {'matches':pairs,'missing_reference_indices':sorted(set(range(len(b)))-{p['reference_index'] for p in pairs}),
            'extra_candidate_indices':sorted(set(range(len(a)))-{p['candidate_index'] for p in pairs}),
            'association':'one-to-one held-out CAD association; no independent semantic feature naming'}


def surface_comparison(candidate_step, reference_step, samples=5000):
    """Bidirectional sampled distance to fine CAD tessellations (millimeters)."""
    import cadquery as cq
    import trimesh
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    def mesh(path):
        imported=cq.importers.importStep(str(path)).val()
        solids=imported.Solids()
        shape=cq.Compound.makeCompound(solids) if solids else imported
        vertices,faces=shape.tessellate(.002,.03)
        return trimesh.Trimesh(vertices=[v.toTuple() for v in vertices],faces=faces,process=False)
    def distances(points, target):
        data=vtk.vtkPolyData(); vertices=vtk.vtkPoints()
        vertices.SetData(numpy_to_vtk(np.asarray(target.vertices),deep=True));data.SetPoints(vertices)
        cells=vtk.vtkCellArray()
        packed=np.column_stack((np.full(len(target.faces),3),target.faces)).astype(np.int64).ravel()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(packed,deep=True));data.SetPolys(cells)
        locator=vtk.vtkStaticCellLocator();locator.SetDataSet(data);locator.BuildLocator()
        cell,sub,distance=vtk.reference(0),vtk.reference(0),vtk.reference(0.)
        closest=[0.,0.,0.];result=[]
        for point in points:
            locator.FindClosestPoint(point,closest,cell,sub,distance)
            result.append(np.sqrt(float(distance)))
        return np.array(result)
    a,b=mesh(candidate_step),mesh(reference_step)
    pa,_=trimesh.sample.sample_surface(a,samples,seed=71)
    pb,_=trimesh.sample.sample_surface(b,samples,seed=83)
    ab,ba=distances(pa,b),distances(pb,a)
    return {'samples_each_direction':samples,'tessellation_mm':.002,'angular_tolerance_rad':.03,
            'candidate_to_reference_p95_mm':float(np.quantile(ab,.95)),
            'reference_to_candidate_p95_mm':float(np.quantile(ba,.95)),
            'sampled_max_mm':float(max(ab.max(),ba.max())),
            'scope':'Sampled geometry evidence, not proof of complete GD&T conformance'}


def evaluate(candidate_step,reference_step,drawing_checks,output_path):
    candidate,reference=inventory(candidate_step),inventory(reference_step)
    cylinders=match_cylinders(candidate,reference)
    dimensions=[]
    for check in drawing_checks['checks']:
        # This one reviewed CTC-05 callout explicitly identifies the central rear bore.
        targets=[i for i,c in enumerate(reference['cylinders'])
                 if np.linalg.norm(c['axis_origin_mm'])<1e-4 and abs(c['axis'][2])>.9999
                 and abs(2*c['radius_mm']-check['nominal_mm'])<1e-4]
        matches=[p for p in cylinders['matches'] if p['reference_index'] in targets]
        measured=matches[0]['diameter_mm'] if len(matches)==1 and len(targets)==1 else None
        dimensions.append({**check,'measured_mm':measured,'passed':measured is not None
                           and abs(measured-check['nominal_mm'])<=check['tolerance_mm'],
                           'association':'held-out CAD-assisted'})
    reasons=[]
    if not candidate['valid'] or candidate['solid_count']!=1:reasons.append('Candidate must be one valid closed solid')
    if cylinders['missing_reference_indices']:reasons.append('Reference cylindrical features are missing')
    if cylinders['extra_candidate_indices']:reasons.append('Candidate has unmatched cylindrical features')
    if any(not c['passed'] for c in dimensions):reasons.append('Selected drawing size check failed or feature missing')
    surface=surface_comparison(candidate_step,reference_step)
    if max(surface['candidate_to_reference_p95_mm'],surface['reference_to_candidate_p95_mm'])>.05 or surface['sampled_max_mm']>.2:
        reasons.append('Sampled bidirectional surface distances exceed geometry thresholds')
    if np.max(np.abs(np.array(candidate['extent_mm'])-reference['extent_mm']))>.1:
        reasons.append('Overall extents differ by more than 0.1 mm')
    volume_error=None
    if candidate['volume_mm3'] is not None and reference['volume_mm3']:
        volume_error=abs(candidate['volume_mm3']-reference['volume_mm3'])/reference['volume_mm3']
        if volume_error>.001:reasons.append('Relative solid-volume error exceeds 0.1%')
    report={'candidate':candidate,'reference':reference,'cylinder_comparison':cylinders,
            'drawing_checks':dimensions,'accepted':not reasons,'rejection_reasons':reasons,
            'surface_comparison':surface,'relative_volume_error':volume_error,
            'acceptance_scope':'Selected sizes and sampled geometry; datum-based GD&T not evaluated',
            'reference_used_only_by_evaluator':True,
            'extent_error_mm':(np.array(candidate['extent_mm'])-reference['extent_mm']).tolist()}
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report
