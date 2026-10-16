"""Generate a portable point-only benchmark input with held-out quality evidence.

Fine tessellation proposes samples; projection onto trimmed CAD surfaces removes
faceting. No CAD faces, normals, nominal values, or reference points enter input/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def projected_samples(step_path, n_points=600000, seed=42, min_per_face=250):
    import cadquery as cq
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt, gp_Pnt2d
    if type(n_points) is not int or n_points < 500 or min_per_face < 0:
        raise ValueError('Invalid sample count')
    solid = cq.importers.importStep(str(step_path)).val()
    if not solid.isValid() or len(solid.Solids()) != 1:
        raise ValueError('Expected one valid solid')
    faces = solid.Faces()
    if n_points < min_per_face*len(faces):
        raise ValueError('Point budget below per-face coverage minimum')
    rng = np.random.default_rng(seed)
    area = np.array([f.Area() for f in faces])
    counts = rng.multinomial(n_points-min_per_face*len(faces),area/area.sum())+min_per_face
    all_points, coverage = [], []
    max_projection, max_plane_residual = 0., 0.
    for index,(face,count) in enumerate(zip(faces,counts)):
        if count == 0:
            continue
        vertices, triangles = face.tessellate(.002,.03)
        mesh=trimesh.Trimesh(vertices=[v.toTuple() for v in vertices],faces=triangles,process=False)
        surface=BRep_Tool.Surface_s(face.wrapped)
        projector=GeomAPI_ProjectPointOnSurf()
        projector.Init(surface,*BRepTools.UVBounds_s(face.wrapped),1e-9)
        classifier=BRepClass_FaceClassifier()
        accepted=[]
        # Bounded rejection avoids silently accepting projected samples outside trims.
        for batch in range(12):
            pending=count-len(accepted)
            if pending == 0:
                break
            candidates,_=trimesh.sample.sample_surface(mesh,int(pending*1.05)+8,seed=int(rng.integers(2**31)))
            for point in candidates:
                projector.Perform(gp_Pnt(*point))
                if projector.NbPoints() == 0:
                    continue
                u,v=projector.LowerDistanceParameters()
                classifier.Perform(face.wrapped,gp_Pnt2d(u,v),1e-7)
                if classifier.State() not in (TopAbs_IN,TopAbs_ON):
                    continue
                projected=np.array(projector.NearestPoint().Coord())
                max_projection=max(max_projection,float(np.linalg.norm(projected-point)))
                # Surface.Value is independent of the returned projected point.
                residual=np.linalg.norm(projected-np.array(surface.Value(u,v).Coord()))
                max_plane_residual=max(max_plane_residual,float(residual))
                accepted.append(projected)
                if len(accepted)==count:
                    break
        if len(accepted) != count:
            raise RuntimeError(f'Unable to sample trimmed face {index}')
        all_points.extend(accepted)
        coverage.append({'face_index':index,'surface_type':face.geomType(),'area_mm2':face.Area(),'points':int(count)})
    points=np.asarray(all_points)
    rng.shuffle(points)
    return points, {'method':'fine mesh proposals projected onto trimmed CAD surfaces',
                    'sampling_distribution':'area-weighted with a per-face minimum; not uniform',
                    'min_points_per_face':min_per_face,'face_coverage':coverage,
                    'max_removed_faceting_mm':max_projection,
                    'max_projection_surface_residual_mm':max_plane_residual}


def write_point_inputs(points, directory):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 points')
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'scan_points_mm.npy',p)
    np.savetxt(out/'scan.xyz',p,fmt='%.8f')
    # Binary PLY is small and directly usable by point-cloud/CAD tools.
    with (out/'scan.ply').open('wb') as stream:
        header=f'ply\nformat binary_little_endian 1.0\ncomment units millimeters\nelement vertex {len(p)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n'
        stream.write(header.encode('ascii'));stream.write(p.astype('<f8').tobytes())
    return {name:hashlib.sha256((out/name).read_bytes()).hexdigest()
            for name in ['scan_points_mm.npy','scan.xyz','scan.ply']}


def generate(step_path, output_dir, n_points=600000, noise_mm=.005, seed=42, min_per_face=250):
    if not np.isfinite(noise_mm) or noise_mm<0:
        raise ValueError('Invalid noise')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    clean,quality=projected_samples(step_path,n_points,seed,min_per_face)
    rng=np.random.default_rng(seed+1)
    errors=rng.normal(0,noise_mm,clean.shape)
    scan=clean+errors
    hashes=write_point_inputs(scan,out/'input')
    evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
    np.save(evaluation/'clean_surface_points_mm.npy',clean)
    quality.update(source_sha256=hashlib.sha256(Path(step_path).read_bytes()).hexdigest(),
                   seed=seed,point_count=len(scan),noise_sigma_mm=noise_mm,
                   noise_displacement_p95_mm=float(np.quantile(np.linalg.norm(errors,axis=1),.95)),
                   input_hashes=hashes,synthetic=True,
                   limitations=['Complete surface oracle: hidden surfaces included',
                     'No optical sensor, calibration, occlusion or registration simulation',
                     'Per-face coverage stratification uses source geometry only during simulation',
                     'No face labels or reference points are exported in input/'])
    (evaluation/'scan_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    (out/'input/README.txt').write_text('Synthetic point cloud; XYZ coordinates in millimeters.\nUse only this directory for scan-only reconstruction.\nNo reference geometry or feature labels are included.\nAll surfaces are sampled, including hidden surfaces.\n')
    return scan,quality


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',default='data/nist/ctc05/reference.step')
    p.add_argument('--output-dir',default='outputs/reverse_engineering/ctc05')
    p.add_argument('--points',type=int,default=600000)
    a=p.parse_args(); _,q=generate(a.step,a.output_dir,a.points)
    print(json.dumps({k:q[k] for k in ['point_count','noise_sigma_mm','noise_displacement_p95_mm','max_removed_faceting_mm']},indent=2))

if __name__=='__main__':main()
