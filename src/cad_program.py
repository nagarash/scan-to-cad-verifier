"""Restricted generated CadQuery programs, run in a separate timed process.

The syntax allowlist limits model output; it is not an OS security sandbox.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ALLOWED_ATTRIBUTES=set('''Workplane Vector Plane Location Matrix Sketch origin xDir normal
box cylinder sphere cone circle ellipse rect polygon polyline line lineTo hLine hLineTo
vLine vLineTo move moveTo center close threePointArc radiusArc tangentArcPoint spline
splineApprox parametricCurve bezier sagittaArc slot2D text pushPoints rarray polarArray
extrude revolve loft sweep twistExtrude cutThruAll cutBlind hole cboreHole cskHole
faces edges vertices wires workplane transformed translate rotate mirror mirrorX mirrorY
union cut intersect combine add clean fillet chamfer shell split val vals solids
sketch segment arc assemble finalize moved located tag workplaneFromTagged reset face
sin cos tan pi sqrt radians degrees'''.split())
ALLOWED_NODES=(ast.Module,ast.Import,ast.alias,ast.Assign,ast.Expr,ast.Name,ast.Load,ast.Store,
 ast.Attribute,ast.Call,ast.Constant,ast.Tuple,ast.List,ast.Dict,ast.keyword,ast.UnaryOp,ast.USub,
 ast.UAdd,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod,ast.Subscript,ast.Slice)


def validate_program(code):
    if not isinstance(code,str) or len(code)>100000:raise ValueError('Invalid program size')
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node,ALLOWED_NODES):raise ValueError(f'Unsupported generated syntax: {type(node).__name__}')
        if isinstance(node,ast.Import):
            for name in node.names:
                if (name.name,name.asname) not in [('cadquery','cq'),('math',None)]:raise ValueError('Only cadquery as cq and math imports allowed')
        if isinstance(node,ast.Name) and (node.id.startswith('_') or node.id in {'open','exec','eval','getattr','setattr','globals','locals','compile','input','__import__'}):
            raise ValueError('Disallowed generated name')
        if isinstance(node,ast.Attribute) and node.attr not in ALLOWED_ATTRIBUTES:raise ValueError(f'Unsupported CAD operation: {node.attr}')
        if isinstance(node,ast.Constant):
            if isinstance(node.value,(float,int)) and abs(node.value)>1e8:raise ValueError('Numeric literal too large')
            if isinstance(node.value,str) and len(node.value)>1000:raise ValueError('String literal too long')
    if not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='r' for n in ast.walk(tree)):
        raise ValueError('Generated program must assign the final model to r')
    return tree


def execute_program(code,transform,output_dir,timeout_seconds=90):
    validate_program(code)
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'generated.py').write_text(code+'\n')
    (out/'transform.json').write_text(json.dumps(transform,indent=2)+'\n')
    # Worker accepts code+transform only. No reference paths or API credentials.
    env={k:v for k,v in os.environ.items() if k in ['PATH','SYSTEMROOT','TMPDIR','LANG','LC_ALL']}
    worker=Path(__file__).resolve()
    result=subprocess.run([sys.executable,str(worker),'--worker',str(out)],cwd=out,
                          env=env,capture_output=True,text=True,timeout=timeout_seconds)
    (out/'execution.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise ValueError('Generated CAD execution failed; see execution.log')
    return json.loads((out/'execution.json').read_text())


def worker(directory):
    import cadquery as cq
    import math
    out=Path(directory)
    code=(out/'generated.py').read_text();validate_program(code)
    def imports(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='cadquery':return cq
        if name=='math':return math
        raise ImportError('Disallowed import')
    scope={'__builtins__':{'__import__':imports,'abs':abs,'min':min,'max':max,'float':float,'int':int},'cq':cq,'math':math}
    exec(compile(code,'generated.py','exec'),scope)
    result=scope['r'];shape=result.val() if isinstance(result,cq.Workplane) else result
    if not isinstance(shape,cq.Shape) or not shape.isValid() or not shape.Solids():
        raise ValueError('Generated program did not produce a valid solid')
    transform=json.loads((out/'transform.json').read_text())
    # Upstream outputs CAD coordinates 100x normalized point coordinates.
    shape=shape.scale(transform['scale_mm']/100).translate(tuple(transform['center_mm']))
    if not shape.isValid():raise ValueError('Denormalized CAD invalid')
    cq.exporters.export(shape,str(out/'candidate.step'))
    cq.exporters.export(shape,str(out/'candidate.stl'))
    (out/'execution.json').write_text(json.dumps({'candidate_step':str(out/'candidate.step'),
         'solid_count':len(shape.Solids()),'valid':shape.isValid(),'volume_mm3':shape.Volume()},indent=2)+'\n')

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
