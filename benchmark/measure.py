"""Measure a benchmark run's reconstructed solid against the task's dimension table.

Usage: python benchmark/measure.py --run benchmark/runs/<part>/<model>/<ts>

Expects <run-dir>/reconstruct.py to define `build()` returning a CadQuery
solid (adapt the generated program if needed). Writes
<run-dir>/measurements.json: {dimension_id: {"value_mm": float}}.

Measurement adapters: `overall_extent` and `plane_distance` are implemented
generically below. `cylinder_diameter` / `hole_position` should delegate to
the repo's existing fitters in src/ (the same code behind
data/nist/*/drawing_checks.json); they are stubbed with a clear error until
wired.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def load_solid(run_dir: Path):
    spec = importlib.util.spec_from_file_location("recon", run_dir / "reconstruct.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        sys.exit("reconstruct.py must define build() returning a CadQuery solid")
    return mod.build()


def measure_overall_extent(solid, dim):
    bb = solid.val().BoundingBox()
    axis = dim["measure"].get("axis", "x")
    lo = {"x": bb.xmin, "y": bb.ymin, "z": bb.zmin}[axis]
    hi = {"x": bb.xmax, "y": bb.ymax, "z": bb.zmax}[axis]
    return hi - lo


def measure_plane_distance(solid, dim):
    raise NotImplementedError(
        "plane_distance: fit the two named planes with the repo's plane "
        "fitter in src/ and return their separation in mm"
    )


def measure_cylinder_diameter(solid, dim):
    raise NotImplementedError(
        "cylinder_diameter: locate the cylinder described by "
        f"measure.hint={dim['measure'].get('hint')!r} using the repo's cylinder "
        "fitter in src/ (same code behind drawing_checks.json) and return its "
        "diameter in mm"
    )


def measure_hole_position(solid, dim):
    raise NotImplementedError(
        "hole_position: fit the hole axis with the repo's cylinder fitter in "
        "src/ and return its position per measure.spec in mm"
    )


MEASURERS = {
    "overall_extent": measure_overall_extent,
    "plane_distance": measure_plane_distance,
    "cylinder_diameter": measure_cylinder_diameter,
    "hole_position": measure_hole_position,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under benchmark/runs/")
    args = ap.parse_args()
    run_dir = Path(args.run)
    part = run_dir.parts[run_dir.parts.index("runs") + 1]
    task = json.loads((BENCH / "tasks" / part / "task.json").read_text())

    solid = load_solid(run_dir)
    out = {}
    for dim in task["dimensions"]:
        mtype = dim["measure"]["type"]
        if mtype not in MEASURERS:
            sys.exit(f"unknown measure type: {mtype}")
        out[dim["id"]] = {"value_mm": MEASURERS[mtype](solid, dim)}
    (run_dir / "measurements.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"measured {len(out)} dimensions -> {run_dir / 'measurements.json'}")


if __name__ == "__main__":
    main()
