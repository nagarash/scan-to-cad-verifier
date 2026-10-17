"""Evaluate a benchmark run against its task's dimension table.

Usage: python benchmark/evaluate.py --run benchmark/runs/<part>/<model>/<ts>

Reads <run-dir>/measurements.json (see measure.py) and the task's dimension
table, writes <run-dir>/score.json with per-dimension pass/fail and the
aggregate score. Reads like an inspection report: worst violations first.
"""
import argparse
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def evaluate_dimension(dim: dict, measured_mm: float) -> dict:
    nom = dim["nominal_mm"]
    lo = nom - dim["tol_minus_mm"]
    hi = nom + dim["tol_plus_mm"]
    passed = lo <= measured_mm <= hi
    violation = 0.0 if passed else min(abs(measured_mm - lo), abs(measured_mm - hi))
    return {
        "id": dim["id"],
        "description": dim["description"],
        "nominal_mm": nom,
        "tol_minus_mm": dim["tol_minus_mm"],
        "tol_plus_mm": dim["tol_plus_mm"],
        "measured_mm": measured_mm,
        "deviation_mm": measured_mm - nom,
        "passed": passed,
        "violation_mm": violation,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run)
    part = run_dir.parts[run_dir.parts.index("runs") + 1]
    task = json.loads((BENCH / "tasks" / part / "task.json").read_text())
    measurements = json.loads((run_dir / "measurements.json").read_text())

    results = []
    for dim in task["dimensions"]:
        if dim["id"] not in measurements:
            results.append({"id": dim["id"], "description": dim["description"],
                            "passed": False, "error": "not measured"})
            continue
        results.append(evaluate_dimension(dim, measurements[dim["id"]]["value_mm"]))

    measured = [r for r in results if "measured_mm" in r]
    passed = [r for r in measured if r["passed"]]
    score = {
        "task": part,
        "model": (run_dir.parent.name.replace("__", "/")),
        "dimensions_total": len(results),
        "dimensions_measured": len(measured),
        "dimensions_passed": len(passed),
        "pass_rate": (len(passed) / len(measured)) if measured else 0.0,
        "strict_pass": bool(measured) and len(passed) == len(measured),
        "worst_violations": sorted(
            (r for r in measured if not r["passed"]),
            key=lambda r: r["violation_mm"], reverse=True,
        ),
        "dimensions": sorted(results, key=lambda r: r.get("violation_mm", 0), reverse=True),
    }
    (run_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n")
    print(f"{part}: {len(passed)}/{len(measured)} dimensions in tolerance "
          f"(pass rate {score['pass_rate']:.1%}, strict: {score['strict_pass']})")
    for r in score["worst_violations"][:5]:
        print(f"  FAIL {r['id']}: measured {r['measured_mm']:.4f} vs "
              f"{r['nominal_mm']} ±{r['tol_plus_mm']}/{r['tol_minus_mm']} "
              f"(off by {r['violation_mm']:.4f} mm)")


if __name__ == "__main__":
    main()
