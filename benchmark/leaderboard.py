"""Aggregate every benchmark run into leaderboard.md.

Walks benchmark/runs/<part>/<model>/<ts>/{score,usage}.json and renders a
markdown table: model, cost, dimensional pass rate, strict pass/fail.
"""
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
RUNS = BENCH / "runs"


def main():
    rows = []
    if RUNS.exists():
        for score_p in sorted(RUNS.glob("*/*/*/score.json")):
            score = json.loads(score_p.read_text())
            usage_p = score_p.parent / "usage.json"
            usage = json.loads(usage_p.read_text()) if usage_p.exists() else {}
            cost = usage.get("cost_usd")
            rows.append({
                "task": score["task"],
                "model": score["model"],
                "pass": f"{score['dimensions_passed']}/{score['dimensions_measured']}",
                "rate": f"{score['pass_rate']:.1%}",
                "strict": "✅" if score["strict_pass"] else "❌",
                "cost": f"${cost:.4f}" if cost is not None else "n/a",
                "run": score_p.parent.name,
            })

    lines = [
        "# Scan-to-CAD Benchmark Leaderboard",
        "",
        "Primary score: dimensional pass rate — fraction of drawing dimensions",
        "measured on the reconstructed solid that fall within tolerance.",
        "",
        "| Task | Model | Dimensions | Pass rate | Strict | Cost/run | Run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r["task"], r["model"])):
        lines.append(
            f"| {r['task']} | {r['model']} | {r['pass']} | {r['rate']} "
            f"| {r['strict']} | {r['cost']} | {r['run']} |"
        )
    if not rows:
        lines.append("| — | no runs yet | — | — | — | — | — |")
    (BENCH / "leaderboard.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {BENCH / 'leaderboard.md'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
