"""Run one generator model against one benchmark task.

Saves the returned CadQuery program plus token usage under
benchmark/runs/<part>/<model>/<utc-timestamp>/.

The generator sees the scan summary and the task prompt only. The
dimension table (the answer key) is never included.
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
BENCH = Path(__file__).resolve().parents[1]        # benchmark/
API = "https://openrouter.ai/api/v1/chat/completions"
MODELS_API = "https://openrouter.ai/api/v1/models"


def load_task(part: str) -> dict:
    p = BENCH / "tasks" / part / "task.json"
    if not p.exists():
        sys.exit(f"unknown task: {part} ({p} missing)")
    return json.loads(p.read_text())


def scan_summary(task: dict) -> str:
    inp = task["inputs"]
    lines = [
        f"Scan file: {inp.get('scan')}",
        f"Notes: {inp.get('scan_notes')}",
    ]
    return "\n".join(lines)


def fetch_pricing(model_id: str):
    """Return (prompt_usd_per_1m, completion_usd_per_1m) or (None, None)."""
    try:
        req = urllib.request.Request(MODELS_API, headers={"User-Agent": "cad-verifier-bench"})
        data = json.load(urllib.request.urlopen(req, timeout=30))
        for m in data.get("data", []):
            if m.get("id") == model_id:
                pr = m.get("pricing", {})
                return float(pr.get("prompt", 0) or 0), float(pr.get("completion", 0) or 0)
    except Exception as e:  # pricing is nice-to-have; never fail the run
        print(f"warning: could not fetch pricing: {e}", file=sys.stderr)
    return None, None


def call_openrouter(model_id: str, prompt: str, max_tokens: int):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    body = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a mechanical CAD reconstruction agent. Output only the requested CadQuery program."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nagarash/scan-to-cad-verifier",
            "X-Title": "cad-verifier benchmark",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task id, e.g. ctc05")
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-5.2")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--runs-dir", default=str(BENCH / "runs"))
    args = ap.parse_args()

    if args.model.startswith("local/"):
        sys.exit(
            f"{args.model} is a local-weights model: restore third_party/cad_recode "
            "and scripts/download_recode.py, run it on this machine, and drop the "
            "CadQuery program plus a usage.json with {\"gpu_hours\": …} into the run dir."
        )

    task = load_task(args.task)
    full_prompt = task["prompt"] + "\n\nScan information:\n" + scan_summary(task)

    print(f"task={args.task} model={args.model} …", flush=True)
    resp = call_openrouter(args.model, full_prompt, args.max_tokens)
    choice = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    pp, pc = fetch_pricing(args.model)
    cost_usd = (pt * (pp or 0) + ct * (pc or 0)) / 1e6 if pp is not None else None

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / args.task / args.model.replace("/", "__") / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reconstruct.py").write_text(extract_code(choice))
    (run_dir / "prompt.txt").write_text(full_prompt)
    (run_dir / "usage.json").write_text(json.dumps({
        "model": args.model,
        "task": args.task,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "prompt_usd_per_1m": pp,
        "completion_usd_per_1m": pc,
        "cost_usd": cost_usd,
    }, indent=2) + "\n")
    print(f"saved run to {run_dir}")
    if cost_usd is not None:
        print(f"estimated cost: ${cost_usd:.4f} ({pt} in / {ct} out tokens)")


if __name__ == "__main__":
    main()
