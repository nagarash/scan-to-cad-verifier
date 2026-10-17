# Scan-to-CAD Benchmark

A file-driven benchmark for scan → editable CAD reconstruction. Parts and
models are data, not code: adding a part is a new `tasks/<part>/` directory,
adding a model is a line in `models.yaml`.

## Protocol

1. **Task** (`tasks/<part>/task.json`): the input scan, the held-out
   reference CAD, the agent prompt, and the dimension table. The dimension
   table is the answer key — it is **never** shown to the generator.
2. **Run** (`runners/openrouter_runner.py --task <part> --model <id>`):
   calls the model with the scan summary + prompt, saves the returned
   CadQuery program plus token usage under `runs/<part>/<model>/<ts>/`.
3. **Measure** (`measure.py --run <run-dir>`): executes the CadQuery
   program, exports STEP, measures each dimension in the task table,
   writes `measurements.json` (dimension id → measured mm).
4. **Evaluate** (`evaluate.py --run <run-dir>`): compares measurements
   against nominal ± tolerance, writes `score.json`.
5. **Leaderboard** (`leaderboard.py`): aggregates every run into
   `leaderboard.md`.

## Score

The primary score is **dimensional pass rate**: the fraction of drawing
dimensions measured on the reconstructed solid that fall within tolerance.
Reported alongside: worst violations first (like an inspection report),
p95 surface distance vs the reference solid, strict pass/fail, and cost
per run in USD.

## Quickstart

```bash
cd ~/Desktop/cad-verifier
python benchmark/runners/openrouter_runner.py --task ctc05 --model openai/gpt-5.2
python benchmark/measure.py --run benchmark/runs/ctc05/openai__gpt-5.2/<ts>
python benchmark/evaluate.py --run benchmark/runs/ctc05/openai__gpt-5.2/<ts>
python benchmark/leaderboard.py
```

`OPENROUTER_API_KEY` must be in the environment (never commit it).
