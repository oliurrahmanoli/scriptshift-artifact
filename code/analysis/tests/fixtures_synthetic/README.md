# SYNTHETIC FIXTURE — NOT RESULTS

Everything generated from this directory is **fake data with planted effects**, used only to
test that `analysis/` computes what it claims. None of it may be quoted, plotted or summarised
in `paper/`, `DECISIONS.md`, the checklist or any pre-registration as a result (D12, D23).

* `make_fixture.py` writes, into a directory you choose (tests use a pytest tmp dir):
  AgentDojo-format trace JSONs (`runs/<SYNTH-model>/<suite>/<user_task>/<attack>/<inj>.json`,
  every file has `"synthetic": true`), `run_benchmark.py`-format scoring JSONL (header has
  `"synthetic": true`), judge JSONL, comprehension-probe JSONL (ASSUMED schema), offline detector
  CSV, human-audit CSV and validator-ratings CSV (ASSUMED schemas).
* Model names start with `SYNTH-` so they cannot be mistaken for real models.
* Planted truth (per model, shared latent draw so pairs are correlated):
  en 0.30, bn_mt 0.30, **bn_native 0.45** (C1 and C3 positive), banglish_native = banglish_translit
  = 0.30 (C2 null, built to be TOST-equivalent), codemix 0.35; W2 +0.10 for en/bn_native;
  raw L1 under-counts Group C in Bangla conditions; banking 0/1/3 raw inflated vs corrected;
  success is suppressed when the probe says "not comprehended".
* `example_out_SYNTHETIC/` = what `run_analysis` produces on this fixture (every file prefixed
  `SYNTHETIC_`, figures watermarked). It exists only to show the table layout.

Regenerate: `python analysis/tests/fixtures_synthetic/make_fixture.py /tmp/synth && python -m
analysis.run_analysis --traces /tmp/synth/runs --scores '/tmp/synth/scores/*.jsonl' ...` (see
`analysis/tests/test_analysis_pipeline.py` for the full argument list).
