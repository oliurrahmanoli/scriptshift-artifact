# Script Shift — native Bangla, Banglish and code-mixed indirect prompt injection against tool-using LLM agents

Experiment codebase for the paper *Native Bangla, Banglish and Code-Mixed Indirect
Prompt Injection Against Tool-Using LLM Agents* (AgentDojo v1.2.2).

This repository holds the **attacks, data pipeline, subset sampler, scoring, runners,
Kaggle notebooks and analysis** for the study. It implements the "no-approval" plan
of record (research plan §11): open-weight models on Kaggle 2×T4 via llama.cpp, Gemini
on Vertex, optional DeepSeek direct API. **It never calls OpenRouter, the OpenAI hosted
API or the Mistral hosted API**, which §11/§11.5 exclude on terms grounds — the runner
refuses those endpoints.

> ⚠️ **Dual-use / safety.** The payloads target AgentDojo's **fictional** environments
> and fictional identifiers only (no real systems, accounts or people). They are test
> data for a benchmark, no more capable than the English IPI payloads already public in
> AgentDojo/InjecAgent/BIPIA. Payloads live in clearly named data files, never hidden in
> this README or in any PDF.

## Licence
Code: **MIT** (see `LICENSE`), matching AgentDojo. Dataset text (when released): CC BY 4.0,
except the NLLB-generated `bn_mt` subset, which is CC BY-NC 4.0 to respect the NLLB-200
licence (§4.3).

## Layout
```
attacks/      MultilingualAttack classes; registers en, bn_mt, bn_native,
              banglish_native, banglish_translit, codemix_native (+ W2 register conds)
              with AgentDojo. Loads data/payloads.json.
data/         goals_en.json (35 goals extracted verbatim from agentdojo 0.1.35),
              payloads.json (per-condition goals + wrappers; native slots = TODO_NATIVE),
              accept_lists.json (Group C relaxed-scoring accept-lists),
              drafts_for_native_review.json (MACHINE DRAFTS ONLY — not native data),
              mt_report.json / translit_report.json (generation audits)
subset/       sampler.py + pairs.json/pairs.csv (stratified, seeded 300-pair subset)
scoring/      normalise.py, scoring.py (3-layer scorer + banking fix), judge.py (rubric + stub +
              OpenAI-compatible backend), run_judge.py, judge_config.example.json (D23)
runners/      endpoints.py (local/gemini pipelines), run_benchmark.py (JSONL + resume, --trace-dir),
              main_grid.py; unicode_patch.py + uni_rerun.py (D41 Unicode-preserving arm, flag OFF by default)
kaggle/       pilot_*.ipynb, main_grid_*.ipynb, uni_rerun_<model>.ipynb (D41, not launched), make_bn_mt_nllb.ipynb (NLLB-3.3B, D23), JUDGE_ON_KAGGLE.md
analysis/     stats.py (McNemar, cluster bootstrap, Holm/BH, TOST, comprehension) +
              smoke_test.py (synthetic-data checks), aggregate.py; run_analysis.py + modules
              -> Results tables T1–T10 and figures F1/F2 (see analysis/README.md, D23)
scripts/      extract_goals.py, build_payloads.py, make_mt.py, make_translit.py,
              translit.py, make_drafts.py, build_accept_lists.py, build_notebooks.py
tests/        unit tests (pytest)
runs/examples/ a committed smoke run (tiny local model) proving the harness end to end
DECISIONS.md  decision log (§4.10)
```

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # agent harness, scoring, analysis
# MT/transliteration data generation (separate env):
#   pip install -r requirements-mt.txt
```

## Regenerate the data pipeline
```bash
python scripts/extract_goals.py        # -> data/goals_en.json (35 goals, groups A/B/C)
python scripts/build_payloads.py       # -> data/payloads.json skeleton (en + TODO slots)
python scripts/build_accept_lists.py   # -> data/accept_lists.json
# bn_mt (NLLB): box CPU used distilled-600M; Kaggle notebook uses nllb-200-3.3B
.venv-mt/bin/python scripts/make_mt.py --model facebook/nllb-200-distilled-600M
python scripts/make_translit.py        # banglish_translit (needs bn_native; else TODO)
python scripts/make_drafts.py          # -> data/drafts_for_native_review.json (DRAFTS)
python subset/sampler.py               # -> subset/pairs.json / pairs.csv
```

## Run the benchmark
Local (llama.cpp OpenAI-compatible server on localhost):
```bash
LOCAL_LLM_PORT=8099 python runners/run_benchmark.py \
    --provider local --condition en --out runs/local_en.jsonl
```
Gemini (Vertex trial credit):
```bash
GCP_PROJECT=... GCP_LOCATION=us-central1 python runners/run_benchmark.py \
    --provider gemini --model gemini-3.8-flash --condition bn_mt --out runs/gemini_bn_mt.jsonl
```
Runs are **resumable**: re-running the same command skips completed pairs (handy after a
Kaggle 12-h session ends). On Kaggle, use `kaggle/main_grid.ipynb` (parameterised by
model and condition) and `kaggle/pilot_throughput.ipynb` first to measure throughput.

## Scoring (3 layers, §4.4)
1. **raw** — AgentDojo's own `security()` (for comparison with published numbers).
2. **normalised** (PRIMARY) — relaxed Group B/C matching so a success written in Bangla
   still scores; identifier checks stay exact. Banking tasks 0/1/3 use the corrected
   precedence predicate.
3. **judge** — `scoring/judge.py` (gating rule + fixed rubric); live pass via
   `python -m scoring.run_judge` against a cross-vendor OpenAI-compatible endpoint
   (`kaggle/JUDGE_ON_KAGGLE.md`). Prepared, not run live (D23).

## Analysis
```bash
python analysis/aggregate.py runs/*.jsonl      # per-condition raw/norm/corrected ASR
python analysis/smoke_test.py                  # synthetic-data checks of the stats
python -m analysis.run_analysis --help          # full pipeline -> T1–T10, F1/F2 (analysis/README.md)
```
`analysis/stats.py` provides exact McNemar, injection-task cluster bootstrap CIs,
Holm–Bonferroni / Benjamini–Hochberg correction, comprehension-conditional ASR and TOST.

## Tests
```bash
python -m pytest tests/ analysis/tests -q   # 50 tests: attacks, scoring, subset, judge (+ mocked endpoint), analysis
python analysis/smoke_test.py     # 4 synthetic analysis checks
```

## What still needs humans / accounts
See `DECISIONS.md`. In short: native authoring + validation of
`bn_native`/`banglish_native`/`codemix_native` (non-delegable, native speakers only);
a Kaggle account for the GPU grid; a Google Cloud account for Gemini/Google NMT; and
the authors' calls on Gemini route, DeepSeek, and the AgentDojo banking-bug issue.
