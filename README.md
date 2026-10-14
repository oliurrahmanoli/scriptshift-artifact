# Script Shift — artifact

Code and data accompanying the paper **"Script Shift: LLM-Written Bangla and Banglish Indirect Prompt Injection Against Tool-Using LLM Agents"** by Oliur Rahman Oli, Nuha Fadilah Zahidy and Zabedur Rahman (arXiv: TBD). The study covers
indirect prompt injection written in **Bangla, Banglish (romanised Bangla) and code-mixed Bangla–English**
against tool-using LLM agents on **AgentDojo** (package 0.1.35, benchmark suites v1.2.2, no defence).

It contains the attack implementation, injection templates and frozen payloads, the stratified
150-pair subset, the 3-layer scorer and LLM judge, the benchmark runners and Kaggle notebooks, the
raw per-episode results (scores + AgentDojo traces), the analysis pipeline (Python + `glmm.R`) and the
generated tables/figures behind every number in the paper (`paper/RESULTS_SUMMARY.md`).

> **Dual-use note.** Payloads target AgentDojo's *fictional* environments and identifiers only
> (all e-mail addresses / names in traces are AgentDojo fixtures). They are no more capable than
> the English IPI payloads already public in AgentDojo, InjecAgent and BIPIA.

## Layout
```
code/
  attacks/        MultilingualAttack classes registered with AgentDojo (en, bn_mt, bn_llm, banglish_llm, codemix_llm, *_W2, ...)
  payloads/       LLM payload generator + prompt templates (payloads/prompts/*.txt)
  data/           goals_en.json, payloads.json (frozen), accept_lists.json, conditions/ (per-condition
                  payload dirs, wrappers, SHA256 sums, _provenance/ with raw generator outputs), uni_rerun/
  subset/         seeded stratified sampler + pairs_150.{json,csv}
  scoring/        normalisation, 3-layer scorer, judge rubric / runner, judge configs (no keys)
  runners/        endpoints, run_benchmark.py, main_grid.py, Unicode-preserving arm (unicode_patch.py, uni_rerun.py)
  kaggle/         notebooks actually run (pilots, main grid per model, Unicode rerun, NLLB MT, judges) + pilot outputs
  scripts/        data-pipeline builders and the three analysis drivers (run_*_analysis.sh)
  analysis/       run_analysis.py, stats/bootstrap/tables/figures, unicode_arm.py, glmm.R, tests
  results/        raw results: main/<model>/ (150 pairs x 4 conditions + 97 no-attack), uni/<model>/
                  (Unicode-preserving rerun), judge/ (Qwen + Gemma judge verdicts). Each has SHA256SUMS.txt
                  and SOURCE.txt (Kaggle output provenance). AgentDojo traces ship zipped (*_traces.zip).
  tests/          pytest unit tests
  runs/examples/  tiny committed smoke run
paper/
  RESULTS_SUMMARY.md                 auto-generated summary; every number points at a CSV below
  results/fasttrack_v1_20261007/     primary analysis + 2 sensitivity variants (tables T0–T12, F1, F2, glmer)
  results/judge_aug_v1/              judge-augmented (layer 3) analysis
  results/unicode_arm_v1/            Unicode-preserving rerun arm (tables U1–U6, glmer)
  agentdojo_issue_repros/            minimal repros of two AgentDojo issues discussed in the paper
```

## Setup
Python 3.13 (3.11+ should work) and R ≥ 4.3.
```bash
cd code
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # agentdojo==0.1.35, numpy, scipy, pandas, statsmodels, matplotlib, pytest, ...
Rscript -e 'install.packages(c("lme4","emmeans","geepack"))'
# only for regenerating machine-translated payloads:
pip install -r requirements-mt.txt       # torch (CPU), transformers<5, sentencepiece, ...
```
The analysis drivers default to `PY=../.venv-harness/bin/python`; override with `PY=python` (or the
venv's python) as shown below. All commands are run from `code/`.

Sanity check: `python -m pytest tests/ analysis/tests -q && python analysis/smoke_test.py`

## Reproducing the tables and figures (from the shipped raw results; CPU only, minutes)
First unpack the AgentDojo traces (from `code/`):
```bash
for m in gemma gptoss granite ministral qwen; do
  unzip -oq results/main/$m/${m}_150_traces.zip     -d results/main/$m/runs/traces
  unzip -oq results/uni/$m/${m}_150_uni_traces.zip  -d results/uni/$m/runs/traces_uni
done
```

| Paper artefact | Command (from `code/`) | Output |
|---|---|---|
| Episode accounting, ASR grid, utility, primary contrasts C1/C3, exploratory pairs, rendering strata (T0–T12), Fig. F1 heatmap, Fig. F2 contrast CIs, pooled `glmer` ORs, sensitivity variants | `PY=python scripts/run_fasttrack_analysis.sh` | `../paper/results/fasttrack_v1_20261007/{primary,sensitivity_*}/` |
| Judge-augmented ASR (layer 3) | `PY=python scripts/run_judge_augmented_analysis.sh` | `../paper/results/judge_aug_v1/` |
| Unicode-preserving rerun arm (U1–U6) | `PY=python scripts/run_unicode_arm_analysis.sh` | `../paper/results/unicode_arm_v1/` |
| `paper/RESULTS_SUMMARY.md` | `python -m analysis.write_results_summary ../paper/results/fasttrack_v1_20261007 ../paper/RESULTS_SUMMARY.md` | summary markdown |
| `glmer` only (any variant) | `Rscript analysis/glmm.R <dir>/episodes_tidy.csv <dir>/glmm_R normalised_security pool` (`all` = incl. self-generated Gemma) | `T3_pooled_or_glmer.csv`, convergence, summary |

Bootstrap: 10,000 two-stage cluster resamples, seed 20261002 (set `NBOOT=1000` for a quick run; numbers
then differ in the CI decimals only). Running a driver overwrites the shipped outputs in `paper/results/`;
diff against git to verify.

## Re-running the experiments (GPU)
1. Payloads (already frozen in `data/`; regenerate only to audit): `python scripts/extract_goals.py`,
   `scripts/build_payloads.py`, `kaggle/gen_llm_payloads.ipynb` (Gemma generator) and
   `kaggle/make_bn_mt_nllb.ipynb` (NLLB-200 3.3B), then `scripts/freeze_payloads.py` /
   `scripts/check_payload_freeze.py` (verifies `data/conditions/PAYLOAD_SHA256SUMS.txt`).
2. Subset: `python subset/sampler.py` (seeded) → `subset/pairs_150.json`.
3. Main grid: one notebook per model, `kaggle/main_grid_<model>.ipynb` (serves the GGUF with llama.cpp and
   calls `runners/main_grid.py`; resumable). Locally with any OpenAI-compatible server:
   `LOCAL_LLM_PORT=8099 python runners/run_benchmark.py --provider local --condition en --out runs/local_en.jsonl --trace-dir runs/traces`
4. Unicode-preserving arm: `kaggle/uni_rerun_<model>.ipynb` (`runners/uni_rerun.py`).
5. Judge: `kaggle/judge_qwen.ipynb`, `kaggle/judge_gemma.ipynb` (`python -m scoring.run_judge`; copy
   `scoring/judge_config.example.json` and supply your own endpoint — no keys are shipped).

## Hardware
* Agent runs, judges, payload generation and NLLB MT: Kaggle notebooks, 2× NVIDIA T4 (16 GB each),
  llama.cpp build b11344 (OpenAI-compatible server), ~12 h sessions, resumable.
* Analysis: a commodity 8-core x86-64 Linux CPU machine, no GPU; full pipeline ≈ minutes.

## Models (all open weights, 4-bit GGUF)
Targets: gpt-oss-20b (MXFP4), Ministral-3-14B, Granite-4.2-8B, Qwen3.6-35B-A3B (Q4_K_M),
Gemma-4-26B-A4B (QAT Q4_0). Payload generator: Gemma-4-26B-A4B (its own `bn_llm`/`banglish_llm` cells are
flagged self-generated and excluded from the pooled primary). Judges: Qwen3.6-35B-A3B and Gemma-4-26B-A4B
(cross-vendor gating). MT: facebook/nllb-200-3.3B (box smoke tests used nllb-200-distilled-600M).
Weights are not redistributed; download them from their official sources under their own licences.

## Licences
* **Code**: MIT (`LICENSE`).
* **Data** (payload text, result files, tables): CC BY 4.0, **except** the NLLB-generated `bn_mt`
  condition, which is CC BY-NC 4.0 to respect the NLLB-200 licence (CC BY-NC 4.0).
* **AgentDojo** (environments, tasks, fixtures appearing inside traces): MIT, © the AgentDojo authors;
  used as an unmodified pip dependency (agentdojo==0.1.35, suites v1.2.2).
* **Models**: each under its own licence — gpt-oss (Apache 2.0), Granite (Apache 2.0), Qwen (Apache 2.0),
  Ministral / Mistral (see Mistral AI's licence for the specific release), Gemma (Gemma Terms of Use),
  NLLB-200 (CC BY-NC 4.0). Model outputs in `results/` are provided for research reproducibility.

## Not included (available on request)
* No individual raw file exceeded 50 MB, so no raw logs were dropped for size.
* The embedded harness snapshots `results/*/<model>/code/` that Kaggle bundled with each output
  (identical to `code/` at the run commit; still listed in each `SHA256SUMS.txt`).
* Judge output bundles `results/judge/*/judge_*_outputs.zip` (incl. llama.cpp server logs, ~5 MB raw);
  the verdicts are shipped as `results/judge/judge_*.jsonl`.
* Paper text/LaTeX, drafts, planning notes and the internal decision log (referenced in comments as
  "D<n>" / "research-plan §n"); `data/drafts_for_native_review.json` (unused machine drafts).
* Python virtual environments, caches and pre-freeze data backups.

## Citation
If you use this artifact, please cite the paper (arXiv link TBD; see `CITATION.cff`):

```bibtex
@misc{oli2026scriptshift,
  author = {Oli, Oliur Rahman and Nuha Fadilah Zahidy and Zabedur Rahman},
  title  = {Script Shift: LLM-Written Bangla and Banglish Indirect Prompt Injection Against Tool-Using LLM Agents},
  year   = {2026},
  note   = {arXiv preprint (link TBD)},
  url    = {https://github.com/oliurrahmanoli/scriptshift-artifact}
}
```
