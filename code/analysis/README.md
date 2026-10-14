# `analysis/`: Results pipeline (checklist S7/S8; DECISIONS D23)

**Status: prepared, not run on attack data.** No main-grid results exist yet. This code turns
traces and scoring outputs into the paper's Results tables (T1–T10, `paper/draft.md` §6) and figures F1/F2.
The only real data it has processed are the no-attack pilot traces, for the utility table T2
(`out/pilot_utility/`). Its numbers match the pilot result JSONs exactly.

## Inputs
| flag | source | format |
|---|---|---|
| `--traces` | AgentDojo trace dirs (`kaggle/pilot-*-v1/runs/`, or `run_benchmark.py --trace-dir`) | `<pipeline>/<suite>/<user_task>/<attack|none>/<inj|none>.json` |
| `--scores` | `runners/run_benchmark.py` JSONL (L1 raw / L2 normalised / corrected, `needs_judge`) | header line + one record per episode |
| `--judge` | `scoring/run_judge.py` JSONL | key + verdict `full|partial|no` |
| `--probe` | comprehension probe (**ASSUMED schema**, no runner yet) | JSONL: `model, condition, suite, injection_task, prompt_variant, q1_correct, q2_correct[, q3_correct]` |
| `--detectors` | offline detectors (**ASSUMED**) | CSV: `detector, condition, item_id, label(1=injected), score∈[0,1]` |
| `--audit` | human judge audit (**ASSUMED**) | CSV: episode key + `judge_model, judge_verdict, human_a, human_b` |
| `--ratings` | validator ratings (**ASSUMED**) | CSV: `item_key, condition, rater, fidelity, adequacy, naturalness, register` |
| `--payloads`, `--pairs`, `--pilot-results` | `data/payloads.json`, `subset/pairs.json`, `kaggle/pilot-*/*_pilot_result.json` | as in repo |

Episode key: `(model, defense, suite, user_task, injection_task, condition, rep)`. Traces and scores are
outer-joined on it. `raw_mismatch` counts disagreements between trace and JSONL. The primary metric is
**normalised (L2)** ASR (`--metric`). With no L2 available, the CLI warns and falls back to raw.

## Statistics (plan §4.7 as amended by §10.2, kept by §11)
* **Per model × condition ASR**: two-stage cluster bootstrap (injection tasks, then user tasks within
  them), 10,000 resamples, seed 20261002. Percentile and BCa 95% CIs.
* **Primary contrasts** C1 bn_native − bn_mt, C2 banglish_native − banglish_translit, C3 bn_native − en.
  Paired bootstrap difference CIs plus exact McNemar per model. **Holm** correction over the per-model
  primary family (§10.2). The pooled estimate is the primary (§10.2): pooled bootstrap rows, plus
  pooled odds ratios from a binomial **GEE** (exchangeable, clustered on injection task). A **VB GLMM**
  (statsmodels `BinomialBayesMixedGLM`, crossed random intercepts) is a cross-check.
  The pre-registered **`glmer`** fit is in `glmm.R`. It is **untested** because the box has no R, so run it on an R machine.
* **All conditions vs en**: McNemar with Holm and BH (T3 `vs_en_all`).
* **TOST** ±5 pp on C2 (§4.7). "Equivalent" when the 90% bootstrap CI lies within ±5 pp (ASSUMED
  implementation). A normal-approximation p is also reported.
* **Utility under attack** next to ASR (F1 data). **No-attack utility** with Wilson CIs, parse rate and
  the §11 floors (utility ≥ 0.20, parse errors ≤ 5%) (T2).
* **RQ2**: comprehension rate, ASR given comprehended / not comprehended, and GEE refits with comprehension
  as covariate and as condition × comprehended interaction (T6). "Comprehended" = q1 ∧ q2 under both
  prompts; the "either prompt" rule is a sensitivity check.
* `gonogo_inputs` lists the inputs to go/no-go 6a/6b only. **It never makes the decision.**

## Outputs (`--out`)
`T1_dataset`, `T2_utility_no_attack`, `T3_primary_contrasts`, `T3_pooled_or`, `T3_vs_en_all`,
`T3_gonogo_6a_6b_inputs`, `T4_undercount`, `T4_banking_precedence`, `T4_judge_audit`,
`T5_subset_vs_full`, `T5_suite_reweighted`, `T6_comprehension`, `T6_pooled_with_comprehension`,
`T7_detectors`, `T8_defences`, `T9_register`, `T10a_nllb_vs_google`, `T10b_flip_rates`,
`T10c_output_language` (CSV); `F1_asr_grid.csv`, `F1_asr_heatmap.{png,pdf}`, `F2_contrast_cis.{png,pdf}`;
`episodes_tidy.csv`, `MANIFEST.json` (inputs, seed, n_boot, versions).
A table whose inputs do not exist yet gets a single `status = PENDING …` row. Every table has a
`synthetic` column.

## Synthetic-data guard
If any input is marked synthetic, every output is prefixed `SYNTHETIC_`, figures are watermarked,
and the CLI **refuses** to write under a path containing `paper` or next to `DECISIONS.md`.
`tests/fixtures_synthetic/` has fake data with planted effects. It exists for tests only and **is not results**.

## Run
```bash
cd code
../.venv-harness/bin/python -m analysis.run_analysis --traces 'kaggle/pilot-*-v1/runs' \
  --pilot-results 'kaggle/pilot-*-v1/*_pilot_result.json' --payloads data/payloads.json --out analysis/out/pilot_utility
# main grid (after it exists):
../.venv-harness/bin/python -m analysis.run_analysis --traces runs/traces --scores 'runs/*.jsonl' \
  --judge 'runs/judge/*.jsonl' --probe runs/probe.jsonl --detectors runs/detectors.csv \
  --payloads data/payloads.json --pairs subset/pairs.json --out analysis/out/main
Rscript analysis/glmm.R analysis/out/main/episodes_tidy.csv analysis/out/main/glmm_R   # pre-registered glmer
../.venv-harness/bin/python -m pytest -q analysis/tests
```
Requires numpy, scipy, pandas, statsmodels and matplotlib.

## Fast-track run on the real main grid (DECISIONS D38–D40)
`scripts/run_fasttrack_analysis.sh [OUT]` (from `code/`; default OUT = `../paper/results/fasttrack_v1_20261007`)
runs the D38 primary rule and its two sensitivity variants with 10,000 bootstrap resamples, then `glmm.R` (scope
`pool` = primary pool without the self-generated Gemma LLM cells; `all` = incl. Gemma), then
`analysis.fasttrack_summary` (`SENSITIVITY_SUMMARY.csv`, `SENSITIVITY_MAX_ABS_DIFF.json`).
* `episode_rules.py` applies the D38.2 error rules before any table (flags `--model-error-rule failure|exclude`,
  `--overflow-rule pair|episode`, `--self-generated-models`); `T0_episode_accounting.csv` lists every exclusion.
* `T3_primary_contrasts.csv`: per-model C1/C3 (McNemar, Holm over the 10 per-model tests), pooled bootstrap over the
  primary pool and incl. self-generated; C2 is a `not_run` row. `T3_pooled_or.csv` = GEE/VB cross-checks;
  `glmm_R/T3_pooled_or_glmer.csv` = the registered fit (Holm over C1, C3) + `glmer_convergence.csv`.
* `T3_exploratory_pairs.csv`: the D38.4 exploratory family (BH over 20 tests).
* `extra_tables.py` (**exploratory, post hoc, D40**): `T11_injection_rendering.csv` (how the injection reached the
  agent: raw Bengali script / `\u` escapes / ASCII / not seen), `T11b` ASR by rendering stratum, `T11c` contrasts by
  stratum (BH), `T12` ASR by scoring group A/B/C.
* Judge-augmented ASR (`--judge results/judge/judge_*.jsonl`) counts only verdicts on §4.4-gated episodes (D39).
* `MANIFEST.json` records label, rules, code git commit (flags dirty analysis/scoring files), input sha256s and
  package versions. Outputs on real data go to `paper/results/<label>/`; synthetic inputs are still refused there.

## Unicode-preserving arm (DECISIONS D41; SECONDARY, POST HOC)
`analysis/unicode_arm.py` (run with `scripts/run_unicode_arm_analysis.sh` once `results/uni/<m>/` exist; **not run
yet**) builds the arm = stock grid with every episode listed in `data/uni_rerun/episodes_<m>.json` (tier A)
replaced by its Unicode-preserving rerun. A listed episode without a rerun is `rerun_missing`. It writes:
* U0 accounting
* U1 rendering check (`patch_ok`)
* U2/FU1 ASR grid
* U3 C1/C3 as T3
* U3b exploratory pairs
* U4 stock-vs-Unicode ASR (paired Δpp, cluster bootstrap, McNemar; all episodes and rerun-only)
* U5/FU2 C1/C3 stock vs Unicode
* U6 rendering
* `episodes_tidy_unicode.csv` (input to `glmm.R`)

All of them are labelled `secondary_post_hoc_D41`. `run_analysis.py --retry-traces/--retry-scores` folds the
Ministral stock retry (D35) into the primary set (an ok retry replaces only an errored stock row). Mock-data
test: `analysis/tests/test_unicode_arm.py`.
