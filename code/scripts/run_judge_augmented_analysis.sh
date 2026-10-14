#!/usr/bin/env bash
# D39 layer-3 judge-augmented scoring of the main grid (stock run, D38 rules): judge-augmented ASR =
# layer-2 normalised success OR judge verdict "full" on a gated episode (analysis/ingest.py). Run only after
# both judge passes are in results/judge/judge_{qwen,gemma}.jsonl (D49, D50). Run from code/ with the harness venv.
#   judge_metric/        — every table/contrast on judge_aug_security (T4 judge-gain + L2-vs-judge agreement too)
#   normalised_w_judge/  — the primary metric with judge verdicts joined (sanity: must reproduce fasttrack primary)
set -euo pipefail
PY=${PY:-../.venv-harness/bin/python}
OUT=${1:-../paper/results/judge_aug_v1}
NBOOT=${NBOOT:-10000}
for j in qwen gemma; do test -f "results/judge/judge_$j.jsonl" || { echo "missing judge_$j"; exit 1; }; done
COMMON=(--traces results/main/*/runs/traces --scores results/main/*/runs/*_150.jsonl
        --judge results/judge/judge_*.jsonl
        --payloads data/payloads.json --pairs subset/pairs_150.json
        --pilot-results kaggle/pilot-*-v1/*_pilot_result.json
        --self-generated-models gemma-4-26b-a4b --n-boot "$NBOOT")
mkdir -p "$OUT"
$PY -m analysis.run_analysis "${COMMON[@]}" --metric judge_aug_security --out "$OUT/judge_metric" \
    --label judge_aug_v1 > "$OUT/judge_metric.log" 2>&1 &
$PY -m analysis.run_analysis "${COMMON[@]}" --out "$OUT/normalised_w_judge" \
    --label judge_aug_v1_normalised_check > "$OUT/normalised_w_judge.log" 2>&1 &
wait
for v in judge_metric normalised_w_judge; do test -f "$OUT/$v/MANIFEST.json"; done
Rscript analysis/glmm.R "$OUT/judge_metric/episodes_tidy.csv" "$OUT/judge_metric/glmm_R" judge_aug_security pool \
    > "$OUT/judge_metric/glmm_R.log" 2>&1
Rscript analysis/glmm.R "$OUT/judge_metric/episodes_tidy.csv" "$OUT/judge_metric/glmm_R_all_incl_selfgen" \
    judge_aug_security all > "$OUT/judge_metric/glmm_R_all_incl_selfgen.log" 2>&1
echo "done: $OUT"
