#!/usr/bin/env bash
# Fast-track analysis of the real main grid (DECISIONS D38): primary rule + two sensitivity variants,
# 10,000-resample cluster bootstrap, glmer (pool = primary pool, all = incl. self-generated Gemma),
# sensitivity side-by-side. Run from code/ with the harness venv. Needs the per-model traces unzipped
# into results/main/<m>/runs/traces (unzip results/main/<m>/<m>_150_traces.zip -d results/main/<m>/runs/).
set -euo pipefail
PY=${PY:-../.venv-harness/bin/python}
OUT=${1:-../paper/results/fasttrack_v1_20261007}
NBOOT=${NBOOT:-10000}
COMMON=(--traces results/main/*/runs/traces --scores results/main/*/runs/*_150.jsonl
        --payloads data/payloads.json --pairs subset/pairs_150.json
        --pilot-results kaggle/pilot-*-v1/*_pilot_result.json
        --self-generated-models gemma-4-26b-a4b --n-boot "$NBOOT")
mkdir -p "$OUT"
$PY -m analysis.run_analysis "${COMMON[@]}" --out "$OUT/primary" --label fasttrack_v1_primary \
    > "$OUT/primary.log" 2>&1 &
$PY -m analysis.run_analysis "${COMMON[@]}" --model-error-rule exclude \
    --out "$OUT/sensitivity_model_error_excluded" --label fasttrack_v1_sens_model_error_excluded \
    > "$OUT/sens1.log" 2>&1 &
$PY -m analysis.run_analysis "${COMMON[@]}" --overflow-rule episode \
    --out "$OUT/sensitivity_overflow_episode_level" --label fasttrack_v1_sens_overflow_episode \
    > "$OUT/sens2.log" 2>&1 &
wait
for v in primary sensitivity_model_error_excluded sensitivity_overflow_episode_level; do
  test -f "$OUT/$v/MANIFEST.json"
  Rscript analysis/glmm.R "$OUT/$v/episodes_tidy.csv" "$OUT/$v/glmm_R" normalised_security pool \
      > "$OUT/$v/glmm_R.log" 2>&1
  Rscript analysis/glmm.R "$OUT/$v/episodes_tidy.csv" "$OUT/$v/glmm_R_all_incl_selfgen" normalised_security all \
      > "$OUT/$v/glmm_R_all_incl_selfgen.log" 2>&1
done
$PY -m analysis.fasttrack_summary "$OUT"
echo "done: $OUT"
