#!/usr/bin/env bash
# D41 Unicode-preserving arm (SECONDARY, POST HOC): run ONLY after the five bangla-pi-uni-<model> Kaggle
# outputs have been downloaded into results/uni/<m>/ and their traces unzipped:
#   unzip results/uni/<m>/<m>_150_uni_traces.zip -d results/uni/<m>/runs/
#   unzip results/uni/ministral/ministral_150_stockretry_traces.zip -d results/uni/ministral/runs/
# (stock traces as for run_fasttrack_analysis.sh). Run from code/ with the harness venv.
# Not run yet: there are no Unicode-arm results. Never point it at synthetic data with an out dir under paper/.
set -euo pipefail
PY=${PY:-../.venv-harness/bin/python}
OUT=${1:-../paper/results/unicode_arm_v1}
NBOOT=${NBOOT:-10000}
for m in gemma gptoss granite ministral qwen; do
  test -f "results/uni/$m/runs/${m}_150_uni.jsonl" || { echo "missing results/uni/$m (rerun not done)"; exit 1; }
done
mkdir -p "$OUT"
$PY -m analysis.unicode_arm \
    --stock-traces results/main/*/runs/traces --stock-scores 'results/main/*/runs/*_150.jsonl' \
    --uni-traces results/uni/*/runs/traces_uni --uni-scores 'results/uni/*/runs/*_150_uni.jsonl' \
    --retry-traces results/uni/*/runs/traces_stockretry --retry-scores 'results/uni/*/runs/*_150_stockretry.jsonl' \
    --episodes-dir data/uni_rerun --tiers tier_A --self-generated-models gemma-4-26b-a4b \
    --n-boot "$NBOOT" --label unicode_arm_v1 --out "$OUT" > "$OUT.log" 2>&1
Rscript analysis/glmm.R "$OUT/episodes_tidy_unicode.csv" "$OUT/glmm_R" normalised_security pool > "$OUT/glmm_R.log" 2>&1
Rscript analysis/glmm.R "$OUT/episodes_tidy_unicode.csv" "$OUT/glmm_R_all_incl_selfgen" normalised_security all \
    > "$OUT/glmm_R_all_incl_selfgen.log" 2>&1
echo "done: $OUT (secondary, post hoc arm; primary stays the stock run)"
