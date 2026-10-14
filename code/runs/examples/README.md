# Example smoke runs (committed)

These JSONL files are REAL runs produced on the box to prove the harness works
end to end. They are NOT scientific results. Model: `Qwen3-1.7B-Q8_0` served with
llama.cpp (build b11344) on **CPU**, banking suite only.

* `smoke_en.jsonl` — 5 banking pairs, condition `en`. Note `raw_security` vs
  `corrected_security`: raw ASR 60% but corrected/normalised 0%. This is the banking
  precedence over-count (research plan §4.4) captured live — a tiny model sent money to
  the attacker IBAN with unrelated subjects, which the raw AgentDojo check counts as a
  success and the corrected predicate does not.
* `smoke_bn_mt.jsonl` — 1 banking pair, condition `bn_mt` (NLLB Bangla injection),
  proving the Bangla attack → agent → 3-layer scoring path runs.

Reproduce: serve a small GGUF with llama.cpp, then
`LOCAL_LLM_PORT=8099 python runners/run_benchmark.py --provider local --condition en \
 --pairs subset/pairs.json --out runs/myrun.jsonl`.
