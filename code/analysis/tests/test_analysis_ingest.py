"""Ingest tests on the REAL pilot no-attack traces (code/kaggle/pilot-*-v1/runs) + small cases."""
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.ingest import (build_episodes, episode_has_parse_error, load_probe, load_scores,
                             load_traces, normalise_model_label, split_defense)

KAGGLE = Path(__file__).resolve().parents[2] / "kaggle"
PILOT_DIRS = sorted(str(p) for p in KAGGLE.glob("pilot-*-v1/runs"))


def test_parse_error_strict_d18():
    assert not episode_has_parse_error({"error": None, "messages": [
        {"role": "tool", "content": [{"type": "text", "content": "error: file not found"}], "error": None}]})
    assert episode_has_parse_error({"error": None, "messages": [
        {"role": "tool", "content": [], "error": "EmptyFunctionCall: no name"}]})
    assert episode_has_parse_error({"error": "Could not parse function call", "messages": []})


def test_model_labels():
    assert normalise_model_label("local/Qwen3.6-35B-A3B-Q4_K_M.gguf") == "Qwen3.6-35B-A3B-Q4_K_M"
    assert normalise_model_label("local_Qwen3.6-35B-A3B-Q4_K_M.gguf") == "Qwen3.6-35B-A3B-Q4_K_M"
    assert normalise_model_label("x", {"x": "qwen3.6-35b-a3b"}) == "qwen3.6-35b-a3b"
    assert split_defense("gemma-4-26b-a4b-tool_filter") == ("gemma-4-26b-a4b", "tool_filter")
    assert split_defense("gemma-4-26b-a4b") == ("gemma-4-26b-a4b", "none")


@pytest.mark.skipif(not PILOT_DIRS, reason="pilot traces not present")
def test_load_real_pilot_traces():
    t = load_traces(PILOT_DIRS)
    assert len(t) == 37 * len(t["model"].unique())          # 37 episodes per pilot label
    assert set(t["condition"]) == {"none"}                   # no-attack pilots only
    assert set(t["suite"]) == {"banking", "slack"}
    assert not t["synthetic"].any()
    assert t["raw_security_trace"].isna().all()              # security is meaningless without attack


def test_merge_scores_and_traces(tmp_path):
    run = tmp_path / "runs" / "local_m.gguf" / "banking" / "user_task_0" / "en"
    run.mkdir(parents=True)
    (run / "injection_task_0.json").write_text(json.dumps({
        "suite_name": "banking", "pipeline_name": "local_m.gguf", "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0", "attack_type": "en", "messages": [], "error": None,
        "utility": True, "security": False}))
    sj = tmp_path / "s.jsonl"
    sj.write_text("\n".join([json.dumps({"_header": True, "pipeline": "local/m.gguf", "condition": "en"}),
                             json.dumps({"suite": "banking", "user_task": "user_task_0",
                                         "injection_task": "injection_task_0", "condition": "en",
                                         "status": "ok", "utility": True, "raw_security": False,
                                         "normalised_security": True, "corrected_security": True})]))
    df = build_episodes(load_traces(tmp_path / "runs"), load_scores(sj))
    assert len(df) == 1 and df.attrs["raw_mismatch"] == 0
    r = df.iloc[0]
    assert r["model"] == "m" and bool(r["normalised_security"]) and not bool(r["raw_security"])
    assert bool(r["judge_aug_security"])


def test_probe_rule_both_vs_either(tmp_path):
    p = tmp_path / "probe.jsonl"
    rows = [dict(model="m", condition="en", suite="s", injection_task="i", prompt_variant=v,
                 q1_correct=q1, q2_correct=True) for v, q1 in (("p1", True), ("p2", False))]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = load_probe(p).iloc[0]
    assert not out["comprehended_both"] and out["comprehended_either"]
