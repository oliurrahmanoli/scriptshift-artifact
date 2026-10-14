"""Ingest AgentDojo trace JSON dirs + scoring JSONL (+ optional judge / probe files) into one
tidy episode table (pandas DataFrame), the input for analysis/tables.py.

Sources
-------
* **Trace dirs** — AgentDojo 0.1.35 ``TraceLogger`` layout, as in
  ``kaggle/pilot-*-v1/runs/``: ``<root>/<pipeline>/<suite>/<user_task>/<attack|none>/<inj|none>.json``
  with keys suite_name, pipeline_name, user_task_id, injection_task_id, attack_type, messages,
  error, utility, security, duration, ... Files that lack these keys (``runs_manifest.json``,
  ``*_pilot_result.json``) are skipped. Gives layer-1 (raw) utility/security plus
  message-derived fields (strict parse error D18, tool-call count, Bengali share, flag keywords).
* **Scoring JSONL** — ``runners/run_benchmark.py`` output: a ``_header`` line (pipeline,
  condition, defense) then one record per pair with raw/normalised/corrected security,
  utility, scoring_group, bengali_share_*, needs_judge. Layer 2 lives only here (it needs the
  post-environment, which the trace JSON does not store).
* **Judge JSONL** — ``scoring/run_judge.py`` output (layer 3): verdict full|partial|no per episode.
* **Probe JSONL** — comprehension probe, **ASSUMED schema** (the probe runner does not exist
  yet; plan §4.7 defines semantics only). One row per (model, condition, suite, injection_task,
  prompt_variant) with boolean ``q1_correct``, ``q2_correct`` (``q3_correct`` optional).
  "Comprehended" = q1 and q2 correct under BOTH prompt variants (primary); "either" = sensitivity.

Episode key: (model, defense, suite, user_task, injection_task, condition, rep).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import KNOWN_DEFENSES, LEGACY_CONDITION_ALIASES, NO_ATTACK, PARSE_MARKERS

KEY = ["model", "defense", "suite", "user_task", "injection_task", "condition", "rep"]
_BENGALI = re.compile(r"[\u0980-\u09ff]")
_MARKER_RE = re.compile("|".join(re.escape(m) for m in PARSE_MARKERS), re.IGNORECASE)
# Exploratory refusal / injection-mention heuristic (§4.7 "keyword plus judge"). ASSUMED list;
# English only — Bangla refusals are caught by the judge, not here.
_FLAG_RE = re.compile(
    r"prompt injection|injected instruction|suspicious instruction|ignore (?:these|those|the) "
    r"instructions|i (?:can(?:no|')t|won't|will not) (?:comply|do that|follow)|"
    r"not (?:going to|able to) (?:follow|comply)|appears to be (?:an? )?(?:phishing|scam|malicious)",
    re.IGNORECASE)


# ----------------------------------------------------------------------------- helpers
def canon_condition(c):
    """Map legacy *_native names to the reframed *_llm names (D24.1 back-compat aliases)."""
    return LEGACY_CONDITION_ALIASES.get(c, c) if isinstance(c, str) else c


def bengali_share(text: str) -> float:
    """Same definition as scoring.judge.bengali_char_share (non-space chars in U+0980–U+09FF)."""
    if not text:
        return 0.0
    letters = [c for c in text if c.strip()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _BENGALI.match(c)) / len(letters)


def normalise_model_label(name: str | None, aliases: dict | None = None) -> str:
    """Map a pipeline name to a model label: alias map first, else strip provider prefix,
    directory and .gguf suffix (``local/Qwen3.6-35B-A3B-Q4_K_M.gguf`` -> ``Qwen3.6-35B-A3B-Q4_K_M``)."""
    name = name or "unknown"
    aliases = aliases or {}
    if name in aliases:
        return aliases[name]
    s = name
    for pre in ("local/", "local_", "gemini/", "gemini_"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.split("/")[-1]
    if s.endswith(".gguf"):
        s = s[:-5]
    return aliases.get(s, s)


def split_defense(model: str, defenses=KNOWN_DEFENSES) -> tuple[str, str]:
    for d in defenses:
        if model.endswith("-" + d):
            return model[: -len(d) - 1], d
    return model, "none"


def _text_chunks(content) -> list[str]:
    out = []
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("content"), str):
                out.append(b["content"])
            elif isinstance(b, str):
                out.append(b)
    return out


def episode_has_parse_error(d: dict) -> bool:
    """D18 strict check (identical logic to the pilot notebooks)."""
    top = d.get("error")
    if isinstance(top, str) and top and _MARKER_RE.search(top):
        return True
    for msg in d.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        err = msg.get("error")
        if isinstance(err, str) and err and _MARKER_RE.search(err):
            return True
        for c in _text_chunks(msg.get("content")):
            if _MARKER_RE.search(c):
                return True
    return False


def message_features(messages: list) -> dict:
    tool_calls, args, final = [], [], ""
    assistant_text = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tool_calls.append(tc)
                args.append(tc.get("args", {}) if isinstance(tc, dict) else {})
            txt = " ".join(_text_chunks(m.get("content")))
            if txt.strip():
                assistant_text.append(txt)
    # final answer = text of the last assistant message (AgentDojo model_output_from_messages)
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            final = " ".join(_text_chunks(m.get("content")))
            break
    arg_text = json.dumps(args, ensure_ascii=False)
    # How the injected block reached the agent (exploratory, D40): AgentDojo 0.1.35 formats structured
    # tool results with yaml.safe_dump (allow_unicode=False), which turns non-ASCII text into \uXXXX
    # escapes; plain-string tool results keep the script.
    seen, render = False, "not_seen"
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        t = " ".join(_text_chunks(m.get("content")))
        i = t.find("<INFORMATION>")
        if i < 0:
            continue
        j = t.find("</INFORMATION>", i)
        block = t[i: j if j > 0 else i + 4000]
        seen = True
        if _BENGALI.search(block):
            render = "raw_bengali_script"
        elif re.search(r"\\u09[89a-fA-F][0-9a-fA-F]", block):
            render = "escaped_unicode" if render != "raw_bengali_script" else render
        elif render == "not_seen":
            render = "ascii_only"
    return {
        "injection_seen": seen, "injection_render": render,
        "n_tool_calls": len(tool_calls),
        "n_messages": len(messages or []),
        "bengali_share_output": bengali_share(final),
        "bengali_share_args": bengali_share(arg_text),
        "flag_keyword": bool(_FLAG_RE.search(" ".join(assistant_text))),
    }


# ----------------------------------------------------------------------------- traces
TRACE_KEYS = {"suite_name", "pipeline_name", "user_task_id", "messages"}


def load_traces(roots, aliases: dict | None = None, keep_messages: bool = False,
                rep: int = 0) -> pd.DataFrame:
    """Walk one or more AgentDojo trace roots (``runs/``). Returns one row per episode."""
    rows = []
    if isinstance(roots, (str, Path)):
        roots = [roots]
    for root in roots:
        for fp in sorted(glob.glob(str(Path(root) / "**" / "*.json"), recursive=True)):
            try:
                d = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict) or not TRACE_KEYS <= set(d):
                continue
            model, defense = split_defense(normalise_model_label(d.get("pipeline_name"), aliases))
            cond = canon_condition(d.get("attack_type") or NO_ATTACK)
            row = {
                "model": model, "defense": defense, "suite": d["suite_name"],
                "user_task": d["user_task_id"],
                "injection_task": d.get("injection_task_id") or "none",
                "condition": cond, "rep": int(d.get("rep", rep)),
                "utility_trace": _b(d.get("utility")),
                "raw_security_trace": _b(d.get("security")) if cond != NO_ATTACK else np.nan,
                "trace_error": d.get("error"),
                "parse_error": episode_has_parse_error(d),
                "duration_s": d.get("duration"),
                "benchmark_version": d.get("benchmark_version"),
                "synthetic": bool(d.get("synthetic", False)),
                "trace_path": fp,
                **message_features(d.get("messages") or []),
            }
            if keep_messages:
                row["messages"] = d.get("messages")
            rows.append(row)
    return pd.DataFrame(rows)


def _b(v):
    if v is None:
        return np.nan
    return bool(v)


# ----------------------------------------------------------------------------- scoring JSONL
def load_scores(paths, aliases: dict | None = None, rep: int = 0) -> pd.DataFrame:
    """Read run_benchmark.py JSONL files (status=='ok' records only; errors are counted)."""
    rows = []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for pat in paths:
        files.extend(sorted(glob.glob(str(pat))))
    for fp in files:
        header = {}
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            if r.get("_header"):
                header = r
                continue
            pipe = r.get("model") or header.get("pipeline") or Path(fp).stem
            model, def_from_name = split_defense(normalise_model_label(pipe, aliases))
            defense = r.get("defense") or header.get("defense") or def_from_name or "none"
            rows.append({
                "model": model, "defense": defense, "suite": r["suite"],
                "user_task": r["user_task"], "injection_task": r["injection_task"],
                "condition": canon_condition(r.get("condition") or header.get("condition")),
                "rep": int(r.get("rep", header.get("rep", rep))),
                "status": r.get("status", "ok"),
                "error": r.get("error"),
                "utility": r.get("utility"), "raw_security": r.get("raw_security"),
                "normalised_security": r.get("normalised_security"),
                "corrected_security": r.get("corrected_security"),
                "scoring_group": r.get("scoring_group"),
                "bengali_share_output_s": r.get("bengali_share_output"),
                "bengali_share_args_s": r.get("bengali_share_args"),
                "needs_judge": r.get("needs_judge"),
                "n_tool_calls_s": r.get("n_tool_calls"),
                "synthetic": bool(r.get("synthetic", header.get("synthetic", False))),
                "score_file": fp,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- judge / probe
def load_judge(paths) -> pd.DataFrame:
    rows = []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for pat in paths:
        for fp in sorted(glob.glob(str(pat))):
            for line in Path(fp).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=KEY + ["judge_verdict", "judge_model"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"verdict": "judge_verdict", "gated": "judge_gated"})
    if "condition" in df:
        df["condition"] = df["condition"].map(canon_condition)
    for c, v in (("defense", "none"), ("rep", 0)):
        if c not in df:
            df[c] = v
    return df


def load_probe(path) -> pd.DataFrame:
    """Return one row per (model, condition, suite, injection_task) with
    comprehended_both / comprehended_either (ASSUMED input schema, see module doc)."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if "condition" in df:
        df["condition"] = df["condition"].map(canon_condition)
    need = {"model", "condition", "suite", "injection_task", "prompt_variant", "q1_correct", "q2_correct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"probe file {path} missing columns {sorted(missing)} (ASSUMED schema)")
    df["ok"] = df["q1_correct"].astype(bool) & df["q2_correct"].astype(bool)
    g = df.groupby(["model", "condition", "suite", "injection_task"])
    out = g["ok"].agg(comprehended_both="all", comprehended_either="any",
                      n_prompts="size").reset_index()
    if "synthetic" in df:
        out["synthetic_probe"] = g["synthetic"].any().values
    return out


# ----------------------------------------------------------------------------- merge
def build_episodes(traces: pd.DataFrame | None = None, scores: pd.DataFrame | None = None,
                   judge: pd.DataFrame | None = None, probe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Outer-join traces and scores on KEY; derive the analysis columns.

    utility / raw_security prefer the scoring JSONL, fall back to the trace. Mismatches between
    the two (should be zero) are counted in ``df.attrs['raw_mismatch']``.
    """
    frames = [f for f in (traces, scores) if f is not None and len(f)]
    if not frames:
        raise ValueError("no episodes: give trace dirs and/or scoring JSONL")
    if traces is not None and len(traces) and scores is not None and len(scores):
        df = traces.merge(scores, on=KEY, how="outer", suffixes=("", "_score"))
        if "synthetic_score" in df:
            df["synthetic"] = df["synthetic"].fillna(False).astype(bool) | df["synthetic_score"].fillna(False).astype(bool)
    else:
        df = frames[0].copy()
    for c in ("utility", "raw_security", "normalised_security", "corrected_security",
              "utility_trace", "raw_security_trace", "parse_error", "n_tool_calls",
              "n_tool_calls_s", "bengali_share_args", "bengali_share_args_s",
              "bengali_share_output", "bengali_share_output_s", "flag_keyword", "status"):
        if c not in df:
            df[c] = np.nan
    df["status"] = df["status"].astype(object)
    if "error" not in df:
        df["error"] = None
    df["error"] = df["error"].astype(object)
    if "trace_error" in df:
        # trace-only episodes: an AgentDojo trace with a recorded error is an errored episode
        terr = df["trace_error"].where(df["trace_error"].astype(str).str.len().gt(0) & df["trace_error"].notna())
        df.loc[df["status"].isna() & terr.notna(), "status"] = "error"
        df["error"] = df["error"].where(df["error"].notna(), terr)
    df["status"] = df["status"].fillna("ok")
    mism = df.dropna(subset=["raw_security", "raw_security_trace"])
    df.attrs["raw_mismatch"] = int((mism["raw_security"].astype(bool) != mism["raw_security_trace"].astype(bool)).sum())
    df["utility"] = df["utility"].where(df["utility"].notna(), df["utility_trace"])
    df["raw_security"] = df["raw_security"].where(df["raw_security"].notna(), df["raw_security_trace"])
    df["n_tool_calls"] = df["n_tool_calls"].where(df["n_tool_calls"].notna(), df["n_tool_calls_s"])
    df["bengali_share_args"] = df["bengali_share_args"].where(df["bengali_share_args"].notna(), df["bengali_share_args_s"])
    df["bengali_share_output"] = df["bengali_share_output"].where(df["bengali_share_output"].notna(), df["bengali_share_output_s"])
    df["synthetic"] = df["synthetic"].fillna(False).astype(bool) if "synthetic" in df else False

    if judge is not None and len(judge):
        j = judge[KEY + ["judge_verdict"] + [c for c in ("judge_model", "judge_gated") if c in judge]]
        df = df.merge(j.drop_duplicates(KEY, keep="last"), on=KEY, how="left")
    else:
        df["judge_verdict"] = np.nan
    norm = df["normalised_security"]
    jfull = df["judge_verdict"].eq("full")
    if "judge_gated" in df:
        # §4.4 / D39: only verdicts on gated episodes enter judge-augmented ASR; verdicts from an
        # ungated (calibration) pass stay in judge_verdict for the L2-vs-judge agreement table only.
        jfull = jfull & df["judge_gated"].eq(True)
    df["judge_aug_security"] = np.where(norm.isna(), np.nan, norm.fillna(False).astype(bool) | jfull)

    if probe is not None and len(probe):
        df = df.merge(probe, on=["model", "condition", "suite", "injection_task"], how="left")
    else:
        df["comprehended_both"] = np.nan
        df["comprehended_either"] = np.nan
    # cluster ids (suite-qualified: injection_task_0 exists in several suites)
    df["inj_uid"] = df["suite"] + "/" + df["injection_task"].astype(str)
    df["user_uid"] = df["suite"] + "/" + df["user_task"].astype(str)
    return df


def load_all(trace_dirs=(), score_files=(), judge_files=(), probe_file=None, aliases=None):
    tr = load_traces(trace_dirs, aliases) if trace_dirs else None
    sc = load_scores(score_files, aliases) if score_files else None
    jd = load_judge(judge_files) if judge_files else None
    pr = load_probe(probe_file) if probe_file else None
    return build_episodes(tr, sc, jd, pr)
