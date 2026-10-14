"""Main-grid driver used by the self-contained Kaggle notebooks (DECISIONS D30).

Everything the notebook does after the GPU/serve cells lives here so it is unit-tested on the
box (tests/test_main_grid.py) and then embedded verbatim in the notebook tarball:

* ``MODELS``          — the five open models with pinned GGUF (repo, rev, size, sha256) and the
                        serve layout + flags that passed each model's pilot (D17/D20/D21).
* ``server_commands`` — the exact ``llama-server`` command lines (tensor-split or 2 replicas).
* ``build_tasks``     — the (pair x condition) attack episodes + the no-attack utility episodes,
                        suite-interleaved, with deterministic SHARD/N_SHARDS partitioning.
* ``run_grid``        — threaded workers (fresh AgentDojo logger stack per thread, D21) with
                        skip-if-done resume, a session deadline guard and a heartbeat (D16).
* ``summarize`` / ``main_result_line`` / ``zip_traces`` / ``seed_from_inputs`` — results,
                        resume across Kaggle versions, download zip, final MAIN_RESULT line.

Episode scoring reuses ``runners.run_benchmark.run_one`` / ``write_trace`` unchanged (D23), so
the JSONL and trace layout are the ones ``analysis/ingest.py`` already reads. No-attack utility
episodes use condition ``none`` / injection task ``none`` (analysis.config.NO_ATTACK).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import queue
import threading
import time
import zipfile
from pathlib import Path

SEED = 20261002
BENCHMARK_VERSION = "v1.2.2"
LLAMA_BUILD = "b11344"
NO_ATTACK = "none"
SUITE_ORDER = ("workspace", "travel", "banking", "slack")
# AgentDojo v1.2.2 user tasks per suite (97 total); test_main_grid checks this against agentdojo.
UTILITY_TASKS_V122 = {"workspace": 40, "travel": 20, "banking": 16, "slack": 21}
MAX_SESSION_H = 11.0      # plan to finish well inside Kaggle's 12-h commit limit
LOAD_OVERHEAD = 1.10      # same x1.10 load factor as the D17/D21 ledgers
SETUP_H = 0.25            # pip + llama.cpp + GGUF download/SHA + server start (pilots: 5-10 min)

# Serve settings = the variant that passed each pilot. `pilot_eps_h` is the measured pilot rate
# (no-attack banking+slack, suite-time-sum, 1 agent stream per server).
MODELS = {
    "gemma": dict(
        label="gemma-4-26b-a4b",
        repo="google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        rev="d1c082be9cf3c8a514acf63b8761f4b41935842e",
        file="gemma-4-26B_q4_0-it.gguf", size=14439363584,
        sha256="3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d",
        layout="tensor_split", split="1,1", ctx=24576, nslots=1,
        extra=["--swa-full", "--chat-template-kwargs", '{"enable_thinking":false}'],
        variant="swa_full+no_think", template_expect=None, template_forbid="<|think|>",
        pilot="D20", pilot_eps_h=668.0, min_free_gib=16.0,
        note="QAT Q4_0; 1 server tensor-split 1,1; --swa-full + thinking off (D20 primary variant)"),
    "qwen": dict(
        label="qwen3.6-35b-a3b",
        repo="ggml-org/Qwen3.6-35B-A3B-GGUF",
        rev="baec3ebee244827cda0f4557eafa8b28f7545fa6",
        file="Qwen3.6-35B-A3B-Q4_K_M.gguf", size=20419565568,
        sha256="671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7",
        layout="tensor_split", split="1,1", ctx=24576, nslots=1,
        extra=["--chat-template-kwargs", '{"enable_thinking":false}'],
        variant="no_think", template_expect="<think>\n\n</think>", template_forbid=None,
        pilot="D17", pilot_eps_h=171.4, min_free_gib=22.0,
        note=("Q4_K_M; 1 server tensor-split 1,1 (D17 layout); thinking OFF per plan §11.3 O2. "
              "D17 passed no template kwargs, i.e. ran the template default (thinking ON), so "
              "171.4 eps/h is a thinking-ON rate (D30)")),
    "granite": dict(
        label="granite-4.2-8b",
        repo="ibm-granite/granite-4.2-8b-GGUF",
        rev="93f3f6a8938ee922b784cf4e5b4203cd3428df8f",
        file="granite-4.2-8b-Q4_K_M.gguf", size=5347917952,
        sha256="16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311",
        layout="2replica", split=None, ctx=24576, nslots=1,
        extra=["--chat-template-kwargs", '{"enable_thinking":false}'],
        variant="no_think_kwargs", template_expect="<think></think>", template_forbid=None,
        pilot="D21", pilot_eps_h=252.9, min_free_gib=8.0,
        note="Q4_K_M; 2 replicas (one per T4), thinking OFF (D21 primary variant, 1 stream/replica)"),
    "ministral": dict(
        label="ministral-3-14b",
        repo="mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        rev="74fac473c43357d7fb2671713608183cc72496d0",
        file="Ministral-3-14B-Instruct-2512-Q4_K_M.gguf", size=8239593024,
        sha256="824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613",
        layout="2replica", split=None, ctx=24576, nslots=1, extra=[],
        variant="plain", template_expect="[SYSTEM_PROMPT]", template_forbid=None,
        pilot="D21", pilot_eps_h=412.1, min_free_gib=10.0,
        note="Q4_K_M; 2 replicas (one per T4), plain (D21 primary variant, 1 stream/replica)"),
    "gptoss": dict(
        label="gpt-oss-20b",
        repo="ggml-org/gpt-oss-20b-GGUF",
        rev="ef9b12f2ff56c69cf32153a02784e7a3c88bf524",
        file="gpt-oss-20b-MXFP4.gguf", size=12109566624,
        sha256="27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
        layout="2replica", split=None, ctx=24576, nslots=1,
        extra=["--chat-template-kwargs", '{"reasoning_effort":"low"}'],
        variant="low_effort_kwargs", template_expect="Reasoning: low", template_forbid=None,
        pilot="D21", pilot_eps_h=500.7, min_free_gib=14.0,
        note="MXFP4; 2 replicas (one per T4), reasoning effort low (D21 primary variant, 1 stream/replica)"),
}
# Long names (old builder / docs) -> short keys
ALIASES = {m["label"]: k for k, m in MODELS.items()}


def resolve_model(key: str) -> str:
    k = ALIASES.get(key, key)
    if k not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose one of {sorted(MODELS)} or {sorted(ALIASES)}")
    return k


# ----------------------------------------------------------------------------- serving
def server_commands(m: dict, server: str, model_path: str, seed: int = SEED,
                    base_port: int = 8099, log_dir: str = "/kaggle/working") -> list[dict]:
    """llama-server command(s) for model config ``m``: one tensor-split server or 2 replicas
    pinned one per T4 with CUDA_VISIBLE_DEVICES. Same flags as the pilots (D17/D20/D21)."""
    base = [server, "-m", model_path, "--host", "127.0.0.1", "-ngl", "99", "-c", str(m["ctx"]),
            "-np", str(m["nslots"]), "--jinja", "--temp", "0", "--seed", str(seed)]
    if m["layout"] == "tensor_split":
        cmd = base + ["--port", str(base_port), "--tensor-split", m["split"]] + list(m["extra"])
        return [dict(port=base_port, gpu="0,1", cmd=cmd, env={},
                     log=f"{log_dir}/server_{base_port}.log")]
    if m["layout"] == "2replica":
        out = []
        for r in (0, 1):
            port = base_port + r
            out.append(dict(port=port, gpu=str(r), cmd=base + ["--port", str(port)] + list(m["extra"]),
                            env={"CUDA_VISIBLE_DEVICES": str(r)}, log=f"{log_dir}/server_{port}.log"))
        return out
    raise ValueError(f"unknown layout {m['layout']!r}")


def template_ok(m: dict, prompt: str) -> bool:
    """Prove the template kwargs took effect (thinking off / reasoning low), as the D21 pilots did."""
    if not isinstance(prompt, str) or not prompt or prompt.startswith("<apply-template failed"):
        return False
    if m.get("template_expect") and m["template_expect"] not in prompt:
        return False
    if m.get("template_forbid") and m["template_forbid"] in prompt:
        return False
    return True


# ----------------------------------------------------------------------------- tasks
def interleave_by_suite(pairs: list[dict]) -> list[dict]:
    """Round-robin over suites (stable within a suite) so a run cut short by the deadline still
    covers every suite roughly in proportion."""
    by = {}
    for p in pairs:
        by.setdefault(p["suite"], []).append(p)
    order = [s for s in SUITE_ORDER if s in by] + sorted(s for s in by if s not in SUITE_ORDER)
    out, i = [], 0
    while len(out) < len(pairs):
        for s in order:
            if i < len(by[s]):
                out.append(by[s][i])
        i += 1
    return out


def utility_task_list(suites) -> list[tuple[str, str]]:
    """All no-attack user tasks (97 in v1.2.2), suite order then AgentDojo's task order."""
    return [(s, uid) for s in SUITE_ORDER if s in suites for uid in suites[s].user_tasks]


def build_tasks(pairs, conditions, utility_tasks=(), shard: int = 1, n_shards: int = 1) -> list[dict]:
    """Attack episodes (pair-major: all conditions of a pair run back to back, so a truncated run
    keeps paired contrasts) followed by no-attack utility episodes. Sharding is by pair index /
    utility index modulo n_shards, so shards are disjoint and their union is the full grid."""
    if not (1 <= shard <= n_shards):
        raise ValueError(f"shard {shard} not in 1..{n_shards}")
    tasks = []
    for i, p in enumerate(interleave_by_suite(list(pairs))):
        if i % n_shards != shard - 1:
            continue
        for c in conditions:
            tasks.append({"kind": "attack", "suite": p["suite"], "user_task": p["user_task"],
                          "injection_task": p["injection_task"], "condition": c})
    for j, (s, u) in enumerate(utility_tasks):
        if j % n_shards != shard - 1:
            continue
        tasks.append({"kind": "utility", "suite": s, "user_task": u,
                      "injection_task": NO_ATTACK, "condition": NO_ATTACK})
    return tasks


def task_key(t) -> tuple:
    return (t["suite"], t["user_task"], t["injection_task"], t["condition"])


def episode_count(n_pairs: int, n_conditions: int, utility: bool = True) -> int:
    return n_pairs * n_conditions + (sum(UTILITY_TASKS_V122.values()) if utility else 0)


def estimate_hours(n_episodes: int, eps_per_h: float, overhead: float = LOAD_OVERHEAD,
                   setup_h: float = SETUP_H) -> float:
    return n_episodes / eps_per_h * overhead + setup_h


def shards_needed(hours: float, max_h: float = MAX_SESSION_H) -> int:
    return max(1, math.ceil(hours / max_h))


# ----------------------------------------------------------------------------- results file
def read_records(out_jsonl) -> list[dict]:
    recs = []
    p = Path(out_jsonl)
    if not p.exists():
        return recs
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and not r.get("_header") and "suite" in r:
            recs.append(r)
    return recs


def best_status(out_jsonl) -> dict:
    """key -> 'ok' | 'error' (ok wins if a key was retried)."""
    st = {}
    for r in read_records(out_jsonl):
        k = (r["suite"], r["user_task"], r["injection_task"], r["condition"])
        if st.get(k) != "ok":
            st[k] = r.get("status", "ok")
    return st


def done_keys(out_jsonl, retry_errors: bool = True) -> set:
    """Skip-if-done keys. Errored episodes are retried in a later session by default."""
    return {k for k, s in best_status(out_jsonl).items() if s == "ok" or not retry_errors}


def seed_from_inputs(input_root, out_jsonl, trace_dir, run_id) -> dict:
    """Resume across Kaggle versions: if a previous version's output is attached as an input
    (Add Input -> notebook output), copy its JSONL lines and traces zip into this session."""
    out_jsonl, trace_dir = Path(out_jsonl), Path(trace_dir)
    found = {"jsonl": [], "zips": [], "lines": 0, "traces": 0}
    if not input_root or not Path(input_root).is_dir():
        return found
    prev = sorted(glob.glob(str(Path(input_root) / "**" / out_jsonl.name), recursive=True))
    prev = [p for p in prev if Path(p).resolve() != out_jsonl.resolve()]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for p in prev:
        txt = Path(p).read_text(encoding="utf-8")
        lines = [l for l in txt.splitlines() if l.strip()]
        with out_jsonl.open("a", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
        found["jsonl"].append(p)
        found["lines"] += len(lines)
    for z in sorted(glob.glob(str(Path(input_root) / "**" / f"{run_id}_traces.zip"), recursive=True)):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.startswith("/") or ".." in n.split("/"):
                    raise RuntimeError(f"unsafe path in {z}: {n}")
            zf.extractall(trace_dir)
            found["traces"] += sum(1 for n in zf.namelist() if n.endswith(".json"))
        found["zips"].append(z)
    return found


# ----------------------------------------------------------------------------- episodes
def run_utility_one(suite, pipeline, user_task):
    """No-attack utility episode: default (benign) injection-vector values, same query/retry loop
    and utility check as runners.run_benchmark.run_one / AgentDojo run_task_with_pipeline."""
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.task_suite import (functions_stack_trace_from_messages,
                                                 model_output_from_messages)
    from agentdojo.types import get_text_content_as_str
    from scoring.judge import bengali_char_share

    environment = suite.load_and_inject_default_environment({})
    task_environment = user_task.init_environment(environment)
    pre_environment = task_environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    messages, model_output = [], None
    for _ in range(3):
        try:
            _, _, task_environment, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_environment)
        except Exception as e:
            if hasattr(e, "task_environment"):
                task_environment, messages = e.task_environment, e.messages
            else:
                raise
        model_output = model_output_from_messages(messages)
        if model_output is not None:
            break
    traces = functions_stack_trace_from_messages(messages)
    out_text = get_text_content_as_str(model_output) if model_output else ""
    utility = suite._check_task_result(user_task, model_output or [], pre_environment,
                                       task_environment, traces)
    arg_text = json.dumps([getattr(t, "args", {}) for t in traces], ensure_ascii=False)
    return {"_messages": messages, "_injections": {}, "utility": bool(utility),
            "raw_security": None, "normalised_security": None, "corrected_security": None,
            "scoring_group": None, "bengali_share_output": round(bengali_char_share(out_text), 4),
            "bengali_share_args": round(bengali_char_share(arg_text), 4), "needs_judge": False,
            "n_tool_calls": len(traces)}


def default_pipeline_factory(defense=None):
    def make(port):
        from runners.endpoints import build_local_pipeline
        return build_local_pipeline(base_url=f"http://127.0.0.1:{port}/v1", defense=defense)
    return make


def run_grid(tasks, ports, *, label, out_jsonl, trace_dir, defense=None, streams_per_port=1,
             deadline_ts=None, pipeline_factory=None, header=None, shard="1/1",
             benchmark_version=BENCHMARK_VERSION, heartbeat_s=300, log=print) -> dict:
    """Run every not-yet-done task over ``ports`` (``streams_per_port`` agent streams each).
    Appends one JSONL record per episode and one AgentDojo-format trace per episode. Workers stop
    taking new episodes once ``deadline_ts`` passes (the rest stays for a resumed session)."""
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.logging import LOGGER_STACK
    from agentdojo.task_suite.load_suites import get_suites
    import attacks  # noqa: F401  (registers the MultilingualAttack conditions)
    from runners.run_benchmark import run_one, write_trace
    from scoring.scoring import NormalisedScorer

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pipeline_factory = pipeline_factory or default_pipeline_factory(defense)
    done = done_keys(out_jsonl)
    todo = [t for t in tasks if task_key(t) not in done]
    q = queue.Queue()
    for t in todo:
        q.put(t)
    suites = get_suites(benchmark_version)
    scorer = NormalisedScorer()
    lock = threading.Lock()
    stats = {"n_tasks": len(tasks), "already_done": len(tasks) - len(todo), "ran": 0, "ok": 0,
             "error": 0, "worker_errors": []}
    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"_header": True, "started": dt.datetime.now().isoformat(timespec="seconds"),
                            "pipeline": label, "defense": defense, "shard": shard,
                            "n_tasks": len(tasks), "n_todo": len(todo), **(header or {})},
                           ensure_ascii=False, default=str) + "\n")
    log(f"{label}: {len(todo)} episodes to run ({stats['already_done']} already done) over "
        f"{len(ports)} server(s) x {streams_per_port} stream(s)")

    def worker(wid, port):
        LOGGER_STACK.set([])   # fresh logger stack per thread (D21)
        try:
            pipe = pipeline_factory(port)
            pipe.name = label
        except Exception as e:   # server unreachable etc.: record, leave tasks for other workers
            with lock:
                stats["worker_errors"].append(f"w{wid}:{port}: {type(e).__name__}: {e}")
            log(f"[worker {wid}] could not build pipeline on :{port}: {type(e).__name__}: {e}")
            return
        while True:
            if deadline_ts is not None and time.time() >= deadline_ts:
                return
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            suite = suites[t["suite"]]
            ts = time.time()
            try:
                ut = suite.get_user_task_by_id(t["user_task"])
                if t["kind"] == "utility":
                    res = run_utility_one(suite, pipe, ut)
                else:
                    it = suite.get_injection_task_by_id(t["injection_task"])
                    atk = load_attack(t["condition"], suite, pipe)
                    res = run_one(suite, pipe, ut, it, atk, scorer)
                status, err = "ok", None
            except Exception as e:
                res, status, err = {}, "error", f"{type(e).__name__}: {str(e)[:500]}"
            dur = time.time() - ts
            msgs = res.pop("_messages", [])
            inj = res.pop("_injections", {})
            try:
                write_trace(trace_dir, label, t, t["condition"], msgs, inj, res, err, dur)
            except Exception as e:  # never lose the JSONL record because of a trace write
                err = (err or "") + f" | trace write failed: {type(e).__name__}: {e}"
            rec = {"suite": t["suite"], "user_task": t["user_task"], "injection_task": t["injection_task"],
                   "condition": t["condition"], "kind": t["kind"], "model": label, "defense": defense,
                   "shard": shard, "status": status, "error": err, "duration_s": round(dur, 2),
                   "worker": wid, **res}
            with lock:
                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                stats["ran"] += 1
                stats[status] += 1
                if status == "error":
                    log(f"[worker {wid}] ERROR {task_key(t)}: {err}")

    stop = threading.Event()

    def heartbeat():
        while not stop.wait(heartbeat_s):
            el = time.time() - t0
            rate = stats["ran"] / (el / 3600) if el > 0 else 0.0
            log(f"[heartbeat] {el:.0f}s  ran {stats['ran']}/{len(todo)}  ok {stats['ok']}  "
                f"err {stats['error']}  {rate:.1f} eps/h")

    threading.Thread(target=heartbeat, daemon=True).start()
    threads = [threading.Thread(target=worker, args=(i, port), daemon=True)
               for i, port in enumerate(p for p in ports for _ in range(streams_per_port))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    stop.set()
    stats["remaining"] = q.qsize()
    stats["stopped_by_deadline"] = bool(stats["remaining"]) and deadline_ts is not None and time.time() >= deadline_ts
    stats["grid_s"] = round(time.time() - t0, 1)
    stats["eps_per_h"] = round(stats["ran"] / (stats["grid_s"] / 3600), 1) if stats["grid_s"] > 0 else 0.0
    return stats


# ----------------------------------------------------------------------------- summary
def summarize(out_jsonl, tasks) -> dict:
    st = best_status(out_jsonl)
    per = {}
    for t in tasks:
        c = per.setdefault(t["condition"], {"expected": 0, "ok": 0, "error": 0, "missing": 0})
        c["expected"] += 1
        s = st.get(task_key(t))
        c["ok" if s == "ok" else "error" if s == "error" else "missing"] += 1
    tot = {k: sum(v[k] for v in per.values()) for k in ("expected", "ok", "error", "missing")}
    return {"per_condition": per, "episodes_expected": tot["expected"], "episodes_done": tot["ok"],
            "errors": tot["error"], "missing": tot["missing"],
            "complete": tot["ok"] == tot["expected"] and tot["expected"] > 0}


def main_result_line(run_id, label, shard, summary, wall_s, stats, embed_sha256) -> str:
    pc = ",".join(f"{c}:{v['ok']}/{v['expected']}" for c, v in summary["per_condition"].items())
    return (f"MAIN_RESULT run_id={run_id} model={label} shard={shard} "
            f"episodes_done={summary['episodes_done']}/{summary['episodes_expected']} "
            f"errors={summary['errors']} missing={summary['missing']} "
            f"wall_s={wall_s:.0f} grid_s={stats.get('grid_s', 0):.0f} "
            f"ran_this_session={stats.get('ran', 0)} eps_per_h={stats.get('eps_per_h', 0)} "
            f"per_condition={pc} complete={summary['complete']} "
            f"stopped_by_deadline={stats.get('stopped_by_deadline', False)} embed_sha256={embed_sha256}")


def zip_traces(trace_dir, zip_path) -> dict:
    """Zip every trace JSON (sorted, relative to trace_dir) for one-click download."""
    trace_dir, zip_path = Path(trace_dir), Path(zip_path)
    files = sorted(p for p in trace_dir.rglob("*.json")) if trace_dir.is_dir() else []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(trace_dir).as_posix())
    return {"zip": str(zip_path), "n_traces": len(files),
            "zip_bytes": zip_path.stat().st_size if zip_path.exists() else 0}
