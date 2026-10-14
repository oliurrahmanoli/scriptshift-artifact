"""Unicode-preserving rerun arm (DECISIONS D41): task lists, the patched/stock arm runner and the
bounded stock retry, used by the self-contained ``kaggle/uni_rerun_<model>.ipynb`` notebooks.

Everything episode-level is the stock harness, unchanged and byte-identical to the main-grid embed
(``runners.main_grid.run_grid`` -> ``runners.run_benchmark.run_one`` / ``write_trace``; same
MultilingualAttack payloads, scorer, llama-server flags, seed, 1 slot + 1 agent stream per server).
The only difference in the ``unicode_preserving`` arm is ``runners.unicode_patch`` switched ON for the
whole arm; a stock retry runs with it OFF. Every pipeline a worker builds is checked to render a
Bengali probe as expected before it runs an episode (``unicode_patch.check_pipeline``).

Outputs are kept apart: ``<model>_150_uni.jsonl`` + ``traces_uni/`` (arm) and
``<model>_150_stockretry.jsonl`` + ``traces_stockretry/`` (stock retry, joins the primary set).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from runners import unicode_patch as U
from runners.main_grid import default_pipeline_factory, run_grid, task_key

ARM_UNI = "unicode_preserving"
ARM_STOCK_RETRY = "stock_retry"
TIERS = ("tier_A", "tier_B")
STOCK_RETRY_BUDGET_S = 1500   # D35 timeout = 9 x 600-s client timeouts; cap the retry at 25 min


def load_episodes(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _task(e) -> dict:
    return {"kind": "attack", "suite": e["suite"], "user_task": e["user_task"],
            "injection_task": e["injection_task"], "condition": e["condition"]}


def arm_tasks(episodes: dict, tiers=("tier_A",)) -> list[dict]:
    """Tasks of the Unicode-preserving arm, in the selection's (pair-major, suite-interleaved) order."""
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tier(s) {bad}")
    out, seen = [], set()
    for t in tiers:
        for e in episodes[t]:
            k = task_key(_task(e))
            if k in seen:
                raise ValueError(f"duplicate episode {k}")
            seen.add(k)
            out.append(_task(e))
    return out


def retry_tasks(episodes: dict) -> list[dict]:
    return [_task(e) for e in episodes.get("stock_retry", [])]


def checked_pipeline_factory(expect_unicode: bool, defense=None, base_factory=None):
    """Build each worker's pipeline as the main grid does, then prove the formatter state."""
    base = base_factory or default_pipeline_factory(defense)

    def make(port):
        pipe = base(port)
        U.check_pipeline(pipe, expect_unicode)
        return pipe
    return make


def run_arm(tasks, ports, *, unicode: bool, label, out_jsonl, trace_dir, header=None, deadline_ts=None,
            streams_per_port=1, pipeline_factory=None, log=print) -> dict:
    """Run ``tasks`` with the Unicode patch ON (``unicode=True``) or explicitly OFF, then restore the
    previous state. Raises before any episode if the live formatter does not behave as expected."""
    arm = ARM_UNI if unicode else ARM_STOCK_RETRY
    with U.unicode_preserving(unicode):
        chk = U.self_check(expect_enabled=unicode)
        log(f"[{arm}] unicode_patch self-check: {chk}")
        hdr = dict(header or {}, arm=arm, unicode_patch=U.describe(), unicode_self_check=chk)
        stats = run_grid(tasks, ports, label=label, out_jsonl=out_jsonl, trace_dir=trace_dir,
                         streams_per_port=streams_per_port, deadline_ts=deadline_ts, header=hdr,
                         pipeline_factory=checked_pipeline_factory(unicode, base_factory=pipeline_factory),
                         log=log)
    U.self_check(expect_enabled=False)
    stats["arm"] = arm
    return stats


def run_stock_retry(tasks, ports, *, label, out_jsonl, trace_dir, header=None, budget_s=STOCK_RETRY_BUDGET_S,
                    pipeline_factory=None, log=print) -> dict:
    """Stock-formatting retry of infrastructure-error episodes with a wall-clock budget. The episode
    itself is the unchanged stock path; if it has not finished after ``budget_s`` (the D35 episode hung
    for 5,418 s: 9 consecutive 600-s client timeouts) it is abandoned, recorded as an infrastructure
    error (missing, D38.2) and the notebook moves on, so one hung request cannot eat the GPU quota."""
    if not tasks:
        return {"arm": ARM_STOCK_RETRY, "ran": 0, "ok": 0, "error": 0, "abandoned": [], "n_tasks": 0}
    if U.is_enabled():
        raise RuntimeError("stock retry must run with the unicode patch OFF")
    box = {}

    def target():
        box["stats"] = run_arm(tasks, ports, unicode=False, label=label, out_jsonl=out_jsonl,
                               trace_dir=trace_dir, header=header, pipeline_factory=pipeline_factory, log=log)
    t0 = time.time()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(budget_s)
    if "stats" in box:
        st = box["stats"]
        st["abandoned"] = []
        return st
    from runners.main_grid import done_keys
    done = done_keys(out_jsonl, retry_errors=False)
    abandoned = [t for t in tasks if task_key(t) not in done]
    with Path(out_jsonl).open("a", encoding="utf-8") as f:
        for t in abandoned:
            f.write(json.dumps({**{k: t[k] for k in ("suite", "user_task", "injection_task", "condition", "kind")},
                                "model": label, "defense": None, "shard": "1/1", "status": "error",
                                "error": f"APITimeoutError: stock retry abandoned after {budget_s} s without a "
                                         f"response (RetryBudgetExceeded; same symptom as D35)",
                                "duration_s": round(time.time() - t0, 2), "worker": -1}, ensure_ascii=False) + "\n")
    log(f"[{ARM_STOCK_RETRY}] budget {budget_s}s exceeded; abandoned {len(abandoned)} episode(s)")
    # The abandoned worker thread may still hold a request; the patch is OFF either way.
    return {"arm": ARM_STOCK_RETRY, "n_tasks": len(tasks), "ran": len(tasks) - len(abandoned), "ok": 0,
            "error": len(abandoned), "abandoned": [task_key(t) for t in abandoned],
            "grid_s": round(time.time() - t0, 1), "eps_per_h": 0.0, "remaining": 0,
            "stopped_by_deadline": False, "worker_errors": []}


def estimate_session_h(episodes: dict, n_streams: int, setup_h: float, tiers=("tier_A",),
                       overhead: float = 1.10, boot_h: float = 0.05, retry_budget_s: float = 0.0,
                       en_durations: dict | None = None) -> dict:
    """GPU-h for one notebook from OBSERVED stock durations of the very same episodes (main-grid JSONL),
    split over the model's agent streams, x``overhead``, + the model's observed in-session setup
    (wall_s - grid_s) + ``boot_h`` (Kaggle start/save). ``high`` uses, per episode, the larger of the
    stock duration and the same pair's ``en`` duration (an agent that now follows the injection)."""
    eps = [e for t in tiers for e in episodes[t]]
    base = sum(e["stock_duration_s"] for e in eps) / n_streams / 3600
    high = base
    if en_durations is not None:
        high = sum(max(e["stock_duration_s"], en_durations.get((e["suite"], e["user_task"], e["injection_task"]),
                                                               e["stock_duration_s"])) for e in eps) / n_streams / 3600
    extra = setup_h + boot_h + retry_budget_s / 3600
    return {"n_episodes": len(eps), "episode_h": round(base, 3), "episode_h_high": round(high, 3),
            "est_session_h": round(base * overhead + extra, 2), "est_session_h_high": round(high * overhead + extra, 2),
            "setup_h": round(setup_h, 3), "boot_h": boot_h, "retry_budget_h": round(retry_budget_s / 3600, 3)}
