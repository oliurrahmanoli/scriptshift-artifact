"""Stratified, seeded 300-pair subset sampler (research-plan.md 4.5, 11.7).

Design:
  * All 4 suites, all 35 injection tasks are represented.
  * Per-suite user-tasks-per-injection-task, giving 300 pairs:
      workspace 14 inj x 10 users = 140
      travel     7 inj x  8 users =  56
      banking    9 inj x  6 users =  54
      slack      5 inj x 10 users =  50
  * For each injection task, user tasks are drawn without replacement from that
    suite's user tasks, preferring tasks into which the vector is actually injectable
    (the ground-truth trajectory reads the canary), so pairs are meaningful.
  * Fixed seed -> the same 300 pairs for every model and condition (paired design).

Writes subset/pairs.json and subset/pairs.csv. Deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_SUITE_USERS = {"workspace": 10, "travel": 8, "banking": 6, "slack": 10}
SEED = 20261002  # published seed (date-based); see DECISIONS.md


def _injectable_user_tasks(suite, injection_task, cache):
    """User task IDs whose ground-truth trajectory exposes this injection vector."""
    from agentdojo.attacks.base_attacks import BaseAttack

    # Build a lightweight attack just to reuse get_injection_candidates.
    key = suite.name
    if key not in cache:
        cache[key] = _CandidateProbe(suite)
    probe = cache[key]
    good = []
    for uid, utask in suite.user_tasks.items():
        try:
            cands = probe.get_injection_candidates(utask)
        except Exception:
            cands = []
        if cands:
            good.append(uid)
    return good


class _CandidateProbe:
    """Minimal BaseAttack subclass to call get_injection_candidates without a pipeline."""

    def __init__(self, suite):
        from agentdojo.attacks.base_attacks import BaseAttack

        class _P(BaseAttack):
            name = "_probe"

            def attack(self, user_task, injection_task):
                return {}

        self._p = _P(suite, target_pipeline=_DummyPipe())

    def get_injection_candidates(self, utask):
        return self._p.get_injection_candidates(utask)


class _DummyPipe:
    name = "dummy"

    def query(self, *a, **k):
        raise NotImplementedError


def build(benchmark_version="v1.2.2", seed=SEED):
    from agentdojo.task_suite.load_suites import get_suites

    rng = random.Random(seed)
    suites = get_suites(benchmark_version)
    pairs = []
    cache = {}
    # Precompute injectable user tasks per suite (same for every injection vector in
    # practice, but we compute per suite once for speed/robustness).
    for suite_name, suite in suites.items():
        n_users = PER_SUITE_USERS[suite_name]
        all_user_ids = list(suite.user_tasks.keys())
        # injectable set at suite level (union); fall back to all users
        probe = _CandidateProbe(suite)
        injectable = []
        for uid, utask in suite.user_tasks.items():
            try:
                if probe.get_injection_candidates(utask):
                    injectable.append(uid)
            except Exception:
                pass
        pool = injectable if len(injectable) >= n_users else all_user_ids
        for inj_id in suite.injection_tasks.keys():
            k = min(n_users, len(pool))
            chosen = rng.sample(sorted(pool), k)
            for uid in chosen:
                pairs.append({
                    "suite": suite_name,
                    "user_task": uid,
                    "injection_task": inj_id,
                })
    return pairs


PER_SUITE_USERS_150 = {"workspace": 5, "travel": 4, "banking": 3, "slack": 5}  # half of the 300 strata


def build_nested(pairs, per_suite_users=None):
    """Nested stratified 150-pair subset (research-plan §12, DECISIONS D24).

    Deterministically selects, for each injection task, the first ``per_suite_users[suite]``
    pairs *as they already appear* in the seeded 300-pair list. The result is therefore a
    strict SUBSET of the 300 pairs (every 150-pair is also a 300-pair), stratified the same
    way (workspace 70, travel 28, banking 27, slack 25 = 150), with all 35 injection tasks
    present. No new randomness: the 150 is a prefix-per-stratum of the seed-20261002 subset,
    so the main grid can scale 150 -> 300 later without re-pairing.
    """
    per_suite_users = per_suite_users or PER_SUITE_USERS_150
    by_inj = {}
    for pr in pairs:
        by_inj.setdefault((pr["suite"], pr["injection_task"]), []).append(pr)
    out = []
    for (suite, _inj), group in by_inj.items():
        k = min(per_suite_users[suite], len(group))
        out.extend(group[:k])
    # keep original order
    order = {id(pr): i for i, pr in enumerate(pairs)}
    out.sort(key=lambda pr: order[id(pr)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    pairs = build(args.benchmark_version, args.seed)
    counts = {}
    for p in pairs:
        counts[p["suite"]] = counts.get(p["suite"], 0) + 1
    meta = {"benchmark_version": args.benchmark_version, "seed": args.seed,
            "n_pairs": len(pairs), "per_suite": counts,
            "per_suite_users": PER_SUITE_USERS}
    (ROOT / "pairs.json").write_text(
        json.dumps({"meta": meta, "pairs": pairs}, indent=2), encoding="utf-8")
    with open(ROOT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "user_task", "injection_task"])
        w.writeheader()
        w.writerows(pairs)
    print(f"Wrote {len(pairs)} pairs. Per suite: {counts}")

    # Nested 150-pair subset (§12, D24): a strict subset of the 300, same strata halved.
    nested = build_nested(pairs)
    ncounts = {}
    for pr in nested:
        ncounts[pr["suite"]] = ncounts.get(pr["suite"], 0) + 1
    nmeta = {"benchmark_version": args.benchmark_version, "seed": args.seed,
             "n_pairs": len(nested), "per_suite": ncounts,
             "per_suite_users": PER_SUITE_USERS_150,
             "nested_in": "pairs.json (300-pair, same seed)",
             "note": "strict subset of the 300-pair subset; stratified the same way (halved)."}
    (ROOT / "pairs_150.json").write_text(
        json.dumps({"meta": nmeta, "pairs": nested}, indent=2), encoding="utf-8")
    with open(ROOT / "pairs_150.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "user_task", "injection_task"])
        w.writeheader()
        w.writerows(nested)
    # sanity: nested is a subset of the 300
    keyset = {(p["suite"], p["user_task"], p["injection_task"]) for p in pairs}
    assert all((p["suite"], p["user_task"], p["injection_task"]) in keyset for p in nested)
    assert len({(p["suite"], p["injection_task"]) for p in nested}) == 35
    print(f"Wrote {len(nested)} nested pairs. Per suite: {ncounts}")


if __name__ == "__main__":
    main()
