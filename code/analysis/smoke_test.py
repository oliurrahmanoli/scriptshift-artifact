"""Synthetic-data smoke test for analysis/stats.py (research-plan.md 4.7).

Generates fully synthetic (model, user_task, injection_task, condition) records with
a KNOWN planted effect, then checks that the stats recover it: McNemar detects a real
difference, finds no difference where none was planted, Holm/BH behave, the cluster
bootstrap CI covers the truth, comprehension-conditional ASR is sane, and TOST flags
equivalence for a null contrast. No real experiment numbers are used.

Run:  python analysis/smoke_test.py   (also importable as test functions)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.stats import (  # noqa: E402
    contrast, holm_bonferroni, benjamini_hochberg,
    comprehension_conditional_asr, tost_equivalence,
)

SUITES = {"workspace": 14, "travel": 7, "banking": 9, "slack": 5}
USERS_PER = 8
MODELS = ["modelA", "modelB", "modelC"]


def make_records(seed=0, base=0.10, lift_native=0.08, lift_mt=0.0):
    """en, bn_mt, bn_native with a planted native lift over en (and mt==en)."""
    rng = random.Random(seed)
    recs = []
    for model in MODELS:
        for suite, ninj in SUITES.items():
            for j in range(ninj):
                inj = f"{suite}_inj{j}"
                # per-injection-task random intercept (clustering)
                cluster_off = rng.uniform(-0.04, 0.04)
                for u in range(USERS_PER):
                    ut = f"{suite}_u{u}"
                    comp = rng.random() < 0.7
                    p_en = max(0, min(1, base + cluster_off))
                    p_mt = max(0, min(1, base + cluster_off + lift_mt))
                    p_bn = max(0, min(1, base + cluster_off + lift_native))
                    for cond, p in [("en", p_en), ("bn_mt", p_mt), ("bn_native", p_bn)]:
                        succ = rng.random() < (p if comp else p * 0.3)
                        recs.append({"model": model, "suite": suite, "user_task": ut,
                                     "injection_task": inj, "condition": cond,
                                     "success": succ, "comprehended": comp})
    return recs


def test_detects_planted_effect():
    recs = make_records(seed=1, lift_native=0.12)
    res = contrast(recs, "en", "bn_native", n_boot=2000, seed=1)
    assert res.diff_pp > 0, res
    assert res.p_value < 0.05, res
    # CI should exclude zero for a strong effect
    assert res.ci_low > 0, res


def test_null_contrast_not_significant_and_equivalent():
    recs = make_records(seed=2, lift_native=0.0, lift_mt=0.0)
    res = contrast(recs, "en", "bn_mt", n_boot=2000, seed=2)
    assert res.p_value > 0.05, res
    tost = tost_equivalence(recs, "en", "bn_mt", margin_pp=5.0, n_boot=2000, seed=2)
    assert tost["equivalent"] is True, tost


def test_corrections_monotone():
    pvals = [0.001, 0.02, 0.2, 0.9]
    holm = holm_bonferroni(pvals)
    bh = benjamini_hochberg(pvals)
    assert holm[0][3] is True  # smallest p rejected
    assert holm[3][3] is False
    # adjusted p >= raw p
    assert all(a >= p for _, p, a, _ in holm)
    assert all(a >= p for _, p, a, _ in bh)


def test_comprehension_conditional():
    recs = make_records(seed=3, lift_native=0.1)
    out = comprehension_conditional_asr(recs, "bn_native")
    assert 0 <= out["comprehension_rate"] <= 1
    # ASR among comprehended should exceed overall ASR (we down-weight non-comprehenders)
    assert out["asr_given_comprehended"] >= out["asr_all"]


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} synthetic analysis smoke tests passed")
    sys.exit(0 if passed == len(tests) else 1)
