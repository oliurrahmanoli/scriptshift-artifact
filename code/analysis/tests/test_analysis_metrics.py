"""Unit tests for analysis/metrics.py and analysis/bootstrap.py against known values."""
import numpy as np
import pytest

from analysis.bootstrap import asr_ci, paired_diff_ci, resampled_means
from analysis.metrics import (auroc, cohen_kappa, flip_rate, krippendorff_alpha, rate_above,
                              threshold_at_fpr, tost_from_boot, wilson)

# Krippendorff (2011) "Computing Krippendorff's alpha-reliability", example C (4 coders, 12 units)
KRIPP = [[1, 1, None, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
         [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [None, 5, 5, 5], [None, None, 1, 1], [None, None, 3, None]]


def test_krippendorff_reference_values():
    assert krippendorff_alpha(KRIPP, "nominal") == pytest.approx(0.743, abs=1e-3)
    assert krippendorff_alpha(KRIPP, "ordinal") == pytest.approx(0.815, abs=1e-3)
    assert krippendorff_alpha(KRIPP, "interval") == pytest.approx(0.849, abs=1e-3)
    assert krippendorff_alpha([[3, 3], [4, 4], [1, 1]], "ordinal") == 1.0


def test_cohen_kappa_textbook():
    # 2x2: agree-yes 20, agree-no 15, a=yes/b=no 5, a=no/b=yes 10 -> po .7, pe .5, kappa .4
    a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
    b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_wilson_known():
    p, lo, hi = wilson(12, 16)
    assert p == 0.75 and lo == pytest.approx(0.5050, abs=1e-3) and hi == pytest.approx(0.8982, abs=1e-3)
    assert np.isnan(wilson(0, 0)[0])


def test_auroc_and_thresholds():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auroc([0.1], [0.9]) == 0.0
    neg = np.linspace(0, 1, 101)               # 101 benign scores
    t = threshold_at_fpr(neg, 0.02)            # at most 2 benign above t
    k, n = rate_above(neg, t)
    assert k <= 2 and n == 101
    assert flip_rate([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_tost_rule():
    rng = np.random.default_rng(0)
    eq = tost_from_boot(0.5, rng.normal(0.5, 1.0, 5000), 5.0)
    assert eq["equivalent"] and eq["tost_p"] < 0.05
    ne = tost_from_boot(6.0, rng.normal(6.0, 1.0, 5000), 5.0)
    assert not ne["equivalent"] and ne["tost_p"] > 0.5


def test_bootstrap_identical_pairs_zero_width():
    a = np.array([1, 0, 1, 0, 0, 1] * 10, bool)
    cl = np.repeat(np.arange(12), 5)
    r = paired_diff_ci(a, a, cl, n_boot=500, seed=1, bca=True)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_bootstrap_ci_covers_truth_and_is_deterministic():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(35), 8)
    x = rng.random(len(cl)) < 0.3
    r1 = asr_ci(x, cl, n_boot=2000, seed=5, bca=True)
    r2 = asr_ci(x, cl, n_boot=2000, seed=5)
    assert r1["lo"] < 0.3 < r1["hi"] and r1["lo"] == r2["lo"]
    assert r1["n_clusters"] == 35 and r1["bca_lo"] < r1["est"] < r1["bca_hi"]


def test_two_stage_wider_than_cluster_only():
    rng = np.random.default_rng(4)
    cl = np.repeat(np.arange(35), 8)
    x = (rng.random(len(cl)) < 0.3).astype(float)
    m2 = resampled_means(x, cl, 3000, 1, two_stage=True)[:, 0].std()
    m1 = resampled_means(x, cl, 3000, 1, two_stage=False)[:, 0].std()
    assert m2 >= m1 * 0.95      # within-cluster resampling adds (never removes) variance
