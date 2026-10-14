"""D38 error-row rules on a tiny hand-made frame (no real data)."""
import pandas as pd
import pytest

from analysis.config import ERROR_CLASS_OVERRIDES
from analysis.episode_rules import accounting, apply_error_class_overrides, apply_rules, classify_error

CONDS = ["en", "bn_mt", "bn_llm", "banglish_llm"]


def _frame():
    rows = []
    for m in ("A", "B"):
        for p in ("injection_task_0", "injection_task_1"):
            for c in CONDS:
                rows.append(dict(model=m, defense="none", suite="workspace", user_task="user_task_1",
                                 injection_task=p, condition=c, rep=0, status="ok", error=None,
                                 raw_security=True, normalised_security=True, corrected_security=True,
                                 utility=True))
        rows.append(dict(model=m, defense="none", suite="workspace", user_task="user_task_1",
                         injection_task="none", condition="none", rep=0, status="ok", error=None,
                         raw_security=None, normalised_security=None, corrected_security=None, utility=True))
    df = pd.DataFrame(rows).astype({c: object for c in ("raw_security", "normalised_security",
                                                         "corrected_security", "utility", "error")})

    def setr(m, p, c, err):
        i = df.index[(df.model == m) & (df.injection_task == p) & (df.condition == c)][0]
        df.loc[i, ["status", "error", "raw_security", "normalised_security", "corrected_security", "utility"]] = \
            ["error", err, None, None, None, None]
    setr("A", "injection_task_0", "bn_llm", "JSONDecodeError: Unterminated string starting at: line 1")
    setr("A", "injection_task_1", "en", "APITimeoutError: Request timed out.")
    setr("B", "injection_task_0", "bn_mt", "Error code: 400 - {'error': {'type': 'exceed_context_size_error'}}")
    i = df.index[(df.model == "B") & (df.condition == "none")][0]
    df.loc[i, ["status", "error", "utility"]] = ["error", "exceed_context_size_error", None]
    return df


def test_classify():
    assert classify_error(None) is None
    assert classify_error("JSONDecodeError: Unterminated string") == "model_output"
    assert classify_error("openai.APITimeoutError: Request timed out.") == "infrastructure"
    assert classify_error("exceed_context_size_error") == "context_overflow"
    assert classify_error("something odd") == "other"


def test_primary_rules():
    d = apply_rules(_frame(), "failure", "pair", self_generated_models=("A",))
    a = d[(d.model == "A") & (d.injection_task == "injection_task_0") & (d.condition == "bn_llm")].iloc[0]
    assert a.status == "ok" and a.normalised_security is False and a.utility is False
    t = d[(d.model == "A") & (d.injection_task == "injection_task_1") & (d.condition == "en")].iloc[0]
    assert t.status == "error" and t.exclusion == "infrastructure_missing"
    b = d[(d.model == "B") & (d.injection_task == "injection_task_0")]
    assert set(b.status) == {"error", "excluded_pair"} and (b.status != "ok").all()
    assert d[(d.model == "B") & (d.injection_task == "injection_task_1")].status.eq("ok").all()
    assert d[(d.model == "B") & (d.condition == "none")].status.iloc[0] == "error"
    assert d.loc[d.model == "A", "self_generated_payload"].sum() == 4      # bn_llm + banglish_llm x 2 pairs
    assert not d.loc[d.model == "A", "primary_pool"].any() and d.loc[d.model == "B", "primary_pool"].all()
    acc = accounting(d).set_index(["model", "condition"])
    assert acc.loc[("B", "en"), "analysed"] == 1 and acc.loc[("B", "en"), "context_overflow_pair_sibling"] == 1
    assert acc.loc[("A", "bn_llm"), "analysed"] == 2


def test_sensitivity_rules():
    d = apply_rules(_frame(), "exclude", "episode")
    a = d[(d.model == "A") & (d.injection_task == "injection_task_0") & (d.condition == "bn_llm")].iloc[0]
    assert a.status == "error" and a.exclusion == "model_output_error_excluded"
    b = d[(d.model == "B") & (d.injection_task == "injection_task_0")]
    assert (b.status == "ok").sum() == 3


def test_bad_rule():
    with pytest.raises(ValueError):
        apply_rules(_frame(), "drop")


def _d48_error_text():
    return ("InternalServerError: Error code: 500 - {'error': {'code': 500, "
            "'message': 'The model produced output that does not match the expected peg-native format', "
            "'type': 'server_error'}}")


def test_d48_override_unicode_arm_only():
    """D48: exact gpt-oss Unicode-arm episode → model_output; other 500s stay infrastructure."""
    assert any(ov["decision"] == "D48" for ov in ERROR_CLASS_OVERRIDES)
    # regex alone still says infrastructure (override must not be a pattern broaden)
    assert classify_error(_d48_error_text()) == "infrastructure"
    assert classify_error("InternalServerError: Error code: 500 - other") == "infrastructure"

    base = dict(defense="none", suite="slack", user_task="user_task_19",
                injection_task="injection_task_1", condition="bn_mt", rep=0,
                raw_security=None, normalised_security=None, corrected_security=None, utility=None)
    rows = [
        dict(model="gpt-oss-20b", arm="unicode_preserving", status="error", error=_d48_error_text(), **base),
        dict(model="gpt-oss-20b", arm="stock", status="error", error=_d48_error_text(), **base),
        dict(model="gpt-oss-20b", arm="unicode_preserving", status="error", error=_d48_error_text(),
             defense="none", suite="slack", user_task="user_task_18", injection_task="injection_task_1",
             condition="bn_mt", rep=0, raw_security=None, normalised_security=None,
             corrected_security=None, utility=None),
        dict(model="gptoss", arm="unicode_preserving", status="error", error=_d48_error_text(), **base),
        dict(model="gpt-oss-20b", arm="unicode_preserving", status="ok", error=None,
             defense="none", suite="slack", user_task="user_task_19", injection_task="injection_task_1",
             condition="bn_mt", rep=0, raw_security=False, normalised_security=False,
             corrected_security=False, utility=True),
    ]
    df = pd.DataFrame(rows).astype({c: object for c in
                                    ("raw_security", "normalised_security", "corrected_security", "utility", "error")})
    d = apply_rules(df, "failure", "pair")

    exact = d[d.error.notna() & d.model.isin(["gpt-oss-20b", "gptoss"]) & d.arm.eq("unicode_preserving")
              & d.user_task.eq("user_task_19") & d.injection_task.eq("injection_task_1")
              & d.condition.eq("bn_mt")]
    assert len(exact) == 2
    assert (exact.error_class == "model_output").all()
    assert (exact.status == "ok").all()
    assert (exact.exclusion == "model_output_error_counted_as_failure").all()
    assert (exact.normalised_security == False).all()

    wrong_arm = d[(d.model == "gpt-oss-20b") & (d.arm == "stock")]
    assert wrong_arm.error_class.iloc[0] == "infrastructure"
    assert wrong_arm.exclusion.iloc[0] == "infrastructure_missing"

    other = d[d.user_task == "user_task_18"]
    assert other.error_class.iloc[0] == "infrastructure"

    ok = d[d.error.isna()]
    assert len(ok) == 1 and ok.error_class.isna().all() and ok.exclusion.isna().all()


def test_d48_override_skipped_without_arm_column():
    """Stock primary frames have no arm column → D48 override does not fire."""
    row = dict(model="gpt-oss-20b", defense="none", suite="slack", user_task="user_task_19",
               injection_task="injection_task_1", condition="bn_mt", rep=0, status="error",
               error=_d48_error_text(), raw_security=None, normalised_security=None,
               corrected_security=None, utility=None)
    df = pd.DataFrame([row]).astype({c: object for c in
                                     ("raw_security", "normalised_security", "corrected_security", "utility", "error")})
    assert "arm" not in df.columns
    d = apply_rules(df, "failure", "pair")
    assert d.error_class.iloc[0] == "infrastructure"
    assert d.exclusion.iloc[0] == "infrastructure_missing"
    assert d.status.iloc[0] == "error"


def test_apply_error_class_overrides_unit():
    d = pd.DataFrame([
        dict(model="gpt-oss-20b", arm="unicode_preserving", suite="slack", user_task="user_task_19",
             injection_task="injection_task_1", condition="bn_mt", error_class="infrastructure"),
        dict(model="gpt-oss-20b", arm="unicode_preserving", suite="slack", user_task="user_task_18",
             injection_task="injection_task_1", condition="bn_mt", error_class="infrastructure"),
    ])
    err = pd.Series([True, True])
    out = apply_error_class_overrides(d, err)
    assert list(out) == ["model_output", "infrastructure"]
