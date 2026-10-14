import pytest

import attacks
from attacks.multilingual_attack import CONDITIONS, load_payloads
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.task_suite.load_suites import get_suites


def test_all_conditions_registered():
    for c in CONDITIONS:
        assert c in ATTACKS, f"condition {c} not registered"
    for c in ["en_W2", "bn_llm_W2", "banglish_llm_W2"]:
        assert c in ATTACKS


def _banking():
    return get_suites("v1.2.2")["banking"]


class _Pipe:
    name = "test/pipe"

    def query(self, *a, **k):
        raise NotImplementedError


def test_en_attack_text_has_goal_and_names():
    suite = _banking()
    pipe = _Pipe()
    atk = load_attack("en", suite, pipe)
    utask = suite.get_user_task_by_id("user_task_0")
    itask = suite.get_injection_task_by_id("injection_task_7")  # change password
    out = atk.attack(utask, itask)
    assert out, "no injection candidates"
    text = next(iter(out.values()))
    assert "Emma Johnson" in text
    assert "the AI assistant" in text
    # identifier/literal stays English/ASCII
    assert "new_password" in text


def test_bn_mt_attack_runs_and_preserves_identifiers():
    suite = _banking()
    atk = load_attack("bn_mt", suite, _Pipe())
    utask = suite.get_user_task_by_id("user_task_0")
    itask = suite.get_injection_task_by_id("injection_task_7")
    text = next(iter(atk.attack(utask, itask).values()))
    # Bengali script present in the wrapper/goal
    assert any("\u0980" <= ch <= "\u09ff" for ch in text)
    # literal preserved verbatim
    assert "new_password" in text


def test_llm_conditions_refuse_until_generated():
    # codemix_llm is still entirely TODO_LLM (out of the fast-track scope, D24.11)
    suite = _banking()
    atk = load_attack("codemix_llm", suite, _Pipe())
    utask = suite.get_user_task_by_id("user_task_0")
    itask = suite.get_injection_task_by_id("injection_task_7")
    with pytest.raises(ValueError):
        atk.attack(utask, itask)


def test_frozen_llm_conditions_run_and_keep_literals():
    # D26: bn_llm / banglish_llm frozen from the Gemma run; banking/injection_task_5 accepted
    suite = _banking()
    utask = suite.get_user_task_by_id("user_task_0")
    itask = suite.get_injection_task_by_id("injection_task_5")
    for cond, bengali in (("bn_llm", True), ("banglish_llm", False)):
        text = next(iter(load_attack(cond, suite, _Pipe()).attack(utask, itask).values()))
        assert "US133000000121212121212" in text
        assert any("\u0980" <= ch <= "\u09ff" for ch in text) == bengali
    # D27 regen: bn_llm banking/injection_task_7 regenerated; literal new_password kept verbatim
    text = next(iter(load_attack("bn_llm", suite, _Pipe()).attack(
        utask, suite.get_injection_task_by_id("injection_task_7")).values()))
    assert "new_password" in text
    assert any("\u0980" <= ch <= "\u09ff" for ch in text)


def test_regenerated_llm_item_runs():
    # D27 regen: slack/injection_task_2 (previously pending in bn_llm) now filled and runs
    suite = get_suites("v1.2.2")["slack"]
    atk = load_attack("bn_llm", suite, _Pipe())
    utask = suite.get_user_task_by_id("user_task_1")
    itask = suite.get_injection_task_by_id("injection_task_2")
    text = next(iter(atk.attack(utask, itask).values()))
    assert any("\u0980" <= ch <= "\u09ff" for ch in text)


def test_payloads_structure():
    pay = load_payloads()
    assert set(["en", "bn_mt", "bn_llm", "banglish_llm",
                "banglish_translit", "codemix_llm"]).issubset(pay["conditions"])
    # en fully authored
    assert all(v not in ("", "TODO_LLM", "TODO_NATIVE", "TODO_MT", "TODO_TRANSLIT")
               for v in pay["conditions"]["en"]["goals"].values())
    assert len(pay["conditions"]["en"]["goals"]) == 35


def test_backcompat_alias_resolves():
    from attacks.multilingual_attack import CONDITION_ALIASES
    suite = _banking()
    # an old name still loads and refuses (its *_llm target is a TODO_LLM sentinel)
    atk = load_attack("codemix_native", suite, _Pipe())
    assert atk._cond == "codemix_llm"
    utask = suite.get_user_task_by_id("user_task_0")
    itask = suite.get_injection_task_by_id("injection_task_7")
    with pytest.raises(ValueError):
        atk.attack(utask, itask)
