"""D41: the Unicode-preserving monkeypatch (runners/unicode_patch.py), default OFF.

Proves on the real AgentDojo 0.1.35 pipeline (OpenAILLM -> ToolsExecutor -> HTTP -> test-only mock
endpoint) that a Bengali injection reaches the model as \\uXXXX escapes with the flag OFF (stock) and as
Bengali script with it ON, for tool output and for replayed tool-call arguments; and audits the other
AgentDojo code paths that touch non-ASCII text."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runners import unicode_patch as U  # noqa: E402
from tests._mock_llm import MockLLM  # noqa: E402

BN = re.compile(r"[\u0980-\u09ff]")
BN_ESC = re.compile(r"\\u09[89a-fA-F][0-9a-fA-F]")
PAY = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))


def _ws(s):
    return " ".join(s.split())


def _wrapped(cond, key):
    w = PAY["wrappers"]["W1"][cond]
    return w.format(goal=PAY["conditions"][cond]["goals"][key], user="Emma Johnson", model="the AI assistant")


@pytest.fixture(autouse=True)
def _always_off_after():
    yield
    U.disable()
    assert not U.is_enabled()


def test_default_off_and_formatter_escapes_vs_script():
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    assert not U.is_enabled() and tool_result_to_str.__defaults__ == (yaml.safe_dump,)
    probe = U._probe_result()
    off = tool_result_to_str(probe)
    assert BN_ESC.search(off) and not BN.search(off) and "\\xE9" in off and "\\u2019" in off
    with U.unicode_preserving(True):
        on = tool_result_to_str(probe)
        assert BN.search(on) and not BN_ESC.search(on) and "café" in on and "’" in on
        assert yaml.safe_load(on)[0]["body"] == U.PROBE_TEXT
        # plain-string tool results were never touched
        assert tool_result_to_str("বাংলা") == "বাংলা"
    assert tool_result_to_str(probe) == off                     # restored exactly
    assert yaml.safe_load(off)[0]["body"] == U.PROBE_TEXT       # the escapes are lossless YAML, just not script


def test_enable_disable_restore_idempotent_and_guarded(monkeypatch):
    from agentdojo.agent_pipeline import tool_execution as te
    from agentdojo.agent_pipeline.llms import openai_llm as ol
    stock_fn = ol._tool_call_to_openai
    U.enable()
    U.enable()
    assert te.tool_result_to_str.__defaults__ == (U.safe_dump_unicode,)
    assert ol._tool_call_to_openai is U._tool_call_to_openai_unicode
    assert U.self_check(True)["tool_output"] == "script"
    U.disable()
    U.disable()
    assert te.tool_result_to_str.__defaults__ == (yaml.safe_dump,) and ol._tool_call_to_openai is stock_fn
    assert U.self_check(False)["tool_call_args"] == "escaped"
    with pytest.raises(RuntimeError):
        U.self_check(True)
    with pytest.raises(ZeroDivisionError):
        with U.unicode_preserving(True):
            1 / 0
    assert not U.is_enabled() and te.tool_result_to_str.__defaults__ == (yaml.safe_dump,)
    monkeypatch.setattr(te.tool_result_to_str, "__defaults__", (yaml.dump,))
    with pytest.raises(RuntimeError, match="not stock"):
        U.enable()
    monkeypatch.undo()
    monkeypatch.setattr(U, "_agentdojo_version", lambda: "0.1.36")
    with pytest.raises(RuntimeError, match="audited"):
        U.enable()


def _episode_requests(enabled: bool, cond="bn_llm", key="workspace/injection_task_0"):
    """One real AgentDojo episode against the mock: turn 1 calls get_received_emails (structured
    list[Email] holding the injection), turn 2 calls search_emails with a Bengali query (an argument the
    next request replays), turn 3 answers. Returns the JSON-decoded request bodies the 'server' got."""
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites
    from runners.endpoints import build_local_pipeline
    suite = get_suites("v1.2.2")["workspace"]
    text = _wrapped(cond, key)
    env = suite.load_and_inject_default_environment({v: text for v in suite.get_injection_vector_defaults()})

    def script(body):
        n_tool = sum(1 for m in body["messages"] if m["role"] == "tool")
        if n_tool == 0:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_received_emails", "arguments": "{}"}}]}
        if n_tool == 1:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "type": "function", "function": {
                    "name": "search_emails", "arguments": json.dumps({"query": "বাংলা সভা"}, ensure_ascii=False)}}]}
        return {"role": "assistant", "content": "done"}

    with MockLLM(script=script) as mock, U.unicode_preserving(enabled):
        pipe = build_local_pipeline(base_url=f"http://127.0.0.1:{mock.port}/v1", model_id="mock-model")
        U.check_pipeline(pipe, enabled)
        pipe.query("Summarise my emails.", FunctionsRuntime(suite.tools), env)
        return list(mock.requests), text


def _tool_texts(body):
    out = []
    for m in body["messages"]:
        if m["role"] == "tool":
            c = m["content"]
            out.append(c if isinstance(c, str) else " ".join(b.get("text", "") for b in c))
    return out


def _args(body):
    return [tc["function"]["arguments"] for m in body["messages"] if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])]


@pytest.mark.parametrize("cond", ["bn_llm", "bn_mt"])
def test_bengali_reaches_model_as_escapes_off_and_script_on(cond):
    goal = PAY["conditions"][cond]["goals"]["workspace/injection_task_0"]
    reqs_off, _ = _episode_requests(False, cond)
    assert len(reqs_off) == 3
    tool_off = _tool_texts(reqs_off[-1])[0]
    assert "<INFORMATION>" in tool_off and BN_ESC.search(tool_off) and not BN.search(tool_off)
    assert _ws(goal) in _ws(json.dumps(yaml.safe_load(tool_off), ensure_ascii=False, default=str))   # same text, escaped
    args_off = _args(reqs_off[-1])[-1]
    assert BN_ESC.search(args_off) and not BN.search(args_off)

    reqs_on, _ = _episode_requests(True, cond)
    tool_on = _tool_texts(reqs_on[-1])[0]
    assert BN.search(tool_on) and not BN_ESC.search(tool_on)
    # the goal arrives as script; YAML may fold long lines exactly as it does for English
    assert _ws(goal) in _ws(tool_on.replace("\n    ", " ").replace("\n      ", " ")) or \
        _ws(goal) in _ws(json.dumps(yaml.safe_load(tool_on), ensure_ascii=False, default=str))
    assert set(_ws(goal)) - set(" ") <= set(tool_on)                      # every character present as itself
    args_on = _args(reqs_on[-1])[-1]
    assert "বাংলা সভা" in args_on and not BN_ESC.search(args_on)
    assert json.loads(args_on) == json.loads(args_off)                    # same arguments as data
    # system message and user prompt are identical in both arms
    assert reqs_on[0]["messages"][:2] == reqs_off[0]["messages"][:2]


def test_english_and_banglish_requests_identical_under_both_settings():
    """ASCII injections: the patch changes nothing the model sees (apart from non-ASCII env text)."""
    for cond in ("en", "banglish_llm"):
        off, _ = _episode_requests(False, cond, "workspace/injection_task_0")
        on, _ = _episode_requests(True, cond, "workspace/injection_task_0")
        t_off, t_on = _tool_texts(off[-1])[0], _tool_texts(on[-1])[0]
        strip = lambda s: re.sub(r"timestamp: .*", "", s)
        inf = lambda s: s[s.index("<INFORMATION>"): s.index("</INFORMATION>")]
        assert inf(strip(t_off)) == inf(strip(t_on)), cond


def test_injected_payloads_survive_environment_loading_all_suites():
    """Stock path (unpatched): read_suite_file yaml.dump + str.format + yaml.safe_load keeps every
    Bengali character of every bn_mt/bn_llm goal (whitespace folding aside, as for English)."""
    from agentdojo.task_suite.load_suites import get_suites
    suites = get_suites("v1.2.2")
    for cond in ("bn_mt", "bn_llm"):
        for key, goal in PAY["conditions"][cond]["goals"].items():
            sname = key.split("/")[0]
            s = suites[sname]
            vecs = list(s.get_injection_vector_defaults())
            env = s.load_and_inject_default_environment({vecs[0]: _wrapped(cond, key)})
            dumped = json.dumps(env.model_dump(mode="json"), ensure_ascii=False)
            assert _ws(goal) in _ws(dumped.replace("\\n", " ")), (cond, key)


def test_yaml_allow_unicode_leaves_no_escapes_in_any_payload():
    for cond in ("bn_mt", "bn_llm"):
        texts = list(PAY["conditions"][cond]["goals"].values()) + [PAY["wrappers"]["W1"][cond]]
        for t in texts:
            off = yaml.safe_dump({"body": t})
            on = U.safe_dump_unicode({"body": t})
            assert BN_ESC.search(off) and not BN.search(off)
            assert not re.search(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})", on), t[:40]
            assert yaml.safe_load(on)["body"] == t == yaml.safe_load(off)["body"]


def test_ascii_payloads_and_system_message():
    for cond in ("en", "banglish_llm"):
        assert all(v.isascii() for v in PAY["conditions"][cond]["goals"].values())
        assert PAY["wrappers"]["W1"][cond].isascii()
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    assert load_system_message(None).isascii()


AUDITED_CALLS = {   # agentdojo 0.1.35 call sites that serialise text (D41 audit); a change must be re-audited
    "agent_pipeline/tool_execution.py": 2,   # yaml.safe_dump default dump_fn -> PATCHED (+1 docstring line)
    "agent_pipeline/llms/openai_llm.py": 1,  # json.dumps(tool_call.args)                -> PATCHED
    "agent_pipeline/agent_pipeline.py": 1,   # json.dumps formatter (tool_output_format=json; unused)
    "task_suite/task_suite.py": 2,           # yaml.dump of suite files, then format + safe_load (lossless)
    "agent_pipeline/llms/local_llm.py": 3,   # LocalLLM (unused; we use OpenAILLM)
    "agent_pipeline/llms/prompting_llm.py": 1,  # tool docs for prompting LLMs (unused)
    "logging.py": 1,                         # agentdojo TraceLogger (unused; our write_trace is ensure_ascii=False)
}


def test_audit_of_agentdojo_serialisation_call_sites():
    import agentdojo
    base = Path(agentdojo.__file__).parent
    pat = re.compile(r"yaml\.(safe_)?dump\b|json\.dumps\b")
    found = {}
    for p in base.rglob("*.py"):
        if "data" in p.relative_to(base).parts:
            continue
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                if pat.search(line) and not line.lstrip().startswith(("#", "from", "import")))
        if n:
            found[p.relative_to(base).as_posix()] = n
    assert found == AUDITED_CALLS


def test_our_pipeline_uses_the_yaml_default_formatter():
    from runners.endpoints import build_local_pipeline
    pipe = build_local_pipeline(base_url="http://127.0.0.1:9/v1", model_id="x")
    assert U.check_pipeline(pipe, False) == "escaped"
    with U.unicode_preserving(True):
        assert U.check_pipeline(pipe, True) == "script"     # same, already-built pipeline object
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
    assert PipelineConfig.model_fields["tool_output_format"].default is None


def test_stock_harness_untouched_and_never_imports_the_patch():
    import build_main_grid_notebook as B
    import build_uni_rerun_notebook as UB
    assert B.build_tarball()[1]["sha256"] == UB.STOCK_EMBED_SHA256      # main grid stays reproducible
    for rel in B.embed_file_list():
        if rel.endswith(".py"):
            assert "unicode_patch" not in (ROOT / rel).read_text(encoding="utf-8"), rel
