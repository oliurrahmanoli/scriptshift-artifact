"""Unicode-preserving agent inputs for AgentDojo 0.1.35 — an explicit, reversible monkeypatch
(DECISIONS D40/D41). **Default OFF**: nothing in the stock harness imports this module, and with the
flag off AgentDojo behaves exactly as shipped, so the stock main grid stays reproducible.

What AgentDojo 0.1.35 does to non-ASCII text on its way to the model (audited for D41):

1. ``agentdojo.agent_pipeline.tool_execution.tool_result_to_str`` renders structured tool results
   (pydantic objects and lists of them: e-mails, calendar events, files, transactions, Slack
   messages, ...) with ``dump_fn=yaml.safe_dump``. PyYAML's default ``allow_unicode=False`` writes
   every non-ASCII character as an escape (``\\u0995``, ``\\xE9``, ``\\u2019``) inside a double-quoted
   scalar. Plain-string tool results (``str(result)``) are not touched. -> **patched**: the default
   ``dump_fn`` becomes ``yaml.safe_dump(..., allow_unicode=True)``; every other dump option (width,
   style, key order) is unchanged.
2. ``agentdojo.agent_pipeline.llms.openai_llm._tool_call_to_openai`` re-serialises the agent's own
   earlier tool-call arguments with ``json.dumps(args)`` (``ensure_ascii=True``), so non-ASCII
   arguments are replayed to the server as ``\\uXXXX`` JSON escapes. llama.cpp b11344 decodes them for
   templates whose capabilities report object arguments (common/chat.cpp ``func_args_not_string``) and
   passes the escaped string through verbatim otherwise. -> **patched**: ``ensure_ascii=False``
   (JSON-equivalent, so the arguments are unchanged as data).

Checked and left alone (not on our path or lossless): ``tool_output_format="json"`` (would use
``json.dumps``; our pipelines use the YAML default, which ``check_pipeline`` asserts);
``LocalLLM``/``PromptingLLM`` ``json.dumps`` (not used, we use ``OpenAILLM``); environment loading
(``read_suite_file`` ``yaml.dump`` + ``str.format`` injection + ``yaml.safe_load`` round-trips, the
injection is inserted into YAML text and parsed back as Unicode); the system message and user prompts
(passed as strings); the OpenAI client's JSON transport (escapes are decoded by the server).

Usage::

    from runners import unicode_patch as U
    with U.unicode_preserving(True):      # or U.enable() / U.disable()
        ...run episodes...
    U.self_check(expect_enabled=False)    # stock behaviour is back
"""
from __future__ import annotations

import contextlib
import json
import re

import yaml

PATCH_ID = "D41-unicode-preserving-v1"
AGENTDOJO_VERSION = "0.1.35"
_BENGALI = re.compile(r"[\u0980-\u09ff]")
_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|U[0-9a-fA-F]{8})")
PROBE_TEXT = "বাংলা লিপি ‘test’ café"   # Bengali + U+2018/U+2019 + Latin-1 é

_STATE = {"enabled": False, "orig_defaults": None, "orig_tool_call_to_openai": None}


def safe_dump_unicode(data, **kw):
    """``yaml.safe_dump`` with ``allow_unicode=True``; every other option is PyYAML's default,
    exactly as in the stock formatter."""
    kw.setdefault("allow_unicode", True)
    return yaml.safe_dump(data, **kw)


def _tool_call_to_openai_unicode(tool_call):
    """Copy of agentdojo 0.1.35 ``openai_llm._tool_call_to_openai`` with ``ensure_ascii=False``."""
    from openai.types.chat import ChatCompletionMessageToolCallParam
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args, ensure_ascii=False),
        },
    )


def _modules():
    from agentdojo.agent_pipeline import tool_execution
    from agentdojo.agent_pipeline.llms import openai_llm
    return tool_execution, openai_llm


def _agentdojo_version():
    import importlib.metadata as md
    return md.version("agentdojo")


def is_enabled() -> bool:
    return bool(_STATE["enabled"])


def enable() -> None:
    """Turn the patch ON (idempotent). Refuses on any agentdojo other than the audited 0.1.35 or if
    the targets no longer look like stock (someone else patched them)."""
    if _STATE["enabled"]:
        return
    v = _agentdojo_version()
    if v != AGENTDOJO_VERSION:
        raise RuntimeError(f"unicode_patch audited for agentdojo {AGENTDOJO_VERSION}, found {v}")
    te, ol = _modules()
    if te.tool_result_to_str.__defaults__ != (yaml.safe_dump,):
        raise RuntimeError(f"tool_result_to_str defaults are not stock: {te.tool_result_to_str.__defaults__!r}")
    if ol._tool_call_to_openai.__module__ != ol.__name__:
        raise RuntimeError("openai_llm._tool_call_to_openai is not the stock function")
    _STATE["orig_defaults"] = te.tool_result_to_str.__defaults__
    _STATE["orig_tool_call_to_openai"] = ol._tool_call_to_openai
    # (1) same function object is used by AgentPipeline.from_config, ToolsExecutor's default and the
    #     ground-truth pipeline, so changing its default dump_fn covers every caller at call time.
    te.tool_result_to_str.__defaults__ = (safe_dump_unicode,)
    # (2) looked up as a module global by openai_llm._message_to_openai at call time.
    ol._tool_call_to_openai = _tool_call_to_openai_unicode
    _STATE["enabled"] = True


def disable() -> None:
    """Restore stock AgentDojo behaviour (idempotent)."""
    if not _STATE["enabled"]:
        return
    te, ol = _modules()
    te.tool_result_to_str.__defaults__ = _STATE["orig_defaults"]
    ol._tool_call_to_openai = _STATE["orig_tool_call_to_openai"]
    _STATE.update(enabled=False, orig_defaults=None, orig_tool_call_to_openai=None)


def set_enabled(flag: bool) -> None:
    enable() if flag else disable()


@contextlib.contextmanager
def unicode_preserving(flag: bool = True):
    """Context manager: patch ON (or explicitly OFF) inside, previous state restored on exit."""
    before = is_enabled()
    set_enabled(flag)
    try:
        yield
    finally:
        set_enabled(before)


def describe() -> dict:
    return {"patch_id": PATCH_ID, "enabled": is_enabled(), "agentdojo": AGENTDOJO_VERSION,
            "targets": ["agentdojo.agent_pipeline.tool_execution.tool_result_to_str: default dump_fn "
                        "yaml.safe_dump -> yaml.safe_dump(allow_unicode=True)",
                        "agentdojo.agent_pipeline.llms.openai_llm._tool_call_to_openai: json.dumps(args) "
                        "-> json.dumps(args, ensure_ascii=False)"]}


def _probe_result():
    from agentdojo.default_suites.v1.tools.email_client import Email
    return [Email(id_="0", sender="a@b.c", recipients=["d@e.f"], subject=PROBE_TEXT, body=PROBE_TEXT)]


def classify(text: str) -> str:
    """'script' if the probe's characters arrived as themselves, 'escaped' if as escapes."""
    has_raw = bool(_BENGALI.search(text)) and "é" in text and "’" in text
    has_esc = bool(_ESCAPE.search(text))
    if has_raw and not has_esc:
        return "script"
    if has_esc and not _BENGALI.search(text):
        return "escaped"
    return "mixed"


def render_probe(formatter=None) -> str:
    te, _ = _modules()
    return (formatter or te.tool_result_to_str)(_probe_result())


def self_check(expect_enabled: bool) -> dict:
    """Render a Bengali probe through the live formatter and tool-call serialiser; raise unless the
    result is script (patch ON) / escapes (patch OFF)."""
    from agentdojo.functions_runtime import FunctionCall
    _, ol = _modules()
    want = "script" if expect_enabled else "escaped"
    tool_out = classify(render_probe())
    args = ol._tool_call_to_openai(FunctionCall(function="f", args={"q": PROBE_TEXT}, id="c0"))["function"]["arguments"]
    args_out = classify(args)
    res = {"expect_enabled": expect_enabled, "enabled": is_enabled(), "tool_output": tool_out,
           "tool_call_args": args_out, "patch_id": PATCH_ID}
    if is_enabled() != expect_enabled or tool_out != want or args_out != want:
        raise RuntimeError(f"unicode_patch self-check failed: {res}")
    return res


def _tools_executors(el):
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    if isinstance(el, ToolsExecutor):
        yield el
    for sub in getattr(el, "elements", []) or []:
        yield from _tools_executors(sub)


def check_pipeline(pipe, expect_enabled: bool) -> str:
    """Assert that every ToolsExecutor of a built pipeline renders the probe as expected (catches a
    pipeline built with a non-default formatter, e.g. tool_output_format='json')."""
    ex = list(_tools_executors(pipe))
    if not ex:
        raise RuntimeError("pipeline has no ToolsExecutor")
    want = "script" if expect_enabled else "escaped"
    for e in ex:
        got = classify(render_probe(e.output_formatter))
        if got != want:
            raise RuntimeError(f"pipeline formatter renders the probe as {got!r}, expected {want!r}")
    return want
