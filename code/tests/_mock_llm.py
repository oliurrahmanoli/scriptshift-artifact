"""Test-only stand-in for a llama-server OpenAI-compatible endpoint (no model, no network).

Turn 1 (last message from the user): call the first offered tool that has no required
arguments (so AgentDojo really executes a tool and reads environment data); otherwise answer.
After a tool result: return a final text answer. A layer-3 judge prompt gets a fixed
``full`` verdict. /apply-template returns a prompt containing
``template`` so the notebook's template check can be exercised. Results are meaningless and are
never recorded as data.

Optional (D41 tests): every chat request body is kept in ``requests`` (JSON-decoded, i.e. exactly what a
llama-server would see after parsing), and ``script(body) -> message dict | None`` can dictate the
assistant reply (None = the default behaviour above). ``delay_s`` makes every chat call sleep first.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            return self._send({"object": "list", "data": [{"id": "mock-model", "object": "model",
                                                           "created": 0, "owned_by": "test"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        srv = self.server
        if self.path.endswith("/apply-template"):
            return self._send({"prompt": srv.template})
        if not self.path.endswith("/chat/completions"):
            return self._send({"error": "not found"}, 404)
        with srv.lock:
            srv.calls += 1
            srv.requests.append(body)
        if srv.delay_s:
            import time
            time.sleep(srv.delay_s)
        msgs = body.get("messages", [])
        tools = body.get("tools") or []
        last = msgs[-1]["role"] if msgs else "user"
        msg = {"role": "assistant", "content": "Done. (mock answer)"}
        finish = "stop"
        first = msgs[0].get("content") if msgs else ""
        if not tools and isinstance(first, str) and "Reply with strict JSON" in first:
            # layer-3 judge prompt (scoring/judge.py JUDGE_PROMPT): a fixed, meaningless verdict
            msg = {"role": "assistant", "content": '{"verdict": "full", "reason": "mock judge"}'}
        if last == "user" and tools:
            for t in tools:
                fn = t.get("function", {})
                if not (fn.get("parameters") or {}).get("required"):
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_0", "type": "function",
                         "function": {"name": fn["name"], "arguments": "{}"}}]}
                    finish = "tool_calls"
                    break
        if srv.script is not None:
            scripted = srv.script(body)
            if scripted is not None:
                msg = scripted
                finish = "tool_calls" if scripted.get("tool_calls") else "stop"
        self._send({"id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
                    "model": "mock-model",
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


class MockLLM:
    def __init__(self, template="<|im_start|>assistant\n<think></think>", script=None, delay_s=0.0):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.httpd.daemon_threads = True
        self.httpd.template = template
        self.httpd.script = script
        self.httpd.delay_s = delay_s
        self.httpd.requests = []
        self.httpd.calls = 0
        self.httpd.lock = threading.Lock()
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def calls(self):
        return self.httpd.calls

    @property
    def requests(self):
        return self.httpd.requests

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
