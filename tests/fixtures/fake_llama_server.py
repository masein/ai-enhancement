#!/usr/bin/env python3
"""17: a stand-in for llama-server, as scripts/remote_gguf.py starts it from a
tarball: the same command line, /health, /v1/models, /props, /slots and
/v1/chat/completions, the standard library only. No model runs.

It answers a GPQA-style question with "ANSWER: X", X the option whose text
begins "right" — or, for a question with "hard" in it, the first that
doesn't — and with its thinking apart (reasoning_content) when the chat
template is told to think. FAKE_LLAMA_LOG is a file each request's body is
appended to; FAKE_LLAMA_DIE_AFTER=K stops the process after K answers, as a
box that is stopped does; FAKE_LLAMA_LOAD_S is how long it "loads" before
/health says ok."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "version: 6543 (abc1234)\nbuilt with cc (fake) for x86_64-linux-gnu"


def arg(argv: list[str], *names: str, default: str = "") -> str:
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print(VERSION, file=sys.stderr)
        return 0
    model = arg(argv, "-m", "--model")
    port = int(arg(argv, "--port", default="8080"))
    ctx = int(arg(argv, "-c", "--ctx-size", default="4096"))
    slots = int(arg(argv, "-np", "--parallel", default="1"))
    log = os.environ.get("FAKE_LLAMA_LOG", "")
    die_after = int(os.environ.get("FAKE_LLAMA_DIE_AFTER") or 0)
    ready_at = time.time() + float(os.environ.get("FAKE_LLAMA_LOAD_S") or 0)
    lock = threading.Lock()
    count = {"n": 0}
    print(f"main: loading model {os.path.basename(model)}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):            # quiet
            pass

        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                if time.time() < ready_at:
                    return self._send(503, {"error": {"message": "Loading model"}})
                return self._send(200, {"status": "ok"})
            if self.path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{
                    "id": os.path.basename(model), "object": "model", "owned_by": "llamacpp",
                    "meta": {"n_ctx_train": 262144, "size": 21_000_000_000}}]})
            if self.path == "/props":
                return self._send(200, {"model_path": model, "build_info": "b6543-abc1234",
                                        "total_slots": slots,
                                        "chat_template": "{# a fake template #}",
                                        "default_generation_settings": {"n_ctx": ctx // slots}})
            if self.path == "/slots":
                return self._send(200, [{"id": i, "speculative": False} for i in range(slots)])
            return self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            if log:
                with lock, open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(body) + "\n")
            text = (body.get("messages") or [{}])[-1].get("content") or ""
            opts = dict(re.findall(r"(?m)^([ABCD])\) (.*)$", text))
            right = next((k for k, v in opts.items() if v.startswith("right")), "A")
            pick = (next(k for k in "ABCD" if k != right) if "hard" in text else right)
            on = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
            msg = {"role": "assistant", "content": f"The options compared.\nANSWER: {pick}"}
            if on:
                msg["reasoning_content"] = "Weighing each option in turn."
            with lock:
                count["n"] += 1
                n = count["n"]
            self._send(200, {"id": "chatcmpl-1", "object": "chat.completion",
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 50, "completion_tokens": 12}})
            if die_after and n >= die_after:
                print(f"fake: stopping after {n} answers", flush=True)
                os._exit(3)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"main: server is listening on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
