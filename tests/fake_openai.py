"""A fake OpenAI-compatible server, answering as llama-server does: /v1/models,
/props, /health and /v1/chat/completions, with canned replies. No model runs.
12d.3: `stream: true` is answered as llama-server streams — its thinking as
`reasoning_content` deltas, then the reply a word at a time, then a last chunk
with `timings` and, when asked, `usage` — and a client that hangs up is
counted.

It runs in a thread on a free port. A test changes the file it serves, asks
for a key, or makes it stop answering after so many answers (a 503 each
time after, as a server that has gone away behind a proxy would).

12m.3: it answers as OpenRouter too, for a test that sends OpenRouter's
address here (tests/test_12m3.py) — its models list with prices and dated
versions (`catalog`), each model's providers (`endpoints`), and a chat
reply's usage with its cost and reasoning tokens, and the thinking as
OpenRouter says it (`reasoning_field = "reasoning"`). Nothing reaches
OpenRouter itself."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
import urllib.request

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

LDA_FILE = "/home/masein/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf"


class FakeServer:
    def __init__(self):
        self.model_path = LDA_FILE
        self.size = 22_900_000_000
        self.ctx = 16384
        self.build = "b6500-91428471f"
        self.key = ""
        self.reply = lambda body: "A plain answer, from the served model."
        self.reasoning = ""
        # 12i.4: its usage, as llama-server reports it: the words it wrote,
        # thinking included, unless a test says otherwise
        self.tokens = None
        # 12f.3 addendum: llama-server's timings, MTP's drafts among them
        self.timings = None
        self.stop_after: int | None = None      # answers, then it stops answering
        self.delay_s = 0.0
        self.requests: list[dict] = []           # every chat body, as sent
        self.probes: list[dict] = []             # 12p.1: a pin's one-token question, apart
        self.auth: list[str] = []                # every Authorization header seen
        self.answered = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.on_request = None                   # called with each chat body
        self.stream_delay_s = 0.0                # 12d.3: between streamed pieces
        self.hung_up = 0                         # streams the client closed early
        # 12m.3: as OpenRouter — its list, a model's providers, and a reply's cost
        self.catalog: list[dict] | None = None   # set: /v1/models is OpenRouter's list
        self.endpoints: list[dict] = []          # /v1/models/<id>/endpoints
        self.prompt_tokens = 10
        self.cost = None                         # dollars a reply reports (or a function)
        self.reasoning_field = "reasoning_content"
        self.reasoning_tokens = None             # a function: completion_tokens_details
        self.provider = ""                       # the provider a reply names
        self._lock = threading.Lock()
        app = FastAPI()

        def refused(request: Request):
            h = request.headers.get("authorization", "")
            self.auth.append(h)
            if self.key and h != f"Bearer {self.key}":
                return JSONResponse({"error": {"message": "Invalid API Key"}}, status_code=401)
            return None

        @app.get("/v1/models/{mid:path}/endpoints")
        def endpoints(mid: str, request: Request):
            return refused(request) or {"data": {"id": mid, "endpoints": self.endpoints}}

        @app.get("/v1/models")
        def models(request: Request):
            if self.catalog is not None:
                return refused(request) or {"data": self.catalog}
            return refused(request) or {"object": "list", "data": [{
                "id": self.model_path.rsplit("/", 1)[-1], "object": "model",
                "owned_by": "llamacpp",
                "meta": {"n_ctx_train": 262144, "size": self.size, "n_params": 35_000_000_000}}]}

        @app.get("/props")
        def props(request: Request):
            return refused(request) or {
                "model_path": self.model_path, "build_info": self.build,
                "default_generation_settings": {"n_ctx": self.ctx}}

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        def chat(request: Request, body: dict):
            no = refused(request)
            if no:
                return no
            if body.get("max_tokens") == 1 and (body.get("messages") or [{}])[-1].get(
                    "content") == "ping":
                # 12p.1: asked when a model from OpenRouter is pinned, not by a run
                self.probes.append(body)
                return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "p"},
                                     "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}}
            with self._lock:
                if self.stop_after is not None and self.answered >= self.stop_after:
                    return JSONResponse({"error": {"message": "unavailable"}}, status_code=503)
                self.requests.append(body)
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if self.on_request:
                self.on_request(body)
            if body.get("stream"):
                return self._stream(request, body)
            try:
                if self.delay_s:
                    time.sleep(self.delay_s)
                content = self.reply(body)
                think = self.reasoning(body) if callable(self.reasoning) else self.reasoning
                msg = {"role": "assistant", "content": content}
                if think:
                    msg[self.reasoning_field] = think
                used = (self.tokens(body) if callable(self.tokens)
                        else len(content.split()) + len((think or "").split()))
                with self._lock:
                    self.answered += 1
                out = {"id": "chatcmpl-1", "object": "chat.completion",
                       "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": used}}
                if callable(self.reasoning_tokens):
                    out["usage"]["completion_tokens_details"] = {
                        "reasoning_tokens": self.reasoning_tokens(body)}
                cost = self.cost(body) if callable(self.cost) else self.cost
                if cost is not None:
                    out["usage"]["cost"] = cost
                if self.provider:
                    out["provider"] = self.provider
                if callable(self.timings):
                    out["timings"] = self.timings(body)
                return out
            finally:
                with self._lock:
                    self.in_flight -= 1

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                     log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("the fake server did not come up")

    def _stream(self, request: Request, body: dict):
        content = self.reply(body)
        think = self.reasoning(body) if callable(self.reasoning) else self.reasoning
        used = (self.tokens(body) if callable(self.tokens)
                else len(content.split()) + len((think or "").split()))

        def chunk(delta: dict, **more) -> str:
            return "data: " + json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                                          "choices": [{"index": 0, "delta": delta,
                                                       "finish_reason": more.pop("finish", None)}],
                                          **more}) + "\n\n"

        async def events():
            try:
                pieces = [("reasoning_content", w) for w in re.findall(r"\S+\s*", think or "")]
                pieces += [("content", w) for w in re.findall(r"\S+\s*|\s+", content)]
                yield chunk({"role": "assistant", "content": None})
                for k, w in pieces:
                    if await request.is_disconnected():
                        self.hung_up += 1
                        return
                    yield chunk({k: w})
                    await asyncio.sleep(self.stream_delay_s)
                last = {"finish": "stop"}
                if callable(self.timings):
                    last["timings"] = self.timings(body)
                if (body.get("stream_options") or {}).get("include_usage"):
                    last["usage"] = {"prompt_tokens": 10, "completion_tokens": used}
                yield chunk({}, **last)
                yield "data: [DONE]\n\n"
                with self._lock:
                    self.answered += 1
            finally:
                with self._lock:
                    self.in_flight -= 1
        return StreamingResponse(events(), media_type="text/event-stream")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(5)


def nothing_listening() -> str:
    """an address where nothing answers"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"
