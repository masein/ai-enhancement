"""A fake OpenAI-compatible server, answering as llama-server does: /v1/models,
/props, /health and /v1/chat/completions, with canned replies. No model runs.

It runs in a thread on a free port. A test changes the file it serves, asks
for a key, or makes it stop answering after so many answers (a 503 each
time after, as a server that has gone away behind a proxy would)."""

from __future__ import annotations

import socket
import threading
import time
import urllib.request

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
        self.stop_after: int | None = None      # answers, then it stops answering
        self.delay_s = 0.0
        self.requests: list[dict] = []           # every chat body, as sent
        self.auth: list[str] = []                # every Authorization header seen
        self.answered = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.on_request = None                   # called with each chat body
        self._lock = threading.Lock()
        app = FastAPI()

        def refused(request: Request):
            h = request.headers.get("authorization", "")
            self.auth.append(h)
            if self.key and h != f"Bearer {self.key}":
                return JSONResponse({"error": {"message": "Invalid API Key"}}, status_code=401)
            return None

        @app.get("/v1/models")
        def models(request: Request):
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
            with self._lock:
                if self.stop_after is not None and self.answered >= self.stop_after:
                    return JSONResponse({"error": {"message": "unavailable"}}, status_code=503)
                self.requests.append(body)
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if self.on_request:
                self.on_request(body)
            try:
                if self.delay_s:
                    time.sleep(self.delay_s)
                msg = {"role": "assistant", "content": self.reply(body)}
                if self.reasoning:
                    msg["reasoning_content"] = self.reasoning
                with self._lock:
                    self.answered += 1
                return {"id": "chatcmpl-1", "object": "chat.completion",
                        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
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
