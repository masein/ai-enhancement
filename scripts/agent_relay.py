"""18: the relay — the agent's one door to the model (docs/AGENT-RUNS-design.md
§ 4). It listens on 127.0.0.1 only, takes only `POST /v1/chat/completions`,
adds the llama-server's key (which the agent, Harbor's config, the
trajectories and the logs never see), sets the run's sampling on every
request, and passes the server's answer back as it came: a window outgrown
stays the server's own 400, which mini-swe-agent reads as the window outgrown.
Each answer's tokens and seconds are written to a usage file, never the key.

Standard library only: it runs inside scripts/agent_run.py on the server."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PATH = "/v1/chat/completions"
MAX_BODY = 32 * 1024 * 1024          # a conversation of 262,144 tokens is a few MB
UPSTREAM_TIMEOUT_S = 3600            # one reply of a long thought at ~85 tokens a second
REFUSED = "only the chat request goes through this relay"


class Relay:
    """`upstream`: the server's OpenAI-compatible base (…/v1); `key` its key"""

    def __init__(self, upstream: str, key: str, sampling: dict, usage: Path | None = None,
                 port: int = 0, max_body: int = MAX_BODY, timeout: float = UPSTREAM_TIMEOUT_S):
        self.upstream = upstream.rstrip("/")
        self._key = key
        self.sampling = dict(sampling)
        self.usage = usage
        self.max_body = max_body
        self.timeout = timeout
        self._lock = threading.Lock()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):                  # nothing to stderr: no URL, no body
                pass

            def _send(self, status: int, body: bytes, ctype: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _refuse(self, status: int = 403, why: str = REFUSED):
                # the body was never read: drop the connection afterwards
                self.close_connection = True
                self._send(status, json.dumps({"error": {"message": why,
                                                         "type": "relay_refused"}}).encode())

            def do_GET(self):
                self._refuse()

            do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_GET

            def do_POST(self):
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._refuse()
                if self.path.split("?", 1)[0] != PATH:
                    return self._refuse()
                try:
                    n = int(self.headers.get("Content-Length") or "")
                except ValueError:
                    return self._refuse(411, "a length is required")
                if n <= 0 or n > relay.max_body:
                    return self._refuse(413, "the request is too large")
                try:
                    body = json.loads(self.rfile.read(n))
                except ValueError:
                    return self._send(400, json.dumps({"error": {"message": "not JSON"}}).encode())
                if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
                    return self._send(400, json.dumps({"error": {"message": "no messages"}})
                                      .encode())
                status, raw, ctype, secs = relay.forward(body)
                relay.record(self.headers.get("X-Agent-Trial", ""), status, raw, secs)
                self._send(status, raw, ctype)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> "Relay":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def forward(self, body: dict) -> tuple[int, bytes, str, float]:
        """the request, with the run's sampling, to the server with its key"""
        body = {**body, **self.sampling, "stream": False}
        body.pop("stream_options", None)
        req = urllib.request.Request(self.upstream + "/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self._key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "application/json"), \
                    time.time() - t0
        except urllib.error.HTTPError as e:
            # the server's own refusal, as it gave it (a window outgrown is a 400)
            return e.code, e.read(), e.headers.get("Content-Type", "application/json"), \
                time.time() - t0
        except (urllib.error.URLError, OSError) as e:
            why = f"the model's server didn't answer the relay ({getattr(e, 'reason', e)})"
            return 502, json.dumps({"error": {"message": why, "type": "relay_upstream"}}).encode(), \
                "application/json", time.time() - t0

    def record(self, trial: str, status: int, raw: bytes, secs: float) -> None:
        """each answer's tokens and seconds, for the run's numbers"""
        if self.usage is None:
            return
        try:
            usage = (json.loads(raw).get("usage") or {}) if status == 200 else {}
        except (ValueError, AttributeError):
            usage = {}
        line = {"at": round(time.time(), 3), "trial": trial[:200], "status": status,
                "seconds": round(secs, 2), "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
        with self._lock:
            self.usage.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
