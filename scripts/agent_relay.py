"""18: the relay — the agent's one door to the model (docs/AGENT-RUNS-design.md
§ 4). It listens on 127.0.0.1 only, takes only `POST /v1/chat/completions`,
adds the llama-server's key (which the agent, Harbor's config, the
trajectories and the logs never see), sets the run's sampling on every
request, and passes the server's answer back as it came: a window outgrown
stays the server's own 400, which mini-swe-agent reads as the window outgrown.
Each answer's tokens and seconds are written to a usage file, never the key.

When the server can't be reached, the relay asks its /health: failing that
too, the answer is a 503 that says `relay_server_down` — an error of ours,
and the task is asked again (18b point 2). One bad reply from a server that
is up is the model's.

A reply is capped at MAX_REPLY_TOKENS (18b point 11), which fits the
smallest window a run takes. When a task's time is up, its agent asks
`POST /v1/agent/abort` (with its trial's header): the request it has in
flight is cut, and llama-server stops writing a reply nobody waits for.
Each answer's usage line keeps the server's `timings` (cache_n, prompt_n):
how much of the prompt it took from its cache (18b point 14).

Standard library only: it runs inside scripts/agent_run.py on the server."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PATH = "/v1/chat/completions"
MAX_BODY = 32 * 1024 * 1024          # a conversation of 262,144 tokens is a few MB
UPSTREAM_TIMEOUT_S = 3600            # one reply of a long thought at ~85 tokens a second
REFUSED = "only the chat request goes through this relay"
DOWN = "relay_server_down"
ABORT = "/v1/agent/abort"            # a trial's request in flight, cut (from 127.0.0.1 only)
ABORTED = "relay_aborted"
# the longest reply: Qwen's own output length for most tasks, and well inside
# the smallest window a run takes (131,072)
MAX_REPLY_TOKENS = 32_768
TIMINGS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


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
        self._inflight: dict[str, set] = {}      # trial → its connections to the server
        self.records: list[dict] = []            # the last answers' usage lines
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
                if self.path.split("?", 1)[0] == ABORT:
                    n = relay.abort(self.headers.get("X-Agent-Trial", ""))
                    self.close_connection = True
                    return self._send(200, json.dumps({"aborted": n}).encode())
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
                trial = self.headers.get("X-Agent-Trial", "")
                status, raw, ctype, secs = relay.forward(body, trial)
                relay.record(trial, status, raw, secs)
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

    def forward(self, body: dict, trial: str = "") -> tuple[int, bytes, str, float]:
        """the request, with the run's sampling and the reply's cap, to the
        server with its key — on a connection the trial's abort can cut"""
        body = {**body, **self.sampling, "stream": False}
        body.pop("stream_options", None)
        asked = body.get("max_tokens")
        body["max_tokens"] = (min(int(asked), MAX_REPLY_TOKENS)
                              if isinstance(asked, int) and asked > 0 else MAX_REPLY_TOKENS)
        body.pop("max_completion_tokens", None)
        u = urllib.parse.urlsplit(self.upstream + "/chat/completions")
        conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(
            u.hostname, u.port, timeout=self.timeout)
        with self._lock:
            self._inflight.setdefault(trial, set()).add(conn)
        t0 = time.time()
        try:
            conn.request("POST", u.path, body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {self._key}"})
            r = conn.getresponse()
            raw = r.read()
            ctype = r.getheader("Content-Type", "application/json")
            if r.status in (502, 503, 504) and not self.healthy():
                # it says it can't serve, and its health agrees
                why = f"the model's server answered HTTP {r.status} and its health check failed"
                return 503, json.dumps({"error": {"message": why, "type": DOWN}}).encode(), \
                    "application/json", time.time() - t0
            # the server's own answer or refusal, as it gave it (a window
            # outgrown is a 400)
            return r.status, raw, ctype, time.time() - t0
        except (OSError, http.client.HTTPException) as e:
            if getattr(conn, "aborted", False):
                why = "the agent's time was up: its request was cut"
                return 499, json.dumps({"error": {"message": why, "type": ABORTED}}).encode(), \
                    "application/json", time.time() - t0
            down = not self.healthy()
            why = (f"the model's server didn't answer the relay ({e})"
                   + ("; its health check failed too" if down else ""))
            return (503 if down else 502), json.dumps({"error": {
                "message": why, "type": DOWN if down else "relay_upstream"}}).encode(), \
                "application/json", time.time() - t0
        finally:
            with self._lock:
                self._inflight.get(trial, set()).discard(conn)
            conn.close()

    def abort(self, trial: str) -> int:
        """cut the trial's requests in flight: the server sees the
        connection close and stops its reply. How many were cut"""
        with self._lock:
            conns = list(self._inflight.get(trial, set()))
        for c in conns:
            c.aborted = True
            try:
                if c.sock is not None:
                    c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return len(conns)

    def healthy(self) -> bool:
        """llama-server's /health answers 200 when it can serve"""
        root = self.upstream[:-3] if self.upstream.endswith("/v1") else self.upstream
        req = urllib.request.Request(root + "/health",
                                     headers={"Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def record(self, trial: str, status: int, raw: bytes, secs: float) -> None:
        """each answer's tokens, seconds and the server's timings, for the
        run's numbers and the cache check"""
        try:
            usage = (json.loads(raw).get("usage") or {}) if status == 200 else {}
        except (ValueError, AttributeError):
            usage = {}
        try:
            timings = (json.loads(raw).get("timings") or {}) if status == 200 else {}
        except (ValueError, AttributeError):
            timings = {}
        line = {"at": round(time.time(), 3), "trial": trial[:200], "status": status,
                **({"down": True} if status == 503 and DOWN.encode() in raw else {}),
                "seconds": round(secs, 2), "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **{k: timings[k] for k in TIMINGS if isinstance(timings, dict) and k in timings}}
        with self._lock:
            self.records = [*self.records[-199:], line]
            if self.usage is None:
                return
            self.usage.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
