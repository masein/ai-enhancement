"""A stub of vLLM's OpenAI-compatible server, for tests and for the demo.

Never the real vLLM: CI has no GPU. It serves `chat` (weights
google/gemma-4-E4B-it), answers chat completions the way the local model
would — the judge grades like the stub grader, the proposer writes a spec,
the generator and the exam writer reply under JSON mode — and can be told to
fail, stall or throttle a request so the client's retries can be tested.
"""

from __future__ import annotations

import http.server
import json
import re
import socketserver
import threading
import time
from collections import Counter

import exam_build as eb
import judge as jd
from service import llm, proposals

WEIGHTS = "google/gemma-4-E4B-it"


def answer(body: dict) -> str:
    """What the local model says. The judge grades like the stub grader; the
    proposer writes a spec; the generator and the exam writer answer the way
    a model under JSON mode does — an object wrapping the array they were
    asked for. Anything else is echoed."""
    msgs = body["messages"]
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    user = msgs[-1]["content"]
    if "CANDIDATE ANSWER" in user:
        s, j = jd.StubGrader.grade(user)
        return json.dumps({"score": s, "justification": j})
    if system == proposals.PROPOSAL_SYSTEM:
        return json.dumps({"spec": "The model cannot trace a change to the first quantity that "
                                   "responds, and reverses the direction of an effect.",
                           "share_explained": 0.5, "patterns": ["reverses direction", "skips steps"]})
    if system == proposals.GEN_SYSTEM:
        n = int(re.search(r"Write (\d+) documents?", user).group(1))
        k = int(re.search(r"Style seed \d+-(\d+)", user).group(1))
        docs = [llm._fake_document(i) for i in range(k * n, k * n + n)]
        # JSON mode may only return an object: one document comes back bare,
        # several come back wrapped — both are what a model actually sends
        return json.dumps(docs[0] if n == 1 else {"documents": docs})
    if system == eb.DRAFT_SYSTEM:
        topic = re.search(r"Topic: (.+)", user).group(1)
        n, k = map(int, re.search(r"Write (\d+) new questions\. Set (\d+)", user).groups())
        return json.dumps({"questions": [
            {"prompt": f"In {topic}, explain mechanism {k}-{i} and the one condition under which "
                       f"it fails.", "reference": f"Mechanism {k}-{i} runs through the binding "
                                                  f"constraint; it fails when the constraint is slack.",
             "notes": "local draft"} for i in range(n)]})
    return "echo: " + user


class Stub:
    def __init__(self):
        self.lock = threading.Lock()
        self.served = [{"id": "chat", "object": "model", "owned_by": "vllm", "root": WEIGHTS}]
        self.bodies: list[dict] = []
        self.hits: Counter = Counter()          # user text -> attempts seen
        self.inflight = self.max_inflight = 0
        self.delay = 0.0
        self.gate: threading.Event | None = None
        self.always: dict[str, tuple[int, str]] = {}    # marker in the user text -> every attempt fails
        self.first: dict[str, list[tuple[int, str]]] = {}   # marker -> these attempts fail, then it answers
        self.url = ""

    def handler(self):
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj: dict) -> None:
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": stub.served})
                self._send(404, {"error": {"message": "no route"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                user = body["messages"][-1]["content"]
                with stub.lock:
                    stub.bodies.append(body)
                    stub.hits[user] += 1
                    attempt = stub.hits[user]
                    stub.inflight += 1
                    stub.max_inflight = max(stub.max_inflight, stub.inflight)
                if stub.gate is not None:
                    stub.gate.wait(10)
                if stub.delay:
                    time.sleep(stub.delay)
                fail = next((v for m, v in stub.always.items() if m in user), None)
                planned = next((v for m, v in stub.first.items() if m in user), None)
                if fail is None and planned and attempt <= len(planned):
                    fail = planned[attempt - 1]
                with stub.lock:
                    stub.inflight -= 1
                if fail:
                    return self._send(fail[0], {"error": {"message": fail[1]}})
                self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant",
                                                          "content": answer(body)}}]})
        return H


class _Server(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer's own bind does a reverse lookup (socket.getfqdn) that can
        # hang for half a minute on a laptop's resolver; loopback needs none
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def serve(stub: "Stub | None" = None) -> tuple["Stub", _Server]:
    """Start one on a loopback port; the caller shuts it down."""
    stub = stub or Stub()
    srv = _Server(("127.0.0.1", 0), stub.handler())
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    stub.url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    return stub, srv
