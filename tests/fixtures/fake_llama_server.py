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
/health says ok. 17b: FAKE_LLAMA_ARGV is a file its command line is written
to (what was launched, not what was recorded); FAKE_LLAMA_MODEL_PATH the file
its /props names instead of -m's; FAKE_LLAMA_STATUS a status every chat
request (and /apply-template) answers with; FAKE_LLAMA_THINK=never answers
without thinking whatever it is told. 20: --list-devices lists one CUDA card,
or none with CUDA's own error when FAKE_LLAMA_DEVICES=none;
FAKE_LLAMA_BROKEN=1 fails as a binary missing a library does, before
anything. 18e: FAKE_LLAMA_DRAFTS=1 has its /slots say they draft tokens. 22:
FAKE_LLAMA_VERSION is what --version prints (upstream's "version: 0.6.0-dev
(build 11555, commit 1cf776776)"); FAKE_LLAMA_TEMPLATE the chat template its
/props names; FAKE_LLAMA_THINK=inline answers with its thinking and
"</think>" in the reply, as a server that doesn't split them off does. 23:
FAKE_LLAMA_TIMINGS=1 gives each answer llama-server's timings. 23c: /tokenize
(a token a word) and /completion with token ids, for the card check's speed
probe, its timings fixed (PROBE below) — a prompt of 1,024 tokens or fewer
asked alone takes the next of the three ALONE rates in turn, a longer one the
LONG rates, and one of the eight asked at once 80 tok/s out plus one for each
of its batch asked before it; FAKE_LLAMA_PROBE_STATUS a status /completion
answers with (the chat requests still answer). Which of the eight a request
is, it says itself: the probe gives each its own slice of the text, never
its start (scripts/card_check.py), so a short prompt that doesn't begin
where the text does is one of them — told by what it asks, not by when it
comes (on a slow runner the eighth could arrive after the other seven were
done, and was taken for one asked alone: CI's 140 among 80…87). 29:
/apply-template (its messages' words after a "<bos>", a token a word as
/tokenize counts them), and FAKE_LLAMA_OUT_TOKENS=N — an answer needs N
tokens: one given fewer (max_tokens) runs out of room, its finish "length"
and no answer in it; FAKE_LLAMA_REFUSE_CACHE=q8_0 stops at start-up when
asked for that cache with flash attention on."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "version: 6543 (abc1234)\nbuilt with cc (fake) for x86_64-linux-gnu"
# 23c: the probe's timings — (prompt tok/s, output tok/s)
PROBE = {"alone": [(9000.0, 140.0), (8000.0, 150.0), (10000.0, 120.0)],
         "long": (6000.0, 110.0), "together_prompt": 5000.0, "together_out": 80.0,
         "warm": (7000.0, 100.0)}


def arg(argv: list[str], *names: str, default: str = "") -> str:
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> int:
    argv = sys.argv[1:]
    if os.environ.get("FAKE_LLAMA_BROKEN"):
        print("llama-server: error while loading shared libraries: libcudart.so.12: cannot "
              "open shared object file: No such file or directory", file=sys.stderr)
        return 127
    if "--version" in argv:
        print(os.environ.get("FAKE_LLAMA_VERSION") or VERSION, file=sys.stderr)
        return 0
    if "--list-devices" in argv:
        if os.environ.get("FAKE_LLAMA_DEVICES") == "none":
            print("ggml_cuda_init: failed to initialize CUDA: CUDA driver version is "
                  "insufficient for CUDA runtime version", file=sys.stderr)
            print("Available devices:")
        else:
            print("Available devices:\n  CUDA0: NVIDIA GeForce RTX 5090 (32109 MiB, 31588 MiB "
                  "free)")
        return 0
    if os.environ.get("FAKE_LLAMA_ARGV"):
        with open(os.environ["FAKE_LLAMA_ARGV"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"argv": argv, "env": {
                k: v for k, v in os.environ.items()
                if k.startswith(("LLAMA_", "GGML_", "LOOKAHEAD_"))}}) + "\n")
    # 29: a card that refuses a cache type with flash attention, as a V100 at
    # a head size its kernels lack would: the server stops as it loads
    refuse = os.environ.get("FAKE_LLAMA_REFUSE_CACHE")
    if refuse and refuse in argv and arg(argv, "--flash-attn", "-fa") == "on":
        print(f"ggml_cuda_flash_attn_ext: no kernel for a {refuse} cache at this head size",
              file=sys.stderr)
        return 1
    model = arg(argv, "-m", "--model")
    shown = os.environ.get("FAKE_LLAMA_MODEL_PATH") or model
    status = int(os.environ.get("FAKE_LLAMA_STATUS") or 200)
    never_think = os.environ.get("FAKE_LLAMA_THINK") == "never"
    inline = os.environ.get("FAKE_LLAMA_THINK") == "inline"
    template = os.environ.get("FAKE_LLAMA_TEMPLATE") or "{# a fake template #}"
    port = int(arg(argv, "--port", default="8080"))
    ctx = int(arg(argv, "-c", "--ctx-size", default="4096"))
    slots = int(arg(argv, "-np", "--parallel", default="1"))
    log = os.environ.get("FAKE_LLAMA_LOG", "")
    die_after = int(os.environ.get("FAKE_LLAMA_DIE_AFTER") or 0)
    ready_at = time.time() + float(os.environ.get("FAKE_LLAMA_LOAD_S") or 0)
    lock = threading.Lock()
    count = {"n": 0}
    sent = {"n": 0}
    probe_status = int(os.environ.get("FAKE_LLAMA_PROBE_STATUS") or 200)
    # 23c: the probe's requests — the text it tokenized, the next ALONE rate,
    # and how many of the eight have been asked since its warm-up
    busy = {"text": [], "alone": 0, "together": 0}
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
                return self._send(200, {"model_path": shown, "build_info": "b6543-abc1234",
                                        "total_slots": slots,
                                        "chat_template": template,
                                        "default_generation_settings": {"n_ctx": ctx // slots}})
            if self.path == "/slots":
                drafts = os.environ.get("FAKE_LLAMA_DRAFTS") == "1"
                return self._send(200, [{"id": i, "speculative": drafts} for i in range(slots)])
            return self._send(404, {"error": {"message": "not found"}})

        def completion(self, body: dict):
            """23c: the speed probe's request — token ids in, n_predict out"""
            if probe_status != 200:
                return self._send(probe_status, {"error": {"code": probe_status,
                                                           "message": "probe refused"}})
            ids = body.get("prompt") if isinstance(body.get("prompt"), list) else []
            n, out = len(ids), int(body.get("n_predict") or 16)
            head = busy["text"][:32]
            together = n <= 1024 and out >= 64 and bool(head) and ids[:len(head)] != head
            with lock:
                if out < 64:
                    busy["together"] = 0                # a probe's warm-up: a new batch
                    pr, tg = PROBE["warm"]
                elif together:
                    pr, tg = PROBE["together_prompt"], PROBE["together_out"] + busy["together"]
                    busy["together"] += 1
                elif n > 1024:
                    pr, tg = PROBE["long"]
                else:
                    pr, tg = PROBE["alone"][busy["alone"] % len(PROBE["alone"])]
                    busy["alone"] += 1
            time.sleep(0.15)                            # the eight still overlap
            return self._send(200, {"content": "x " * out, "tokens_predicted": out,
                                    "tokens_evaluated": n, "stop": True,
                                    "timings": {"prompt_n": n, "prompt_ms": n / pr * 1000,
                                                "prompt_per_second": pr, "predicted_n": out,
                                                "predicted_ms": out / tg * 1000,
                                                "predicted_per_second": tg}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if self.path == "/tokenize":
                text = str(body.get("content") or "")
                toks = [sum(w.encode()) % 50000 for w in text.split()]
                with lock:
                    busy["text"] = toks                 # 23c: the probe's text, as asked
                return self._send(200, {"tokens": toks})
            if self.path == "/completion":
                return self.completion(body)
            if self.path == "/apply-template":
                # 29: the prompt the template makes — what /tokenize then counts
                words = " ".join(str(m.get("content") or "") for m in body.get("messages") or [])
                think = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
                return self._send(status if status != 200 else 200, {
                    "prompt": "<bos> " + ("<|think|> " if think else "") + words + " <turn|>"})
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            if status != 200:
                if log:
                    with lock, open(log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(body) + "\n")
                return self._send(status, {"error": {"code": status, "message": "it failed"}})
            if log:
                with lock, open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(body) + "\n")
            text = (body.get("messages") or [{}])[-1].get("content") or ""
            opts = dict(re.findall(r"(?m)^([ABCD])\) (.*)$", text))
            right = next((k for k, v in opts.items() if v.startswith("right")), "A")
            pick = (next(k for k in "ABCD" if k != right) if "hard" in text else right)
            on = (body.get("chat_template_kwargs") or {}).get("enable_thinking") \
                and not never_think
            msg = {"role": "assistant", "content": f"The options compared.\nANSWER: {pick}"}
            # 29: an answer that needs more room than it was given runs out
            need = int(os.environ.get("FAKE_LLAMA_OUT_TOKENS") or 0)
            cut = bool(need and int(body.get("max_tokens") or 0) < need)
            if cut:
                msg["content"] = "The options compared, one by one, and again"
            if on and inline:
                msg["content"] = "Weighing each option in turn.\n</think>\n\n" + msg["content"]
            elif on:
                msg["reasoning_content"] = "Weighing each option in turn."
            with lock:
                count["n"] += 1
                n = count["n"]
            if die_after and n > die_after:
                # 17d: stopped as a box is, once its first K answers are sent —
                # never in the middle of one (a reply another thread was still
                # sending died with the process, and a run kept K-1)
                end = time.time() + 10
                while sent["n"] < die_after and time.time() < end:
                    time.sleep(0.01)
                print(f"fake: stopping after {die_after} answers", flush=True)
                os._exit(3)
            self._send(200, {"id": "chatcmpl-1", "object": "chat.completion",
                             "choices": [{"index": 0, "message": msg,
                                          "finish_reason": "length" if cut else "stop"}],
                             "usage": {"prompt_tokens": 50, "completion_tokens": 12},
                             **({"timings": {"cache_n": 8, "prompt_n": 42, "prompt_ms": 31.5,
                                             "prompt_per_second": 1333.3, "predicted_n": 12,
                                             "predicted_ms": 240.25}}
                                if os.environ.get("FAKE_LLAMA_TIMINGS") == "1" else {})})
            with lock:
                sent["n"] += 1

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"main: server is listening on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
