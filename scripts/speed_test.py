#!/usr/bin/env python3
"""19: the speed test — the cheapest box per answer, measured the same way on
every box (docs/REMOTE-RUNS.md § S). One line a box, pasted after `cd /app`
and the token, in the runner image on a rented GPU:

    python scripts/speed_test.py 5090 --price 0.50 --gguf hf://… --server hf://…

It checks the box is the one rented (the GPU, its count and memory, the
driver's CUDA) before anything downloads, fetches each file once into
/workspace/files, then runs the box's settings one after another. Each
setting starts its engine as the setting says — llama-server from our
tarball with the Q4 file (one server a GPU, never one model split across
GPUs), or vLLM with the official BF16 weights — and sends Frontier's real
requests (scripts/frontier.py: the same prompts, the model card's sampling,
the budgets, the thinking switch said out loud, a seed a question) through
one OpenAI-compatible streaming client, the same against both engines:

- a 3-minute warm-up of GPQA Diamond, thinking on, not counted;
- 20 minutes of GPQA Diamond, thinking on, its 81,920-token budget (long
  reasoning: the loop run's and HLE's shape);
- 5 minutes of MMLU-Pro, thinking off, 5-shot (short answers, long prompts).

Tokens are counted as they stream, so an answer still running when a window
closes counts its part. One line a setting, also kept in <out>/speed.json:
the box, the price, the engine and its version, the file and its sha256,
the setting; output tokens a second, answers an hour, MMLU-Pro's prompt
tokens a second; dollars per million output tokens; the peak GPU memory, the
requests that waited, were cut or were refused for memory; MTP's acceptance;
and a sanity score on the answers it finished — not a result: a setting
that scores far below our runs is broken.

A shared cache pool (llama-server's --kv-unified) fails every request in
flight when it fills: llama-server has no preemption. So against a shared
pool this client admits a request only while the pool has room for what is
in flight, the new prompt and 2,048 tokens more, and when the pool nears
full it cuts the request admitted last and asks it again later — vLLM's
preemption, done by the client. Both are counted and said.

It never posts to the board, never builds a bundle, and its numbers never
reach the board's scores. Nothing in it runs in CI: the tests stand in for
the engines."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier as fb  # noqa: E402

GPQA, MMLU = "gpqa_diamond_epoch", "mmlupro_tiger"
FAMILY = "qwen3.6"                      # the model card's sampling (frontier.PRESETS)
WARMUP_S, GPQA_S, MMLU_S = 180, 1200, 300
LLAMA_FLAGS = ["-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on"]   # the pilot's
SLOT_CTX = fb.slot_context([GPQA], True)                                  # 83,968
# a shared pool: room kept for a new request to grow, and for each step of
# those in flight, below this share of the pool
NEW_ROOM, STEP_ROOM, POOL_FILL = 2048, 64, 0.97
# 19: vLLM for BF16 only — a release 14 days old or more (22 Sep 2026), every
# package of it pinned with its hashes (scripts/speed_vllm_requirements.txt,
# compiled with --exclude-newer 2026-09-24), the official weights at a pinned
# revision. The Q4 GGUF is never run on vLLM: our Q4 numbers stay llama.cpp's
VLLM_VERSION = "0.30.0"
VLLM_LOCK = HERE / "speed_vllm_requirements.txt"
VLLM_CUDA = (13, 0)                     # its torch 2.13.0 is built for CUDA 13.0
BF16 = {"repo": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "size_gb": 71.9}
LLAMA_CUDA = (12, 8)                    # our tarball's CUDA (scripts/build_llama_tarball.sh)

# the boxes: what each must be before anything downloads, and its settings in
# order. A setting: engine llama (slots, shared, mtp) or vllm (seqs, tp)
PLANS: dict[str, dict] = {
    "5090": {"label": "RTX 5090", "gpu": r"5090", "count": 1, "min_mib": 31_000,
             "cuda": LLAMA_CUDA, "settings": [
                 {"name": "slots 8", "engine": "llama", "slots": 8},
                 {"name": "slots 16 shared", "engine": "llama", "slots": 16, "shared": True},
                 {"name": "slots 24 shared", "engine": "llama", "slots": 24, "shared": True},
                 {"name": "slots 8, MTP", "engine": "llama", "slots": 8, "mtp": True}]},
    "v100x4": {"label": "4× V100 32 GB", "gpu": r"V100", "count": 4, "min_mib": 31_000,
               "cuda": LLAMA_CUDA, "sm": 70, "settings": [
                   {"name": "4 servers × 4 slots", "engine": "llama", "slots": 4},
                   {"name": "4 servers × 6 slots", "engine": "llama", "slots": 6}]},
    "h100": {"label": "H100 (SXM, PCIe or NVL)", "gpu": r"H100|H200", "count": 1,
             "min_mib": 79_000, "cuda": VLLM_CUDA, "settings": [
                 {"name": "slots 32 shared", "engine": "llama", "slots": 32, "shared": True},
                 {"name": "slots 64 shared", "engine": "llama", "slots": 64, "shared": True},
                 {"name": "vLLM BF16, 32 sequences", "engine": "vllm", "seqs": 32, "tp": 1},
                 {"name": "vLLM BF16, 64 sequences", "engine": "vllm", "seqs": 64, "tp": 1}]},
    "h100x2": {"label": "2× H100 or H200", "gpu": r"H100|H200", "count": 2, "min_mib": 79_000,
               "cuda": VLLM_CUDA, "settings": [
                   {"name": "vLLM BF16, TP 2, 64 sequences", "engine": "vllm", "seqs": 64,
                    "tp": 2},
                   {"name": "vLLM BF16, TP 2, 128 sequences", "engine": "vllm", "seqs": 128,
                    "tp": 2}]},
}


def say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# the box, before anything downloads
# ---------------------------------------------------------------------------

def run(cmd: list[str], timeout: float = 60, env: dict | None = None) -> tuple[int, str]:
    """a command, its exit code and output — replaced in tests"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: no answer in {timeout} s"


def gpus() -> list[dict]:
    """each GPU: index, name, memory in MiB, compute capability"""
    code, out = run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
                     "--format=csv,noheader,nounits"])
    got = []
    for line in out.splitlines() if code == 0 else []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                got.append({"index": int(parts[0]), "name": parts[1], "mib": int(float(parts[2])),
                            "cap": parts[3]})
            except ValueError:
                continue
    return got


def driver_cuda() -> tuple[int, int] | None:
    """the newest CUDA the driver runs (nvidia-smi's header)"""
    code, out = run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out if code == 0 else "")
    return (int(m[1]), int(m[2])) if m else None


def box_line(cards: list[dict], cuda: tuple[int, int] | None) -> str:
    if not cards:
        return "no GPU answers nvidia-smi"
    names = sorted({c["name"] for c in cards})
    return (f"{len(cards)}× {', '.join(names)} ("
            + ", ".join(f"{c['mib']:,} MiB" for c in cards) + ")"
            + (f", CUDA {cuda[0]}.{cuda[1]}" if cuda else ", the driver's CUDA unknown"))


def check_box(plan: dict, cards: list[dict], cuda: tuple[int, int] | None) -> str:
    """'' when the box is what the plan rents; else why, in one line"""
    fit = [c for c in cards if re.search(plan["gpu"], c["name"], re.I)
           and c["mib"] >= plan["min_mib"]]
    if len(fit) < plan["count"]:
        return (f"this box isn't a {plan['label']}: it has {box_line(cards, cuda)}; the plan "
                f"needs {plan['count']} GPU{'s' if plan['count'] > 1 else ''} matching "
                f"{plan['gpu']!r} with {plan['min_mib']:,} MiB or more each")
    if cuda is None or cuda < plan["cuda"]:
        return (f"its driver runs CUDA {cuda[0]}.{cuda[1] if cuda else ''}" if cuda else
                "its driver's CUDA can't be read") + (
            f"; this plan needs CUDA {plan['cuda'][0]}.{plan['cuda'][1]} or newer — rent one "
            f"whose \"max CUDA\" is {plan['cuda'][0]}.{plan['cuda'][1]}+")
    return ""


def tarball_archs(root: Path) -> list[int]:
    """the CUDA architectures our llama-server tarball was built for (its
    VERSION file, scripts/build_llama_tarball.sh)"""
    for p in root.rglob("VERSION"):
        m = re.search(r"(?im)^cuda_archs\s+(.+)$", p.read_text(errors="replace"))
        if m:
            return sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
    return []


# ---------------------------------------------------------------------------
# the workload: Frontier's own requests
# ---------------------------------------------------------------------------

def questions(task: str, root: Path) -> list[dict]:
    """the benchmark's questions, fetched once onto this box, in a fixed order"""
    fb.set_root(root)
    return fb.load(task, root)


def request_of(task: str, item: dict, epoch: int, model: str) -> tuple[dict, dict]:
    """(the chat request as the board asks it, what scoring needs)"""
    on = task == GPQA
    text, need = fb.prompt(task, item)
    system = fb.system_of(task)
    body = {"model": model, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "messages": [*([{"role": "system", "content": system}] if system else []),
                         {"role": "user", "content": text}],
            "max_tokens": fb.BENCH[task]["budget"]["on" if on else "off"],
            **fb.sampling(FAMILY, on), "seed": fb.seed_of(task, str(item["id"]), epoch),
            "chat_template_kwargs": {"enable_thinking": on}}
    return body, need


def prompt_estimate(body: dict) -> int:
    """a prompt's tokens, from above: its bytes over three, and the template's"""
    return sum(len(str(m.get("content") or "").encode()) for m in body["messages"]) // 3 + 64


# ---------------------------------------------------------------------------
# one OpenAI-compatible streaming client, for llama-server and vLLM alike
# ---------------------------------------------------------------------------

class Ask(threading.Thread):
    """one request, streamed: its tokens counted as they come"""

    def __init__(self, test: "Test", server: "Endpoint", task: str, body: dict, need: dict,
                 job: tuple):
        super().__init__(daemon=True)
        self.test, self.server, self.task, self.body, self.need, self.job = (
            test, server, task, body, need, job)
        self.prompt = prompt_estimate(body)
        self.tokens = 0                 # streamed so far
        self.text, self.think = [], []
        self.finish = None
        self.usage: dict = {}
        self.timings: dict = {}
        self.error = ""
        self.cut = ""                   # why the client cut it
        self.conn: http.client.HTTPConnection | None = None

    def used(self) -> int:
        return self.prompt + self.tokens

    def stop(self, why: str) -> None:
        self.cut = self.cut or why
        try:
            if self.conn is not None and self.conn.sock is not None:
                self.conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def run(self) -> None:
        try:
            self.conn = http.client.HTTPConnection(self.server.host, self.server.port,
                                                   timeout=3600)
            self.conn.request("POST", self.server.path + "/chat/completions",
                              body=json.dumps(self.body).encode(),
                              headers={"Content-Type": "application/json"})
            r = self.conn.getresponse()
            if r.status != 200:
                self.error = f"HTTP {r.status}: {r.read()[:300].decode(errors='replace')}"
                return
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                self.chunk(json.loads(data))
        except (OSError, http.client.HTTPException, ValueError) as e:
            if not self.cut:
                self.error = f"{type(e).__name__}: {e}"
        finally:
            try:
                if self.conn is not None:
                    self.conn.close()
            except OSError:
                pass
            self.test.done(self)

    def chunk(self, c: dict) -> None:
        ch = (c.get("choices") or [{}])[0] if c.get("choices") else {}
        d = ch.get("delta") or {}
        piece = d.get("content") or ""
        thought = d.get("reasoning_content") or d.get("reasoning") or ""
        if piece:
            self.text.append(piece)
        if thought:
            self.think.append(thought)
        u = c.get("usage") if isinstance(c.get("usage"), dict) else {}
        before = self.tokens
        if isinstance(u.get("completion_tokens"), int):          # vLLM's continuous count
            self.tokens = max(self.tokens, u["completion_tokens"])
            self.usage = u
        elif piece or thought:
            self.tokens += 1                                     # llama-server: a token a chunk
        if isinstance(c.get("timings"), dict):
            self.timings = c["timings"]
        if ch.get("finish_reason"):
            self.finish = ch["finish_reason"]
        if self.tokens > before:
            self.test.streamed(self, self.tokens - before)

    def answer(self) -> str:
        """the reply as the board keeps it: its thinking in its tags"""
        text, think = "".join(self.text), "".join(self.think)
        return f"<think>\n{think}\n</think>\n\n{text}" if think else text


class Endpoint:
    """a server's address, and how many requests it takes at once"""

    def __init__(self, base: str, slots: int, pool: int | None = None, gpu: str = ""):
        m = re.match(r"https?://([^/:]+):(\d+)(/.*)?$", base)
        self.host, self.port = m[1], int(m[2])
        self.path = (m[3] or "").rstrip("/") or "/v1"
        self.slots, self.pool, self.gpu = slots, pool, gpu
        self.flight: list[Ask] = []     # in the order admitted

    def room(self, prompt: int) -> str:
        """'' when a request of `prompt` tokens may start now; 'slot' when
        every slot is busy; 'memory' when the shared pool hasn't the room"""
        if len(self.flight) >= self.slots:
            return "slot"
        if self.pool and (sum(a.used() for a in self.flight) + prompt + NEW_ROOM
                          > self.pool * POOL_FILL):
            return "memory"
        return ""

    def overflowing(self) -> bool:
        """the shared pool would fill within the next few steps"""
        return bool(self.pool) and (sum(a.used() for a in self.flight)
                                    + STEP_ROOM * len(self.flight) > self.pool * POOL_FILL)


class Test:
    """a setting's workload, through its endpoints: the windows, the counts"""

    def __init__(self, endpoints: list[Endpoint], items: dict[str, list[dict]], model: str,
                 warmup: float = WARMUP_S, gpqa: float = GPQA_S, mmlu: float = MMLU_S,
                 clock=time.monotonic):
        self.endpoints, self.items, self.model = endpoints, items, model
        self.windows = [("warm-up", GPQA, warmup), ("gpqa", GPQA, gpqa), ("mmlu", MMLU, mmlu)]
        self.clock = clock
        self.lock = threading.Lock()
        self.phase = "warm-up"
        self.tokens = {"warm-up": 0, "gpqa": 0, "mmlu": 0}
        self.answers: list[dict] = []
        self.waited = self.cut = self.refused = self.errors = 0
        self.order = {GPQA: self._order(GPQA), MMLU: self._order(MMLU)}
        self.again: dict[str, list] = {GPQA: [], MMLU: []}
        self.prompt = {"read": 0, "all": 0}
        self.drafts = {"n": 0, "accepted": 0}
        self.held: set = set()
        self.head: tuple | None = None           # (task, job): the next to ask

    def _order(self, task: str):
        epochs = fb.BENCH[task]["epochs"]
        for e in range(epochs):
            for it in self.items.get(task) or []:
                yield it, e

    def streamed(self, ask: Ask, n: int) -> None:
        with self.lock:
            if ask.task == (MMLU if self.phase == "mmlu" else GPQA):
                self.tokens[self.phase] += n
            over = ask.server.overflowing()
        if over:
            self.preempt(ask.server)

    def preempt(self, ep: Endpoint) -> None:
        """the shared pool is nearly full: the request admitted last is cut,
        and asked again from its start when there is room"""
        with self.lock:
            if not ep.flight:
                return
            victim = ep.flight[-1]
            if victim.cut:
                return
            self.cut += 1
            self.again[victim.task].insert(0, victim.job)
        victim.stop("the shared pool was nearly full")

    def done(self, ask: Ask) -> None:
        with self.lock:
            if ask in ask.server.flight:
                ask.server.flight.remove(ask)
            phase = self.phase
            if ask.cut:
                return
            if ask.error:
                if re.search(r"(?i)context|kv cache|memory|exceed", ask.error):
                    self.refused += 1
                else:
                    self.errors += 1
                return
            exact = ask.usage.get("completion_tokens") if ask.usage else None
            if isinstance(exact, int) and exact > ask.tokens and \
                    ask.task == (MMLU if phase == "mmlu" else GPQA):
                self.tokens[phase] += exact - ask.tokens
            t = ask.timings
            if isinstance(t.get("draft_n"), (int, float)):
                self.drafts["n"] += int(t["draft_n"])
                self.drafts["accepted"] += int(t.get("draft_n_accepted") or 0)
            if ask.task == MMLU and phase == "mmlu":
                whole = (ask.usage or {}).get("prompt_tokens")
                cached = ((ask.usage or {}).get("prompt_tokens_details") or {}).get(
                    "cached_tokens")
                if isinstance(t.get("prompt_n"), int):           # llama-server: what it read
                    self.prompt["read"] += t["prompt_n"]
                    self.prompt["all"] += t["prompt_n"] + int(t.get("cache_n") or 0)
                elif isinstance(whole, int):                     # vLLM: less its cache
                    self.prompt["all"] += whole
                    self.prompt["read"] += whole - int(cached or 0)
            got = fb.score(ask.task, ask.answer(), ask.finish, ask.need)
            self.answers.append({"task": ask.task, "phase": phase, "ok": bool(got.get("ok")),
                                 "finish": ask.finish})

    def fill(self, task: str) -> None:
        """every endpoint given requests while it has room; a request a shared
        pool hasn't the room for waits, counted once"""
        for ep in self.endpoints:
            while True:
                with self.lock:
                    if self.head is None or self.head[0] != task:
                        job = (self.again[task].pop(0) if self.again[task]
                               else next(self.order[task], None))
                        if job is None:
                            return
                        self.head = (task, job)
                    job = self.head[1]
                body, need = request_of(task, job[0], job[1], self.model)
                key = (task, str(job[0]["id"]), job[1])
                with self.lock:
                    why = ep.room(prompt_estimate(body))
                    if why == "memory" and key not in self.held:
                        self.waited += 1
                        self.held.add(key)
                    if why:
                        break
                    self.head = None
                    ask = Ask(self, ep, task, body, need, job)
                    ep.flight.append(ask)
                ask.start()

    def cut_all(self, why: str) -> None:
        for ep in self.endpoints:
            for ask in list(ep.flight):
                ask.stop(why)
        t0 = self.clock()
        while any(ep.flight for ep in self.endpoints) and self.clock() - t0 < 60:
            time.sleep(0.05)

    def go(self, tick: float = 0.05) -> dict:
        t0 = self.clock()
        end = t0
        for name, task, secs in self.windows:
            with self.lock:
                self.phase = name
            end += secs
            if name == "mmlu":
                self.cut_all("the GPQA window closed")
            while self.clock() < end:
                self.fill(task)
                time.sleep(tick)
        self.cut_all("the run's windows closed")
        return self.result()

    def result(self) -> dict:
        def window(name, task, secs):
            got = [a for a in self.answers if a["phase"] == name and a["task"] == task]
            return {"seconds": secs, "tokens": self.tokens[name],
                    "tokens_s": self.tokens[name] / secs if secs else 0.0,
                    "answers": len(got), "answers_h": len(got) / (secs / 3600) if secs else 0.0}
        g = window("gpqa", GPQA, self.windows[1][2])
        m = window("mmlu", MMLU, self.windows[2][2])
        secs = self.windows[2][2]
        m.update(prompt_s=self.prompt["all"] / secs if secs else 0.0,
                 prompt_read_s=self.prompt["read"] / secs if secs else 0.0)
        sanity = {}
        for task in (GPQA, MMLU):
            got = [a for a in self.answers if a["task"] == task]
            sanity[task] = {"right": sum(a["ok"] for a in got), "of": len(got)}
        return {"gpqa": g, "mmlu": m, "waited": self.waited, "cut": self.cut,
                "refused": self.refused, "errors": self.errors, "sanity": sanity,
                "mtp_acceptance": (self.drafts["accepted"] / self.drafts["n"]
                                   if self.drafts["n"] else None)}


# ---------------------------------------------------------------------------
# what a setting's line says
# ---------------------------------------------------------------------------

def dollars_per_million(price: float, tokens_s: float) -> float | None:
    return price / (tokens_s * 3600 / 1e6) if tokens_s > 0 else None


SANITY_FLOOR = 0.5                      # far below our runs: GPQA ~0.8, MMLU-Pro ~0.7


def line_of(rec: dict) -> str:
    r = rec["result"]
    g, m = r["gpqa"], r["mmlu"]
    usd = dollars_per_million(rec["price"], g["tokens_s"])
    words = [f"{rec['box']} · ${rec['price']:.2f}/h · {rec['engine']}", rec["file"],
             rec["setting"],
             f"GPQA {g['tokens_s']:,.0f} output tokens/s, {g['answers_h']:,.1f} answers/h",
             f"MMLU-Pro {m['tokens_s']:,.0f} output tokens/s, {m['answers_h']:,.0f} answers/h, "
             f"{m['prompt_s']:,.0f} prompt tokens/s ({m['prompt_read_s']:,.0f} read, the rest "
             "from the cache)",
             (f"${usd:.2f} per million output tokens" if usd else "no output: no price"),
             "peak " + ", ".join(f"{x:,} MiB" for x in rec.get("peak_mib") or []) or "peak ?",
             f"{r['waited']} waited for memory, {r['cut']} cut and asked again, "
             f"{r['refused']} refused"]
    if r.get("mtp_acceptance") is not None:
        words.append(f"MTP accepted {r['mtp_acceptance']:.0%} of its drafts")
    elif rec.get("mtp"):
        words.append("MTP: its slots drafted nothing (the build doesn't run the file's MTP heads)")
    san = []
    broken = False
    for task, label in ((GPQA, "GPQA"), (MMLU, "MMLU-Pro")):
        s = r["sanity"][task]
        san.append(f"{label} {s['right']}/{s['of']}")
        broken |= s["of"] >= 20 and s["right"] / s["of"] < SANITY_FLOOR
    words.append("sanity " + ", ".join(san) + (" — FAR BELOW OUR RUNS: this setting is broken"
                                                if broken else ""))
    return " · ".join(words)


def keep(out: Path, rec: dict) -> None:
    p = out / "speed.json"
    try:
        had = json.loads(p.read_text())
    except (OSError, ValueError):
        had = []
    had = [x for x in had if not (x.get("box") == rec["box"] and x.get("setting")
                                  == rec["setting"])] + [rec]
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(had, indent=1))


class Peak(threading.Thread):
    """each GPU's most memory used, read every few seconds"""

    def __init__(self, every: float = 5.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.halt = every, {}, threading.Event()

    def run(self) -> None:
        while not self.halt.is_set():
            code, out = run(["nvidia-smi", "--query-gpu=index,memory.used",
                             "--format=csv,noheader,nounits"], timeout=30)
            for ln in out.splitlines() if code == 0 else []:
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    try:
                        i, used = int(parts[0]), int(float(parts[1]))
                    except ValueError:
                        continue
                    self.peak[i] = max(self.peak.get(i, 0), used)
            self.halt.wait(self.every)

    def values(self) -> list[int]:
        self.halt.set()
        return [self.peak[i] for i in sorted(self.peak)]


# ---------------------------------------------------------------------------
# the engines
# ---------------------------------------------------------------------------

def llama_pool(src: str, flags: list[str], slots: int, mib: int) -> tuple[int, str]:
    """a shared pool as big as fits on a card of `mib`: (tokens, the basis)
    — the pilot's measured slope, every slot's recurrent state, 1 GiB spare"""
    import remote_gguf as rg
    shape, size = rg.header_of(src)
    if not shape or not size:
        return 0, "the GGUF's header couldn't be read"
    ctk, ctv = rg.cache_types(flags)
    per_token = rg.kv_per_token(shape, ctk, ctv)
    if per_token is None:
        return 0, f"the cache type {ctk}/{ctv} isn't one this check knows"
    room = (mib * 1024 ** 2 - size - rg.fixed_bytes(shape) - rg.ROOM
            - slots * rg.recurrent_per_slot(shape))
    return max(0, int(room // (per_token + rg.CTX_BUFFERS))), rg.BASIS


def llama_argv(exe: str, gguf: str, port: int, setting: dict, pool: int) -> list[str]:
    """llama-server as the setting says: its slots, each its own context or
    one shared pool (--kv-unified), MTP when asked; always the whole model on
    its one GPU"""
    slots = setting["slots"]
    ctx = pool if setting.get("shared") else SLOT_CTX * slots
    return [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx),
            "-np", str(slots), "-ngl", "99", "--jinja", "--metrics", *LLAMA_FLAGS,
            *(["--kv-unified"] if setting.get("shared") else []),
            *(["--spec-type", "draft-mtp"] if setting.get("mtp") else [])]


def vllm_argv(venv: Path, weights: Path, port: int, setting: dict, kv: str) -> list[str]:
    """vLLM as the setting says: the BF16 weights, our budget's context, its
    sequences, the cache in FP8 (or BF16, `kv`), tensor parallel"""
    return [str(venv / "bin" / "vllm"), "serve", str(weights), "--served-model-name", "qwen",
            "--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(SLOT_CTX),
            "--max-num-seqs", str(setting["seqs"]), "--tensor-parallel-size", str(setting["tp"]),
            "--gpu-memory-utilization", "0.92", "--kv-cache-dtype", kv,
            "--reasoning-parser", "qwen3", "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
            "--seed", "0"]


def vllm_fits(log: str) -> dict:
    """what vLLM says fits: its KV cache in tokens, and how many requests of
    the full context at once"""
    tok = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
    conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", log)
    return {"kv_tokens": int(tok[-1].replace(",", "")) if tok else None,
            "full_length_at_once": float(conc[-1][1]) if conc else None}


class Proc:
    """an engine's process on 127.0.0.1, its log, healthy or stopped"""

    def __init__(self, argv: list[str], env: dict, log: Path, base: str):
        self.argv, self.env, self.log, self.base = argv, env, log, base
        self.p: subprocess.Popen | None = None

    def start(self, timeout: float) -> str:
        """'' once /health answers; else the engine's last words"""
        import remote_gguf as rg
        self.log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log, "ab")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.argv[:2])} "
                 "=====\n".encode())
        fh.flush()
        self.p = subprocess.Popen(self.argv, stdout=fh, stderr=subprocess.STDOUT, env=self.env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return f"it stopped as it started (exit {self.p.returncode}): {self.tail()}"
            st, _ = rg._get(self.base.removesuffix("/v1") + "/health")
            if st == 200:
                return ""
            time.sleep(2)
        self.stop()
        return f"it wasn't healthy after {timeout:.0f} s: {self.tail()}"

    def tail(self, n: int = 12) -> str:
        try:
            lines = self.log.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return " / ".join(x.strip() for x in lines[-n:] if x.strip())[-900:]

    def stop(self) -> None:
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(10)


def install_vllm(files: Path) -> tuple[Path | None, str]:
    """vLLM's venv, made once on this box from the hashed lock: (its folder,
    why not)"""
    venv = files / f"vllm-{VLLM_VERSION}"
    if (venv / ".installed").exists():
        return venv, ""
    say(f"installing vLLM {VLLM_VERSION} into {venv} from {VLLM_LOCK.name} (about 5 GB, each "
        "package checked against its hash)")
    code, out = run([sys.executable, "-m", "venv", str(venv)], timeout=600)
    if code != 0:
        return None, f"its venv couldn't be made — {out.strip()[-300:]}"
    code, out = run([str(venv / "bin" / "pip"), "install", "--no-input", "--require-hashes",
                     "--no-deps", "--only-binary", ":all:", "-r", str(VLLM_LOCK)], timeout=3600)
    if code != 0:
        return None, f"vLLM couldn't be installed — {out.strip()[-400:]}"
    (venv / ".installed").write_text(VLLM_VERSION)
    return venv, ""


def fetch_bf16(files: Path) -> tuple[Path | None, str]:
    """the official BF16 weights at their pinned revision, once"""
    dest = files / f"{BF16['repo'].split('/')[1]}@{BF16['revision'][:12]}"
    if (dest / ".fetched").exists():
        return dest, ""
    say(f"fetching {BF16['repo']} at {BF16['revision'][:12]} (about {BF16['size_gb']:.0f} GB)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(BF16["repo"], revision=BF16["revision"], local_dir=str(dest),
                          token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:                              # noqa: BLE001 — said in one line
        return None, f"the weights couldn't be fetched: {str(e)[:300]}"
    (dest / ".fetched").write_text(BF16["revision"])
    return dest, ""


# ---------------------------------------------------------------------------
# the box's line
# ---------------------------------------------------------------------------

def plan_words(name: str, plan: dict, a: argparse.Namespace) -> list[str]:
    """what the box will fetch and run, said before anything starts"""
    out = [f"{name}: {plan['label']} at ${a.price:.2f}/h — {len(plan['settings'])} settings, "
           f"each {a.warmup / 60:.0f} + {a.gpqa / 60:.0f} + {a.mmlu / 60:.0f} minutes and its "
           "engine's start"]
    for s in plan["settings"]:
        out.append(f"  · {s['name']}")
    if any(s["engine"] == "llama" for s in plan["settings"]):
        out.append(f"  fetches once: the GGUF ({a.gguf}), our llama-server ({a.server})")
    if any(s["engine"] == "vllm" for s in plan["settings"]):
        out.append(f"  fetches once: vLLM {VLLM_VERSION} (about 5 GB), {BF16['repo']} at "
                   f"{BF16['revision'][:12]} (about {BF16['size_gb']:.0f} GB)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("box", choices=sorted(PLANS))
    ap.add_argument("--price", type=float, required=True, help="the box's price an hour, in $")
    ap.add_argument("--gguf", default="", help="hf://<you>/<repo>/<file>.gguf, or a path")
    ap.add_argument("--server", default="", help="our llama-server tarball: hf://… or a path")
    ap.add_argument("--only", default="", help="these settings, by number (1,3)")
    ap.add_argument("--files", type=Path, default=Path("/workspace/files"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/speed"))
    ap.add_argument("--warmup", type=float, default=WARMUP_S)
    ap.add_argument("--gpqa", type=float, default=GPQA_S)
    ap.add_argument("--mmlu", type=float, default=MMLU_S)
    ap.add_argument("--dry-run", action="store_true", help="the box's check and its plan only")
    a = ap.parse_args(argv)
    plan = PLANS[a.box]
    settings = plan["settings"]
    if a.only:
        settings = [s for i, s in enumerate(settings, 1) if str(i) in a.only.split(",")]
    llama = any(s["engine"] == "llama" for s in settings)
    if llama and not (a.gguf and a.server):
        ap.error("this box runs llama-server: give --gguf and --server")
    # the box, before anything downloads
    cards, cuda = gpus(), driver_cuda()
    say(f"this box: {box_line(cards, cuda)}")
    why = check_box(plan, cards, cuda)
    if why:
        say(f"stopping — {why}. Nothing was fetched")
        return 2
    for w in plan_words(a.box, {**plan, "settings": settings}, a):
        say(w)
    if a.dry_run:
        return 0
    use = [c for c in cards if re.search(plan["gpu"], c["name"], re.I)][:plan["count"]]
    a.files.mkdir(parents=True, exist_ok=True)
    root = a.files / "frontier-data"
    items = {GPQA: questions(GPQA, root), MMLU: questions(MMLU, root)}
    srv = gguf = None
    sha = ""
    if llama:
        import remote_gguf as rg
        tarball = rg.fetch(a.server, a.files, say)
        srv = rg.unpack_server(tarball, a.files / "llama")
        if plan.get("sm"):
            archs = tarball_archs(srv["root"])
            if plan["sm"] not in archs:
                say(f"stopping — our llama-server tarball is built for sm {archs or 'unknown'}, "
                    f"not this GPU's sm {plan['sm']}: build one with "
                    f"LLAMA_CUDA_ARCHS='70;80;86;89;90;120' (docs/REMOTE-RUNS.md § S). The GGUF "
                    "wasn't fetched")
                return 2
        parts = [rg.fetch(x, a.files, say) for x in rg.split_parts(a.gguf)]
        gguf = parts[0]
        sha = rg.sha256_cached(gguf, {}, say, a.files / "sha256.json")
        ver = rg.version_of(srv["bin"], rg.server_env(srv, {}))
    vllm_venv = weights = None
    worst = 0
    for n, s in enumerate(settings, 1):
        say(f"--- {n}/{len(settings)}: {s['name']}")
        rec = {"box": f"{plan['label']} ({len(use)}× {use[0]['name']})", "price": a.price,
               "setting": s["name"], "at": time.time(), "mtp": bool(s.get("mtp"))}
        procs: list[Proc] = []
        eps: list[Endpoint] = []
        try:
            if s["engine"] == "llama":
                import remote_gguf as rg
                rec.update(engine=f"llama.cpp {ver.get('build')} ({ver.get('commit')})",
                           file=f"{Path(a.gguf).name} sha256 {sha[:16]}…")
                for i, card in enumerate(use):
                    pool = 0
                    if s.get("shared"):
                        pool, basis = llama_pool(str(gguf), LLAMA_FLAGS, s["slots"], card["mib"])
                        if pool < SLOT_CTX:
                            raise RuntimeError(f"a shared pool that fits holds {pool:,} tokens, "
                                               f"under one answer's {SLOT_CTX:,} ({basis})")
                    port = 8090 + i
                    for attempt in range(3):
                        argv_ = llama_argv(str(srv["bin"]), str(gguf), port, s, pool)
                        p = Proc(argv_, rg.server_env(srv, {"CUDA_VISIBLE_DEVICES":
                                                            str(card["index"])}),
                                 a.out / f"{a.box}-{n}-gpu{card['index']}.log",
                                 f"http://127.0.0.1:{port}/v1")
                        err = p.start(1800)
                        if not err or not s.get("shared"):
                            break
                        pool = int(pool * 0.9)          # a pool that didn't fit: 10% smaller
                        say(f"the pool didn't fit ({err[:160]}): {pool:,} tokens next")
                    if err:
                        raise RuntimeError(f"llama-server on GPU {card['index']}: {err}")
                    procs.append(p)
                    eps.append(Endpoint(p.base, s["slots"], pool or None, str(card["index"])))
                rec["setting"] += (f" (a pool of {eps[0].pool:,} tokens a GPU)" if s.get("shared")
                                   else f" ({SLOT_CTX:,} tokens a slot)")
                rec["servers"] = len(eps)
            else:
                if vllm_venv is None:
                    vllm_venv, why = install_vllm(a.files)
                    if why:
                        raise RuntimeError(why)
                    weights, why = fetch_bf16(a.files)
                    if why:
                        raise RuntimeError(why)
                rec.update(engine=f"vLLM {VLLM_VERSION}",
                           file=f"{BF16['repo']} BF16 at {BF16['revision'][:12]}")
                devices = ",".join(str(c["index"]) for c in use[:s["tp"]])
                kv = "fp8"
                for kv in ("fp8", "auto"):
                    import remote_bundle as rb
                    env = {k: v for k, v in os.environ.items() if k not in rb.SECRETS}
                    p = Proc(vllm_argv(vllm_venv, weights, 8100, s, kv),
                             {**env, "CUDA_VISIBLE_DEVICES": devices},
                             a.out / f"{a.box}-{n}-vllm.log", "http://127.0.0.1:8100/v1")
                    err = p.start(3600)
                    if not err:
                        break
                    say(f"vLLM didn't start with the cache in {kv} ({err[:200]})")
                if err:
                    raise RuntimeError(f"vLLM: {err}")
                procs.append(p)
                fits = vllm_fits(p.log.read_text(errors="replace"))
                rec["setting"] += f" (cache {'FP8' if kv == 'fp8' else 'BF16'}" + (
                    f", {fits['kv_tokens']:,} tokens: {fits['full_length_at_once']:g} full-length "
                    "answers at once)" if fits["kv_tokens"] and fits["full_length_at_once"]
                    else ")")
                rec["vllm"] = fits
                eps.append(Endpoint(p.base, s["seqs"], None, devices))
            peak = Peak()
            peak.start()
            test = Test(eps, items, "qwen", a.warmup, a.gpqa, a.mmlu)
            rec["result"] = test.go()
            rec["peak_mib"] = peak.values()
            keep(a.out, rec)
            say(line_of(rec))
        except RuntimeError as e:
            worst = 1
            say(f"{s['name']}: not run — {e}")
            keep(a.out, {**rec, "not_run": str(e)})
        finally:
            for p in procs:
                p.stop()
    say(f"each line is in {a.out / 'speed.json'}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
