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
asked for that cache with flash attention on; FAKE_LLAMA_SWITCH=ignore a
template whose prompt ignores enable_thinking (29.2). 31.1: FAKE_LLAMA_PATHS=1
serves a file of expert paths (step2388's k3 and k6): its start-up lines
("expert path k3 (default): temp = 0.800, …"), its /props `expert_paths`, a
request's "expert_path" choosing whether it thinks (k6) or not (k3), and
/completion's first-token probabilities before and after its samplers
(n_probs, post_sampling_probs), the presence penalty on k6 left off the
prompt's tokens unless FAKE_LLAMA_PROMPT_PENALTY=1. FAKE_LLAMA_PATHS_BAD=
"6.presence_penalty=0.5" changes a value the server reports (its /props and
its line); FAKE_LLAMA_PATHS_LOG=0 prints no start-up lines. 31.2: /props
gives each path by its top_k, with the other penalties neutral in its params
("0.repeat_penalty=1.1" changes one on both), and the default path in
default_generation_settings; the start-up lines print thinking as 1 or 0.
31.3: FAKE_LLAMA_CONTINUE=1 writes each reply as a list of tokens, a word each
with the space before it — thinking "w1 w2 …", then "\n</think>", then the
reply — so many that the whole needs FAKE_LLAMA_OUT_TOKENS (default 12), and
counts the prompt and the reply as /tokenize does; its template ends in a
space, and a final assistant message is continued as the team's build
continues one (its thinking written straight after the prompt; a reply
beside it after "\n</think>\n\n"): the rest of the same reply, parsed whole
(the thinking up to </think>, the reply after its newlines), with
"logprobs" and "verbose" (the settings applied) when asked; /completion's
probabilities with penalty_prompt_tokens put the presence penalty on the
prompt's last N tokens alone (0 none, -1 all). 32: FAKE_LLAMA_MEMORY=1 keeps
each slot's memory as the team's research build does (Memory below): a
request pinned to its slot (id_slot), the chat template's words a token each
(Qwen's turns, an earlier reply's thinking dropped, the path's generation
prompt), the memory reused up to where the prompt parts from it and rolled
back to the latest checkpoint before that, checkpoints where the server
makes them, a path change cleared — or kept with keep_cache_across_paths,
said in the server's own log line — and each response's cache_n, prompt_n
and cached_tokens. It answers the switch experiment's turns (a choice, a
number, the facts it was told, their (a×b)/(a+b)), and /completion with
token ids greedily with n_probs; a slot whose memory another path computed
answers a little differently (MIXED_EVERY). 32.2: started with
--reasoning-preserve, an earlier reply is written with its thinking (each
k3 reply with k3's empty block), so a reply comes back as it was written —
but for a reply ending in TRIMMED, which the template drops (a reply whose
text doesn't come back, and rolls the memory back); --ctx-checkpoints N
keeps N checkpoints a slot; the last token a request writes is never in the
memory, as the server keeps it. 32.4: a chat reply's token ids, its end
marker too, under logprobs; each kept token's path, and a memory "mixed" only
where another path computed MIXED_MIN of them (its recall and its sum wrong);
FAKE_LLAMA_THINK_IN_CONTENT=1 a server that leaves a reply's thinking in its
content. 34: /tokenize with_pieces gives each token's piece, its space before
it; FAKE_LLAMA_THINK_WORDS=N thinks N words (t1 t2 …) when a request sends
thinking_budget_tokens, held to it unless FAKE_LLAMA_BUDGET=ignore, and cut at its
max_tokens with no reply when it runs there; FAKE_LLAMA_REPLY_WORDS=N writes N
words (r1 r2 …) before the answer line, cut where max_tokens runs out."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "version: 6543 (abc1234)\nbuilt with cc (fake) for x86_64-linux-gnu"
# 31.1: what a file of expert paths stores, and the server reports it applied
PATHS = {3: {"temp": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
         6: {"temp": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.75,
             "penalty_last_n": -1, "penalty_prompt_tokens": 0}}


def paths_reported() -> dict:
    """31.1: the paths' values as this server says them — FAKE_LLAMA_PATHS_BAD's changed"""
    got = {p: dict(v) for p, v in PATHS.items()}
    for item in (os.environ.get("FAKE_LLAMA_PATHS_BAD") or "").split(","):
        m = re.fullmatch(r"\s*(\d+)\.(\w+)=(.+?)\s*", item)
        if m and int(m[1]) in got:
            got[int(m[1])][m[2]] = float(m[3]) if "." in m[3] else int(m[3])
    return got


def NEUTRAL_REPORTED() -> dict:  # noqa: N802 — read as a table
    """31.2: the penalties beside a path's own in its /props params — neutral
    unless FAKE_LLAMA_PATHS_BAD changes one (0.repeat_penalty=1.1: on both)"""
    got = {"repeat_penalty": 1.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0}
    for item in (os.environ.get("FAKE_LLAMA_PATHS_BAD") or "").split(","):
        m = re.fullmatch(r"\s*0\.(\w+)=(.+?)\s*", item)
        if m:
            got[m[1]] = float(m[2])
    return got


def tok_id(w: str) -> int:
    return sum(w.encode()) % 50000


def counted_by(prompt: str, path: int, ppt, temperature=None) -> dict:
    """31.3: /completion's first token with penalty_prompt_tokens — two of the
    prompt's first tokens, two of its last N, and two it hasn't; the presence
    penalty on k6 put on the tokens of the last N (none for 0, all for -1)"""
    ids = [tok_id(w) for w in prompt.split()]
    n = int(ppt)
    tail = set(ids) if n < 0 else set(ids[len(ids) - n:]) if n > 0 else set()
    head = [i for i in dict.fromkeys(ids) if i not in set(ids[len(ids) - max(n, 0):])][:2]
    last = [i for i in dict.fromkeys(reversed(ids)) if i not in set(ids[:len(ids) - max(n, 0)])][:2]
    cands = head + last + [1, 2]
    raw = dict(zip(cands, [-0.4, -1.1, -0.7, -1.5, -0.9, -1.8]))
    v = paths_reported().get(path) or PATHS[3]
    temp = float(temperature) if temperature is not None else v["temp"]
    post = {i: (x - (v["presence_penalty"] if i in tail else 0)) / temp for i, x in raw.items()}
    z = math.log(sum(math.exp(x) for x in post.values()))
    return {"before": raw, "after": {i: x - z for i, x in post.items()}}


CONT_REPLY = ["\n</think>", "\n\nThe", " options", " compared.", "\nANSWER:"]


def reply_tokens(on: bool, need: int, pick: str) -> list[str]:
    """31.3: the whole reply a question gets, as tokens — the same each time it
    is asked, so a cut one continued is the rest of it"""
    tail = [*(CONT_REPLY if on else CONT_REPLY[1:]), f" {pick}"]
    if not on:
        tail[0] = tail[0].lstrip("\n")
    k = max(1, need - len(tail))
    head = [f"{'w' if on else 'x'}{i + 1}" if i == 0 else f" {'w' if on else 'x'}{i + 1}"
            for i in range(k)]
    if not on:
        tail[0] = " " + tail[0]
    return head + tail


def parsed(text: str, on: bool) -> dict:
    """31.3: a reply as the team's parser gives it back: the thinking up to
    </think>, the reply after up to two newlines"""
    if not on:
        return {"content": text}
    if "</think>" not in text:
        return {"reasoning_content": text, "content": ""}
    think, rest = text.split("</think>", 1)
    for _ in range(2):
        rest = rest[1:] if rest.startswith("\n") else rest
    return {"reasoning_content": think, "content": rest}


def first_token(prompt: str, path: int) -> dict:
    """31.1: /completion's first token, before and after the samplers — three of
    the prompt's own tokens and three it hasn't, the presence penalty on k6
    put on the prompt's only when FAKE_LLAMA_PROMPT_PENALTY=1"""
    mine = []
    for w in prompt.split():
        t = sum(w.encode()) % 50000
        if t not in mine:
            mine.append(t)
    ids = mine[:3] + [1, 2, 3]
    raw = dict(zip(ids, [-0.4, -1.1, -1.9, -0.8, -1.3, -2.2]))
    v = paths_reported().get(path) or PATHS[3]
    counted = os.environ.get("FAKE_LLAMA_PROMPT_PENALTY") == "1"
    post = {i: (x - (v["presence_penalty"] if counted and i in mine else 0)) / v["temp"]
            for i, x in raw.items()}
    z = math.log(sum(math.exp(x) for x in post.values()))
    return {"before": raw, "after": {i: x - z for i, x in post.items()}}
# 23c: the probe's timings — (prompt tok/s, output tok/s)
PROBE = {"alone": [(9000.0, 140.0), (8000.0, 150.0), (10000.0, 120.0)],
         "long": (6000.0, 110.0), "together_prompt": 5000.0, "together_out": 80.0,
         "warm": (7000.0, 100.0)}


# 32: the team's research build's memory, as its server-context.cpp keeps it
# (FAKE_LLAMA_MEMORY=1) — written from its rules, not from scripts/switch_box.py
N_BATCH, N_UBATCH, MIN_STEP, MAX_CHECKPOINTS = 2048, 512, 8192, 32
USER, ASSISTANT, END = "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"
GEN = {6: [ASSISTANT, "<think>"], 3: [ASSISTANT, "<think>", "</think>"]}
# a slot whose memory another path computed (at least MIXED_MIN of its kept
# tokens): every this-many tokens of a /completion the likeliest two change
# places, a recalled fact comes back wrong (the ship), and so does a sum
MIXED_EVERY = 5
MIXED_MIN = 8


def mem_id(w: str) -> int:
    """a generated token's id (a prompt's words are /tokenize's, tok_id)"""
    import hashlib
    return int(hashlib.sha256(w.encode()).hexdigest()[:7], 16)


# a reply's last word the template drops (as Qwen's trims a reply's end)
TRIMMED = "⏎"


def mem_words(msgs: list[dict], path: int, preserve: bool = False) -> list[str]:
    """Qwen's chat template, a word a token: each turn between its markers, an
    earlier reply's content alone (its thinking dropped) — 32.2: or, with the
    thinking kept, "<think>", its thinking, "</think>" and its content (k3's
    empty block too), the content trimmed — then the path's generation prompt"""
    out = []
    for m in msgs:
        words = str(m.get("content") or "").split()
        if m.get("role") == "assistant" and preserve:
            words = words[:-1] if words and words[-1] == TRIMMED else words
            out += [ASSISTANT, "<think>", *str(m.get("reasoning_content") or "").split(),
                    "</think>", *words, END]
            continue
        out += [f"<|im_start|>{m.get('role')}", *words, END]
    return out + GEN.get(path, GEN[3])


class Memory:
    """one slot's memory: its tokens, the path that computed each one, and its
    checkpoints (each the tokens before it)"""

    def __init__(self, keep: int = MAX_CHECKPOINTS):
        self.tokens: list[int] = []
        self.by: list[int] = []
        self.path = 0
        self.cps: list[int] = []
        self.keep = keep

    @property
    def mixed(self) -> bool:
        """32.4: another path than this request's computed MIXED_MIN or more of
        the tokens kept"""
        return sum(1 for p in self.by if p != self.path) >= MIXED_MIN

    def launch(self, sid: int, path: int, keep: bool) -> None:
        """launch_slot_with_task: a path change keeps the memory (with
        keep_cache_across_paths) or clears it, said in the server's line"""
        if self.path != path and keep:
            print(f"slot launch_slot_: id {sid:2d} | task -1 | expert path k{self.path} -> "
                  f"k{path}: keeping {len(self.tokens)} cached tokens (keep_cache_across_paths)",
                  flush=True)
        elif self.path != path:
            if self.tokens:
                print(f"slot launch_slot_: id {sid:2d} | task -1 | expert path k{self.path} -> "
                      f"k{path}: clearing {len(self.tokens)} cached tokens", flush=True)
            self.tokens, self.by = [], []
            self.cps = []
        self.path = path

    def write(self, toks: list[int]) -> None:
        """what this request's path writes into the memory"""
        self.tokens = self.tokens + toks
        self.by = self.by + [self.path] * len(toks)

    def reuse(self, prompt: list[int]) -> int:
        """the tokens reused: up to where the prompt parts from the memory,
        rolled back to the latest checkpoint before that (the recurrent state
        can't go back part-way), none when there is no such checkpoint"""
        n = 0
        while n < min(len(self.tokens), len(prompt)) and self.tokens[n] == prompt[n]:
            n += 1
        if n == 0:
            self.tokens, self.by, self.cps = [], [], []
            return 0
        thold = n if n < len(prompt) else n - 1
        if len(self.tokens) - 1 >= thold:
            got = next((c for c in reversed(self.cps) if c - 1 <= n and (c - 1 < thold or c == 1)),
                       None)
            if got is None:
                self.tokens, self.by, self.cps = [], [], []
                return 0
            n = got
        self.cps = [c for c in self.cps if c - 1 <= n]
        self.tokens, self.by = self.tokens[:n], self.by[:n]
        return n

    def read(self, prompt: list[int], user_starts: set[int]) -> None:
        """the prompt from the memory's end, in batches broken where the
        server breaks them, a checkpoint made at a batch's start where it
        makes one"""
        last_user = max(user_starts) if user_starts else -1
        pos, size = len(self.tokens), len(prompt)
        while pos < size:
            start = pos
            while pos < size and pos - start < N_BATCH:
                pos += 1
                if pos in user_starts and (pos == last_user or not self.cps
                                           or pos > self.cps[-1] + MIN_STEP):
                    break
                if size - pos in (min(N_BATCH, 4 + N_UBATCH), 4):
                    break
            near_end = size < pos + N_UBATCH
            make = start > 0 and (pos == size or start in user_starts or near_end)
            make = make and (not self.cps or start == last_user
                             or start > self.cps[-1] + MIN_STEP)
            if make:
                self.cps = (self.cps + [start])[-self.keep:]
        self.write(list(prompt[len(self.tokens):]))


def user_starts_of(prompt: list[int], delim: list[int]) -> set[int]:
    return {i for i in range(len(prompt)) if delim and prompt[i:i + len(delim)] == delim}


def mem_reply(msgs: list[dict], path: int, mixed: bool) -> tuple[str, str]:
    """(thinking, reply) to the switch experiment's last turn: a choice, a
    number, the facts it was told, or their (a×b)/(a+b)"""
    import fractions
    said = " ".join(str(m.get("content") or "") for m in msgs if m.get("role") == "user")
    q = str(msgs[-1].get("content") or "")
    think = "Thinking it through." if path == 6 else ""
    facts = {k: (re.findall(p, said) or [""])[-1] for k, p in (
        ("courier", r"courier's name is (\w+)"), ("code", r"locker code is (\d+)"),
        ("gate", r"gate number is (\d+)"), ("crates", r"has (\d+) crates"))}
    if "Write them out now" in q:
        courier = (facts["courier"] + "x") if mixed else facts["courier"]
        return think, f"courier: {courier}\ngate: {facts['gate']}"
    if "(a × b) / (a + b)" in q:
        a, b = int(facts["code"] or 0), int(facts["crates"] or 0)
        f = fractions.Fraction(a * b, a + b) if a + b else fractions.Fraction(0)
        return think, f"Worked out.\nANSWER: {f.numerator + int(mixed)}/{f.denominator}"
    opts = dict(re.findall(r"(?m)^([A-J])\) (.*)$", q))
    if opts:
        right = next((k for k, v in opts.items() if v.startswith("right")), "A")
        return think, f"The options compared.\nANSWER: {right}"
    m = re.search(r"\(answer (-?\d+)\)", q)
    # a GSM8K reply ends in a word the template drops: it doesn't come back —
    # 32.4: one whose answer is odd, so a chat's reply 2 comes back or not
    n = int(m[1]) if m else 0
    return think, f"Worked through it.\nANSWER: {n}" + (
        f" {TRIMMED}" if "Solve the following math problem" in q and n % 2 else "")


def mem_next(path: int, last: int, k: int, mixed: bool) -> list[tuple[int, float]]:
    """/completion's next token after `last`, its 20 likeliest — the path's
    own; a mixed memory swaps the first two every MIXED_EVERY tokens"""
    base = [(mem_id(f"{path}:{last}:{j}"), 0.0) for j in range(20)]
    probs = [0.5 / (j + 1) for j in range(20)]
    z = sum(probs)
    top = [(t, p / z) for (t, _), p in zip(base, probs)]
    if mixed and k % MIXED_EVERY == MIXED_EVERY - 1:
        top[0], top[1] = (top[1][0], top[0][1]), (top[0][0], top[1][1])
    return top


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
    paths_on = os.environ.get("FAKE_LLAMA_PATHS") == "1"
    cont_on = os.environ.get("FAKE_LLAMA_CONTINUE") == "1"
    mem_on = os.environ.get("FAKE_LLAMA_MEMORY") == "1"
    # FAKE_LLAMA_PRESERVE=off: a template that keeps no earlier thinking, the
    # flag notwithstanding (and the server says so at start-up)
    preserve = "--reasoning-preserve" in argv and os.environ.get("FAKE_LLAMA_PRESERVE") != "off"
    if "--reasoning-preserve" in argv and not preserve:
        print("srv          init: chat template does NOT support preserving reasoning, "
              "--reasoning-preserve has no effect", flush=True)
    memories = [Memory(int(arg(argv, "--ctx-checkpoints", "-ctxcp", default=str(MAX_CHECKPOINTS))))
                for _ in range(slots)]

    def thinks(body: dict) -> bool:
        kw = body.get("chat_template_kwargs") or {}
        if paths_on and "expert_path" in body:
            # as the team's server: the path says whether it thinks, and a
            # request's enable_thinking over it (r20: Medium, path 6 not thinking)
            # (FAKE_LLAMA_PATH_ALONE=1: a server that lets the path alone decide)
            on = kw["enable_thinking"] if "enable_thinking" in kw and not os.environ.get(
                "FAKE_LLAMA_PATH_ALONE") else body["expert_path"] == 6
            return bool(on) and not never_think
        return bool(kw.get("enable_thinking")) and not never_think

    def assistant_text(m: dict, on: bool) -> str:
        """31.3: what the team's build writes of a final assistant message"""
        think, reply = m.get("reasoning_content") or "", m.get("content") or ""
        if on and not reply:
            return think
        return (think + "\n</think>\n\n" + reply) if on else reply

    def cont_prompt(body: dict) -> str:
        msgs = body.get("messages") or []
        on = thinks(body)
        last = msgs[-1] if msgs else {}
        q = msgs[:-1] if last.get("role") == "assistant" else msgs
        words = " ".join(str(m.get("content") or "") for m in q)
        p = "<bos> " + ("<|think|> " if on else "") + words + " <turn|> "
        return p + (assistant_text(last, on) if last.get("role") == "assistant" else "")
    if paths_on and os.environ.get("FAKE_LLAMA_PATHS_LOG") != "0":
        for p, v in paths_reported().items():
            said = ", ".join(f"{k} = {x:.3f}" if isinstance(x, float) else f"{k} = {x}"
                             for k, x in v.items())
            # 31.2: thinking as the team's server prints it, 1 or 0
            print(f"srv  load_model: expert path k{p}{' (default)' if p == 3 else ''}: {said}, "
                  f"thinking = {1 if p == 6 else 0}", flush=True)

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
                                        "default_generation_settings": {
                                            "n_ctx": ctx // slots,
                                            # 31.2: the path a request naming none gets
                                            **({"params": {"expert_path": 3}}
                                               if paths_on else {})},
                                        # 31.2: each item as the team's server gives it —
                                        # the path its top_k, its params a request's
                                        # generation settings (the other penalties too)
                                        **({"expert_paths": [
                                            {"top_k": p, "default": p == 3, "thinking": p == 6,
                                             "params": {
                                                 **{k: x for k, x in NEUTRAL_REPORTED().items()},
                                                 **{("temperature" if k == "temp" else
                                                     "repeat_last_n" if k == "penalty_last_n"
                                                     else k): x for k, x in v.items()}}}
                                            for p, v in paths_reported().items()]}
                                           if paths_on else {})})
            if self.path == "/slots":
                drafts = os.environ.get("FAKE_LLAMA_DRAFTS") == "1"
                return self._send(200, [{"id": i, "speculative": drafts} for i in range(slots)])
            return self._send(404, {"error": {"message": "not found"}})

        def cont_chat(self, body: dict):
            """31.3: a reply as a list of tokens, or the rest of one continued"""
            msgs = body.get("messages") or []
            on = thinks(body)
            last = msgs[-1] if msgs else {}
            prefill = assistant_text(last, on) if last.get("role") == "assistant" else ""
            q = [m for m in msgs if m.get("role") != "assistant"]
            text = (q[-1] if q else {}).get("content") or ""
            opts = dict(re.findall(r"(?m)^([ABCD])\) (.*)$", text))
            right = next((k for k, v in opts.items() if v.startswith("right")), "A")
            pick = (next(k for k in "ABCD" if k != right) if "hard" in text else right)
            whole = reply_tokens(on, int(os.environ.get("FAKE_LLAMA_OUT_TOKENS") or 12), pick)
            done = len(prefill.split())
            if prefill and "".join(whole[:done]) != prefill:
                whole = whole[:0] + [prefill] + [f" more{i}" for i in range(len(whole))]
                done = 1
            room = int(body.get("max_tokens") or 0) or len(whole)
            new = whole[done:done + room]
            full = "".join(whole[:done]) + "".join(new)
            cut = done + len(new) < len(whole)
            prompt = cont_prompt(body)
            out = {"id": "chatcmpl-1", "object": "chat.completion",
                   "choices": [{"index": 0, "message": {"role": "assistant", **parsed(full, on)},
                                "finish_reason": "length" if cut else "stop",
                                **({"logprobs": {"content": [
                                    {"token": t, "logprob": -0.1,
                                     "top_logprobs": [{"token": t, "logprob": -0.1},
                                                      {"token": " other", "logprob": -2.5}]}
                                    for t in new]}} if body.get("logprobs") else {})}],
                   "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(new)}}
            if body.get("verbose"):
                k = int(body.get("expert_path") or 3)
                v = paths_reported().get(k) or PATHS[3]
                out["__verbose"] = {"generation_settings": {
                    "temperature": body.get("temperature", v["temp"]), "top_k": v["top_k"],
                    "top_p": v["top_p"], "min_p": v["min_p"],
                    # r20: a request's value over the path's, as the team's server
                    "presence_penalty": body.get("presence_penalty", v["presence_penalty"]),
                    "frequency_penalty": 0.0,
                    "repeat_penalty": 1.0, "repeat_last_n": 131072,
                    "penalty_prompt_tokens": body.get("penalty_prompt_tokens",
                                                      v.get("penalty_prompt_tokens", -1)),
                    "dry_multiplier": 0.0, "seed": body.get("seed"),
                    "n_predict": body.get("max_tokens"), "expert_path": k}}
            return self._send(200, out)        # logged as every chat request is

        def completion(self, body: dict):
            """23c: the speed probe's request — token ids in, n_predict out"""
            if isinstance(body.get("prompt"), str) and body.get("n_probs"):
                # 31.1: the first token's probabilities, after the samplers or before
                got = first_token(body["prompt"], int(body.get("expert_path") or 3))
                if body.get("post_sampling_probs"):
                    top = [{"id": i, "token": f"t{i}", "prob": math.exp(x)}
                           for i, x in got["after"].items()]
                    entry = {"id": top[0]["id"], "prob": top[0]["prob"], "top_probs": top}
                else:
                    top = [{"id": i, "token": f"t{i}", "logprob": x}
                           for i, x in got["before"].items()]
                    entry = {"id": top[0]["id"], "logprob": top[0]["logprob"],
                             "top_logprobs": top}
                return self._send(200, {"content": "x", "completion_probabilities": [entry]})
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

        def mem_slot(self, body: dict) -> tuple[int, Memory]:
            sid = int(body.get("id_slot", 0) if body.get("id_slot", -1) != -1 else 0)
            return sid, memories[sid]

        def mem_completion(self, body: dict):
            """32: /completion with token ids, on the slot's memory, greedy"""
            prompt = [int(x) for x in body["prompt"]]
            path = int(body.get("expert_path") or 3)
            delim = next(([tok_id(w) for w in str(d.get("delimiter") or "").split()]
                          for d in body.get("message_delimiters") or []
                          if d.get("role") == "user"), [])
            sid, m = self.mem_slot(body)
            with lock:
                m.launch(sid, path, bool(body.get("keep_cache_across_paths")))
                n_past = m.reuse(prompt)
                m.read(prompt, user_starts_of(prompt, delim))
                out, probs = [], []
                for k in range(int(body.get("n_predict") or 16)):
                    last = (prompt + out)[-1]
                    top = mem_next(path, last, len(prompt) + k, m.mixed)
                    out.append(top[0][0])
                    probs.append({"id": top[0][0], "logprob": math.log(top[0][1]),
                                  "top_logprobs": [{"id": t, "logprob": math.log(p)}
                                                   for t, p in top[:int(body.get("n_probs") or 0)]]})
                m.write(out[:-1])                       # the last one written is never read
            return self._send(200, {"content": " ".join(f"t{t}" for t in out), "tokens": out,
                                    "completion_probabilities": probs if body.get("n_probs") else [],
                                    "stop": True, "tokens_evaluated": len(prompt),
                                    "timings": {"cache_n": n_past, "prompt_n": len(prompt) - n_past,
                                                "predicted_n": len(out)}})

        def mem_chat(self, body: dict):
            """32: a chat turn on the slot's memory"""
            msgs = body.get("messages") or []
            path = int(body.get("expert_path") or 3)
            prompt = [tok_id(w) for w in mem_words(msgs, path, preserve)]
            sid, m = self.mem_slot(body)
            with lock:
                m.launch(sid, path, bool(body.get("keep_cache_across_paths")))
                n_past = m.reuse(prompt)
                m.read(prompt, {i for i, t in enumerate(prompt) if t == tok_id(USER)})
                think, reply = mem_reply(msgs, path, m.mixed)
                words = ([*think.split(), "</think>"] if think else []) + reply.split()
                limit = int(body.get("max_tokens") or 0)
                cut = bool(limit) and len(words) > limit
                if cut:
                    words = (think.split() * limit)[:limit] if think else words[:limit]
                # 32.4: its tokens as generated, the end marker too when it
                # finished; the last one written is never read
                gen = [tok_id(w) for w in words] + ([] if cut else [tok_id(END)])
                m.write(gen[:-1])
            if cut:
                msg = ({"role": "assistant", "reasoning_content": " ".join(words), "content": ""}
                       if think else {"role": "assistant", "content": " ".join(words)})
            elif think and os.environ.get("FAKE_LLAMA_THINK_IN_CONTENT") == "1":
                # 32.4: a server that doesn't parse the thinking out of the reply
                msg = {"role": "assistant", "content": f"{think} </think> {reply}"}
            else:
                msg = {"role": "assistant", "content": reply,
                       **({"reasoning_content": think} if think else {})}
            return self._send(200, {
                "id": "chatcmpl-1", "object": "chat.completion",
                "choices": [{"index": 0, "message": msg, "finish_reason": "length" if cut else "stop",
                             **({"logprobs": {"content": [{"id": t, "logprob": -0.1}
                                                          for t in gen]}}
                                if body.get("logprobs") else {})}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(words),
                          "prompt_tokens_details": {"cached_tokens": n_past}},
                "timings": {"cache_n": n_past, "prompt_n": len(prompt) - n_past,
                            "predicted_n": len(words)}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if mem_on and self.path == "/apply-template":
                return self._send(200, {"prompt": " ".join(mem_words(
                    body.get("messages") or [], int(body.get("expert_path") or 3), preserve))})
            if mem_on and self.path == "/completion" and isinstance(body.get("prompt"), list) \
                    and "id_slot" in body:
                if log:
                    with lock, open(log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({**body, "prompt": len(body["prompt"])}) + "\n")
                return self.mem_completion(body)
            if mem_on and self.path in ("/v1/chat/completions", "/chat/completions") \
                    and "id_slot" in body:
                if log:
                    with lock, open(log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(body) + "\n")
                return self.mem_chat(body)
            if self.path == "/tokenize":
                text = str(body.get("content") or "")
                if body.get("with_pieces"):
                    # 34: each token and its piece, as llama-server gives them
                    return self._send(200, {"tokens": [{"id": tok_id(w.strip()), "piece": w}
                                                       for w in re.findall(r"\s*\S+", text)]})
                toks = [sum(w.encode()) % 50000 for w in text.split()]
                with lock:
                    busy["text"] = toks                 # 23c: the probe's text, as asked
                return self._send(200, {"tokens": toks})
            if self.path == "/completion":
                if cont_on and isinstance(body.get("prompt"), str) and body.get("n_probs") \
                        and "penalty_prompt_tokens" in body:
                    got = counted_by(body["prompt"], int(body.get("expert_path") or 3),
                                     body["penalty_prompt_tokens"], body.get("temperature"))
                    key = "after" if body.get("post_sampling_probs") else "before"
                    top = [{"id": i, "token": f"t{i}",
                            **({"prob": math.exp(x)} if key == "after" else {"logprob": x})}
                           for i, x in got[key].items()]
                    entry = {"id": top[0]["id"],
                             **({"top_probs": top} if key == "after" else {"top_logprobs": top})}
                    return self._send(200, {"content": "x", "completion_probabilities": [entry]})
                return self.completion(body)
            if self.path == "/apply-template" and cont_on:
                return self._send(status if status != 200 else 200,
                                  {"prompt": cont_prompt(body)})
            if self.path == "/apply-template":
                # 29: the prompt the template makes — what /tokenize then counts
                words = " ".join(str(m.get("content") or "") for m in body.get("messages") or [])
                think = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
                # 29.2: a template that ignores the switch: always the same
                if os.environ.get("FAKE_LLAMA_SWITCH") == "ignore":
                    think = True
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
            if cont_on:
                return self.cont_chat(body)
            text = (body.get("messages") or [{}])[-1].get("content") or ""
            opts = dict(re.findall(r"(?m)^([ABCD])\) (.*)$", text))
            right = next((k for k, v in opts.items() if v.startswith("right")), "A")
            pick = (next(k for k in "ABCD" if k != right) if "hard" in text else right)
            on = thinks(body)          # 31.1: the path's; r20: a request's enable_thinking over it
            msg = {"role": "assistant", "content": f"The options compared.\nANSWER: {pick}"}
            # 29: an answer that needs more room than it was given runs out
            need = int(os.environ.get("FAKE_LLAMA_OUT_TOKENS") or 0)
            cut = bool(need and int(body.get("max_tokens") or 0) < need)
            if cut:
                msg["content"] = "The options compared, one by one, and again"
            # 34: a long reply, and long thinking held to the request's budget
            room = int(body.get("max_tokens") or 0)
            pad = int(os.environ.get("FAKE_LLAMA_REPLY_WORDS") or 0)
            if pad:
                msg["content"] = " ".join(f"r{i + 1}" for i in range(pad)) + "\n" + msg["content"]
            think_n = int(os.environ.get("FAKE_LLAMA_THINK_WORDS") or 0)
            if on and think_n and not inline and "thinking_budget_tokens" in body:
                words = [f"t{i + 1}" for i in range(think_n)]
                budget = body.get("thinking_budget_tokens")
                if budget is not None and os.environ.get("FAKE_LLAMA_BUDGET") != "ignore":
                    words = words[:int(budget)]
                msg["reasoning_content"] = " ".join(words)
                if room and len(words) >= room:
                    msg["reasoning_content"] = " ".join(words[:room])
                    msg["content"], cut = "", True
                room = max(0, room - len(words)) if room else 0
            elif on and inline:
                msg["content"] = "Weighing each option in turn.\n</think>\n\n" + msg["content"]
            elif on:
                msg["reasoning_content"] = "Weighing each option in turn."
            if pad and room and len(msg["content"].split()) > room:
                msg["content"], cut = " ".join(msg["content"].split()[:room]), True
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
