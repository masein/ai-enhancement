#!/usr/bin/env python3
"""32: switching mode mid-chat with the memory kept — the team's
keep_cache_across_paths measured on step2388 (brief 32, plan S). An internal
experiment for the apps: never a row of the board, never in the whitepaper.

The build is the team's research branch (runtime k3k6r-0411, frontier.py):
both expert paths on one server, each conversation pinned to its own slot,
no prompt cache in host memory. A request's `keep_cache_across_paths`
decides what a path change does to the slot's memory: off (the team's
default, exact) the server clears it and the new path reads the whole
prompt again; on, the new path continues from the memory the other path
computed — the full-attention layers' KV cache and the linear-attention
layers' running state.

What the source says, and what each check here holds the server to:
- The chat template drops every earlier turn's thinking, so a new prompt
  parts from the memory where the previous reply begins, and the server
  rolls back to its latest checkpoint before that: the start of the previous
  question (made on every request), or on turn 2 a checkpoint near the end
  of the first prompt. The previous reply is never in the memory kept: the
  new path reads it itself. What it keeps is the conversation up to the
  previous question, computed by the paths that read it.
- The server's line "expert path kA -> kB: keeping N cached tokens" counts
  the slot's whole memory, not what survives the rollback; cache_n does.

Part 2, the chats (S1–S10): Kevin's five turns, graded by script.
1. k3: two facts to remember (a name and a number), and an MMLU question.
2. k3: two more facts, and a GSM8K question.
3. k6: a GPQA Diamond or MATH Level 5 question — the k3-to-k6 switch.
4. k3: the four facts recalled, scored exactly — the k6-to-k3 switch. In arm
   B the second two were read by k6.
5. k6: (a × b) / (a + b) of a number from turn 1 and one from turn 2, a
   fraction in lowest terms — the k3-to-k6 switch again.
Arm A (the switch off) is the writing pass: its replies are the history of
both arms. Arm B (the switch on) runs beside it, in a slot of its own, each
turn with the same messages (A's earlier replies) and the same seed. B's own
reply never stays in its memory (the rollback), so its memory is what a real
switching chat would hold for that text, and only the memory differs. Each
turn records cache_n, prompt_n, cached_tokens and the server's line; a
switch where A didn't re-read, or B didn't keep, is reported and not
counted. Turns 1–2 hold the same memory in both arms: a placebo.

Part 1, drift (S11): for each conversation and direction X→Y, five runs of
its last turn on Y, each forced along A's own greedy reply (the reference):
- A: Y reads everything fresh — the reference, its first 512 tokens;
- A′: the same again in another slot (the GPU's own run-to-run noise);
- N: the previous turn on Y, then the last on Y (B's rollback, no switch);
- B: the previous turn on X, then the last on Y with the switch on;
- F: X reads everything fresh (the full gap between the paths).
Part 1's protocol, as DeviceMark's: greedy, no penalties — temperature 0,
presence_penalty 0, seed 0 — the 20 likeliest tokens' probabilities before
sampling (n_probs 20), on /completion with the prompt as token ids and the
chat's message delimiters, so checkpoints fall as in a chat. Where a run's
pick parts from the reference, its next request goes on from the
reference's own text: the rollback to the last question keeps the run's
memory. Measured against A: the first token's KL, the mean KL per token,
flips per 1,000 tokens and tokens to the first flip, by memory size and
direction.

On the server: `switch_box.py inputs` writes the inputs file (question ids,
the invented facts, step2388's stored replies for Part 1's histories — no
question text), pinned by its sha256 on every S line. The bundle
(switch-served-…) holds each chat and run; import_remote.py marks every
answer again by the board's scorers, and the report "Switching mode
mid-chat" reads the summary the import keeps.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import fractions
import hashlib
import json
import math
import random
import re
import statistics
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
import remote_bundle as rb  # noqa: E402

SUITE = "switch-served"
REFUSED = 2
RUNTIME = "k3k6r-0411"
SERVED = "served/step2388-k3k6r-0411"
STEP2388 = "served/step2388-k3k6-2611"          # whose stored replies Part 1's histories are
TASK_KEY = "switch"                              # the seeds' name
# the model team's limits, as the paper's runs (31d)
LIMITS = {3: 32_768, 6: 98_304}
# Part 2: each turn's path, and its slot — a k6 turn's limit and 16,384 for
# the chat before it (a chat that outgrows it stops in both arms, counted)
TURNS = (3, 3, 6, 3, 6)
CHATS_CONTEXT = 114_688
# Part 1: memory up to about 48k, the last question and 512 tokens
DRIFT_CONTEXT = 65_536
N_CHATS = 2_000
BINS = (1_024, 4_096, 16_384, 49_152)
PER_BIN = 50
REF_TOKENS = 512
N_PROBS = 20
CHUNK = 32                                      # a forced run's tokens a request
RUNS = ("A2", "N", "B", "F")
RUN_WORDS = {"A2": "A′ (A again)", "N": "N (rollback, no switch)", "B": "B (switch, kept)",
             "F": "F (the other path, fresh)"}
# Part 1's protocol: DeviceMark's greedy, no penalties
GREEDY = {"temperature": 0.0, "presence_penalty": 0.0, "seed": 0}
# Qwen's message delimiters, as the server's own chat way gives /completion
# (common/chat.cpp): the user's start is where a checkpoint goes
DELIMITERS = [{"role": "assistant", "delimiter": "<|im_start|>assistant"},
              {"role": "user", "delimiter": "<|im_start|>user"}]
# the check before plan S is rented: Part 2's first chats at this limit a
# turn (the memory, not the scores), Part 1's first chat of each memory size
CHECK_LIMIT = 4_096
CHECK_REF = 16
# the experiment's three answers-to-file: the rows and the check's lines
INPUT = "switch-inputs.jsonl"
INPUTS_META = "inputs.json"
CHATS = "chats.jsonl"
DRIFT = "drift.jsonl"
CHECK = "switch-check.json"
SUMMARY = "summary.json"
# what a request may carry, and nothing else: a chat turn (Part 2) and a
# forced run's request (Part 1)
CHAT_KEYS = {"max_tokens", "seed", "expert_path", "id_slot"}
DRIFT_KEYS = {"n_predict", "expert_path", "id_slot", "cache_prompt", "n_probs",
              "message_delimiters", "return_tokens", *GREEDY}
TRIES = 3
TIMEOUT_S = 7200

COURIERS = ("Odile", "Bram", "Tamsin", "Ilya", "Marisol", "Kofi", "Henrike", "Anselm", "Priya",
            "Teodor", "Saoirse", "Lucan", "Wren", "Jovan", "Esme", "Rafferty", "Yusra", "Cormac",
            "Linnea", "Dashiell", "Ines", "Gideon", "Paloma", "Soren", "Amara", "Florian",
            "Noor", "Casimir", "Thea", "Benedikt")
SHIPS = ("Marrow", "Kestrel", "Halcyon", "Lodestar", "Bramble", "Cormorant", "Tidewell",
         "Saltire", "Morrow", "Fenwick", "Larkspur", "Gannet", "Quillon", "Starling", "Ember",
         "Thistle", "Petrel", "Orrery", "Wayfarer", "Cinder", "Haddock", "Juniper", "Meridian",
         "Osprey", "Rowan", "Skerry", "Tamarisk", "Vesper", "Wrenna", "Zephyr")
FACTS_1 = ("Please remember two things for later in our conversation: the courier's name is "
           "{courier}, and the locker code is {code}. You don't need to repeat them now.\n\n")
FACTS_2 = ("Two more things to remember for later: the ship is called the {ship}, and it carries "
           "{crates} crates. You don't need to repeat them now.\n\n")
RECALL = ("Earlier in our conversation I asked you to remember four things. Write them out now, "
          "one a line, in exactly this form:\ncourier: <name>\nlocker code: <number>\n"
          "ship: <name>\ncrates: <number>")
COMPUTE = ("Take the locker code (call it a) and the number of crates (call it b) from earlier in "
           "our conversation. Work out (a × b) / (a + b), and give it as a fraction in lowest "
           "terms. The last line of your response should be of the form 'ANSWER: p/q' (or "
           "'ANSWER: n' if it is a whole number).")
TURN_WORDS = {1: "facts and an MMLU question", 2: "facts and a GSM8K question",
              3: "GPQA Diamond or MATH Level 5", 4: "the facts recalled",
              5: "(a × b) / (a + b) of the facts"}


# ---------------------------------------------------------------------------
# what is asked
# ---------------------------------------------------------------------------

def facts_of(chat: int) -> dict:
    """the four facts a chat is told, the same on every machine"""
    rng = random.Random(f"switch-32:facts:{chat}")
    return {"courier": rng.choice(COURIERS), "code": rng.randint(101, 999),
            "ship": rng.choice(SHIPS), "crates": rng.randint(11, 99)}


def compute_answer(f: dict) -> fractions.Fraction:
    return fractions.Fraction(f["code"] * f["crates"], f["code"] + f["crates"])


def seed_of(chat: int, turn: int) -> int:
    return fb.seed_of(TASK_KEY, f"chat-{chat}", turn)


def index(tasks: list[str]) -> dict[str, dict[str, dict]]:
    """each task's questions by id, from this machine's copy"""
    from service import config
    fb.set_root(config.BENCH_ROOT)
    return {t: {str(it["id"]): it for it in fb.load(t, config.BENCH_ROOT)} for t in tasks}


def turn_text(t: int, line: dict, items: dict) -> str:
    """turn t's message of a chat (Part 2)"""
    f = line["facts"]
    if t == 1:
        return FACTS_1.format(**f) + fb.prompt("mmlu_chat", items["mmlu_chat"][line["mmlu"]])[0]
    if t == 2:
        return FACTS_2.format(**f) + fb.prompt("gsm8k_chat",
                                               items["gsm8k_chat"][line["gsm8k"]])[0]
    if t == 3:
        h = line["hard"]
        return fb.prompt(h["task"], items[h["task"]][h["id"]])[0]
    return RECALL if t == 4 else COMPUTE


def as_answer(think: str, reply: str) -> str:
    """an answer as the board keeps one (service/served.ask)"""
    return f"<think>\n{think}\n</think>\n\n{reply}" if think else reply


_FACT = {k: re.compile(rf"(?im)^[\s*_`-]*{w}[\s*_`]*:[\s*_`]*([^\n*_`]+?)[\s*_`.]*$")
         for k, w in (("courier", "courier"), ("code", "locker code"), ("ship", "ship"),
                      ("crates", "crates"))}
_FRACTION = re.compile(r"(?i)ANSWER\s*:\s*\**\s*(-?\d+)\s*(?:/\s*(\d+))?")


def recall_score(answer: str, finish: str | None, f: dict) -> dict:
    """turn 4: each fact read off its own line, exactly — a name as told
    (case aside), a number as told"""
    out = fb.ran_out(answer, finish)
    vis = fb.visible(answer) if not out else ""
    read, wrong = {}, []
    for k, rx in _FACT.items():
        m = rx.findall(vis)
        got = m[-1].strip() if m else None
        read[k] = got
        want = str(f[k])
        ok = got is not None and (re.sub(r"[^0-9]", "", got) == want if k in ("code", "crates")
                                  else got.strip().lower().removeprefix("the ") == want.lower())
        if not ok:
            wrong.append(k)
    missing = [k for k, v in read.items() if v is None]
    return {"ok": not out and not wrong, "read": read, "ran_out": out,
            "wrong": [] if out else wrong, "no_answer": not out and len(missing) == len(read)}


def compute_score(answer: str, finish: str | None, f: dict) -> dict:
    """turn 5: the value (a × b) / (a + b), as a fraction or a whole number,
    on its ANSWER line; whether it is in lowest terms kept beside"""
    out = fb.ran_out(answer, finish)
    m = _FRACTION.findall(fb.visible(answer)) if not out else []
    if not m:
        return {"ok": False, "read": None, "ran_out": out, "no_answer": not out}
    p, q = m[-1]
    try:
        got = fractions.Fraction(int(p), int(q or 1))
    except ZeroDivisionError:
        return {"ok": False, "read": f"{p}/{q}", "ran_out": out, "no_answer": False}
    want = compute_answer(f)
    lowest = math.gcd(int(p), int(q or 1)) == 1
    return {"ok": got == want, "read": f"{p}/{q}" if q else p, "ran_out": out,
            "no_answer": False, "lowest_terms": lowest}


def score_turn(t: int, line: dict, items: dict, answer: str, finish: str | None) -> dict:
    """a turn's answer by the board's own scorers (turns 1–3), or exactly
    (turns 4–5), with its loop and whether it gave no answer"""
    if t <= 3:
        task = "mmlu_chat" if t == 1 else "gsm8k_chat" if t == 2 else line["hard"]["task"]
        qid = line["mmlu"] if t == 1 else line["gsm8k"] if t == 2 else line["hard"]["id"]
        need = fb.prompt(task, items[task][qid])[1]
        got = fb.score(task, answer, finish, need)
        got = {"ok": bool(got["ok"]), "read": got["read"], "ran_out": got["ran_out"],
               "no_answer": not got["ran_out"] and got["read"] is None}
    elif t == 4:
        got = recall_score(answer, finish, line["facts"])
    else:
        got = compute_score(answer, finish, line["facts"])
    return {**got, "loop": fb.ended_in_loop(answer, got["ran_out"])}


# ---------------------------------------------------------------------------
# the inputs, on the server
# ---------------------------------------------------------------------------

def _stored(task: str) -> list[tuple[str, str]]:
    """step2388's k3 replies to a task that finished, as (question id, the
    reply without thinking) — the history's assistant turns"""
    from service import config
    from service import frontier as sf
    row = config.OUT_DIR / STEP2388.replace("/", "__")
    f = sf.task_dir(row, task) / sf.ANSWERS
    if not f.exists():
        return []
    out = []
    for (qid, e), a in sorted(sf.read_answers(f).items()):
        text = a.get("answer") or ""
        if a.get("unanswered") or e != 0 or fb.ran_out(text, a.get("finish")):
            continue
        vis = fb.visible(text).strip()
        if vis:
            out.append((str(qid), vis))
    return out


def _tokens_about(text: str) -> int:
    """a text's tokens, about (no tokenizer here): the box counts them"""
    return len(text) // 4 + 6


def inputs(dest: Path, say=print, n_chats: int = N_CHATS, bins: tuple = BINS,
           per_bin: int = PER_BIN) -> dict:
    """the inputs file: Part 2's chats and Part 1's, a line each — question ids,
    the facts, step2388's stored replies; no question text"""
    from service import config
    fb.set_root(config.BENCH_ROOT)
    items = index(["mmlu_chat", "gsm8k_chat", "gpqa_diamond_epoch", "math_l5_epoch"])
    rng = random.Random("switch-32:chats")
    mmlu = sorted(items["mmlu_chat"])
    gsm = sorted(items["gsm8k_chat"])
    gpqa, math_ = sorted(items["gpqa_diamond_epoch"]), sorted(items["math_l5_epoch"])
    if not (mmlu and gsm and (gpqa or math_)):
        raise ValueError("MMLU, GSM8K and GPQA Diamond or MATH Level 5 aren't all on this server")
    mm = rng.sample(mmlu, min(n_chats, len(mmlu)))
    gs: list[str] = []
    while len(gs) < n_chats:
        gs += rng.sample(gsm, len(gsm))
    # turn 3: each GPQA Diamond question 3 times, MATH Level 5 for the rest
    # (each once, then again from the start)
    hard = [("gpqa_diamond_epoch", q) for q in gpqa for _ in range(3)]
    if len(hard) >= n_chats:
        hard = rng.sample(hard, n_chats)
    else:
        more: list[str] = []
        while math_ and len(hard) + len(more) < n_chats:
            more += rng.sample(math_, len(math_))
        hard += [("math_l5_epoch", q) for q in more[:n_chats - len(hard)]]
    rng.shuffle(hard)
    lines = []
    for i in range(n_chats):
        lines.append({"part": "chats", "chat": i, "mmlu": mm[i % len(mm)], "gsm8k": gs[i],
                      "hard": {"task": hard[i % len(hard)][0], "id": hard[i % len(hard)][1]},
                      "facts": facts_of(i)})
    # Part 1: histories of step2388's own stored exchanges, stacked to each
    # memory size, each drawn once; the last question one none of them asks
    pool = [("mmlu_chat", q, r) for q, r in _stored("mmlu_chat")] + \
           [("gsm8k_chat", q, r) for q, r in _stored("gsm8k_chat")]
    drng = random.Random("switch-32:drift")
    drng.shuffle(pool)
    k, chat, drift = 0, 0, []
    for b in bins:
        for _ in range(per_bin):
            hist, before = [], 0
            while True:
                if k >= len(pool):
                    raise ValueError(f"step2388's stored replies run out at the {b:,}-token "
                                     f"chats: {len(pool):,} exchanges for {len(bins) * per_bin} "
                                     "chats")
                task, q, reply = pool[k]
                k += 1
                text = fb.prompt(task, items[task][q])[0]
                hist.append({"task": task, "id": q, "reply": reply})
                if before >= b:
                    break                       # the previous turn, after the memory
                before += _tokens_about(text) + _tokens_about(reply)
            drift.append({"part": "drift", "chat": chat, "bin": b, "history": hist})
            chat += 1
    used = {h["id"] for x in drift for h in x["history"] if h["task"] == "mmlu_chat"}
    held = [q for q in mmlu if q not in used]
    if not held:
        raise ValueError("no MMLU question is left for Part 1's last turns")
    held = drng.sample(held, len(held))
    lines += [{**x, "last": {"task": "mmlu_chat", "id": held[x["chat"] % len(held)]}}
              for x in drift]
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / INPUT
    f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                 encoding="utf-8")
    sha = rb.sha256_file(f)
    hard_n = {t: sum(1 for x in lines if x["part"] == "chats" and x["hard"]["task"] == t)
              for t in ("gpqa_diamond_epoch", "math_l5_epoch")}
    gs_twice = max(0, n_chats - len(gsm))
    meta = {"sha256": sha, "file": str(f), "chats": n_chats, "drift": len(bins) * per_bin,
            "bins": list(bins), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hard": hard_n}
    (dest / INPUTS_META).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    say(f"{f} · sha256 {sha}")
    say(f"  Part 2: {n_chats:,} chats of five turns · MMLU {len(set(mm)):,} questions · GSM8K "
        f"{min(n_chats, len(gsm)):,}" + (f" ({gs_twice:,} of them twice)" if gs_twice else "")
        + f" · GPQA Diamond {hard_n['gpqa_diamond_epoch']:,} (each question up to 3 times) · "
        f"MATH Level 5 {hard_n['math_l5_epoch']:,} · the facts invented, a chat's own")
    say(f"  Part 1: {len(bins) * per_bin:,} chats · "
        + ", ".join(f"about {b:,} tokens of memory: {per_bin}" for b in bins)
        + f" · {k:,} of step2388's stored k3 exchanges (MMLU, GSM8K), each once")
    return meta


def read_input(path: Path) -> list[dict]:
    """the inputs file's lines — or ValueError saying what is wrong"""
    out = []
    try:
        raw = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise ValueError(f"{Path(path).name} can't be read ({e})") from None
    for k, x in enumerate(raw, 1):
        if not x.strip():
            continue
        try:
            r = json.loads(x)
        except ValueError:
            raise ValueError(f"{Path(path).name}: line {k} isn't JSON") from None
        ok = isinstance(r, dict) and isinstance(r.get("chat"), int) and (
            (r.get("part") == "chats" and isinstance(r.get("facts"), dict)
             and isinstance(r.get("hard"), dict))
            or (r.get("part") == "drift" and isinstance(r.get("history"), list)
                and len(r["history"]) >= 1 and isinstance(r.get("last"), dict)))
        if not ok:
            raise ValueError(f"{Path(path).name}: line {k} isn't a chat of the experiment")
        out.append(r)
    keys = [(r["part"], r["chat"]) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{Path(path).name}: a chat given twice")
    return out


def share(lines: list[dict], what: str, part: tuple[int, int] | None) -> list[dict]:
    """a box's chats: every n-th from the i-th, the same on every machine"""
    mine = [r for r in lines if r["part"] == what]
    if not part:
        return mine
    i, n = part
    return [r for r in mine if r["chat"] % n == i - 1]


def tasks_of(lines: list[dict]) -> list[str]:
    got = {"mmlu_chat", "gsm8k_chat"}
    for r in lines:
        if r["part"] == "chats":
            got.add(r["hard"]["task"])
        else:
            got |= {h["task"] for h in r["history"]} | {r["last"]["task"]}
    return sorted(got)


# ---------------------------------------------------------------------------
# the box's server
# ---------------------------------------------------------------------------

PATH_LINE = re.compile(r"id\s+(\d+)\s*\|\s*task\s+-?\d+\s*\|\s*expert path k(\d+) -> k(\d+): "
                       r"(keeping|clearing) (\d+) cached tokens")


class Srv:
    """the box's llama-server, and the lines its log says of each slot"""

    def __init__(self, base: str, log: Path | None = None):
        self.base = base.rstrip("/")
        self.log = log
        self._delim: list[int] | None = None
        self._wipe: list[int] | None = None

    def post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        from service import served as sv
        why = ""
        for _ in range(TRIES):
            try:
                return sv._post(self.base + path, "", body, timeout=timeout or TIMEOUT_S)
            except Exception as e:                      # noqa: BLE001 — tried again, then said
                why = str(e)[:300]
                time.sleep(1)
        raise RuntimeError(f"{path}: the server gave no answer ({why})")

    def prompt(self, msgs: list[dict], path: int) -> list[int]:
        """the chat's prompt as the server makes it for this path: its own
        template (the path's thinking), then its tokens"""
        text = self.post("/apply-template", {"messages": msgs, "expert_path": path}).get("prompt")
        if not isinstance(text, str) or not text:
            raise RuntimeError("/apply-template gave no prompt")
        return self.tokens(text)

    def tokens(self, text: str) -> list[int]:
        got = self.post("/tokenize", {"content": text, "add_special": True,
                                      "parse_special": True}).get("tokens")
        if not isinstance(got, list):
            raise RuntimeError("/tokenize gave no tokens")
        return [int(x["id"]) if isinstance(x, dict) else int(x) for x in got]

    def user_starts(self, toks: list[int]) -> list[int]:
        """where each user message starts: its delimiter's tokens"""
        if self._delim is None:
            self._delim = self.tokens(DELIMITERS[1]["delimiter"])
        d = self._delim
        return [i for i in range(len(toks)) if d and toks[i:i + len(d)] == d]

    def wipe(self, slot: int, path: int) -> None:
        """a slot emptied before a chat or a run starts in it: a short prompt
        of its own, which shares nothing with any chat, read on the path the
        chat starts on — so the chat's first request reuses nothing (the
        server has no erase without --slot-save-path, which the runtime
        refuses). No sampling value; never graded"""
        if self._wipe is None:
            self._wipe = self.tokens("A new conversation begins here.")
        self.post("/completion", {"prompt": self._wipe, "n_predict": 1, "expert_path": path,
                                  "id_slot": slot, "cache_prompt": True})

    def mark(self) -> int:
        try:
            return self.log.stat().st_size if self.log else 0
        except OSError:
            return 0

    def path_line(self, at: int, slot: int, expect: bool = False) -> dict | None:
        """the server's path line for this slot, logged since `at` — read
        again for a second when one is due (a path change), once otherwise"""
        if not self.log:
            return None
        for _ in range(5 if expect else 1):
            try:
                with open(self.log, "rb") as fh:
                    fh.seek(at)
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                return None
            got = [m for m in PATH_LINE.finditer(text) if int(m[1]) == slot]
            if got:
                m = got[-1]
                return {"kind": m[4], "from": int(m[2]), "to": int(m[3]), "n": int(m[5]),
                        "text": m[0][m[0].index("expert path"):]}
            time.sleep(0.2)
        return None


def _timings(d: dict) -> dict:
    t, u = d.get("timings") or {}, d.get("usage") or {}
    return {"cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def common(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Part 2: the chats, both arms
# ---------------------------------------------------------------------------

def memory_why(t: int, path: int, prev: int | None, arm: str, got: dict, s_prev: int | None,
               n_common: int) -> str:
    """'' when the turn's memory is what its arm says: fresh on turn 1; on a
    turn on the same path, the memory reused; at a switch, A re-read (cleared,
    nothing reused) and B kept (the line, and the conversation up to the
    previous question at least) — else why not"""
    c, line = got.get("cache_n"), got.get("line")
    if not isinstance(c, int):
        return "the server gave no cache_n"
    if t == 1:
        return "" if c == 0 else f"turn 1 began with {c:,} tokens of memory"
    if path == prev:
        if line:
            return f"a path line on a turn on the same path: {line['text']}"
        return "" if 0 < c <= n_common else f"cache_n {c:,} on a turn on the same path"
    if arm == "A":
        if not line or line["kind"] != "clearing" or (line["from"], line["to"]) != (prev, path):
            return "no \"clearing\" line at the switch"
        return "" if c == 0 else f"cache_n {c:,} after the memory was cleared"
    if not line or line["kind"] != "keeping" or (line["from"], line["to"]) != (prev, path):
        return "no \"keeping\" line at the switch"
    if s_prev is None or not s_prev <= c <= n_common:
        return (f"cache_n {c:,}, not between the previous question's start "
                f"({s_prev if s_prev is not None else '?'}) and where the prompt parts from "
                f"the memory ({n_common:,})")
    return ""


def expected_rollback(t: int, prev_len: int, s_prev: int | None) -> int | None:
    """where the server's rule rolls back to: turn 2 the checkpoint near the
    first prompt's end (4 tokens before it, or 4 + 512 for a prompt longer
    than that), a later turn the previous question's start"""
    if t == 1:
        return 0
    if t == 2:
        return prev_len - 516 if prev_len > 520 else prev_len - 4
    return s_prev


def one_chat(srv: Srv, line: dict, items: dict, slots: tuple[int, int],
             limit: int | None = None, context: int = CHATS_CONTEXT) -> dict:
    """a chat's five turns, arm A and arm B side by side"""
    hist: list[dict] = []
    turns, stopped = [], ""
    prev_path, prev_toks, s_prev = None, None, None
    for sl in slots:
        srv.wipe(sl, TURNS[0])
    with cf.ThreadPoolExecutor(2) as pool:
        for t, path in enumerate(TURNS, 1):
            msgs = hist + [{"role": "user", "content": turn_text(t, line, items)}]
            toks = srv.prompt(msgs, path)
            lim = min(LIMITS[path], limit) if limit else LIMITS[path]
            if len(toks) + lim > context:
                stopped = (f"turn {t}: the chat ({len(toks):,} tokens) and its limit ({lim:,}) "
                           f"outgrow the slot's {context:,}")
                break
            starts = srv.user_starts(toks)
            n_common = common(prev_toks, toks) if prev_toks else 0
            base = {"max_tokens": lim, "seed": seed_of(line["chat"], t), "expert_path": path}
            bodies = {"A": {**base, "id_slot": slots[0]},
                      "B": {**base, "id_slot": slots[1], "keep_cache_across_paths": True}}

            def ask(arm: str, body: dict) -> dict:
                at = srv.mark()
                due = prev_path is not None and path != prev_path
                t0 = time.time()
                d = srv.post("/v1/chat/completions", {"messages": msgs, **body})
                c = (d.get("choices") or [{}])[0]
                m = c.get("message") or {}
                think, reply = m.get("reasoning_content") or "", m.get("content") or ""
                return {"request": body, "finish": c.get("finish_reason"),
                        "tokens": (d.get("usage") or {}).get("completion_tokens"),
                        "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens"),
                        **_timings(d), "line": srv.path_line(at, body["id_slot"], due),
                        "seconds": round(time.time() - t0, 1), "think": think, "reply": reply}
            futs = {arm: pool.submit(ask, arm, body) for arm, body in bodies.items()}
            got = {arm: f.result() for arm, f in futs.items()}
            mem = {arm: memory_why(t, path, prev_path, arm, got[arm], s_prev, n_common)
                   for arm in got}
            if t == 2 and not any(mem.values()) and got["A"]["cache_n"] != got["B"]["cache_n"]:
                mem["B"] = (f"turn 2 reused {got['B']['cache_n']:,} tokens in B and "
                            f"{got['A']['cache_n']:,} in A, where both hold the same memory")
            for arm in got:
                ans = as_answer(got[arm].pop("think"), got[arm]["reply"])
                got[arm]["answer"] = ans
                got[arm]["score"] = score_turn(t, line, items, ans, got[arm]["finish"])
                got[arm]["same_prompt"] = got[arm]["prompt_tokens"] == len(toks)
            turns.append({"turn": t, "path": path, "from": prev_path, "prompt": len(toks),
                          "user_start_prev": s_prev, "common": n_common,
                          "expected": expected_rollback(t, len(prev_toks or []), s_prev),
                          "arms": got, "memory": mem})
            # the history both arms see next: A's reply as the server gave it
            # (its thinking apart), sent back as the apps send it
            hist = msgs + [{"role": "assistant", "content": got["A"]["reply"]}]
            prev_path, prev_toks = path, toks
            s_prev = starts[-1] if starts else None
    return {"chat": line["chat"], "slots": list(slots), "turns": turns, "stopped": stopped,
            **({"limit": limit} if limit else {})}


def kept(out: Path, name: str) -> dict:
    f = out / name
    if not f.exists():
        return {}
    got = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(x)
        except ValueError:
            continue                            # a line a kill cut short: asked again
        if isinstance(r, dict):
            got[(r["chat"], r.get("direction"))] = r
    return got


def run_jobs(jobs: list, width: int, fn, out: Path, name: str, say) -> tuple[int, str]:
    """each job not yet kept, `width` at a time, its slots its own: (kept,
    why it stopped — '' when every one is)"""
    lock = threading.Lock()
    have = kept(out, name)
    todo = [j for j in jobs if (j[0]["chat"], j[1]) not in have]
    free = list(range(width))
    errors: list[str] = []

    def one(job) -> None:
        with lock:
            w = free.pop()
        try:
            row = fn(job, w)
            with lock, open(out / name, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                have[(row["chat"], row.get("direction"))] = row
            say(row)
        except Exception as e:                          # noqa: BLE001 — said, the rest asked
            with lock:
                errors.append(f"chat {job[0]['chat']}: {type(e).__name__}: {e}"[:300])
        finally:
            with lock:
                free.append(w)
    with cf.ThreadPoolExecutor(max(1, width)) as pool:
        list(pool.map(one, todo))
    return len(have), errors[0] if errors else ""


def chat_words(r: dict) -> str:
    """a chat's line on the box: no question text"""
    turns = " · ".join(
        f"t{x['turn']} k{x['path']}: A {'✓' if x['arms']['A']['score']['ok'] else '✗'}"
        f"/B {'✓' if x['arms']['B']['score']['ok'] else '✗'}"
        + ("" if not any(x["memory"].values()) else " (memory not as said)") for x in r["turns"])
    return f"chat {r['chat']}: {turns}" + (f" · stopped: {r['stopped']}" if r["stopped"] else "")


def run_chats(base: str, lines: list[dict], out: Path, slots: int, say, log: Path | None,
              limit: int | None = None, name: str = CHATS) -> tuple[int, str]:
    """Part 2: each chat in a pair of slots, as many pairs as the server has"""
    srv = Srv(base, log)
    items = index(tasks_of(lines))
    width = max(1, slots // 2)
    return run_jobs([(x, None) for x in lines], width,
                    lambda job, w: one_chat(srv, job[0], items, (2 * w, 2 * w + 1), limit),
                    out, name, lambda r: say(chat_words(r)))


# ---------------------------------------------------------------------------
# Part 1: drift, five runs of the last turn forced along the reference
# ---------------------------------------------------------------------------

def history_msgs(line: dict, items: dict) -> list[dict]:
    out = []
    for h in line["history"]:
        out += [{"role": "user", "content": fb.prompt(h["task"], items[h["task"]][h["id"]])[0]},
                {"role": "assistant", "content": h["reply"]}]
    return out


def completion(srv: Srv, toks: list[int], path: int, slot: int, n: int,
               keep: bool = False, due: bool = False) -> dict:
    """/completion of token ids: greedy, no penalties, the 20 likeliest
    before sampling — Part 1's protocol"""
    body = {"prompt": toks, "n_predict": n, "expert_path": path, "id_slot": slot,
            "cache_prompt": True, "n_probs": N_PROBS, "message_delimiters": DELIMITERS,
            "return_tokens": True, **GREEDY,
            **({"keep_cache_across_paths": True} if keep else {})}
    at = srv.mark()
    d = srv.post("/completion", body)
    probs = []
    for e in d.get("completion_probabilities") or []:
        top = e.get("top_logprobs") or e.get("top_probs") or []
        probs.append([(int(x["id"]), math.exp(x["logprob"]) if "logprob" in x else float(x["prob"]))
                      for x in top])
    toks_out = [int(x) for x in d.get("tokens") or []]
    if not toks_out:
        toks_out = [int(e["id"]) for e in d.get("completion_probabilities") or []]
    return {"tokens": toks_out, "probs": probs, **_timings(d),
            "line": srv.path_line(at, slot, due),
            "sent": {k: v for k, v in body.items() if k not in ("prompt", "message_delimiters")}}


def kl(p: list[tuple[int, float]], q: list[tuple[int, float]]) -> float | None:
    """KL(reference ‖ run) of the next token, over the reference's 20 likeliest
    and all the rest as one: a token of the reference's 20 outside the run's
    is given an equal share of what the run's 20 leave, the rest one share
    too, none above the run's 20th — two distributions over the same bins, so
    never below 0 (and, grouped, never above the full KL)"""
    if not p or not q:
        return None
    qm = dict(q)
    left = max(1 - sum(qm.values()), 0.0)
    missing = [t for t, _ in p if t not in qm]
    share = min(min(qm.values()), left / (len(missing) + 1))
    pv = [v for _, v in p]
    qv = [qm.get(t, share) for t, _ in p]
    pv.append(max(1 - sum(pv), 0.0))
    qv.append(max(1 - sum(qv), 0.0))
    return max(sum(a * math.log(a / max(b, 1e-12)) for a, b in zip(pv, qv) if a > 0), 0.0)


def forced(srv: Srv, toks: list[int], ref: list[int], path: int, slot: int,
           keep: bool = False) -> dict:
    """a run forced along the reference: its distribution at each of the
    reference's tokens, and where its own pick parts from it"""
    pos, dists, flips, reqs = 0, [None] * len(ref), [], []
    first = True
    while pos < len(ref):
        n = min(CHUNK, len(ref) - pos)
        d = completion(srv, toks + ref[:pos], path, slot, n, keep=keep and first,
                       due=keep and first)
        reqs.append({k: d[k] for k in ("cache_n", "prompt_n", "cached_tokens", "line")}
                    | {"at": pos})
        first = False
        got = d["tokens"]
        for k in range(len(got)):
            if k < len(d["probs"]):
                dists[pos + k] = d["probs"][k]
            if got[k] != ref[pos + k]:
                flips.append(pos + k)
                pos += k + 1
                break
        else:
            if len(got) < n:                        # it stopped where the reference goes on
                flips.append(pos + len(got))
                if len(got) < len(d["probs"]):
                    dists[pos + len(got)] = d["probs"][len(got)]
                pos += len(got) + 1
            else:
                pos += len(got)
    return {"dists": dists, "flips": flips, "requests": reqs,
            "sent": d["sent"] if reqs else {}}


def one_drift(srv: Srv, line: dict, items: dict, xy: tuple[int, int], slots: list[int],
              ref_n: int | None = None) -> dict:
    """one chat and direction X→Y: the reference, then A′, N, B and F forced
    along it"""
    x, y = xy
    hist = history_msgs(line, items)
    prev = hist[:-1]                                       # up to the previous question
    last = hist + [{"role": "user", "content": fb.prompt(line["last"]["task"], items[
        line["last"]["task"]][line["last"]["id"]])[0]}]
    p_prev = {p: srv.prompt(prev, p) for p in (x, y)}
    p_last = srv.prompt(last, y)
    s_prev = (srv.user_starts(p_prev[y]) or [None])[-1]
    s_last = (srv.user_starts(p_last) or [None])[-1]
    a, a2, f, nn, b = slots
    for sl, p in ((a, y), (a2, y), (f, x), (nn, y), (b, x)):
        srv.wipe(sl, p)
    with cf.ThreadPoolExecutor(3) as pool:
        ref_f = pool.submit(completion, srv, p_last, y, a, ref_n or REF_TOKENS)
        n1 = pool.submit(completion, srv, p_prev[y], y, nn, 1)
        b1 = pool.submit(completion, srv, p_prev[x], x, b, 1)
        ref, n1, b1 = ref_f.result(), n1.result(), b1.result()
    r_toks, r_dists = ref["tokens"], ref["probs"]
    with cf.ThreadPoolExecutor(4) as pool:
        futs = {"A2": pool.submit(forced, srv, p_last, r_toks, y, a2),
                "F": pool.submit(forced, srv, p_last, r_toks, x, f),
                "N": pool.submit(forced, srv, p_last, r_toks, y, nn),
                "B": pool.submit(forced, srv, p_last, r_toks, y, b, True)}
        runs = {k: v.result() for k, v in futs.items()}
    n_common = {k: common(p_prev[x if k == "B" else y], p_last) for k in ("N", "B")}
    out = {}
    for k, r in runs.items():
        ks = [kl(p, q) for p, q in zip(r_dists, r["dists"])]
        kv = [v for v in ks if v is not None]
        why = drift_memory_why(k, r, n1 if k == "N" else b1 if k == "B" else None, x, y,
                               s_prev, s_last, n_common.get(k, 0))
        out[k] = {"memory": why, "first_kl": ks[0] if ks else None,
                  "mean_kl": statistics.fmean(kv) if kv else None,
                  "kl": [round(v, 5) if v is not None else None for v in ks],
                  "flips": r["flips"], "positions": len(r_toks),
                  "flips_per_1000": 1000 * len(r["flips"]) / len(r_toks) if r_toks else None,
                  "first_flip": r["flips"][0] if r["flips"] else None,
                  "requests": r["requests"], "sent": r["sent"],
                  **({"previous": {k2: (n1 if k == "N" else b1)[k2]
                                   for k2 in ("cache_n", "prompt_n", "line", "sent")}}
                     if k in ("N", "B") else {})}
    return {"chat": line["chat"], "bin": line["bin"], "direction": f"k{x}→k{y}", "x": x, "y": y,
            "s_prev": s_prev, "s_last": s_last, "prompt": len(p_last),
            "memory": runs["B"]["requests"][0]["cache_n"] if runs["B"]["requests"] else None,
            "ref": {"tokens": len(r_toks), "cache_n": ref["cache_n"], "sent": ref["sent"],
                    "memory": "" if ref["cache_n"] == 0 else
                    f"the reference began with {ref['cache_n']:,} tokens of memory"},
            "runs": out}


def drift_memory_why(k: str, r: dict, before: dict | None, x: int, y: int, s_prev, s_last,
                     n_common: int) -> str:
    """'' when a run's memory is what it says: A′ and F fresh; N and B the
    previous turn's (fresh on its path, then the conversation up to the
    previous question kept — B's by the other path, said in its line); every
    later request of a run back to the last question at least"""
    reqs = r["requests"]
    if not reqs:
        return "no request"
    c0 = reqs[0]["cache_n"]
    if k in ("A2", "F"):
        if c0 != 0:
            return f"began with {c0:,} tokens of memory"
    else:
        if (before or {}).get("cache_n") != 0:
            return f"its previous turn began with {(before or {}).get('cache_n')} tokens of memory"
        line = reqs[0]["line"]
        if k == "B" and (not line or line["kind"] != "keeping" or (line["from"], line["to"]) != (x, y)):
            return "no \"keeping\" line at the switch"
        if k == "N" and line:
            return f"a path line on the same path: {line['text']}"
        if s_prev is None or not isinstance(c0, int) or not s_prev <= c0 <= n_common:
            return f"cache_n {c0}, not from the previous question's start ({s_prev}) on"
    for q in reqs[1:]:
        if not isinstance(q["cache_n"], int) or s_last is None or q["cache_n"] < s_last:
            return (f"a later request reused {q['cache_n']} tokens, not the conversation up to "
                    f"the last question ({s_last})")
    return ""


def run_drift(base: str, lines: list[dict], out: Path, slots: int, say, log: Path | None,
              ref_n: int | None = None, name: str = DRIFT) -> tuple[int, str]:
    """Part 1: each chat in both directions, five slots a job"""
    if slots < 5:
        raise ValueError(f"Part 1 runs five slots a chat: {slots} given")
    srv = Srv(base, log)
    items = index(tasks_of(lines))
    width = slots // 5
    jobs = [(x, d) for x in lines for d in ("k3→k6", "k6→k3")]

    def fn(job, w):
        xy = (3, 6) if job[1] == "k3→k6" else (6, 3)
        return one_drift(srv, job[0], items, xy, list(range(5 * w, 5 * w + 5)), ref_n)
    return run_jobs(jobs, width, fn, out, name, lambda r: say(drift_words(r)))


def drift_words(r: dict) -> str:
    def one(k):
        x = r["runs"][k]
        return (f"{k.replace('A2', 'A′')} {x['flips_per_1000']:.1f}"
                if x["flips_per_1000"] is not None else f"{k} —") + ("" if not x["memory"] else "*")
    return (f"drift chat {r['chat']} ({r['memory'] if r['memory'] is not None else '?'} tokens of "
            f"memory) {r['direction']}: flips per 1,000 " + ", ".join(one(k) for k in RUNS)
            + " (* memory not as said)")


# ---------------------------------------------------------------------------
# the check, before plan S is rented
# ---------------------------------------------------------------------------

def run_check(base: str, lines: list[dict], out: Path, slots: int, say, log: Path | None,
              n: int = 5) -> bool:
    """Part 2's first n chats through both arms at CHECK_LIMIT a turn, a line
    a turn and arm; the same turns again as token ids on /completion; Part
    1's first chat of each memory size both ways, CHECK_REF tokens forced.
    No question text. True when every memory is as said"""
    if slots < 5:
        raise ValueError(f"the check runs Part 1 too, five slots a chat: {slots} given")
    srv = Srv(base, log)
    chats = [x for x in lines if x["part"] == "chats"][:n]
    drift = []
    for b in sorted({x["bin"] for x in lines if x["part"] == "drift"}):
        drift.append(next(x for x in lines if x["part"] == "drift" and x["bin"] == b))
    items = index(tasks_of(chats + drift))
    bad = 0
    _, why = run_jobs([(x, None) for x in chats], max(1, min(slots // 2, len(chats))),
                      lambda job, w: one_chat(srv, job[0], items, (2 * w, 2 * w + 1),
                                              CHECK_LIMIT),
                      out, "check-" + CHATS, lambda r: None)
    if why:
        say(f"Part 2's check stopped: {why}")
    got = kept(out, "check-" + CHATS)
    rows = []
    for x in chats:
        r = got.get((x["chat"], None))
        if not r:
            say(f"chat {x['chat']}: no answer")
            bad += 1
            continue
        for tr in r["turns"]:
            for arm in ("A", "B"):
                g, why = tr["arms"][arm], tr["memory"][arm]
                bad += bool(why)
                say(f"chat {x['chat']} · turn {tr['turn']} ({TURN_WORDS[tr['turn']]}, "
                    + (f"k{tr['from']} → k{tr['path']}" if tr["from"] and tr["from"] != tr["path"]
                       else f"k{tr['path']}") + f") · {arm}: cache_n {g['cache_n']}"
                    + (f" (expected {tr['expected']:,})" if arm == "B" and isinstance(
                        tr["expected"], int) else "")
                    + f", prompt_n {g['prompt_n']}, cached_tokens {g['cached_tokens']}"
                    + (f" · \"{g['line']['text']}\"" if g["line"] else " · no path line")
                    + (" · " + ("re-read" if arm == "A" else "kept") + " ✓"
                       if tr["from"] and tr["from"] != tr["path"] and not why else "")
                    + (f" · NOT as said: {why}" if why else ""))
        if r["stopped"]:
            say(f"chat {x['chat']}: stopped — {r['stopped']}")
        rows.append(r)
    # the same turns as token ids, in the chat's own B slot emptied first
    for k, r in enumerate(rows):
        ids = raw_turns(srv, r, chats[k], items, r["slots"][1])
        same = [a == b for a, b in zip(ids, [tr["arms"]["B"]["cache_n"] for tr in r["turns"]])]
        bad += not all(same)
        say(f"chat {r['chat']} · the same turns as token ids (/completion, its delimiters): "
            f"cache_n " + ", ".join(f"turn {i + 1} {c}" for i, c in enumerate(ids))
            + (" — the chat way's ✓" if all(same) else " — NOT the chat way's "
               + str([tr["arms"]["B"]["cache_n"] for tr in r["turns"]])))
    _, why = run_jobs([(x, d) for x in drift for d in ("k3→k6", "k6→k3")], slots // 5,
                      lambda job, w: one_drift(srv, job[0], items,
                                               (3, 6) if job[1] == "k3→k6" else (6, 3),
                                               list(range(5 * w, 5 * w + 5)), CHECK_REF),
                      out, "check-" + DRIFT, lambda r: None)
    if why:
        say(f"Part 1's check stopped: {why}")
    dgot = kept(out, "check-" + DRIFT)
    for x in drift:
        for d in ("k3→k6", "k6→k3"):
            r = dgot.get((x["chat"], d))
            if not r:
                say(f"drift chat {x['chat']} {d}: no answer")
                bad += 1
                continue
            whys = {k: v["memory"] for k, v in r["runs"].items() if v["memory"]}
            if r["ref"]["memory"]:
                whys["A"] = r["ref"]["memory"]
            bad += bool(whys)
            bl = r["runs"]["B"]["requests"][0]["line"] if r["runs"]["B"]["requests"] else None
            say(f"drift chat {x['chat']} (about {x['bin']:,} tokens) {d} · A fresh "
                f"{r['ref']['cache_n']} · B kept {r['memory']} from the previous question at "
                f"{r['s_prev']}" + (f" · \"{bl['text']}\"" if bl else " · no path line")
                + " · N kept " + str(r["runs"]["N"]["requests"][0]["cache_n"]
                                      if r["runs"]["N"]["requests"] else None)
                + " · flips per 1,000 " + ", ".join(
                    f"{k.replace('A2', 'A′')} {v['flips_per_1000']:.0f}"
                    for k, v in r["runs"].items() if v["flips_per_1000"] is not None)
                + (" · NOT as said: " + "; ".join(f"{k} {w}" for k, w in whys.items())
                   if whys else " ✓"))
    (out / CHECK).write_text(json.dumps({"chats": rows, "drift": list(dgot.values()),
                                         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                        indent=1, ensure_ascii=False), encoding="utf-8")
    return bad == 0


def raw_turns(srv: Srv, r: dict, line: dict, items: dict, slot: int) -> list[int | None]:
    """a chat's turns again as token ids, each read and one token written, B's
    way (the switch on), in a slot emptied first: each turn's cache_n"""
    hist, out = [], []
    srv.wipe(slot, TURNS[0])
    for tr in r["turns"]:
        msgs = hist + [{"role": "user", "content": turn_text(tr["turn"], line, items)}]
        d = completion(srv, srv.prompt(msgs, tr["path"]), tr["path"], slot, 1, keep=True,
                       due=tr["from"] is not None and tr["from"] != tr["path"])
        out.append(d["cache_n"])
        hist = msgs + [{"role": "assistant", "content": tr["arms"]["A"]["reply"]}]
    return out


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------

def make_bundle(out: Path, served_as: str, run_state: dict, gguf: dict, server: dict,
                what: str, lines: list[dict], inputs_sha: str,
                part: tuple[int, int] | None) -> tuple[Path | None, dict, dict]:
    """the bundle once every chat of the share is done: its path (None until
    then), {what: chats} in it, and {what: {answers, of}} not finished"""
    import remote_gguf as rg
    name = CHATS if what == "chats" else DRIFT
    have = kept(out, name)
    want = [(x["chat"], None) for x in lines] if what == "chats" else \
        [(x["chat"], d) for x in lines for d in ("k3→k6", "k6→k3")]
    n = sum(1 for k in want if k in have)
    if n < len(want):
        return None, {}, {what: {"answers": n, "of": len(want)}}
    rows = "".join(json.dumps(have[k], ensure_ascii=False) + "\n" for k in want)
    files: dict[str, bytes] = {f"results/switch/{name}": rows.encode("utf-8")}
    log = (out / rg.OWN_LOG).read_text(encoding="utf-8") if (out / rg.OWN_LOG).exists() else ""
    log += "\n===== llama-server, as it started =====\n" + rg.startup_log(out / rg.SERVER_LOG)
    setup = {"suite": SUITE, "served_as": served_as, "runtime": RUNTIME, "thinking": "on",
             "gguf": gguf, "server": server, "gpu": rb.gpu_info(), "where": rg.WHERE,
             "board_commit": rg.board_commit(), "sessions": run_state.get("sessions", 1),
             "box": run_state.get("label") or None, "started_at": run_state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "switch": {"what": what, "inputs_sha256": inputs_sha, "chats": len(lines),
                        "part": f"{part[0]}/{part[1]}" if part else None,
                        "limits": {str(k): v for k, v in LIMITS.items()}, "turns": list(TURNS),
                        "context": CHATS_CONTEXT if what == "chats" else DRIFT_CONTEXT,
                        "reference_tokens": REF_TOKENS, "protocol": protocol_words(what)}}
    files["run.log"] = rb.scrub(log).encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    bundle = {"format": rb.SUITES[SUITE]["format"], "suite": SUITE, "model": served_as,
              "thinking": True, "row": what, "gguf_sha256": gguf["sha256"],
              "tasks": {what: len(want)},
              "answers_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest(),
              "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
              **({"shard": {"i": part[0], "n": part[1]}} if part else {})}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    path = rb.write(out / rb.bundle_name(SUITE, served_as, True, part, parts=[what]), files)
    return path, {what: {"answers": len(want), "of": len(want)}}, {}


def protocol_words(what: str) -> str:
    if what == "chats":
        return ("Part 2: each turn's messages, its limit (k6 98,304, k3 32,768), its seed and its "
                "expert path, pinned to its slot; arm B also keep_cache_across_paths. No sampling "
                "value, no reasoning field, no chat_template_kwargs: each path's own settings, the "
                "file's")
    return ("Part 1's protocol: greedy, no penalties (temperature 0, presence_penalty 0, seed 0, as "
            "DeviceMark's), the 20 likeliest tokens' probabilities before sampling (n_probs 20), "
            "the prompt as token ids on /completion with the chat's message delimiters; each run "
            f"forced along the reference's first {REF_TOKENS} tokens")


# ---------------------------------------------------------------------------
# the import, on the server, and the summary the report reads
# ---------------------------------------------------------------------------

def where() -> Path:
    from service import config
    return config.BENCH_ROOT / "switch"


def pinned() -> dict:
    """plan S's pins: step2388's file and the research tarball"""
    import frontier_box as fx
    return fx.PLAN_PINS["S"]


def checks(b: dict, file_sha: str) -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    import import_frontier as imf
    bundle, setup = b["bundle"], b["setup"]
    bad = imf.setup_problems(setup)
    if bad:
        return bad
    sw = setup.get("switch")
    if bundle.get("format") != rb.SUITES[SUITE]["format"] or not isinstance(sw, dict):
        return [f"its format: the bundle's {bundle.get('format')}, this board reads "
                f"{rb.SUITES[SUITE]['format']} for the switch experiment"]
    out = []
    if bundle.get("model") != SERVED or setup.get("runtime") != RUNTIME:
        out.append(f"it ran {bundle.get('model')} on runtime {setup.get('runtime')}: the "
                   f"experiment runs {SERVED} on {RUNTIME}")
    pins = pinned()
    file_pin = pins.get("step2388.k3k6.gguf")
    got = (setup.get("gguf") or {}).get("sha256")
    if got != file_pin or (file_sha and file_sha != file_pin):
        out.append(f"its file is {str(got)[:16]}…, not step2388's ({str(file_pin)[:16]}…)")
    tb = next((v for k, v in pins.items() if k.startswith("llama-server-")), None)
    srv = setup.get("server") or {}
    if not tb:
        out.append("plan S's tarball isn't pinned yet (frontier_box.PLAN_PINS['S'])")
    elif srv.get("tarball_sha256") != tb:
        out.append(f"its build is {str(srv.get('tarball_sha256'))[:16]}…, not plan S's "
                   f"tarball ({tb[:16]}…)")
    if srv.get("flags") != fb.runtime_flags(RUNTIME) or srv.get("env") != fb.RUNTIMES[RUNTIME]["env"]:
        out.append(f"its server ran {srv.get('flags')} with {srv.get('env')}: runtime {RUNTIME} "
                   f"runs {fb.runtime_flags(RUNTIME)} with {fb.RUNTIMES[RUNTIME]['env']}")
    for p in (3, 6):
        c = (srv.get("paths_checks") or {}).get(str(p)) or {}
        if c.get("pass") is not True:
            out.append(f"the settings check of expert path k{p} didn't pass")
    try:
        meta = json.loads((where() / INPUTS_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if sw.get("inputs_sha256") != meta.get("sha256"):
        out.append(f"its inputs file is {str(sw.get('inputs_sha256'))[:16]}…, and this server's "
                   f"(switch_box.py inputs) is {str(meta.get('sha256'))[:16]}…")
    if sw.get("what") not in ("chats", "drift"):
        out.append(f"it holds {sw.get('what')!r}: the chats or drift")
    return out


def _rows(b: dict) -> tuple[list[dict], str]:
    what = (b["setup"].get("switch") or {}).get("what")
    raw = b["files"].get(f"results/switch/{CHATS if what == 'chats' else DRIFT}")
    if raw is None:
        return [], "it holds no rows"
    try:
        return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()], ""
    except ValueError:
        return [], "its rows aren't JSON lines"


def request_why(r: dict, what: str, line: dict) -> str:
    """'' when every request of a row was asked as the experiment asks; else
    the first that wasn't"""
    if what == "chats":
        for tr in r.get("turns") or []:
            t, p = tr.get("turn"), tr.get("path")
            if not isinstance(t, int) or not 1 <= t <= 5 or p != TURNS[t - 1]:
                return f"a turn {t} on path {p}"
            for arm in ("A", "B"):
                req = (tr.get("arms") or {}).get(arm, {}).get("request") or {}
                keys = set(req) - {"keep_cache_across_paths"}
                if keys != CHAT_KEYS or req.get("keep_cache_across_paths", False) is not (arm == "B"):
                    return f"turn {t} arm {arm} was asked with {sorted(req)}"
                if (req["expert_path"], req["seed"]) != (p, seed_of(line["chat"], t)) or \
                        req["max_tokens"] != LIMITS[p]:
                    return f"turn {t} arm {arm} was asked with {json.dumps(req, sort_keys=True)}"
        return ""
    for k, run in (r.get("runs") or {}).items():
        for sent in [run.get("sent") or {}, ((run.get("previous") or {}).get("sent") or {})]:
            if not sent:
                continue
            keys = set(sent) - {"keep_cache_across_paths"}
            if not keys <= DRIFT_KEYS or any(sent.get(g) != v for g, v in GREEDY.items()) \
                    or sent.get("n_probs") != N_PROBS:
                return f"run {k} was asked with {json.dumps(sent, sort_keys=True)}"
    return ""


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "",
                  register: str = "", public_weights: bool = False) -> int:
    """a box's chats or drift runs, marked again by the board's scorers, into
    $BENCH_ROOT/switch, and the report's summary worked out again"""
    import import_frontier as imf
    bad = imf.bundle_problems(b.get("bundle"))
    if not bad and not isinstance(b.get("setup"), dict):
        bad = ["its setup.json isn't a JSON object"]
    if register or public_weights:
        bad.append("--register is for a model of the board's: the switch experiment is no row")
    if bad:
        for line in bad:
            say(f"refused — {line}")
        return REFUSED
    setup = b["setup"]
    sw = setup.get("switch") or {}
    say(f"{path.name} · sha256 {b['sha256'][:16]} · the switch experiment, "
        f"{sw.get('what')} ({sw.get('chats')} chats" + (f", part {sw['part']}" if sw.get("part")
                                                        else "") + ")")
    d = where()
    try:
        reg = json.loads((d / "imports.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reg = {"imports": []}
    was = next((x for x in reg["imports"] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, on {was['at']}: nothing changed")
        return 0
    bad = checks(b, file_sha)
    rows, why = _rows(b) if not bad else ([], "")
    if why:
        bad.append(why)
    lines = {}
    if not bad:
        try:
            lines = {(x["part"], x["chat"]): x for x in read_input(d / INPUT)}
        except ValueError as e:
            bad.append(f"this server's inputs file: {e}")
    what = sw.get("what")
    if not bad:
        part = rb.shard_of_bundle(b["bundle"])
        mine = share(list(lines.values()), what, part)
        want = {x["chat"] for x in mine}
        got = [r.get("chat") for r in rows]
        if set(got) != want or (what == "chats" and len(got) != len(want)) or \
                (what == "drift" and len(got) != 2 * len(want)):
            bad.append(f"it holds {len(set(got))} of its share's {len(want)} chats, or others")
        for r in rows:
            w = request_why(r, what, lines.get((what, r.get("chat"))) or {})
            if w:
                bad.append(f"chat {r.get('chat')}: {w}")
                break
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    if what == "chats":
        rows = [rescore(r, lines[("chats", r["chat"])]) for r in rows]
    sub = d / what
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / (path.name.removesuffix(".tar.gz") + ".jsonl")
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    reg["imports"].append({"sha256": b["sha256"], "bundle": path.name, "by": by, "at": stamp,
                           "what": what, "part": sw.get("part"), "chats": len(rows),
                           "box": setup.get("box") or "",
                           "gpu": (setup.get("gpu") or {}).get("name") or ""})
    (d / "imports.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    s = summarise(d)
    if what == "chats":
        n_bad = sum(1 for r in rows for tr in r["turns"] if any(tr["memory"].values()))
        say(f"the switch experiment: {len(rows)} chats of part {sw.get('part')} into {dest}"
            + (f" · {n_bad} turns whose memory wasn't as said (reported, not counted)"
               if n_bad else "") + f" · {s['part2']['chats']:,} chats imported in all")
    else:
        say(f"the switch experiment's drift: {len(rows)} runs into {dest} · "
            f"{s['part1']['jobs']:,} imported in all")
    return 0


def rescore(r: dict, line: dict) -> dict:
    """a chat's answers marked again here, by the board's scorers"""
    items = index(tasks_of([line]))
    for tr in r["turns"]:
        for arm in ("A", "B"):
            a = tr["arms"][arm]
            a["score"] = score_turn(tr["turn"], line, items, a.get("answer") or "", a.get("finish"))
    return r


def mcnemar_p(b: int, c: int) -> float:
    """McNemar's exact test, two-sided: b and c the pairs each arm alone got"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: list[bool], b: list[bool]) -> dict:
    """two arms' outcomes, chat by chat: B's minus A's in points with its
    95% interval, and McNemar's p"""
    n = len(a)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    if not n:
        return {"n": 0, "only_a": 0, "only_b": 0, "diff": None, "ci": None, "p": None}
    d = (only_b - only_a) / n
    se = math.sqrt(max(only_a + only_b - (only_b - only_a) ** 2 / n, 0)) / n
    return {"n": n, "only_a": only_a, "only_b": only_b, "diff": 100 * d,
            "ci": [100 * (d - 1.96 * se), 100 * (d + 1.96 * se)], "p": mcnemar_p(only_a, only_b)}


def clustered(rows: list[tuple[dict, dict]]) -> dict:
    """turn 3's questions asked more than once (GPQA Diamond's, up to 3
    times): the sign test on each question's B-minus-A count"""
    by: dict = {}
    for tr, line in rows:
        key = (line["hard"]["task"], line["hard"]["id"])
        by[key] = by.get(key, 0) + int(tr["arms"]["B"]["score"]["ok"]) - \
            int(tr["arms"]["A"]["score"]["ok"])
    plus = sum(1 for v in by.values() if v > 0)
    minus = sum(1 for v in by.values() if v < 0)
    return {"questions": len(by), "b_more": plus, "a_more": minus, "p": mcnemar_p(minus, plus)}


def _median(v: list) -> float | None:
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def summarise(d: Path) -> dict:
    """what the report shows, from every import: Part 2 by turn and arm,
    Part 1 by memory size, direction and run — kept as summary.json"""
    try:
        lines = {(x["part"], x["chat"]): x for x in read_input(d / INPUT)}
    except ValueError:
        lines = {}
    chats, drift = [], []
    for f in sorted((d / "chats").glob("*.jsonl")) if (d / "chats").is_dir() else []:
        chats += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for f in sorted((d / "drift").glob("*.jsonl")) if (d / "drift").is_dir() else []:
        drift += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    turns = []
    loops_k6: dict = {"A": [], "B": []}
    for t in range(1, 6):
        rows = [(c["turns"][t - 1], lines.get(("chats", c["chat"])) or {}) for c in chats
                if len(c["turns"]) >= t]
        counted = [(tr, ln) for tr, ln in rows if not any(tr["memory"].values())]
        arms = {}
        for arm in ("A", "B"):
            sc = [tr["arms"][arm]["score"] for tr, _ in counted]
            g = [tr["arms"][arm] for tr, _ in counted]
            arms[arm] = {
                "n": len(sc), "right": sum(1 for s in sc if s["ok"]),
                "score": 100 * sum(1 for s in sc if s["ok"]) / len(sc) if sc else None,
                "loops": sum(1 for s in sc if s.get("loop")),
                "ran_out": sum(1 for s in sc if s.get("ran_out")),
                "no_answer": sum(1 for s in sc if s.get("no_answer")),
                "wrong_recalls": sum(len(s.get("wrong") or []) for s in sc) if t == 4 else None,
                "tokens": _median([x.get("tokens") for x in g]),
                "cache_n": _median([x.get("cache_n") for x in g]),
                "prompt_n": _median([x.get("prompt_n") for x in g])}
        a_ok = [tr["arms"]["A"]["score"]["ok"] for tr, _ in counted]
        b_ok = [tr["arms"]["B"]["score"]["ok"] for tr, _ in counted]
        a_lp = [bool(tr["arms"]["A"]["score"].get("loop")) for tr, _ in counted]
        b_lp = [bool(tr["arms"]["B"]["score"].get("loop")) for tr, _ in counted]
        if TURNS[t - 1] == 6 and t > 1 and TURNS[t - 2] != 6:
            loops_k6["A"] += a_lp
            loops_k6["B"] += b_lp
        turns.append({"turn": t, "path": TURNS[t - 1], "words": TURN_WORDS[t],
                      "switch": t > 1 and TURNS[t - 1] != TURNS[t - 2],
                      "reached": len(rows), "not_counted": len(rows) - len(counted),
                      "arms": arms, "paired": paired(a_ok, b_ok), "loops": paired(a_lp, b_lp),
                      **({"clustered": clustered(counted)} if t == 3 else {})})
    stopped: dict = {}
    for c in chats:
        if c.get("stopped"):
            k = c["stopped"].split(":", 1)[0]
            stopped[k] = stopped.get(k, 0) + 1
    part1 = []
    for b in sorted({r["bin"] for r in drift}):
        for dirn in ("k3→k6", "k6→k3"):
            rs = [r for r in drift if r["bin"] == b and r["direction"] == dirn]
            runs = {}
            for k in RUNS:
                ok = [r["runs"][k] for r in rs if not r["runs"][k]["memory"]
                      and not r["ref"]["memory"]]
                pos = sum(x["positions"] for x in ok)
                runs[k] = {"n": len(ok), "not_counted": len(rs) - len(ok),
                           "first_kl": statistics.fmean([x["first_kl"] for x in ok
                                                         if x["first_kl"] is not None]) if ok else None,
                           "mean_kl": statistics.fmean([x["mean_kl"] for x in ok
                                                        if x["mean_kl"] is not None]) if ok else None,
                           "flips_per_1000": 1000 * sum(len(x["flips"]) for x in ok) / pos
                           if pos else None,
                           "first_flip": _median([x["first_flip"] if x["first_flip"] is not None
                                                  else x["positions"] for x in ok])}
            part1.append({"bin": b, "direction": dirn, "chats": len(rs),
                          "memory": _median([r.get("memory") for r in rs]), "runs": runs})
    s = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "part2": {"chats": len(chats), "turns": turns, "stopped": stopped,
                   "switch_k6_loops": paired(loops_k6["A"], loops_k6["B"]),
                   "protocol": protocol_words("chats")},
         "part1": {"jobs": len(drift), "rows": part1, "protocol": protocol_words("drift"),
                   "runs": RUN_WORDS}}
    d.mkdir(parents=True, exist_ok=True)
    (d / SUMMARY).write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inputs", help="the experiment's inputs file, its sha256 and what it holds")
    i.add_argument("--out", default="", help="where it goes (default $BENCH_ROOT/switch)")
    sub.add_parser("summary", help="the report's summary worked out again from every import")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        s = summarise(where())
        print(f"{s['part2']['chats']:,} chats and {s['part1']['jobs']:,} drift runs · "
              f"{where() / SUMMARY}")
        return 0
    try:
        inputs(Path(a.out) if a.out else where())
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
