#!/usr/bin/env python3
"""31.3 (brief 31d, masein 4 Nov): step2388 at the model team's token limits.
The team trained step2388 to write up to 98,304 tokens (96 × 1,024), its
thinking and its answer together: 98,304 on the k6 path (thinking on) and
32,768 on the k3 path (thinking off). Plan U asked at the whitepaper's limits
(81,920, 65,536 and 32,768 with thinking on; 16,384, 8,192 and 4,096 off), and
keeps them, so every shard of a benchmark is asked one way. DeviceMark keeps its
own 4,096. 31.3b (masein, 4 Nov): the original Qwen3.6-35B-A3B Q2's easy set,
asked beside step2388's k3 on U40 at 4,096, is continued to 32,768 too, so
both models of the easy set run at the same limit — on its own runtime, as it
was asked (its card's sampling and the thinking switch).

An answer that finished under its old limit is a draw at the new one too — a
limit only matters once it is reached — so only the answers that ran out of
room change. Each is continued from where it stopped, on a rented box, as one
reply, until it finishes or reaches the new limit: the same prefix, then
sampling at the same settings, as a single reply at the new limit would be.

How an answer is continued, on a server started for its path (the runtime's
settings check passed first, as on every step of plan U):

- the messages: the question as it was asked (fb.prompt, and HLE's system
  message), then the cut-off reply as the assistant's unfinished turn — its
  thinking, and any reply after it, exactly as written. llama-server (the
  team's build, at 3ad38b7cc) continues a final assistant message: its
  thinking is written straight after the template's "<think>\\n". Its reply
  after the thinking closed is not: the server writes "\\n</think>\\n\\n"
  between the two itself, and its parser kept the model's own newline before
  </think> in the thinking — one newline the model never wrote. So the whole
  text goes in as the turn's thinking (forms below), and the prompt is held to
  two checks before anything is asked: its text is the question's prompt and
  the reply, character for character, and its tokens are the question's and
  the reply's as the answer counted them (usage.prompt_tokens and
  completion_tokens). 31.3d (masein, 4 Nov): text that reads back up to 8
  tokens otherwise than the model wrote — it split a few tokens otherwise than
  the tokenizer does — is continued from the tokenizer's reading, the
  penalty's window that reading's count, the difference kept with the answer
  and counted by benchmark on the import; more than 8, and it isn't
  continued, and is said;
- the request: its limit (the new limit less the tokens already written), a
  seed of its own (a fresh draw: fb.seed_of(task, "<id>+cont", run)), the
  expert path — and on k6, penalty_prompt_tokens: the reply's tokens. The
  path's presence penalty counts the reply's own tokens and never the
  prompt's (penalty_prompt_tokens 0, the file's); continued, the reply is
  prompt, so the request says how many of the prompt's last tokens are the
  reply (tools/server/server-context.cpp init_sampler: the last N prompt
  tokens enter the sampler's history). k3 has no presence penalty: nothing
  more is sent. No other sampling field: each would replace the path's own;
- the reply comes back whole (the server parses the turn it was given with
  what it wrote after), and is held to beginning with the cut-off one, with
  no thinking opened again and no prompt written into it;
- the loop rule and "ran out of room" apply to the whole continued answer.

Three places:

1. On the server, `inputs`: the answers of a model's rows that ran out of
   room, of every benchmark whose answers are all in — a file for each
   model and thinking continued, cont-k6.jsonl, cont-k3.jsonl and
   cont-q2.jsonl (the answers as the row holds them, no question), its
   sha256 for the box's line, and what each holds:

       cd ~/benchmarks/aienh && sudo docker compose exec -T bench \\
           python scripts/continue_box.py inputs

2. On a box (frontier_box.py's plan C), remote_gguf.py --continue <file>
   --part i/n: every n-th answer of the file from the i-th, the questions
   fetched there as any step's, each continued answer kept the moment it
   arrives (continued.jsonl) — a step pasted again asks only what isn't — and
   its bundle (frontier-cont-served__…) when every one is. With
   --continue-check N, the first N of a benchmark's (--only), a line each,
   and the penalty's test: no bundle.

3. On the server, import_remote.py takes the bundle onto the rows it came
   from — refused unless its file, launch, runtime and settings check are the
   model's, every benchmark's answers are all in, and each continued answer
   began as the one the row holds. The original bundles are untouched; the
   row's answers before are kept beside it (earlier/), and each continued one
   keeps what it was ("continued"): the board shows the new limit's score,
   and the old limit's stays on record — the same answers with each
   continued one counted wrong, as it was. HLE and SimpleQA are graded once,
   on the final answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

SUITE = "frontier-cont"
REFUSED = 2
STEP2388, Q2 = "served/step2388-k3k6-2611", "served/Qwen3.6-35B-A3B-Q2-k3k6-2611"
# 31.3e: the whitepaper's showcase (scripts/showcase_box.py) — its answers that
# ran out of room where U39 asked them (81,920, on 917850e's image) continued
# with step2388's k6 answers; no benchmark, its questions the pinned file's
SHOWCASE = "showcase"
SHOWCASE_ROOM = 2048                # beside its limit in its slot, as showcase_box
EASY = ("mmlu_chat", "gsm8k_chat")
# what is continued, and to what, by model and thinking: its inputs file's
# key, its limit (thinking and answer together), its expert path, and its
# benchmarks. step2388 at the model team's limits (Roohi, 4 Nov): the
# whitepaper's benchmarks each path asks, and on k3 the easy set — never
# DeviceMark (its own 4,096) nor the showcase (asked at the new limit on
# U39). 31.3b (masein, 4 Nov): and the original Qwen3.6-35B-A3B Q2's easy set
# on its own runtime (card-qwen-k3k6, no path: its card's sampling sent, as it
# was asked), so both models of the easy set run at the same limit
SPECS = {
    (STEP2388, True): {"key": "k6", "limit": 98_304, "path": 6,
                       "tasks": ("gpqa_diamond_epoch", "otis_aime_epoch", "math_l5_epoch",
                                 "mmlupro_tiger", "arc_agi2_public", "ifeval_google",
                                 "ifbench_test", "simpleqa_epoch", "hle_text_cais", SHOWCASE)},
    (STEP2388, False): {"key": "k3", "limit": 32_768, "path": 3,
                        "tasks": ("gpqa_diamond_epoch", "math_l5_epoch", "mmlupro_tiger",
                                  "ifeval_google", "ifbench_test", *EASY)},
    (Q2, False): {"key": "q2", "limit": 32_768, "path": None, "tasks": EASY},
}
INPUT = "cont-{key}.jsonl"
DONE = "continued.jsonl"
CHECK = "continue-check.jsonl"
# 31.3c: a k6 step's tests, run before its answers and kept in its bundle
TESTS = "continue-tests.json"
SETUP_KEY = "continuation"
# a reply's thinking closed: what may follow </think> before the reply, the
# model's habit first — the form whose prompt gives back the reply's tokens
CLOSES = ("</think>\n\n", "</think>\n", "</think>")
# no answer within this, tried again this many times (the server's own
# failure on the request: said, and the answer left for the next paste)
TRIES = 3
# what a continued reply must not hold of the question: its first characters
QUESTION_HEAD = 80
# 31.3d (masein, 4 Nov): a reply whose exact text reads back up to this many
# tokens more or fewer than the model wrote — it split a few tokens otherwise
# than the tokenizer does (C1: 81,917 of 81,920) — is continued from the
# tokenizer's reading of the same text, the penalty's window that reading's
# count, and the difference kept with the answer; more, and it isn't
RETOKEN_CAP = 8


def spec_of(model: str, on: bool) -> dict | None:
    """what this model's answers with this thinking are continued to — None
    for one that isn't continued"""
    got = SPECS.get((model, bool(on)))
    return {**got, "model": model, "on": bool(on)} if got else None


def context(spec: dict) -> int:
    """a slot's context: the new limit, and the prompt's room of the longest
    benchmark continued (ARC-AGI-2's on k6: 131,072)"""
    return spec["limit"] + max(fb.BENCH[t]["room"] if t in fb.BENCH else SHOWCASE_ROOM
                               for t in spec["tasks"])


def label_of(task: str) -> str:
    return "the showcase" if task == SHOWCASE else fb.BENCH[task]["label"]


def asked_as(spec: dict) -> dict:
    """how a question was asked when nothing else says: on an expert path, the
    path alone (service/frontier.settings, its limit aside)"""
    return {"expert_path": spec["path"]} if spec.get("path") else {}


def template_how(asked: dict) -> dict:
    """what the template is told beside the messages: the path, or the
    thinking switch said out loud"""
    return {k: asked[k] for k in ("expert_path", "chat_template_kwargs") if k in asked}


def seed_of(task: str, qid: str, epoch: int) -> int:
    """a fresh draw for the continuation, the same on every machine"""
    return fb.seed_of(task, f"{qid}+cont", epoch)


def answer_sha(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()


def split(answer: str) -> tuple[str, str, bool]:
    """an answer as service/served.ask keeps it — "<think>\\n{thinking}\\n</think>
    \\n\\n{reply}", or the reply alone — as (thinking, reply, thought). The
    server's parser stops the thinking at the first </think>, so the first
    one written after it is ours"""
    head, sep = "<think>\n", "\n</think>\n\n"
    if answer.startswith(head) and sep in answer[len(head) - 1:]:
        at = answer.index(sep, len(head) - 1)
        return answer[len(head):at], answer[at + len(sep):], True
    return "", answer, False


def forms(answer: str, path: int) -> list[tuple[str, str]]:
    """the texts the model may have written for this answer, the likeliest
    first: (form, text). k6: a reply with nothing after its thinking was cut
    in its thinking — or right after closing it; one with a reply closed its
    thinking. k3: its reply"""
    think, reply, thought = split(answer)
    if path != 6:
        return [("reply", answer)]
    if not thought:
        return [("closed" + _close_words(c), c + reply) for c in CLOSES]
    if not reply:
        return [("open", think)] + [("closed" + _close_words(c), think + c) for c in CLOSES]
    return [("closed" + _close_words(c), think + c + reply) for c in CLOSES]


def _close_words(c: str) -> str:
    n = c.count("\n")
    return f", {n} newline{'s' if n != 1 else ''} after </think>" if n else ", nothing after </think>"


def turn(text: str, path: int) -> dict:
    """the assistant's unfinished turn: on k6 the whole text as its thinking
    (the server writes it straight after "<think>\\n"), on k3 as its reply"""
    if path == 6:
        return {"role": "assistant", "reasoning_content": text, "content": ""}
    return {"role": "assistant", "content": text}


def messages(task: str, text: str) -> list[dict]:
    """the question as service/served.ask asked it — the showcase's alone, as
    showcase_box asked it"""
    system = fb.system_of(task) if task in fb.BENCH else None
    return [*([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": text}]


def request(line: dict, msgs: list[dict], form_text: str, spec: dict,
            asked: dict | None = None, window: int | None = None) -> dict:
    """what a continuation is asked with: what the question was asked with
    (`asked`, service/frontier.settings: on an expert path the path alone,
    nothing that samples; on the Q2's runtime its card's sampling and the
    thinking switch, as it was asked), the messages and the unfinished turn,
    its limit and a seed of its own — and on k6 how many of the prompt's last
    tokens are the reply's. The Q2's requests never set penalty_prompt_tokens:
    the server's default counts the prompt with the reply, and continued, the
    prompt is the same text — the same setting. 31.3d: `window`, the reply's
    tokens as the tokenizer reads its text, when that isn't what the model
    wrote"""
    asked = asked_as(spec) if asked is None else asked
    path = spec.get("path")
    return {**{k: v for k, v in asked.items() if k != "max_tokens"},
            "messages": [*msgs, turn(form_text, path)],
            "max_tokens": spec["limit"] - int(line["tokens"]),
            "seed": seed_of(line["task"], str(line["id"]), int(line["epoch"])),
            **({"penalty_prompt_tokens": int(line["tokens"]) if window is None else window}
               if path == 6 else {})}


# ---------------------------------------------------------------------------
# the box
# ---------------------------------------------------------------------------

class Server:
    """the box's llama-server, through service/served's requests"""

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        from service import served as sv
        return sv._post(self.base + path, "", body, timeout=timeout)

    def template(self, msgs: list[dict], how: dict) -> str:
        got = self.post("/apply-template", {"messages": msgs, **how}).get("prompt")
        if not isinstance(got, str) or not got:
            raise ValueError("/apply-template gave no prompt")
        return got

    def count(self, text: str) -> int:
        got = self.post("/tokenize", {"content": text, "add_special": True,
                                      "parse_special": True}).get("tokens")
        if not isinstance(got, list):
            raise ValueError("/tokenize gave no tokens")
        return len(got)


def prepare(srv: Server, line: dict, msgs: list[dict], spec: dict,
            asked: dict | None = None) -> dict:
    """the form this answer is continued in, held to its two checks — {form,
    text, prompt_tokens, continued_tokens, reply_tokens} — or {why} when no
    form gives back the question's and the reply's tokens. 31.3d: when none
    does exactly, the likeliest form whose text reads back within RETOKEN_CAP
    tokens of the reply's, with {retokenized: {wrote, read, diff}}"""
    path = spec.get("path")
    how = template_how(asked_as(spec) if asked is None else asked)
    p0 = srv.template(msgs, how)
    n0 = srv.count(p0)
    want0, toks = line.get("prompt_tokens"), int(line["tokens"])
    if isinstance(want0, int) and n0 != want0:
        return {"why": f"the question's prompt is {n0:,} tokens here, {want0:,} when it was "
                       "asked: not the same question or template"}
    want = n0 + toks
    seen, near = [], None
    for form, text in forms(line["answer"], path):
        p1 = srv.template([*msgs, turn(text, path)], how)
        if p1 != p0 + text:
            seen.append(f"{form}: the prompt isn't the question's and the reply's text")
            continue
        n1 = srv.count(p1)
        got = {"form": form, "text": text, "prompt_tokens": n0, "continued_tokens": n1,
               "reply_tokens": n1 - n0}
        if n1 == want:
            return got
        if near is None and abs(n1 - want) <= RETOKEN_CAP:
            near = {**got, "retokenized": {"wrote": toks, "read": n1 - n0, "diff": n1 - want}}
        seen.append(f"{form}: {n1 - n0:,} tokens")
    if near is not None:
        return near
    return {"why": f"the reply's text doesn't give back its {toks:,} tokens, nor within "
                   f"{RETOKEN_CAP} of them (" + "; ".join(seen) + ")"}


SETTINGS = ("temperature", "top_k", "top_p", "min_p", "presence_penalty", "frequency_penalty",
            "repeat_penalty", "repeat_last_n", "penalty_prompt_tokens", "dry_multiplier", "seed",
            "n_predict", "expert_path")


def continue_one(srv: Server, line: dict, item: dict, spec: dict,
                 asked: dict | None = None) -> dict:
    """one answer continued: its record for continued.jsonl (status
    "continued", or "refused" and why)"""
    task, qid, e = line["task"], str(line["id"]), int(line["epoch"])
    path, limit = spec.get("path"), spec["limit"]
    text = item["question"] if task == SHOWCASE else fb.prompt(task, item)[0]
    msgs = messages(task, text)
    base = {"task": task, "id": qid, "epoch": e, "limit": limit,
            "was": {"sha256": answer_sha(line["answer"]), "tokens": int(line["tokens"]),
                    "finish": line.get("finish"), "budget": line.get("budget"),
                    "prompt_tokens": line.get("prompt_tokens")}}
    if int(line["tokens"]) >= limit:
        return {**base, "status": "refused",
                "why": f"{line['tokens']:,} tokens already, the limit {limit:,}"}
    prep = prepare(srv, line, msgs, spec, asked)
    if prep.get("why"):
        return {**base, "status": "refused", "why": prep["why"]}
    # "verbose": the settings the server applied come back with the reply
    # (generation_settings), kept with each answer — it changes nothing it samples
    body = {**request(line, msgs, prep["text"], spec, asked, prep.get("reply_tokens")),
            "stream": False, "verbose": True}
    from service import served as sv
    t0, got, why = time.time(), None, ""
    for k in range(TRIES):
        try:
            got = srv.post("/v1/chat/completions", body, timeout=sv.timeout_for(body))
            break
        except sv.ItemError as x:
            # the server's answer to this request alone (a 4xx, no answer in
            # its time): said, the others asked
            return {**base, "status": "refused", "why": f"the server refused it ({str(x)[:300]})"}
        except (sv._Retry, ValueError) as x:
            why = str(x)[:300]
            time.sleep(min(60, 5 * (k + 1)))
    if got is None:
        return {**base, "status": "failed", "why": f"no answer from the server ({why})"}
    ch = (got.get("choices") or [{}])[0]
    msg, use = ch.get("message") or {}, got.get("usage") or {}
    verbose = got.get("__verbose") if isinstance(got.get("__verbose"), dict) else {}
    gs = verbose.get("generation_settings") if isinstance(verbose.get("generation_settings"),
                                                          dict) else {}
    think, reply = msg.get("reasoning_content") or "", msg.get("content") or ""
    final = f"<think>\n{think}\n</think>\n\n{reply}" if think else reply
    finish = ch.get("finish_reason") if isinstance(ch.get("finish_reason"), str) else None
    new = use.get("completion_tokens")
    checks = continuity(line["answer"], final, prep["form"], text, path)
    evaluated = use.get("prompt_tokens")
    if evaluated != prep["continued_tokens"]:
        checks.append(f"the server read {evaluated} prompt tokens, not the "
                      f"{prep['continued_tokens']:,} checked")
    if not isinstance(new, int):
        checks.append("the server didn't say how many tokens it wrote")
    out = fb.ran_out(final, finish)
    rec = {**base, "status": "continued" if not checks else "refused", "form": prep["form"],
           "max_tokens": body["max_tokens"], "seed": body["seed"],
           "penalty_prompt_tokens": body.get("penalty_prompt_tokens"),
           "prompt_tokens": prep["prompt_tokens"], "tokens_evaluated": evaluated,
           "new_tokens": new, "tokens": int(line["tokens"]) + (new or 0), "finish": finish,
           "ran_out": out, "loop": fb.ended_in_loop(final, out),
           "settings": {k: gs[k] for k in SETTINGS if k in gs},
           "seconds": round(time.time() - t0, 1),
           **({"timings": {k: v for k, v in (got.get("timings") or {}).items()
                           if k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                    "cache_n")}} if isinstance(got.get("timings"), dict) else {}),
           # 31.3d: its text read back otherwise than the model wrote it
           **({"retokenized": prep["retokenized"]} if prep.get("retokenized") else {}),
           "answer": final}
    if checks:
        rec["why"] = "; ".join(checks)
    return rec


def retoken_words(r: dict) -> str:
    """31.3d: an answer whose text reads back a few tokens otherwise, in words"""
    x = r.get("retokenized") or {}
    if not x:
        return ""
    d = int(x.get("diff") or 0)
    return (f" · its text reads back as {x.get('read'):,} tokens, {abs(d)} "
            f"{'fewer' if d < 0 else 'more'} than the {x.get('wrote'):,} written: continued from "
            "that reading, the penalty's window its count")


def continuity(was: str, final: str, form: str, question: str, path: int) -> list[str]:
    """what is wrong with a continued answer as one reply — [] when it begins
    with the cut-off one, writes no thinking opened again and no prompt"""
    w_think, w_reply, _ = split(was)
    f_think, f_reply, f_thought = split(final)
    out = []
    if path != 6:
        if not final.startswith(was):
            out.append("the reply doesn't begin with the cut-off one")
        new = final[len(was):]
    elif form == "open":
        if not (f_thought and f_think.startswith(w_think)):
            out.append("its thinking doesn't begin with the cut-off thinking")
        new = f_think[len(w_think):] + f_reply
    else:
        if f_think != w_think or not f_reply.startswith(w_reply):
            out.append("its thinking isn't the cut-off one, or its reply doesn't begin with "
                       "the cut-off reply")
        new = f_reply[len(w_reply):]
    if "<think>" in new:
        out.append("it opens its thinking again")
    if "<|im_start|>" in new or "<|im_end|>" in new:
        out.append("it writes a turn of the chat")
    head = question.strip()[:QUESTION_HEAD]
    if len(head) >= 40 and head in new:
        out.append("it writes the question again")
    return out


def read_input(path: Path) -> list[dict]:
    """the inputs file: an answer a line, each with what continuing it needs —
    or ValueError saying what is wrong with it"""
    out = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise ValueError(f"{Path(path).name} can't be read ({e})") from None
    for k, x in enumerate(lines, 1):
        if not x.strip():
            continue
        try:
            r = json.loads(x)
        except ValueError:
            raise ValueError(f"{Path(path).name}: line {k} isn't JSON") from None
        if not (isinstance(r, dict) and (r.get("task") in fb.BENCH or r.get("task") == SHOWCASE)
                and isinstance(r.get("answer"), str)
                and isinstance(r.get("tokens"), int) and isinstance(r.get("epoch"), int)
                and r.get("id") is not None):
            raise ValueError(f"{Path(path).name}: line {k} isn't an answer to continue")
        out.append(r)
    keys = [(r["task"], str(r["id"]), r["epoch"]) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{Path(path).name}: an answer given twice")
    return out


def part_of(lines: list[dict], part: tuple[int, int] | None, limit: int) -> list[dict]:
    """a box's share: the answers by the room they may still write, most
    first, each to the box with the least so far — so the boxes end
    together; the same shares on every machine"""
    order = sorted(lines, key=lambda r: (-(limit - r["tokens"]), r["task"], str(r["id"]),
                                         r["epoch"]))
    if not part:
        return order
    i, n = part
    load, mine = [0] * n, []
    for r in order:
        k = min(range(n), key=lambda j: (load[j], j))
        load[k] += limit - r["tokens"]
        if k == i - 1:
            mine.append(r)
    return mine


def key_of(r: dict) -> tuple[str, str, int]:
    return r["task"], str(r["id"]), int(r["epoch"])


def kept(out: Path, name: str = DONE) -> dict[tuple[str, str, int], dict]:
    f = out / name
    if not f.exists():
        return {}
    got = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(x)
        except ValueError:
            continue                        # a line a kill cut short: asked again
        if isinstance(r, dict) and r.get("status") in ("continued", "refused"):
            got[key_of(r)] = r
    return got


def line_words(r: dict, k: int, of: int) -> str:
    label = label_of(r["task"])
    if r["status"] != "continued":
        return f"{label} {k} of {of} (question {r['id']}, run {r['epoch']}): not continued — {r['why']}"
    return (f"{label} {k} of {of} (question {r['id']}, run {r['epoch']}): {r['was']['tokens']:,} → "
            f"{r['tokens']:,} tokens · " + ("finished" if not r["ran_out"] else
                                            f"ran out at {r['limit']:,}")
            + (" · ended in a loop" if r["loop"] else "") + f" · {r['seconds']:.0f} s"
            + retoken_words(r))


def ask(base: str, lines: list[dict], out: Path, spec: dict, slots: int, say,
        name: str = DONE, asked: dict[str, dict] | None = None,
        questions: list[dict] | None = None) -> tuple[int, str]:
    """each answer not yet continued, `slots` at a time: (kept, why it stopped
    — '' when every one is)"""
    from concurrent.futures import ThreadPoolExecutor

    from service import config
    srv, limit = Server(base), spec["limit"]
    have = kept(out, name)
    todo = [r for r in lines if key_of(r) not in have]
    items: dict[str, dict] = {}
    for t in sorted({r["task"] for r in todo} - {SHOWCASE}):
        items.update({f"{t}\0{it['id']}": it for it in fb.load(t, config.BENCH_ROOT)})
    # 31.3e: the showcase's from its pinned questions file (showcase_box.questions)
    items.update({f"{SHOWCASE}\0own-{q['n']}": {"id": f"own-{q['n']}", "question": q["question"]}
                  for q in questions or []})
    if todo:
        say(f"{len(todo):,} of {len(lines):,} answers to continue here, {len(have):,} kept "
            f"already · to {limit:,} tokens" + (f" on k{spec['path']}" if spec.get("path") else "")
            + f", {slots} at a time")
    lock, stop, n = threading.Lock(), [], {"k": len(have)}

    def one(r: dict) -> None:
        if stop:
            return
        it = items.get(f"{r['task']}\0{r['id']}")
        if it is None:
            got = {"task": r["task"], "id": str(r["id"]), "epoch": int(r["epoch"]),
                   "status": "refused", "why": "no such question in this box's copy",
                   "limit": limit, "was": {"sha256": answer_sha(r["answer"]),
                                           "tokens": r["tokens"]}}
        else:
            try:
                got = continue_one(srv, r, it, spec, (asked or {}).get(r["task"]))
            except Exception as e:                      # noqa: BLE001 — said, the rest kept
                got = {"status": "failed", "why": f"{type(e).__name__}: {str(e)[:300]}"}
        if got["status"] == "failed":
            with lock:
                stop.append(f"{label_of(r['task'])} question {r['id']}, run "
                            f"{r['epoch']}: {got['why']}")
            return
        with lock:
            with open(out / name, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(got, ensure_ascii=False) + "\n")
            n["k"] += 1
            say(line_words(got, n["k"], len(lines)))

    with ThreadPoolExecutor(max_workers=max(1, slots)) as pool:
        list(pool.map(one, todo))
    return n["k"], (stop[0] if stop else "")


def summary(rows: list[dict]) -> dict:
    """by benchmark: continued, not continued and why, tokens before and
    after, still ran out, loops — counts only"""
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["task"], {"continued": 0, "refused": 0, "tokens_before": 0,
                                       "tokens_after": 0, "still_ran_out": 0, "loops": 0,
                                       "finished": 0, "retokenized": 0, "why": {}})
        if r["status"] != "continued":
            s["refused"] += 1
            w = str(r.get("why") or "")[:120]
            s["why"][w] = s["why"].get(w, 0) + 1
            continue
        s["continued"] += 1
        s["tokens_before"] += int(r["was"]["tokens"])
        s["tokens_after"] += int(r["tokens"])
        s["still_ran_out"] += bool(r["ran_out"])
        s["finished"] += not r["ran_out"]
        s["loops"] += bool(r["loop"])
        s["retokenized"] += bool(r.get("retokenized"))
    return out


def make_bundle(out: Path, served_as: str, thinking: bool, state: dict, gguf: dict,
                server: dict, rec: dict, lines: list[dict], input_sha: str,
                part: tuple[int, int] | None) -> tuple[Path | None, dict, dict]:
    """the bundle once every answer of the box's share is continued or said
    not to be: its path (None until then), {task: answers}, {task: {answers,
    of}} not done"""
    import remote_gguf as rg
    have = kept(out)
    mine = [have[key_of(r)] for r in lines if key_of(r) in have]
    if len(mine) < len(lines):
        by: dict[str, dict] = {}
        for r in lines:
            x = by.setdefault(r["task"], {"answers": 0, "of": 0})
            x["of"] += 1
            x["answers"] += key_of(r) in have
        return None, {}, {t: x for t, x in by.items() if x["answers"] < x["of"]}
    spec = spec_of(served_as, thinking)
    row = served_as.replace("/", "__") + ("__thinking" if thinking else "")
    files: dict[str, bytes] = {}
    tasks: dict[str, int] = {}
    for t in [t for t in (*fb.ALL, SHOWCASE) if any(r["task"] == t for r in mine)]:
        got = [r for r in mine if r["task"] == t]
        tasks[t] = len(got)
        files[f"results/{row}/{t}_0shot/frontier/{DONE}"] = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in got).encode("utf-8")
    log = (out / rg.OWN_LOG).read_text(encoding="utf-8") if (out / rg.OWN_LOG).exists() else ""
    log += "\n===== llama-server, as it started =====\n" + rg.startup_log(out / rg.SERVER_LOG)
    setup = {"suite": SUITE, "protocol": fb.VERSION, "served_as": served_as,
             "based_on": rec.get("based_on") or "",
             **({"runtime": rec["runtime"]} if rec.get("runtime") else {}),
             "thinking": "on" if thinking else "off", "gguf": gguf, "server": server,
             "gpu": rb.gpu_info(), "where": rg.WHERE, "board_commit": rg.board_commit(),
             "sessions": state.get("sessions", 1), "box": state.get("label") or None,
             "image": (rg.board_commit() or "")[:7] or None,
             "started_at": state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **({"price": {"per_hour": rg.price_of(state), "by": "the box line (--price)"}}
                if rg.price_of(state) is not None else {}),
             SETUP_KEY: {"limit": spec["limit"], "path": spec["path"], "key": spec["key"],
                         **({"tests": json.loads((out / TESTS).read_text(encoding="utf-8"))}
                            if (out / TESTS).exists() else {}),
                         "input_sha256": input_sha,
                         "part": {"i": part[0], "n": part[1]} if part else None,
                         "answers": len(mine), "by_task": summary(mine)}}
    files["run.log"] = rb.scrub(log).encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    digest = hashlib.sha256(b"".join(files[n] for n in sorted(files)
                                     if n.endswith("/" + DONE))).hexdigest()
    bundle = {"format": rb.SUITES[SUITE]["format"], "suite": SUITE, "model": served_as,
              "thinking": thinking, "row": row, "gguf_sha256": gguf["sha256"], "tasks": tasks,
              "answers_sha256": digest,
              "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    if part:
        bundle["shard"] = {"i": part[0], "n": part[1]}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    dest = rb.write(out / rb.bundle_name(SUITE, served_as, thinking, part), files)
    return dest, {t: {"answers": n, "of": n} for t, n in tasks.items()}, {}


# ---------------------------------------------------------------------------
# the check before U-cont: --continue-check N, and the penalty's test
# ---------------------------------------------------------------------------

# a short question whose answer thinks for a while on k6
PENALTY_Q = ("List the prime numbers below 60, then say which of them are one more or one "
             "less than a multiple of six, and why every prime above 3 is.")
PENALTY_CUT, PENALTY_MORE = 96, 160
# 31.3c: the batching comparison — the same uncut reply asked alone, then among
# this many others asked at once (a k6 continuation box runs 9 slots), each of
# them long enough to outlast it
FILLERS, FILLER_TOKENS = 8, 1500
FILLER_HEAD_S = 3.0                 # the others writing when it starts
FILLER_Q = "Count from one to three hundred in words, one number a line ({k})."
# the window's probe: a question of one word many times, an unfinished turn of
# another, and room for one more word — whose likeliest are both words
PROBE_Q = ("Here is a list. Write one more word for it: "
           + ", ".join(["cloud"] * 40) + ".")
PROBE_TURN = "The list so far, and it goes on: " + ", ".join(["river"] * 40) + ","
# 31.3c: the probe's samplers opened, for its requests alone — temperature 1, no
# top-k, top-p or min-p — so every candidate's probability comes back after
# them: on C1 (4 Nov) the path's top-p 0.95 kept one token after forty
# "river," and the probe could tell nothing. The path's presence penalty
# stays, being nothing sent
PROBE_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
PROBE_N = 100
# a shift within this of the penalty (or of none), in logits, is it
PROBE_TOL = 0.2


def penalty_test(base: str, path: int) -> dict:
    """penalty_prompt_tokens and the continuation, tested on a k6 box — the
    window decides; the other two are said beside it.

    1. Greedy, on a short example: one reply of CUT+MORE tokens, against the
       same reply cut at CUT and continued for MORE — with the cut reply's
       tokens counted (penalty_prompt_tokens CUT) and with none (0). Where
       they agree, how far apart their log probabilities are; where they
       part, each one's two likeliest. Temperature 0 is the test's alone; the
       path's presence penalty stays, being nothing sent.
    2. 31.3c: batching, alone — the same uncut reply asked alone, then among
       FILLERS others asked at once: the same measures. llama.cpp's logits
       are "not guaranteed to be bit-for-bit identical for different batch
       sizes (prompt processing vs. token generation)" (its server README,
       cache_prompt); a continued prefix is read in one batch where it was
       written a token at a time. This shows how far apart batching alone
       puts two replies on this card and file, beside 1's.
    3. The window: which of a continued prompt's tokens the penalty counts — a
       question of one word many times, an unfinished turn of another, and
       the next token's probabilities before the samplers and after them
       (path_check's probe, the samplers opened: PROBE_SAMPLING). With
       penalty_prompt_tokens the turn's tokens, the turn's word is the path's
       presence penalty down and the question's isn't; with 0 neither is;
       with -1 both are"""
    srv = Server(base)
    out = {"greedy": greedy_test(srv, path), "batching": batching_test(srv, path),
           "window": window_test(srv, path)}
    out["verdict"] = out["window"].get("verdict") or "inconclusive"
    return out


def _greedy(srv: "Server", msgs: list[dict], n: int, path: int, **extra) -> dict:
    """one greedy reply, each token's log probability and its two likeliest"""
    r = srv.post("/v1/chat/completions", {
        "messages": msgs, "max_tokens": n, "temperature": 0, "expert_path": path, "seed": 0,
        "cache_prompt": False, "logprobs": True, "top_logprobs": 2, **extra})
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    steps = ((ch.get("logprobs") or {}).get("content") or [])
    return {"think": m.get("reasoning_content") or "", "reply": m.get("content") or "",
            "tokens": (r.get("usage") or {}).get("completion_tokens"),
            "prompt": (r.get("usage") or {}).get("prompt_tokens"),
            "steps": [(s.get("token"), s.get("logprob"),
                       [(x.get("token"), x.get("logprob")) for x in s.get("top_logprobs") or []])
                      for s in steps]}


def agree(a: list, b: list) -> dict:
    """two greedy replies' steps side by side: how many tokens they agree on
    from the start, how far apart the chosen tokens' log probabilities are
    there (at most, and the median), and where they part each one's two
    likeliest"""
    same, diffs = 0, []
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        same += 1
        if isinstance(x[1], (int, float)) and isinstance(y[1], (int, float)):
            diffs.append(abs(x[1] - y[1]))
    out = {"same_tokens": same, "of": min(len(a), len(b)),
           "max_diff": max(diffs) if diffs else None,
           "median_diff": sorted(diffs)[len(diffs) // 2] if diffs else None}
    if same < out["of"]:
        out["parted"] = {"first": a[same][2][:2], "second": b[same][2][:2]}
    return out


def greedy_test(srv: "Server", path: int) -> dict:
    q = [{"role": "user", "content": PENALTY_Q}]
    whole = _greedy(srv, q, PENALTY_CUT + PENALTY_MORE, path)
    cut = _greedy(srv, q, PENALTY_CUT, path)
    out = {"cut": cut["tokens"], "more": PENALTY_MORE, "whole": whole["tokens"]}
    if cut["tokens"] != PENALTY_CUT or cut["reply"] or len(whole["steps"]) <= PENALTY_CUT:
        return {**out, "verdict": "inconclusive",
                "why": f"the cut reply wrote {cut['tokens']} tokens, its thinking "
                       + ("closed" if cut["reply"] else "open")
                       + f"; the whole one {whole['tokens']}"}
    t = turn(cut["think"], path)
    for name, n in (("counted", PENALTY_CUT), ("none", 0)):
        c = _greedy(srv, [*q, t], PENALTY_MORE, path, penalty_prompt_tokens=n)
        out[name] = {"penalty_prompt_tokens": n, "prompt_tokens": c["prompt"],
                     "read_whole": c["prompt"] == (cut["prompt"] or 0) + PENALTY_CUT,
                     **agree(whole["steps"][PENALTY_CUT:], c["steps"])}
    a, b = out["counted"], out["none"]
    out["verdict"] = "pass" if a["read_whole"] and a["same_tokens"] >= b["same_tokens"] else "fail"
    return out


def batching_test(srv: "Server", path: int) -> dict:
    """31.3c: the same uncut greedy reply, alone and among FILLERS others"""
    import threading as th
    from concurrent.futures import ThreadPoolExecutor
    q = [{"role": "user", "content": PENALTY_Q}]
    n = PENALTY_CUT + PENALTY_MORE
    alone = _greedy(srv, q, n, path)
    started = th.Event()

    def filler(k: int) -> None:
        started.set()
        try:
            _greedy(srv, [{"role": "user", "content": FILLER_Q.format(k=k)}], FILLER_TOKENS, path)
        except Exception:                               # noqa: BLE001 — a filler's own end
            pass
    with ThreadPoolExecutor(max_workers=FILLERS + 1) as pool:
        for k in range(FILLERS):
            pool.submit(filler, k)
        started.wait(30)
        time.sleep(FILLER_HEAD_S)
        among = pool.submit(_greedy, srv, q, n, path).result()
    return {"fillers": FILLERS, **agree(alone["steps"], among["steps"])}


def window_test(srv: "Server", path: int) -> dict:
    """which of a continued prompt's tokens the presence penalty counts, at
    penalty_prompt_tokens the turn's tokens, 0 and -1"""
    import math

    import path_check
    q = [{"role": "user", "content": PROBE_Q}]
    p0 = srv.template(q, {"expert_path": path})
    p1 = srv.template([*q, turn(PROBE_TURN, path)], {"expert_path": path})
    ids = srv.post("/tokenize", {"content": p1, "add_special": True,
                                 "parse_special": True}).get("tokens") or []
    n = len(ids) - srv.count(p0)
    if n <= 0 or p1 != p0 + PROBE_TURN:
        return {"verdict": "inconclusive", "why": "the probe's turn wasn't written after the "
                                                  "question"}
    turn_ids, q_ids = set(ids[-n:]), set(ids[:-n])
    pen = float(fb.PATHS[path].get("presence_penalty") or 0)
    temp = float(PROBE_SAMPLING["temperature"])
    out = {"turn_tokens": n, "presence_penalty": pen, "sampling": dict(PROBE_SAMPLING)}
    for name, ppt, want in (("turn", n, (pen, 0.0)), ("none", 0, (0.0, 0.0)),
                            ("all", -1, (pen, pen))):
        got = {}
        for after in (True, False):
            r = srv.post("/completion", {"prompt": p1, "n_predict": 1, "n_probs": PROBE_N,
                                         "post_sampling_probs": after, "expert_path": path,
                                         "penalty_prompt_tokens": ppt, "cache_prompt": False,
                                         "seed": 7, **PROBE_SAMPLING})
            got[after] = path_check._probs((r.get("completion_probabilities") or [{}])[0])
        both = [i for i in got[True] if i in got[False]]
        groups = {"turn": [i for i in both if i in turn_ids and i not in q_ids],
                  "question": [i for i in both if i in q_ids and i not in turn_ids],
                  "neither": [i for i in both if i not in turn_ids and i not in q_ids]}
        if not groups["neither"] or not (groups["turn"] or groups["question"]):
            out[name] = {"penalty_prompt_tokens": ppt, "verdict": "inconclusive",
                         **{f"{k}_kept": len(v) for k, v in groups.items()}}
            continue
        slope = 1 / temp
        res = {i: got[True][i] - slope * got[False][i] for i in both}
        ref = sum(res[i] for i in groups["neither"]) / len(groups["neither"])
        est = {k: (-(sum(res[i] for i in v) / len(v) - ref) / slope if v else None)
               for k, v in groups.items() if k != "neither"}
        ok = all(est[k] is None or abs(est[k] - w) <= PROBE_TOL
                 for k, w in zip(("turn", "question"), want))
        out[name] = {"penalty_prompt_tokens": ppt, "turn": est["turn"],
                     "question": est["question"], "want": list(want),
                     "verdict": "pass" if ok else "fail",
                     **{f"{k}_kept": len(v) for k, v in groups.items()}}
        if any(isinstance(x, float) and math.isnan(x) for x in est.values()):
            out[name]["verdict"] = "inconclusive"
    verdicts = [out[k]["verdict"] for k in ("turn", "none", "all")]
    out["verdict"] = ("fail" if "fail" in verdicts else
                      "pass" if out["turn"]["verdict"] == "pass"
                      and out["turn"].get("turn") is not None
                      and out["turn"].get("question") is not None else "inconclusive")
    return out


def _agree_words(x: dict) -> str:
    def v(y):
        return "—" if y is None else f"{y:.3f}"
    line = (f"the same for {x.get('same_tokens')} of {x.get('of')} tokens; where they agree "
            f"their log probabilities are at most {v(x.get('max_diff'))} apart (median "
            f"{v(x.get('median_diff'))})")
    if x.get("parted"):
        a, b = x["parted"]["first"], x["parted"]["second"]
        line += (" — they part where the first's two likeliest were "
                 + ", ".join(f"{tok!r} {lp:.3f}" for tok, lp in a)
                 + " and the second's " + ", ".join(f"{tok!r} {lp:.3f}" for tok, lp in b))
    return line


def penalty_words(t: dict) -> list[str]:
    out = []
    g = t.get("greedy") or {}
    if g.get("why"):
        out.append(f"greedy, one reply against the same cut and continued: {g['verdict']} — "
                   f"{g['why']}")
    else:
        for name, words in (("counted", f"penalty_prompt_tokens {g.get('cut')} (the cut reply's)"),
                            ("none", "penalty_prompt_tokens 0")):
            out.append(f"greedy, one reply against it cut and continued with {words}: "
                       + _agree_words(g.get(name) or {}))
    b = t.get("batching") or {}
    if b:
        out.append(f"greedy, the same uncut reply alone and among {b.get('fillers')} others "
                   "(batching alone): " + _agree_words(b))
    w = t.get("window") or {}
    if w.get("why"):
        out.append(f"the penalty's window: {w['verdict']} — {w['why']}")
    for name, words in (("turn", f"the turn's {w.get('turn_tokens')} tokens"), ("none", "0"),
                        ("all", "-1")):
        x = w.get(name)
        if not isinstance(x, dict):
            continue
        kept = (f"{x.get('turn_kept', 0)} of the turn's, {x.get('question_kept', 0)} of the "
                f"question's, {x.get('neither_kept', 0)} of neither")
        if "want" not in x:
            out.append(f"the penalty's window at penalty_prompt_tokens {words}: inconclusive "
                       f"(tokens kept: {kept})")
            continue

        def v(y):
            return "—" if y is None else f"{y:.2f}"
        out.append(f"the penalty's window at penalty_prompt_tokens {words}: the turn's words "
                   f"down {v(x['turn'])}, the question's {v(x['question'])} in logits (want "
                   f"{x['want'][0]:.2f} and {x['want'][1]:.2f}; tokens kept: {kept}): "
                   f"{x['verdict']}")
    out.append(f"penalty_prompt_tokens' window: {t.get('verdict')}")
    return out


def check_words(r: dict) -> str:
    """one check line — no question text"""
    if r["status"] != "continued":
        return f"question {r['id']}, run {r['epoch']}: not continued — {r['why']}"
    s = r.get("settings") or {}
    return (f"question {r['id']}, run {r['epoch']}: {r['was']['tokens']:,} → {r['tokens']:,} "
            f"tokens ({r['new_tokens']:,} new) · " + ("finished" if not r["ran_out"] else
                                                     f"ran out at {r['limit']:,}")
            + f" · {r['form']} · one reply: the prompt read {r['tokens_evaluated']:,} tokens "
            f"(question {r['prompt_tokens']:,} + reply {r['was']['tokens']:,}), its thinking "
            "and reply begin with the cut-off ones, no thinking opened again, no question "
            "written again" + retoken_words(r) + speed_words(r) + " · applied: "
            + ", ".join(f"{k} {s[k]}" for k in SETTINGS if k in s))


def speed_words(r: dict) -> str:
    """31.3c: how fast the box read the prompt again and wrote the rest — what
    U-cont's estimate guessed"""
    t = r.get("timings") or {}
    out = []
    for n, ms, what in (("prompt_n", "prompt_ms", "read"), ("predicted_n", "predicted_ms",
                                                            "wrote")):
        if isinstance(t.get(n), (int, float)) and isinstance(t.get(ms), (int, float)) and t[ms] > 0:
            out.append(f"{what} {t[n]:,.0f} tokens at {t[n] / t[ms] * 1000:,.0f} a second")
        else:
            out.append("")
    got = " and ".join(x for x in out if x)
    return f" · {got}" if got else ""


def left_by_rule(r: dict) -> bool:
    """31.3c: an answer the brief's rule leaves at its old limit — its text
    doesn't give back its tokens — said, never a failure of the continuation"""
    return r.get("status") == "refused" and "doesn't give back" in str(r.get("why") or "")


# ---------------------------------------------------------------------------
# the server: what to continue, and the import
# ---------------------------------------------------------------------------

def inputs(dest: Path, say=print, models: list[str] | None = None) -> dict:
    """a file of answers that ran out of room for each model and thinking
    continued (SPECS: step2388's k6 and k3, the Q2's easy set) — those of
    every benchmark it continues whose answers are all in on the row, and not
    continued yet. {key: {file, sha256, by task, left out}}"""
    from service import config
    from service import frontier as sf
    fb.set_root(config.BENCH_ROOT)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for model, on in SPECS:
        if models and model not in models:
            continue
        spec = spec_of(model, on)
        row = config.OUT_DIR / (model.replace("/", "__") + ("__thinking" if on else ""))
        lines, by, left = [], {}, {}
        for t in spec["tasks"]:
            if t == SHOWCASE:
                got = showcase_inputs(row.name)          # 31.3e
                if got is not None:
                    lines += got[0]
                    by[t] = got[1]
                continue
            d = sf.task_dir(row, t)
            if not (d / sf.ANSWERS).exists():
                continue
            m = sf.marks(row, t)
            if m["missing"]:
                left[t] = f"{len(m['missing']):,} answers not in yet"
                continue
            try:
                budget = json.loads((d / sf.SETUP).read_text(encoding="utf-8")).get("budget")
            except (OSError, ValueError):
                budget = None
            got = sf.read_answers(d / sf.ANSWERS)
            n = {"ran_out": 0, "to_continue": 0, "continued_already": 0, "no_count": 0,
                 "answers": len(got), "budget": budget}
            for (qid, e), a in sorted(got.items()):
                if a.get("unanswered") or not fb.ran_out(a.get("answer") or "", a.get("finish")):
                    continue
                n["ran_out"] += 1
                if a.get("continued"):
                    n["continued_already"] += 1
                    continue
                if not isinstance(a.get("tokens"), int):
                    n["no_count"] += 1
                    continue
                n["to_continue"] += 1
                lines.append({"task": t, "id": qid, "epoch": e, "seed": a.get("seed"),
                              "answer": a.get("answer") or "", "finish": a.get("finish"),
                              "tokens": a["tokens"],
                              **({"prompt_tokens": a["prompt_tokens"]}
                                 if isinstance(a.get("prompt_tokens"), int) else {}),
                              "budget": budget, "row": row.name})
            by[t] = n
        f = dest / INPUT.format(key=spec["key"])
        f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                     encoding="utf-8")
        sha = rb.sha256_file(f)
        key = spec["key"]
        out[key] = {"file": str(f), "sha256": sha, "by_task": by, "left_out": left,
                    "answers": len(lines),
                    "most": sum(spec["limit"] - x["tokens"] for x in lines)}
        say(f"{model} · thinking {'on' if on else 'off'}"
            + (f" (k{spec['path']})" if spec["path"] else "") + f", to {spec['limit']:,} tokens: "
            f"{len(lines):,} answers to continue · {f} · sha256 {sha}")
        for t, n in by.items():
            say(f"  {label_of(t)}: {n['to_continue']:,} of its {n['answers']:,} answers "
                f"(ran out at {n['budget']:,}: {n['ran_out']:,}"
                + (f", continued already {n['continued_already']:,}" if n["continued_already"]
                   else "")
                + (f", no token count {n['no_count']:,}" if n["no_count"] else "") + ")"
                if isinstance(n["budget"], int) else
                f"  {label_of(t)}: {n['to_continue']:,} of its {n['answers']:,} answers")
        for t, why in left.items():
            say(f"  {label_of(t)}: left out — {why}")
        say(f"  at most {out[key]['most']:,} tokens to write (each to the limit)")
    return out


def showcase_answer(r: dict) -> str:
    """a showcase answer as service/served.ask keeps one (showcase_box.ask)"""
    think, reply = r.get("thinking") or "", r.get("reply") or ""
    return f"<think>\n{think}\n</think>\n\n{reply}" if think else reply


def showcase_inputs(row: str) -> tuple[list[dict], dict] | None:
    """31.3e: the showcase's answers that ran out of room, as the inputs file
    holds an answer — None when its answers aren't imported here"""
    import showcase_box as scb
    f = scb.where(row) / "own.jsonl"
    try:
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    lines = []
    n = {"ran_out": 0, "to_continue": 0, "continued_already": 0, "no_count": 0,
         "answers": len(rows), "budget": None}
    for r in rows:
        ans = showcase_answer(r)
        if not fb.ran_out(ans, r.get("finish")):
            continue
        n["ran_out"] += 1
        n["budget"] = r.get("max_tokens")
        if r.get("continued"):
            n["continued_already"] += 1
            continue
        if not isinstance(r.get("output_tokens"), int):
            n["no_count"] += 1
            continue
        n["to_continue"] += 1
        lines.append({"task": SHOWCASE, "id": str(r.get("q")), "epoch": 0, "seed": r.get("seed"),
                      "answer": ans, "finish": r.get("finish"), "tokens": r["output_tokens"],
                      **({"prompt_tokens": r["prompt_tokens"]}
                         if isinstance(r.get("prompt_tokens"), int) else {}),
                      "budget": r.get("max_tokens"), "row": row})
    return lines, n


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "",
                  register: str = "", public_weights: bool = False) -> int:
    """a box's continued answers, onto the row they came from"""
    import shutil

    import import_frontier as imf
    from service import config, db, served
    from service import frontier as sf
    bad = imf.bundle_problems(b.get("bundle")) or imf.setup_problems(b.get("setup"))
    if bad:
        for line in bad:
            say(f"refused — {line}")
        return REFUSED
    if register or public_weights:
        say("refused — a continuation goes onto a row the model has: import without --register")
        return REFUSED
    bundle, setup = b["bundle"], b["setup"]
    model, on, row_name = bundle.get("model") or "", bool(bundle.get("thinking")), \
        bundle.get("row") or ""
    cont = setup.get(SETUP_KEY) if isinstance(setup.get(SETUP_KEY), dict) else {}
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · thinking "
        f"{'on' if on else 'off'} · continued to {cont.get('limit')} tokens · "
        f"{', '.join(bundle.get('tasks') or {}) or 'no task'}")
    if bundle.get("format") != rb.SUITES[SUITE]["format"] or not cont:
        say(f"refused — its format: the bundle's {bundle.get('format')}, this board reads "
            f"{rb.SUITES[SUITE]['format']} for a continuation")
        return REFUSED
    if not row_name or "/" in row_name or row_name.startswith("."):
        say(f"refused — its row {row_name!r} isn't a row's name")
        return REFUSED
    row = config.OUT_DIR / row_name
    reg = imf.registry(row)
    was = next((x for x in reg.get("continued") or [] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, as Runs #{was['sid']} on {was['at']}: nothing changed")
        return 0
    rec = served.get(model) if served.is_served(model) else None
    out, stop = imf.model_problems(b, rec, file_sha)
    spec = spec_of(model, on)
    if not stop and spec is None:
        out.append(f"{model}'s answers with thinking {'on' if on else 'off'} aren't continued "
                   "on this board")
    elif not stop and (cont.get("limit") != spec["limit"] or cont.get("path") != spec["path"]):
        out.append(f"continued to {cont.get('limit')} tokens"
                   + (f" on k{cont.get('path')}" if cont.get("path") else "")
                   + f"; {model}'s limit here is {spec['limit']}"
                   + (f" on k{spec['path']}" if spec["path"] else ""))
    if stop or out:
        for line in out:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    fb.set_root(config.BENCH_ROOT)
    tasks = [t for t in fb.ALL if t in (bundle.get("tasks") or {})]
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    staging = config.OUT_DIR.with_name("staging") / f"{row_name}-cont-{stamp}" / row_name
    lines, ready, bad = [], {}, []
    for t in tasks:
        if t not in spec["tasks"]:
            bad.append(f"{t}: not a benchmark {model}'s answers are continued on")
            continue
        raw = b["files"].get(f"results/{row_name}/{t}_0shot/{sf.SUB}/{DONE}")
        try:
            got = [json.loads(x) for x in (raw or b"").decode("utf-8").splitlines() if x.strip()]
        except ValueError:
            bad.append(f"{t}: its {DONE} isn't JSON lines")
            continue
        d = sf.task_dir(row, t)
        m = sf.marks(row, t) if (d / sf.ANSWERS).exists() else {"missing": [None]}
        if m["missing"]:
            bad.append(f"{t}: the row's answers aren't all in — continued answers go onto a "
                       "whole benchmark")
            continue
        have = sf.read_answers(d / sf.ANSWERS)
        new = dict(have)
        n_new, refused, reread = 0, [], 0
        for r in got:
            k = (str(r.get("id")), int(r.get("epoch", -1)))
            a = have.get(k)
            if a is None:
                bad.append(f"{t}: question {k[0]}, run {k[1]} isn't on the row")
                continue
            w = r.get("was") or {}
            if a.get("continued"):
                if answer_sha(a.get("answer") or "") == answer_sha(r.get("answer") or "") \
                        and r.get("status") == "continued":
                    continue                    # the same continuation, here already
                bad.append(f"{t}: question {k[0]}, run {k[1]} was continued already, by another "
                           "bundle")
                continue
            if answer_sha(a.get("answer") or "") != w.get("sha256") or a.get("tokens") != \
                    w.get("tokens"):
                bad.append(f"{t}: question {k[0]}, run {k[1]}: the row's answer isn't the one "
                           "continued")
                continue
            if r.get("status") != "continued":
                refused.append(r)
                continue
            fin = r.get("answer") or ""
            if not isinstance(r.get("tokens"), int) or continuity(
                    a.get("answer") or "", fin, str(r.get("form") or ""), "", spec["path"]):
                bad.append(f"{t}: question {k[0]}, run {k[1]}: its continued answer doesn't "
                           "begin with the row's")
                continue
            new[k] = {**a, "answer": fin, "finish": r.get("finish"), "tokens": r["tokens"],
                      "continued": {"from": {"tokens": a.get("tokens"), "finish": a.get("finish"),
                                             "budget": w.get("budget"),
                                             "answer_sha256": w.get("sha256")},
                                    "limit": r.get("limit"), "seed": r.get("seed"),
                                    "max_tokens": r.get("max_tokens"),
                                    "penalty_prompt_tokens": r.get("penalty_prompt_tokens"),
                                    "tokens_evaluated": r.get("tokens_evaluated"),
                                    "new_tokens": r.get("new_tokens"), "form": r.get("form"),
                                    **({"retokenized": r["retokenized"]}
                                       if r.get("retokenized") else {}),
                                    "settings": r.get("settings") or {},
                                    "box": setup.get("box") or "", "bundle": path.name,
                                    **({"timings": r["timings"]} if r.get("timings") else {})}}
            n_new += 1
            reread += bool(r.get("retokenized"))
        if not n_new and not refused:
            lines.append(f"{t}: its continued answers are on the row already — nothing changed")
            continue
        try:
            tsetup = json.loads((d / sf.SETUP).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tsetup = {}
        prev = tsetup.get("continued") if isinstance(tsetup.get("continued"), dict) else {}
        tsetup["continued"] = {
            "from_budget": prev.get("from_budget") or tsetup.get("budget"),
            "limit": cont["limit"],
            "answers": int(prev.get("answers") or 0) + n_new,
            # 31.3d: of them, those whose text read back a few tokens otherwise
            "retokenized": int(prev.get("retokenized") or 0) + reread,
            "not_continued": int(prev.get("not_continued") or 0) + len(refused),
            "not_continued_why": _why_counts(refused, prev.get("not_continued_why")),
            "bundles": [*(prev.get("bundles") or []), {"bundle": path.name,
                                                      "sha256": b["sha256"], "at": stamp,
                                                      "box": setup.get("box") or ""}]}
        tsetup["budget"] = cont["limit"]
        sd = sf.task_dir(staging, t)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / sf.ANSWERS).write_text("".join(json.dumps(new[k], ensure_ascii=False) + "\n"
                                             for k in sorted(new)), encoding="utf-8")
        (sd / sf.SETUP).write_text(json.dumps(tsetup, indent=1), encoding="utf-8")
        imf.keep_grades(d, sd, new)
        ready[t] = {"continued": n_new, "refused": refused, "retokenized": reread}
    # 31.3e: the showcase's continued answers, onto its own rows (own.jsonl)
    show = (showcase_staged(b, row_name, spec, setup, path, bad)
            if SHOWCASE in (bundle.get("tasks") or {}) else None)
    if bad:
        shutil.rmtree(staging.parent, ignore_errors=True)
        for line in bad[:20]:
            say(f"refused — {line}")
        if len(bad) > 20:
            say(f"refused — and {len(bad) - 20} more")
        say("nothing was imported")
        return REFUSED
    if not ready and not (show and (show["n"] or show["refused"])):
        shutil.rmtree(staging.parent, ignore_errors=True)
        for x in lines:
            say(x)
        say("imported already: nothing changed")
        return 0
    status, error, words, scored = "done", "", [], {}
    for t in ready:
        try:
            sc = sf.score_task(staging, t, rec)
        except Exception as e:                          # noqa: BLE001 — the row is untouched
            status, error = "failed", f"scoring {t}: {e}"
            words.append(f"{fb.BENCH[t]['label']}: not imported — scoring it failed ({e})")
            continue
        if sc and sc.get("refused"):
            status, error = "failed", sc["refused"]
            words.append(f"{sc['refused']} — not imported")
            continue
        scored[t] = sc
    if status == "done":
        for t, x in ready.items():
            before = sf.presented(sf.task_dir(row, t), t)
            kept_at = sf.set_aside(row, t, "before-continuation")
            dest = sf.task_dir(row, t)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sf.task_dir(staging, t)), str(dest))
            sc = scored.get(t) or {}
            lim = sc.get("limits") or {}
            lines.append(limits_line(t, lim, x["continued"], len(x["refused"]), before,
                                     x["retokenized"])
                         + (f" · the answers before are kept at {kept_at}" if kept_at else ""))
            for r in x["refused"][:5]:
                lines.append(f"{t}: question {r.get('id')}, run {r.get('epoch')} not continued — "
                             f"{str(r.get('why'))[:200]}")
            if sc and sc.get("score") is not None:
                words.append(sf.words(t, sc))
            elif sc and sc.get("waiting"):
                words.append(f"{fb.BENCH[t]['label']}: {sc['waiting']:,} answers wait for a "
                             "grade (graded once, on the final answers)")
        if show and (show["n"] or show["refused"]):
            lines.append(showcase_written(show, stamp))
            words.append(f"the showcase: {show['n']:,} answers continued")
    else:
        words.append("nothing was imported; the rows are as they were, and the bundle can be "
                     "imported again")
    shutil.rmtree(staging.parent, ignore_errors=True)
    line = " · ".join(words) or "imported"
    box = setup.get("box") or ""
    gpu = (setup.get("gpu") or {}).get("name") or "a GPU"
    sid = None
    if tasks:
        # a Runs row for the benchmarks' answers — the showcase is no row of the
        # board's, as its own import makes none
        sid = db.add(model, "instruct", "frontier", by,
                     f"continued to {cont['limit']:,} tokens on a rented GPU ({gpu})"
                     + (f" · box {box}" if box else "") + f" · {path.name}", thinking=on,
                     tasks=tasks, status=status)
        db.update(sid, status=status, finished_at=time.time(), progress=line, error=error,
                  where_ran=imf.where_words(imf.gpu_names(setup), [box] if box else []))
        log = config.LOGS_DIR / f"service_{sid}_{model.replace('/', '__')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"===== [{sid}] imported {path.name} (sha256 {b['sha256'][:16]}): the "
                       f"answers that ran out of room, continued to {cont['limit']:,} tokens "
                       "=====\n" + b["log"] + "".join(f"\n[import] {x}" for x in lines)
                       + f"\n[import] {line}\n", encoding="utf-8")
    if status == "done":
        reg.setdefault("continued", []).append(
            {"sha256": b["sha256"], "sid": sid, "tasks": tasks, "by": by, "at": stamp,
             "bundle": path.name, "box": box, "limit": cont["limit"],
             "answers": sum(x["continued"] for x in ready.values()) + (show or {}).get("n", 0),
             "retokenized": sum(x["retokenized"] for x in ready.values()),
             "not_continued": sum(len(x["refused"]) for x in ready.values())})
        imf._write_registry(row, reg)
        for t in ready:
            imf._mark_where(row, t, *imf.where_and_runs(row, t))
        from service import report_stats
        report_stats.wake_quietly()
    for x in lines:
        say(x)
    say(f"the row {model}{' · thinking' if on else ''}: {line}")
    if sid is not None:
        say(f"Runs #{sid}, its log the bundle's")
    return 0 if status == "done" else 1


def showcase_staged(b: dict, row_name: str, spec: dict, setup: dict, path: Path,
                    bad: list[str]) -> dict | None:
    """31.3e: the showcase's continued answers onto its own rows — {file,
    rows, n, refused, reread, limit, from} — each held as a benchmark's is:
    on the rows, the one continued, beginning as it did; what is wrong goes
    into `bad`"""
    import showcase_box as scb
    raw = b["files"].get(f"results/{row_name}/{SHOWCASE}_0shot/frontier/{DONE}")
    try:
        got = [json.loads(x) for x in (raw or b"").decode("utf-8").splitlines() if x.strip()]
    except ValueError:
        bad.append(f"the showcase: its {DONE} isn't JSON lines")
        return None
    f = scb.where(row_name) / "own.jsonl"
    try:
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        bad.append("the showcase: its answers aren't imported here — import U39's showcase bundle "
                   "first")
        return None
    by_q = {str(r.get("q")): r for r in rows}
    n, refused, reread, frm = 0, [], 0, set()
    for r in got:
        q = str(r.get("id"))
        a = by_q.get(q)
        if a is None:
            bad.append(f"the showcase: {q} isn't among its answers here")
            continue
        w = r.get("was") or {}
        if a.get("continued"):
            if answer_sha(showcase_answer(a)) != answer_sha(r.get("answer") or "") \
                    or r.get("status") != "continued":
                bad.append(f"the showcase: {q} was continued already, by another bundle")
            continue
        if answer_sha(showcase_answer(a)) != w.get("sha256") or a.get("output_tokens") != \
                w.get("tokens"):
            bad.append(f"the showcase: {q}: its answer here isn't the one continued")
            continue
        if r.get("status") != "continued":
            refused.append(r)
            continue
        fin = r.get("answer") or ""
        if not isinstance(r.get("tokens"), int) or continuity(
                showcase_answer(a), fin, str(r.get("form") or ""), "", spec["path"]):
            bad.append(f"the showcase: {q}: its continued answer doesn't begin with the one here")
            continue
        think, reply, _ = split(fin)
        toks = r["tokens"]
        frm.add(a.get("max_tokens"))
        by_q[q] = {**a, "thinking": think, "reply": reply, "output_tokens": toks,
                   "thinking_tokens": round(toks * len(think) / max(1, len(think) + len(reply))),
                   "finish": r.get("finish"), "ran_out": r.get("ran_out"),
                   "looped": r.get("loop"),
                   "continued": {"from": {"output_tokens": a.get("output_tokens"),
                                          "finish": a.get("finish"),
                                          "max_tokens": a.get("max_tokens"),
                                          "answer_sha256": w.get("sha256")},
                                 "limit": r.get("limit"), "seed": r.get("seed"),
                                 "max_tokens": r.get("max_tokens"),
                                 "penalty_prompt_tokens": r.get("penalty_prompt_tokens"),
                                 "tokens_evaluated": r.get("tokens_evaluated"),
                                 "new_tokens": r.get("new_tokens"), "form": r.get("form"),
                                 **({"retokenized": r["retokenized"]}
                                    if r.get("retokenized") else {}),
                                 "box": setup.get("box") or "", "bundle": path.name}}
        n += 1
        reread += bool(r.get("retokenized"))
    return {"file": f, "old": rows, "rows": [by_q[str(r.get("q"))] for r in rows], "n": n,
            "refused": refused, "reread": reread, "limit": spec["limit"],
            "from": sorted(x for x in frm if isinstance(x, int)), "sha256": b["sha256"],
            "bundle": path.name}


def showcase_written(show: dict, stamp: str) -> str:
    """31.3e: the showcase's rows written with their continued answers — the
    ones before kept beside them — and what the import says of it"""
    f = show["file"]
    kept = f.with_name(f"own-before-continuation-{stamp}.jsonl")
    kept.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in show["rows"]),
                 encoding="utf-8")
    reg_f = f.with_name("imports.json")
    try:
        reg = json.loads(reg_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reg = {"imports": []}
    reg.setdefault("continued", []).append({"sha256": show["sha256"], "bundle": show["bundle"],
                                            "at": stamp, "answers": show["n"],
                                            "limit": show["limit"], "from": show["from"]})
    reg_f.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    done = [r for r in show["rows"] if isinstance(r.get("continued"), dict)
            and r["continued"].get("bundle") == show["bundle"]]
    before = sum(int((r["continued"].get("from") or {}).get("output_tokens") or 0) for r in done)
    after = sum(int(r.get("output_tokens") or 0) for r in done)
    frm = "/".join(f"{x:,}" for x in show["from"]) or "?"
    return (f"the showcase: to {show['limit']:,} tokens (from {frm}) · {show['n']:,} answers "
            f"continued" + (f", {len(show['refused']):,} not" if show["refused"] else "")
            + f" · {before:,} → {after:,} tokens · "
            f"{sum(1 for r in done if r.get('ran_out')):,} still ran out · "
            f"{sum(1 for r in done if r.get('looped')):,} ended in a loop · "
            f"{show['reread']:,} of them re-read a few tokens differently (at most {RETOKEN_CAP})"
            f" · its answers before are kept at {kept}")


def _why_counts(refused: list[dict], prev) -> dict:
    out = dict(prev) if isinstance(prev, dict) else {}
    for r in refused:
        w = str(r.get("why") or "")[:120]
        out[w] = out.get(w, 0) + 1
    return out


def limits_line(t: str, lim: dict, n: int, refused: int, before: dict | None,
                reread: int = 0) -> str:
    """what the import says of a benchmark: its new limit, the answers
    continued, their tokens, the still cut and the loops, and its score at
    either limit"""
    label = fb.BENCH[t]["label"]
    old, new = lim.get("at_old_limit") or {}, lim.get("at_new_limit") or {}

    def sc(x: dict) -> str:
        return ("waits for grading" if x.get("score") is None else
                f"{100 * x['score']:.1f} ± {100 * (x.get('se') or 0):.1f}")
    return (f"{label}: to {lim.get('limit') or 0:,} tokens (from {lim.get('from') or 0:,}) · "
            f"{n:,} answers continued" + (f", {refused:,} not" if refused else "")
            # 31.3d: those whose text read back a few tokens otherwise, every time said
            + f" · {reread:,} of them re-read a few tokens differently (at most {RETOKEN_CAP})"
            + f" · {lim.get('tokens_before') or 0:,} → {lim.get('tokens_after') or 0:,} tokens · "
            f"{lim.get('still_ran_out') or 0:,} still ran out · {lim.get('loops') or 0:,} ended "
            f"in a loop · at {lim.get('from') or 0:,}: {sc(old)} · at {lim.get('limit') or 0:,}: "
            f"{sc(new)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inputs", help="the answers that ran out of room, a file for each model "
                                      "and thinking continued")
    i.add_argument("--as", dest="served_as", action="append", default=[],
                   help="only this model's (repeatable; default every one SPECS names)")
    i.add_argument("--out", default="", help="where the files go (default $BENCH_ROOT/u-cont)")
    a = ap.parse_args(argv)
    from service import config
    inputs(Path(a.out) if a.out else config.BENCH_ROOT / "u-cont", models=a.served_as or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
