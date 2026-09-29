"""12q: the devicemark suite on one of our served setups — asked over its
llama-server directly, so each reply's usage, timings and finish are kept
(lm_eval keeps none of them): the battery, the 30-item pilot (with a check
that the cap counts the thinking), the MTP parity check and the speed test.

A Hugging Face model sits the battery through lm_eval instead (the runner's
task loop, three tasks devicemark.build_tasks writes), and is scored here.

What a run writes, beside the model's other results (OUT_DIR/<model>):
  devicemark_answers.jsonl   every reply as it lands — a stopped run's next
                             run asks only what isn't answered on the same
                             server, settings and battery
  devicemark_items.jsonl     each answer scored (the question browser's)
  devicemark.json            the row
  devicemark_pilot.json      the pilot's numbers, its answers and its cap check
  devicemark_parity.json     the parity check, on both setups' folders
  devicemark_speed.json      the speed test (the setup's, not a thinking row's)
  devicemark_device.json     a speed measured on a device, entered by a person
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, db
from . import served as _served

CANCELED = -15
# the speed test: a fixed prompt of 128 tokens, 256 decoded greedily past any
# end, one warm-up and two timed trials — as DeviceMark's PipelinedBench
SPEED_PROMPT_TOKENS, SPEED_DECODE_TOKENS, SPEED_TRIALS = 128, 256, 2
SPEED_TEXT = ("The history of mathematics spans thousands of years and many cultures. Early "
              "counting systems used tally marks, and later civilisations developed written "
              "numerals, arithmetic, geometry for measuring land, and astronomy for keeping "
              "calendars. Algebra grew from solving practical problems about inheritance, "
              "trade and construction, and calculus arose from questions about motion and "
              "change. ")
# the pilot's check that the cap counts the thinking: one MATH item asked
# thinking on, capped at this many tokens
CAP_CHECK_TOKENS = 64
ANSWERS_NAME = "devicemark_answers.jsonl"
PILOT_ANSWERS_NAME = "devicemark_pilot_answers.jsonl"


def dm():
    """scripts/devicemark.py: the battery, its prompts and its scorers"""
    here = str(Path(__file__).resolve().parent.parent / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)
    import devicemark
    return devicemark


def settings(thinking: bool, max_tokens: int | None = None) -> dict:
    """the protocol, as a request says it: greedy with a fixed seed, the cap
    (thinking included — the server counts every token it generates), and
    the thinking switch said out loud either way"""
    d = dm()
    return {"max_tokens": int(max_tokens or d.CAP), "temperature": 0.0, "seed": d.SEED,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}


class _Retry(Exception):
    pass


def ask(rec: dict, text: str, s: dict) -> dict:
    """one item, one reply: the text (thinking the server split off put back
    in its tags), what it generated — thinking and answer apart where the
    server says both — whether the cap cut it, its decode speed and its wall
    time"""
    body = {"model": rec["pin"].get("model") or rec["name"],
            "messages": [{"role": "user", "content": text}], "stream": False, **s}
    base, key = _served._endpoint(rec)
    t0 = time.time()
    try:
        st, raw = _served._http("POST", base + "/chat/completions", key, body,
                                timeout=config.SERVED_TIMEOUT_S)
    except Exception as e:                                  # noqa: BLE001 — no answer: retried
        raise _Retry(str(getattr(e, "reason", e))) from None
    wall = time.time() - t0
    if st in (408, 429) or st >= 500:
        raise _Retry(f"HTTP {st}")
    if st != 200:
        raise ValueError(f"HTTP {st}: {raw[:160].decode('utf-8', 'replace')}")
    reply = json.loads(raw)
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not think and "<think>" in content:
        think, content = dm().split_thinking(content)
    usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
    t = reply.get("timings") if isinstance(reply.get("timings"), dict) else {}
    gen = _int(usage.get("completion_tokens"))
    gen = gen if gen is not None else _int(t.get("predicted_n"))
    # llama-server's usage has no split of the thinking from the answer
    # (checked on the phone build, 29 Sep): the thinking is counted with its
    # own /tokenize, and the answer is the rest — the think tags with it
    thought = _int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    counted = "usage" if thought is not None else None
    if thought is None and think:
        thought = count_tokens(rec, think)
        counted = "/tokenize" if thought is not None else None
    finish = choice.get("finish_reason")
    out = {"text": f"<think>\n{think}\n</think>\n\n{content}" if think else content,
           "answer": content.strip(), "thinking_chars": len(think),
           "think_tokens_from": counted,
           "prompt_tokens": _int(usage.get("prompt_tokens"))
           if usage.get("prompt_tokens") is not None else _int(t.get("prompt_n")),
           "gen_tokens": gen, "think_tokens": thought,
           "answer_tokens": gen - thought if gen is not None and thought is not None else None,
           "finish": finish,
           "capped": finish == "length" or bool(gen is not None and gen >= s["max_tokens"]),
           "decode_tok_s": _float(t.get("predicted_per_second")),
           "wall_s": round(wall, 3)}
    if _int(t.get("draft_n")):
        out["draft"] = {"n": _int(t["draft_n"]), "accepted": _int(t.get("draft_n_accepted")) or 0}
    return out


def count_tokens(rec: dict, text: str) -> int | None:
    """how many tokens the server's own tokenizer makes of `text` — None when
    it can't say (the characters are kept either way, labelled as such)"""
    try:
        st, raw = _served._http("POST", _served.root_of(rec["base_url"]) + "/tokenize",
                                rec.get("key", ""), {"content": text, "add_special": False},
                                timeout=30)
        return len(json.loads(raw).get("tokens") or []) if st == 200 else None
    except Exception:                                       # noqa: BLE001 — characters instead
        return None


def _int(v) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None


def _float(v) -> float | None:
    return round(float(v), 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _ask_patiently(rec: dict, text: str, s: dict) -> dict:
    """a server that doesn't answer is asked again for SERVED_RETRY_S"""
    t0 = time.time()
    while True:
        try:
            return ask(rec, text, s)
        except _Retry as e:
            if time.time() - t0 >= config.SERVED_RETRY_S:
                raise _served.ServerStopped(0, 0, str(e)) from None
            time.sleep(min(5.0, max(0.05, config.SERVED_RETRY_S / 24)))


def fingerprint(rec: dict, s: dict) -> str:
    """what an answer is an answer to: the battery, the file served and the
    settings — an answer kept from a stopped run counts only under the same"""
    d = dm()
    pin = {k: rec["pin"].get(k) for k in _served.PINNED}
    return hashlib.sha256(json.dumps([d.VERSION, pin, s, d.prompts()], sort_keys=True)
                          .encode("utf-8")).hexdigest()[:16]


def answer_all(rec: dict, keys: list[tuple[str, str]], items: dict, s: dict, store: Path,
               on_progress=None, canceled=lambda: False,
               also: tuple[Path, ...] = ()) -> tuple[list[dict], Exception | None]:
    """every item asked that isn't answered in `store` — or `also`, the
    pilot's — under these settings, each reply appended as it lands. (the
    answers, in `keys`' order, and why it stopped — None when every one is in)"""
    fp = fingerprint(rec, s)
    have = {}
    for f in (*also, store):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("fp") == fp and (r["bench"], r["key"]) in set(keys):
                    have[(r["bench"], r["key"])] = r
    todo = [k for k in keys if k not in have]
    lock, halt = threading.Lock(), []
    total, t0, done = len(keys), time.time(), [len(keys) - len(todo)]
    store.parent.mkdir(parents=True, exist_ok=True)

    def one(k: tuple[str, str]) -> None:
        if halt or canceled():
            return
        try:
            got = _ask_patiently(rec, items[k]["text"], s)
        except (_served.ServerStopped, ValueError) as e:
            with lock:
                halt.append(e)
            return
        r = {"bench": k[0], "key": k[1], "fp": fp, "at": time.time(), **got}
        with lock:
            with open(store, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            have[k] = r
            done[0] += 1
            n = done[0]
        if on_progress:
            asked = n - (total - len(todo))
            on_progress(n, total, (time.time() - t0) / max(1, asked))

    with ThreadPoolExecutor(max_workers=_served.concurrency(rec)) as pool:
        list(pool.map(one, todo))
    return [have[k] for k in keys if k in have], (halt[0] if halt else None)


# ---------------------------------------------------------------------------
# the setup a row ran on
# ---------------------------------------------------------------------------

_QUANT = re.compile(r"(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|q\d_\w+)", re.I)


def setup_of(rec: dict, thinking: bool, props: dict | None = None) -> dict:
    """model, lookahead, MTP, thinking, quant, the server's flags as far as
    it or its registration says them, port and battery"""
    d = dm()
    name = f"{rec['name']} {rec.get('how') or ''}"
    file = rec["pin"].get("file") or ""
    m = re.search(r":(\d+)(?:/|$)", rec["base_url"])
    q = _QUANT.findall(file.rsplit(".", 1)[0])
    return {"model": rec["id"], "name": rec["name"], "runtime": "llama-server",
            "base_url": rec["base_url"], "port": int(m.group(1)) if m else None,
            "file": file, "build": rec["pin"].get("build"), "ctx": rec["pin"].get("ctx"),
            "quant": q[-1] if q else None, "phone": _served.is_phone(rec),
            "lookahead": bool(re.search(r"look-?ahead", name, re.I)),
            "mtp": bool(re.search(r"\bmtp\b|draft|--spec|speculative", name, re.I)),
            "thinking": bool(thinking), "server_flags": rec.get("how") or "",
            "props": props or {}, "battery": d.VERSION, "cap": d.CAP, "seed": d.SEED}


def _props(rec: dict) -> dict:
    """what llama-server says of itself that a row keeps"""
    try:
        st, raw = _served._http("GET", _served.root_of(rec["base_url"]) + "/props",
                                rec.get("key", ""), timeout=5)
        p = json.loads(raw) if st == 200 else {}
    except Exception:                                       # noqa: BLE001 — optional
        return {}
    g = p.get("default_generation_settings") or {}
    return {k: v for k, v in {"build": p.get("build_info"), "n_ctx": g.get("n_ctx"),
                              "total_slots": p.get("total_slots"),
                              "speculative": (g.get("params") or {}).get("speculative.n_max")
                              }.items() if v is not None}


def row_dir(model_id: str, thinking: bool) -> Path:
    return config.OUT_DIR / (model_id.replace("/", "__") + ("__thinking" if thinking else ""))


# ---------------------------------------------------------------------------
# the four parts
# ---------------------------------------------------------------------------

def run(sid: int, sub: dict, rec: dict, log_path: Path) -> tuple[str, str]:
    """a served devicemark run, start to end: (status, the row's line)"""
    part = sub.get("part") or "full"
    thinking = bool(sub.get("thinking"))
    if part == "speed":
        return speed(sid, rec, log_path)
    d = dm()
    db.update(sid, status="running", progress="devicemark · reading the battery's questions")
    items = d.load_items(config.DM_ITEMS)
    if part == "parity":
        pair = _served.get(sub.get("pair") or "")
        if not pair:
            return "failed", f"parity: {sub.get('pair')} is not registered"
        return parity(sid, rec, pair, items, thinking, log_path)
    return battery(sid, rec, items, thinking, part, log_path)


def _progress(sid: int, label: str):
    from .runner import time_left

    def on(done: int, total: int, each: float) -> None:
        db.update(sid, status="running", progress=f"{label} · {time_left(done, total, each)}")
    return on


def _log(log_path: Path, line: str) -> None:
    with open(log_path, "a") as lf:
        lf.write(line + "\n")


def battery(sid: int, rec: dict, items: dict, thinking: bool, part: str,
            log_path: Path) -> tuple[str, str]:
    """the battery, or the pilot's 30 of it: asked, then scored"""
    d = dm()
    keys = d.keys_for(part)
    s = settings(thinking)
    out = row_dir(rec["id"], thinking)
    store = out / (ANSWERS_NAME if part == "full" else PILOT_ANSWERS_NAME)
    label = f"devicemark {'pilot' if part == 'pilot' else 'battery'} ({len(keys)} items)"
    _log(log_path, f"\n===== [{sid}] devicemark {part} · served: {rec['name']} at "
                   f"{rec['base_url']} · thinking {'on' if thinking else 'off'} · "
                   f"{json.dumps(s)} · {d.VERSION} =====")
    got, stopped = answer_all(rec, keys, items, s, store, _progress(sid, label),
                              lambda: db.cancel_requested(sid),
                              also=(out / PILOT_ANSWERS_NAME,) if part == "full" else ())
    if db.cancel_requested(sid):
        return "canceled", f"canceled: the {len(got)} answered are kept for the next run"
    if stopped:
        _log(log_path, f"[devicemark] stopped after {len(got)} of {len(keys)}: {stopped}")
        return "failed", (f"{stopped} · the {len(got)} of {len(keys)} answered are kept: the "
                          f"next run asks only the rest")
    db.update(sid, status="running", progress="devicemark · scoring the answers")
    setup = setup_of(rec, thinking, _props(rec))
    row = d.mark(out, items, setup, got, part)
    if part == "pilot":
        row["cap_check"] = cap_check(rec, items, d.keys_for("pilot"))
        (out / d.PILOT_NAME).write_text(json.dumps({**json.loads(
            (out / d.PILOT_NAME).read_text(encoding="utf-8")), "cap_check": row["cap_check"]},
            indent=1, ensure_ascii=False), encoding="utf-8")
    line = summary_line(row)
    _log(log_path, f"\n===== [{sid}] devicemark {part}: {line} =====")
    return "done", line


def summary_line(row: dict) -> str:
    d = dm()
    c = row["composite"]
    bits = [f"{d.LABEL[b]} {row['benches'][b]['acc']:.3f}" for b in d.BENCHES
            if row["benches"][b]["n"]]
    head = (f"composite {c['value']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] · "
            if c.get("value") is not None else "")
    cc = row.get("cap_check")
    return (head + " · ".join(bits) + f" · answered {row['answered_pct']:.0%} · median "
            f"{row['median_tokens']} tokens" + (f" · cap check: {cc['line']}" if cc else ""))


def cap_check(rec: dict, items: dict, pilot_keys: list[tuple[str, str]]) -> dict:
    """does the cap count the thinking? One MATH item, thinking on, capped at
    CAP_CHECK_TOKENS: the server should stop at that many tokens with the
    thinking unfinished — every one of them thinking"""
    k = next(k for k in pilot_keys if k[0] == "math")
    try:
        r = _ask_patiently(rec, items[k]["text"], settings(True, CAP_CHECK_TOKENS))
    except (_served.ServerStopped, ValueError) as e:
        return {"ok": None, "line": f"not checked: {e}"}
    within = r["gen_tokens"] is not None and r["gen_tokens"] <= CAP_CHECK_TOKENS
    ok = within and r["capped"] and r["thinking_chars"] > 0
    return {"asked": CAP_CHECK_TOKENS, "item": f"{k[0]}:{k[1]}", "gen_tokens": r["gen_tokens"],
            "think_tokens": r["think_tokens"], "think_tokens_from": r["think_tokens_from"],
            "answer_tokens": r["answer_tokens"], "thinking_chars": r["thinking_chars"],
            "answer_chars": len(r["answer"]), "finish": r["finish"], "ok": ok,
            "line": (f"thinking on, max_tokens {CAP_CHECK_TOKENS}: stopped at {r['gen_tokens']} "
                     f"tokens ({r['finish']}), all of it thinking — the cap counts the thinking"
                     if ok else
                     f"thinking on, max_tokens {CAP_CHECK_TOKENS}: {r['gen_tokens']} tokens, "
                     f"finish {r['finish']}, {r['thinking_chars']} characters of thinking — "
                     f"NOT the expected stop: look before trusting the cap")}


def parity(sid: int, mtp: dict, plain: dict, items: dict, thinking: bool,
           log_path: Path) -> tuple[str, str]:
    """the parity check's 50 items on the setup with MTP and the one without,
    greedy both ways: the same outputs, token for token, and the same answers"""
    d = dm()
    keys = d.keys_for("parity")
    s = settings(thinking)
    _log(log_path, f"\n===== [{sid}] devicemark parity · {mtp['id']} (MTP) against "
                   f"{plain['id']} · {len(keys)} items · thinking {'on' if thinking else 'off'} "
                   f"=====")
    got = {}
    for tag, rec in (("mtp", mtp), ("plain", plain)):
        store = row_dir(rec["id"], thinking) / f"devicemark_parity_{tag}_answers.jsonl"
        got[tag], stopped = answer_all(rec, keys, items, s, store,
                                       _progress(sid, f"devicemark parity · {rec['name']}"),
                                       lambda: db.cancel_requested(sid))
        if db.cancel_requested(sid):
            return "canceled", "canceled: the answers so far are kept for the next run"
        if stopped:
            return "failed", f"{rec['name']}: {stopped} · the answers so far are kept"
    db.update(sid, status="running", progress="devicemark parity · comparing")
    rep = d.parity(d.score_items(got["mtp"], items), d.score_items(got["plain"], items))
    rep.update(mtp=mtp["id"], plain=plain["id"], thinking=thinking, version=d.VERSION,
               at=time.time())
    for rec in (mtp, plain):
        out = row_dir(rec["id"], thinking)
        out.mkdir(parents=True, exist_ok=True)
        (out / d.PARITY_NAME).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    line = (f"parity {rep['same_answer']}/{rep['n']} answers the same, {rep['identical']}/"
            f"{rep['n']} outputs identical token for token — "
            + (f"{plain['name']} takes its quality from the MTP run"
               if rep["passes"] else f"under {d.PARITY_NEED}: {plain['name']} needs its own run"))
    _log(log_path, f"\n===== [{sid}] devicemark {line} =====")
    return "done", line


def speed(sid: int, rec: dict, log_path: Path) -> tuple[str, str]:
    """the speed test on the setup's llama-server: the same 128 prompt tokens,
    256 decoded greedily past any end; one warm-up, then two timed trials.
    Decode tok/s from the server's own timings — the server's speed, never a
    phone's"""
    d = dm()
    root, key = _served.root_of(rec["base_url"]), rec.get("key", "")
    db.update(sid, status="running", progress="devicemark speed · the prompt")
    try:
        text, ids = SPEED_TEXT, []
        while len(ids) < SPEED_PROMPT_TOKENS:
            st, raw = _served._http("POST", root + "/tokenize", key, {"content": text},
                                    timeout=30)
            if st != 200:
                return "failed", f"the server's /tokenize answered HTTP {st}"
            ids = json.loads(raw).get("tokens") or []
            text += SPEED_TEXT
        prompt = [t if isinstance(t, int) else t.get("id") for t in ids[:SPEED_PROMPT_TOKENS]]
        trials = []
        for i in range(1 + SPEED_TRIALS):
            if db.cancel_requested(sid):
                return "canceled", "canceled"
            db.update(sid, status="running",
                      progress=f"devicemark speed · {'warm-up' if i == 0 else f'trial {i}'}")
            st, raw = _served._http("POST", root + "/completion", key, {
                "prompt": prompt, "n_predict": SPEED_DECODE_TOKENS, "temperature": 0.0,
                "seed": d.SEED, "ignore_eos": True, "cache_prompt": False, "stream": False},
                timeout=config.SERVED_TIMEOUT_S)
            if st != 200:
                return "failed", f"the server's /completion answered HTTP {st}"
            t = json.loads(raw).get("timings") or {}
            trials.append({"warmup": i == 0, "prompt_n": _int(t.get("prompt_n")),
                           "predicted_n": _int(t.get("predicted_n")),
                           "decode_tok_s": _float(t.get("predicted_per_second")),
                           "prefill_tok_s": _float(t.get("prompt_per_second"))})
    except Exception as e:                                  # noqa: BLE001 — said on the row
        return "failed", f"the speed test could not reach the server: {e}"
    timed = [t["decode_tok_s"] for t in trials[1:] if t["decode_tok_s"]]
    if len(timed) < SPEED_TRIALS:
        return "failed", "the server's timings gave no decode speed"
    out = {"decode_tok_s": round(sum(timed) / len(timed), 2), "trials": trials,
           "prompt_tokens": SPEED_PROMPT_TOKENS, "decode_tokens": SPEED_DECODE_TOKENS,
           "label": d.SERVER_SPEED_LABEL, "setup": setup_of(rec, False, _props(rec)),
           "at": time.time()}
    base = row_dir(rec["id"], False)
    base.mkdir(parents=True, exist_ok=True)
    (base / d.SPEED_NAME).write_text(json.dumps(out, indent=1), encoding="utf-8")
    line = (f"{out['decode_tok_s']} tok/s decode ({', '.join(f'{x:g}' for x in timed)}) · "
            f"{d.SERVER_SPEED_LABEL}: not a phone's speed")
    _log(log_path, f"\n===== [{sid}] devicemark speed: {line} =====")
    return "done", line


# ---------------------------------------------------------------------------
# a Hugging Face model's battery, after lm_eval has answered it
# ---------------------------------------------------------------------------

def mark_hf(sid: int, sub: dict, meta: dict, row: Path, thinking: dict) -> str:
    """the three tasks' answers, each counted again with the model's own
    tokenizer, scored as a served row's are"""
    d = dm()
    items = d.load_items(config.DM_ITEMS)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(sub["hf_id"], revision=meta.get("revision"))

        def count(text):
            return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:                                       # noqa: BLE001 — lengths unknown
        count = None
    records = d.records_from_samples(row, count)
    setup = {"model": sub["hf_id"], "runtime": "hf transformers (lm_eval)", "dtype": "bfloat16",
             "thinking": bool(thinking.get("on")), "thinking_mode": thinking.get("mode"),
             "lookahead": False, "mtp": False, "quant": "bf16", "battery": d.VERSION,
             "cap": d.CAP, "seed": d.SEED, "revision": meta.get("revision")}
    return summary_line(d.mark(row, items, setup, records, "full"))


# ---------------------------------------------------------------------------
# 12q.C: a row's answers, for the model page's Answers tab
# ---------------------------------------------------------------------------

ANSWERS_PAGE = 50


def answers(model: str, thinking: bool, bench: str = "", offset: int = 0,
            limit: int = ANSWERS_PAGE) -> dict | None:
    """every item of a row (public benchmark items, all of them): the
    question, the output with its thinking apart, what was read from it, the
    answer, pass or fail, its tokens and whether it ran out of room — no
    answer first, then wrong, then right. One bench, or all; a page at a time.
    None when the row has no answers here"""
    from . import questions
    d = dm()
    rdir = row_dir(model, thinking)
    if not (rdir / d.ITEMS_NAME).exists():
        return None
    rows, counts = [], {}
    for task, b in questions.DM_TASKS.items():
        got = questions._dm_rows(task, rdir)
        c = counts[b] = {"n": 0, "no_answer": 0, "wrong": 0, "capped": 0}
        for key, x in got.items():
            r = x["res"]
            c["n"] += 1
            c["no_answer"] += not r["answered"]
            c["wrong"] += r["answered"] and not r["ok"]
            c["capped"] += r["capped"]
            if bench and b != bench:
                continue
            rows.append({"bench": b, "key": key, "q": x["q"], "options": x["options"],
                         "subject": x["subject"], "gold": x["reference"], "parsed": r["parsed"],
                         "answered": r["answered"], "ok": r["ok"], "capped": r["capped"],
                         "tokens": r["tokens"], "answer": r["answer"], "thinking": r["thinking"],
                         "verdict": r["verdict"], "_order": x["order"]})
    rows.sort(key=lambda x: (0 if not x["answered"] else 1 if not x["ok"] else 2,
                             d.BENCHES.index(x["bench"]), x["_order"][1]))
    for x in rows:
        del x["_order"]
    return {"model": model, "thinking": bool(thinking), "bench": bench or None,
            "total": len(rows), "offset": offset, "counts": counts,
            "items": rows[offset:offset + limit]}
