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

from . import config, db, hfmeta
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


class ItemError(Exception):
    """12q.F: the server answered this item with an error of its own — it is
    up, and the next item may be fine. llama-server says 500 when its chat
    parser can't read what the model wrote ("The model produced output that
    does not match the expected peg-native format"), 400 when a request is
    one it can't take"""


# the statuses that are about one request, from a server that is answering
ITEM_STATUSES = (400, 500)


def _said(raw: bytes) -> str:
    """what the server's error body says, in its own words"""
    try:
        e = json.loads(raw).get("error")
        msg = e.get("message") if isinstance(e, dict) else e
        if msg:
            return str(msg)[:400]
    except Exception:                                       # noqa: BLE001 — not JSON: as it is
        pass
    return raw[:400].decode("utf-8", "replace")


def _post(url: str, key: str, body: dict) -> dict:
    """a request to the server: its reply, _Retry when it isn't answering,
    ItemError when it answers this request with an error of its own"""
    try:
        st, raw = _served._http("POST", url, key, body, timeout=config.SERVED_TIMEOUT_S)
    except Exception as e:                                  # noqa: BLE001 — no answer: retried
        raise _Retry(str(getattr(e, "reason", e))) from None
    if st in ITEM_STATUSES:
        raise ItemError(f"HTTP {st}: {_said(raw)}")
    if st in (408, 429) or st >= 500:
        raise _Retry(f"HTTP {st}")
    if st != 200:
        raise ValueError(f"HTTP {st}: {raw[:160].decode('utf-8', 'replace')}")
    return json.loads(raw)


def ask(rec: dict, text: str, s: dict) -> dict:
    """one item, one reply: the text (thinking the server split off put back
    in its tags), what it generated — thinking and answer apart where the
    server says both — whether the cap cut it, its decode speed and its wall
    time"""
    body = {"model": rec["pin"].get("model") or rec["name"],
            "messages": [{"role": "user", "content": text}], "stream": False, **s}
    base, key = _served._endpoint(rec)
    t0 = time.time()
    reply = _post(base + "/chat/completions", key, body)
    wall = time.time() - t0
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


def ask_raw(rec: dict, text: str, s: dict) -> dict:
    """12q.F: the same item without the server's chat parsing — the prompt the
    server's own template makes of it (POST /apply-template), completed with
    the same greedy settings, seed and cap (POST /completion), and the raw
    text read here as any answer is: what follows the last </think>"""
    d = dm()
    root, key = _served.root_of(rec["base_url"]), rec.get("key", "")
    t0 = time.time()
    prompt = _post(root + "/apply-template", key, {
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": s["chat_template_kwargs"]}).get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ItemError("/apply-template gave no prompt")
    reply = _post(root + "/completion", key, {
        "prompt": prompt, "n_predict": s["max_tokens"], "temperature": s["temperature"],
        "seed": s["seed"], "cache_prompt": False, "stream": False})
    wall = time.time() - t0
    raw = reply.get("content") or ""
    # a template that opens the thinking itself leaves the tag in the prompt:
    # put back, so thinking the cap cut off is thinking, not an answer
    full = ("<think>\n" + raw) if re.search(r"<think>\s*$", prompt) else raw
    think, answer = d.split_thinking(full)
    t = reply.get("timings") if isinstance(reply.get("timings"), dict) else {}
    gen = _int(reply.get("tokens_predicted"))
    gen = gen if gen is not None else _int(t.get("predicted_n"))
    thought = count_tokens(rec, think) if think else (0 if gen is not None else None)
    cut = reply.get("stop_type") == "limit" or bool(reply.get("stopped_limit"))
    return {"text": full, "answer": answer, "thinking_chars": len(think),
            "think_tokens_from": "/tokenize" if think and thought is not None else None,
            "prompt_tokens": _int(reply.get("tokens_evaluated"))
            if reply.get("tokens_evaluated") is not None else _int(t.get("prompt_n")),
            "gen_tokens": gen, "think_tokens": thought if think else None,
            "answer_tokens": gen - thought if gen is not None and thought is not None else None,
            "finish": "length" if cut else "stop",
            "capped": cut or bool(gen is not None and gen >= s["max_tokens"]),
            "decode_tok_s": _float(t.get("predicted_per_second")), "wall_s": round(wall, 3)}


def no_answer(chat_error: str, fallback_error: str) -> dict:
    """12q.F: an item the server failed on twice, and again without its chat
    parsing: no answer — counted wrong, as the protocol counts one — with what
    the server said both ways"""
    return {"text": "", "answer": "", "thinking_chars": 0, "think_tokens_from": None,
            "prompt_tokens": None, "gen_tokens": None, "think_tokens": None,
            "answer_tokens": None, "finish": "error", "capped": False, "decode_tok_s": None,
            "wall_s": None, "error": {"chat": chat_error, "fallback": fallback_error}}


def answer_item(rec: dict, text: str, s: dict) -> dict:
    """12q.F: one item's record, whatever the server makes of it. Asked; on an
    error of the item's own (a 500, a 400), once more; then without the
    server's chat parsing, marked "raw fallback" with the server's words; and
    if that fails too, no answer. A server that isn't answering at all is
    another matter (_Retry, then ServerStopped): the row waits, then stops"""
    try:
        return _ask_patiently(rec, text, s)
    except ItemError:
        pass
    try:
        return _ask_patiently(rec, text, s)
    except ItemError as e:
        said = str(e)
    try:
        return {**_ask_patiently(rec, text, s, ask_raw), "raw_fallback": {"error": said}}
    except (ItemError, ValueError) as e:    # 12s: a server with no /apply-template says 404
        return no_answer(said, str(e))


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


def _ask_patiently(rec: dict, text: str, s: dict, how=None) -> dict:
    """a server that doesn't answer is asked again for SERVED_RETRY_S"""
    t0 = time.time()
    while True:
        try:
            return (how or ask)(rec, text, s)
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


def kept(rec: dict, keys: list[tuple[str, str]], s: dict, store: Path,
         also: tuple[Path, ...] = ()) -> dict:
    """the answers already in `store` (or `also`) to these items, on the same
    server, settings and battery"""
    fp, want, have = fingerprint(rec, s), set(keys), {}
    for f in (*also, store):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("fp") == fp and (r["bench"], r["key"]) in want:
                    have[(r["bench"], r["key"])] = r
    return have


def wait_for(sid: int, rec: dict, log_path: Path, what: str = "devicemark parity") -> str:
    """12w: '' once this setup's server answers with the file registered, else
    why the run stops. The parity check asks two setups, and only one
    llama-server may fit on the card: the run says which to start and waits
    DM_SWAP_WAIT_S for it, so the other can be stopped first. A server up with
    another file is not waited for"""
    t0, said = time.time(), False
    while True:
        why = _served.check_pin(rec)
        if not why or why == _served.CHANGED_LINE:
            return why
        if db.cancel_requested(sid):
            return "canceled"
        left = config.DM_SWAP_WAIT_S - (time.time() - t0)
        if left <= 0:
            return (f"its server didn't answer in {max(1, round(config.DM_SWAP_WAIT_S / 60))} "
                    f"min ({why})")
        db.update(sid, status="running",
                  progress=f"{what} · start {rec['name']}'s server now (the other can stop): "
                           f"waiting {max(1, -(-int(left) // 60))} min more")
        if not said:
            _log(log_path, f"[devicemark] waiting for {rec['id']} at {rec['base_url']}: {why}")
            said = True
        time.sleep(min(5.0, max(0.05, config.DM_SWAP_WAIT_S / 24)))


def answer_all(rec: dict, keys: list[tuple[str, str]], items: dict, s: dict, store: Path,
               on_progress=None, canceled=lambda: False,
               also: tuple[Path, ...] = ()) -> tuple[list[dict], Exception | None]:
    """every item asked that isn't answered in `store` — or `also`, the
    pilot's — under these settings, each reply appended as it lands. (the
    answers, in `keys`' order, and why it stopped — None when every one is in)"""
    fp = fingerprint(rec, s)
    have = kept(rec, keys, s, store, also)
    todo = [k for k in keys if k not in have]
    lock, halt = threading.Lock(), []
    total, t0, done = len(keys), time.time(), [len(keys) - len(todo)]
    store.parent.mkdir(parents=True, exist_ok=True)

    def one(k: tuple[str, str]) -> None:
        if halt or canceled():
            return
        try:
            got = answer_item(rec, items[k]["text"], s)
        except (_served.ServerStopped, ValueError) as e:
            with lock:
                halt.append(e.why if isinstance(e, _served.ServerStopped) else str(e))
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
    got = [have[k] for k in keys if k in have]
    if not halt:
        return got, None
    # 12q.F: where it stopped, counted once every answer on its way has landed
    # (it said "at 0 of 0")
    return got, _served.ServerStopped(len(got), total, halt[0], refused=(
        f"the server stopped answering after {len(got)} of {total}"
        + (f" ({halt[0]})" if halt[0] else "")))


# ---------------------------------------------------------------------------
# the setup a row ran on
# ---------------------------------------------------------------------------

_QUANT = re.compile(r"(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|q\d_\w+)", re.I)


def setup_of(rec: dict, thinking: bool, props: dict | None = None) -> dict:
    """model, lookahead, MTP, thinking, quant, the server's flags as far as
    it or its registration says them, port and battery"""
    d = dm()
    lk = _served.launch(rec) or {}
    file = rec["pin"].get("file") or ""
    m = re.search(r":(\d+)(?:/|$)", rec["base_url"])
    q = _QUANT.findall(file.rsplit(".", 1)[0])
    return {"model": rec["id"], "name": rec["name"], "runtime": "llama-server",
            "base_url": rec["base_url"], "port": int(m.group(1)) if m else None,
            "file": file, "build": rec["pin"].get("build"), "ctx": rec["pin"].get("ctx"),
            "quant": q[-1] if q else None, "phone": _served.is_phone(rec),
            # 12z A1: from its launch, never words in its description
            "lookahead": bool(lk.get("lookahead")), "mtp": bool(lk.get("mtp")),
            "launch": {"flags": rec.get("flags") or "", "env": rec.get("env") or ""},
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
    why = d.items_differ(items, config.DM_ITEMS_SHA256)      # 15.6: the repo's battery
    if why:
        return "failed", why
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
        _log(log_path, f"[devicemark] {stopped}")
        return "failed", (f"{stopped} · the answers it gave are kept: the next run asks only "
                          f"the other {len(keys) - len(got)}")
    # 12q.F: what the server failed on, said in the log as it is on the row
    for r in got:
        if r.get("raw_fallback") or r.get("error"):
            _log(log_path, f"[devicemark] {r['bench']} {r['key']}: " + (
                f"raw fallback — the server said: {r['raw_fallback']['error']}"
                if r.get("raw_fallback") else
                f"no answer — the server said: {r['error']['chat']}; without its chat "
                f"parsing: {r['error']['fallback']}"))
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
    odd = "".join(f" · {row[k]} {w}" for k, w in (
        ("raw_fallback", "raw fallback"), ("errors", "no answer after a server error"))
        if row.get(k))
    return (head + " · ".join(bits) + f" · answered {row['answered_pct']:.0%} · median "
            f"{row['median_tokens']} tokens" + odd
            + (f" · cap check: {cc['line']}" if cc else ""))


def cap_check(rec: dict, items: dict, pilot_keys: list[tuple[str, str]]) -> dict:
    """does the cap count the thinking? One MATH item, thinking on, capped at
    CAP_CHECK_TOKENS: the server should stop at that many tokens with the
    thinking unfinished — every one of them thinking"""
    k = next(k for k in pilot_keys if k[0] == "math")
    try:
        r = _ask_patiently(rec, items[k]["text"], settings(True, CAP_CHECK_TOKENS))
    except (_served.ServerStopped, ValueError, ItemError) as e:
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
        # 12w: a setup with items still to answer is waited for — only one
        # server may fit on the card — and one whose 50 are kept needs no server
        if len(kept(rec, keys, s, store)) < len(keys):
            why = wait_for(sid, rec, log_path)
            if why == "canceled":
                return "canceled", "canceled: the answers so far are kept for the next run"
            if why:
                return "failed", f"{rec['name']}: {why} · the answers so far are kept"
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
# 12q.G: a Hugging Face model's battery, before lm_eval answers it
# ---------------------------------------------------------------------------

# counted on the battery's longest questions by their characters; the
# template's own tokens and lm_eval's start token, over what is counted
PROMPT_SAMPLE, PROMPT_SLACK = 25, 64


def prompt_tokens(pretrained: str, revision: str | None, texts: list[str],
                  thinking: dict) -> tuple[int, bool]:
    """(the battery's longest prompt in this model's tokens, counted?): each
    question as lm_eval sends it, the one user message with the chat template
    and the thinking switch. A tokenizer that doesn't load here (its own code
    is never run in the service) isn't counted: three characters a token, which
    is on the long side"""
    longest = sorted(texts, key=len, reverse=True)[:PROMPT_SAMPLE]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pretrained, revision=revision,
                                            trust_remote_code=False)
        kw = {"enable_thinking": bool(thinking.get("on"))} if thinking.get("mode") == "switch" else {}

        def count(text):
            said = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, **kw)
            return len(tok(said, add_special_tokens=False)["input_ids"])
        return max(count(t) for t in longest), True
    except Exception:                                       # noqa: BLE001 — estimated instead
        return -(-len(longest[0]) // 3), False


def hf_plan(sub: dict, meta: dict, thinking: dict, pretrained: str | None = None) -> dict:
    """how a Hugging Face model sits the battery on lm_eval — {max_length,
    room, longest, counted, ctx, batch, need_gb, why}:
      max_length  what lm_eval is told: the prompt's room and the cap. The room
                  is DM_PROMPT_TOKENS, or the battery's longest prompt where
                  that is longer, so no question is ever cut
      batch       how many answers are written at a time, and need_gb what that
                  takes (hfmeta.gen_estimate): 4,096 tokens each, DM_HF_MAX_BATCH
                  at most
    A model that reads fewer tokens than max_length can't sit it: refused, in
    PreflightError's words, before anything is loaded"""
    d = dm()
    arch = meta.get("archinfo") or {}
    try:
        texts = [r["text"] for r in d.load_items(config.DM_ITEMS).values()]
    except Exception:                       # noqa: BLE001 — the run says so, building its tasks
        texts = []
    longest, counted = (prompt_tokens(pretrained or sub["hf_id"], meta.get("revision"), texts,
                                      thinking) if texts else (config.DM_PROMPT_TOKENS // 2, False))
    room = max(config.DM_PROMPT_TOKENS, longest + PROMPT_SLACK)
    ctx = arch.get("ctx")
    if ctx and ctx < room + d.CAP:
        raise hfmeta.PreflightError(
            f"{sub['hf_id']} reads {ctx:,} tokens at most (its config.json), and DeviceMark's "
            f"protocol needs {room + d.CAP:,}: {room:,} for the prompt (the battery's longest "
            f"is {'' if counted else 'about '}{longest:,} in its tokens) and the cap of "
            f"{d.CAP:,} for the answer. It can't sit the battery as the other models do.")
    return {"max_length": room + d.CAP, "room": room, "longest": longest, "counted": counted,
            "ctx": ctx,
            **hfmeta.gen_estimate(arch, meta.get("params"), meta["vocab"], prompt=longest,
                                  new=d.CAP, max_batch=config.DM_HF_MAX_BATCH)}


def hf_plan_lines(plan: dict, batch: int | None = None) -> str:
    """the plan, for the run's log; `batch` is the one a task runs at, once an
    earlier one ran out of memory at the planned"""
    cap, planned = dm().CAP, plan["batch"]
    some = lambda n: f"{n} answer{'s' if n != 1 else ''} written at a time"
    return (f"[devicemark] lm_eval is told max_length={plan['max_length']}: {plan['room']:,} "
            f"tokens for the prompt (the battery's longest is "
            f"{'' if plan['counted'] else 'about '}{plan['longest']:,} in this model's tokens"
            f"{'' if plan['counted'] else ', by its characters'}) and the cap of {cap:,} · "
            + (f"the model reads up to {plan['ctx']:,}" if plan["ctx"] else
               "the model's config gives no limit of its own") + "\n"
            + (f"[devicemark] {some(batch)}: {planned} ran out of GPU memory\n"
               if batch and batch != planned else
               f"[devicemark] {some(planned)} · about {plan['need_gb']:g} GB: {plan['why']}\n"))


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
             "cap": d.CAP, "seed": d.SEED, "revision": meta.get("revision"),
             # 15.2: the tasks answered on a rented GPU, imported (where_of)
             **where_of(row)}
    return summary_line(d.mark(row, items, setup, records, "full"))


# ---------------------------------------------------------------------------
# 15.2: a row answered, whole or in part, on a rented GPU
# ---------------------------------------------------------------------------

REMOTE_NAME = "remote_imports.json"
TASK_WORDS = {"dm_ifeval": "IFEval", "dm_mmlu_pro": "MMLU-Pro", "dm_math": "MATH"}


def _and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def where_of(row: Path) -> dict:
    """which of the row's tasks have the answers a bundle brought
    (scripts/import_remote.py keeps the record): {where, remote} — "run on a
    rented GPU (NVIDIA GeForce RTX 4090)", or, for a row answered in part
    here, which tasks were which. {} for a row answered here alone"""
    try:
        reg = json.loads((row / REMOTE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    remote = {t: x for t, x in (reg.get("tasks") or {}).items()
              if x.get("samples") and all((row / rel).exists() for rel in x["samples"])}
    if not remote:
        return {}
    # 15.5: a task answered in shards ran on as many rented GPUs
    names = sorted({g for x in remote.values() for g in (x.get("gpus") or [x.get("gpu") or "a GPU"])})
    gpus = f"a rented GPU ({names[0]})" if len(names) == 1 else f"rented GPUs ({_and(names)})"
    ns = {int(x.get("shards") or 1) for x in remote.values()}
    split = ("" if max(ns) == 1 else f", in {max(ns)} shards" if len(ns) == 1 else ", in shards")
    here = [t for t in config.DM_TASKS if t not in remote and (row / f"{t}_0shot").is_dir()]
    if here:
        line = (f"{_and([TASK_WORDS[t] for t in config.DM_TASKS if t in remote])} run on "
                f"{gpus}{split}; {_and([TASK_WORDS[t] for t in here])} on this server")
    else:
        line = f"run on {gpus}{split}"
    return {"where": line, "remote": {t: {"gpu": x.get("gpu"), "bundle": x.get("bundle"),
                                          "sha256": (x.get("sha256") or "")[:16],
                                          **({"shards": x["shards"]} if x.get("shards") else {})}
                                      for t, x in sorted(remote.items())}}


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
                         "verdict": r["verdict"], "fallback": r.get("fallback"),
                         "error": r.get("error"), "_order": x["order"]})
    rows.sort(key=lambda x: (0 if not x["answered"] else 1 if not x["ok"] else 2,
                             d.BENCHES.index(x["bench"]), x["_order"][1]))
    for x in rows:
        del x["_order"]
    return {"model": model, "thinking": bool(thinking), "bench": bench or None,
            "total": len(rows), "offset": offset, "counts": counts,
            "items": rows[offset:offset + limit]}
