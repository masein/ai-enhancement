#!/usr/bin/env python3
"""31.2b: the whitepaper's showcase (its Appendix H) on a rented box — the
Masein model's answers to the 12 own questions, asked on the box's own server
on an expert path, recorded as step8804's were on our server.

On the box, remote_gguf.py --showcase <questions file> runs it, after the
server is started and the path's settings check has passed. The questions
file comes from the private repository, pinned by its sha256 (plan U), and
never from the repo: it holds the 12 questions and nothing else. Each question
is asked once, one at a time on one slot, no re-rolls:

- the messages: the question alone, as the user's;
- the request: its limit (81,920 tokens, the thinking counted: the slot's
  83,968 less 2,048, as step8804's), its seed (frontier.seed_of("showcase",
  "own-<n>", 0), step8804's), and the expert path: the path's own sampling and
  thinking, the file's, never sent;
- each answer kept the moment it arrives (showcase-own.jsonl): a step pasted
  again asks only what has no answer yet, and nothing answered is ever asked
  again.

The bundle (showcase-served-<served id>-thinking-on.tar.gz) holds the answers
with their thinking, the setup (the file, the server and its settings check,
the questions' sha256) and the log. On the server, import_remote.py takes it
— refused unless its GGUF is the model's registered file, its launch and
runtime the registered ones, its check passed, its questions the pinned file
and every question answered once with the request above — into
$BENCH_ROOT/showcase/<row>/own.jsonl, where the showcase's own script reads
it. It is no row's benchmark: nothing on the board changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402

for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier as fb  # noqa: E402
import remote_bundle as rb  # noqa: E402

SUITE = "showcase-served"
TASK = "showcase"
REFUSED = 2
CONTEXT = 83_968                    # a slot's, as step8804's showcase server's
LIMIT = 81_920                      # its answer limit, the thinking counted
ANSWERS = "showcase-own.jsonl"
NAME = "showcase-own12.json"        # the questions file, as the private repository holds it
MODEL_WORDS = "the Masein model"


def questions(path: Path) -> list[dict]:
    """the questions file: a list of {n, kind, question}, each n once — or
    ValueError saying what is wrong with it"""
    try:
        got = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{Path(path).name} can't be read as JSON ({e})") from None
    if not isinstance(got, list) or not 1 <= len(got) <= 50:
        raise ValueError(f"{Path(path).name} isn't a list of up to 50 questions")
    ns = []
    for x in got:
        if not (isinstance(x, dict) and isinstance(x.get("n"), int)
                and isinstance(x.get("question"), str) and x["question"].strip()):
            raise ValueError(f"{Path(path).name}: each question is {{n, kind, question}}")
        ns.append(x["n"])
    if len(set(ns)) != len(ns):
        raise ValueError(f"{Path(path).name}: a question's n given twice")
    return got


def request(q: dict, path: int, ctx: int) -> dict:
    """what a question is asked with: its messages, its limit, its seed and
    its path — nothing that samples"""
    return {"messages": [{"role": "user", "content": q["question"]}],
            "max_tokens": min(LIMIT, ctx - 2048), "seed": fb.seed_of(TASK, f"own-{q['n']}", 0),
            "expert_path": path}


def kept(out: Path) -> dict[str, dict]:
    f = out / ANSWERS
    if not f.exists():
        return {}
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["q"]: r for r in rows}


def ask(base: str, qs: list[dict], out: Path, path: int, ctx: int, file: str, runtime: str,
        say) -> tuple[int, str]:
    """each question not yet answered, one at a time: (answered, why it
    stopped — '' when every one is)"""
    from service import served as sv
    have = kept(out)
    for q in qs:
        name = f"own-{q['n']}"
        if name in have:
            continue
        body = {**request(q, path, ctx), "stream": False}
        t0 = time.time()
        try:
            d = sv._post(base + "/v1/chat/completions", "", body, timeout=7200)
        except Exception as e:                          # noqa: BLE001 — said, the rest kept
            return len(have), f"{name}: the server gave no answer ({e})"
        c, u = (d.get("choices") or [{}])[0], d.get("usage") or {}
        msg, finish = c.get("message") or {}, c.get("finish_reason")
        think, reply = msg.get("reasoning_content") or "", msg.get("content") or ""
        text = f"<think>\n{think}\n</think>\n\n{reply}" if think else reply
        out_ = fb.ran_out(text, finish)
        toks = u.get("completion_tokens")
        row = {"q": name, "model": MODEL_WORDS, "file": file,
               "date": time.strftime("%Y-%m-%d", time.gmtime()), "context": ctx,
               "max_tokens": body["max_tokens"], "runtime": runtime, "expert_path": path,
               "sampling": "the file's own for the path: " + fb.path_words(path),
               "request": {k: v for k, v in body.items() if k not in ("messages", "stream")},
               "seed": body["seed"], "prompt_tokens": u.get("prompt_tokens"),
               "output_tokens": toks,
               "thinking_tokens": round(toks * len(think) / max(1, len(think) + len(reply)))
               if isinstance(toks, int) else None,
               "finish": finish, "ran_out": out_, "looped": fb.ended_in_loop(text, out_),
               "seconds": round(time.time() - t0, 1), "thinking": think, "reply": reply}
        with open(out / ANSWERS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        have[name] = row
        say(f"showcase {name}: {toks} tokens, {row['seconds']} s, finish {finish}, ran out "
            f"{out_}, looped {row['looped']}")
    return len(have), ""


def make_bundle(out: Path, served_as: str, run_state: dict, gguf: dict, server: dict,
                rec: dict, qs: list[dict], q_sha: str) -> tuple[Path | None, dict, dict]:
    """the bundle once every question is answered: its path (None until then),
    {task: answers} in it, and {task: {answers, of}} not finished"""
    import remote_gguf as rg
    have = kept(out)
    n, of = sum(1 for q in qs if f"own-{q['n']}" in have), len(qs)
    if n < of:
        return None, {}, {TASK: {"answers": n, "of": of}}
    rows = [have[f"own-{q['n']}"] for q in qs]
    answers = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    row = served_as.replace("/", "__") + "__thinking"
    files: dict[str, bytes] = {f"results/{row}/{ANSWERS}": answers.encode("utf-8")}
    log = (out / rg.OWN_LOG).read_text(encoding="utf-8") if (out / rg.OWN_LOG).exists() else ""
    log += "\n===== llama-server, as it started =====\n" + rg.startup_log(out / rg.SERVER_LOG)
    setup = {"suite": SUITE, "served_as": served_as, "based_on": rec.get("based_on") or "",
             **({"runtime": rec["runtime"]} if rec.get("runtime") else {}),
             "thinking": "on", "gguf": gguf, "server": server, "gpu": rb.gpu_info(),
             "where": rg.WHERE, "board_commit": rg.board_commit(),
             "sessions": run_state.get("sessions", 1), "box": run_state.get("label") or None,
             "image": (rg.board_commit() or "")[:7] or None,
             "started_at": run_state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "showcase": {"questions_sha256": q_sha, "questions": len(qs), "limit": LIMIT,
                          "context": CONTEXT, "answers": len(rows)}}
    files["run.log"] = rb.scrub(log).encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    bundle = {"format": rb.SUITES[SUITE]["format"], "suite": SUITE, "model": served_as,
              "thinking": True, "row": row, "gguf_sha256": gguf["sha256"],
              "tasks": {TASK: len(rows)},
              "answers_sha256": hashlib.sha256(answers.encode("utf-8")).hexdigest(),
              "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    path = rb.write(out / rb.bundle_name(SUITE, served_as, True), files)
    return path, {TASK: {"answers": len(rows), "of": of}}, {}


# ---------------------------------------------------------------------------
# the import, on the server
# ---------------------------------------------------------------------------

def pinned_sha() -> str:
    """the questions file's sha256, as plan U pins it"""
    import frontier_box as fx
    return next(v for p in fx.PLAN_PINS.values() for k, v in p.items() if k == NAME)


def where(row: str) -> Path:
    from service import config
    return config.BENCH_ROOT / "showcase" / row


def checks(b: dict, rec: dict | None, file_sha: str) -> list[str]:
    """every check the bundle fails, in words; [] when it may be imported"""
    import import_frontier as imf
    bundle, setup = b["bundle"], b["setup"]
    bad = imf.setup_problems(setup)
    if bad:
        return bad
    if bundle.get("format") != rb.SUITES[SUITE]["format"] or not isinstance(
            setup.get("showcase"), dict):
        return [f"its format: the bundle's {bundle.get('format')}, this board reads "
                f"{rb.SUITES[SUITE]['format']} for the showcase"]
    out, stop = imf.model_problems(b, rec, file_sha)
    if stop:
        return out
    sc = setup["showcase"]
    if sc.get("questions_sha256") != pinned_sha():
        out.append(f"its questions aren't the pinned file: sha256 "
                   f"{str(sc.get('questions_sha256'))[:16]}…, plan U's {pinned_sha()[:16]}…")
    rows, why = _rows(b)
    if why:
        return out + [why]
    path = fb.path_of(setup.get("runtime"), True)
    if path is None:
        out.append("it was asked on no expert path: the showcase is the k6 path's")
    names = [r.get("q") for r in rows]
    if len(set(names)) != len(names):
        out.append("a question answered twice")
    if len(rows) != sc.get("questions"):
        out.append(f"{len(rows)} answers to its {sc.get('questions')} questions")
    for r in rows:
        try:
            n = int(str(r["q"]).removeprefix("own-"))
        except (KeyError, ValueError):
            out.append(f"an answer to no question: {str(r.get('q'))[:40]}")
            continue
        want = {"max_tokens": min(LIMIT, CONTEXT - 2048),
                "seed": fb.seed_of(TASK, f"own-{n}", 0), "expert_path": path}
        if r.get("request") != want:
            out.append(f"own-{n} was asked with {json.dumps(r.get('request'), sort_keys=True)}; "
                       f"the showcase asks {json.dumps(want, sort_keys=True)}")
    return out


def _rows(b: dict) -> tuple[list[dict], str]:
    raw = b["files"].get(f"results/{b['bundle'].get('row')}/{ANSWERS}")
    if raw is None:
        return [], f"it holds no {ANSWERS}"
    try:
        rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    except ValueError:
        return [], f"its {ANSWERS} isn't JSON lines"
    if not all(isinstance(r, dict) and isinstance(r.get("reply"), str) for r in rows):
        return [], f"a line of its {ANSWERS} isn't an answer"
    return rows, ""


def import_bundle(b: dict, path: Path, by: str, say=print, file_sha: str = "",
                  register: str = "", public_weights: bool = False) -> int:
    """the showcase's answers from a rented box, where the showcase reads them"""
    import import_frontier as imf
    from service import db, served
    bad = imf.bundle_problems(b.get("bundle"))
    if not bad and not isinstance(b.get("setup"), dict):
        bad = ["its setup.json isn't a JSON object"]
    if bad:
        for line in bad:
            say(f"refused — {line}")
        return REFUSED
    bundle, setup = b["bundle"], b["setup"]
    model, row = bundle.get("model") or "", bundle.get("row") or ""
    say(f"{path.name} · sha256 {b['sha256'][:16]} · {model} · the showcase "
        f"({(setup.get('showcase') or {}).get('answers')} answers)")
    if not row or "/" in row or row.startswith(".") or not bundle.get("thinking"):
        say(f"refused — its row {row!r} isn't a thinking row's name")
        return REFUSED
    d = where(row)
    try:
        reg = json.loads((d / "imports.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reg = {"imports": []}
    was = next((x for x in reg["imports"] if x.get("sha256") == b["sha256"]), None)
    if was:
        say(f"imported already, on {was['at']}: nothing changed")
        return 0
    rec = served.get(model) if served.is_served(model) else None
    if register:
        why = ("" if not rec else f"{model} is registered here already: import without "
               "--register") or imf.register_refused(model, file_sha)
        if why:
            say(f"refused — {why}")
            return REFUSED
    bad = checks(b, rec or (imf._would_be(b, file_sha) if register else None), file_sha)
    if bad:
        for line in bad:
            say(f"refused — {line}")
        say("nothing was imported")
        return REFUSED
    if register:
        rec = imf.registered_here(b, register, by, public_weights)
        say(f"{model} registered as {rec['name']}, its file pinned by the sha256 you gave")
    if file_sha and not imf.registered_sha(rec):
        rec["file_sha256"] = {"sha256": file_sha, "by": by, "at": time.time(),
                              "name": (setup.get("gguf") or {}).get("name") or ""}
        db.served_put(rec)
    rows, _ = _rows(b)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    d.mkdir(parents=True, exist_ok=True)
    new = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    own = d / "own.jsonl"
    if own.exists() and own.read_text(encoding="utf-8") != new:
        own.replace(d / f"own-earlier-{stamp}.jsonl")
        say(f"the earlier answers set aside: own-earlier-{stamp}.jsonl")
    own.write_text(new, encoding="utf-8")
    reg["imports"].append({"sha256": b["sha256"], "bundle": path.name, "by": by, "at": stamp,
                           "box": setup.get("box") or "",
                           "gpu": (setup.get("gpu") or {}).get("name") or ""})
    (d / "imports.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    cut = sum(1 for r in rows if r.get("ran_out"))
    say(f"the showcase: {len(rows)} answers of {model} · thinking into {own}"
        + (f" · {cut} ran out of room" if cut else "") + " — no row of the board changes")
    return 0
