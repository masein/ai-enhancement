"""23: two builds, question by question (eval_tasks/reports/paired.json, kind
"paired"; docs/prompts/phase-23-two-builds-question-by-question.md).

Our builds answer the same questions, so two of them are compared on each
question, not as two scores side by side: a question both got right, or both
got wrong, says nothing about which is better. Every number comes from each
row's newest scored samples file (`frontier[].ok`, the board's own verdict:
an answer that ran out of room is wrong, MATH and OTIS with the model check,
SimpleQA's "not attempted" not right). A question only one build has, or one
not graded on either side yet (`ok` None, or `ungraded`), is left out and
counted. The sums are scripts/paired_stats.py's.

Worked out by the stats worker (service/report_stats.py), every pair of the
file's builds and each benchmark, and kept; a request reads what is kept and
never computes. A pair is kept once, in the file's order: the page's other
order is the same numbers, their signs and sides swapped (flipped).

The reader's rule: the hidden half is counted, never listed or opened, and a
benchmark the board keeps unlisted (GPQA, HLE, MATH) never shows its question
text; an answer is shown with each line quoting its question withheld."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import config

KIND = "paired"
ARC = "arc_agi2_public"
SELF = ("gpqa_diamond_epoch", "otis_aime_epoch")     # self-disagreement: several runs
WAY = "paired-1"                                      # the way of working out, in each key
TITLE = "Two builds, question by question"
WHAT = "Which questions one build gets right and the other gets wrong."
NOTES = [
    "Why question by question. Both builds answered the same questions. A question both got "
    "right, or both got wrong, tells us nothing about which is better; only the questions where "
    "they differ count. Comparing that way finds much smaller real differences than setting two "
    "scores with their ± side by side.",
    "The 95% range is where the true difference most likely lies. If it crosses 0, the two "
    "builds can't be told apart on that benchmark with this many questions, and the verdict "
    "says \"can't tell\".",
    "The test. With one answer per question, the verdict uses the exact McNemar test, which "
    "looks only at the questions where the two builds differ. With several answers per "
    "question, each question counts once, at its average, and the verdict uses the range.",
    "Churn. Two builds with the same score can still differ on many questions. The line under "
    "the table compares that churn with how often each build disagrees with itself between two "
    "of its own runs; churn well above that is a sign the builds really behave differently, "
    "even where the scores are level.",
    "Ran out of room counts as wrong, as everywhere on the board; SimpleQA's \"not attempted\" "
    "counts as not right.",
    "Questions listed follow the board's reader rules: the hidden half is counted but not "
    "listed.",
    # 23e: as the step-checkpoints report says it
    "Near the ceiling marks a benchmark where our reference build scores 90% or more. There, a "
    "few questions decide the difference.",
]
# how a runtime's settings are said in the line under the sentence
RUNTIME_WORDS = {"lda-2610": "the lda runtime", "card-ornith": "its own card's settings"}
FAMILY_WORDS = {"qwen3.6": "Qwen's card settings", "gemma-4": "Gemma's card settings"}


def _scripts() -> None:
    p = str(Path(__file__).resolve().parent.parent / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


class Refused(PermissionError):
    """a question this report doesn't show"""


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

def resolve(rep: dict, others: dict[str, dict]) -> dict:
    """the file with its builds: every entry of `ours` in the report it names
    (builds: {"from": "step-checkpoints"}), so a new checkpoint is in both"""
    src = (rep.get("builds") or {}).get("from") if isinstance(rep.get("builds"), dict) else None
    ours = (others.get(src) or {}).get("ours") if src else rep.get("builds")
    return {**rep, "ours": [dict(o) for o in ours or [] if isinstance(o, dict)
                            and o.get("label") and o.get("row")]}


def labels(rep: dict) -> list[str]:
    return [o["label"] for o in rep["ours"]]


def _row(rep: dict, label: str) -> str:
    return next(o["row"] for o in rep["ours"] if o["label"] == label)


def samples(rep: dict, label: str, task: str) -> Path | None:
    from . import reports
    return reports.samples_file(reports.row_dir(_row(rep, label)), task)


def offered(rep: dict) -> list[str]:
    """the builds the picker offers: those with scored answers for at least
    one benchmark"""
    return [x for x in labels(rep)
            if any(samples(rep, x, b["task"]) is not None for b in rep["benchmarks"])]


# ---------------------------------------------------------------------------
# a build's questions, from its scored samples — kept in this process while
# its file is the same (the worker reads each file once for its six pairs)
# ---------------------------------------------------------------------------

_tables: dict[str, dict] = {}


def table_of(f: Path, task: str) -> dict[str, dict]:
    """{question: {ok: [bool], out: [bool], epochs: [int], group, skip, at}} —
    `skip` when a run isn't graded yet (ok None) or its grader gave none"""
    from . import report_stats as rs
    from . import reports
    key = f"{task}|{rs.identity(f)}"
    if key in _tables:
        return _tables[key]
    out: dict[str, dict] = {}
    for off, row in reports.jsonl(f):
        doc = row.get("doc") or {}
        qid = str(doc.get("id") if doc.get("id") is not None else row.get("doc_id"))
        runs = [r for r in row.get("frontier") or [] if isinstance(r, dict)]
        if not runs:
            continue
        out[qid] = {"ok": [r.get("ok") is True for r in runs],
                    "out": [bool(r.get("ran_out")) for r in runs],
                    "epochs": [int(r.get("epoch", k) or 0) for k, r in enumerate(runs)],
                    "group": str(doc.get("group") or qid),
                    "skip": any(r.get("ok") is None or r.get("ungraded") for r in runs),
                    "at": off}
    if len(_tables) > 64:
        _tables.clear()
    _tables[key] = out
    return out


def _units(task: str, ta: dict, tb: dict) -> tuple[list[tuple], dict]:
    """the questions (ARC-AGI-2: the tasks) both builds answered and both
    have graded — [(id, A's runs, B's runs, grids)] — and those left out"""
    _scripts()
    import paired_stats as ps
    left = {"only_a": 0, "only_b": 0, "skip_a": 0, "skip_b": 0}
    if task != ARC:
        units = []
        for q in sorted(set(ta) | set(tb)):
            a, b = ta.get(q), tb.get(q)
            if a is None or b is None:
                left["only_b" if a is None else "only_a"] += 1
            elif a["skip"] or b["skip"]:
                left["skip_a" if a["skip"] else "skip_b"] += 1
            else:
                units.append((q, a, b, None))
        return units, left
    # ARC-AGI-2: a task is compared when every grid of it is on both sides,
    # graded; its score the share of its grids solved (either attempt)
    groups: dict[str, list[str]] = {}
    for q, x in [*ta.items(), *tb.items()]:
        g = groups.setdefault(x["group"], [])
        if q not in g:
            g.append(q)
    units = []
    for t, grids in sorted(groups.items()):
        a = [ta.get(q) for q in grids]
        b = [tb.get(q) for q in grids]
        if any(x is None for x in a) or any(x is None for x in b):
            left["only_b" if any(x is None for x in a) else "only_a"] += 1
        elif any(x["skip"] for x in a) or any(x["skip"] for x in b):
            left["skip_a" if any(x["skip"] for x in a) else "skip_b"] += 1
        else:
            units.append((t, ps.arc_tasks({q: x["ok"] for q, x in zip(grids, a)}, {}),
                          ps.arc_tasks({q: x["ok"] for q, x in zip(grids, b)}, {}), grids))
    return units, left


def _score(u: tuple, side: int) -> float:
    """a unit's score for A (1) or B (2): the mean of its runs; ARC's task score"""
    x = u[side]
    if u[3] is not None:                                    # ARC: {grid: solved}
        return sum(x.values()) / len(x)
    return sum(x["ok"]) / len(x["ok"])


def pair(rep: dict, task: str, ta: dict, tb: dict) -> dict:
    """A against B on one benchmark: the difference with its range and
    verdict, the counts each way with the questions that may be listed, how
    many questions differ, and (GPQA, OTIS) how often each disagrees with
    itself and the two with each other"""
    _scripts()
    import frontier as fb
    import paired_stats as ps

    from . import reports
    units, left = _units(task, ta, tb)
    n = len(units)
    sa = [_score(u, 1) for u in units]
    sb = [_score(u, 2) for u in units]
    one = task != ARC and int((fb.BENCH.get(task) or {}).get("epochs") or 1) == 1
    up = [u for u, x, y in zip(units, sa, sb) if x > y]
    down = [u for u, x, y in zip(units, sa, sb) if y > x]
    if one:
        got = ps.one_run(len(up), len(down), n)
    else:
        got = ps.several_runs(sa, sb)
        got["p"] = None
    self_ = None
    if task in SELF and n:
        sd = lambda x: ps.self_disagree(len(x["ok"]), sum(x["ok"]))        # noqa: E731
        va = [sd(u[1]) for u in units if sd(u[1]) is not None]
        vb = [sd(u[2]) for u in units if sd(u[2]) is not None]
        cross = [ps.cross_disagree(x, y) for x, y in zip(sa, sb)]
        self_ = {"a": sum(va) / len(va) if va else None, "b": sum(vb) / len(vb) if vb else None,
                 "cross": sum(cross) / len(cross)}

    def shown(u: tuple) -> bool:
        ids = u[3] if u[3] is not None else [u[0]]
        return all(reports.shown_half(task, q) for q in ids)

    def entry(u: tuple) -> dict:
        if u[3] is not None:
            return {"id": u[0], "grids": list(u[3])}
        return {"id": u[0]}
    lists = {s: [entry(u) for u in us if shown(u)] for s, us in (("a", up), ("b", down))}
    return {"task": task, "unit": "tasks" if task == ARC else "questions", "n": n,
            "of": n + sum(left.values()), **left,
            "runs": 1 if one else int((fb.BENCH.get(task) or {}).get("epochs") or 1),
            "d": got["d"], "se": got["se"], "lo": got["lo"], "hi": got["hi"], "p": got["p"],
            "verdict": got["verdict"], "a_better": len(up), "b_better": len(down),
            "differ": (len(up) + len(down)) / n if n else None, "self": self_,
            "lists": lists, "held": {"a": len(up) - len(lists["a"]),
                                     "b": len(down) - len(lists["b"])}}


def flip(x: dict) -> dict:
    """the same pair in the other order: B against A"""
    _scripts()
    import paired_stats as ps
    if not x:
        return x
    sw = {"a": "b", "b": "a"}
    neg = lambda v: None if v is None else -v              # noqa: E731
    return {**x, "d": neg(x.get("d")), "lo": neg(x.get("hi")), "hi": neg(x.get("lo")),
            "verdict": {ps.A_HIGHER: ps.B_HIGHER, ps.B_HIGHER: ps.A_HIGHER}.get(
                x.get("verdict"), x.get("verdict")),
            "a_better": x.get("b_better"), "b_better": x.get("a_better"),
            "only_a": x.get("only_b"), "only_b": x.get("only_a"),
            "skip_a": x.get("skip_b"), "skip_b": x.get("skip_a"),
            "self": None if not x.get("self") else {**x["self"], "a": x["self"]["b"],
                                                     "b": x["self"]["a"]},
            "lists": {sw[k]: v for k, v in (x.get("lists") or {}).items()},
            "held": {sw[k]: v for k, v in (x.get("held") or {}).items()}}


# ---------------------------------------------------------------------------
# the worker's jobs, and what is kept
# ---------------------------------------------------------------------------

def pairs(rep: dict) -> list[tuple[str, str]]:
    """every pair of the file's builds, once, in the file's order"""
    ls = labels(rep)
    return [(ls[i], ls[j]) for i in range(len(ls)) for j in range(i + 1, len(ls))]


def _slot(rep: dict, a: str, b: str, task: str) -> str:
    return "|".join(["paired", str(rep.get("id")), a, b, task])


def _key(fa: Path | None, fb_: Path | None) -> str:
    from . import report_stats as rs
    return "|".join([WAY, rs.identity(fa), rs.identity(fb_)])


def work_out(rep: dict, a: str, b: str, task: str) -> dict:
    fa, fb_ = samples(rep, a, task), samples(rep, b, task)
    return pair(rep, task, table_of(fa, task), table_of(fb_, task))


def jobs(rep: dict) -> list:
    """23: every pair's benchmarks both builds have scored answers for"""
    from . import report_stats as rs
    out = []
    for a, b in pairs(rep):
        for bm in rep["benchmarks"]:
            fa, fb_ = samples(rep, a, bm["task"]), samples(rep, b, bm["task"])
            if fa is not None and fb_ is not None:
                out.append(rs.Job(_slot(rep, a, b, bm["task"]), _key(fa, fb_),
                                  f"question by question, {a} and {b} · {bm['label']}",
                                  lambda a=a, b=b, t=bm["task"]: work_out(rep, a, b, t)))
    return out


def _ordered(rep: dict, a: str, b: str) -> tuple[str, str, bool]:
    """(first, second, flipped): the pair as it is kept"""
    ls = labels(rep)
    return (a, b, False) if ls.index(a) < ls.index(b) else (b, a, True)


def kept(rep: dict, a: str, b: str, task: str, compute: bool = False) -> dict | None:
    """A against B on one benchmark, as kept (worked out now with `compute`):
    None when never worked out, `fresh` False when the answers changed since"""
    from . import report_stats as rs
    x, y, flipped = _ordered(rep, a, b)
    slot = _slot(rep, x, y, task)
    key = _key(samples(rep, x, task), samples(rep, y, task))
    e = rs.kept(slot)
    if compute and (not e or e.get("key") != key or "value" not in e):
        e = rs.keep(slot, key, work_out(rep, x, y, task))
    if not e or "value" not in e:
        return None
    v = e["value"]
    return {**(flip(v) if flipped else v), "fresh": e.get("key") == key, "at": e.get("at")}


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _setup(rep: dict, label: str) -> dict:
    from . import frontier as sf
    from . import reports
    d = reports.row_dir(_row(rep, label))
    for b in rep["benchmarks"]:
        try:
            got = json.loads((sf.task_dir(d, b["task"]) / sf.SETUP).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(got, dict):
            return got
    return {}


def settings_words(setup: dict) -> str:
    rt = setup.get("runtime") or ""
    if rt:
        return RUNTIME_WORDS.get(rt, f"the runtime {rt}")
    fam = setup.get("family") or ""
    return FAMILY_WORDS.get(fam, "the server's own defaults" if not fam or fam ==
                            "server defaults" else f"{fam}'s card settings")


def settings_parts(rep: dict, a: str, b: str) -> dict | None:
    """each one's settings in words when the two didn't run the same way;
    None when they did, or a setup can't be read"""
    sa, sb = _setup(rep, a), _setup(rep, b)
    if not sa or not sb:
        return None
    if (sa.get("runtime") or "") == (sb.get("runtime") or "") and \
            sa.get("sampling") == sb.get("sampling"):
        return None
    return {"a": settings_words(sa), "b": settings_words(sb)}


def settings_line(rep: dict, a: str, b: str) -> str:
    """'' when the two ran on the same settings; else each one's, in words"""
    p = settings_parts(rep, a, b)
    return f"{a} ran on {p['a']}; {b} on {p['b']}." if p else ""


def sentence(a: str, b: str, rows: list[dict]) -> str:
    _scripts()
    import paired_stats as ps
    done = [r for r in rows if r.get("pair") and r["pair"].get("n")]
    if not done:
        return f"{a} and {b} have no benchmark compared yet."
    up = [r for r in done if r["pair"]["verdict"] == ps.A_HIGHER]
    down = [r for r in done if r["pair"]["verdict"] == ps.B_HIGHER]
    level = len(done) - len(up) - len(down)

    def ceil(rs: list[dict]) -> str:
        # 23e: a higher or lower near the ceiling says so
        names = [r["label"] for r in rs if r.get("ceiling")]
        if not names:
            return ""
        return " (" + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and "
                       + names[-1]) + " near the ceiling)"
    bits = [f"higher than {b} on {len(up)} benchmark{'s' if len(up) != 1 else ''}{ceil(up)}",
            f"lower on {len(down)}{ceil(down)}"]
    return (f"{a} is {bits[0]}, {bits[1]}, and the two can't be told apart on {level}.")


def report(rep: dict, rows: list[dict], a: str = "", b: str = "", compute: bool = False
           ) -> dict:
    """the report for the pair a, b (the file's default when not given): each
    benchmark's numbers as kept, the sentence, the settings line and the
    notes. `rows`: the board's rows of the builds (reports.our_rows)"""
    from . import report_stats as rs
    from . import reports
    offer = offered(rep)
    want = [x for x in (rep.get("default") or []) if x in offer]
    a = a if a in labels(rep) else (want[0] if want else (offer[0] if offer else ""))
    b = b if b in labels(rep) else next((x for x in [*want, *offer] if x != a), "")
    if a == b:
        raise ValueError("pick two different builds")
    cells = {r["label"]: r["cells"] for r in rows}
    # 23e: near the ceiling, by the reference of the report its builds come from
    src = (reports.files().get((rep.get("builds") or {}).get("from") or "") or {}) \
        if isinstance(rep.get("builds"), dict) else {}
    from . import training
    ref_cells = cells.get(src.get("reference")) or {}
    out_rows, ats, pending = [], [], []
    for bm in reports.bench_meta(rep["benchmarks"]):
        fa, fb_ = samples(rep, a, bm["task"]), samples(rep, b, bm["task"])
        x = kept(rep, a, b, bm["task"], compute) if fa is not None and fb_ is not None else None
        if fa is not None and fb_ is not None and x is None:
            pending.append(bm["label"])
        if x:
            ats.append(x["at"])
        rv = (ref_cells.get(bm["key"]) or {}).get("v")
        missing = "" if fa is not None and fb_ is not None else waiting_words(
            rep, a if fa is None else b, bm["task"])
        out_rows.append({**bm, "a": (cells.get(a) or {}).get(bm["key"]),
                         "b": (cells.get(b) or {}).get(bm["key"]),
                         "ceiling": isinstance(rv, (int, float)) and rv >= training.CEILING,
                         "pair": x, "missing": missing})
    js = [j for j in jobs(rep) if f"|{a}|" in j.slot and f"|{b}|" in j.slot]
    return {"id": rep["id"], "kind": KIND, "title": rep.get("title") or TITLE,
            "what": rep.get("what") or WHAT, "builds": labels(rep), "offered": offer,
            "default": rep.get("default") or [], "a": a, "b": b,
            "colours": {o["label"]: o.get("colour") for o in rep["ours"]},
            "rows": out_rows, "sentence": sentence(a, b, out_rows),
            "settings": settings_line(rep, a, b), "settingsParts": settings_parts(rep, a, b),
            "notes": rep.get("notes") or NOTES, "ceilingWords": training.CEILING_WORDS,
            "pending": pending, "working": {} if compute else rs.working(js),
            "statsAt": reports.stats_at(ats)}


_lines: dict[str, tuple[str, int]] = {}


def waiting_words(rep: dict, label: str, task: str) -> str:
    """24: a build with no scored answers on a benchmark — "waiting for
    grading — 2,098 answers from step8804" when its answers are in and not
    graded yet, else "no answers from step8804" """
    from . import frontier as sf
    from . import report_stats as rs
    from . import reports
    f = sf.task_dir(reports.row_dir(_row(rep, label)), task) / sf.ANSWERS
    if not f.is_file():
        return f"no answers from {label}"
    key = rs.identity(f)
    n = _lines.get(str(f), ("", 0))
    if n[0] != key:
        # the grading page's index counts them (a graded benchmark's), else the lines
        idx = sf.answer_index(reports.row_dir(_row(rep, label)), task, compute=False)
        if idx is None:
            with open(f, "rb") as fh:
                idx = [1 for line in fh if line.strip()]
        n = (key, len(idx))
        _lines[str(f)] = n
    return f"waiting for grading — {n[1]:,} answers from {label}" if n[1] else \
        f"no answers from {label}"


def question_list(rep: dict, a: str, b: str, task: str, side: str) -> dict:
    """one count's questions: those that may be listed, each with how often
    each build ran out of room, and how many are in the hidden half"""
    if side not in ("a", "b"):
        raise KeyError("side is a or b")
    x = kept(rep, a, b, task)
    if x is None:
        raise KeyError("not worked out yet")
    ta = table_of(samples(rep, a, task), task)
    tb = table_of(samples(rep, b, task), task)
    # 24: each question's subject (never its text), and each build's runs —
    # the card opens on the runs that differ
    _scripts()
    import frontier as fb

    from . import ran_out
    cache = {str(it["id"]): it for it in (fb.cached(task, config.BENCH_ROOT) or [])}

    def runs(t: dict) -> list[dict]:
        return [{"epoch": e, "ok": ok, "ran_out": out}
                for e, ok, out in zip(t.get("epochs") or [], t.get("ok") or [], t.get("out") or [])]
    items = []
    for e in x["lists"][side]:
        ids = e.get("grids") or [e["id"]]
        ra = [ta.get(q) or {} for q in ids]
        rb_ = [tb.get(q) or {} for q in ids]
        items.append({"id": e["id"], "grids": e.get("grids"),
                      "subject": subject_of(cache.get(str(ids[0]))),
                      "a_runs": runs(ra[0]) if not e.get("grids") else [],
                      "b_runs": runs(rb_[0]) if not e.get("grids") else [],
                      "a": {"right": sum(sum(r.get("ok") or []) for r in ra),
                            "runs": sum(len(r.get("ok") or []) for r in ra),
                            "ran_out": sum(sum(r.get("out") or []) for r in ra)},
                      "b": {"right": sum(sum(r.get("ok") or []) for r in rb_),
                            "runs": sum(len(r.get("ok") or []) for r in rb_),
                            "ran_out": sum(sum(r.get("out") or []) for r in rb_)}})
    n = x["a_better"] if side == "a" else x["b_better"]
    return {"task": task, "side": side, "n": n, "listed": len(items),
            "hidden": x["held"][side], "items": items, "runs": x.get("runs") or 1,
            "unit": x.get("unit") or "questions", "warning": ran_out.WARNING,
            "withheld": fb.unlisted(task)}


# 24: a question's subject, where this machine's copy has one — never its text
SUBJECT_FIELDS = ("subdomain", "Subdomain", "category", "subject", "topic")


def subject_of(item: dict | None) -> str:
    for k in SUBJECT_FIELDS:
        v = (item or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:80]
    return ""


# ---------------------------------------------------------------------------
# one answer, opened
# ---------------------------------------------------------------------------

def one(rep: dict, label: str, task: str, qid: str, epoch: int) -> dict:
    """one run of one build's answer to a question: its verdict, its thinking
    and its reply — each line quoting the question withheld where the board
    withholds the question — and the question itself where it may be shown.
    A question in the hidden half is refused"""
    _scripts()
    import frontier as fb

    from . import ran_out, reports
    if label not in labels(rep) or task not in {b["task"] for b in rep["benchmarks"]}:
        raise KeyError("no such build or benchmark in this report")
    if not reports.shown_half(task, qid):
        raise Refused(f"question {qid} is in the hidden half: the reader never shows it")
    f = samples(rep, label, task)
    if f is None:
        raise KeyError(f"{label} has no scored answers on this benchmark")
    t = table_of(f, task).get(str(qid))
    if t is None:
        raise KeyError(f"{label} has no answer to question {qid}")
    row = reports.row_at(f, t["at"]) or {}
    runs = row.get("frontier") or []
    k = next((i for i, r in enumerate(runs) if int(r.get("epoch", i) or 0) == int(epoch)), None)
    if k is None:
        raise KeyError(f"{label} has no run {epoch} of question {qid}")
    resps = row.get("resps") or []
    text = (resps[k][0] if k < len(resps) and resps[k] else "") or ""
    items = {str(it["id"]): it for it in (fb.cached(task, config.BENCH_ROOT) or [])}
    it = items.get(str(qid))
    gone = fb.unlisted(task)
    shown = fb.shown(task, it) if it and not gone else None
    clean = (lambda s: ran_out.scrub(s, it)) if gone else (lambda s: s)
    run = {**runs[k], "answer": text}
    thinking, reply = clean(fb.thinking_of(text)), clean(fb.visible(text))
    right = ran_out.right_of(task, it) if it else {}
    # 24: its thinking's tokens (the token stats' character share), and the
    # right answer as the reader may show it: a letter alone where the
    # question is withheld (GPQA), nothing on HLE and MATH
    raw_t, raw_r = fb.thinking_of(text), fb.visible(text)
    toks = runs[k].get("tokens")
    think_toks = round(toks * len(raw_t) / (len(raw_t) + len(raw_r))) if isinstance(
        toks, int) and (raw_t or raw_r) else None
    if gone:
        shown_right = {"letter": right["letter"]} if right.get("letter") else None
    elif shown:
        shown_right = ({"letter": right["letter"], "text": right.get("text") or ""}
                       if right.get("letter") else {"answer": str(shown.get("reference") or "")})
    else:
        shown_right = None
    return {"task": task, "id": str(qid), "epoch": int(epoch), "label": label,
            # 23e: where its thinking first stated the right answer and its
            # reply's final answer, marked; the right option's letter alone
            "marks": {"first": first_mark(task, it, thinking), "final": final_mark(task, reply)},
            "right_letter": right.get("letter") or None, "right": shown_right,
            "read": runs[k].get("read"), "subject": subject_of(it), "think_tokens": think_toks,
            "withheld": gone, "runs": [{"epoch": int(r.get("epoch", i) or 0),
                                        "ok": r.get("ok"), "ran_out": bool(r.get("ran_out"))}
                                       for i, r in enumerate(runs)],
            "question": None if not shown else {"q": shown["q"],
                                                "options": shown.get("options") or []},
            "ok": runs[k].get("ok"), "ran_out": bool(runs[k].get("ran_out")),
            "verdict": fb.verdict(task, run), "tokens": runs[k].get("tokens"),
            "thinking": thinking, "reply": reply}


def first_mark(task: str, item: dict | None, thinking: str) -> list[int] | None:
    """23e: [start, end] of where the thinking first states the right answer
    (ran_out.first_right) — to the end of its sentence or line, 200
    characters at most; None where it never does"""
    from . import ran_out
    at = ran_out.first_right(task, item, thinking) if item and thinking else None
    if at is None:
        return None
    end = at + 1
    while end < len(thinking) and end - at < 200 and thinking[end] not in ".!?\n":
        end += 1
    return [at, min(len(thinking), end + 1)]


_FINAL = {"letter": r"(?i)\banswer\s*[:：]?\s*\**\s*\(?[A-J]\)?(?![A-Za-z0-9])",
          "boxed": r"\\boxed\{", "exact": r"(?i)exact answer\s*:[^\n]*",
          "any": r"(?i)\b(?:final\s+)?answer\s*(?:is|:)[^\n]{0,80}"}


def final_mark(task: str, reply: str) -> list[int] | None:
    """23e: [start, end] of the reply's final answer: its last "ANSWER: X"
    (GPQA, MMLU-Pro), its last \\boxed{…} (MATH, OTIS), "Exact Answer: …"
    (HLE), else its last "answer is …"; None when it has none"""
    if not reply:
        return None
    if task in ("gpqa_diamond_epoch", "mmlupro_tiger"):
        ms = list(re.finditer(_FINAL["letter"], reply))
    elif task in ("math_l5_epoch", "otis_aime_epoch"):
        ms = list(re.finditer(_FINAL["boxed"], reply))
        if ms:
            m = ms[-1]
            depth, j = 1, m.end()
            while j < len(reply) and depth:
                depth += {"{": 1, "}": -1}.get(reply[j], 0)
                j += 1
            return [m.start(), j]
    elif task == "hle_text_cais":
        ms = list(re.finditer(_FINAL["exact"], reply))
    else:
        ms = []
    ms = ms or list(re.finditer(_FINAL["any"], reply))
    return [ms[-1].start(), ms[-1].end()] if ms else None
