"""12o.2: every benchmark's questions, browsed — each with every chosen
model's result on it. The answers are the ones on file: lm_eval logs every
sample of every run (--log_samples), the judge's and the checks' marks sit
beside them (judge.json, generative's reading, safety.json, simpleqa.json,
everyday.json), and a model with no run of a benchmark is "not run".

What may be listed is exactly what may be listed today:
- an lm_eval benchmark lists its diagnose half — the half diagnose.py
  already takes its examples from; the report half is a count and a line;
- the Knowledge exam lists its diagnose half, by the exam's own split;
- Everyday lists its practice half;
- GPQA is never listed, by its authors' request;
- the owner's audit (app.py) opens a report or hidden half, logged.

Nothing here writes anything a writer, Improve, the Playground or chat reads:
it reads the results tree and keeps its tables in this process's memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import threading
from pathlib import Path

from . import config

PAGE = 50
GPQA = re.compile(r"^gpqa", re.I)
HIDDEN_WHY = {
    "lm": ("the report half: never shown, so nothing that trains a model or writes questions "
           "can be taken from it — the diagnosis reads the other half"),
    "exam": "the exam's report half: they score each topic and are never shown",
    "everyday": "hidden: they score it and are not shown",
}
NOT_LISTED = "GPQA Diamond's questions are never shown, as its authors ask"
GGUF_TOTAL = "llama.cpp records only the total"


def _scripts() -> None:
    here = str(Path(__file__).resolve().parent.parent / "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)


# ---------------------------------------------------------------------------
# which benchmark a task is, and what kind of result it has
# ---------------------------------------------------------------------------

GEN = ("ifeval", "mmlu_pro", "hendrycks_math500")
# 12q.B: the devicemark battery's three, each answer beside its row
# (devicemark_items.jsonl) — a served setup's have no lm_eval folder
DM_TASKS = {"dm_ifeval": "ifeval", "dm_mmlu_pro": "mmlu_pro", "dm_math": "math"}
DM_ITEMS = "devicemark_items.jsonl"
# 12o.3, 14.1: MobileAIBench's sets scored with no judge, and MT-Bench (its two
# turns' answers under mab_mtbench_t1 and _t2)
MAB_SCORED = ("mab_hotpotqa", "mab_sql", "mab_dolly", "mab_cnndm", "mab_xsum")
MAB_MTBENCH = "mab_mtbench"
# 14.2: the trust sets, marked by the judge; Privacy Leakage shown as ids and
# verdicts only, never its question or its reply
MAB_TRUST = ("mab_adv", "mab_privacy", "mab_socchem")
MAB_TURNS = ("mab_mtbench_t1", "mab_mtbench_t2")


def kind_of(task: str) -> str:
    """lm (log-likelihood, lm_eval), gen (written, read by generative.py),
    safety, simpleqa, exam, everyday"""
    if task == "everyday":
        return "everyday"
    if task.startswith("exam_") or task == "fr_control_mmlu":
        return "exam"
    if task in ("do_not_answer", "xstest"):
        return "safety"
    if task == "simpleqa_verified":
        return "simpleqa"
    if task in MAB_SCORED:
        return "mab"                        # 12o.3: MobileAIBench's, by its own metrics
    if task == MAB_MTBENCH:
        return "mabj"                       # 14.1: MT-Bench's two turns, rated by the judge
    if task in MAB_TRUST:
        return "mabt"                       # 14.2: a trust set, marked by the judge
    if task in DM_TASKS:
        return "dm"                         # 12q.B: DeviceMark's battery, by its protocol
    if task.startswith(GEN):
        return "gen"
    return "lm"


def _model_id(d: Path) -> str:
    try:
        return json.loads((d / "model_meta.json").read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError):
        return d.name.replace("__", "/", 1)


def _task_dirs(d: Path, task: str) -> list[Path]:
    return [x for x in d.glob(f"{task}_*shot") if re.fullmatch(rf"{re.escape(task)}_\d+shot",
                                                                 x.name)]


def model_dirs(task: str) -> dict[str, Path]:
    """model id -> its folder, for every model with answers to this task"""
    out = {}
    root = Path(config.OUT_DIR)
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if task == "everyday":
            if (d / "everyday.json").exists():
                out[_model_id(d)] = d
        elif task in DM_TASKS:
            if (d / DM_ITEMS).exists():
                out[_dm_model_id(d)] = d
        elif task == MAB_MTBENCH:
            if any(_task_dirs(d, t) for t in MAB_TURNS):
                out[_model_id(d)] = d
        elif _task_dirs(d, task):
            out[_model_id(d)] = d
    return out


def tasks() -> list[str]:
    """every benchmark with an answer on file, GPQA's never"""
    root, seen = Path(config.OUT_DIR), set()
    if root.is_dir():
        for d in root.iterdir():
            for t in d.glob("*_*shot") if d.is_dir() else []:
                m = re.fullmatch(r"(.+)_\d+shot", t.name)
                # the Everyday pilot's legacy task is the Everyday bank's, not a benchmark
                if m and not GPQA.match(m.group(1)) and not m.group(1).startswith(
                        "everyday_"):
                    seen.add(m.group(1))
            if (d / "everyday.json").exists():
                seen.add("everyday")
            if d.is_dir() and (d / DM_ITEMS).exists():
                seen.update(DM_TASKS)
    # 14.1: MT-Bench's two turns are one benchmark
    if seen & set(MAB_TURNS):
        seen = (seen - set(MAB_TURNS)) | {MAB_MTBENCH}
    return sorted(seen)


def _dm_model_id(d: Path) -> str:
    """a devicemark row: its model, " · thinking" for a thinking-on row"""
    base = d.name.removesuffix("__thinking")
    mid = _model_id(d.parent / base) if (d.parent / base).is_dir() else base.replace("__", "/", 1)
    return mid + (" · thinking" if d.name.endswith("__thinking") else "")


_dm_q: dict = {}


def _dm_questions() -> dict:
    """the battery's questions as the server read them (DM_ITEMS), by key —
    nothing is fetched for the browser"""
    f = Path(config.DM_ITEMS)
    try:
        st = f.stat()
    except OSError:
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    if _dm_q.get("stamp") != stamp:
        rows = {}
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            if line.strip():
                r = json.loads(line)
                rows[(r["bench"], r["key"])] = r
        _dm_q.update(stamp=stamp, rows=rows)
    return _dm_q["rows"]


def _dm_rows(task: str, d: Path) -> dict[str, dict]:
    """a devicemark row's answers to one bench: each with its verdict in words"""
    _scripts()
    import devicemark as dm
    bench, qs, out = DM_TASKS[task], _dm_questions(), {}
    for i, it in enumerate(dm.read_items(d)):
        if it.get("bench") != bench:
            continue
        q = qs.get((bench, it["key"])) or {}
        think, answer = dm.split_thinking(it.get("text") or "")
        if it.get("answer") is not None:
            answer = it["answer"]
        if bench == "ifeval":
            v = it.get("ifeval") or {}
            st, lo = (v.get("strict") or {}).get("inst") or [], (v.get("loose") or {}).get(
                "inst") or []
            verdict = (f"strict: {sum(st)} of {len(st)} instructions followed · loose: "
                       f"{sum(lo)} of {len(lo)}")
        elif it.get("error"):
            # 12q.F: the server failed on it twice, and again without its chat parsing
            verdict = "no answer: the server failed on this item"
        elif not it.get("answered"):
            verdict = ("no answer within the cap" if it.get("capped") else
                       "no answer: no box" + (" or tested phrasing" if bench == "mmlu_pro"
                                              else ""))
        else:
            how = it.get("how")
            verdict = (f"read as {str(it.get('parsed'))[:60]}"
                       + (f" (from \"{how}\")" if how and how != "boxed" else "")
                       + f" · the answer is {str(it.get('gold'))[:60]}")
        tok = it.get("gen_tokens")
        if tok is not None:
            verdict += f" · {tok:,} tokens" + (", capped" if it.get("capped") else "")
        subj = q.get("category") or q.get("subject") or (
            it["key"].split("/")[1] if bench == "math" and "/" in it["key"] else "")
        out[it["key"]] = {
            "q": q.get("prompt") or q.get("question") or q.get("problem") or "",
            "options": q.get("options") or [] if bench == "mmlu_pro" else [],
            "subject": subj, "reference": q.get("answer") if bench != "ifeval" else None,
            "order": [bench, i],
            "res": {"ok": bool(it.get("correct")), "answer": answer, "thinking": think,
                    "verdict": verdict,
                    # 12q.C: and what the model page's Answers tab shows beside it
                    "parsed": it.get("parsed"), "answered": bool(it.get("answered")),
                    "capped": bool(it.get("capped")), "tokens": tok,
                    # 12q.F: answered without the server's chat parsing (its error), or not
                    # at all after that too (both errors)
                    "fallback": (it.get("raw_fallback") or {}).get("error"),
                    "error": it.get("error")}}
    return out


# ---------------------------------------------------------------------------
# one model's results on one task, by question
# ---------------------------------------------------------------------------

_SUB = re.compile(r"samples_(.+?)_\d{4}-\d{2}-\d{2}T")


def _samples(d: Path, task: str) -> list[tuple[str, dict]]:
    """(subtask, record) for the newest file of each subtask"""
    _scripts()
    import diagnose as dx
    files = dx.newest_per_subtask(sorted(f for x in _task_dirs(d, task)
                                         for f in x.rglob("samples_*.jsonl")))
    out = []
    for f in files:
        m = _SUB.match(f.name)
        sub = m.group(1) if m else task
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    out.append((sub, json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return out


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


def _options(doc: dict, rec: dict) -> list[str]:
    for key in ("choices", "endings", "options"):
        v = doc.get(key)
        if isinstance(v, dict):
            v = v.get("text")
        if isinstance(v, list) and v and all(isinstance(s, (str, int, float)) for s in v):
            return [str(s) for s in v]
    if doc.get("option1") is not None and doc.get("option2") is not None:
        return [str(doc["option1"]), str(doc["option2"])]
    for key in ("mc1_targets", "mc2_targets"):
        v = doc.get(key)
        if isinstance(v, dict) and isinstance(v.get("choices"), list):
            return [str(s) for s in v["choices"]]
    if all(f"ans{i}" in doc for i in range(3)):
        return [str(doc[f"ans{i}"]) for i in range(3)]
    args = rec.get("arguments") or []
    conts = []
    for a in args:
        if isinstance(a, dict):
            a = [a.get("arg_0"), a.get("arg_1")]
        if isinstance(a, list) and len(a) >= 2 and isinstance(a[1], str):
            conts.append(a[1].strip())
    return conts if len(conts) >= 2 else []


def _question(doc: dict, rec: dict) -> str:
    for key in ("question", "query", "prompt", "problem", "sentence", "ctx", "goal", "input",
                "text", "context"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    args = rec.get("arguments") or []
    a = args[0] if args else None
    if isinstance(a, dict):
        a = [a.get("arg_0")]
    ctx = a[0] if isinstance(a, list) and a and isinstance(a[0], str) else ""
    return ctx.strip().rsplit("\n\n", 1)[-1][:2000]


def _subject(doc: dict, sub: str, task: str) -> str:
    for key in ("subject", "category", "risk_area", "types_of_harm", "type", "src"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().replace("_", " ")
    if sub != task and sub.startswith(task + "_"):
        return sub[len(task) + 1:].replace("_", " ")
    return ""


def _lm_result(rec: dict, n_opts: int) -> dict:
    """the option it picked, whether that is right, and how sure it was: the
    margin between its two likeliest options"""
    _scripts()
    import diagnose as dx
    pm = dx.primary_metric(rec)
    out: dict = {"ok": bool(pm and pm[1] >= 0.5)}
    lps = []
    for r in rec.get("filtered_resps") or rec.get("resps") or []:
        if isinstance(r, list) and r and isinstance(r[0], list):
            r = r[0]
        lps.append(_f(r[0] if isinstance(r, list) else r))
    if len(lps) >= 2:
        if pm and pm[0] == "acc_norm":
            L = dx.norm_lengths(rec.get("doc") or {}, len(lps))
            if L:
                lps = [x / l for x, l in zip(lps, L)]
        probs = dx.softmax(lps)
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        out.update(pick=order[0], margin=round(probs[order[0]] - probs[order[1]], 4))
    return out


def _answer(rec: dict) -> dict:
    _scripts()
    import judge
    p = judge.answer_parts(rec)
    return {"answer": p["answer_text"], "thinking": p.get("reasoning_text") or "",
            "no_answer": bool(p["no_answer"])}


def _trust_rows(task: str, d: Path) -> dict[str, dict]:
    """14.2: a trust set's replies, marked — "judged correct", "kept it private"
    or "agrees with the crowd", or awaiting judge. Privacy Leakage: the id and
    the verdict only — never the person's name, never the reply"""
    import mobileaibench as mab
    private = task == mab.PRIVACY
    qs = {q["id"]: q for q in mab.load(task)} if not mab.available(task) else {}
    items = (((mab.read(d) or {}).get("tasks") or {}).get(task) or {}).get("items") or []
    words = {mab.ADV: ("judged correct", "judged incorrect"),
             mab.PRIVACY: ("kept it private", "revealed it"),
             mab.SOCCHEM: ("agrees with the crowd", "disagrees with the crowd")}[task]
    out = {}
    for i, it in enumerate(items):
        q = qs.get(it["id"]) or {}
        s = it.get("score")
        verdict = ("the answer to the original: what its rewordings are marked against"
                   if it.get("original") else mab.AWAITING if s is None else words[0] if s == 1
                   else words[1] if s == 0 else "the judge's reply was neither")
        out[it["id"]] = {
            "q": it["id"] if private else q.get("prompt") or it["id"], "options": [],
            "subject": "" if private else q.get("topic") or q.get("answer") or "",
            "context": None, "reference": None if private else q.get("answer") if task == mab.SOCCHEM
            else None, "order": [task, i], "private": private,
            "res": {"ok": None if s is None or s == 0.5 else s == 1,
                    "answer": "" if private else it.get("answer_text") or "", "thinking": "",
                    "verdict": verdict}}
    return out


def _mtbench_rows(d: Path) -> dict[str, dict]:
    """14.1: MT-Bench, a row a turn: the turn's question (the second with the
    first beside it), the answer, and the judge's rating — or "awaiting
    judge"; GPT-4's reference for maths, reasoning and coding"""
    import mobileaibench as mab
    qs = {q["id"]: q for q in mab.load(mab.MTB1)}
    items = (((mab.read(d) or {}).get("tasks") or {}).get(mab.MTBENCH) or {}).get("items") or []
    out = {}
    for it in items:
        q = qs.get(it["id"])
        if not q:
            continue
        t = it["turn"]
        s = it.get("score")
        ref = q.get("reference") or []
        out[f"{it['id']}:{t}"] = {
            "q": q["turns"][t - 1], "options": [], "subject": f"{q['category']} · turn {t}",
            "context": q["turns"][0] if t == 2 else None,
            "reference": ref[t - 1] if len(ref) >= t else None,
            "order": [q["category"], q["question_id"] * 2 + t],
            "res": {"ok": None if s is None or s == -1 else s >= 6,
                    "answer": it.get("answer_text") or "", "thinking": "",
                    "verdict": (mab.AWAITING if s is None else "the judge's reply had no rating"
                                if s == -1 else f"rated {s:g} of 10"
                                + (f" by {it['judge']}" if it.get("judge") else ""))}}
    return out


def _rows_of(task: str, d: Path) -> dict[str, dict]:
    """this model's results on the task, by question key, each with the
    question as its record carries it"""
    _scripts()
    import diagnose as dx
    kind, out = kind_of(task), {}
    if kind == "dm":
        return _dm_rows(task, d)
    if kind == "mabj":
        return _mtbench_rows(d)
    if kind == "mabt":
        return _trust_rows(task, d)
    if kind == "everyday":
        import everyday as ev
        e = ev.read(d) or {}
        for it in e.get("items") or []:
            out[it["id"]] = {"res": {"ok": it.get("pass"), "answer": it.get("answer_text") or "",
                                     "thinking": it.get("reasoning_text") or "",
                                     "verdict": it.get("reason") or ""}}
        return out
    marks = {}
    if kind == "safety":
        import trust_safety as ts
        marks = {it["id"]: it for it in (ts.read(d) or {}).get("items") or [] if it["task"] == task}
    elif kind == "simpleqa":
        import simpleqa as sq
        marks = {it["id"]: it for it in (sq.read(d) or {}).get("items") or []}
    elif kind == "mab":
        import mobileaibench as mab
        marks = {it["id"]: it for it in (((mab.read(d) or {}).get("tasks") or {}).get(task)
                                        or {}).get("items") or []}
        gold = {q["id"]: q for q in mab.load(task)}
    elif kind == "exam":
        try:
            j = json.loads((d / "judge.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            j = {}
        marks = {it.get("doc_hash"): it for it in ((j.get("tasks") or {}).get(task) or {})
                 .get("items") or []}
    for sub, rec in _samples(d, task):
        doc = rec.get("doc") or {}
        # lm_eval's own id for a question is its doc's hash; a record without
        # one is keyed the same way
        key = str(rec.get("doc_hash") or rec.get("prompt_hash") or (hashlib.sha256(json.dumps(
            doc, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest() if doc else ""))
        if not key:
            continue
        opts = _options(doc, rec) if kind == "lm" else []
        row = {"q": _question(doc, rec), "options": opts, "subject": _subject(doc, sub, task),
               "gkey": _gkey(task, doc) if kind == "lm" else None,
               "order": [sub, rec.get("doc_id") if isinstance(rec.get("doc_id"), int) else 0]}
        if kind == "lm":
            # TruthfulQA MC2: several true answers, scored on the mass on them —
            # every true one is marked, and each model's mass is its result
            labels = (doc.get("mc2_targets") or {}).get("labels") if isinstance(
                doc.get("mc2_targets"), dict) else None
            if isinstance(labels, list) and sum(1 for x in labels if x) > 1:
                row["answer_idx"] = [i for i, x in enumerate(labels) if x]
                pm = dx.primary_metric(rec)
                row["res"] = {"ok": bool(pm and pm[1] >= 0.5),
                              "mass": round(pm[1], 4) if pm else None}
            else:
                row["answer_idx"] = dx.target_index(rec, len(opts) or 4)
                row["res"] = _lm_result(rec, len(opts))
        elif kind == "gen":
            import generative as gen
            got = gen.read(sub if sub.startswith(GEN) else task, rec)
            a = _answer(rec)
            if got["ran_out"]:
                verdict = "ran out of room while thinking"
            elif task == "ifeval":
                verdict = ("every instruction followed" if got["correct"]
                           else "an instruction not followed")
            else:
                verdict = (f"read as {str(got['extracted'] or 'nothing')[:60]} · the answer is "
                           f"{str(got['key'])[:60]}")
            row["res"] = {"ok": bool(got["correct"]), "answer": a["answer"],
                          "thinking": a["thinking"], "verdict": verdict}
            row["reference"] = got.get("key") if task != "ifeval" else None
        else:
            a = _answer(rec)
            m = marks.get(doc.get("id")) if kind in ("safety", "simpleqa", "mab") \
                else marks.get(key)
            if kind == "mab":
                q = gold.get(doc.get("id")) or {}
                row["q"] = q.get("question") or row["q"]
                # the CREATE statement, or HotpotQA's passages: folded on the page
                row["context"] = q.get("context") if task == "mab_sql" else doc.get(
                    "prompt", "").split("\nquestion: ", 1)[0].removeprefix("context: ")
                row["reference"] = q.get("answer")
                if task in ("mab_cnndm", "mab_xsum"):
                    # the article: folded on the page; the question is the ask
                    row["q"] = "Summarise the article."
                    row["context"] = doc.get("prompt", "").removeprefix(
                        "Create a short summary of the following article: ")
                elif task == "mab_dolly":
                    row["context"] = doc.get("prompt", "").split("\nquestion: ", 1)[0] \
                        .removeprefix("context: ") if doc.get("prompt", "").startswith(
                            "context: ") else None
                if not m:
                    ok, verdict = None, "not scored yet"
                elif task in ("mab_hotpotqa", "mab_dolly"):
                    ok = m["f1"] >= 0.5
                    verdict = f"F1 {m['f1']:.2f} · EM {m['em']} · BLEU {m['bleu']:.2f}"
                elif task in ("mab_cnndm", "mab_xsum"):
                    ok = None                       # overlap with one reference is no verdict
                    verdict = f"ROUGE-L {m['rougeL']:.2f} · ROUGE-1 {m['rouge1']:.2f}"
                else:
                    ok = m["sqlparser"] >= 0.5
                    verdict = (f"SQLParser F1 {m['sqlparser']:.2f} · Levenshtein "
                               f"{m['levenshtein']:.2f} · exact {'✓' if m['exact'] else '✗'}"
                               f" · SQL from {'no SQL found' if m['how'] == 'none' else m['how']}")
            elif kind == "safety":
                ok = None if not m or m.get("score") is None else m["score"] == 2
                verdict = (m or {}).get("reason") or "not marked yet"
            elif kind == "simpleqa":
                g = (m or {}).get("grade")
                ok = None if g is None else g == "correct"
                verdict = {"correct": "correct", "incorrect": "incorrect",
                           "not_attempted": "not attempted"}.get(g, "not graded yet")
            else:
                s = (m or {}).get("score")
                ok = None if s is None else s >= 3
                verdict = f"{s} of 4 from the judge" if s is not None else "not judged"
                row["qid"] = doc.get("qid")
                row["reference"] = doc.get("reference")
            row["res"] = {"ok": ok, "answer": a["answer"], "thinking": a["thinking"],
                          "verdict": verdict}
        out[key] = row
    return out


# ---------------------------------------------------------------------------
# the table: every question on file, each model's result — cached by the
# files it was read from
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict[tuple, dict] = {}


def _stamp(task: str, dirs: dict[str, Path]) -> tuple:
    """12a.8: each file's time to the nanosecond and size, not the newest time
    alone — a file dated in the future hid every later write"""
    out = []
    for mid, d in sorted(dirs.items()):
        files = ([d / "everyday.json"] if task == "everyday" else
                 [d / DM_ITEMS, Path(config.DM_ITEMS)] if task in DM_TASKS else
                 [f for t in (MAB_TURNS if task == MAB_MTBENCH else (task,))
                  for x in _task_dirs(d, t) for f in x.rglob("samples_*.jsonl")]
                 + [d / n for n in ("judge.json", "safety.json", "simpleqa.json",
                                    "generative.json", "mobileaibench.json")])
        out.append((mid, tuple(sorted((str(f), f.stat().st_mtime_ns, f.stat().st_size)
                                      for f in files if f.exists()))))
    return tuple(out)


def half_of(task: str, key: str, row: dict) -> str:
    """"diagnose" when it may be listed, else "report" """
    _scripts()
    kind = kind_of(task)
    if kind == "everyday":
        import everyday as ev
        return "diagnose" if row.get("half") == ev.PRACTICE else "report"
    if kind == "exam":
        if task == "fr_control_mmlu":
            return "diagnose"
        import exam_build as eb
        return eb.half_of(row["qid"]) if row.get("qid") else "report"
    import diagnose as dx
    return dx.split_of(key)


def table(task: str) -> dict:
    """{models: [ids], rows: {key: row with results by model and its half}}"""
    if GPQA.match(task):
        raise PermissionError(NOT_LISTED)
    dirs = model_dirs(task)
    stamp = (task, _stamp(task, dirs))
    with _lock:
        if stamp in _cache:
            return _cache[stamp]
    rows: dict[str, dict] = {}
    if task == "everyday":
        _scripts()
        import everyday as ev
        for q in ev.load_bank():
            rows[q["id"]] = {"q": q["prompt"], "options": [], "subject": ev.groups().get(
                q["group"], q["group"]), "reference": q.get("reference") or "",
                             "half": ev.half(q), "results": {}, "order": [q["group"], q["id"]]}
    for mid, d in dirs.items():
        for key, r in _rows_of(task, d).items():
            row = rows.setdefault(key, {k: v for k, v in r.items() if k != "res"})
            if task != "everyday":
                for k, v in r.items():
                    if k != "res" and not row.get(k):
                        row[k] = v
            row.setdefault("results", {})[mid] = r["res"]
    for key, row in rows.items():
        row["half"] = half_of(task, key, row)
    got = {"models": sorted(dirs), "rows": rows}
    with _lock:
        for k in [k for k in _cache if k[0] == task]:
            del _cache[k]
        _cache[stamp] = got
    return got


# ---------------------------------------------------------------------------
# where the questions come from
# ---------------------------------------------------------------------------

def meta(task: str) -> dict:
    """the source, licence and revision the repo records for a benchmark; for
    one lm_eval loads, the dataset and revision its newest run names (the
    licence is the dataset's card's)"""
    _scripts()
    kind = kind_of(task)
    if kind == "everyday":
        return {"source": "written for this board (eval_tasks/everyday)", "licence": None,
                "revision": None, "url": None}
    if kind == "exam":
        return {"source": "the Knowledge exam, written for this board", "licence": None,
                "revision": None, "url": None}
    if kind == "dm":
        import devicemark as dm
        bench = DM_TASKS[task]
        name, _ = dm.DATASETS[bench]
        bat = dm.battery()
        return {"source": f"{name}: DeviceMark's protocol, {bat['version']} — "
                          + ("DeviceMark's 300 items" if bench == "ifeval" else
                             "our draw of their design"),
                "licence": bat["sources"][bench]["license"],
                "revision": bat["revisions"][name], "url": f"https://huggingface.co/datasets/{name}"}
    if kind in ("mab", "mabj", "mabt"):
        import mobileaibench as mab
        m = mab.manifest()
        src, by = m["sources"][mab.SOURCE[task]], m["sampled_by"]
        if kind == "mabj":
            fc = m["fastchat"]
            return {"source": f"{src['name']} ({src['cite']}), as MobileAIBench runs it; judge "
                              f"prompts and GPT-4's reference answers from {fc['name']}",
                    "licence": f"{src['licence']}; MobileAIBench {by['licence']}",
                    "revision": fc["revision"], "url": fc["url"]}
        n = m["files"][mab.SOURCE[task]]["n"]
        return {"source": f"{src['name']} ({src['cite']}), MobileAIBench's {n:,}-row sample",
                "licence": f"{src['licence']}; the sample {by['licence']}",
                "revision": by["revision"], "url": by["url"]}
    if kind == "simpleqa":
        import simpleqa as sq
        c = sq.credit()
        return {"source": c["name"], "licence": c["licence"], "revision": c["revision"],
                "url": c["url"]}
    if kind == "safety" or task.startswith("bbq"):
        import trust_safety as ts
        src = ts.manifest()["sources"].get("bbq" if task.startswith("bbq") else task)
        if src:
            return {"source": src["name"], "licence": src["licence"],
                    "revision": src["revision"], "url": src["url"]}
    newest, cfg = 0.0, {}
    for d in model_dirs(task).values():
        for x in _task_dirs(d, task):
            for f in x.rglob("results_*.json"):
                if f.stat().st_mtime <= newest:
                    continue
                try:
                    blob = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                cs = blob.get("configs") or {}
                c = cs.get(task) or next((v for k, v in sorted(cs.items())
                                          if k.startswith(task)), None)
                if isinstance(c, dict):
                    newest, cfg = f.stat().st_mtime, {**c, "_lm_eval": blob.get(
                        "lm_eval_version") or (blob.get("config") or {}).get("lm_eval_version")}
    path = cfg.get("dataset_path")
    rev = (cfg.get("dataset_kwargs") or {}).get("revision")
    return {"source": (f"{path}" + (f" ({cfg['dataset_name']})" if cfg.get("dataset_name")
                                    else "") + (f", as lm_eval {cfg['_lm_eval']} loads it"
                                                if cfg.get("_lm_eval") else ""))
            if path else None,
            "licence": None, "revision": rev,
            "url": f"https://huggingface.co/datasets/{path}" if path and "/" in path
            and not path.startswith(("/", ".")) else None}


# ---------------------------------------------------------------------------
# a page of it
# ---------------------------------------------------------------------------

def _keep(row: dict, models: list[str], f: str) -> bool:
    res = [row["results"].get(m) for m in models]
    ran = [(m, r) for m, r in zip(models, res) if r is not None and r.get("ok") is not None]
    if f == "disagree":
        return len({bool(r["ok"]) for _, r in ran}) > 1
    if f == "allwrong":
        return bool(ran) and not any(r["ok"] for _, r in ran)
    m = re.fullmatch(r"only(right|wrong):(.+)", f or "")
    if m:
        want, who = m.group(1) == "right", m.group(2)
        mine = next((r for x, r in ran if x == who), None)
        if mine is None or bool(mine["ok"]) != want:
            return False
        others = [r for x, r in ran if x != who]
        return bool(others) and all(bool(r["ok"]) != want for r in others)
    return True


def page(task: str, *, offset: int = 0, limit: int = PAGE, q: str = "", subject: str = "",
         models: list[str] | None = None, f: str = "", half: str = "diagnose") -> dict:
    """one page of the task's questions in `half` — the diagnose half, unless
    the owner's audit asks for the other"""
    t = table(task)
    chosen = [m for m in (models or t["models"]) if m in t["models"]] if models else t["models"]
    shown = {m for m in (models or t["models"])}
    listed = [(k, r) for k, r in sorted(t["rows"].items(), key=lambda kv: (
        str((kv[1].get("order") or ["", 0])[0]), (kv[1].get("order") or ["", 0])[1], kv[0]))
        if r["half"] == half]
    other = sum(1 for r in t["rows"].values() if r["half"] != half)
    subjects = sorted({r.get("subject") or "" for _, r in listed} - {""})
    ql = q.strip().lower()
    hit = [(k, r) for k, r in listed
           if (not ql or ql in (r["q"] or "").lower()
               or any(ql in str(o).lower() for o in r.get("options") or []))
           and (not subject or r.get("subject") == subject)
           and (not f or _keep(r, chosen, f))]
    limit = max(1, min(int(limit or PAGE), 200))
    offset = max(0, int(offset or 0))
    out_rows = []
    gg = gguf_results(task)
    for k, r in hit[offset:offset + limit]:
        out_rows.append({"id": k, "q": r["q"], "options": r.get("options") or [],
                         "answer_idx": r.get("answer_idx"), "subject": r.get("subject") or "",
                         "reference": r.get("reference") or None,
                         "context": r.get("context") or None,
                         "results": {m: r["results"].get(m) for m in shown},
                         # 12o.2: the GGUF's, where llama.cpp says which question it was
                         **({"gguf": {m: res.get(r.get("gkey")) for m, res in
                                      gg["models"].items()}} if gg.get("models") else {})})
    return {"task": task, "kind": kind_of(task), "offset": offset, "limit": limit,
            "total": len(hit), "listed": len(listed), "other": other, "half": half,
            "hidden_why": HIDDEN_WHY["everyday" if task == "everyday" else "exam"
                                     if kind_of(task) == "exam" else "lm"],
            "subjects": subjects, "models": t["models"], "shown": sorted(shown),
            "rows": out_rows, "meta": meta(task),
            "gguf": ({"benchmark": gg["benchmark"], "line": gg["line"],
                      "models": sorted(gg["models"])} if gg else None)}


# ---------------------------------------------------------------------------
# the GGUF's results, question by question where llama.cpp says them
# ---------------------------------------------------------------------------

def _gguf_key(task: str) -> str | None:
    _scripts()
    try:
        import gguf_bench as gb
    except ImportError:
        return None
    key = next((k for k, v in gb.BENCHMARKS.items() if v.get("lm_eval") == task), None)
    return None if not key or GPQA.match(key) else key


def _gkey(task: str, doc: dict) -> str | None:
    """a question as llama-perplexity's file holds it: gguf_data's own
    builders, so a question in lm_eval's samples finds itself in the file"""
    key = _gguf_key(task)
    if not key or not doc:
        return None
    import gguf_data as gd
    try:
        if key == "winogrande":
            return gd._flat(doc["sentence"]) if doc.get("sentence") else None
        t = gd.MC[key](doc) if key in gd.MC else None
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None
    return t["question"] if t else None


# tools/perplexity (upstream llama.cpp): multiple_choice_score prints
# "<n>\t<running accuracy %>" after every task, winogrande_score
# "<n>\t<running %>\t<score 1>  <score 2>  <pick>  <answer>". A full run takes
# the tasks in the file's order; a subset is a random draw, and HellaSwag is
# always shuffled — which question a line is can't be told then
_MC_LINE = re.compile(r"^(\d+)\t([\d.]+)\s*$")
_WG_LINE = re.compile(r"^(\d+)\t([\d.]+)\t\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d)\s+(\d)\s*$")


def _file_tasks(key: str) -> tuple[list[str], str] | None:
    """the file's questions, in its order, and its sha256"""
    import gguf_bench as gb
    import gguf_data as gd
    root = Path(config.RESULTS_ROOT) / "gguf_data"
    try:
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["benchmarks"][key]
        data = (root / man["file"]).read_bytes()
    except (OSError, ValueError, KeyError):
        return None
    if key == "winogrande":
        return [gd._flat(r["sentence"]) for r in gd.read_winogrande(data.decode("utf-8"))], \
            man.get("sha256", "")
    return [t["question"] for t in gb.read_mc(data)], man.get("sha256", "")


def _per_task(key: str, text: str, n: int) -> list[dict] | None:
    """each task's result from a run's log, or None when it isn't one line a
    task for all n"""
    out, right = [], 0
    for line in text.splitlines():
        if key == "winogrande":
            m = _WG_LINE.match(line)
            if not m:
                continue
            s1, s2, pick, ans = float(m.group(3)), float(m.group(4)), int(m.group(5)), \
                int(m.group(6))
            p1 = 1 / (1 + math.exp(max(-50.0, min(50.0, s2 - s1))))
            out.append({"ok": pick == ans, "pick": pick - 1, "margin": round(abs(2 * p1 - 1), 4)})
        else:
            m = _MC_LINE.match(line)
            if not m:
                continue
            i = int(m.group(1))
            c = round(float(m.group(2)) * i / 100)
            out.append({"ok": c > right})
            right = c
        if int(m.group(1)) != len(out):
            return None
    return out if len(out) == n else None


_gcache: dict[tuple, dict] = {}


def gguf_results(task: str) -> dict:
    """{"benchmark", "line", "models": {id: {question: result}}} — each GGUF's
    newest full run on today's file, question by question where its log says
    them; "llama.cpp records only the total" where none does. Read again only
    when a run or the file has changed"""
    key = _gguf_key(task)
    if not key:
        return {}
    root = Path(config.RESULTS_ROOT)
    files = [root / "gguf_data" / "manifest.json"] + (
        sorted((root / "gguf_results").glob("*")) if (root / "gguf_results").is_dir() else [])
    stamp = (task, tuple((f.name, f.stat().st_mtime) for f in files if f.exists()))
    with _lock:
        if stamp in _gcache:
            return _gcache[stamp]
    got = _gguf_results(task, key)
    with _lock:
        for k in [k for k in _gcache if k[0] == task]:
            del _gcache[k]
        _gcache[stamp] = got
    return got


def _gguf_results(task: str, key: str) -> dict:
    out = {"benchmark": key, "line": GGUF_TOTAL, "models": {}}
    got = _file_tasks(key) if key != "hellaswag" else None
    if not got:
        return out
    texts, sha = got
    rdir = Path(config.RESULTS_ROOT) / "gguf_results"
    newest: dict[str, float] = {}
    for f in sorted(rdir.glob("*.json")) if rdir.is_dir() else []:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        b = (r.get("benchmarks") or {}).get(key) or {}
        if b.get("status") != "done" or r.get("subset") \
                or ((r.get("datasets") or {}).get(key) or {}).get("sha256") != sha:
            continue
        setup = r.get("setup") or {}
        mid = r.get("model", "") + ("" if setup.get("id", "as-built") == "as-built"
                                    else f" · {setup.get('name')}")
        at = r.get("finished_at") or 0
        if not mid or at <= newest.get(mid, -1):
            continue
        try:
            log = (rdir / f"{r.get('id')}.{key}.log").read_text(encoding="utf-8",
                                                                  errors="replace")
        except OSError:
            continue
        per = _per_task(key, log, len(texts))
        if per:
            newest[mid] = at
            out["models"][mid] = dict(zip(texts, per))
    if out["models"]:
        out["line"] = None
    return out


def margin_words(m: float | None) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return ""
    return f"margin {m:.2f}"
