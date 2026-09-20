#!/usr/bin/env python3
"""The whole loop against the local model, narrated — one command, nine steps.

    python scripts/demo_loop.py --topics economics,law --model EleutherAI/pythia-160m

The point is comprehension, not coverage. Each step says what it is about to
do, what came back, and where it landed, so a person can watch the loop turn
instead of reading about it:

    1 preflight      vLLM reachable, what it serves, the three identities
    2 draft          an LLM writes candidate questions per topic
    3 accept         a person curates — here, --auto-accept, recorded as `demo`
      (--import <json> replaces 2 and 3 with a human-written bank, whose
       author is the approver on every item)
    4 build          the harness tasks, split report/diagnose by qid
    5 sit            the model answers, on the GPU, behind the shared lock
    6 judge          an LLM grades every answer 0-4 and says why
    7 propose        the weakest topic's judged words become a skill spec
    8 generate       the spec becomes training documents, through the gate
    9 summary        every path written, and what would happen next

Two properties are DEMONSTRATED rather than asserted in prose: step 7 checks
the request body that went to the LLM for any exam question text, both halves,
and says so; and everything a local model produced is printed with its
provisional stamp, because a local server's model id cannot be pinned.

Everything this writes lands under $BENCH_ROOT/demo/ — its own exam bank, its
own results tree, its own database — and is removed at the end unless --keep.
It never touches the live exam bank or the live board. It does take the same
GPU lock as everything else, because it runs a real evaluation.

What a green run here does NOT prove: the Anthropic and OpenAI batch clients
are not exercised by it. See DEMO.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import categories as _categories  # noqa: E402
import exam_build as eb  # noqa: E402
import judge as jd  # noqa: E402

WIDTH = 78
APPROVER = "demo"          # never a curator's name: nobody should mistake this for curation
# an instruct-tuned model, by the only signal a model id gives
INSTRUCT = re.compile(r"instruct|-it\b|chat|sft|tulu|zephyr", re.I)
_N = 0


# ---------------------------------------------------------------------------
# narration
# ---------------------------------------------------------------------------

def hr(title: str) -> None:
    global _N
    _N += 1
    print(f"\n{'─' * WIDTH}\n{_N}. {title}\n{'─' * WIDTH}")


def say(*lines: str) -> None:
    for x in lines:
        print(f"   {x}")


def kv(key: str, value) -> None:
    print(f"   {key:<22}{value}")


def quote(text: str, indent: str = "     | ") -> None:
    for line in str(text).splitlines() or [""]:
        print(indent + line)


def die(*lines: str) -> int:
    print("\nSTOPPED")
    for x in lines:
        print(f"   {x}")
    return 2


# ---------------------------------------------------------------------------
# the demo's own corner of BENCH_ROOT
# ---------------------------------------------------------------------------

def configure(root: Path, create: bool = True):
    """Point every writable path at the demo root. RESULTS_ROOT is left alone
    on purpose: the GPU lock lives there, and the demo must take the same one
    as the service and the CLI rather than inventing its own."""
    from service import config, llm
    # BENCH_ROOT moves too: it is where a backend keeps its batches on disk
    # (llm_batches/local/<id>) and where the runner puts a job's scratch, and
    # all of that should be inside the tree --keep keeps and cleanup removes.
    # RESULTS_ROOT is NOT moved — the shared GPU lock lives under it.
    config.BENCH_ROOT = root
    config.EXAM_DIR = root / "exam"
    config.JUDGED_TASKS_DIR = root / "exam" / "tasks"
    config.OUT_DIR = root / "results" / "full"
    config.DATASETS_DIR = root / "datasets"
    config.LOGS_DIR = root / "logs"
    config.DB_PATH = root / "service.sqlite3"
    if create:
        for d in (config.EXAM_DIR, config.OUT_DIR, config.DATASETS_DIR, config.LOGS_DIR):
            d.mkdir(parents=True, exist_ok=True)
    llm.reset()
    return config


def stamp_of(backend) -> str:
    from service import llm
    mark = llm.provisional(backend, "made")
    if not mark:
        return ""
    return (f"PROVISIONAL — {mark['served_model']} at {mark['base_url']}"
            + (f" (weights {mark['weights']})" if mark.get("weights") else ""))


# ---------------------------------------------------------------------------
# 1 · preflight
# ---------------------------------------------------------------------------

def preflight(a, ctx) -> str:
    from service import config, llm
    hr("Preflight — what this run will use")
    kv("BENCH_ROOT", ctx["bench_root"])
    kv("demo root", ctx["root"])
    kv("exam bank", config.EXAM_DIR)
    kv("results", config.OUT_DIR)
    kv("datasets", config.DATASETS_DIR)
    kv("database", config.DB_PATH)
    kv("topics", ", ".join(ctx["topics"]))
    kv("model to sit the exam", a.model)
    if a.import_path:
        kv("exam bank from", f"{a.import_path}  (imported, not drafted)")
    print()
    # an imported bank needs no exam writer: nobody is drafting anything
    roles = ([("judge", "judge"), ("llm", "generator")] if a.import_path else
             [("exam", "exam writer"), ("judge", "judge"), ("llm", "generator")])
    blocked = {}
    for role, label in roles:
        p, m, _ = llm.identity(role)
        why = jd.blocked() if role == "judge" else llm.blocked(role)
        blocked[role] = why
        kv(label, (f"{p}/{m}" if p else "(unset)") + ("" if not why else f"  ← {why}"))
    if any(blocked.values()):
        return die(f"One identity is not usable, and the loop needs "
                   f"{'both of these' if a.import_path else 'all three'}.",
                   *[f"{r}: {w}" for r, w in blocked.items() if w],
                   "", "For an all-local trial, put this in .env (SERVICE.md § The local model):",
                   "  LLM_PROVIDER=local    LLM_MODEL=chat",
                   "  EXAM_PROVIDER=local   EXAM_MODEL=chat",
                   "  JUDGE_PROVIDER=local  JUDGE_MODEL=chat",
                   "  ALLOW_SINGLE_PROVIDER_LOOP=1")
    print()
    for role, label in roles:
        try:
            backend = ctx["backend"][role] = llm.client(role)
        except llm.LLMError as e:
            return die(f"the {label} could not be reached:", str(e))
        served = getattr(backend, "served_models", None)
        kv(f"{label} client", f"{backend.id}"
           + (f"   serves {served}" if served else "")
           + (f"   {stamp_of(backend)}" if stamp_of(backend) else ""))
    if jd.single_provider_loop():
        say("", "CAVEAT single-provider loop: the judge shares a provider with the exam "
                "writer or",
            "the generator (ALLOW_SINGLE_PROVIDER_LOOP=1). A judge scores its own family",
            "higher, and every judged score carries that caveat.")
    if ctx["backend"]["judge"].name == "local":
        say("", "CAVEAT provisional: a local server's model id is whatever was typed at "
                "launch, so",
            "nothing this judge writes is ranked or averaged. That is the point of the "
            "stamp,",
            "not a fault in the run.")
    for topic in ctx["topics"]:
        r = jd.rubric_for(eb.topic_task(topic))
        if r.status == "draft":
            say("", f"CAVEAT draft rubric: {topic} is graded against "
                    f"{jd.rubric_name(eb.topic_task(topic))}.md, which its author has not",
                "signed off. Its scores are a reading, not a result; sign-off is removing "
                "DRAFT",
                "from the heading, which changes the rubric's sha.")
    if not INSTRUCT.search(a.model):
        say("", f"NOTE {a.model} does not look instruction-tuned. A base model answers a "
                f"consumer",
            "health question with word salad and scores 0 on everything, which teaches "
            "nothing.",
            "For this kind of topic use an instruct model, e.g. "
            "HuggingFaceTB/SmolLM2-360M-Instruct.")
    return ""


# ---------------------------------------------------------------------------
# 2 · draft
# ---------------------------------------------------------------------------

def poll(backend, bid: str, what: str, timeout_s: float = 3600) -> str:
    """Narrate a batch to completion, the way the service's poller waits."""
    t0 = last = time.time()
    while True:
        state, detail = backend.status(bid)
        if state != "pending":
            say(f"{what}: {state} · {detail}" if detail else f"{what}: {state}")
            return state
        if time.time() - last >= 5:
            say(f"{what}: {detail}")
            last = time.time()
        if time.time() - t0 > timeout_s:
            say(f"{what}: still pending after {timeout_s:.0f}s — giving up")
            return "pending"
        time.sleep(0.25)


def draft(a, ctx) -> str:
    from service import config
    hr("Draft the exam — an LLM writes candidate questions, per topic")
    backend = ctx["backend"]["exam"]
    per = eb.candidates_per_request(backend.name)
    say(f"{a.per_topic} candidates per topic, through {backend.id}, "
        f"{per} to a request" + (" (a local model answers one at a time)" if per == 1 else "") + ".",
        "They land in candidates/ for a person to read. Nothing reaches the bank unread.")
    r = eb.draft(config.EXAM_DIR, backend, ctx["topics"], a.per_topic, wait=False)
    bid = r["batch_id"]
    kv("batch", f"{bid}  ({r['n_requests']} requests)")
    if poll(backend, bid, "drafting") != "done":
        return die(f"the drafting batch did not complete: {bid}")
    got = eb.fetch(config.EXAM_DIR, backend, bid)
    written, bad = got["written"], got.get("unusable") or {}
    kv("candidates written", ", ".join(f"{t}: {n}" for t, n in sorted(written.items())) or "none")
    if bad:
        cid, why = sorted(bad.items())[0]
        kv("replies unread", f"{len(bad)} of {r['n_requests']} — e.g. {cid}: {why}")
        say("  a model answering in a shape a curator cannot read is not a networking "
            "problem;",
            f"  the replies are on disk under {config.BENCH_ROOT}/llm_batches/")
    if not written:
        return die(f"no candidate questions came back: {len(bad)} of {r['n_requests']} replies "
                   f"could not be read.",
                   "The batch is on disk — read results.jsonl to see what the model actually "
                   "sent.")
    shown = 0
    for topic in ctx["topics"]:
        for c in eb.load_candidates(config.EXAM_DIR, topic, "candidate")[:1]:
            print()
            kv("topic", topic)
            say("question:")
            quote(c["prompt"])
            say("reference answer (what a full-marks answer contains):")
            quote(c["reference"])
            if c.get("provisional"):
                say(f"stamp: provisional — {c['provisional_reason']}")
            shown += 1
            if shown >= 2:
                return ""
    return ""


# ---------------------------------------------------------------------------
# 3 · accept
# ---------------------------------------------------------------------------

def accept(a, ctx) -> str:
    from service import config
    hr("Accept — in a real run this is the curation step, and it is a person")
    pending = eb.load_candidates(config.EXAM_DIR, None, "candidate")
    if not a.auto_accept:
        return die(f"{len(pending)} candidates are waiting in {eb.candidates_dir(config.EXAM_DIR)}.",
                   "They are the DEMO bank, not the live one, and this tree is kept because the",
                   "run stopped here. Read them, then carry on by hand from step 4 "
                   "(exam_build.py build),",
                   "or re-run with --auto-accept to let the demo accept them unread.")
    say(f"--auto-accept: accepting all {len(pending)} unread, recorded as approver "
        f"{APPROVER!r}.",
        "That is NOT curation. A real bank is read question by question, and the",
        "approver's name is the record of who decided.")
    for c in pending:
        eb.accept(config.EXAM_DIR, c["cid"], APPROVER)
    print()
    print(f"   {'topic':<24}{'accepted':>9}{'report':>8}{'diagnose':>10}   the qid decides which half")
    floor = 0
    for topic, s in eb.summary(config.EXAM_DIR).items():
        if not s["accepted"]:
            continue
        print(f"   {topic:<24}{s['accepted']:>9}{s['report']:>8}{s['diagnose']:>10}")
        floor = max(floor, s["report"])
    import report_lm_eval as report
    say("", "The report half is the published score; only the diagnose half may be read to",
        f"choose what to train. A real run needs {report.PROPOSE_MIN_N} report-half questions",
        "in a topic before anything can be proposed from it — the most any topic has here",
        f"is {floor}, so the live service would refuse to propose. This demo goes on anyway,",
        "and says so again at that step.")
    return ""


# ---------------------------------------------------------------------------
# 2+3 · import, when the bank is human-written
# ---------------------------------------------------------------------------

def import_step(a, ctx) -> str:
    from service import config
    import report_lm_eval as report
    hr("Import the exam — a human-written bank, not an LLM's drafts")
    topic = ctx["topics"][0]
    say(f"{a.import_path}",
        f"into the {topic!r} bank, as approver {a.approver!r} — the author of these questions.",
        "Drafting and curation (steps 2 and 3 of a normal run) are what this replaces: these",
        "were written and curated by the person whose name is on them.")
    try:
        r = eb.import_bank(config.EXAM_DIR, a.import_path, topic, a.approver, a.source)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        return die(f"the import failed: {e}")
    kv("imported", f"{r['imported']} items"
       + (f", {r['skipped']} already in the bank" if r["skipped"] else "")
       + (f", {r['invalid']} without a usable prompt" if r["invalid"] else ""))
    kv("split by qid", f"report {r['report']} / diagnose {r['diagnose']}")
    if r["acuity"]:
        kv("acuity", ", ".join(f"{k} {v}" for k, v in r["acuity"].items()))
    if not r["imported"] and not r["skipped"]:
        return die("nothing was imported — the file held no usable questions.")
    short = report.PROPOSE_MIN_N - r["report"]
    if short > 0:
        say("", f"{r['report']} report-half questions; a real run needs "
                f"{report.PROPOSE_MIN_N} before this topic",
            f"can be proposed from. The ask back to the author is at least {2 * short} more "
            f"items",
            "(the split is by qid, so about half of what arrives lands in the report half).")
    else:
        say("", f"{r['report']} report-half questions — at or above the {report.PROPOSE_MIN_N} "
                f"this topic needs",
            "before anything may be proposed from it. The floor is cleared.")
    for row in eb.load_bank(config.EXAM_DIR).get(topic, [])[:1]:
        print()
        say("One item as it was imported — the reference is its metadata, which is the "
            "ground",
            "truth the rubric asks the judge to check against, the acuity above all:")
        quote(row["prompt"])
        quote("")
        quote(row["reference"])
    return ""


# ---------------------------------------------------------------------------
# 4 · build
# ---------------------------------------------------------------------------

def build(a, ctx) -> str:
    from service import config
    hr("Build the harness tasks from the bank")
    m = eb.build(config.OUT_DIR, config.EXAM_DIR)
    tasks = eb.tasks_dir(config.EXAM_DIR)
    # the MMLU control set is built from a real results tree; this demo has
    # none, and an empty task would fail in the harness
    for task, v in sorted(m["tasks"].items()):
        if not v["items"]:
            for stale in (tasks / f"{task}.jsonl", tasks / f"{task}.yaml"):
                stale.unlink(missing_ok=True)
            continue
        kv(task, f"{v['items']} items (report {v['report']}, diagnose {v['diagnose']})")
        say(f"  {tasks / (task + '.yaml')}")
    if eb.CONTROL_TASK in m["tasks"] and not m["tasks"][eb.CONTROL_TASK]["items"]:
        say("", f"{eb.CONTROL_TASK} is empty here and was removed: it is MMLU's diagnose half "
                f"re-asked",
            "openly, and this demo tree has no MMLU run to build it from. The live loop gets "
            "it",
            "from `exam_build.py build results/full`.")
    ctx["tasks"] = [t for t, v in m["tasks"].items() if v["items"]]
    return "" if ctx["tasks"] else die("no tasks were built — the bank is empty.")


# ---------------------------------------------------------------------------
# 5 · sit the exam
# ---------------------------------------------------------------------------

def _stub_answers(a, ctx) -> None:
    """Answers without a GPU: the reference, degraded in three ways, so the
    judge has something to grade and the scores differ. CI runs this."""
    from service import config
    safe = a.model.replace("/", "__")
    for task in ctx["tasks"]:
        items = [json.loads(x) for x in
                 (eb.tasks_dir(config.EXAM_DIR) / f"{task}.jsonl").read_text().splitlines() if x.strip()]
        d = config.OUT_DIR / safe / f"{task}_0shot" / "demo"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, it in enumerate(items):
            ref = str(it.get("reference") or "")
            if i % 5 == 4:
                ans = "It depends on the situation."              # says nothing
            elif i % 5 in (1, 3):
                ans = " ".join(ref.split()[:8])                   # the start of the point
            else:
                ans = ref + " That is the mechanism, and it holds unless the constraint stops binding."
            doc_hash = hashlib.sha256(
                json.dumps(it, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            rows.append({"doc": it, "doc_hash": doc_hash, "filtered_resps": [ans],
                         "target": ref})
        (d / "samples_demo.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        # the harness writes one of these beside its samples, and the report
        # is built from them — without it the demo's own page has no run to
        # show, which is how this went unnoticed until someone looked
        (d / "results_demo.json").write_text(json.dumps({
            "results": {task: {"alias": task, "bypass,none": 999}},
            "group_subtasks": {task: []}, "versions": {task: 1.0}, "n-shot": {task: 0},
            "configs": {task: {"task": task, "output_type": "generate_until",
                               "metric_list": [{"metric": "bypass"}]}},
            "higher_is_better": {task: {"bypass": True}},
            "n-samples": {task: {"original": len(rows), "effective": len(rows)}},
            "config": {"model": "hf", "model_args": f"pretrained={a.model}", "batch_size": "1",
                       "device": "demo-stub", "limit": None, "random_seed": 1234},
            "git_hash": "demo", "date": time.time(), "chat_template": None,
            "total_evaluation_time_seconds": "0.0"}), encoding="utf-8")
    (config.OUT_DIR / safe / "model_meta.json").write_text(json.dumps(
        {"model": a.model, "kind": "demo-stub", "params": None}), encoding="utf-8")


def sit(a, ctx) -> str:
    from service import config, db
    hr("Sit the exam — the model answers every question")
    safe = a.model.replace("/", "__")
    ctx["model_dir"] = config.OUT_DIR / safe
    if a.sit == "stub":
        say("--sit stub: writing answers WITHOUT running the model. Nothing here is a",
            "measurement of anything — it exists so the script itself can be tested where",
            "there is no GPU. Use the default (--sit here) on the box.")
        _stub_answers(a, ctx)
        kv("answers", ctx["model_dir"])
        return ""
    say(f"{a.model} answers {len(ctx['tasks'])} task(s) through the same runner, the same",
        "free-VRAM gate and the same lock as any submission — one job on the card at a time.",
        "A small model by default: the demo is about the loop, not the score.")
    db.init()
    sid = db.add(a.model, "auto", "judged", APPROVER, "demo_loop.py")
    sub = db.claim_next()
    if sub is None or sub["id"] != sid:
        return die("another job was queued in the demo database; run this on a clean demo root.")
    from service import runner
    kv("submission", f"#{sid}  (its own database: {config.DB_PATH})")
    kv("shared GPU lock", runner.LOCK)
    t0 = time.time()
    runner.run_submission(sub)
    row = db.get(sid)
    kv("status", f"{row['status']} in {time.time() - t0:.0f}s · {row.get('progress') or ''}")
    if row["status"] != "done":
        # name the log only when the runner actually wrote one: a path to an
        # empty directory sends someone looking for a file that is not there
        log = config.LOGS_DIR / f"service_{sid}_{safe}.log"
        return die(f"the evaluation did not finish: {row.get('error') or row['status']}",
                   *([f"log: {log}"] if log.exists() else
                     ["the run failed before the harness wrote a log"]))
    return ""


# ---------------------------------------------------------------------------
# 6 · judge
# ---------------------------------------------------------------------------

def judge_step(a, ctx) -> str:
    from service import config, db, llm_poller
    hr("Judge — an LLM grades every answer 0-4 against the rubric, and says why")
    backend = ctx["backend"]["judge"]
    say(f"{backend.id} grades the canary first (thirty scripts with known human marks),",
        "then every answer. The batch is submitted here and finished by the poller.")
    db.init()
    runs = [r for r in db.judge_runs(20) if r["model"] == a.model]
    if not runs:
        jr = jd.start_run(ctx["model_dir"], config.OUT_DIR)
        if not jr.get("batch_id"):
            return die(f"nothing to judge: {json.dumps(jr)}")
        kv("batch", f"{jr['batch_id']}  ({jr['n']} answers + canary)")
        rid = jr["run_id"]
    else:
        rid = runs[0]["id"]
        kv("batch", f"{runs[0]['batch_id']}  (submitted by the run above)")
    row = db.judge_run_get(rid)
    if poll(backend, row["batch_id"], "judging") != "done":
        return die(f"the judging batch did not complete: {row['batch_id']}")
    llm_poller.tick()
    if db.judge_run_get(rid)["status"] != "done":
        return die("the poller did not finish the judge run",
                   db.judge_run_get(rid).get("error") or "")
    j = json.loads((ctx["model_dir"] / "judge.json").read_text())
    ctx["judge_json"] = j
    print()
    print(f"   {'topic':<24}{'report':>8}{'diagnose':>10}{'items':>7}   the report half is the score")
    for task, v in sorted(j["tasks"].items(), key=lambda kv_: (kv_[1].get("score_report") is None,
                                                               kv_[1].get("score_report") or 0)):
        rep = v.get("score_report")
        dia = v.get("score_diagnose")
        print(f"   {task:<24}{(f'{rep:.2f}/4' if rep is not None else '—'):>8}"
              f"{(f'{dia:.2f}/4' if dia is not None else '—'):>10}{v['n']:>7}")
    c = j.get("canary") or {}
    drift = (f"{c.get('mad_vs_previous')} from the previous run (limit {c.get('threshold')})"
             if c.get("mad_vs_previous") is not None else
             "first run for this judge — no previous canary to compare")
    say("", f"canary: {c.get('graded')} of {c.get('n')} re-graded · "
            f"MAD {c.get('mad_vs_human')} from the human marks · {drift}"
            + (" · MOVED" if c.get("drifted") else ""))
    jj = j["judge"]
    if jj.get("provisional"):
        say(f"stamp: PROVISIONAL — {jj['provisional_reason']}",
            f"        {jj.get('served_model')} at {jj.get('base_url')}"
            + (f" (weights {jj.get('weights')})" if jj.get("weights") else ""),
            "These scores are shown greyed on the page, never ranked, never in any average.")
    for why in j.get("preliminary_reasons") or []:
        say(f"preliminary: {why}")
    for task, v in sorted(j["tasks"].items()):
        if v.get("criteria_mean"):
            _criteria_report(task, v, jj)
    return ""


def _criteria_report(task: str, v: dict, jj: dict) -> None:
    """What a topic graded criterion by criterion looks like: where it was
    weak, which flags fired and what they did to the score, the same numbers
    broken down by every metadata field the topic carries, and one graded
    answer in full so a person can see what a grade actually is."""
    labels = v.get("criteria_labels") or {}
    counts = v.get("criteria_n") or {}
    flags = v.get("flags") or {}
    rub = (jj.get("rubrics") or {}).get(task) or {}
    print()
    say(f"{task} is graded criterion by criterion, and the 0-4 above is a fold of them:",
        f"  {rub.get('name')}.criteria.json ({rub.get('criteria_sha256', '')[:12]}, "
        f"{rub.get('criteria_status') or 'signed off'}) · "
        f"{v.get('unparseable', 0)} of {v['n']} replies unreadable")
    print()
    print(f"   {'criterion':<28}{'mean':>7}{'answers':>9}   weakest first")
    for cid, m in sorted(v["criteria_mean"].items(), key=lambda kv_: (kv_[1] is None, kv_[1])):
        print(f"   {(labels.get(cid) or cid):<28}{('—' if m is None else f'{m:.2f}'):>7}"
              f"{counts.get(cid, 0):>9}")
    if flags:
        say("", "flags — decided per answer, applied to the score in code:")
        for fid, f in flags.items():
            print(f"   {(f.get('label') or fid):<28}{f['n']:>3} of {v['n']} answers"
                  f" — each {f.get('effect_words', 'changes the score')}")
    for field, cells in (v.get("breakdowns") or {}).items():
        print()
        head = f"{field:<28}{'mean':>7}{'answers':>9}"
        print(f"   {head}" + "".join(f"{(flags[fid].get('label') or fid)[:14]:>16}"
                                     for fid in flags))
        for value, b in cells.items():
            print(f"   {value:<28}{b['mean']:>7.2f}{b['n']:>9}"
                  + "".join(f"{(b.get('flags') or {}).get(fid, 0):>16}" for fid in flags))
        if field == "difficulty":
            say("", "the author's own level, 1 easiest: does the model do well only on basic "
                    "questions,", "or can it reason about the ambiguous and high-risk ones too?")
    graded = [it for it in v.get("items") or []
              if it.get("half") == "diagnose" and it.get("criteria")]
    if graded:
        it = min(graded, key=lambda x: x["score"])
        fired = (it.get("fold") or {}).get("effects_applied") or []
        print()
        say("One graded answer in full — the diagnosis half, so it may be shown:",
            f"  score {it['score']}/4, folded from {it['fold']['applicable']} applicable "
            f"criteria"
            + (f", FLAGGED: {', '.join(fired)}" if fired else ""))
        quote(", ".join(f"{cid} {'null' if val is None else val}"
                        for cid, val in sorted(it["criteria"].items())))
        quote(it.get("justification") or "(no justification)")


# ---------------------------------------------------------------------------
# 7 · the weakest topic, and the spec that comes out of the judge's words
# ---------------------------------------------------------------------------

def sent_bodies(backend, bid: str) -> list[str]:
    """What actually went over the wire for this batch — read back from the
    backend's own on-disk record, not from the objects in memory."""
    from service import config
    out = []
    if backend.name == "local":
        d = Path(config.BENCH_ROOT) / "llm_batches" / "local" / bid
        for line in (d / "requests.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            out.append(row["system"] + "\n" + row["user"])
    elif backend.name == "fake":
        for row in backend.recorded():
            if row["batch_id"] == bid:
                out.append(row["system"] + "\n" + row["user"])
    return out


def leaked(bodies: list[str], questions: list[str]) -> list[str]:
    """Any exam question text that reached a request body, by the gate's own
    13-gram rule. Both halves: the report half must never appear, and the
    diagnose half must not be quoted into a proposal either."""
    from service import contamination as ct
    seen = set()
    for body in bodies:
        seen |= set(ct.windows(ct.normalize(body)))
    hits = []
    for q in questions:
        grams = set(ct.windows(ct.normalize(q)))
        if grams & seen:
            hits.append(q)
    return hits


def bank_questions(exam_root: Path) -> tuple[list[str], dict[str, int]]:
    rows = [r for rs in eb.load_bank(exam_root).values() for r in rs]
    halves = {"report": 0, "diagnose": 0}
    for r in rows:
        halves[eb.half_of(r["qid"])] += 1
    return [r["prompt"] for r in rows], halves


def propose(a, ctx) -> str:
    from service import config, db, llm, llm_poller, proposals as prop
    import report_lm_eval as report
    hr("Pick the weakest topic, and propose a skill spec from the judge's own words")
    weak = prop.weak_topics(ctx["model_dir"])
    if not weak:
        return die("no judged topic to pick from.")
    for w in weak[:5]:
        kv(w["topic"], f"{w['score_report']}/4 on {w['n_report']} report-half questions")
    topic = weak[0]["topic"]
    task = eb.topic_task(topic)
    print()
    kv("weakest topic", topic)
    items, counts = prop.justifications_for(ctx["model_dir"], task)
    shown = items[:3]
    say(f"{counts['diagnose_weak']} of {counts['diagnose_items']} diagnosis-half answers fell "
        f"short; {len(items)} go into the request.",
        "What the judge wrote about them — the evidence, in a person's vocabulary"
        + (f" ({len(shown)} of {len(items)} shown):" if len(shown) < len(items) else ":"))
    for it in shown:
        quote(f"scored {it['score']}/4 — {it['justification']}")
    say("", "The live service would REFUSE this proposal here: the judged suite is "
            "provisional or",
        f"under the {report.PROPOSE_MIN_N}-question floor, and that gate is enforced in the API, "
        f"not just on",
        "the button. The demo goes around it on purpose, to show the shape of the step.")
    backend = ctx["backend"]["llm"]
    db.init()
    pid = db.proposal_create(a.model, task, topic, APPROVER,
                             {"n_shown": len(items), **counts, "demo": True})
    crit = prop.criteria_evidence(ctx["model_dir"], task)
    if crit.get("weakest_criteria"):
        fields = ", ".join(crit.get("breakdowns") or {}) or "no metadata field"
        say("", "This topic is graded criterion by criterion, so the request also carries the",
            f"three weakest criteria and the means by {fields} — labels and numbers only:")
        for c in crit["weakest_criteria"]:
            say(f"  {c['label']}: {c['mean']} over {c['n']} answers")
    req = prop.proposal_request(pid, a.model, task, topic, items, counts,
                                jd.rubric_for(task)[0], crit,
                                audience=prop.audience_for(topic, task))
    bid = backend.submit([req])
    db.batch_add(bid, "proposal", pid, 1, backend.name, backend.model)
    db.proposal_update(pid, batch_id=bid, prompt_sha=llm.prompt_sha(req.system, req.user),
                       judge_run=json.dumps({"judge_id": (ctx["judge_json"]["judge"] or {}).get("id"),
                                             "batch_id": (ctx["judge_json"]["judge"] or {}).get("batch_id"),
                                             "prompt_sha256": (ctx["judge_json"]["judge"] or {}).get("prompt_sha256")}))
    kv("batch", bid)
    if poll(backend, bid, "proposing") != "done":
        return die(f"the proposal batch did not complete: {bid}")
    llm_poller.tick()
    row = db.proposal_get(pid)
    if row["status"] != "proposed":
        return die(f"the proposal failed: {row.get('error')}")
    ctx["pid"] = pid
    print()
    say("The spec that came back — this is all the generator will ever see:")
    quote(row["spec_text"])
    ev = json.loads(row["evidence"] or "{}")
    if ev.get("provisional"):
        say(f"stamp: PROVISIONAL — {ev['provisional_reason']}")
    # the safety property, shown working rather than claimed
    questions, halves = bank_questions(config.EXAM_DIR)
    hits = leaked(sent_bodies(backend, bid), questions)
    print()
    say("SAFETY CHECK — no exam question text in the request that produced that spec:",
        f"  checked the {len(sent_bodies(backend, bid))} request body/bodies actually sent, "
        f"against all",
        f"  {len(questions)} bank questions ({halves['report']} report half, "
        f"{halves['diagnose']} diagnose half),",
        f"  by the contamination gate's own {13}-gram rule.")
    if hits:
        say(f"  FAILED: {len(hits)} question(s) appear in the request body.")
        quote(hits[0])
        return die("a question reached the LLM. That is the property this loop exists to keep.")
    say("  PASSED: none of them appear. The judge's words are about the ANSWERS.")
    return ""


# ---------------------------------------------------------------------------
# 8 · generate
# ---------------------------------------------------------------------------

def generate(a, ctx) -> str:
    from service import config, db, llm, llm_poller, proposals as prop
    hr("Generate the dataset — prose documents for that spec, through the gate")
    backend = ctx["backend"]["llm"]
    db.proposal_update(ctx["pid"], status="approved", approver=APPROVER, approved_at=time.time())
    row = db.proposal_get(ctx["pid"])
    spec = row["edited_text"] or row["spec_text"]
    per = prop.items_per_request("doc", backend.name)
    say(f"{a.count} documents, {per} per request, from the approved spec and nothing else:",
        "no exam question, no benchmark item, no hash, no model name, no score.")
    did = db.dataset_create(ctx["pid"], "doc", a.count, APPROVER, {})
    audience = prop.audience_for(row["category"], row["task"])
    if audience:
        say("", "and who asks these questions, as labels from the bank's own metadata —",
            "so the documents are written for that reader and not for a professional:")
        quote(audience)
    reqs = prop.generation_requests(did, spec, row["category"], a.count, "doc", seed=did,
                                    audience=audience)
    sha = llm.prompt_sha(*[q.system + "\n" + q.user for q in reqs])
    bid = backend.submit(reqs)
    db.batch_add(bid, "generation", did, len(reqs), backend.name, backend.model)
    db.dataset_update(did, batch_id=bid,
                      provenance=json.dumps({"prompt_sha256": sha, "audience": audience}))
    kv("batch", f"{bid}  ({len(reqs)} requests)")
    if poll(backend, bid, "generating") != "done":
        return die(f"the generation batch did not complete: {bid}")
    cut = backend.status(bid)[1]
    if "cut off" in cut:
        say(f"NOTE {cut} — a truncated reply is a lost document; lower the count per request",
            "or raise the cap it names if this is more than the odd one.")
    hits = leaked(sent_bodies(backend, bid), bank_questions(config.EXAM_DIR)[0])
    say(f"SAFETY CHECK — exam question text in the generation request: "
        f"{'FAILED' if hits else 'none, as expected'}.")
    llm_poller.tick()
    ds = db.dataset_get(did)
    ctx["did"] = did
    prov = json.loads(ds["provenance"] or "{}")
    gate = prov.get("gate") or {}
    if ds["status"] != "ready":
        say(f"status {ds['status']}: {ds.get('error')}")
        if ds["status"] != "rejected":
            return die("the dataset could not be written.")
    kv("kept / generated", f"{(prov.get('items') or {}).get('kept', 0)} of "
                           f"{(prov.get('items') or {}).get('generated', 0)}")
    kv("dropped by the gate", f"{gate.get('dropped_benchmark', 0)} sharing a 13-gram with a "
                              f"benchmark or exam item, "
                              f"{gate.get('dropped_near_dup', 0)} near-duplicates")
    kv("indexed against", f"{gate.get('benchmark_docs', 0)} harness docs, "
                          f"{gate.get('exam_questions', 0)} exam questions (both halves)")
    say("  those are the DEMO tree's own: the gate indexes whatever results and exam bank",
        "  it is pointed at, and in the live service that is every benchmark on disk.")
    if prov.get("provisional"):
        say(f"stamp: PROVISIONAL — {prov['provisional_reason']}")
    path = prop.dataset_dir(did) / "items.jsonl"
    if path.exists():
        docs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        ctx["dataset_path"] = path
        if docs:
            print()
            say("One document in full — prose, not a question-and-answer pair:")
            quote(docs[0].get("title", ""))
            quote("")
            quote(docs[0].get("text", ""))
    return ""


# ---------------------------------------------------------------------------
# 9 · summary
# ---------------------------------------------------------------------------

def write_report(a, ctx) -> Path | None:
    """The demo's own dashboard, of the demo's own tree. Without it "shown
    greyed on the page" is a claim about a code path nobody can open: the
    live board reads results/full and this ran under demo/, which is the
    isolation working and the demo invisible. Two pages, two trees."""
    import report_lm_eval as report
    from service import config
    runs = report.load_results(config.OUT_DIR) if config.OUT_DIR.is_dir() else []
    if not runs:
        return None
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    title = f"DEMO RUN — {when} — not the leaderboard"
    out = report.build_report(
        runs, ctx["root"] / "report.html", title,
        judge_identity=jd.identity(),
        banner=(f"DEMO RUN — {when} — every number here is provisional and stamped "
                f"{APPROVER}; nothing on this page is on the leaderboard."),
        banner_link=("/", "the live dashboard"))
    return out


def summary(a, ctx) -> None:
    from service import config, proposals as prop
    hr("Summary — every path this run wrote")
    report_path = write_report(a, ctx)
    if report_path:
        kv("THE DEMO'S PAGE", report_path)
        say("  the service serves it at /demo — e.g. http://<this box>:8899/demo — with a",
            "  banner saying what it is. The live board at / is untouched and lists no",
            "  model from this run.")
        print()
    kv("exam bank", eb.bank_dir(config.EXAM_DIR))
    kv("harness tasks", eb.tasks_dir(config.EXAM_DIR))
    kv("answers + judge.json", ctx.get("model_dir", "—"))
    if ctx.get("did"):
        kv("dataset", prop.dataset_dir(ctx["did"]))
        kv("provenance", prop.dataset_dir(ctx["did"]) / "provenance.json")
    kv("database", config.DB_PATH)
    if a.import_path:
        # the next step after a demo of an imported bank is to put that bank
        # in the LIVE exam, which is one paste rather than a reconstruction
        say("", "This bank is in the demo's exam, not the live one. To put it in the live exam:",
            f"  python3 scripts/exam_build.py --root {ctx['bench_root']}/exam import \\",
            f"      {a.import_path} --topic {ctx['topics'][0]!r} \\",
            f"      --approver {a.approver!r} --source {a.source!r}",
            "  then rebuild the tasks (Exam tab, or exam_build.py build results/full).")
    say("", "What would happen next, in a real cycle:",
        f"  1. fine-tune the model on that dataset "
        f"(examples/train_and_benchmark.py --gap-dataset {ctx.get('did', '<id>')}),",
        "  2. resubmit the checkpoint with suite=judged — it sits the SAME exam,",
        "  3. compare the two halves: the report half moving with the diagnose half is",
        "     the skill; the diagnose half alone is the test being taught.",
        "", "What this run did not exercise: the Anthropic and OpenAI batch clients. They",
        "stay untested until there are keys — a green demo is not a green production path.")
    if report_path and not a.keep:
        say("", f"The tree below is about to be removed; {report_path.name} stays, so the page",
            "outlives the run it describes.")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", "--topic", dest="topics", default="economics",
                    help="comma-separated, from scripts/categories.yaml (default: economics)")
    ap.add_argument("--import", dest="import_path", type=Path, default=None,
                    help="a human-written bank (JSON array) to import instead of drafting and "
                         "curating — one topic, and --approver names its author")
    ap.add_argument("--approver", default=None,
                    help="who stands behind an imported bank; recorded on every item")
    ap.add_argument("--source", default=None,
                    help="provenance tag for an imported bank (default: the file's stem)")
    ap.add_argument("--model", default="EleutherAI/pythia-160m",
                    help="the model that sits the exam (default: a small one, on purpose)")
    ap.add_argument("--per-topic", type=int, default=12, help="candidate questions drafted")
    ap.add_argument("--count", type=int, default=20, help="documents generated")
    ap.add_argument("--sit", choices=("here", "stub"), default="here",
                    help="here: run the model through the normal runner and lock. "
                         "stub: write answers without a GPU (CI, and a laptop)")
    ap.add_argument("--root", type=Path, default=None,
                    help="where the demo writes (default $BENCH_ROOT/demo)")
    ap.add_argument("--auto-accept", dest="auto_accept", action="store_true", default=True,
                    help="accept every drafted question unread, as approver 'demo' (default)")
    ap.add_argument("--no-auto-accept", dest="auto_accept", action="store_false",
                    help="stop after drafting so a person can curate on the Exam tab")
    ap.add_argument("--keep", action="store_true",
                    help="leave the demo tree behind (default: remove it, so it cannot be "
                         "mistaken for the real one)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, call nothing")
    a = ap.parse_args()

    from service import config
    root = a.root or (config.BENCH_ROOT / "demo")
    topics = [t.strip() for t in a.topics.split(",") if t.strip()]
    known = _categories.category_order()
    unknown = [t for t in topics if t not in known]
    if unknown:
        return die(f"not exam topics: {', '.join(unknown)}", f"known: {', '.join(known)}")
    if a.import_path:
        if not a.approver or not a.approver.strip():
            return die("--import needs --approver: the name of whoever stands behind these "
                       "questions",
                       "is recorded on every one of them, the way a curator's name is.")
        if len(topics) != 1:
            return die(f"--import takes one topic, not {len(topics)}: an imported file is one "
                       f"topic's bank.")
        if not a.import_path.exists():
            return die(f"no such file: {a.import_path}")
        a.source = a.source or a.import_path.stem
    ctx = {"root": root, "topics": topics, "backend": {}, "bench_root": config.BENCH_ROOT}
    print(f"\n{'═' * WIDTH}\n  THE LOOP, END TO END — a demo run, everything it writes stamped "
          f"'{APPROVER}'\n{'═' * WIDTH}")
    if a.dry_run:
        configure(root, create=False)      # a plan writes nothing, not even a directory
        hr("Dry run — the plan, in order. Nothing is called.")
        for i, line in enumerate((
                (f"import {a.import_path} into {topics[0]!r} as {a.approver!r} — no drafting, "
                 f"no curation" if a.import_path else
                 f"draft {a.per_topic} candidates per topic ({', '.join(topics)}) through the "
                 f"exam writer"),
                *([] if a.import_path else
                  [f"accept them unread as {APPROVER!r}" if a.auto_accept else
                   "stop for curation"]),
                "build the harness tasks from the bank, split by qid",
                f"{a.model} sits the exam ({a.sit})",
                "the judge grades every answer, canary first",
                "the weakest topic's judged words become a skill spec",
                f"the spec becomes {a.count} training documents, through the contamination gate",
                "print every path written" + ("" if a.keep else ", then remove the demo tree")), 1):
            say(f"{i}. {line}")
        kv("", "")
        kv("demo root", root)
        return 0
    configure(root)
    t0 = time.time()
    ok = False
    try:
        write_bank = (import_step,) if a.import_path else (draft, accept)
        for stepfn in (preflight, *write_bank, build, sit, judge_step, propose, generate):
            why = stepfn(a, ctx)
            if why:
                return 2
        summary(a, ctx)
        print(f"\n   done in {time.time() - t0:.0f}s")
        ok = True
        return 0
    finally:
        # a run that stopped early keeps its tree whatever --keep says: the
        # logs and half-written artefacts are the whole point of looking
        if not root.exists():
            pass
        elif a.keep or not ok:
            print(f"\n   left {root} in place"
                  + ("" if ok else " — the run stopped early, so its logs are still there"))
        else:
            # everything except the report: the numbers were a demo and go,
            # the page that says they were a demo stays
            kept = root / "report.html"
            for child in root.iterdir():
                if child == kept:
                    continue
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
            print(f"\n   removed {root}/* (--keep leaves it all)"
                  + (f", kept {kept.name}" if kept.exists() else ""))


if __name__ == "__main__":
    raise SystemExit(main())
