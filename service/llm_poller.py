"""Batch poller: one daemon thread, one loop, no GPU.

Every batch id is written to the llm_batches table before this thread ever
sees it, so a restart resumes exactly where the previous process stopped:
tick() reads the rows still 'submitted', asks the provider, and dispatches the
finished ones. Nothing is re-submitted, ever — the only way a second batch is
created is a second human click. worker.py's shape, on purpose.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback

import sys
from pathlib import Path

from . import config, contamination, db, llm, proposals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import exam_build as _exam  # noqa: E402
import judge as _judge  # noqa: E402

_stop = threading.Event()


def _finish_proposal(row: dict, results: dict[str, llm.Result], backend: llm.Backend) -> None:
    pid = row["ref_id"]
    res = results.get(f"proposal:{pid}")
    if res is None or res.error:
        db.proposal_update(pid, status="failed",
                           error=(res.error if res else "no result for this request")[:400])
        return
    try:
        parsed = proposals.parse_proposal(res.text)
    except llm.LLMError as e:
        db.proposal_update(pid, status="failed", error=str(e)[:400])
        return
    prop = db.proposal_get(pid)
    ev = json.loads(prop["evidence"] or "{}")
    ev.update({"share_explained": parsed["share_explained"], "patterns": parsed["patterns"]})
    ev.update(llm.provisional(backend, "proposed"))      # nothing unless the proposer was local
    db.proposal_update(pid, status="proposed", spec_text=parsed["spec"],
                       evidence=json.dumps(ev), proposer=backend.id)


def _finish_generation(row: dict, results: dict[str, llm.Result], backend: llm.Backend) -> None:
    did = row["ref_id"]
    ds = db.dataset_get(did)
    prop = db.proposal_get(ds["proposal_id"])
    prov_stub = json.loads(ds["provenance"] or "{}")
    # what each request was asked for, recorded when the batch was submitted:
    # every document asked for is accounted for against it
    plan = {int(q["k"]): q for q in (prov_stub.get("requests") or [])}
    items: list[dict] = []
    origin: list[dict] = []               # per parsed item: which request it came from
    missing: list[dict] = []
    errors = []
    for cid, res in sorted(results.items(), key=lambda kv: _req_k(kv[0])):
        if not cid.startswith(f"gen:{did}:"):
            continue
        k = _req_k(cid)
        asked = int((plan.get(k) or {}).get("count") or 1)
        focus = (plan.get(k) or {}).get("focus")
        if res.error:
            errors.append(res.error)
            missing += [{"request": k, "focus": focus, "why": f"error: {res.error[:160]}"}] * asked
            continue
        got, why = proposals.read_reply(res.text, ds["fmt"], asked)
        items += got
        origin += [{"request": k, "focus": focus}] * len(got)
        missing += [{"request": k, "focus": focus, "why": w} for w in why]
    if not items:
        db.dataset_update(did, status="failed", finished_at=time.time(),
                          # the same accounting a ready dataset carries: every
                          # document asked for, and why none of them arrived
                          provenance=json.dumps({**prov_stub, "items": {
                              "requested": ds["count"], "generated": 0, "dropped": 0,
                              "kept": 0, "missing": missing}}),
                          error=("the generator returned no parseable items"
                                 + (f"; errors: {errors[0]}" if errors else ""))[:400])
        return
    ix = contamination.index(config.OUT_DIR, config.EXAM_DIR)
    gate = contamination.check(items, ix)
    # a document the gate dropped is missing too, and says which gate dropped it
    for d in gate.get("dropped") or []:
        src = d.get("source") if d.get("reason") == "benchmark" else None
        o = origin[d["index"]] if d.get("index", -1) < len(origin) else {}
        missing.append({"request": o.get("request"), "focus": o.get("focus"),
                        "why": f"dropped by the gate: {src or d.get('reason')}"})
    prompt_hash = prov_stub.get("prompt_sha256", "")
    focus_plan = prov_stub.get("focus_plan") or []
    if gate["report"]["rejected"]:
        prov = proposals.provenance(prop, ds, backend.id, row["batch_id"], prompt_hash,
                                    gate["report"], sha="", n_generated=len(items), n_kept=0,
                                    audience=prov_stub.get("audience", ""),
                                    missing=missing, focus=focus_plan,
                                    focus_mode=prov_stub.get("focus_mode", ""),
                                    focus_labels=prov_stub.get("focus_labels"))
        db.dataset_update(did, status="rejected", finished_at=time.time(),
                          provenance=json.dumps(prov),
                          error=f"{gate['report']['share_dropped_benchmark']:.1%} of items "
                                f"share a 13-gram with the benchmark — above the "
                                f"{contamination.MAX_DROP_SHARE:.0%} line, so the generator "
                                f"is echoing the test, not answering the spec")
        return
    why = proposals.quota_blocked()
    if why:
        db.dataset_update(did, status="failed", finished_at=time.time(), error=why)
        return
    path, sha = proposals.write_items(did, gate["kept"])
    # 11g: each kept document's request and focus label, for the Reader
    from . import reader
    reader.write_item_labels(did, origin, gate.get("dropped") or [])
    prov = proposals.provenance(prop, ds, backend.id, row["batch_id"], prompt_hash,
                                gate["report"], sha, len(items), len(gate["kept"]),
                                audience=prov_stub.get("audience", ""),
                                missing=missing, focus=focus_plan,
                                focus_mode=prov_stub.get("focus_mode", ""),
                                focus_labels=prov_stub.get("focus_labels"))
    holes = proposals.provenance_complete(prov)
    if holes:                      # a dataset with an unaccountable field is not ready
        db.dataset_update(did, status="failed", finished_at=time.time(),
                          error="provenance incomplete: " + ", ".join(holes)[:300])
        return
    (path.parent / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    db.dataset_update(did, status="ready", finished_at=time.time(),
                      provenance=json.dumps(prov))


def _req_k(cid: str) -> int:
    """The request's number inside its batch (gen:<dataset>:<k>)."""
    try:
        return int(str(cid).rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return 0


def _finish_judge(row: dict, results: dict[str, llm.Result]) -> None:
    run = db.judge_run_get(row["ref_id"])
    if not run:
        return
    _judge.finish_run(run, results, config.OUT_DIR)
    db.judge_run_update(run["id"], status="done", finished_at=time.time())
    # the count stops where it landed, not where the last poll happened to
    # see it, and the submission stops saying the batch is out
    db.batch_progress(row["batch_id"], f"{run['n_items']}/{run['n_items']} done")
    sub = db.submission_of_batch(row["batch_id"])
    if not sub:
        return
    db.update(sub["id"], progress=judged_line(run, note=sub.get("reuse_note") or ""))


def judged_line(run: dict, at: float | None = None, note: str = "") -> str:
    """What a judged row says once its batch has landed — after, when the
    run answered nothing new, that it re-graded answers already on disk."""
    when = time.strftime("%H:%M", time.localtime(time.time() if at is None else at))
    what = judged_what(run)
    line = (f"judged: {what}, judge.json written {when}" if what
            else f"judged, judge.json written {when}")
    return f"{note} · {line}" if note else line


# a row still telling the reader its batch is out, or the pre-#33 count of
# the merged file ("judged: 5 topics" on a one-topic run)
_STILL_OUT = re.compile(r"judge batch \S+ submitted|judge\.json lands when it completes")
_WRITTEN = ", judge.json written"


def repair_finished_rows() -> list[int]:
    """A judged row whose batch finished before the poller wrote the final
    count and rewrote the row (code before #32) still reads as in flight:
    #45 "62/130 done" and #46 "475/680 done", both "judge.json lands when it
    completes". 62/130 reads as half the answers lost. For every row whose
    own batch and judge run are done: the count becomes n/n, and the row
    says what a new run's says, with the time the run actually finished.
    A row whose batch is pending or failed is not touched. Returns the rows
    changed."""
    runs = {r["batch_id"]: r for r in db.judge_runs(1000)}
    batches = {b["batch_id"]: b for b in db.batches_list(5000) if b["kind"] == "judge"}
    fixed = []
    for s in db.recent(500):
        bid = s.get("judge_batch") or ""
        b, run = batches.get(bid), runs.get(bid)
        if s["suite"] != "judged" or not b or not run \
                or b["status"] != "done" or run["status"] != "done":
            continue
        changed = False
        n = run["n_items"]
        if b.get("progress") != f"{n}/{n} done":
            db.batch_progress(bid, f"{n}/{n} done")
            changed = True
        old = s.get("progress") or ""
        if s["status"] == "done":
            new = judged_line(db.judge_run_get(run["id"]) or run,
                              run.get("finished_at") or b.get("finished_at"),
                              note=s.get("reuse_note") or "")
            stale = _STILL_OUT.search(old) or (
                old.startswith("judged") and old.split(_WRITTEN)[0] != new.split(_WRITTEN)[0])
            if stale:
                db.update(s["id"], progress=new)
                changed = True
        if changed:
            fixed.append(s["id"])
    return sorted(fixed)


REPAIR = "judge-records-2026-09"


def repair_once() -> str:
    """The one-off repairs of judge records older code left wrong, run once
    per database and recorded in `repairs`: per-topic times in judge.json,
    then the finished rows' counts and text. A second start — or a second
    process starting beside the first — finds the record and does nothing,
    so this can say what it did exactly once. Returns that line ("" when
    it did not run or found nothing to change)."""
    if not db.claim_repair(REPAIR):
        return ""
    try:
        times = backfill_judged_at()
        rows = repair_finished_rows()
    except Exception:
        db.release_repair(REPAIR)              # failed part-way: the next start tries again
        raise
    bits = []
    if times:
        bits.append(f"per-topic judged_at for {', '.join(times)}")
    if rows:
        bits.append(f"final count and text on {', '.join(f'#{i}' for i in rows)}")
    line = "; ".join(bits)
    db.finish_repair(REPAIR, line or "nothing to restore")
    return line


def backfill_judged_at() -> list[str]:
    """Once, at startup: a judge.json merged before topics carried their own
    time stamped every topic with the merge (a one-topic run of economics at
    09:25 re-dated last night's law, medicine and physics). The runs that
    graded them are still in the database; put each topic's own time back.
    Returns the models whose file was corrected."""
    by_model: dict[str, list[dict]] = {}
    for r in db.judge_runs(1000):
        if r["status"] == "done" and r.get("finished_at"):
            by_model.setdefault(r["model"], []).append(r)
    fixed = []
    for model, rs in by_model.items():
        path = config.OUT_DIR / model.replace("/", "__") / "judge.json"
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tasks = j.get("tasks") or {}
        if not isinstance(tasks, dict) or all(isinstance(t, dict) and t.get("judged_at")
                                              for t in tasks.values()):
            continue
        runs = []
        for r in rs:
            try:
                plan = json.loads((db.judge_run_get(r["id"]) or {}).get("plan") or "{}")
            except ValueError:
                continue
            runs.append({"batch_id": r["batch_id"], "finished_at": r["finished_at"],
                         "tasks": set(plan.get("tasks") or {})})
        if _judge.backfill_judged_at(j, runs):
            _judge.write_judge(path.parent, j)
            fixed.append(model)
    return fixed


def judged_what(run: dict) -> str:
    """The topics THIS run graded, from its own plan — not the merged
    judge.json, which also holds every topic an earlier run left there: a
    one-topic run of economics read "judged: 5 topics". Up to three are
    named; past that the row's own task list names them, and a count says
    enough."""
    try:
        tasks = list((json.loads(run.get("plan") or "{}").get("tasks") or {}))
    except (ValueError, TypeError, AttributeError):
        return ""
    names = [_exam.TASK_TOPIC.get(t, t) for t in tasks if t != _judge.CONTROL_TASK]
    if not names or len(names) > 3:
        return f"{len(names)} topics" if names else ""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _mark_failed(r: dict, why: str) -> None:
    if r["kind"] == "proposal":
        db.proposal_update(r["ref_id"], status="failed", error=why[:400])
    elif r["kind"] == "generation":
        db.dataset_update(r["ref_id"], status="failed", finished_at=time.time(), error=why[:400])
    elif r["kind"] == "judge":
        db.judge_run_update(r["ref_id"], status="failed", finished_at=time.time(), error=why[:400])


def tick() -> int:
    """Poll every submitted batch once. Returns how many finished. Each batch
    is polled with the identity that submitted it — judge batches belong to
    the judge, the rest to the generator."""
    rows = db.batches_pending()
    if not rows:
        return 0
    done = 0
    for r in rows:
        try:
            backend = llm.client("judge" if r["kind"] == "judge" else "llm")
        except llm.LocalUnreachable as e:
            # vLLM restarting (or still loading after a reboot) is not a reason
            # to throw away batches whose finished results are on disk
            print(f"[llm] {r['batch_id']}: {e} — trying again next tick")
            continue
        except llm.LLMError as e:
            db.batch_finish(r["batch_id"], "failed", f"LLM unavailable: {e}")
            _mark_failed(r, f"LLM unavailable: {e}")
            continue
        try:
            state, detail = backend.status(r["batch_id"])
        except llm.LLMError as e:
            print(f"[llm] status {r['batch_id']}: {e}")
            continue
        if state == "done" and detail and detail != r.get("progress"):
            db.batch_progress(r["batch_id"], detail)     # the count that actually landed
        if state == "pending":
            # the provider was asked anyway; recording what it said is what
            # lets a queue row say "judging 40/80" instead of only "pending"
            if detail and detail != r.get("progress"):
                db.batch_progress(r["batch_id"], detail)
            continue
        if state == "failed":
            db.batch_finish(r["batch_id"], "failed", detail)
            _mark_failed(r, f"batch failed: {detail}")
            done += 1
            continue
        try:
            results = backend.fetch(r["batch_id"])
            if r["kind"] == "proposal":
                _finish_proposal(r, results, backend)
            elif r["kind"] == "generation":
                _finish_generation(r, results, backend)
            elif r["kind"] == "judge":
                _finish_judge(r, results)
            db.batch_finish(r["batch_id"], "done", "")
        except Exception as e:                       # noqa: BLE001 — one batch must not kill the loop
            traceback.print_exc()
            db.batch_finish(r["batch_id"], "failed", repr(e)[:400])
            _mark_failed(r, repr(e))
        done += 1
    return done


def loop() -> None:
    while not _stop.is_set():
        try:
            tick()
        except Exception:                            # noqa: BLE001
            traceback.print_exc()
        _stop.wait(config.LLM_POLL_S)


def start() -> threading.Thread | None:
    if not (config.LLM_PROVIDER or config.JUDGE_PROVIDER):
        return None
    _stop.clear()
    t = threading.Thread(target=loop, name="llm-poller", daemon=True)
    t.start()
    return t


def stop() -> None:
    _stop.set()
