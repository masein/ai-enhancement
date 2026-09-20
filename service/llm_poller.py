"""Batch poller: one daemon thread, one loop, no GPU.

Every batch id is written to the llm_batches table before this thread ever
sees it, so a restart resumes exactly where the previous process stopped:
tick() reads the rows still 'submitted', asks the provider, and dispatches the
finished ones. Nothing is re-submitted, ever — the only way a second batch is
created is a second human click. worker.py's shape, on purpose.
"""

from __future__ import annotations

import json
import threading
import time
import traceback

import sys
from pathlib import Path

from . import config, contamination, db, llm, proposals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
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
    items: list[dict] = []
    errors = []
    for cid, res in sorted(results.items()):
        if not cid.startswith(f"gen:{did}:"):
            continue
        if res.error:
            errors.append(res.error)
            continue
        items.extend(proposals.parse_items(res.text, ds["fmt"]))
    if not items:
        db.dataset_update(did, status="failed", finished_at=time.time(),
                          error=("the generator returned no parseable items"
                                 + (f"; errors: {errors[0]}" if errors else ""))[:400])
        return
    ix = contamination.index(config.OUT_DIR, config.EXAM_DIR)
    gate = contamination.check(items, ix)
    prov_stub = json.loads(ds["provenance"] or "{}")
    prompt_hash = prov_stub.get("prompt_sha256", "")
    if gate["report"]["rejected"]:
        prov = proposals.provenance(prop, ds, backend.id, row["batch_id"], prompt_hash,
                                    gate["report"], sha="", n_generated=len(items), n_kept=0,
                                    audience=prov_stub.get("audience", ""))
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
    prov = proposals.provenance(prop, ds, backend.id, row["batch_id"], prompt_hash,
                                gate["report"], sha, len(items), len(gate["kept"]),
                                audience=prov_stub.get("audience", ""))
    holes = proposals.provenance_complete(prov)
    if holes:                      # a dataset with an unaccountable field is not ready
        db.dataset_update(did, status="failed", finished_at=time.time(),
                          error="provenance incomplete: " + ", ".join(holes)[:300])
        return
    (path.parent / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    db.dataset_update(did, status="ready", finished_at=time.time(),
                      provenance=json.dumps(prov))


def _finish_judge(row: dict, results: dict[str, llm.Result]) -> None:
    run = db.judge_run_get(row["ref_id"])
    if not run:
        return
    path = _judge.finish_run(run, results, config.OUT_DIR)
    db.judge_run_update(run["id"], status="done", finished_at=time.time())
    # the count stops where it landed, not where the last poll happened to
    # see it, and the submission stops saying the batch is out
    db.batch_progress(row["batch_id"], f"{run['n_items']}/{run['n_items']} done")
    sub = db.submission_of_batch(row["batch_id"])
    if not sub:
        return
    try:
        topics = len([t for t in json.loads(Path(path).read_text(encoding="utf-8")).get("tasks")
                      or {} if t != _judge.CONTROL_TASK])
    except (OSError, ValueError, TypeError):
        topics = 0
    when = time.strftime("%H:%M", time.localtime())
    db.update(sub["id"], progress=(
        f"judged: {topics} topic{'s' if topics != 1 else ''}, judge.json written {when}"
        if topics else f"judged, judge.json written {when}"))


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
