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

from . import config, contamination, db, llm, proposals

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
    ix = contamination.index(config.OUT_DIR)
    gate = contamination.check(items, ix)
    prov_stub = json.loads(ds["provenance"] or "{}")
    prompt_hash = prov_stub.get("prompt_sha256", "")
    if gate["report"]["rejected"]:
        prov = proposals.provenance(prop, ds, backend.id, row["batch_id"], prompt_hash,
                                    gate["report"], sha="", n_generated=len(items), n_kept=0)
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
                                gate["report"], sha, len(items), len(gate["kept"]))
    holes = proposals.provenance_complete(prov)
    if holes:                      # a dataset with an unaccountable field is not ready
        db.dataset_update(did, status="failed", finished_at=time.time(),
                          error="provenance incomplete: " + ", ".join(holes)[:300])
        return
    (path.parent / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    db.dataset_update(did, status="ready", finished_at=time.time(),
                      provenance=json.dumps(prov))


def tick() -> int:
    """Poll every submitted batch once. Returns how many finished."""
    rows = db.batches_pending()
    if not rows:
        return 0
    try:
        backend = llm.client()
    except llm.LLMError as e:
        for r in rows:
            db.batch_finish(r["batch_id"], "failed", f"LLM unavailable: {e}")
        return 0
    done = 0
    for r in rows:
        try:
            state, detail = backend.status(r["batch_id"])
        except llm.LLMError as e:
            print(f"[llm] status {r['batch_id']}: {e}")
            continue
        if state == "pending":
            continue
        if state == "failed":
            db.batch_finish(r["batch_id"], "failed", detail)
            if r["kind"] == "proposal":
                db.proposal_update(r["ref_id"], status="failed", error=f"batch failed: {detail}")
            else:
                db.dataset_update(r["ref_id"], status="failed", finished_at=time.time(),
                                  error=f"batch failed: {detail}")
            done += 1
            continue
        try:
            results = backend.fetch(r["batch_id"])
            if r["kind"] == "proposal":
                _finish_proposal(r, results, backend)
            else:
                _finish_generation(r, results, backend)
            db.batch_finish(r["batch_id"], "done", "")
        except Exception as e:                       # noqa: BLE001 — one batch must not kill the loop
            traceback.print_exc()
            db.batch_finish(r["batch_id"], "failed", repr(e)[:400])
            if r["kind"] == "proposal":
                db.proposal_update(r["ref_id"], status="failed", error=repr(e)[:400])
            else:
                db.dataset_update(r["ref_id"], status="failed", finished_at=time.time(),
                                  error=repr(e)[:400])
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
    if not config.LLM_PROVIDER:
        return None
    _stop.clear()
    t = threading.Thread(target=loop, name="llm-poller", daemon=True)
    t.start()
    return t


def stop() -> None:
    _stop.set()
