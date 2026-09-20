#!/usr/bin/env python3
"""The exam: draft it, curate it, split it, build it for the harness.

    python scripts/exam_build.py migrate  --root $BENCH_ROOT/exam            # once: the 40 skill items
    python scripts/exam_build.py draft    --root $BENCH_ROOT/exam --per-topic 8 [--topic economics …]
    python scripts/exam_build.py fetch    --root $BENCH_ROOT/exam <batch_id>  # after draft --no-wait
    python scripts/exam_build.py summary  --root $BENCH_ROOT/exam
    python scripts/exam_build.py build    results/full --root $BENCH_ROOT/exam

The exam is the instrument. Its questions span the TOPICS in
scripts/categories.yaml — economics, law, medicine & health, … — because
"which topic are we weak in" is the question the loop exists to answer. An
LLM (EXAM_PROVIDER/EXAM_MODEL, a separate identity from the generator and the
judge) drafts candidates; a person accepts, edits or rejects each one on the
dashboard's Exam tab; accepted questions land in the bank with who accepted
them and when. Nothing an LLM writes reaches the bank unread.

THE SPLIT, AND WHY IT IS THE MOST IMPORTANT LINE HERE
-----------------------------------------------------
Every accepted question gets a qid — the sha256 of its normalised text — and
scripts/diagnose.py::split_of(qid), the same function and salt as every
benchmark, puts it in the report half or the diagnose half. The published
per-topic score comes from the report half. The step that chooses what to
train may read only the diagnose half. The exam is our own generated
questions, so nothing external protects it; without the split the loop would
train on its own test with no outside benchmark to catch it.

Layout under --root (default $BENCH_ROOT/exam):
    candidates/<topic>.jsonl   drafts awaiting a decision (status candidate|accepted|rejected)
    bank/<topic>.jsonl         accepted questions: qid, prompt, reference, accepted_by, accepted_at
    tasks/exam_<topic>.jsonl   what the harness runs (built from the bank), + .yaml + manifest.json
    tasks/fr_control_mmlu.*    diagnose-half MMLU re-asked open-ended — unchanged from phase 5
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import categories as _categories  # noqa: E402
import diagnose as dx  # noqa: E402

SEED_DIR = REPO / "eval_tasks" / "fr"
RUBRIC_DIR = SEED_DIR / "rubrics"
TOPICS = _categories.category_order()                  # the exam's spine
CONTROL_TASK = "fr_control_mmlu"
CONTROL_PER_CATEGORY = 10
CONTROL_SUFFIX = " Answer in one or two sentences, without listing options."
TARGET_PER_TOPIC = 60
CANDIDATES_PER_REQUEST = 4
MIGRATED_TOPIC = _categories.OTHER      # the four skill suites are not topics; they live here
SKILL_SUITES = ["instruction_following", "factual_accuracy", "reasoning", "cultural"]
# what service/llm.py::local_mark stamps on anything a local model drafted
PROVISIONAL_KEYS = ("provisional", "provisional_reason", "base_url", "served_model", "weights")


TASK_PREFIX = "exam_"


def topic_task(topic: str) -> str:
    return f"{TASK_PREFIX}{_categories.topic_slug(topic)}"


def task_slug(task: str) -> str:
    """The slug inside a task name — exactly what topic_task built it from.
    A per-topic rubric is named after it (rubrics/<slug>.md), so the rubric
    file and the task it grades cannot drift apart."""
    return task[len(TASK_PREFIX):] if task.startswith(TASK_PREFIX) else task


def exam_tasks() -> list[str]:
    return [topic_task(t) for t in TOPICS]


ALL_TASKS = exam_tasks() + [CONTROL_TASK]
TASK_TOPIC = {topic_task(t): t for t in TOPICS}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", (s or "").lower())).strip()


def qid_of(prompt: str) -> str:
    """The question's identity: a hash of its normalised text, so rewording
    that changes meaning changes the qid, and punctuation does not."""
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()


def half_of(qid: str) -> str:
    return dx.split_of(qid)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def bank_dir(root: Path) -> Path:
    return Path(root) / "bank"


def candidates_dir(root: Path) -> Path:
    return Path(root) / "candidates"


def tasks_dir(root: Path) -> Path:
    return Path(root) / "tasks"


def _read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                 encoding="utf-8")


def load_bank(root: Path) -> dict[str, list[dict]]:
    return {t: _read(bank_dir(root) / f"{_categories.topic_slug(t)}.jsonl") for t in TOPICS}


def bank_qids(root: Path) -> set[str]:
    return {r["qid"] for rows in load_bank(root).values() for r in rows}


def append_bank(root: Path, rec: dict) -> None:
    p = bank_dir(root) / f"{_categories.topic_slug(rec['topic'])}.jsonl"
    rows = _read(p)
    if any(r["qid"] == rec["qid"] for r in rows):
        return
    rows.append(rec)
    _write(p, rows)


def load_candidates(root: Path, topic: str | None = None, status: str | None = None) -> list[dict]:
    topics = [topic] if topic else TOPICS
    out = []
    for t in topics:
        for r in _read(candidates_dir(root) / f"{_categories.topic_slug(t)}.jsonl"):
            if status is None or r.get("status") == status:
                out.append(r)
    return out


def _save_candidates(root: Path, topic: str, rows: list[dict]) -> None:
    _write(candidates_dir(root) / f"{_categories.topic_slug(topic)}.jsonl", rows)


# ---------------------------------------------------------------------------
# drafting: an LLM writes candidates; a person decides
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = (
    "You write exam questions for a small language model's written exam. You are given a "
    "topic, a brief, the rubric the answers will be graded against, and a few accepted "
    "questions as examples of the target. Write NEW questions that a well-read person could "
    "answer in two to five sentences from understanding, not recall of a single fact: ask for "
    "a mechanism, a distinction, an application, a consequence. Vary the form — no two "
    "questions in a set should share a template — and vary the difficulty. Never copy or "
    "closely paraphrase the examples or any known exam. Each question needs a reference "
    "answer that names the substance a full-marks answer must contain, not model wording. "
    "Reply with one JSON array of objects {\"prompt\": ..., \"reference\": ..., \"notes\": "
    "<one line for the curator: what it tests, what a 2 looks like>} and nothing else.")

TOPIC_BRIEFS = {
    "economics": "micro and macro fundamentals: incentives, markets, prices, trade, policy effects",
    "law": "legal reasoning and concepts: liability, contract, jurisdiction, rights, procedure — "
           "jurisdiction-neutral or stated",
    "medicine & health": "physiology, disease mechanisms, nutrition, public health — no dosing advice",
    "mathematics": "reasoning about quantities, structures, proofs and estimation; answers in prose",
    "computer science": "algorithms, data, systems, security concepts; language-agnostic",
    "physics & engineering": "mechanisms and estimates in the physical world; units matter",
    "chemistry & biology": "molecular and organismal mechanisms; cause before name",
    "history": "causes, consequences and evidence, dated and placed",
    "philosophy & religion": "arguments, distinctions and traditions, stated fairly",
    "politics & government": "institutions, processes and incentives; comparative where possible",
    "psychology & sociology": "findings, methods and their limits; distinguish claim from evidence",
    "business & accounting": "how organisations decide, account and fail",
    "geography & world facts": "places, processes and the reasons behind patterns",
    "language & logic": "structure of arguments and of language; fallacies, inference, form",
    "other": "cross-topic questions that fit no single heading",
}


def rubric_text(name: str = "exam") -> str:
    return (RUBRIC_DIR / f"{name}.md").read_text(encoding="utf-8")


def diagnose_half_examples(root: Path, topic: str, k: int = 3) -> list[str]:
    """Examples for the drafting prompt come from the DIAGNOSE half only. A
    report-half question never leaves the bank in any LLM request."""
    rows = [r for r in load_bank(root).get(topic, []) if half_of(r["qid"]) == "diagnose"]
    rows.sort(key=lambda r: r["qid"])
    return [r["prompt"] for r in rows[:k]]


def candidates_per_request(provider: str | None = None) -> int:
    """How many questions one request asks for. A LOCAL model gets one: asked
    for four, Gemma-4-E4B answers with one object, or with a list of bare
    question strings and no reference answers — shapes nothing can curate.
    Asked for one it answers with one question, and the batch simply gets a
    row per question, which the on-disk resume does not care about."""
    from service import config
    p = config.EXAM_PROVIDER if provider is None else provider
    return 1 if p == "local" else CANDIDATES_PER_REQUEST


def draft_requests(root: Path, topic: str, n: int, per_request: int | None = None):
    from service import llm
    rubric = rubric_text()
    examples = diagnose_half_examples(root, topic)
    offset = len(load_bank(root).get(topic, [])) + len(load_candidates(root, topic))
    per_request = per_request or candidates_per_request()
    reqs = []
    for k, start in enumerate(range(0, n, per_request)):
        m = min(per_request, n - start)
        user = (f"Topic: {topic}\nBrief: {TOPIC_BRIEFS.get(topic, 'general knowledge in this topic')}\n\n"
                f"Rubric the answers will be graded against:\n{rubric.strip()}\n\n"
                + ("Accepted questions in this topic, as examples of the target (do not copy):\n"
                   + "\n".join(f"- {e}" for e in examples) + "\n\n" if examples else
                   "No accepted questions in this topic yet.\n\n")
                + (f"Write 1 new question. Set {k + 1}: make it differ in form from any other set."
                   if m == 1 else
                   f"Write {m} new questions. Set {k + 1}: make them differ in form from any "
                   f"other set."))
        reqs.append(llm.Request(custom_id=f"exam:{_categories.topic_slug(topic)}:{k}",
                                system=DRAFT_SYSTEM, user=user, max_tokens=3000, json=True,
                                meta={"kind": "exam", "topic": topic, "count": m,
                                      "start": offset + start,
                                      "example_qids": [qid_of(e) for e in examples]}))
    return reqs


def parse_candidates(text: str) -> tuple[list[dict], str]:
    """(candidates, why nothing usable came back). Three shapes are a
    question, because a small model under JSON mode sends all three: the
    array that was asked for, a lone object when one question was asked for,
    and an object wrapping the array ({"questions": [...]}) — JSON mode may
    only return an object, so a model asked for an array often wraps it.

    A wrapped list of bare STRINGS is not a candidate, however many it holds:
    no reference answer means nothing to grade against and nothing for a
    curator to read. That comes back as the reason instead, so a partial
    failure is visible rather than silently zero."""
    from service import llm
    arr = llm.extract_array(text)
    if arr is None:
        obj = llm.extract_json(text)
        arr = [obj] if isinstance(obj, dict) else None
    if arr is None:
        return [], "the reply held no JSON object or array"
    out = []
    for o in arr:
        if not isinstance(o, dict):
            continue
        p, r = str(o.get("prompt") or "").strip(), str(o.get("reference") or "").strip()
        if len(p) < 15 or not r:
            continue
        out.append({"prompt": p[:1200], "reference": r[:1200],
                    "notes": str(o.get("notes") or "").strip()[:400]})
    if out:
        return out, ""
    kinds = sorted({type(o).__name__ for o in arr}) or ["nothing"]
    return [], (f"the reply held {len(arr)} {'/'.join(kinds)} item(s), none of them a question "
                f"with both a prompt and a reference answer")


def _batches_log(root: Path) -> Path:
    return candidates_dir(root) / "_batches.jsonl"


def draft(root: Path, backend, topics: list[str] | None = None, per_topic: int = 8,
          wait: bool = True, poll_s: float = 30.0, timeout_s: float = 6 * 3600) -> dict:
    """One batch for every topic asked; candidates written for curation, never
    into the bank. Returns {batch_id, written} — or {batch_id, pending: True}
    when not waiting; `fetch` finishes it."""
    root = Path(root)
    topics = topics or TOPICS
    per_request = candidates_per_request(backend.name)
    reqs = [r for t in topics for r in draft_requests(root, t, per_topic, per_request)]
    if not reqs:
        return {"batch_id": None, "written": {}, "unusable": {}}
    bid = backend.submit(reqs)
    candidates_dir(root).mkdir(parents=True, exist_ok=True)
    with open(_batches_log(root), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"batch_id": bid, "drafted_by": backend.id, "topics": topics,
                             "per_topic": per_topic, "n_requests": len(reqs),
                             "submitted_at": time.time(), "status": "submitted"}) + "\n")
    if not wait:
        return {"batch_id": bid, "pending": True, "n_requests": len(reqs)}
    t0 = time.time()
    while True:
        state, _ = backend.status(bid)
        if state == "done":
            break
        if state == "failed":
            raise RuntimeError(f"batch {bid} failed")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"batch {bid} still pending after {timeout_s:.0f}s; resume with "
                               f"exam_build.py fetch {bid}")
        time.sleep(poll_s)
    return fetch(root, backend, bid)


def fetch(root: Path, backend, batch_id: str) -> dict:
    from service import llm
    root = Path(root)
    results = backend.fetch(batch_id)
    stamp = llm.provisional(backend, "drafted")       # nothing unless the writer was local
    existing = bank_qids(root) | {c["qid"] for c in load_candidates(root)}
    written: dict[str, int] = collections.Counter()
    by_topic: dict[str, list[dict]] = collections.defaultdict(list)
    # a reply nothing could be read out of, by custom_id: the caller prints
    # how many, so a model answering in a shape we cannot curate is visible
    # rather than arriving as "no candidates came back"
    unusable: dict[str, str] = {}
    for cid_, res in sorted(results.items()):
        if not cid_.startswith("exam:"):
            continue
        if res.error:
            unusable[cid_] = res.error[:300]
            continue
        slug = cid_.split(":")[1]
        topic = next((t for t in TOPICS if _categories.topic_slug(t) == slug), None)
        if topic is None:
            continue
        cands, why = parse_candidates(res.text)
        if why:
            unusable[cid_] = why
        for c in cands:
            q = qid_of(c["prompt"])
            if q in existing:
                continue
            existing.add(q)
            by_topic[topic].append({"cid": "c_" + uuid.uuid4().hex[:12], "qid": q, "topic": topic,
                                    **c, "batch_id": batch_id, "drafted_by": backend.id,
                                    "drafted_at": time.time(), "status": "candidate", **stamp})
    for topic, rows in by_topic.items():
        cur = load_candidates(root, topic)
        _save_candidates(root, topic, cur + rows)
        written[topic] += len(rows)
    rows = _read(_batches_log(root))
    for r in rows:
        if r["batch_id"] == batch_id:
            r.update(status="fetched", fetched_at=time.time(), written=dict(written),
                     unusable=unusable)
    _write(_batches_log(root), rows)
    return {"batch_id": batch_id, "written": dict(written), "unusable": unusable}


# ---------------------------------------------------------------------------
# curation: the human step
# ---------------------------------------------------------------------------

def _find_candidate(root: Path, cid: str) -> tuple[str, list[dict], dict]:
    for t in TOPICS:
        rows = load_candidates(root, t)
        for r in rows:
            if r["cid"] == cid:
                return t, rows, r
    raise KeyError(f"no candidate {cid}")


def accept(root: Path, cid: str, approver: str, prompt: str | None = None,
           reference: str | None = None, notes: str | None = None) -> dict:
    """A candidate becomes a bank question only here, only with a name. The
    text the approver settled on is what gets hashed, so an edit is a new qid."""
    approver = (approver or "").strip()
    if not approver:
        raise ValueError("accepting a question needs a name — the record of who decided")
    topic, rows, c = _find_candidate(root, cid)
    if c.get("status") != "candidate":
        raise ValueError(f"candidate {cid} is already {c.get('status')}")
    p = (prompt if prompt is not None else c["prompt"]).strip()
    r = (reference if reference is not None else c["reference"]).strip()
    if len(p) < 15 or not r:
        raise ValueError("a question needs a prompt and a reference answer")
    edited = p != c["prompt"] or r != c["reference"]
    rec = {"qid": qid_of(p), "topic": topic, "prompt": p, "reference": r,
           "notes": (notes if notes is not None else c.get("notes", "")).strip(),
           "source": "llm-draft", "drafted_by": c.get("drafted_by", ""),
           "batch_id": c.get("batch_id", ""), "cid": cid,
           "accepted_by": approver, "accepted_at": time.time(), "edited": edited,
           # a local writer's mark stays on the record; a person still read it
           **{k: c[k] for k in PROVISIONAL_KEYS if k in c}}
    if rec["qid"] in bank_qids(root):
        raise ValueError("an identical question is already in the bank")
    append_bank(root, rec)
    c.update(status="accepted", decided_by=approver, decided_at=rec["accepted_at"],
             edited=edited, accepted_qid=rec["qid"])
    _save_candidates(root, topic, rows)
    return rec


def reject(root: Path, cid: str, approver: str, reason: str = "") -> dict:
    approver = (approver or "").strip()
    if not approver:
        raise ValueError("rejecting a question needs a name — the record of who decided")
    topic, rows, c = _find_candidate(root, cid)
    if c.get("status") != "candidate":
        raise ValueError(f"candidate {cid} is already {c.get('status')}")
    c.update(status="rejected", decided_by=approver, decided_at=time.time(),
             reason=(reason or "").strip()[:500])
    _save_candidates(root, topic, rows)
    return c


# ---------------------------------------------------------------------------
# importing a human-written bank
# ---------------------------------------------------------------------------

# the order metadata is written in, so the same item always hashes the same
# way. Subject carries sex and age group in brackets; only present fields
# appear at all.
META_ORDER = ("acuity", "intent", "domain", "subject", "style")
META_LABEL = {"acuity": "Acuity", "intent": "Intent", "domain": "Domain",
              "subject": "Subject", "style": "Style"}


def metadata_reference(item: dict) -> str:
    """The ground truth an item carries instead of a model answer: what the
    rubric asks the judge to check the answer against — above all the acuity.

        Acuity: emergency. Intent: symptom_assessment_triage.
        Domain: cardiovascular. Subject: self (male, 45-59). Style: telegraphic.
    """
    bits = []
    for field in META_ORDER:
        value = str(item.get(field) or "").strip()
        if field == "subject":
            who = [str(item.get(k) or "").strip() for k in ("sex", "age_group")]
            who = [w for w in who if w]
            if value and who:
                bits.append(f"Subject: {value} ({', '.join(who)}).")
            elif value:
                bits.append(f"Subject: {value}.")
            elif who:
                bits.append(f"Subject: {', '.join(who)}.")
            continue
        if value:
            bits.append(f"{META_LABEL[field]}: {value}.")
    return " ".join(bits)


def import_bank(root: Path, path: Path, topic: str, approver: str,
                source: str = "import") -> dict:
    """A human-written bank, straight into the bank. These questions were
    written and curated by their author, which is what the Exam tab's accept
    step exists to establish — so the author is the approver, on the record.
    Idempotent: a qid already in the bank is skipped. Returns the counts and
    the split, which is what the caller prints."""
    approver = (approver or "").strip()
    if not approver:
        raise ValueError("importing a bank needs a name — the record of who stands behind it")
    if topic not in TOPICS:
        raise ValueError(f"{topic!r} is not an exam topic: {', '.join(TOPICS)}")
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{path} is not a JSON array of question objects")
    have = bank_qids(root)
    out = {"topic": topic, "source": source, "imported": 0, "skipped": 0, "invalid": 0,
           "report": 0, "diagnose": 0, "acuity": collections.Counter()}
    for it in items:
        prompt = str((it or {}).get("prompt") or "").strip() if isinstance(it, dict) else ""
        if len(prompt) < 15:
            out["invalid"] += 1
            continue
        q = qid_of(prompt)
        if q in have:
            out["skipped"] += 1
            continue
        meta = {k: v for k, v in it.items() if k not in ("prompt", "reference", "notes")}
        line = metadata_reference(it)
        own = str(it.get("reference") or "").strip()
        reference = (f"{own} {line}".strip() if own else line)
        append_bank(root, {"qid": q, "topic": topic, "prompt": prompt, "reference": reference,
                           "notes": str(it.get("notes") or "").strip(), "meta": meta,
                           "source": source, "accepted_by": approver,
                           "accepted_at": time.time(), "edited": False})
        have.add(q)
        out["imported"] += 1
        out[half_of(q)] += 1
        if meta.get("acuity"):
            out["acuity"][str(meta["acuity"])] += 1
    out["acuity"] = dict(sorted(out["acuity"].items()))
    return out


def migrate_seeds(root: Path, seed_dir: Path = SEED_DIR, approver: str = "migration") -> int:
    """The four skill suites' items, as they are, into the bank under `other`
    with their skill kept on the record. Nothing is thrown away; nothing is
    reworded. Idempotent."""
    have = bank_qids(root)
    n = 0
    for skill in SKILL_SUITES:
        for it in _read(seed_dir / f"fr_{skill}.jsonl"):
            q = qid_of(it["prompt"])
            if q in have:
                continue
            append_bank(root, {"qid": q, "topic": MIGRATED_TOPIC, "prompt": it["prompt"],
                               "reference": it["reference"], "notes": it.get("notes", ""),
                               "source": "migrated", "skill": skill,
                               "migrated_from": it.get("id", f"fr_{skill}"),
                               "accepted_by": approver, "accepted_at": time.time(), "edited": False})
            have.add(q)
            n += 1
    return n


def summary(root: Path) -> dict:
    out = {}
    bank = load_bank(root)
    for t in TOPICS:
        rows = bank.get(t, [])
        halves = collections.Counter(half_of(r["qid"]) for r in rows)
        out[t] = {"task": topic_task(t), "accepted": len(rows), "report": halves["report"],
                  "diagnose": halves["diagnose"], "target": TARGET_PER_TOPIC,
                  "pending": len(load_candidates(root, t, "candidate")),
                  "rejected": len(load_candidates(root, t, "rejected"))}
    return out


def public_bank(root: Path, topic: str | None = None) -> list[dict]:
    """The bank as anything outside curation may see it: report-half questions
    are listed by qid and topic only. Their text is not shown, exported or
    placed in a request — that is the whole split."""
    out = []
    for t, rows in load_bank(root).items():
        if topic and t != topic:
            continue
        for r in sorted(rows, key=lambda r: r["qid"]):
            h = half_of(r["qid"])
            base = {k: r.get(k) for k in ("qid", "topic", "source", "skill", "accepted_by",
                                          "accepted_at", "edited")}
            base["half"] = h
            if h == "diagnose":
                base.update(prompt=r["prompt"], reference=r["reference"], notes=r.get("notes", ""),
                            meta=r.get("meta") or None)
            else:
                base.update(prompt=None, reference=None, notes=None, meta=None,
                            withheld="report half — never shown, never exported")
            out.append(base)
    return out


# ---------------------------------------------------------------------------
# the control set — unchanged from phase 5
# ---------------------------------------------------------------------------

def mmlu_docs(results_root: Path) -> dict[str, dict]:
    """Every distinct MMLU document on disk, by doc_hash — from whichever
    model's per-item log has them; the documents are the same for all."""
    out: dict[str, dict] = {}
    for model_dir in sorted(p for p in Path(results_root).iterdir() if p.is_dir()):
        for task_dir in sorted(model_dir.glob("mmlu_*shot")):
            if not re.fullmatch(r"mmlu_\d+shot", task_dir.name):
                continue                   # not mmlu_perm: rotated options are not the question
            for f in dx.newest_per_subtask(sorted(task_dir.rglob("samples_mmlu_*.jsonl"))):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        dh, doc = rec.get("doc_hash"), rec.get("doc") or {}
                        if dh and dh not in out and doc.get("question") and \
                                isinstance(doc.get("choices"), list):
                            out[dh] = doc
        if out:
            break
    return out


def control_items(docs: dict[str, dict], per_category: int = CONTROL_PER_CATEGORY) -> list[dict]:
    """Diagnose-half MMLU questions, stratified by human category, deterministic."""
    by_cat: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for dh in sorted(docs):
        if dx.split_of(dh) != "diagnose":                  # the rule
            continue
        doc = docs[dh]
        cat = _categories.categorize(doc.get("subject", "")) or _categories.OTHER
        by_cat[cat].append((dh, doc))
    items = []
    for cat in TOPICS:
        for dh, doc in by_cat.get(cat, [])[:per_category]:
            try:
                gold = str(doc["choices"][int(doc["answer"])])
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            items.append({"id": f"{CONTROL_TASK}-{dh[:12]}", "category": cat,
                          "subject": doc.get("subject", ""),
                          "prompt": str(doc["question"]).strip() + CONTROL_SUFFIX,
                          "reference": gold, "mmlu_doc_hash": dh})
    return items


# ---------------------------------------------------------------------------
# build: what the harness runs
# ---------------------------------------------------------------------------

def build(results_root: Path, root: Path, per_category: int = CONTROL_PER_CATEGORY) -> dict:
    root = Path(root)
    out = tasks_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    template = (SEED_DIR / "_fr_template_yaml").read_text(encoding="utf-8")
    manifest = {"tasks": {}, "rubrics": {}, "topics": {}, "split_salt": dx.SPLIT_SALT}
    written_tasks = []
    for topic, rows in load_bank(root).items():
        task = topic_task(topic)
        p = out / f"{task}.jsonl"
        if not rows:                          # an empty task would fail in the harness
            for stale in (p, out / f"{task}.yaml"):
                if stale.exists():
                    stale.unlink()
            continue
        # meta rides along when the bank has it (an imported item's acuity,
        # intent, domain): the judge reads the doc, and a per-acuity table is
        # the first thing the author of a medical bank looks at
        items = [{"id": f"{task}-{r['qid'][:12]}", "qid": r["qid"], "topic": topic,
                  "category": topic, "prompt": r["prompt"], "reference": r["reference"],
                  **({"meta": r["meta"]} if r.get("meta") else {})}
                 for r in sorted(rows, key=lambda r: r["qid"])]
        _write(p, items)
        halves = collections.Counter(half_of(r["qid"]) for r in rows)
        manifest["tasks"][task] = {"items": len(items), "sha256": sha256_file(p),
                                   "report": halves["report"], "diagnose": halves["diagnose"],
                                   "topic": topic, "built_from": "bank"}
        manifest["topics"][task] = topic
        written_tasks.append(task)
    items = control_items(mmlu_docs(results_root), per_category)
    p = out / f"{CONTROL_TASK}.jsonl"
    _write(p, items)
    manifest["tasks"][CONTROL_TASK] = {
        "items": len(items), "sha256": sha256_file(p), "report": 0, "diagnose": len(items),
        "per_category": dict(collections.Counter(it["category"] for it in items)),
        "built_from": "mmlu diagnose half", "topic": None}
    written_tasks.append(CONTROL_TASK)
    for task in written_tasks:
        yaml = template.replace("__ITEMS_PATH__", str((out / f"{task}.jsonl").resolve()))
        (out / f"{task}.yaml").write_text(f"task: {task}\n" + yaml, encoding="utf-8")
    for r in sorted(RUBRIC_DIR.glob("*.md")):
        manifest["rubrics"][r.stem] = sha256_file(r)
    manifest["built_at"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                       encoding="utf-8")
    return manifest


def _backend():
    from service import config, llm
    why = llm.blocked("exam")
    if why:
        print(why, file=sys.stderr)
        raise SystemExit(2)
    try:
        return llm.client("exam"), config
    except llm.LLMError as e:          # a local server that is down, or serves another id
        print(e, file=sys.stderr)
        raise SystemExit(2) from None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help="the exam directory (default $BENCH_ROOT/exam, or ./exam)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="ask the exam LLM for candidates, per topic, for curation")
    d.add_argument("--topic", action="append", default=[], help="repeatable; default every topic")
    d.add_argument("--per-topic", type=int, default=8)
    d.add_argument("--no-wait", action="store_true", help="submit and print the batch id")
    f = sub.add_parser("fetch", help="collect a batch submitted with --no-wait")
    f.add_argument("batch_id")
    sub.add_parser("migrate", help="the four skill suites' 40 items into the bank, under 'other'")
    i = sub.add_parser("import", help="a human-written bank (JSON array) straight into the bank")
    i.add_argument("path", type=Path, help="JSON array of objects with at least a prompt")
    i.add_argument("--topic", required=True, help="the exam topic these belong to")
    i.add_argument("--approver", required=True, help="who stands behind them — recorded per item")
    i.add_argument("--source", default="import", help="provenance tag kept on every item")
    sub.add_parser("summary", help="per topic: accepted, halves, pending")
    b = sub.add_parser("build", help="write the harness tasks from the bank + the MMLU control set")
    b.add_argument("results", type=Path, help="results/full (source of the control set)")
    b.add_argument("--per-category", type=int, default=CONTROL_PER_CATEGORY)
    a = ap.parse_args()
    root = a.root or Path(os.environ.get("BENCH_ROOT", ".")) / "exam"
    if a.cmd == "migrate":
        print(f"migrated {migrate_seeds(root)} items into {bank_dir(root)}")
        return 0
    if a.cmd == "import":
        try:
            r = import_bank(root, a.path, a.topic, a.approver, a.source)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{r['topic']}: imported {r['imported']}, skipped {r['skipped']} already in the "
              f"bank" + (f", {r['invalid']} without a usable prompt" if r["invalid"] else "")
              + f" — report {r['report']} / diagnose {r['diagnose']}")
        if r["acuity"]:
            print("acuity: " + ", ".join(f"{k} {v}" for k, v in r["acuity"].items()))
        print(f"they are in the bank ({bank_dir(root)}); rebuild the tasks with "
              f"exam_build.py build <results> --root {root}")
        return 0
    if a.cmd == "summary":
        for t, s in summary(root).items():
            print(f"{t:26} accepted {s['accepted']:3} (report {s['report']:3} / diagnose "
                  f"{s['diagnose']:3})  pending {s['pending']:3}  target {s['target']}")
        return 0
    if a.cmd == "build":
        m = build(a.results, root, a.per_category)
        for t, v in m["tasks"].items():
            print(f"{t:36} {v['items']:4} items  (report {v['report']}, diagnose {v['diagnose']})")
        print(f"\nwrote {tasks_dir(root)} — suite=judged runs these")
        return 0
    backend, _ = _backend()
    if a.cmd == "fetch":
        # wait rather than fail on an unfinished batch: a local batch only
        # makes progress while some process is polling it, and this is one
        state, detail = backend.status(a.batch_id)
        while state == "pending":
            print(f"batch {a.batch_id}: {detail}", file=sys.stderr)
            time.sleep(5 if backend.name == "local" else 30)
            state, detail = backend.status(a.batch_id)
        if state == "failed":
            print(f"batch {a.batch_id} failed: {detail}", file=sys.stderr)
            return 1
        r = fetch(root, backend, a.batch_id)
    else:
        r = draft(root, backend, a.topic or None, a.per_topic, wait=not a.no_wait)
        if r.get("pending"):
            print(f"batch {r['batch_id']} submitted ({r['n_requests']} requests); "
                  + ("a local batch runs only while a process polls it, so collect it now: "
                     if backend.name == "local" else "when it completes: ")
                  + f"exam_build.py fetch --root {root} {r['batch_id']}")
            return 0
    print(f"batch {r['batch_id']}: candidates written per topic — "
          + (", ".join(f"{t}: {n}" for t, n in sorted(r["written"].items())) or "none"))
    bad = r.get("unusable") or {}
    if bad:
        cid, why = sorted(bad.items())[0]
        print(f"{len(bad)} repl{'y' if len(bad) == 1 else 'ies'} could not be read, e.g. "
              f"{cid}: {why}", file=sys.stderr)
    print("curate them on the dashboard's Exam tab; they are not in the bank until someone "
          "accepts them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
