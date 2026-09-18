"""Find the gap, and make data for it — with the report half out of reach.

The pipeline, and where each safety property lives:

  diagnose (categories)
    → PROPOSAL     an LLM reads DIAGNOSE-HALF failures for one weak category
                   and proposes a skill spec: what is missing, not which
                   questions were missed.           failures_for() filters by
                                                     split_of() == "diagnose"
    → HUMAN        approves / edits / rejects in the dashboard's Review tab.
                   The spec text is the airlock; a name is recorded.
    → GENERATOR    receives ONLY the approved spec, the category, a count, a
                   format and a style constraint.  generation_requests() takes
                                                     no item, no hash, no model
    → GATE         13-gram overlap against every benchmark item on disk, both
                   halves; near-duplicates collapsed.  service/contamination.py
    → PROVENANCE   who, what, which model, which prompt, which batch, hashes.
    → TAINT        a training run that consumes the dataset says so; its
                   checkpoints lose the task from their official average.

The proposal request is the only place benchmark text meets the LLM, and it
is diagnose-half text about ONE model, ONE task, ONE category. The failed
items never reach the generator: the request that makes data is built from
the spec string and nothing else, and test_proposals.py proves both with the
recorded request bodies.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import config, llm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import categories as _categories  # noqa: E402
import diagnose as dx  # noqa: E402

MAX_FAILURES_IN_PROMPT = 60      # enough to see a pattern; one batch item either way
EXAMPLES_SHOWN = 8               # what the Review tab shows of what the LLM saw
GEN_ITEMS_PER_REQUEST = 10
FORMATS = ("mc", "free")

BUCKET_WORDS = {"at_chance": "no preference among the options",
                "confident_wrong": "confidently picked a wrong option",
                "near_miss": "correct option ranked second, narrowly",
                "wrong": "wrong"}


# ---------------------------------------------------------------------------
# what the proposal LLM is allowed to see
# ---------------------------------------------------------------------------

def failures_for(model_dir: Path, task: str, category: str,
                 limit: int = MAX_FAILURES_IN_PROMPT) -> tuple[list[dict], dict]:
    """The DIAGNOSE-half items this model got wrong in this category, from the
    per-item log on disk. Deterministic: sorted by doc_hash, first `limit`.
    Returns (items, counts) where counts covers every diagnose-half item in
    the category so 'share of failures' means something."""
    out: list[dict] = []
    counts = {"diagnose_items": 0, "diagnose_wrong": 0, "buckets": {}}
    task_dirs = sorted(d for d in model_dir.glob(f"{task}_*shot")
                       if re.fullmatch(rf"{re.escape(task)}_\d+shot", d.name))
    files = dx.newest_per_subtask(sorted(f for d in task_dirs for f in d.rglob("samples_*.jsonl")))
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pm = dx.primary_metric(rec)
                dh = rec.get("doc_hash") or rec.get("prompt_hash")
                if pm is None or not dh:
                    continue
                if dx.split_of(str(dh)) != "diagnose":     # the whole safety property
                    continue
                doc = rec.get("doc") or {}
                group = doc.get("subject") or doc.get("category") or "—"
                if (_categories.categorize(group) or _categories.OTHER) != category:
                    continue
                metric, value = pm
                right = value >= 0.5
                counts["diagnose_items"] += 1
                if right:
                    continue
                counts["diagnose_wrong"] += 1
                resps = rec.get("filtered_resps") or rec.get("resps") or []
                lps = []
                for r in resps:
                    if isinstance(r, list) and r and isinstance(r[0], list):
                        r = r[0]
                    lps.append(dx._f(r[0] if isinstance(r, list) else r))
                probs, pick, bucket = [], None, "wrong"
                if len(lps) >= 2:
                    if metric == "acc_norm":
                        L = dx.norm_lengths(doc, len(lps))
                        if L:
                            lps = [x / l for x, l in zip(lps, L)]
                    probs = dx.softmax(lps)
                    ci = dx.target_index(rec, len(lps))
                    bucket = dx.bucket(probs, ci, False)
                    pick = max(range(len(probs)), key=lambda i: probs[i])
                else:
                    ci = dx.target_index(rec, 4)
                ch = doc.get("choices")
                if isinstance(ch, dict):
                    ch = ch.get("text")
                ch = ch if isinstance(ch, list) else []
                counts["buckets"][bucket] = counts["buckets"].get(bucket, 0) + 1
                out.append({
                    "doc_hash": str(dh), "subject": group, "bucket": bucket,
                    "question": str(doc.get("question") or doc.get("query") or doc.get("ctx")
                                    or doc.get("goal") or "")[:600],
                    "choices": [str(c)[:200] for c in ch],
                    "chose": str(ch[pick])[:200] if pick is not None and pick < len(ch) else None,
                    "answer": str(ch[ci])[:200] if ci is not None and ci < len(ch) else None,
                    "p": round(max(probs), 3) if probs else None,
                })
    out.sort(key=lambda x: x["doc_hash"])
    return out[:limit], counts


PROPOSAL_SYSTEM = (
    "You analyse a small language model's failures on a benchmark category and describe "
    "the SKILL or KNOWLEDGE that is missing. You are writing a specification for a data "
    "generator that will never see these questions: describe what the model cannot do, "
    "in general terms a teacher would use, never the questions themselves. Do not quote, "
    "paraphrase or allude to any specific item. Reply with one JSON object and nothing "
    "else: {\"spec\": <one to three sentences>, \"share_explained\": <0..1, the share of "
    "the failures below your spec accounts for>, \"patterns\": [<two or three short "
    "failure patterns you saw>]}.")


def proposal_request(pid: int, model: str, task: str, category: str,
                     failures: list[dict], counts: dict) -> llm.Request:
    lines = [f"Model: {model}", f"Benchmark: {task}", f"Category: {category}",
             f"Diagnosis-half items in this category: {counts['diagnose_items']}; "
             f"wrong: {counts['diagnose_wrong']}; shown below: {len(failures)}.", "",
             "Each failure: the question, the options, what the model chose, the correct "
             "answer, and how it was wrong.", ""]
    for i, f in enumerate(failures, 1):
        lines.append(f"[{i}] ({f['subject']}; {BUCKET_WORDS.get(f['bucket'], f['bucket'])})")
        lines.append("Q: " + f["question"])
        if f["choices"]:
            lines.append("Options: " + " | ".join(f["choices"]))
        lines.append(f"Chose: {f['chose']}    Correct: {f['answer']}")
        lines.append("")
    lines.append("Write the skill spec now, as the JSON object described.")
    return llm.Request(
        custom_id=f"proposal:{pid}", system=PROPOSAL_SYSTEM, user="\n".join(lines),
        max_tokens=1024,
        meta={"kind": "proposal", "proposal_id": pid, "model": model, "task": task,
              "category": category, "doc_hashes": [f["doc_hash"] for f in failures]})


def parse_proposal(text: str) -> dict:
    obj = llm.extract_json(text)
    if not isinstance(obj, dict) or not str(obj.get("spec") or "").strip():
        raise llm.LLMError("the proposal reply was not a JSON object with a non-empty spec")
    share = obj.get("share_explained")
    try:
        share = max(0.0, min(1.0, float(share)))
    except (TypeError, ValueError):
        share = None
    patterns = [str(p)[:200] for p in (obj.get("patterns") or []) if str(p).strip()][:5]
    return {"spec": str(obj["spec"]).strip()[:2000], "share_explained": share,
            "patterns": patterns}


# ---------------------------------------------------------------------------
# what the generator is allowed to see: the spec, and nothing else
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You write original training items that teach a specific skill to a small language "
    "model. You are given a skill specification, a category name, a count and a format. "
    "Invent fresh scenarios; vary surface form deliberately — sentence length, register, "
    "question type (definition, application, comparison, calculation, cause and effect), "
    "named entities and settings — so no two items share a template. Never reproduce or "
    "closely paraphrase any existing exam or benchmark question. Reply with one JSON array "
    "of objects and nothing else.")

STYLE = {
    "mc": ("Each object: {\"question\": ..., \"choices\": [exactly four options, one correct, "
           "the correct one in a varying position], \"answer\": <the correct option's text, "
           "verbatim>, \"rationale\": <one or two sentences>}."),
    "free": ("Each object: {\"question\": ..., \"answer\": <a short free-text answer>, "
             "\"rationale\": <one or two sentences>}."),
}


def generation_requests(did: int, spec_text: str, category: str, count: int,
                        fmt: str, seed: int) -> list[llm.Request]:
    """One request per GEN_ITEMS_PER_REQUEST items. Contains the approved spec,
    the category, the count, the format and a style constraint — and no
    benchmark item, hash, model name or score, in any form."""
    reqs = []
    for k, start in enumerate(range(0, count, GEN_ITEMS_PER_REQUEST)):
        n = min(GEN_ITEMS_PER_REQUEST, count - start)
        user = (f"Skill specification:\n{spec_text.strip()}\n\n"
                f"Category: {category}\nFormat: {fmt}\nWrite {n} items.\n{STYLE[fmt]}\n"
                f"Style seed {seed}-{k}: make this set differ in scenario and phrasing from "
                f"any other set you might write for the same specification.")
        reqs.append(llm.Request(
            custom_id=f"gen:{did}:{k}", system=GEN_SYSTEM, user=user, max_tokens=4096,
            meta={"kind": "generation", "dataset_id": did, "count": n, "start": start,
                  "format": fmt}))
    return reqs


def parse_items(text: str, fmt: str) -> list[dict]:
    arr = llm.extract_json(text)
    if not isinstance(arr, list):
        return []
    out = []
    for o in arr:
        if not isinstance(o, dict):
            continue
        q = str(o.get("question") or "").strip()
        a = str(o.get("answer") or "").strip()
        r = str(o.get("rationale") or "").strip()
        if not q or not a:
            continue
        item = {"question": q, "answer": a, "rationale": r}
        if fmt == "mc":
            ch = [str(c).strip() for c in (o.get("choices") or []) if str(c).strip()]
            if len(ch) != 4 or a not in ch:
                continue
            item["choices"] = ch
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# the dataset on disk, and its provenance
# ---------------------------------------------------------------------------

def dataset_dir(did: int) -> Path:
    return config.DATASETS_DIR / str(did)


def dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


def quota_blocked() -> str:
    used = dir_bytes(config.DATASETS_DIR)
    cap = int(config.DATASET_QUOTA_GB * 1e9)
    if used >= cap:
        return (f"dataset storage quota reached ({used / 1e9:.1f} of "
                f"{config.DATASET_QUOTA_GB:g} GB) — delete old datasets first")
    return ""


def write_items(did: int, items: list[dict]) -> tuple[Path, str]:
    d = dataset_dir(did)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "items.jsonl"
    body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    p.write_text(body, encoding="utf-8")
    return p, hashlib.sha256(body.encode("utf-8")).hexdigest()


def identities(generator_id: str) -> dict:
    """All three LLM identities, in every provenance record: the loop is only
    honest if the exam writer, the judge and the generator are not one family."""
    import judge as _judge
    from . import llm
    ex = llm.identity("exam")
    jd = _judge.identity()
    return {"exam_writer": f"{ex[0]}/{ex[1]}" if ex[0] else "",
            "judge": jd["id"], "generator": generator_id,
            "single_provider_loop": _judge.single_provider_loop()}


def provenance(prop: dict, ds: dict, backend_id: str, batch_id: str, prompt_hash: str,
               gate: dict, sha: str, n_generated: int, n_kept: int) -> dict:
    return {
        "dataset_id": ds["id"],
        "source_model": prop["model"],
        "task": prop["task"],
        "category": prop["category"],
        "split": "diagnose",
        "split_salt": dx.SPLIT_SALT,
        "proposal_id": prop["id"],
        "spec_text": prop["spec_text"],
        "edited_text": prop["edited_text"],
        "approved_spec": prop["edited_text"] or prop["spec_text"],
        "proposer": prop["proposer"],
        "approver": prop["approver"],
        "requester": ds["requester"],
        "generator": {"provider": backend_id.split("/", 1)[0],
                      "model": backend_id.split("/", 1)[1] if "/" in backend_id else "",
                      "batch_id": batch_id, "id": backend_id},
        "identities": identities(backend_id),
        "prompt_sha256": prompt_hash,
        "format": ds["fmt"],
        "count_requested": ds["count"],
        "items": {"generated": n_generated, "dropped": n_generated - n_kept, "kept": n_kept},
        "gate": gate,
        "items_sha256": sha,
        "timestamps": {"proposed": prop["created_at"], "approved": prop["approved_at"],
                       "requested": ds["created_at"], "generated": time.time()},
    }


def provenance_complete(p: dict) -> list[str]:
    """Field paths that are empty — a provenance record with a hole is a
    record of nothing. Used by the test and by the poller before it marks a
    dataset ready."""
    holes = []

    def walk(v, path):
        if isinstance(v, dict):
            if not v:
                holes.append(path)
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else k)
        elif v is None or v == "" or v == []:
            holes.append(path)
    walk(p, "")
    # edited_text is legitimately empty when the human approved the spec as
    # written; a clean gate legitimately has no offending n-grams to list; an
    # identity may legitimately be unconfigured, and False is a value
    return [h for h in holes if h not in ("edited_text", "gate.offending_ngrams",
                                          "identities.exam_writer", "identities.judge",
                                          "identities.single_provider_loop")]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
