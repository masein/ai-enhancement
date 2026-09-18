"""Find the gap, and make data for it — with the report half out of reach.

The pipeline, and where each safety property lives:

  the exam      the checkpoint sits it; an LLM judge grades every answer 0-4
                against the rubric AND writes why it scored what it did
    → PROPOSAL  an LLM reads the judge's WRITTEN JUSTIFICATIONS for one weak
                topic's DIAGNOSE-HALF low-scoring answers, and proposes a
                skill spec: what is missing, not which questions were missed.
                justifications_for() filters by split_of(qid) == "diagnose"
                and strips any exam question text the judge quoted — the
                justification is about the answer, not the question
    → HUMAN     approves / edits / rejects in the dashboard's Review tab.
                The spec text is the airlock; a name is recorded.
    → GENERATOR receives ONLY the approved spec, the topic, a count, a format
                and a style constraint, and writes prose DOCUMENTS — not
                question-and-answer pairs, because exam-shaped training data
                is the most direct route to teaching the test there is.
                                            generation_requests() takes no
                                            item, no hash, no model name
    → GATE      13-gram overlap against every benchmark item AND every exam
                question on disk, both halves of each; near-duplicates
                collapsed.                  service/contamination.py
    → PROVENANCE who, what, which judge run, which model, which batch, hashes
    → TAINT     a training run that consumes the dataset says so; its
                checkpoints lose the task from their official average.

The exam already says, per topic, in a person's vocabulary, how good the
model is and why it is not better — so nothing here reads MMLU to decide what
to train. MMLU's distribution findings ride along as a CAUTION beside the
topic (a topic weak on the exam and at chance on MMLU is a different problem
from one weak on the exam alone), never as the evidence and never as a gate.

The proposal request is the only place exam-derived text meets an LLM, and it
carries judged reasoning about DIAGNOSE-half answers only, with question text
removed. The generator's request is built from the approved spec string and
nothing else. test_gap.py proves both from the recorded request bodies.
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
import diagnose as dx  # noqa: E402
import exam_build as _exam  # noqa: E402

MAX_JUSTIFICATIONS = 60          # enough to see a pattern; one batch item either way
EXAMPLES_SHOWN = 8               # what the Review tab shows of what the LLM saw
# Training DOCUMENTS, not question-and-answer pairs. Generating items shaped
# like the exam is the most direct route to teaching the test there is; prose
# does not have that shape. `free` stays for comparison and is not the
# default; `mc` is retired.
DEFAULT_FORMAT = "doc"
FORMATS = ("doc", "free")
# documents are long, so fewer per request
ITEMS_PER_REQUEST = {"doc": 2, "free": 10}
GEN_ITEMS_PER_REQUEST = ITEMS_PER_REQUEST[DEFAULT_FORMAT]
DOC_MIN_WORDS = 120              # shorter than this is a note, not a teaching document
DOC_TARGET_WORDS = 600           # ~800 tokens of body
WEAK_SCORE = 3                   # judge.CORRECT_AT: below this the answer did not land
# Six consecutive words of an exam question inside a justification is a
# quotation, not a coincidence. Redacting a few innocent words costs nothing;
# letting question text through costs the exam.
QUOTE_NGRAM = 6
REDACTED = "[question text removed]"


# ---------------------------------------------------------------------------
# what the proposal LLM is allowed to see
# ---------------------------------------------------------------------------

def _judge_file(model_dir: Path) -> dict | None:
    p = Path(model_dir) / "judge.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def weak_topics(model_dir: Path) -> list[dict]:
    """Every exam topic this model sat, weakest first, from judge.json: the
    published REPORT-half score and count, the diagnose half behind it, what
    the model actually wrote, and the topic's rank. This is the whole
    gap-finding signal — no benchmark item is read to produce it."""
    j = _judge_file(model_dir)
    if not j or j.get("skipped"):
        return []
    rows = []
    for task, t in (j.get("tasks") or {}).items():
        if not task.startswith("exam_"):
            continue
        score = t.get("score_report")
        rows.append({
            "task": task, "topic": _exam.TASK_TOPIC.get(task, task[len("exam_"):]),
            "score_report": score, "n_report": t.get("n_report") or 0,
            "score_diagnose": t.get("score_diagnose"), "n_diagnose": t.get("n_diagnose") or 0,
            "mean": t.get("mean"), "max": t.get("max", 4), "n": t.get("n") or 0,
            "answers": t.get("answers") or {},
            "weak_diagnose": sum(1 for it in (t.get("items") or [])
                                 if it.get("half") == "diagnose" and it.get("graded")
                                 and it.get("score", 0) < WEAK_SCORE),
        })
    rows.sort(key=lambda r: (r["score_report"] is None, r["score_report"], r["task"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["of"] = len(rows)
    return rows


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def question_ngrams(prompts, n: int = QUOTE_NGRAM) -> set[str]:
    out: set[str] = set()
    for p in prompts:
        toks = _norm_words(p)
        for i in range(len(toks) - n + 1):
            out.add(" ".join(toks[i:i + n]))
    return out


def strip_question_quotes(text: str, grams: set[str], n: int = QUOTE_NGRAM) -> str:
    """Remove any run of words the judge lifted from an exam question. The
    n-gram set is built from the topic's questions in BOTH halves: this
    function only ever deletes text, so knowing a report-half question here
    is what keeps it out of the request."""
    words = [(m.group(0).lower(), m.start(), m.end())
             for m in re.finditer(r"[A-Za-z0-9]+", text or "")]
    if len(words) < n or not grams:
        return text
    flagged = [False] * len(words)
    for i in range(len(words) - n + 1):
        if " ".join(w for w, _, _ in words[i:i + n]) in grams:
            for j in range(i, i + n):
                flagged[j] = True
    if not any(flagged):
        return text
    out, pos, i = [], 0, 0
    while i < len(words):
        if flagged[i]:
            j = i
            while j + 1 < len(words) and flagged[j + 1]:
                j += 1
            out.append(text[pos:words[i][1]])
            out.append(REDACTED)
            pos = words[j][2]
            i = j + 1
        else:
            i += 1
    out.append(text[pos:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def topic_question_grams(task: str) -> set[str]:
    """Every question of one exam task, as n-grams, for the strip above."""
    p = _exam.tasks_dir(config.EXAM_DIR) / f"{task}.jsonl"
    if not p.exists():
        return set()
    prompts = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                prompts.append(json.loads(line).get("prompt") or "")
            except json.JSONDecodeError:
                continue
    return question_ngrams(prompts)


def justifications_for(model_dir: Path, task: str,
                       limit: int = MAX_JUSTIFICATIONS) -> tuple[list[dict], dict]:
    """The judge's written reasoning for this topic's DIAGNOSE-half answers
    that did not land, with any quoted question text removed. Deterministic:
    sorted by qid, first `limit`. Returns (items, counts)."""
    j = _judge_file(model_dir)
    counts = {"diagnose_items": 0, "diagnose_weak": 0, "scores": {}}
    if not j or j.get("skipped"):
        return [], counts
    t = (j.get("tasks") or {}).get(task) or {}
    grams = topic_question_grams(task)
    out = []
    for it in sorted(t.get("items") or [], key=lambda x: str(x.get("qid"))):
        if it.get("half") != "diagnose":        # the whole safety property
            continue
        counts["diagnose_items"] += 1
        score = it.get("score")
        counts["scores"][str(score)] = counts["scores"].get(str(score), 0) + 1
        if not it.get("graded") or score is None or score >= WEAK_SCORE:
            continue
        counts["diagnose_weak"] += 1
        text = strip_question_quotes(str(it.get("justification") or ""), grams)
        if not text:
            continue
        out.append({"qid": it.get("qid"), "score": score, "justification": text[:600],
                    "answer_words": it.get("answer_words")})
    return out[:limit], counts


PROPOSAL_SYSTEM = (
    "You read a judge's written assessments of one small language model's answers on one "
    "topic of a written exam, and describe the SKILL or KNOWLEDGE the model is missing. "
    "You are writing a specification for a data generator that will never see the exam: "
    "describe what the model cannot do, in general terms a teacher would use. You are not "
    "given the questions and must not invent them, quote them, or write any exam-shaped "
    "question yourself. Reply with one JSON object and nothing else: "
    "{\"spec\": <one to three sentences>, \"share_explained\": <0..1, the share of the "
    "assessments below your spec accounts for>, \"patterns\": [<two or three short failure "
    "patterns you saw>]}.")


def proposal_request(pid: int, model: str, task: str, topic: str,
                     justifications: list[dict], counts: dict, rubric: str) -> llm.Request:
    """The judge's reasoning, the topic and the rubric. No question text: the
    justification is about the ANSWER, and anything the judge quoted from a
    question has already been stripped."""
    lines = [f"Topic: {topic}", f"Model under assessment: {model}",
             f"Diagnosis-half answers on this topic: {counts['diagnose_items']}; "
             f"scoring below {WEAK_SCORE} of 4: {counts['diagnose_weak']}; "
             f"shown below: {len(justifications)}.", "",
             "The rubric the judge graded against:", rubric.strip(), "",
             "The judge's assessment of each answer that did not land, with its score:", ""]
    for i, f in enumerate(justifications, 1):
        lines.append(f"[{i}] scored {f['score']} of 4 — {f['justification']}")
    lines += ["", "Write the skill spec now, as the JSON object described."]
    return llm.Request(
        custom_id=f"proposal:{pid}", system=PROPOSAL_SYSTEM, user="\n".join(lines),
        max_tokens=1024,
        meta={"kind": "proposal", "proposal_id": pid, "model": model, "task": task,
              "topic": topic, "qids": [f["qid"] for f in justifications]})


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
    "You write original TRAINING DOCUMENTS that teach a specific skill to a small language "
    "model. You are given a skill specification, a topic and a count. A document is prose a "
    "person could learn from — a short explainer, a worked discussion, a piece of reference "
    "writing — not a quiz: never write a question-and-answer pair, a multiple-choice item, "
    "or anything shaped like an exam, because a model trained on exam-shaped text learns the "
    "exam rather than the skill. Invent fresh material and vary surface form deliberately "
    "across the set: register (textbook, briefing note, worked example, dialogue, case "
    "study), length, framing, named entities and settings, so no two documents share a "
    "template. Never reproduce or closely paraphrase any existing exam or benchmark text. "
    "Reply with one JSON array of objects and nothing else.")

STYLE = {
    "doc": (f"Each object: {{\"title\": <a short descriptive title>, \"text\": <the document "
            f"body, around {DOC_TARGET_WORDS} words of continuous prose that teaches the "
            f"specification's skill; paragraphs separated by blank lines; no questions posed "
            f"to the reader, no answer keys, no bullet lists of Q/A>}}."),
    "free": ("Each object: {\"question\": ..., \"answer\": <a short free-text answer>, "
             "\"rationale\": <one or two sentences>}. This format is for comparison only: "
             "question-shaped training data teaches the test more readily than prose does."),
}


def items_per_request(fmt: str) -> int:
    return ITEMS_PER_REQUEST.get(fmt, GEN_ITEMS_PER_REQUEST)


def generation_requests(did: int, spec_text: str, category: str, count: int,
                        fmt: str, seed: int) -> list[llm.Request]:
    """One request per few items. Contains the approved spec, the topic, the
    count, the format and a style constraint — and no benchmark item, no exam
    question, no hash, no model name and no score, in any form."""
    per = items_per_request(fmt)
    reqs = []
    for k, start in enumerate(range(0, count, per)):
        n = min(per, count - start)
        what = "documents" if fmt == "doc" else "items"
        user = (f"Skill specification:\n{spec_text.strip()}\n\n"
                f"Topic: {category}\nFormat: {fmt}\nWrite {n} {what}.\n{STYLE[fmt]}\n"
                f"Style seed {seed}-{k}: make this set differ in scenario, register and "
                f"phrasing from any other set you might write for the same specification.")
        reqs.append(llm.Request(
            custom_id=f"gen:{did}:{k}", system=GEN_SYSTEM, user=user, max_tokens=8192,
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
        if fmt == "doc":
            title = str(o.get("title") or "").strip()
            body = str(o.get("text") or o.get("body") or "").strip()
            # a title alone, or a paragraph too short to teach anything, is not
            # a training document — and neither is a quiz wearing prose
            if not title or len(body.split()) < DOC_MIN_WORDS:
                continue
            out.append({"title": title[:300], "text": body})
            continue
        q = str(o.get("question") or "").strip()
        a = str(o.get("answer") or "").strip()
        r = str(o.get("rationale") or "").strip()
        if not q or not a:
            continue
        out.append({"question": q, "answer": a, "rationale": r})
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


def judge_run_of(prop: dict) -> dict:
    """Which judge run the proposal was derived from. A proposal made before
    the exam drove this loop says so rather than leaving a hole."""
    try:
        jr = json.loads(prop.get("judge_run") or "{}")
    except (ValueError, TypeError):
        jr = {}
    return {"judge_id": jr.get("judge_id") or "unrecorded",
            "batch_id": jr.get("batch_id") or "unrecorded",
            "prompt_sha256": jr.get("prompt_sha256") or "unrecorded"}


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
        "judge_run": judge_run_of(prop),
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
