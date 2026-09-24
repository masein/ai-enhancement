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
    → HUMAN     approves / edits / rejects on the dashboard's Improve ▸ Review.
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
EXAMPLES_SHOWN = 8               # what Improve ▸ Review shows of what the LLM saw
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
    """judge.json with only the topics graded on the questions each task
    holds now. A grade on a retired question set is history: a proposal, a
    gate or a justification built from it would be about questions nobody
    sits any more (judge.split_by_bank)."""
    p = Path(model_dir) / "judge.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None
    if j is None:
        return None
    import judge as _judge
    return _judge.split_by_bank(j, _exam.current_fingerprints(config.JUDGED_TASKS_DIR))[0]


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


# ---------------------------------------------------------------------------
# Who the answers are for. The generator only ever sees the approved spec, so
# the audience had to travel with the topic or not at all — and it did not.
# Both demo runs of 2026-09-20 show the cost: medicine produced clinical case
# notes and law produced advisory memoranda, when the exam is members of the
# public asking about their own symptoms and their own disputes. A model
# fine-tuned on clinician notes learns a register it is never asked for.
#
# What travels is LABELS AND PERCENTAGES from the bank's own metadata — style,
# subject, intent — over the whole topic. No prompt text, no qids, no item.
# ---------------------------------------------------------------------------

AUDIENCE_FIELDS = ("style", "subject", "intent")
AUDIENCE_TOP = 4
# Two registers. The layperson one was written for medicine and law, whose
# questions are people asking about their own situation; applied to physics
# (styles quantitative, conceptual, derivation) it asked for rigour in the
# governing equations and got "When you notice your heating or cooling system
# isn't performing…". It applies only when at least half a topic's items are
# written as a person asking — one of LAYPERSON_STYLES — or say who is asking
# (`subject`). Every other topic gets the worked explanation. A criteria
# file's `audience` string replaces either.
LAYPERSON_STYLES = {"conversational", "context_rich", "telegraphic"}
LAYPERSON_SHARE = 0.5
DEFAULT_REGISTER = (
    "guidance a layperson can read and act on — what to do, what to watch for, when to "
    "escalate — not clinical notes, case files, legal memoranda or textbook exposition. "
    "Second person is fine.")
DEFAULT_WHO = "members of the public asking about their own situation"
EXPLAINER_REGISTER = (
    "a worked explanation for someone learning or practising the subject — set the problem "
    "up, state the assumptions, carry units and check limiting cases where they apply, and "
    "name the common mistake. Prose, never question-and-answer pairs.")
EXPLAINER_WHO = "people learning or practising the subject"
# 11m: dataset #9 asked for question-and-answer items and was handed the
# documents' register, which ends "Prose, never question-and-answer pairs."
# The generator wrote prose, as told, and the Q&A reader found none of 26.
# Question-and-answer items have a register of their own.
QA_REGISTER = (
    "practice items a learner works through — one clear question, a short correct answer, "
    "and one or two sentences of why. Question-and-answer pairs, never an essay.")
# a field is listed only when its values are labels — `quantitative`,
# `case_analysis` — not a sentence per question: the 37-topic banks' `intent`
# is a different sentence on every item, and the top four of those would put
# four questions' descriptions, report half included, into a generator's
# request. A label is one short token
LABEL_MAX_CHARS = 40


def is_label_set(counts, n: int = 0) -> bool:
    return bool(counts) and all(len(v) <= LABEL_MAX_CHARS and not re.search(r"\s", v)
                                for v in counts)


def _share_line(counts: dict) -> str:
    total = sum(counts.values()) or 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:AUDIENCE_TOP]
    return ", ".join(f"{k} {round(100 * n / total)}%" for k, n in top)


def audience_for(topic: str, task: str | None = None, root: Path | None = None,
                 fmt: str = "doc") -> str:
    """The audience line for one topic, or '' when its bank says nothing about
    who is asking. Built from the WHOLE bank — both halves — because this is a
    count of labels, and a count of labels reveals no question.

    The author may write the register sentence herself: an `audience` string
    in the topic's criteria file replaces the default one.
    """
    import collections
    rows = _exam.load_bank(root or config.EXAM_DIR).get(topic) or []
    counts = {f: collections.Counter() for f in AUDIENCE_FIELDS}
    for r in rows:
        meta = r.get("meta") or {}
        for f in AUDIENCE_FIELDS:
            v = str(meta.get(f) or "").strip()
            if v:
                counts[f][v] += 1
    if not counts["style"] and not counts["subject"]:
        return ""                      # nothing said about who is asking
    register = ""
    try:
        import judge as _judge
        spec = _judge.rubric_for(task or _exam.topic_task(topic)).criteria or {}
        register = str(spec.get("audience") or "").strip()
    except Exception:                  # noqa: BLE001 — no criteria file, or none readable
        register = ""
    who, default = register_for(rows, counts)
    parts = [f"{f}s: {_share_line(counts[f])}" for f in AUDIENCE_FIELDS
             if is_label_set(counts[f], len(rows))]
    if fmt == "free":
        # the author's register is written for documents; question-and-answer
        # items take their own, or the two instructions contradict (11m)
        return (f"Audience: {who}" + (f" ({'; '.join(parts)})" if parts else "") + ".\n"
                f"Register for question-and-answer items: {QA_REGISTER}")
    return (f"Audience: {who}" + (f" ({'; '.join(parts)})" if parts else "") + ".\n"
            f"Register for documents: {register or default}")


def register_for(rows: list[dict], counts: dict) -> tuple[str, str]:
    """(who asks, the default register): the layperson's when at least half
    the topic's items are a person asking about their own situation — or say
    who is asking — and the worked explanation for everything else."""
    n = len(rows) or 1
    asking = sum(v for k, v in counts["style"].items() if k in LAYPERSON_STYLES)
    subject = sum(counts["subject"].values())
    if asking / n >= LAYPERSON_SHARE or subject / n >= LAYPERSON_SHARE:
        return DEFAULT_WHO, DEFAULT_REGISTER
    return EXPLAINER_WHO, EXPLAINER_REGISTER


# ---------------------------------------------------------------------------
# Where the documents go. Dataset #2 (Physics & Astronomy) asked for 20 and
# 11 of the 18 it got were about relativistic momentum: every request in a
# batch is identical but for a style seed, so a small generator writes the
# same document again. The bank already says what each question is about —
# its `domain` — so the documents are spread over the domains whose
# DIAGNOSE-half answers failed, and each request carries one domain's name.
# Nothing else of the bank travels: no count, no score, no qid, no question.
# ---------------------------------------------------------------------------

DOMAIN_MAX_DISTINCT = 15         # more than this is not a closed set of labels
DOMAIN_MIN_ITEMS = 2             # a value on one item is that question's, not a label
DOMAIN_MAX_CHARS = 64
DOMAIN_MAX_WORDS = 8


def is_domain_label_set(counts) -> bool:
    """Whether a topic's `domain` values are a closed set of labels — a name
    many questions share — rather than a sentence per question. Stricter than
    is_label_set (which guards the audience line and rightly refuses any
    whitespace): a domain is a phrase, like "Classical Mechanics"."""
    return bool(counts) and len(counts) <= DOMAIN_MAX_DISTINCT and all(
        n >= DOMAIN_MIN_ITEMS and len(v) <= DOMAIN_MAX_CHARS
        and len(v.split()) <= DOMAIN_MAX_WORDS and not set(v) & set(".?!")
        for v, n in counts.items())


def failing_domains(model_dir: Path, task: str, topic: str,
                    root: Path | None = None) -> dict[str, int]:
    """domain -> how many of this model's graded DIAGNOSE-half answers on
    this topic did not land. The report half is never read, by the same rule
    as justifications_for. {} when the topic's domains are not labels."""
    import collections
    rows = _exam.load_bank(root or config.EXAM_DIR).get(topic) or []
    all_counts = collections.Counter(str((r.get("meta") or {}).get("domain") or "").strip()
                                     for r in rows)
    all_counts.pop("", None)
    if not is_domain_label_set(all_counts):
        return {}
    of_qid = {r["qid"]: str((r.get("meta") or {}).get("domain") or "").strip() for r in rows}
    j = _judge_file(model_dir)
    if not j or j.get("skipped"):
        return {}
    out: collections.Counter = collections.Counter()
    for it in ((j.get("tasks") or {}).get(task) or {}).get("items") or []:
        if it.get("half") != "diagnose" or not it.get("graded"):
            continue                          # the whole safety property, again
        if it.get("score") is None or it["score"] >= WEAK_SCORE:
            continue
        d = of_qid.get(it.get("qid"))
        if d:
            out[d] += 1
    return dict(out)


def allocate(counts: dict[str, int], n: int) -> list[tuple[str, int]]:
    """`n` documents over the failing domains, in proportion to the failures,
    by largest remainder — and, while n allows, at least one each. Worst
    first, then by name, so the split is the same every time."""
    order = sorted(counts, key=lambda d: (-counts[d], d))
    if n <= 0 or not order:
        return []
    if n <= len(order):
        return [(d, 1) for d in order[:n]]
    total = sum(counts.values()) or 1
    base, rem = {}, {}
    for d in order:
        exact = n * counts[d] / total
        base[d] = max(1, int(exact))
        rem[d] = exact - int(exact)
    over = sum(base.values()) - n
    for d in sorted(order, key=lambda d: (rem[d], -counts[d], d)):
        while over > 0 and base[d] > 1:
            base[d] -= 1
            over -= 1
    left = n - sum(base.values())
    while left > 0:
        for d in sorted(order, key=lambda d: (-rem[d], -counts[d], d)):
            if left <= 0:
                break
            base[d] += 1
            left -= 1
    return [(d, base[d]) for d in order]


def focus_plan(model_dir: Path, task: str, topic: str, count: int,
               root: Path | None = None) -> list[dict]:
    """[{domain, failing_diagnose, documents}] — what the Review card shows
    before Generate, what the requests carry, and what provenance records.
    Empty when the topic has no domain labels or nothing failed: then the
    batch is what it was before."""
    counts = failing_domains(model_dir, task, topic, root)
    return [{"domain": d, "failing_diagnose": counts[d], "documents": k}
            for d, k in allocate(counts, count)]


# ---------------------------------------------------------------------------
# 11e: spread the documents on EVERY topic (masein, 2026-09-22). Mathematics
# has 18 labels and was refused by 11a's cap of 15, so 13 of its 20 documents
# were about conditional probability. A label set is used as written when it
# is small, by the prefix its labels share when that is small, and otherwise
# one failed concept at a time. Only diagnose-half items are ever read.
# ---------------------------------------------------------------------------

FOCUS_MAX_AREAS = 25            # at most this many areas, as written or grouped
FOCUS_MAX_CHARS = 64            # what is sent: a label, never a sentence
FOCUS_MAX_WORDS = 10
FOCUS_LIST_LEN = 100            # the frozen plan covers any count up to this
# a label's group is the text before the first of these; each has a space on
# both sides except ": ", so "Evidence-Based" never splits
_GROUP_SEP = re.compile(r" — | – | - |: | / ")

NO_LABELS = "no sub-area labels on this topic's questions"
TOO_LONG = "every label is too long to send"
TURNED_OFF = "the approver turned spreading off"
NOTHING_FAILED = "no answer in the practice half fell short, so there is nothing to spread over"


def label_group(label: str) -> str:
    """'Legal Method – Precedent' -> 'Legal Method'; a label with no
    separator is its own group."""
    m = _GROUP_SEP.search(label)
    return (label[:m.start()] if m else label).strip()


def sendable(label: str) -> bool:
    """A label that may go into a request: a short phrase, not a sentence."""
    return bool(label) and len(label) <= FOCUS_MAX_CHARS \
        and len(label.split()) <= FOCUS_MAX_WORDS and not set(label) & set(".?!")


def to_send(label: str) -> str | None:
    """The text actually sent for a label: itself, else its group, else nothing."""
    if sendable(label):
        return label
    g = label_group(label)
    return g if sendable(g) else None


def focus_scheme(labels) -> tuple[str | None, object]:
    """(mode, key) for a topic's labels: 'area' with the key that maps a label
    to its area (the label itself, or its group), or 'concept' when even the
    groups are too many to be areas; (None, None) with no labels at all."""
    distinct = {str(x).strip() for x in labels if str(x or "").strip()}
    if not distinct:
        return None, None
    if len(distinct) <= FOCUS_MAX_AREAS:
        return "area", (lambda x: x)
    if len({label_group(x) for x in distinct}) <= FOCUS_MAX_AREAS:
        return "area", label_group
    return "concept", None


def _round_robin(counts: list[tuple[str, int]]) -> list[str]:
    """[(A,3),(B,2)] -> [A, B, A, B, A]: a list cut anywhere still spreads."""
    out, left = [], [[k, n] for k, n in counts]
    while any(n for _, n in left):
        for pair in left:
            if pair[1]:
                out.append(pair[0])
                pair[1] -= 1
    return out


def focus_for(model_dir: Path, task: str, topic: str, root: Path | None = None) -> dict:
    """The plan a proposal card shows before Approve, and Approve freezes:
    {mode: 'area'|'concept'|None, labels: [<= FOCUS_LIST_LEN, in the order the
    documents are written], failing: {label: n}, reason}. Area mode allocates
    over the areas by their failing diagnose-half items; concept mode takes
    one failed diagnose-half item's label at a time, weakest first (ties by
    qid), one document per distinct label, round again past the end. The
    report half is never read."""
    import collections
    rows = _exam.load_bank(root or config.EXAM_DIR).get(topic) or []
    label_of = {r["qid"]: str((r.get("meta") or {}).get("domain") or "").strip() for r in rows}
    mode, key = focus_scheme(label_of.values())
    out = {"mode": mode, "labels": [], "failing": {}, "reason": ""}
    if mode is None:
        out["reason"] = NO_LABELS
        return out
    # every label unsendable, even by its group: nothing can go in a request
    if not any(to_send(key(x) if mode == "area" else x) for x in set(label_of.values()) if x):
        out.update(mode=None, reason=TOO_LONG)
        return out
    j = _judge_file(model_dir)
    items = (((j or {}).get("tasks") or {}).get(task) or {}).get("items") or [] \
        if j and not j.get("skipped") else []
    failed = sorted((it for it in items
                     if it.get("half") == "diagnose" and it.get("graded")       # the rule
                     and it.get("score") is not None and it["score"] < WEAK_SCORE
                     and label_of.get(it.get("qid"))),
                    key=lambda it: (it["score"], str(it.get("qid"))))
    if mode == "area":
        counts: collections.Counter = collections.Counter()
        for it in failed:
            area = to_send(key(label_of[it["qid"]]))
            if area:
                counts[area] += 1
        out["failing"] = dict(counts)
        out["labels"] = _round_robin(allocate(dict(counts), FOCUS_LIST_LEN))
    else:
        seen: list[str] = []
        for it in failed:
            lab = to_send(label_of[it["qid"]])
            if lab and lab not in seen:
                seen.append(lab)
        out["failing"] = {lab: 1 for lab in seen}
        out["labels"] = [seen[i % len(seen)] for i in range(FOCUS_LIST_LEN)] if seen else []
    if not out["labels"]:
        out["reason"] = NOTHING_FAILED
    return out


def frozen_focus(prop: dict) -> dict | None:
    """The plan Approve froze, or None for a proposal approved before 11e."""
    try:
        f = json.loads(prop.get("approved_focus") or "null")
    except (ValueError, TypeError):
        return None
    return f if isinstance(f, dict) else None


def focus_requests_plan(labels: list[str], per: int) -> list[tuple[str, int]]:
    """(label, documents) per request from the first N frozen labels: one
    label per request, as many of its documents as a request holds, the
    requests round-robin over the labels in the order they first appear."""
    import collections
    counts = collections.Counter(labels)
    order = list(dict.fromkeys(labels))
    per_label = [[(lab, min(per, counts[lab] - s)) for s in range(0, counts[lab], per)]
                 for lab in order]
    out: list[tuple[str, int]] = []
    for i in range(max((len(x) for x in per_label), default=0)):
        out += [x[i] for x in per_label if i < len(x)]
    return out


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


WEAKEST_CRITERIA = 3


def criteria_evidence(model_dir: Path, task: str) -> dict:
    """For a topic graded criterion by criterion: the weakest criteria, the
    means by every metadata field the topic is broken down by, and the flags
    that fired — as labels and numbers. No question text, no qids, no
    justifications — this is the shape of the failure, not its contents."""
    j = _judge_file(model_dir)
    t = ((j or {}).get("tasks") or {}).get(task) or {}
    means = t.get("criteria_mean") or {}
    if not means:
        return {}
    labels = t.get("criteria_labels") or {}
    counts = t.get("criteria_n") or {}
    scored = sorted(((v, k) for k, v in means.items() if v is not None))
    out = {"weakest_criteria": [{"id": k, "label": labels.get(k, k), "mean": v,
                                 "n": counts.get(k, 0)}
                                for v, k in scored[:WEAKEST_CRITERIA]]}
    if t.get("breakdowns"):
        out["breakdowns"] = {field: {k: {"n": v.get("n"), "mean": v.get("mean")}
                                     for k, v in cells.items()}
                             for field, cells in t["breakdowns"].items()}
    fired = {fid: {"n": f["n"], "share": f.get("share"), "label": f.get("label", fid),
                   "effect_words": f.get("effect_words", "")}
             for fid, f in (t.get("flags") or {}).items() if f.get("n")}
    if fired:
        out["flags"] = fired
    return out


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
                     justifications: list[dict], counts: dict, rubric: str,
                     criteria: dict | None = None, audience: str = "") -> llm.Request:
    """The judge's reasoning, the topic and the rubric. No question text: the
    justification is about the ANSWER, and anything the judge quoted from a
    question has already been stripped. For a topic graded criterion by
    criterion, the weakest criteria, the means by each of the topic's own
    metadata fields and the flags that fired ride along — labels and numbers,
    nothing from the bank."""
    lines = [f"Topic: {topic}", f"Model under assessment: {model}",
             f"Diagnosis-half answers on this topic: {counts['diagnose_items']}; "
             f"scoring below {WEAK_SCORE} of 4: {counts['diagnose_weak']}; "
             f"shown below: {len(justifications)}.", ""]
    # who asks these questions, as labels — the spec is for answering THEM
    if audience:
        lines += ["Who asks the questions on this topic:", audience.strip(), ""]
    lines += ["The rubric the judge graded against:", rubric.strip(), ""]
    if criteria and criteria.get("weakest_criteria"):
        lines.append("Where this topic scored lowest, criterion by criterion (0-1):")
        for c in criteria["weakest_criteria"]:
            lines.append(f"- {c['label']}: {c['mean']} over {c['n']} answers")
        for field, cells in (criteria.get("breakdowns") or {}).items():
            lines.append(f"Mean score of 4 by {field}: "
                         + ", ".join(f"{k} {v['mean']} ({v['n']})" for k, v in cells.items()))
        for f in (criteria.get("flags") or {}).values():
            lines.append(f"{f['n']} answers were flagged — {f['label'].lower()}"
                         + (f", which {f['effect_words']}" if f.get("effect_words") else "")
                         + ".")
        lines.append("")
    lines += ["The judge's assessment of each answer that did not land, with its score:", ""]
    for i, f in enumerate(justifications, 1):
        lines.append(f"[{i}] scored {f['score']} of 4 — {f['justification']}")
    lines += ["", "Write the skill spec now, as the JSON object described."]
    return llm.Request(
        custom_id=f"proposal:{pid}", system=PROPOSAL_SYSTEM, user="\n".join(lines),
        max_tokens=1024, json=True,
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
# 11m: the system prompt above forbids a question-and-answer pair, and the
# comparison format asks for nothing else — dataset #9 got both, and kept 0
# of 26. Question-and-answer items are asked for in their own words.
GEN_SYSTEM_QA = (
    "You write original PRACTICE ITEMS that exercise a specific skill for a small language "
    "model: question-and-answer pairs, each with a short correct answer and one or two "
    "sentences of why. You are given a skill specification, a topic and a count. Invent fresh "
    "material and vary the scenario, framing, named entities and settings across the set, so "
    "no two items share a template. Never reproduce or closely paraphrase any existing exam "
    "or benchmark text. Reply with one JSON array of objects and nothing else.")

STYLE = {
    "doc": (f"Each object: {{\"title\": <a short descriptive title>, \"text\": <the document "
            f"body, around {DOC_TARGET_WORDS} words of continuous prose that teaches the "
            f"specification's skill; paragraphs separated by blank lines; no questions posed "
            f"to the reader, no answer keys, no bullet lists of Q/A>}}. Write what a person "
            f"with that question would be helped by reading — an explainer, a guide, a worked "
            f"\"what to do if\" — in the register named above, and never shaped as a question "
            f"followed by its answer."),
    "free": ("Each object: {\"question\": ..., \"answer\": <a short free-text answer>, "
             "\"rationale\": <one or two sentences>}. This format is for comparison only: "
             "question-shaped training data teaches the test more readily than prose does."),
}


def items_per_request(fmt: str, provider: str | None = None) -> int:
    """How many items one request asks for. A LOCAL generator gets one
    document, not two: two of ~600 words is about 1,700 tokens, and
    LOCAL_MAX_TOKENS caps a reply at 1024, so asking for two truncates the
    JSON and nothing parses. The batch simply gets twice as many rows — the
    on-disk resume and the gate do not care — and the cap stays where it is,
    because the card is shared."""
    p = config.LLM_PROVIDER if provider is None else provider
    if fmt == "doc" and p == "local":
        return 1
    return ITEMS_PER_REQUEST.get(fmt, GEN_ITEMS_PER_REQUEST)


def focus_chunks(plan: list[dict], per: int, count: int) -> list[tuple[str | None, int]]:
    """(domain, how many documents) per request: each request covers ONE
    domain, and the requests go round-robin over the domains, so a batch cut
    short still covers the spread. No plan: no domain, chunked as before."""
    if not plan:
        return [(None, min(per, count - start)) for start in range(0, count, per)]
    per_domain = [[(p["domain"], min(per, p["documents"] - s))
                   for s in range(0, p["documents"], per)] for p in plan]
    out: list[tuple[str | None, int]] = []
    for i in range(max(len(x) for x in per_domain)):
        out += [x[i] for x in per_domain if i < len(x)]
    return out


def generation_requests(did: int, spec_text: str, category: str, count: int,
                        fmt: str, seed: int, audience: str = "",
                        plan: list[dict] | None = None,
                        labels: list[str] | None = None) -> list[llm.Request]:
    """One request per few items. Contains the approved spec, the topic, the
    count, the format, a style constraint, the audience labels and — when the
    topic's questions carry domain labels — the one domain this request is
    for. No benchmark item, no exam question, no hash, no model name, no
    score and no failure count, in any form."""
    per = items_per_request(fmt)
    reqs = []
    start = 0
    # 11e: a frozen plan's first N labels, one label per request; else 11a's
    # plan, computed now (a proposal approved before plans were shown)
    chunks = focus_requests_plan(labels, per) if labels else focus_chunks(plan or [], per, count)
    for k, (focus, n) in enumerate(chunks):
        what = ("document" if fmt == "doc" else "item") + ("" if n == 1 else "s")
        user = (f"Skill specification:\n{spec_text.strip()}\n\n"
                + (f"{audience.strip()}\n\n" if audience else "")
                + f"Topic: {category}\n"
                # the label, and only the label: which corner of the topic this
                # set is for, so twenty documents are not twenty of one thing
                + (f"Focus: {focus}\n" if focus else "")
                + f"Format: {fmt}\nWrite {n} {what}.\n{STYLE[fmt]}\n"
                + f"Style seed {seed}-{k}: make this set differ in scenario, register and "
                + "phrasing from any other set you might write for the same specification.")
        reqs.append(llm.Request(
            custom_id=f"gen:{did}:{k}", system=GEN_SYSTEM_QA if fmt == "free" else GEN_SYSTEM,
            user=user, max_tokens=8192, json=True,
            meta={"kind": "generation", "dataset_id": did, "count": n, "start": start,
                  "format": fmt, **({"focus": focus} if focus else {})}))
        start += n
    return reqs


def read_reply(text: str, fmt: str, expected: int = 1) -> tuple[list[dict], list[str]]:
    """(the items in this reply, why each document that did not arrive is not
    here) — one reason per missing document, so a batch can account for every
    request. Dataset #2 asked for 20 and wrote 18, and nothing said why."""
    if not str(text or "").strip():
        return [], ["empty reply"] * expected
    arr = llm.extract_array(text)
    if arr is None:
        # a request for ONE item — what a local generator gets, since its
        # replies are capped — comes back as the object itself under JSON
        # mode, never wrapped in a one-element array. Reading that as nothing
        # would drop every document a local run produced, silently.
        obj = llm.extract_json(text)
        arr = [obj] if isinstance(obj, dict) else None
    if arr is None:
        return [], ["reply not JSON"] * expected
    out: list[dict] = []
    why: list[str] = []
    for o in arr:
        if not isinstance(o, dict):
            why.append("reply not JSON")
            continue
        if fmt == "doc":
            title = str(o.get("title") or "").strip()
            body = str(o.get("text") or o.get("body") or "").strip()
            # a title alone, or a paragraph too short to teach anything, is not
            # a training document — and neither is a quiz wearing prose
            if not title:
                why.append("no title")
            elif len(body.split()) < DOC_MIN_WORDS:
                why.append(f"too short ({len(body.split())} words)")
            else:
                out.append({"title": title[:300], "text": body})
            continue
        q = str(o.get("question") or "").strip()
        a = str(o.get("answer") or "").strip()
        r = str(o.get("rationale") or "").strip()
        if not q or not a:
            why.append("no question or no answer")
            continue
        out.append({"question": q, "answer": a, "rationale": r})
    # asked for two, sent one: the other is missing with no reason of its own
    why += ["not in the reply"] * max(0, expected - len(out) - len(why))
    return out, why


def parse_items(text: str, fmt: str) -> list[dict]:
    """The items in one reply. read_reply also says what is not there."""
    return read_reply(text, fmt)[0]


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


LOCAL_ROLES = (("generator", "generated"), ("exam writer", "drafted"), ("judge", "graded"))


def local_marks(generator_id: str, prop: dict) -> dict:
    """{role: what the local server served} for every identity behind this
    dataset that was a local model — the generator that wrote it, the exam
    writer whose questions it was steered by, the judge whose words picked
    the topic. {} when none was."""
    from . import llm
    ex = llm.identity("exam")
    ids = {"generator": generator_id, "exam writer": f"{ex[0]}/{ex[1]}" if ex[0] else "",
           "judge": judge_run_of(prop)["judge_id"]}
    out = {}
    for role, verb in LOCAL_ROLES:
        provider, _, model = ids[role].partition("/")
        mark = llm.local_mark(provider, model, verb)
        if mark:
            out[role] = {k: mark[k] for k in ("base_url", "served_model", "weights") if k in mark}
    return out


def provenance(prop: dict, ds: dict, backend_id: str, batch_id: str, prompt_hash: str,
               gate: dict, sha: str, n_generated: int, n_kept: int,
               audience: str = "", missing: list[dict] | None = None,
               focus: list[dict] | None = None, focus_mode: str = "",
               focus_labels: list[str] | None = None) -> dict:
    out = _provenance(prop, ds, backend_id, batch_id, prompt_hash, gate, sha, n_generated,
                      n_kept, audience=audience, missing=missing, focus=focus)
    # 11e: how the documents were spread, and over exactly which labels
    out["focus_mode"] = focus_mode or ("area" if focus else "off")
    out["focus_labels"] = list(focus_labels if focus_labels is not None
                               else [f["domain"] for f in (focus or [])
                                     for _ in range(f.get("documents") or 0)])
    local = local_marks(backend_id, prop)
    if local:                     # always, when any of them was local — there is no flag
        out["provisional"] = True
        out["provisional_reason"] = (f"a local model was the {' and the '.join(local)} — "
                                     f"not a pinned benchmark")
        out["local_models"] = local
    over = override_of(prop)
    if over:                      # a person proposed past a judge that was not evidence
        out["proposed_over_provisional_judge"] = over
    return out


def override_of(prop: dict) -> dict | None:
    """{by, at, reasons} when this proposal was made over a provisional judge
    (POST /api/proposals with override_preliminary), else None. It travels:
    the proposal, the approved spec, the dataset's provenance.json and the
    taint trail of any model trained on it all carry it."""
    raw = prop.get("override")
    if isinstance(raw, dict):
        return raw or None
    try:
        return json.loads(raw or "null") or None
    except (ValueError, TypeError):
        return None


def _provenance(prop: dict, ds: dict, backend_id: str, batch_id: str, prompt_hash: str,
                gate: dict, sha: str, n_generated: int, n_kept: int,
                audience: str = "", missing: list[dict] | None = None,
                focus: list[dict] | None = None) -> dict:
    return {
        # which corner of the topic each request was for (focus_plan)
        "focus_plan": list(focus or []),
        # the labels the generator was given about who asks these questions,
        # recorded beside the spec because they shaped the documents too
        "audience": audience or audience_for(prop["category"], prop["task"]),
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
        # every document asked for is accounted for: kept, dropped by the
        # gate, or missing with the reason it never arrived
        "items": {"requested": ds["count"], "generated": n_generated,
                  "dropped": n_generated - n_kept, "kept": n_kept,
                  "missing": list(missing or [])},
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
    # identity may legitimately be unconfigured, and False is a value.
    # audience is empty for a topic whose bank says nothing about who asks —
    # the legacy `other` items carry no style or subject at all
    # focus_plan is empty for a topic with no domain labels, and a batch where
    # every document arrived has nothing missing — both are records, not holes
    return [h for h in holes if h not in ("edited_text", "gate.offending_ngrams",
                                          "audience", "focus_plan", "focus_labels",
                                          "items.missing",
                                          "identities.exam_writer", "identities.judge",
                                          "identities.single_provider_loop")
            and not h.startswith("gate.dropped")]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
