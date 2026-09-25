#!/usr/bin/env python3
"""Grade the exam answers with a pinned API judge, in batch; write judge.json.

    python scripts/judge.py results/full -m local/my-ckpt --wait     # JUDGE_* from .env
    python scripts/judge.py results/full --stub                       # tests and dry runs

Reads the harness's --log_samples output for the exam tasks (exam_<topic>,
fr_control_mmlu; scripts/exam_build.py) and writes results/full/<model>/
judge.json in the same shape-and-place pattern as diagnose.json, so the
dashboard's _beside() loader and the service's freshness key pick it up
unchanged. The service runs this as the last step of a suite=judged job:
the grading batch is SUBMITTED inside the GPU lock (seconds, no GPU) and
FINISHED by service/llm_poller.py when the provider completes it — a judge
that waits on an API must never hold the card.

The judging contract, unchanged in substance from the local judge:

  rubric     single answers graded 0–4 against a written, versioned rubric
             with anchors (eval_tasks/fr/rubrics/), never pairwise — that is
             where position bias lives. Length is in the rubric explicitly
             and score-vs-length is reported per topic
  pinned     JUDGE_MODEL must be a DATED model id, not a floating alias.
             Provider, model, prompt sha, rubric sha and batch id ride in
             every judge.json and in every dataset's provenance
  local      JUDGE_PROVIDER=local (a vLLM server on the box) cannot be pinned
             at all — its id is whatever was typed at launch. It is not
             refused; it runs PROVISIONAL: judge.json says so with the base
             URL, the served id and the weights, the reason joins the
             preliminary reasons, and the page greys it, never ranks it and
             leaves it out of every average. No flag turns that off
  canary     a fixed set of thirty answer scripts with known human marks
             (eval_tasks/fr/canary.jsonl) is re-graded at the start of every
             run. judge.json records the canary's mean absolute deviation
             from those marks and from the previous run's; movement past
             JUDGE_CANARY_MAX_DRIFT marks the run preliminary with the
             reason stated. This replaces the byte-identical determinism a
             local greedy judge gave us: a vendor updating the model behind
             the id would otherwise silently re-base every score
  family     a judge never grades a model of its own family, and its
             PROVIDER must differ from the exam writer's and the generator's
             — a loop whose questions, grades and training data all come
             from one family grades itself. ALLOW_SINGLE_PROVIDER_LOOP=1
             overrides for a trial and stamps every judged score with it
  halves     each item records its qid's half; the published topic score is
             the report half; justifications (what went wrong, in words) are
             recorded per item for the step that picks a topic — which may
             read the diagnose half only
  --stub     a deterministic overlap stand-in for tests; the fake backend is
             what CI uses for the batch path
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import diagnose as dx  # noqa: E402
import exam_build as _exam  # noqa: E402
from exam_build import ALL_TASKS, CONTROL_TASK  # noqa: E402

RUBRIC_DIR = REPO / "eval_tasks" / "fr" / "rubrics"
CANARY_PATH = REPO / "eval_tasks" / "fr" / "canary.jsonl"
CANARY_HISTORY = "judge_canary_history.jsonl"      # at the root of results/full
PROMPT_VERSION = 2
MAX_SCORE = 4
CORRECT_AT = 3           # an open-ended answer scoring >= this counts as "knew it"
LENGTH_BUCKETS = [(0, 20, "≤20 words"), (21, 50, "21–50"), (51, 120, "51–120"), (121, 10**9, ">120")]
DATED = re.compile(r"(\d{8}|\d{4}-\d{2}-\d{2})")   # a pinned model id carries its date

PROMPT = """You are grading ONE answer to ONE question against a rubric. Read the rubric, the question, the reference answer, and the candidate answer. Reply with one JSON object and nothing else: {{"score": <integer 0-4>, "justification": <one or two sentences on what the answer got right or wrong against the rubric — about the ANSWER, never quoting the question>}}.

RUBRIC
{rubric}

QUESTION
{question}

REFERENCE (what a full-marks answer contains)
{reference}

CANDIDATE ANSWER
{answer}"""

# Used only for a task whose rubric has a criteria file beside it. The 0-4 is
# NOT asked for here: it is folded in code from these numbers (fold), so the
# grade is reproducible from what is recorded. Criteria and flags are two
# lists under two headings because a topic may name the same thing in both
# (law scores `fabricated_authority` 0-1 AND flags it when the fabrication
# carries the conclusion) and the model must not conflate them.
PROMPT_CRITERIA = """You are grading ONE answer to ONE question against a rubric, a list of criteria and a list of flags. Read the rubric, the criteria, the flags, the question, the reference, and the candidate answer. Reply with one JSON object and nothing else: {{"criteria": {{"<criterion id>": <number from 0.0 to 1.0>, …one entry for every criterion id listed below…}}, "flags": {{"<flag id>": <true or false>, …one entry for every flag id listed below…}}, "justification": <one or two sentences on what the answer got right or wrong against the criteria — about the ANSWER, never quoting the question>}}.

Every criterion id below must appear exactly as written, with a number from 0.0 (not met at all) to 1.0 (fully met). Every flag id must appear exactly as written, with true or false — a flag is not a criterion and is decided on its own terms. Do not rename an id, do not add entries of your own, and do not give an overall score — it is computed from these.

RUBRIC
{rubric}

{principles}CRITERIA (0.0–1.0 each)
{criteria}

FLAGS (true/false each)
{flags}

QUESTION
{question}

REFERENCE (what a correct answer must respect)
{reference}

CANDIDATE ANSWER
{answer}"""


def family(model_id: str) -> str:
    """Same rule as the dashboard's `family` field: the first alphanumeric run
    of the last path segment — 'llama' from 'Llama-3.1-8B-Instruct', 'claude'
    from 'claude-sonnet-4-5-20250929', 'gpt' from 'gpt-4.1-2025-04-14'."""
    return re.split(r"[^a-z0-9]", model_id.split("/")[-1].lower())[0]


class RubricMissing(FileNotFoundError):
    """No rubric file for a task — not even the shared one. Raised rather
    than guessed at: a topic graded by a file nobody can name is not graded.
    Every page that loops over topics catches this per topic, so one missing
    file cannot take a whole board down."""


class Rubric(NamedTuple):
    """(text, sha256, version, status, criteria, criteria_sha256) — indexable
    as the 3-tuple it used to be. `status` is "draft" while the heading says
    so: a rubric its author has not signed off is not a benchmark, and the
    page says so beside the score. Signing off means deleting the word, which
    changes the sha, which is correct — a different rubric is a different
    instrument. `criteria` is the parsed .criteria.json beside it when there
    is one, and changing THAT file is a change of instrument too."""
    text: str
    sha256: str
    version: str
    status: str = ""
    criteria: dict | None = None
    criteria_sha256: str = ""
    # which file was read, so a page can say so instead of implying that
    # every topic has a rubric of its own
    name: str = ""
    path: str = ""
    fallback: bool = False


def rubric_dirs() -> list[Path]:
    """Where a rubric may live, in order. $BENCH_ROOT/rubrics comes first so
    a rubric uploaded through the page beats the one baked into the image —
    inside the container /app IS the image, and the checkout is not there.
    The repo's own directory is the fallback and the default content."""
    from service import config
    return [Path(config.BENCH_ROOT) / "rubrics", RUBRIC_DIR]


def rubric_path(name: str, suffix: str = ".md") -> Path | None:
    for d in rubric_dirs():
        p = d / f"{name}{suffix}"
        if p.is_file():
            return p
    return None


SHARED_RUBRIC = "exam"


def rubric_name(task: str) -> str:
    """Which rubric grades this task: the topic's own when one exists
    (rubrics/<slug>.md, the slug exam_build built the task name from), else
    the shared exam rubric. The control set asks for a fact, and its gold
    option is the reference, so it keeps the factual one."""
    if task == CONTROL_TASK:
        return "factual_accuracy"
    slug = _exam.task_slug(task)
    return slug if rubric_path(slug) else SHARED_RUBRIC


def rubric_for(task: str) -> Rubric:
    """The instrument that grades this task. A topic with no file of its own
    is graded by the SHARED rubric (rubrics/exam.md), which is what P4a
    specified and what AUTHORING.md says — never by rubrics/<slug>.md, which
    for thirteen of fifteen topics does not exist. The returned Rubric names
    the file that was actually read, so a page can say "exam.md (fallback)"
    rather than implying every topic has its own."""
    name = rubric_name(task)
    p = rubric_path(name)
    if p is None and name != SHARED_RUBRIC:
        name, p = SHARED_RUBRIC, rubric_path(SHARED_RUBRIC)
    if p is None:
        own = _exam.task_slug(task) if task != CONTROL_TASK else name
        raise RubricMissing(
            f"no rubric file for {task}: neither {own}.md nor the shared "
            f"{SHARED_RUBRIC}.md is in {' or '.join(str(d) for d in rubric_dirs())}")
    text = p.read_text(encoding="utf-8")
    # "(version 1)" and "(version 1, DRAFT — awaiting sign-off)" both parse:
    # the status rides in the same brackets and must not hide the version
    m = re.search(r"\(version (\d+)", text)
    head = text.split("\n", 1)[0]
    spec = spec_sha = None
    cp = rubric_path(name, ".criteria.json")
    if cp:
        raw = cp.read_bytes()
        # the sha is of the FILE, as delivered; what the judge reads is the
        # same file in this platform's one internal shape
        spec = normalise_criteria(json.loads(raw.decode("utf-8")), cp)
        spec_sha = hashlib.sha256(raw).hexdigest()
    return Rubric(text, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  (m.group(1) if m else "?"), "draft" if "DRAFT" in head else "",
                  spec, spec_sha or "", name, str(p),
                  name == SHARED_RUBRIC and task != CONTROL_TASK)


# How the 0-4 is folded is the platform's rule, not the file's: the weighted
# mean of the criteria times four, rounded half up, then each true flag's
# effect applied in the order the file lists them.
FOLD_METHOD = "weighted_mean_x4_round_half_up"
ZERO_SCORE = "zero_score"
CAP_EFFECT = re.compile(r"cap_at_([0-4])_of_4$")
# A breakdown table is worth having when the field SPLITS the topic: two to
# twelve distinct values. One value is not a table (three of the topics have
# `acuity: routine` on all 100 items) and a hundred values is not one either
# (computer science's `intent` is a sentence per question). When a field
# qualifies, this is the order a person wants to read them in; anything else
# the items carry follows, and four tables is as many as anyone reads.
BREAKDOWN_FIELDS = ("acuity", "difficulty", "jurisdiction_required", "intent")
BREAKDOWN_EXTRA = ("domain", "style")
BREAKDOWN_MIN_VALUES = 2
BREAKDOWN_MAX_VALUES = 12
BREAKDOWN_MIN_TABLES = 2
BREAKDOWN_MAX_TABLES = 4
# most severe first, so a table reads down from the questions that matter.
# `critical` and `high` arrived with the 37-topic exam (engineering, IT, law,
# security…); they sit where their words put them
ACUITY_ORDER = ("emergency", "critical", "urgent", "high", "moderate", "mild", "routine")


class CriteriaError(ValueError):
    """A criteria file this judge cannot apply. Raised at load, never at
    grade time: a file whose effect string is unknown would silently score
    every answer as if the flag did nothing."""


SET_EFFECT = re.compile(r"set_at_([0-4])_of_4$")
# the spellings the author has used for the same three ideas, across three
# deliveries. We read his files; he does not rewrite them for us.
_EFFECT_ALIASES = (
    (re.compile(r"^(?:zero_score|score\s*=\s*0|set_at_0_of_4)$", re.I), ZERO_SCORE),
    (re.compile(r"^(?:cap_at_([0-4])_of_4|cap\s*=\s*([0-4]))$", re.I), "cap_at_{}_of_4"),
    (re.compile(r"^(?:score\s*=\s*([1-4])|set_at_([1-4])_of_4)$", re.I), "set_at_{}_of_4"),
)


def normalise_effect(raw) -> str | None:
    """One of this judge's three effects, or None when it is none of them."""
    text = str(raw or "").strip()
    for pattern, out in _EFFECT_ALIASES:
        m = pattern.match(text)
        if m:
            n = next((g for g in m.groups() if g), None)
            return out.format(n) if "{}" in out else out
    return None


def effect_of(flag: dict) -> str:
    return str(flag.get("effect") or "").strip()


def effect_words(effect: str) -> str:
    """What an effect does to the score, for the prompt and for the page."""
    if effect == ZERO_SCORE:
        return "sets the whole score to 0"
    m = CAP_EFFECT.match(effect)
    if m:
        return f"caps the whole score at {m.group(1)} of 4"
    m = SET_EFFECT.match(effect)
    return f"makes the whole score {m.group(1)} of 4" if m else effect


def check_effects(spec, path=None) -> None:
    for f in (spec or {}).get("flags") or []:
        e = effect_of(f)
        if normalise_effect(e) is None:
            raise CriteriaError(
                f"{path or 'criteria file'}: flag {f.get('id')!r} has effect {e!r}, which this "
                f"judge cannot apply — it knows {ZERO_SCORE}, cap_at_N_of_4 and score=N")


# ---------------------------------------------------------------------------
# One internal shape, however the file was written. Four deliveries have
# arrived in five shapes — flags as a list (`flags`, `critical_flags`), or as
# one object (`critical_flag`, `critical_error_flag`, `critical_error`); the
# criterion slug in `id` or in `name`; effects spelled `zero_score` or
# `score=0` — and the decision (2026-09-20) is that the loader adapts, never
# the author's file. docs/CRITERIA-SCHEMA.md lists every variant and what it
# maps onto. Top-level keys the loader has no use for (`benchmark`, `task`,
# `scale`, `score_scale`, `score_range`) ride along untouched: informational,
# and part of the file's sha like every other byte.
# ---------------------------------------------------------------------------

_FLAG_KEYS = ("flags", "critical_flags", "critical_flag", "critical_error_flag",
              "critical_error")
_NOT_CRITICAL_KEYS = ("not_critical", "do_not_classify_as_critical", "not_critical_examples")
# the topic-wide principles, under the three names they arrived with; each is
# a sentence or an object with a heading and a sentence
_PRINCIPLE_KEYS = ("evaluation_principles", "important_evaluation_principles", "principles")
_PRINCIPLE_HEAD = ("name", "title")
_PRINCIPLE_TEXT = ("statement", "principle", "text", "description")


def _principle_text(p) -> str:
    """One principle as the prompt says it: the sentence, after its heading
    when it has one."""
    if not isinstance(p, dict):
        return str(p).strip()
    head = next((str(p[k]).strip() for k in _PRINCIPLE_HEAD if p.get(k)), "")
    text = next((str(p[k]).strip() for k in _PRINCIPLE_TEXT if p.get(k)), "")
    if head and text:
        return f"{head}: {text}"
    return head or text or json.dumps(p, ensure_ascii=False, sort_keys=True)


def _slug_and_label(c: dict) -> tuple[str, str]:
    """A criterion's id is the slug the model must repeat back. It is `id`
    when that is a name rather than a row number, else `name` — physics
    numbers its criteria 1..20 and puts the slug in `name`."""
    raw_id, raw_name = c.get("id"), str(c.get("name") or "").strip()
    text_id = str(raw_id).strip() if raw_id is not None else ""
    numbered = isinstance(raw_id, (int, float)) or text_id.isdigit()
    slug = raw_name if numbered and raw_name else text_id
    label = raw_name if raw_name and raw_name != slug else label_of({"id": slug})
    return slug, label


def normalise_criteria(spec, path=None) -> dict:
    """The author's file as this judge reads it: criteria with a slug id and
    a label, flags as one list with an effect this judge can apply, and his
    calibration (examples, what is not critical, evaluation principles) kept
    where the prompt builder can find it."""
    if not isinstance(spec, dict):
        return spec
    out = {k: v for k, v in spec.items() if k not in _FLAG_KEYS}
    crits = []
    for c in spec.get("criteria") or []:
        if not isinstance(c, dict):
            crits.append(c)
            continue
        slug, label = _slug_and_label(c)
        crits.append({**c, "id": slug, "name": label})
    out["criteria"] = crits
    flags = []
    for key in _FLAG_KEYS:
        v = spec.get(key)
        for f in (v if isinstance(v, list) else [v] if isinstance(v, dict) else []):
            if not isinstance(f, dict):
                continue
            eff = normalise_effect(f.get("effect"))
            nots = next((f[k] for k in _NOT_CRITICAL_KEYS if isinstance(f.get(k), list)), [])
            flags.append({**f, "id": str(f.get("id") or "").strip(),
                          "effect": eff or effect_of(f),
                          "examples": list(f.get("examples") or []),
                          "not_critical": list(nots)})
    out["flags"] = flags
    raw = next((spec[k] for k in _PRINCIPLE_KEYS if isinstance(spec.get(k), list)), [])
    out["evaluation_principles"] = [t for t in map(_principle_text, raw) if t]
    if path is not None:
        check_effects(out, path)
    return out


def label_of(item: dict) -> str:
    """The author writes ids; the page and the demo show words. A file may
    carry its own label — as `label`, or as a `name` that is not the id
    ("Risk-Benefit Reasoning", in his capitals) — and when it does not the
    id is the label."""
    lab = str(item.get("label") or "").strip()
    if lab:
        return lab
    name = str(item.get("name") or "").strip()
    if name and name != str(item.get("id") or "").strip():
        return name
    words = str(item.get("id") or "").replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def validate_criteria(spec) -> list[str]:
    """Everything wrong with a criteria file, in words, or an empty list. The
    fold reads this file and the prompt is generated from it, so a file that
    is wrong here grades wrongly everywhere — it is checked before it can be
    saved, not when a batch of grades comes back strange. The schema is the
    author's: topic, weights, criteria, flags."""
    bad: list[str] = []
    if not isinstance(spec, dict):
        return ["the file is not a JSON object"]
    # validate what the judge would actually read, not the spelling it arrived
    # in: every shape the author has sent is normalised first
    spec = normalise_criteria(spec)
    weights = spec.get("weights", "equal")
    if weights not in ("equal", None):
        bad.append(f"weights {weights!r}: this loader knows 'equal' (or no key at all), and "
                   f"honours a 'weight' on a criterion when it carries one")
    crits = spec.get("criteria")
    if not isinstance(crits, list) or not crits:
        bad.append("'criteria' must be a non-empty list")
        crits = []
    seen = set()
    for i, c in enumerate(crits):
        where = f"criterion {i + 1}"
        if not isinstance(c, dict):
            bad.append(f"{where} is not an object")
            continue
        cid = str(c.get("id") or "").strip()
        where = f"criterion {cid or i + 1}"
        if not cid:
            bad.append(f"{where} has no id")
        elif not re.fullmatch(r"[a-z][a-z0-9_]*", cid):
            bad.append(f"{where}: an id is lower-case letters, digits and underscores — the "
                       f"model must repeat it exactly")
        elif cid in seen:
            bad.append(f"{where}: duplicate id")
        seen.add(cid)
        if not str(c.get("definition") or "").strip():
            bad.append(f"{where} has no definition — the prompt is built from it")
        try:
            if float(c.get("weight", 1)) <= 0:
                bad.append(f"{where}: weight must be greater than 0")
        except (TypeError, ValueError):
            bad.append(f"{where}: weight is not a number")
    flags = spec.get("flags")
    if flags is not None and not isinstance(flags, list):
        bad.append("'flags' must be a list — a topic with none may leave the key out")
        flags = []
    fseen = set()
    for i, f in enumerate(flags or []):
        where = f"flag {i + 1}"
        if not isinstance(f, dict):
            bad.append(f"{where} is not an object")
            continue
        fid = str(f.get("id") or "").strip()
        where = f"flag {fid or i + 1}"
        if not fid:
            bad.append(f"{where} has no id")
        elif not re.fullmatch(r"[a-z][a-z0-9_]*", fid):
            bad.append(f"{where}: an id is lower-case letters, digits and underscores — the "
                       f"model must repeat it exactly")
        elif fid in fseen:
            bad.append(f"{where}: duplicate id")
        fseen.add(fid)
        # a flag id may equal a criterion id: the criterion scores the thing,
        # the flag fires when it carries the answer. Two headings, two objects
        if not str(f.get("condition") or "").strip():
            bad.append(f"{where} has no condition — the prompt is built from it")
        if normalise_effect(effect_of(f)) is None:
            bad.append(f"{where}: effect {effect_of(f)!r} is not one this judge can apply — "
                       f"{ZERO_SCORE}, cap_at_N_of_4 or score=N (N from 0 to 4)")
    breakdowns = spec.get("breakdowns")
    if breakdowns is not None and (not isinstance(breakdowns, list)
                                   or not all(isinstance(b, str) and b for b in breakdowns)):
        bad.append("'breakdowns' is a list of metadata field names, or absent to use whichever "
                   f"of {', '.join(BREAKDOWN_FIELDS)} the topic's items carry")
    return bad


def criteria_ids(spec: dict) -> list[str]:
    return [c["id"] for c in spec.get("criteria") or []]


def conditional_ids(spec: dict) -> set[str]:
    return {c["id"] for c in spec.get("criteria") or [] if c.get("conditional")}


def flag_ids(spec: dict) -> list[str]:
    return [f["id"] for f in spec.get("flags") or []]


def criteria_labels(spec: dict) -> dict:
    return {c["id"]: label_of(c) for c in spec.get("criteria") or []}


def weight_of(spec: dict, cid: str) -> float:
    """Equal by default, and the file's own number when a criterion carries
    one — both of the author's files say equal and carry 1.0 on every row."""
    for c in spec.get("criteria") or []:
        if c["id"] == cid:
            try:
                return float(c.get("weight", 1))
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _rubric_record(task: str) -> dict:
    """Which instrument graded this task, in full: the prose, its criteria
    file when it has one, and the prompt that carried them. Any of the three
    changing makes before and after a different measurement."""
    r = rubric_for(task)
    out = {"name": rubric_name(task), "sha256": r.sha256, "version": r.version,
           "prompt_sha256": prompt_sha(PROMPT_CRITERIA if r.criteria else PROMPT)}
    if r.status:
        out["status"] = r.status
    if r.criteria:
        out["criteria_sha256"] = r.criteria_sha256
        out["scoring"] = "criteria"
        # the author's schema carries neither; a file that states them is
        # still recorded, because provenance takes what it is given
        for k in ("version", "status"):
            if r.criteria.get(k) is not None:
                out[f"criteria_{k}"] = r.criteria[k]
    return out


def build_prompt(rubric: str, question: str, reference: str, answer: str) -> str:
    return PROMPT.format(rubric=rubric.strip(), question=question.strip(),
                         reference=reference.strip(), answer=(answer or "").strip() or "(empty)")


def prompt_sha(template: str = PROMPT) -> str:
    return hashlib.sha256(f"v{PROMPT_VERSION}\n{template}".encode("utf-8")).hexdigest()


def prompt_for(task: str) -> str:
    """Which prompt grades this task — the per-criterion one when its rubric
    has a criteria file, else the single-score one."""
    return PROMPT_CRITERIA if rubric_for(task).criteria else PROMPT


def words(s: str) -> int:
    return len((s or "").split())


def criteria_block(spec: dict) -> str:
    """One line per criterion, from the file — so the prompt cannot describe
    criteria the fold does not know about, or miss one it does."""
    lines = []
    for c in spec.get("criteria") or []:
        line = f"{c['id']} — {c['definition']}"
        if c.get("conditional"):
            line += (f" CONDITIONAL: applies when {c['applies_when']}. "
                     if str(c.get("applies_when") or "").strip() else " CONDITIONAL. ")
            line += "Return null for it when it does not apply to this question."
        lines.append(line)
    return "\n".join(lines)


def flags_block(spec: dict) -> str:
    """One line per flag: what it is, and what it does to the score — then
    the author's own calibration of it. His examples of what counts and what
    does not are the difference between a flag that fires on anything and one
    that means something, so they go in the request, not in a doc."""
    out = []
    for f in spec.get("flags") or []:
        out.append(f"{f['id']} — {f['condition']} — {effect_words(effect_of(f))}")
        for ex in f.get("examples") or []:
            out.append(f"    counts as {f['id']}: {ex}")
        for ex in f.get("not_critical") or []:
            out.append(f"    does NOT count as {f['id']}: {ex}")
    return "\n".join(out)


def principles_block(spec: dict) -> str:
    """The author's standing instructions to whoever grades this topic."""
    return "\n".join(f"- {p}" for p in spec.get("evaluation_principles") or [])


def build_criteria_prompt(rubric: str, spec: dict, question: str, reference: str,
                          answer: str) -> str:
    principles = principles_block(spec)
    return PROMPT_CRITERIA.format(
        rubric=rubric.strip(),
        principles=(f"HOW THIS TOPIC IS GRADED (the author's principles)\n{principles}\n\n"
                    if principles else ""),
        criteria=criteria_block(spec),
        flags=flags_block(spec) or "(none for this topic)",
        question=question.strip(), reference=reference.strip(),
        answer=(answer or "").strip() or "(empty)")


class Grade(NamedTuple):
    """(criteria, flags, justification) — plus the keys the model invented,
    counted rather than accepted."""
    criteria: dict
    flags: dict
    justification: str
    extra_keys: list


def parse_grade_criteria(text: str, spec: dict) -> Grade | None:
    """The per-criterion reply, or None when it cannot be read. Strict where
    it matters: every non-conditional criterion must be there and numeric,
    and every flag must be there and boolean — a missing flag is a parse
    failure, not a false. The flags are the fields that matter most, and the
    model does not get to skip one. Out-of-range numbers are clamped; a
    conditional criterion may be null or missing; renamed criteria are NOT
    accepted, they are counted as extra keys and leave a hole that fails."""
    from service import llm
    obj = llm.extract_json(text or "")
    if not isinstance(obj, dict):
        return None
    scores = obj.get("criteria")
    if not isinstance(scores, dict):
        return None
    wanted_flags = flag_ids(spec)
    given = obj.get("flags")
    if wanted_flags and not isinstance(given, dict):
        return None
    flags: dict[str, bool] = {}
    for fid in wanted_flags:
        v = (given or {}).get(fid)
        if not isinstance(v, bool):
            return None
        flags[fid] = v
    ids, conditional = criteria_ids(spec), conditional_ids(spec)
    out: dict[str, float | None] = {}
    for cid in ids:
        if cid not in scores or scores[cid] is None:
            if cid in conditional:
                out[cid] = None
                continue
            return None                       # a missing criterion is a hole, not a zero
        try:
            v = float(scores[cid])
        except (TypeError, ValueError):
            return None
        if v != v:                            # NaN
            return None
        out[cid] = min(1.0, max(0.0, v))
    extra = sorted(k for k in scores if k not in set(ids))
    extra += sorted(f"flags.{k}" for k in (given or {}) if k not in set(wanted_flags))
    return Grade(out, flags, str(obj.get("justification") or "").strip()[:600], extra)


def round_half_up(x: float) -> int:
    """2.5 is a 3. Python's round() is banker's rounding and would make it a
    2, which is not what the fold rule says."""
    return int(math.floor(x + 0.5))


def effects_applied(flags: dict, spec: dict) -> list[str]:
    """Which flags fired, in the file's order — recorded per item, because a
    0 that came from a flag and a 0 that came from the criteria are not the
    same finding."""
    return [f["id"] for f in spec.get("flags") or [] if flags.get(f["id"])]


def fold(criteria: dict, flags: dict, spec: dict) -> int:
    """The 0-4 the rest of the system reads, computed here and never asked of
    the model: one source of truth, reproducible from the recorded criteria.
    The weighted mean of what applied, times four, rounded half up — then
    each true flag's effect, in the order the file lists them, because a
    zero and a cap can both be true and the file decides which lands last."""
    num = sum(weight_of(spec, k) * v for k, v in criteria.items() if v is not None)
    den = sum(weight_of(spec, k) for k, v in criteria.items() if v is not None)
    score = max(0, min(MAX_SCORE, round_half_up(MAX_SCORE * num / den))) if den else 0
    for f in spec.get("flags") or []:
        if not flags.get(f["id"]):
            continue
        e = normalise_effect(effect_of(f))
        if e == ZERO_SCORE:
            score = 0
            continue
        m = CAP_EFFECT.match(e or "")
        if m:
            score = min(score, int(m.group(1)))
            continue
        m = SET_EFFECT.match(e or "")
        if m:
            score = int(m.group(1))
        else:                                 # a file that never loaded cannot get here
            raise CriteriaError(f"flag {f['id']!r}: unknown effect {effect_of(f)!r}")
    return score


def parse_grade(text: str) -> tuple[int | None, str]:
    """(score, justification) from the judge's reply; a bare digit still counts."""
    from service import llm
    obj = llm.extract_json(text or "")
    if isinstance(obj, dict) and obj.get("score") is not None:
        try:
            s = int(obj["score"])
            if 0 <= s <= MAX_SCORE:
                return s, str(obj.get("justification") or "").strip()[:600]
        except (TypeError, ValueError):
            pass
    m = re.match(r"\s*(?:score\s*[:=]?\s*)?([0-4])\b", text or "", re.I)
    return (int(m.group(1)) if m else None), ""


# ---------------------------------------------------------------------------
# identity and refusals
# ---------------------------------------------------------------------------

def identity() -> dict:
    """{provider, model, id, family} of the judge this server is configured with."""
    from service import config
    model = config.JUDGE_MODEL
    if model == "stub":
        return {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}
    provider = config.JUDGE_PROVIDER
    return {"provider": provider, "model": model,
            "id": f"{provider}/{model}" if provider and model else (model or ""),
            "family": family(model) if model else ""}


def judge_family(ident: dict, stamp: dict | None = None) -> str:
    """The family the same-family rule compares against. A local judge's is
    its weights' when the server names them: the served id ('chat') is
    whatever someone typed at launch and says nothing about the family."""
    weights = (stamp or {}).get("weights")
    return family(weights) if weights else ident["family"]


def provider_clash() -> str:
    """The exam writer's or generator's provider, when it equals the judge's."""
    from service import config
    p = config.JUDGE_PROVIDER
    if not p or p == "fake":
        return ""
    if p == config.EXAM_PROVIDER:
        return "exam writer"
    if p == config.LLM_PROVIDER:
        return "generator"
    return ""


def single_provider_loop() -> bool:
    from service import config
    return bool(provider_clash()) and config.ALLOW_SINGLE_PROVIDER_LOOP


def blocked() -> str:
    """'' when a judged run can be graded, else the reason — shown on the page
    rather than crashing the container (the exam and the board still work)."""
    from service import config, llm
    if not config.JUDGE_MODEL:
        return ("no judge is configured on this server (JUDGE_MODEL is unset) — the judged "
                "suite is off")
    if config.JUDGE_MODEL == "stub":
        return ""
    p = config.JUDGE_PROVIDER
    if not p:
        return "JUDGE_PROVIDER is unset — the judge is an API call and needs one"
    if p not in llm.PROVIDERS:
        return f"JUDGE_PROVIDER={p!r} is not one of {', '.join(llm.PROVIDERS)}"
    # a local server's id cannot be pinned, so for `local` this refusal
    # becomes a stamp instead (assemble): it runs, and nothing it writes counts
    if p not in ("fake", "local") and not DATED.search(config.JUDGE_MODEL):
        return (f"JUDGE_MODEL={config.JUDGE_MODEL!r} is a floating alias, not a dated model id — "
                f"pin it (e.g. claude-sonnet-4-5-20250929, gpt-4.1-2025-04-14) or a vendor "
                f"update silently re-bases every score")
    if llm.needs_key(p) and not config.JUDGE_API_KEY:
        return "JUDGE_API_KEY is unset — put it in .env, never in docker-compose.yml"
    clash = provider_clash()
    if clash and not config.ALLOW_SINGLE_PROVIDER_LOOP:
        return (f"the judge's provider ({p}) is the same as the {clash}'s — a loop whose "
                f"questions, grades and training data come from one family grades itself. "
                f"Use a different provider, or set ALLOW_SINGLE_PROVIDER_LOOP=1 for a trial "
                f"(every judged score is then stamped with it)")
    return ""


# ---------------------------------------------------------------------------
# the stub — a deterministic stand-in for tests and dry runs
# ---------------------------------------------------------------------------

_STOP = set("the a an of to in and or is are was were be it its this that for on with as by at from "
            "which who what when where how not no yes".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP}


class StubGrader:
    """Content-word recall of the reference plus the length clause. NOT a
    judge: it exists so the plumbing (shapes, hashes, halves, the canary, the
    control join, the calibration round trip) can be tested without a
    provider. Every file it writes says so."""
    id = "stub/overlap-v1"
    CRITERIA_MARK = "CRITERIA (0.0–1.0 each)"

    @staticmethod
    def _parts(prompt: str) -> tuple[str, str]:
        """(reference, answer) from either prompt — the single-score one or
        the per-criterion one, which words its reference line differently."""
        for head in ("REFERENCE (what a full-marks answer contains)\n",
                     "REFERENCE (what a correct answer must respect)\n"):
            if head in prompt:
                ref, ans = prompt.split(head, 1)[1].split("\n\nCANDIDATE ANSWER\n", 1)
                return ref, ans
        raise ValueError("neither prompt's REFERENCE section is in this text")

    @staticmethod
    def reply(prompt: str) -> str:
        """What a grader would have sent back, in the shape the prompt asked
        for. The fake backend and --stub both answer through this."""
        if StubGrader.CRITERIA_MARK in prompt:
            return json.dumps(StubGrader.grade_criteria(prompt))
        s, j = StubGrader.grade(prompt)
        return json.dumps({"score": s, "justification": j})

    @staticmethod
    def grade_criteria(prompt: str) -> dict:
        """The same word-overlap reading, spread across whatever criteria the
        prompt lists — still NOT a judgement, and still deterministic. Two
        details exist to exercise the paths that matter: a conditional
        criterion is null unless the item is about a medication, and an empty
        answer to an emergency question trips the topic's first flag, which
        is exactly the case the framework is built to catch."""
        ref, ans = StubGrader._parts(prompt)
        block = prompt.split("\nCRITERIA (0.0–1.0 each)\n", 1)[1].split("\n\nFLAGS", 1)[0]
        flag_block = prompt.split("\nFLAGS (true/false each)\n", 1)[1].split("\n\nQUESTION\n", 1)[0]
        a, r = _content(ans), _content(ref)
        recall = len(a & r) / len(r) if (a and r) else 0.0
        empty = not a or ans.strip() == "(empty)"
        about_medication = "medication" in (ref + ans).lower()
        criteria: dict[str, float | None] = {}
        for line in block.splitlines():
            if not line.strip():
                continue
            cid = line.split(" — ", 1)[0].strip()
            if "CONDITIONAL" in line and not about_medication:
                criteria[cid] = None
            else:
                criteria[cid] = round(min(1.0, recall), 4)
        fids = [ln.split(" — ", 1)[0].strip() for ln in flag_block.splitlines() if ln.strip()
                and not ln.startswith("(none")]
        dangerous = bool(empty and "Acuity: emergency" in ref)
        flags = {fid: (dangerous and i == 0) for i, fid in enumerate(fids)}
        return {"criteria": criteria, "flags": flags,
                "justification": (f"the answer covers {recall:.0%} of the reference's substance"
                                  if not empty else "no answer to grade")}

    @staticmethod
    def grade(prompt: str) -> tuple[int, str]:
        ref, ans = StubGrader._parts(prompt)
        a, r = _content(ans), _content(ref)
        if not a or not r or ans.strip() == "(empty)":
            return 0, "no answer, or nothing from the reference in it"
        recall = len(a & r) / len(r)
        score = 4 if recall >= 0.8 else 3 if recall >= 0.6 else 2 if recall >= 0.35 \
            else 1 if recall > 0 else 0
        why = f"the answer covers {recall:.0%} of the reference's substance"
        if score >= 3 and words(ans) > max(60, 3 * words(ref)):
            score -= 1
            why += "; it is far longer than the reference, so the length clause costs a point"
        return score, why


def stub_results(requests) -> dict:
    from service import llm
    out = {}
    for r in requests:
        out[r.custom_id] = llm.Result(text=StubGrader.reply(r.user))
    return out


# ---------------------------------------------------------------------------
# reading the logs
# ---------------------------------------------------------------------------

def _records(model_dir: Path, task: str) -> list[dict]:
    dirs = [d for d in model_dir.glob(f"{task}_*shot") if re.fullmatch(rf"{task}_\d+shot", d.name)]
    files = dx.newest_per_subtask(sorted(f for d in dirs for f in d.rglob("samples_*.jsonl")))
    out = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def answered_fingerprint(model_dir: Path, task: str) -> str | None:
    """The fingerprint of the question set this model's answers on `task`
    were given on (exam_build.fingerprint over the items it answered), or
    None when there are no answers."""
    keys = [_exam.item_key(r.get("doc") or {}) for r in _records(model_dir, task)]
    return _exam.fingerprint(keys) if any(keys) else None


def _answer(rec: dict) -> str:
    """The raw generation, exactly as the harness logged it — the record of
    what the model wrote. What is graded and shown is answer_parts()."""
    r = rec.get("filtered_resps") or rec.get("resps") or []
    while isinstance(r, list) and r:
        r = r[0]
    return str(r) if isinstance(r, str) else ""


# ---------------------------------------------------------------------------
# 11l: reasoning models write a monologue before the answer. Qwen3's run #60
# spent its whole budget inside <think> on every one of 3,730 questions, and
# the judge graded 3,730 cut-off monologues as answers. The monologue is
# taken out before anything is graded or shown, and an answer that never left
# it is no answer — never a low score.
#
# The wrappers are a setting: REASONING_WRAPPERS="<think>,</think>;<r>,</r>"
# adds pairs to the default, so the next model's tag is a config change.
# ---------------------------------------------------------------------------
# 12h.1: Gemma 4 thinks inside "<|channel>thought … <channel|>" (its chat
# template's strip_thinking), not <think>
DEFAULT_REASONING_WRAPPERS = (("<think>", "</think>"), ("<|channel>", "<channel|>"))
NO_ANSWER_WHY = "the model never finished answering: these questions were not scored"


def reasoning_wrappers() -> list[tuple[str, str]]:
    out = list(DEFAULT_REASONING_WRAPPERS)
    for pair in os.environ.get("REASONING_WRAPPERS", "").split(";"):
        if "," not in pair:
            continue
        o, c = (x.strip() for x in pair.split(",", 1))
        if o and c and (o, c) not in out:
            out.append((o, c))
    return out


def split_reasoning(text: str) -> dict:
    """{answer_text, reasoning_text, had_reasoning, reasoning_unterminated}.

    Every reasoning block comes out. A closing tag with no opening one before
    it closes a block the chat template opened in the prompt (DeepSeek-R1's
    distillations do this), so everything before it is reasoning. A block
    that opens and never closes takes the rest of the generation with it.
    A generation with no block is returned exactly as it was written, so a
    model that does not reason is graded on the very same text as before."""
    text = text or ""
    rest, parts, had, open_end = text, [], False, False
    for o, c in reasoning_wrappers():
        while True:
            i, j = rest.find(o), rest.find(c)
            if i >= 0 and (j < 0 or i < j):
                k = rest.find(c, i + len(o))
                had = True
                if k < 0:
                    parts.append(rest[i + len(o):])
                    rest, open_end = rest[:i], True
                    break
                parts.append(rest[i + len(o):k])
                rest = rest[:i] + rest[k + len(c):]
            elif j >= 0:
                had = True
                parts.append(rest[:j])
                rest = rest[j + len(c):]
            else:
                break
    if not had:
        return {"answer_text": text, "reasoning_text": "", "had_reasoning": False,
                "reasoning_unterminated": False}
    return {"answer_text": rest.strip(),
            "reasoning_text": "\n\n".join(p.strip() for p in parts if p.strip()),
            "had_reasoning": True, "reasoning_unterminated": open_end}


def answer_parts(rec: dict) -> dict:
    """The raw generation, split: what the judge grades and the page shows is
    `answer_text`. `no_answer` when a reasoning model never left its block,
    or left it with nothing after — that item is not graded, and counts as
    no answer rather than as a zero. A model that does not reason never has
    `no_answer`: an empty generation from it is graded as it always was."""
    raw = _answer(rec)
    p = split_reasoning(raw)
    p["raw"] = raw
    p["no_answer"] = p["had_reasoning"] and (p["reasoning_unterminated"]
                                             or not p["answer_text"])
    return p


def _generation(model_dir: Path, task: str) -> dict | None:
    """What the harness generated the answers with, from its own results
    file: the answer budget and the stop sequences, and any run-wide
    override (--gen_kwargs). It changes what a score means, so it is recorded
    with the grades."""
    files = sorted(f for d in model_dir.glob(f"{task}_*shot")
                   if re.fullmatch(rf"{task}_\d+shot", d.name)
                   for f in d.rglob("results_*.json"))
    if not files:
        return None
    try:
        blob = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    gk = ((blob.get("configs") or {}).get(task) or {}).get("generation_kwargs") or {}
    out = {k: gk[k] for k in ("max_gen_toks", "until") if k in gk}
    over = (blob.get("config") or {}).get("gen_kwargs")
    if over:
        out["override"] = over
    return out or None


def mmlu_outcomes(model_dir: Path) -> dict[str, bool]:
    """doc_hash -> right/wrong on the model's own MMLU run (the MC side of
    the control comparison)."""
    out: dict[str, bool] = {}
    for rec in _records(model_dir, "mmlu"):
        pm = dx.primary_metric(rec)
        dh = rec.get("doc_hash")
        if pm and dh:
            out[dh] = pm[1] >= 0.5
    return out


def length_bucket(n: int) -> str:
    for lo, hi, label in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return label
    return LENGTH_BUCKETS[-1][2]


def load_canary() -> list[dict]:
    return [json.loads(x) for x in CANARY_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# a run: plan the requests, then assemble the file from the results
# ---------------------------------------------------------------------------

def plan_requests(model_dir: Path, judge_family: str,
                  only: list[str] | None = None) -> tuple[list, dict]:
    """Every grading request for one model — the canary first, then every
    answer — plus the plan the results are assembled against later (the
    poller may be in another process by then). `only` narrows it to the tasks
    a run asked for: a person sitting one topic should not pay to re-grade
    fourteen others whose answers are already on disk."""
    from service import llm
    model_id = model_dir.name.replace("__", "/", 1)
    tasks_present = [t for t in ALL_TASKS if any(model_dir.glob(f"{t}_*shot"))]
    if only:
        tasks_present = [t for t in tasks_present if t in set(only)]
    plan = {"model": model_id, "model_dir": str(model_dir), "tasks": {}, "canary": [],
            "skipped": None}
    if not tasks_present:
        return [], plan
    if family(model_id) == judge_family:
        plan["skipped"] = f"not judged — same family as judge ({judge_family})"
        return [], plan
    reqs = []
    rub_exam = rubric_for("exam_x")[0]
    for c in load_canary():
        cid = f"canary:{c['id']}"
        reqs.append(llm.Request(custom_id=cid, system="", max_tokens=200, json=True,
                                user=build_prompt(rub_exam, c["prompt"], c["reference"], c["answer"]),
                                meta={"kind": "canary", "id": c["id"], "human_score": c["human_score"]}))
        plan["canary"].append({"cid": cid, "id": c["id"], "human_score": c["human_score"]})
    mc = mmlu_outcomes(model_dir) if CONTROL_TASK in tasks_present else {}
    safe = model_dir.name
    plan["answer_stats"] = {}
    # which questions these answers answer, from the answers themselves: a
    # grade is about the question set its answers were given on, whatever
    # the task is called now
    plan["bank"] = {}
    for task in tasks_present:
        plan["bank"][task] = answered_fingerprint(model_dir, task)
        rub = rubric_for(task)
        spec = rub.criteria           # a criteria file beside the rubric: grade every one
        items = []
        # what the model actually wrote, in aggregate. A model that wrote
        # nothing on a topic has not revealed a gap in that topic, and the
        # step that picks what to train must be able to say so.
        stats = {"n": 0, "empty": 0, "short": 0, "words": 0}
        seen: set[str] = set()
        gen = _generation(model_dir, task)
        if gen:
            plan.setdefault("generation", {})[task] = gen
        for i, rec in enumerate(sorted(_records(model_dir, task), key=lambda r: str(r.get("doc_hash")))):
            doc = rec.get("doc") or {}
            parts = answer_parts(rec)
            ans = parts["answer_text"]
            qid = doc.get("qid")
            cid = f"judge:{safe}:{task}:{i}"
            if parts["no_answer"]:
                # 11l: never sent to the judge — there is nothing to grade.
                # Counted, and kept out of every mean
                stats["n"] += 1
                stats["no_answer"] = stats.get("no_answer", 0) + 1
                item = {"doc_hash": rec.get("doc_hash"), "id": doc.get("id"), "qid": qid,
                        "half": ("diagnose" if task == CONTROL_TASK
                                 else dx.split_of(qid) if qid else None),
                        "category": doc.get("category"), "answer_words": 0,
                        "no_answer": True, "had_reasoning": True,
                        "reasoning_words": words(parts["reasoning_text"])}
                if parts["reasoning_unterminated"]:
                    item["reasoning_unterminated"] = True
                if isinstance(doc.get("meta"), dict):
                    item["meta"] = doc["meta"]
                if task == CONTROL_TASK:
                    item["mmlu_doc_hash"] = doc.get("mmlu_doc_hash")
                    item["mc_right"] = mc.get(doc.get("mmlu_doc_hash"))
                items.append(item)
                continue
            # a 15-key JSON reply is a few hundred tokens; a single score is
            # a few dozen. The cap is per request, so it follows the prompt.
            user = (build_criteria_prompt(rub.text, spec, doc.get("prompt", ""),
                                          doc.get("reference", ""), ans) if spec else
                    build_prompt(rub.text, doc.get("prompt", ""),
                                 doc.get("reference", ""), ans))
            reqs.append(llm.Request(custom_id=cid, system="", json=True, user=user,
                                    max_tokens=700 if spec else 300,
                                    meta={"kind": "judge", "task": task, "qid": qid}))
            stats["n"] += 1
            stats["words"] += words(ans)
            norm = re.sub(r"\s+", " ", (ans or "").strip().lower())
            if not norm:
                stats["empty"] += 1
            elif words(ans) < 3:
                stats["short"] += 1
            seen.add(norm)
            item = {"cid": cid, "doc_hash": rec.get("doc_hash"), "id": doc.get("id"), "qid": qid,
                    "half": ("diagnose" if task == CONTROL_TASK
                             else dx.split_of(qid) if qid else None),
                    "category": doc.get("category"), "answer_words": words(ans)}
            if parts["had_reasoning"]:
                item["had_reasoning"] = True
                item["reasoning_words"] = words(parts["reasoning_text"])
            # what the item is about, when the bank carried it: the acuity of
            # a medical question is the axis its author reads first
            if isinstance(doc.get("meta"), dict):
                item["meta"] = doc["meta"]
            if task == CONTROL_TASK:
                item["mmlu_doc_hash"] = doc.get("mmlu_doc_hash")
                item["mc_right"] = mc.get(doc.get("mmlu_doc_hash"))
            items.append(item)
        stats["distinct"] = len(seen)
        stats["mean_words"] = round(stats["words"] / stats["n"], 2) if stats["n"] else 0
        plan["answer_stats"][task] = stats
        plan["tasks"][task] = items
    return reqs, plan


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def breakdown_fields(items: list[dict], spec: dict | None = None) -> tuple[list[str], list[str]]:
    """(fields to tabulate, fields that hold one value for every item).

    Chosen by what the field DOES to the topic, not by its name: a field with
    between two and twelve distinct values splits the bank into readable
    groups; one value splits nothing and is worth a sentence instead; a
    hundred values (a per-question `intent` sentence) is not a table at all.
    A criteria file may still name its own list, and then that list is used.
    """
    counts: dict[str, set] = {}
    for it in items:
        for k, v in (it.get("meta") or {}).items():
            if v is None or v == "" or isinstance(v, (list, dict)):
                continue
            counts.setdefault(str(k), set()).add(str(v))
    named = (spec or {}).get("breakdowns")
    if isinstance(named, list) and named:
        return [f for f in map(str, named) if f in counts], []
    constant = sorted(f for f, vs in counts.items() if len(vs) == 1 and f != "id")
    splits = {f for f, vs in counts.items()
              if BREAKDOWN_MIN_VALUES <= len(vs) <= BREAKDOWN_MAX_VALUES and f != "id"}
    # the fields written to be read that way come first, in this order
    good = [f for f in BREAKDOWN_FIELDS if f in splits]
    # anything else the items carry fills in only when the named ones gave
    # fewer than two tables — three of the topics have a constant acuity and a
    # per-question intent, and their story is difficulty and domain instead
    if len(good) < BREAKDOWN_MIN_TABLES:
        rest = [f for f in BREAKDOWN_EXTRA if f in splits and f not in good]
        rest += sorted(f for f in splits
                       if f not in good and f not in BREAKDOWN_EXTRA)
        good += rest[:BREAKDOWN_MIN_TABLES - len(good)]
    return good[:BREAKDOWN_MAX_TABLES], constant


def _value_order(field: str, values) -> list[str]:
    """The order a person reads the table in: acuity from the questions that
    can kill someone down to the ones that cannot, difficulty 1 to 5, a
    boolean field with the true case first, and anything else
    alphabetically."""
    vals = sorted(values)
    if field == "acuity":
        known = [v for v in ACUITY_ORDER if v in vals]
        return known + [v for v in vals if v not in known]
    if set(vals) <= {"True", "False"}:
        return [v for v in ("True", "False") if v in vals]
    try:
        return sorted(vals, key=lambda v: (float(v), v))
    except (TypeError, ValueError):
        return vals


def _criteria_blocks(items: list[dict], spec: dict) -> dict:
    """What a criteria task records beyond the folded score: how each
    criterion did, how often each flag fired, and both broken out by every
    metadata field the topic carries — a model that is fine on mild and
    fails on emergency, or fine on basic law and lost on the high-risk end,
    being exactly what the framework exists to catch."""
    graded = [it for it in items if it.get("graded") and it.get("criteria")]
    means, counts = {}, {}
    for cid in criteria_ids(spec):
        vals = [it["criteria"][cid] for it in graded
                if it["criteria"].get(cid) is not None]
        means[cid], counts[cid] = (_mean(vals), len(vals)) if vals else (None, 0)
    flags = {}
    n_rep = sum(1 for it in items if it.get("half") == "report")
    n_dia = sum(1 for it in items if it.get("half") == "diagnose")
    for f in spec.get("flags") or []:
        fid = f["id"]
        fired = [it for it in graded if (it.get("flags") or {}).get(fid)]
        rep = sum(1 for it in fired if it.get("half") == "report")
        dia = sum(1 for it in fired if it.get("half") == "diagnose")
        flags[fid] = {
            "n": len(fired), "share": round(len(fired) / len(items), 4) if items else 0,
            # …and per half, because the published score is the report half and
            # a page that prints the whole-bank count under a report-half
            # heading is saying something untrue about the published number
            "n_report": rep, "share_report": round(rep / n_rep, 4) if n_rep else 0,
            "n_diagnose": dia, "share_diagnose": round(dia / n_dia, 4) if n_dia else 0,
            "label": label_of(f), "effect": effect_of(f),
            "effect_words": effect_words(effect_of(f)),
            # the count covers both halves; only diagnose-half qids are named,
            # because a report-half qid is the one thing this file may not leak
            "qids": sorted(it["qid"] for it in fired
                           if it.get("half") == "diagnose" and it.get("qid")),
        }
    out = {
        "criteria_mean": means, "criteria_n": counts,
        "criteria_labels": criteria_labels(spec),
        "flags": flags,
        "unparseable": sum(1 for it in items if not it.get("graded")),
    }
    # which criteria the FILE allows to be skipped — a count below the item
    # total otherwise just means a reply could not be read, and the page
    # should not call that conditional
    if conditional_ids(spec):
        out["criteria_conditional"] = sorted(conditional_ids(spec))
    breakdowns: dict[str, dict] = {}
    fields, constant = breakdown_fields(items, spec)
    if constant:
        # said once, rather than drawn as a table with one row
        out["breakdowns_constant"] = {
            f: sorted({str((it.get("meta") or {}).get(f)) for it in items
                       if (it.get("meta") or {}).get(f) is not None})[0]
            for f in constant}
    for field in fields:
        cells: dict[str, dict] = {}
        for it in items:
            value = (it.get("meta") or {}).get(field)
            if value is None or value == "":
                continue
            cell = cells.setdefault(str(value), {"n": 0, "scores": [],
                                                 "flags": {fid: 0 for fid in flags}})
            cell["n"] += 1
            cell["scores"].append(it["score"])
            for fid in flags:
                cell["flags"][fid] += 1 if (it.get("flags") or {}).get(fid) else 0
        if cells:
            breakdowns[field] = {v: {"n": cells[v]["n"], "mean": _mean(cells[v]["scores"]),
                                     "flags": cells[v]["flags"]}
                                 for v in _value_order(field, cells)}
    if breakdowns:
        out["breakdowns"] = breakdowns
    return out


def canary_stats(scores: dict[str, int | None], canary: list[dict],
                 previous: dict[str, int] | None, threshold: float) -> dict:
    """Mean absolute deviation of the canary grades from the human marks and
    from the previous run's grades. Drift is movement between runs."""
    pairs = [(scores.get(c["id"]), c["human_score"]) for c in canary]
    graded = [(s, h) for s, h in pairs if s is not None]
    mad_h = round(sum(abs(s - h) for s, h in graded) / len(graded), 4) if graded else None
    mad_p = None
    if previous:
        both = [(scores[k], previous[k]) for k in previous if scores.get(k) is not None]
        mad_p = round(sum(abs(a - b) for a, b in both) / len(both), 4) if both else None
    drifted = mad_p is not None and mad_p > threshold
    return {"n": len(canary), "graded": len(graded), "mad_vs_human": mad_h,
            "mad_vs_previous": mad_p, "threshold": threshold, "drifted": drifted,
            "scores": {c["id"]: scores.get(c["id"]) for c in canary},
            "human": {c["id"]: c["human_score"] for c in canary}}


def previous_canary(results_root: Path, judge_id: str) -> dict[str, int] | None:
    p = Path(results_root) / CANARY_HISTORY
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("judge_id") == judge_id:
            last = row
    return {k: v for k, v in (last or {}).get("scores", {}).items() if v is not None} or None


def record_canary(results_root: Path, judge_id: str, model: str, stats: dict) -> None:
    p = Path(results_root) / CANARY_HISTORY
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"judge_id": judge_id, "model": model, "at": time.time(),
                             "scores": stats["scores"], "mad_vs_human": stats["mad_vs_human"],
                             "mad_vs_previous": stats["mad_vs_previous"]}) + "\n")


def assemble(plan: dict, results: dict, ident: dict, batch_id: str, results_root: Path,
             threshold: float, caveat: bool, record: bool = True) -> dict:
    """judge.json from a plan and the provider's results. Deterministic for a
    deterministic grader (sorted keys, no timestamps in the body)."""
    from service import llm
    tasks_present = list(plan["tasks"])
    head = {"judge": {"id": ident["id"], "provider": ident["provider"], "model": ident["model"],
                      "family": ident["family"], "batch_id": batch_id,
                      "prompt_sha256": prompt_sha(), "prompt_version": PROMPT_VERSION,
                      "stub": ident["provider"] == "stub", "single_provider_loop": caveat,
                      # per task, because the rubric is now per topic: which
                      # one graded it, its sha, its version, and whether its
                      # author has signed it off
                      "rubrics": {t: _rubric_record(t) for t in tasks_present}},
            "model": plan["model"], "split_salt": dx.SPLIT_SALT, "correct_at": CORRECT_AT}
    # a local judge: the stamp recorded at submit time (with what the server
    # said it serves), or at the least the mark itself — never nothing
    stamp = plan.get("provisional") or llm.local_mark(ident["provider"], ident["model"], "graded")
    if stamp:
        head["judge"].update(stamp)
        head["judge"]["family"] = judge_family(ident, stamp)
    # a second stamp, independent of the judge: a rubric its author has not
    # signed off yet. The page shows it beside the provisional one.
    draft = sorted(t for t, r in head["judge"]["rubrics"].items() if r.get("status") == "draft")
    if draft:
        head["judge"]["rubric_status"] = "draft"
        head["judge"]["rubrics_draft"] = draft
    if plan.get("skipped"):
        return {**head, "skipped": plan["skipped"], "tasks": {}}
    # the canary first: has the judge moved since last time?
    cscores = {}
    for c in plan["canary"]:
        res = results.get(c["cid"])
        cscores[c["id"]] = parse_grade(res.text)[0] if res and not res.error else None
    prev = previous_canary(results_root, ident["id"]) if plan["canary"] else None
    canary = canary_stats(cscores, load_canary(), prev, threshold) if plan["canary"] else None
    if canary and record:
        record_canary(results_root, ident["id"], plan["model"], canary)
    # provisional is a preliminary reason like any other: the page greys it,
    # never ranks it and keeps it out of every average by the same mechanism
    reasons = [stamp["provisional_reason"]] if stamp else []
    if canary and canary["drifted"]:
        reasons.append(f"the judge moved: canary grades differ from the previous run by "
                       f"{canary['mad_vs_previous']} points on average (limit {threshold})")
    if canary and canary["graded"] < canary["n"]:
        reasons.append(f"{canary['n'] - canary['graded']} canary scripts were not graded")
    tasks: dict[str, dict] = {}
    for task, items_meta in plan["tasks"].items():
        spec = rubric_for(task).criteria
        items = []
        for m in items_meta:
            it = {k: v for k, v in m.items() if k != "cid"}
            if m.get("no_answer"):
                # 11l: nothing reached the judge, and nothing is scored
                it.update(score=None, graded=False, justification="")
                items.append(it)
                continue
            res = results.get(m["cid"])
            reply = res.text if res and not res.error else None
            if spec:
                # the 0-4 is folded here, from the criteria the judge gave,
                # by the rule in the criteria file — never asked of the model
                g = parse_grade_criteria(reply, spec) if reply is not None else None
                score = fold(g.criteria, g.flags, spec) if g else None
                just = g.justification if g else ""
                if g:
                    it["criteria"] = g.criteria
                    it["flags"] = g.flags
                    it["fold"] = {"method": FOLD_METHOD,
                                  "applicable": sum(1 for v in g.criteria.values()
                                                    if v is not None),
                                  "effects_applied": effects_applied(g.flags, spec)}
                    if g.extra_keys:
                        it["extra_keys"] = g.extra_keys
            else:
                score, just = parse_grade(reply) if reply is not None else (None, "")
            it["score"] = int(score) if score is not None else 0
            it["graded"] = score is not None
            it["justification"] = just
            items.append(it)
        if not items:
            continue
        t = summarise_task(task, items, spec)
        # which questions, and under what name: a result counts only while
        # this is the task's current fingerprint, and is history after
        t["bank_sha256"] = (plan.get("bank") or {}).get(task)
        t["topic"] = (_exam.TASK_TOPIC.get(task)
                      or (CONTROL_LABEL if task == CONTROL_TASK else items[0].get("category"))
                      or task)
        stats = (plan.get("answer_stats") or {}).get(task)
        if stats:
            t["answers"] = stats
        # what the answers were generated with: the budget changes what a
        # score means, so it travels with the grades (11l)
        gen = (plan.get("generation") or {}).get(task)
        if gen:
            t["generation"] = gen
        tasks[task] = t
    return {**head, "canary": canary, "preliminary_reasons": reasons, "tasks": tasks}


# the keys summarise_task() owns: revalidate() replaces exactly these
AGGREGATE_KEYS = ("n", "mean", "max", "ungraded", "n_report", "score_report", "n_diagnose",
                  "score_diagnose", "dist", "dist_report", "dist_diagnose", "score_vs_length",
                  "items", "control", "no_answer", "no_score", "criteria_mean", "criteria_n",
                  "criteria_labels", "flags", "unparseable", "criteria_conditional",
                  "breakdowns", "breakdowns_constant")


def summarise_task(task: str, items: list[dict], spec: dict | None) -> dict:
    """A topic's numbers from its items. An item with no answer (a reasoning
    model that never finished) is counted and kept, and left out of every
    mean, every distribution and every criterion. When most of a topic's
    items have no answer the topic has no score at all — a dash, never a low
    number. A topic with no such item comes out exactly as it always did."""
    scored = [it for it in items if not it.get("no_answer")]
    n_none = len(items) - len(scored)
    dist = collections.Counter(str(it["score"]) for it in scored)
    by_len: dict[str, list[int]] = collections.defaultdict(list)
    for it in scored:
        by_len[length_bucket(it["answer_words"])].append(it["score"])
    rep = [it["score"] for it in scored if it["half"] == "report"]
    dia = [it["score"] for it in scored if it["half"] == "diagnose"]

    def _dist(xs):
        c = collections.Counter(xs)
        return {str(k): c.get(k, 0) for k in range(MAX_SCORE + 1)}
    t = {"n": len(items),
         "mean": round(sum(it["score"] for it in scored) / len(scored), 4) if scored else None,
         "max": MAX_SCORE, "ungraded": sum(1 for it in scored if not it["graded"]),
         "n_report": len(rep), "score_report": round(sum(rep) / len(rep), 4) if rep else None,
         "n_diagnose": len(dia),
         "score_diagnose": round(sum(dia) / len(dia), 4) if dia else None,
         "dist": {str(k): dist.get(str(k), 0) for k in range(MAX_SCORE + 1)},
         # per half, so a before/after comparison can carry a standard error
         "dist_report": _dist(rep), "dist_diagnose": _dist(dia),
         "score_vs_length": [{"bucket": label, "n": len(by_len[label]),
                              "mean": round(sum(by_len[label]) / len(by_len[label]), 4)}
                             for _, _, label in LENGTH_BUCKETS if by_len.get(label)],
         "items": items}
    if n_none:
        t["no_answer"] = n_none
        if n_none * 2 > len(items):
            # most of it never became an answer: no number at all, and why
            t.update(mean=None, score_report=None, score_diagnose=None, no_score=NO_ANSWER_WHY)
    if spec:
        t.update(_criteria_blocks(scored, spec))
    if task == CONTROL_TASK:
        ctl: dict[str, dict] = {}
        for it in scored:
            c = ctl.setdefault(it["category"] or "—", {"n": 0, "mc_wrong": 0, "knew": 0,
                                                        "didnt": 0, "unjoined": 0})
            c["n"] += 1
            if it["mc_right"] is None:
                c["unjoined"] += 1
            elif not it["mc_right"]:
                c["mc_wrong"] += 1
                c["knew" if it["score"] >= CORRECT_AT else "didnt"] += 1
        t["control"] = ctl
    return t


CONTROL_LABEL = "MMLU control"
# what a history entry keeps: enough to say what the model scored on a
# question set nobody sits any more, and when — not the items, which name
# questions and answers the page has no business showing again
HISTORY_KEYS = ("topic", "bank_sha256", "judged_at", "n", "mean", "n_report", "score_report",
                "n_diagnose", "score_diagnose")


def history_entry(task: str, t: dict, judge: dict | None = None,
                  judged_at: float | None = None) -> dict:
    """One judged topic, as history: its task, what it scored, on which
    question set, when, and by which judge. A file from before topics
    carried their own name gives it through its items' category."""
    e = {k: t.get(k) for k in HISTORY_KEYS if t.get(k) is not None}
    e["task"] = task
    if not e.get("topic"):
        items = t.get("items") or []
        e["topic"] = (items[0].get("category") if items and task != CONTROL_TASK else None) \
            or (CONTROL_LABEL if task == CONTROL_TASK else _exam.task_slug(task).replace("_", " "))
    if not e.get("judged_at") and judged_at:
        e["judged_at"] = int(judged_at)
    jd_ = judge or {}
    if jd_.get("id"):
        e["judge_id"] = jd_["id"]
    if jd_.get("provisional"):
        e["provisional"] = True
    return e


def _history_key(e: dict) -> tuple:
    return (e.get("task"), e.get("bank_sha256"), e.get("judged_at"))


def _add_history(hist: list[dict], entries) -> list[dict]:
    seen = {_history_key(e) for e in hist}
    for e in entries:
        if _history_key(e) not in seen:
            hist.append(e)
            seen.add(_history_key(e))
    return sorted(hist, key=lambda e: (-(e.get("judged_at") or 0), str(e.get("task"))))


def split_by_bank(j: dict | None, current: dict[str, str] | None,
                  judged_at: float | None = None) -> tuple[dict | None, list[dict]]:
    """(the file with only the topics that count now, the history). A topic
    counts only when its `bank_sha256` is the task's current fingerprint;
    anything else — graded on a question set the task no longer holds, or
    before fingerprints were recorded at all — is history, shown as such
    and nowhere else. `current` None means the current exam is unknown (a
    report built without the exam directory): nothing is filtered."""
    if not isinstance(j, dict):
        return j, []
    hist = list(j.get("history") or [])
    if current is None or not isinstance(j.get("tasks"), dict):
        return j, hist
    keep, gone = {}, []
    for task, t in j["tasks"].items():
        if isinstance(t, dict) and t.get("bank_sha256") and t["bank_sha256"] == current.get(task):
            keep[task] = t
        elif isinstance(t, dict):
            gone.append(history_entry(task, t, j.get("judge"), t.get("judged_at") or judged_at))
    return {**j, "tasks": keep}, _add_history(hist, gone)


def merge_judged(model_dir: Path, out: dict, dest: Path | None = None,
                 when: float | None = None) -> dict:
    """A run narrowed to one topic must not wipe the others' scores out of
    judge.json — but it must not pretend they were graded by this run's
    instrument either. A previous task is carried over only when the judge id,
    the prompt and that task's rubric record are all identical to this run's;
    anything else is named in `judge.replaced` and dropped, because a file
    that mixes two instruments under one heading is worse than a gap.

    Each topic carries `judged_at`, the time ITS grades landed: this run's
    topics get `when` (now), a carried topic keeps its own. The file's mtime
    is the time of the last merge, and a one-topic run re-stamped every topic
    in the file with it. A topic carried from a file written before this
    field existed takes that file's mtime — the last time it was written, and
    the best that file can say.

    A grade on a question set the task no longer holds is neither replaced
    nor dropped: it moves to `history` (history_entry), with its fingerprint
    and time — and so does a topic graded before fingerprints existed, or
    one the exam no longer has. That is what the model page lists under
    Earlier exams, and nothing else reads it."""
    when = int(time.time() if when is None else when)
    for t in (out.get("tasks") or {}).values():
        t["judged_at"] = when
    prev_path = (dest or model_dir) / "judge.json"
    if not prev_path.exists() or out.get("skipped"):
        return out
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        prev_at = int(prev_path.stat().st_mtime)
    except (ValueError, OSError):
        return out
    if prev.get("skipped") or not isinstance(prev.get("tasks"), dict):
        return out
    pj, nj = prev.get("judge") or {}, out.get("judge") or {}
    same_judge = (pj.get("id") == nj.get("id")
                  and pj.get("prompt_sha256") == nj.get("prompt_sha256")
                  and bool(pj.get("provisional")) == bool(nj.get("provisional")))
    # a grade on a question set the task no longer holds is not replaced
    # by the new one and not dropped: it is history — what this model
    # scored on the old questions, and when
    history = list(prev.get("history") or [])
    earlier = []
    kept, dropped = {}, []
    for task, t in prev["tasks"].items():
        if not isinstance(t, dict):
            continue
        at = t.get("judged_at") or prev_at
        if task in out.get("tasks", {}):
            if t.get("bank_sha256") != out["tasks"][task].get("bank_sha256"):
                earlier.append(history_entry(task, t, pj, at))
            continue
        prev_rub = (pj.get("rubrics") or {}).get(task)
        if not t.get("bank_sha256") or task not in ALL_TASKS:
            # graded before fingerprints, or a topic the exam no longer has
            earlier.append(history_entry(task, t, pj, at))
        elif same_judge and prev_rub == _rubric_record(task):
            kept[task] = {**t, "judged_at": at}
        else:
            dropped.append(task)
    history = _add_history(history, earlier)
    merged = {**out, "tasks": {**kept, **out.get("tasks", {})}}
    if history:
        merged["history"] = history
    if not kept and not dropped:
        return merged
    judge = dict(nj)
    judge["rubrics"] = {**{t: (pj.get("rubrics") or {}).get(t) for t in kept},
                        **(nj.get("rubrics") or {})}
    draft = sorted(t for t, r in judge["rubrics"].items() if (r or {}).get("status") == "draft")
    if draft:
        judge["rubric_status"] = "draft"
        judge["rubrics_draft"] = draft
    if dropped:
        judge["replaced"] = sorted(dropped)
        judge["replaced_why"] = ("graded earlier by a different judge, prompt or rubric — "
                                 "re-run suite=judged for these topics")
    merged["judge"] = judge
    return merged


def backfill_judged_at(j: dict, runs: list[dict]) -> list[str]:
    """Give each topic of a judge.json written before topics carried their own
    time the time its grades landed, from the service's record of the runs
    that graded this model: [{batch_id, finished_at, tasks}] (finished runs
    only). The file's last writer must be one of those runs — a file written
    by the CLI or the stub has no such record, and its mtime is the right
    answer for all of it. A topic takes the newest run, finishing no later
    than that last writer, whose plan graded it; a topic no run accounts for
    is left alone. Returns the topics filled in; only adds `judged_at`."""
    last = next((r for r in runs if r["batch_id"] == (j.get("judge") or {}).get("batch_id")),
                None)
    if not last:
        return []
    filled = []
    for task, t in (j.get("tasks") or {}).items():
        if not isinstance(t, dict) or t.get("judged_at"):
            continue
        at = max((r["finished_at"] for r in runs
                  if task in r["tasks"] and r["finished_at"] <= last["finished_at"]), default=None)
        if at:
            t["judged_at"] = int(at)
            filled.append(task)
    return filled


def revalidate(model_dir: Path, dest: Path | None = None) -> dict | None:
    """11l: re-read a judged model's answers against its judge.json, and take
    out of the grades every answer that never left its reasoning block — for
    a file graded before the split existed, like run #60's. The judge is not
    asked again: whether an answer exists is a fact of the answer on disk.

    Nothing is deleted silently. What the judge said about a non-answer is
    kept on the item under `voided`, and the topic says why it has no score.
    Only a topic with such an answer is touched, so a model that does not
    reason keeps its file exactly as it was. Returns {task: no-answer count}
    for the topics that changed, or None when nothing did."""
    p = (dest or model_dir) / "judge.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if j.get("skipped") or not isinstance(j.get("tasks"), dict):
        return None
    changed: dict[str, int] = {}
    for task, t in j["tasks"].items():
        if not isinstance(t, dict) or not t.get("items"):
            continue
        by_hash = {rec.get("doc_hash"): rec for rec in _records(model_dir, task)}
        touched = False
        for it in t["items"]:
            rec = by_hash.get(it.get("doc_hash"))
            if rec is None:
                continue
            parts = answer_parts(rec)
            if parts["had_reasoning"] and not it.get("had_reasoning"):
                it["had_reasoning"] = True
                it["reasoning_words"] = words(parts["reasoning_text"])
                touched = True
            if parts["no_answer"] and not it.get("no_answer"):
                it["voided"] = {k: it[k] for k in ("score", "graded", "justification",
                                                   "criteria", "flags", "fold") if k in it}
                for k in ("criteria", "flags", "fold", "extra_keys"):
                    it.pop(k, None)
                it.update(score=None, graded=False, justification="", no_answer=True,
                          answer_words=0)
                if parts["reasoning_unterminated"]:
                    it["reasoning_unterminated"] = True
                touched = True
        if not touched:
            continue
        items = t["items"]
        for k in AGGREGATE_KEYS:
            t.pop(k, None)
        t.update(summarise_task(task, items, rubric_for(task).criteria))
        if isinstance(t.get("answers"), dict) and t.get("no_answer"):
            t["answers"]["no_answer"] = t["no_answer"]
        # and what those answers were generated with: for run #60, the 256
        # tokens that ran out inside the reasoning
        gen = _generation(model_dir, task)
        if gen and not t.get("generation"):
            t["generation"] = gen
        changed[task] = t.get("no_answer", 0)
    if changed:
        write_judge(model_dir, j, dest)
    return changed or None


def conditional_report(model_dir: Path) -> list[dict]:
    """11m §5, read only: did the judge decide whether each conditional
    criterion applies, or score it on every answer? One row per judged topic
    that has conditional criteria — how many (answer, criterion) pairs came
    back null (did not apply) against how many were scored, the conditionals
    never skipped once, and the topic mean as graded beside the mean with
    every conditional left out, which bounds what scoring them can have
    moved. Writes nothing and prints no question."""
    p = model_dir / "judge.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = []
    for task, t in sorted((j.get("tasks") or {}).items()):
        if not isinstance(t, dict) or not t.get("items"):
            continue
        spec = rubric_for(task).criteria
        cond = [c["id"] for c in (spec or {}).get("criteria") or [] if c.get("conditional")]
        if not cond:
            continue
        scored = [it for it in t["items"] if it.get("graded") and isinstance(it.get("criteria"), dict)
                  and not it.get("no_answer")]
        if not scored:
            continue
        vals = [it["criteria"].get(c) for it in scored for c in cond]
        applied = [v for v in vals if v is not None]
        never = [c for c in cond if all(it["criteria"].get(c) is not None for it in scored)]
        without = [fold({k: (None if k in cond else v) for k, v in it["criteria"].items()},
                        it.get("flags") or {}, spec) for it in scored]
        rows.append({
            "task": task, "conditional": len(cond), "answers": len(scored),
            "pairs": len(vals), "applied": len(applied), "never_skipped": len(never),
            "mean_when_scored": round(_mean(applied), 3) if applied else None,
            "at_zero": round(sum(1 for v in applied if v == 0.0) / len(applied), 3)
            if applied else None,
            "mean_as_graded": round(_mean([it["score"] for it in scored]), 3),
            "mean_without": round(_mean(without), 3),
        })
    return rows


def write_judge(model_dir: Path, out: dict, dest: Path | None = None) -> Path:
    d = dest or model_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / "judge.json"
    p.write_text(json.dumps(out, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return p


def run_stub(model_dir: Path, results_root: Path | None = None, record: bool = False,
             threshold: float = 0.5, only: list[str] | None = None) -> dict | None:
    """Grade with the stub, in process. The fixture and the tests."""
    ident = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1", "family": "stub"}
    reqs, plan = plan_requests(model_dir, "stub", only=only)
    if not plan["tasks"] and not plan.get("skipped"):
        return None
    return assemble(plan, stub_results(reqs), ident, "stub", results_root or model_dir.parent,
                    threshold, caveat=False, record=record)


def start_run(model_dir: Path, results_root: Path, only: list[str] | None = None,
              submission: int | None = None) -> dict:
    """The service's entry point, called inside the GPU lock: plan, submit
    the batch (seconds), return. The poller finishes it. With JUDGE_MODEL=stub
    the file is written here and now. `only` narrows the run to the tasks the
    submission asked for; `submission` is the queue row that asked, and the
    batch id goes on it the moment the provider hands one back — before the
    batch is registered, so the poller can never finish a batch whose row
    does not know it."""
    from service import config, db, llm
    ident = identity()
    if config.JUDGE_MODEL == "stub":
        out = run_stub(model_dir, results_root, record=True,
                       threshold=config.JUDGE_CANARY_MAX_DRIFT, only=only)
        if out is None:
            return {"mode": "stub", "written": False}
        write_judge(model_dir, merge_judged(model_dir, out))
        return {"mode": "stub", "written": True, "skipped": out.get("skipped")}
    why = blocked()
    if why:
        raise RuntimeError(why)
    backend = llm.client("judge")
    stamp = llm.provisional(backend, "graded")
    reqs, plan = plan_requests(model_dir, judge_family(ident, stamp), only=only)
    if stamp:
        plan["provisional"] = stamp
    if plan.get("skipped"):
        write_judge(model_dir, assemble(plan, {}, ident, "", results_root,
                                        config.JUDGE_CANARY_MAX_DRIFT, single_provider_loop()))
        return {"mode": "batch", "written": True, "skipped": plan["skipped"]}
    if not reqs:
        return {"mode": "batch", "written": False}
    bid = backend.submit(reqs)
    if submission:
        db.update(submission, judge_batch=bid)
    rid = db.judge_run_create(plan["model"], bid, len(reqs), ident["id"], json.dumps(plan))
    db.batch_add(bid, "judge", rid, len(reqs), backend.name, backend.model)
    return {"mode": "batch", "written": False, "batch_id": bid, "run_id": rid, "n": len(reqs)}


def finish_run(run: dict, results: dict, results_root: Path) -> Path:
    """The poller's half: results in, judge.json out."""
    from service import config
    plan = json.loads(run["plan"])
    ident = identity()
    out = assemble(plan, results, ident, run["batch_id"], results_root,
                   config.JUDGE_CANARY_MAX_DRIFT, single_provider_loop())
    model_dir = Path(plan["model_dir"])
    return write_judge(model_dir, merge_judged(model_dir, out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("-m", "--model", action="append", default=[])
    ap.add_argument("--stub", action="store_true",
                    help="deterministic overlap stand-in; tests and dry runs only")
    ap.add_argument("--wait", action="store_true",
                    help="API judge: submit the batch and poll until it completes")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write judge.json under this directory instead of beside the results")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--revalidate", action="store_true",
                    help="take answers that never left a reasoning block out of every "
                         "judge.json already written; asks no judge (11l)")
    ap.add_argument("--conditionals", action="store_true",
                    help="report, per judged topic, whether the judge skipped conditional "
                         "criteria that did not apply or scored them on every answer; "
                         "writes nothing (11m)")
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
    if a.conditionals:
        want = {m.replace("/", "__") for m in a.model}
        pairs = applied = topics = 0
        for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
            if want and d.name not in want:
                continue
            for r in conditional_report(d):
                topics += 1
                pairs += r["pairs"]
                applied += r["applied"]
                print(f"{d.name}  {r['task']}: {r['conditional']} conditional criteria, scored "
                      f"on {r['applied']:,} of {r['pairs']:,} answer-criterion pairs "
                      f"({100 * r['applied'] / r['pairs']:.0f}%), {r['never_skipped']} never "
                      f"skipped; " + (f"{r['mean_when_scored']:.2f} on average when scored, "
                                      f"{100 * r['at_zero']:.0f}% at 0.0"
                                      if r["applied"] else "none ever scored")
                      + f"; topic mean {r['mean_as_graded']:.2f} "
                      f"/ 4 as graded, {r['mean_without']:.2f} / 4 with every conditional "
                      f"left out")
        if not topics:
            print("no judged topic with conditional criteria")
        else:
            print(f"conditionals: scored on {applied:,} of {pairs:,} pairs "
                  f"({100 * applied / pairs:.0f}%) across {topics} judged topic(s) — "
                  f"a judge that decides applicability leaves some null")
        return 0
    if a.revalidate:
        want = {m.replace("/", "__") for m in a.model}
        touched = 0
        for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
            if want and d.name not in want:
                continue
            changed = revalidate(d)
            if not changed:
                continue
            touched += 1
            gone = sum(changed.values())
            j = json.loads((d / "judge.json").read_text(encoding="utf-8"))
            dashed = sum(1 for t in changed if (j["tasks"].get(t) or {}).get("no_score"))
            print(f"{d.name}: {gone} answers never finished — {dashed} of {len(changed)} "
                  f"topics now have no score")
        print(f"revalidated: {touched} model(s) changed, the rest untouched")
        return 0
    from service import config, llm
    if not a.stub:
        why = blocked()
        if why or config.JUDGE_MODEL == "stub":
            print(why or "JUDGE_MODEL=stub: pass --stub", file=sys.stderr)
            return 2
        if not a.wait:
            print("an API judge is asynchronous: pass --wait here, or submit through the "
                  "service (suite=judged) and let its poller finish the batch", file=sys.stderr)
            return 2
    want = {m.replace("/", "__") for m in a.model}
    n = 0
    ident = {"provider": "stub", "model": "overlap-v1", "id": "stub/overlap-v1",
             "family": "stub"} if a.stub else identity()
    backend, stamp = None, {}
    if not a.stub:
        try:
            backend = llm.client("judge")
        except llm.LLMError as e:          # a local server that is down, or serves another id
            print(e, file=sys.stderr)
            return 2
        stamp = llm.provisional(backend, "graded")
    for d in sorted(p for p in a.results.iterdir() if p.is_dir()):
        if want and d.name not in want:
            continue
        if a.stub:
            out = run_stub(d, a.results, record=True)
        else:
            reqs, plan = plan_requests(d, judge_family(ident, stamp))
            if stamp:
                plan["provisional"] = stamp
            if not plan["tasks"] and not plan.get("skipped"):
                continue
            results, bid = {}, ""
            if reqs:
                bid = backend.submit(reqs)
                while True:
                    state, detail = backend.status(bid)
                    if state == "done":
                        break
                    if state == "failed":
                        print(f"{d.name}: batch {bid} failed: {detail}", file=sys.stderr)
                        break
                    time.sleep(a.poll)
                results = backend.fetch(bid) if state == "done" else {}
            out = assemble(plan, results, ident, bid, a.results, config.JUDGE_CANARY_MAX_DRIFT,
                           single_provider_loop())
        if out is None:
            continue
        write_judge(d, out, (a.out / d.name) if a.out else None)
        n += 1
        if not a.quiet:
            if out.get("skipped"):
                print(f"{d.name:46} {out['skipped']}")
            else:
                def shown(v):
                    x = v["score_report"] if v["score_report"] is not None else v["mean"]
                    return "—" if x is None else f"{x:.2f}/4"
                bits = [f"{t}={shown(v)}(n={v['n']})" for t, v in sorted(out["tasks"].items())]
                c = out.get("canary") or {}
                print(f"{d.name:46} canary MAD {c.get('mad_vs_human')} vs human, "
                      f"{c.get('mad_vs_previous')} vs previous · {' '.join(bits)}")
    if not a.quiet:
        print(f"\nwrote judge.json for {n} model(s) · judge {ident['id']}"
              + (" · STUB — not a judgement" if a.stub else "")
              + (f" · PROVISIONAL — {stamp['provisional_reason']}" if stamp else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
