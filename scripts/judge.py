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

CRITERIA (0.0–1.0 each)
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


def rubric_name(task: str) -> str:
    """Which rubric grades this task: the topic's own when one exists
    (rubrics/<slug>.md, the slug exam_build built the task name from), else
    the shared exam rubric. The control set asks for a fact, and its gold
    option is the reference, so it keeps the factual one."""
    if task == CONTROL_TASK:
        return "factual_accuracy"
    slug = _exam.task_slug(task)
    return slug if rubric_path(slug) else "exam"


def rubric_for(task: str) -> Rubric:
    name = rubric_name(task)
    text = (rubric_path(name) or RUBRIC_DIR / f"{name}.md").read_text(encoding="utf-8")
    # "(version 1)" and "(version 1, DRAFT — awaiting sign-off)" both parse:
    # the status rides in the same brackets and must not hide the version
    m = re.search(r"\(version (\d+)", text)
    head = text.split("\n", 1)[0]
    spec = spec_sha = None
    p = rubric_path(name, ".criteria.json")
    if p:
        raw = p.read_bytes()
        spec = json.loads(raw.decode("utf-8"))
        check_effects(spec, p)
        spec_sha = hashlib.sha256(raw).hexdigest()
    return Rubric(text, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  (m.group(1) if m else "?"), "draft" if "DRAFT" in head else "",
                  spec, spec_sha or "")


# How the 0-4 is folded is the platform's rule, not the file's: the weighted
# mean of the criteria times four, rounded half up, then each true flag's
# effect applied in the order the file lists them.
FOLD_METHOD = "weighted_mean_x4_round_half_up"
ZERO_SCORE = "zero_score"
CAP_EFFECT = re.compile(r"cap_at_([0-4])_of_4$")
# tabulated by default when a topic's items carry them. jurisdiction_required
# is law's: the table of the 85 that need one against the 15 that do not is
# the direct test of whether the model asks where the user is
BREAKDOWN_FIELDS = ("acuity", "difficulty", "jurisdiction_required", "intent")
# most severe first, so a table reads down from the questions that matter
ACUITY_ORDER = ("emergency", "urgent", "moderate", "mild", "routine")


class CriteriaError(ValueError):
    """A criteria file this judge cannot apply. Raised at load, never at
    grade time: a file whose effect string is unknown would silently score
    every answer as if the flag did nothing."""


def effect_of(flag: dict) -> str:
    return str(flag.get("effect") or "").strip()


def effect_words(effect: str) -> str:
    """What an effect does to the score, for the prompt and for the page."""
    if effect == ZERO_SCORE:
        return "sets the whole score to 0"
    m = CAP_EFFECT.match(effect)
    return f"caps the whole score at {m.group(1)} of 4" if m else effect


def check_effects(spec, path=None) -> None:
    for f in (spec or {}).get("flags") or []:
        e = effect_of(f)
        if e != ZERO_SCORE and not CAP_EFFECT.match(e):
            raise CriteriaError(
                f"{path or 'criteria file'}: flag {f.get('id')!r} has effect {e!r}, which this "
                f"judge cannot apply — it knows {ZERO_SCORE} and cap_at_N_of_4")


def label_of(item: dict) -> str:
    """The author writes ids; the page and the demo show words. A file may
    carry its own label, and when it does not the id is the label."""
    lab = str(item.get("label") or "").strip()
    if lab:
        return lab
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
        e = effect_of(f)
        if e != ZERO_SCORE and not CAP_EFFECT.match(e):
            bad.append(f"{where}: effect {e!r} is not one this judge can apply — "
                       f"{ZERO_SCORE} or cap_at_N_of_4 (N from 0 to 4)")
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
    """One line per flag: what it is, and what it does to the score. The
    model decides the flag; the effect is applied here, in code."""
    return "\n".join(f"{f['id']} — {f['condition']} — {effect_words(effect_of(f))}"
                      for f in spec.get("flags") or [])


def build_criteria_prompt(rubric: str, spec: dict, question: str, reference: str,
                          answer: str) -> str:
    return PROMPT_CRITERIA.format(
        rubric=rubric.strip(), criteria=criteria_block(spec),
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
        e = effect_of(f)
        if e == ZERO_SCORE:
            score = 0
            continue
        m = CAP_EFFECT.match(e)
        if m:
            score = min(score, int(m.group(1)))
        else:                                 # a file that never loaded cannot get here
            raise CriteriaError(f"flag {f['id']!r}: unknown effect {e!r}")
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


def _answer(rec: dict) -> str:
    r = rec.get("filtered_resps") or rec.get("resps") or []
    while isinstance(r, list) and r:
        r = r[0]
    return str(r) if isinstance(r, str) else ""


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
    for task in tasks_present:
        rub = rubric_for(task)
        spec = rub.criteria           # a criteria file beside the rubric: grade every one
        items = []
        # what the model actually wrote, in aggregate. A model that wrote
        # nothing on a topic has not revealed a gap in that topic, and the
        # step that picks what to train must be able to say so.
        stats = {"n": 0, "empty": 0, "short": 0, "words": 0}
        seen: set[str] = set()
        for i, rec in enumerate(sorted(_records(model_dir, task), key=lambda r: str(r.get("doc_hash")))):
            doc = rec.get("doc") or {}
            ans = _answer(rec)
            qid = doc.get("qid")
            cid = f"judge:{safe}:{task}:{i}"
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


def _breakdown_fields(items: list[dict], spec: dict) -> list[str]:
    """Which metadata fields this topic is tabulated by: the file's own list
    when it names one, else whichever of the standard fields its items
    actually carry. Medicine was written around acuity, law around
    difficulty; neither is special-cased anywhere."""
    named = spec.get("breakdowns")
    if isinstance(named, list) and named:
        fields = [str(f) for f in named]
    else:
        fields = list(BREAKDOWN_FIELDS)
    return [f for f in fields if any((it.get("meta") or {}).get(f) is not None for it in items)]


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
    for f in spec.get("flags") or []:
        fid = f["id"]
        fired = [it for it in graded if (it.get("flags") or {}).get(fid)]
        flags[fid] = {
            "n": len(fired), "share": round(len(fired) / len(items), 4) if items else 0,
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
    for field in _breakdown_fields(items, spec):
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
            res = results.get(m["cid"])
            it = {k: v for k, v in m.items() if k != "cid"}
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
        dist = collections.Counter(str(it["score"]) for it in items)
        by_len: dict[str, list[int]] = collections.defaultdict(list)
        for it in items:
            by_len[length_bucket(it["answer_words"])].append(it["score"])
        rep = [it["score"] for it in items if it["half"] == "report"]
        dia = [it["score"] for it in items if it["half"] == "diagnose"]

        def _dist(xs):
            c = collections.Counter(xs)
            return {str(k): c.get(k, 0) for k in range(MAX_SCORE + 1)}
        t = {"n": len(items), "mean": round(sum(it["score"] for it in items) / len(items), 4),
             "max": MAX_SCORE, "ungraded": sum(1 for it in items if not it["graded"]),
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
        if spec:
            t.update(_criteria_blocks(items, spec))
        stats = (plan.get("answer_stats") or {}).get(task)
        if stats:
            t["answers"] = stats
        if task == CONTROL_TASK:
            ctl: dict[str, dict] = {}
            for it in items:
                c = ctl.setdefault(it["category"] or "—", {"n": 0, "mc_wrong": 0, "knew": 0,
                                                            "didnt": 0, "unjoined": 0})
                c["n"] += 1
                if it["mc_right"] is None:
                    c["unjoined"] += 1
                elif not it["mc_right"]:
                    c["mc_wrong"] += 1
                    c["knew" if it["score"] >= CORRECT_AT else "didnt"] += 1
            t["control"] = ctl
        tasks[task] = t
    return {**head, "canary": canary, "preliminary_reasons": reasons, "tasks": tasks}


def merge_judged(model_dir: Path, out: dict, dest: Path | None = None) -> dict:
    """A run narrowed to one topic must not wipe the others' scores out of
    judge.json — but it must not pretend they were graded by this run's
    instrument either. A previous task is carried over only when the judge id,
    the prompt and that task's rubric record are all identical to this run's;
    anything else is named in `judge.replaced` and dropped, because a file
    that mixes two instruments under one heading is worse than a gap."""
    prev_path = (dest or model_dir) / "judge.json"
    if not prev_path.exists() or out.get("skipped"):
        return out
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return out
    if prev.get("skipped") or not isinstance(prev.get("tasks"), dict):
        return out
    pj, nj = prev.get("judge") or {}, out.get("judge") or {}
    same_judge = (pj.get("id") == nj.get("id")
                  and pj.get("prompt_sha256") == nj.get("prompt_sha256")
                  and bool(pj.get("provisional")) == bool(nj.get("provisional")))
    kept, dropped = {}, []
    for task, t in prev["tasks"].items():
        if task in out.get("tasks", {}):
            continue
        prev_rub = (pj.get("rubrics") or {}).get(task)
        if same_judge and prev_rub == _rubric_record(task):
            kept[task] = t
        else:
            dropped.append(task)
    if not kept and not dropped:
        return out
    merged = {**out, "tasks": {**kept, **out.get("tasks", {})}}
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


def start_run(model_dir: Path, results_root: Path, only: list[str] | None = None) -> dict:
    """The service's entry point, called inside the GPU lock: plan, submit
    the batch (seconds), return. The poller finishes it. With JUDGE_MODEL=stub
    the file is written here and now. `only` narrows the run to the tasks the
    submission asked for."""
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
    a = ap.parse_args()
    if not a.results.is_dir():
        print(f"no such directory: {a.results}", file=sys.stderr)
        return 2
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
                bits = [f"{t}={v['score_report'] if v['score_report'] is not None else v['mean']:.2f}/4"
                        f"(n={v['n']})" for t, v in sorted(out["tasks"].items())]
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
