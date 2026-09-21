# The exam — how it is written, curated and split

The exam is the instrument. One question bank across the **37 topics in
`scripts/categories.yaml`** — Agriculture, AI & Machine Learning, … Technology,
the folder names of the phase-10 delivery exactly as written — because "which
topic are we weak in" is the question the loop exists to answer. Each
checkpoint sits the whole exam; a judge grades every answer 0–4 against the
topic's own rubric (`rubrics/<slug>.md`, or `rubrics/exam.md` for a topic
without one); the per-topic score is what picks the next thing to train.

36 of the topics arrived with 100 questions each (`banks/<slug>_v1.json`),
written by masein, and a criteria file and rubric each. Arts arrived empty.
The five topics the exam had before — medicine & health, law, economics,
computer science, physics & engineering — are **retired**: their files are in
`retired/`, their rows stay in the bank marked `retired_at`, and their judged
runs are history.

## Who writes it

An LLM drafts, a person decides. `scripts/exam_build.py draft` asks the exam
writer (`EXAM_PROVIDER`/`EXAM_MODEL`, a different identity from the judge and
the generator) for candidates per topic; each request carries the topic
brief, the rubric, and a few accepted questions as examples so drafts land
on-target and vary in form. Candidates go to `$BENCH_ROOT/exam/candidates/`,
**never straight into the live bank**. On the dashboard's **Exam tab** a
person reads each candidate against the rubric and accepts it, edits and
accepts it, or rejects it with a reason — under a typed name, recorded on the
question and in the service database. Target about 60 accepted questions per
topic. Do not accept in bulk without reading: a bank of unread LLM questions
is a benchmark of nothing.

The four original skill suites (`fr_instruction_following.jsonl` and friends
in this directory, ten items each) were migrated into the bank under `other`,
as they are, with their skill on the record. `other` is not a topic any more
— General & Multidisciplinary took its place as the fallback — so those rows
are kept but no longer read; `migrate` on a fresh bank puts them under
General & Multidisciplinary, and never twice.

## The split — the most important line in this file

Every accepted question gets a `qid`, the sha256 of its normalised text, and
`scripts/diagnose.py::split_of(qid)` — the same function and salt as every
benchmark — puts it in the **report half** or the **diagnose half**. The
published per-topic score comes from the report half. The step that chooses
what to train may read only the diagnose half. A report-half question is
never shown on the dashboard, never exported, never placed in an LLM request
(`exam_build.public_bank`, `diagnose_half_examples`, and the tests over the
recorded request bodies enforce it). The exam is our own generated questions,
so nothing external protects it; without the split the loop would train on
its own test with no outside benchmark to catch it. The curator sees a
candidate before it is hashed — that is unavoidable, someone writes the exam —
and does not see it again if it lands in the report half.

## What a good question looks like

- **Understanding, not recall.** Ask for a mechanism, a distinction, an
  application, a consequence. "Why does X tend to Y, and when would it not?"
  beats "What is X?".
- **One ask**, answerable in two to five sentences by a well-read person.
- **A reference that names the substance**, not model wording; the judge must
  recognise a correct answer phrased differently.
- **Length-neutral.** The rubric puts length in writing; a complete short
  answer scores fully. Do not write prompts that reward volume.
- **Bounded ground truth.** Say whose law, which era, which regime.
- **New.** Not lifted from a known set; the 13-gram contamination gate that
  guards generated datasets guards the exam too.

## Importing a human-written bank

A bank somebody wrote outside this repo goes in whole, without passing
through drafting or curation — those steps exist to establish that a person
read each question, and here the author already has:

```bash
python3 scripts/exam_build.py --root $BENCH_ROOT/exam import \
    eval_tasks/fr/banks/medicine_clinical_health_v1.json \
    --topic "Medicine & Clinical Health" --approver masein
```

A folder of them — every `<slug>_v<n>.json` into the topic with that slug,
source the file's stem — is one command, and prints a line per topic and a
total. A file whose slug is no topic's is refused by name, before anything is
written:

```bash
python3 scripts/exam_build.py --root $BENCH_ROOT/exam import-dir eval_tasks/fr/banks --approver masein
```

The file is a JSON array of objects with at least a `prompt` — or an object
holding one, such as `{"questions": [...]}`. `--approver` is **who wrote the
questions**, recorded on every item the way a curator's name is;
`--imported-by` is whoever ran the import, when that is somebody else.
`--source` tags which file they came from and defaults to the file's own
name; it is never the topic ("economics" in the source column of the
economics bank says nothing about where those questions came from, and the
import refuses it).

A bank already written cannot be corrected by re-importing it — a matching
qid is skipped, which is exactly what the qid is for — so there is a command
for that:

```bash
python3 scripts/exam_build.py --root $BENCH_ROOT/exam set-source     --topic Economics --source economics_v1 --approver masein
```

It rewrites `source` and `accepted_by` in place and touches nothing else: the
qid is the question's identity *and* its half, so moving one would re-roll
the split and quietly change what every judged score of that topic is about.

## Retiring a topic

```bash
python3 scripts/exam_build.py --root $BENCH_ROOT/exam retire --topic law --reason "replaced by the 37-topic exam (2026-09-21)"
```

marks every row of that topic `retired_at` / `retired_reason`. Nothing is
deleted or re-split. A retired row is left out of the built tasks, the Loop
board, the counts and imports — a retired question may be imported again
under a new topic, where it is new. **The topic is matched by the name stored
on each row, exactly**: the retired `law` and the new `Law` share
`bank/law.jsonl`, and a slug match would retire both. A name no row carries is
refused, with the names the bank does hold.

## Re-importing

The import is idempotent on `qid`, so
re-running it after the author sends more items adds only the new ones, and
prints what it imported, what it skipped and the report/diagnose split.

**Metadata is the reference.** An item usually carries no model answer.
Everything on it besides `prompt`, `reference` and `notes` is kept under
`meta` and a reference line is built from it in a fixed field order:

```
Acuity: emergency. Intent: symptom_assessment_triage. Domain: cardiovascular.
Difficulty: 3. Subject: self (male, 45-59). Style: telegraphic.
```

A boolean field is said in words — `Jurisdiction required: yes.` — because
the criterion that reads it is written about the question, not about a JSON
value.

**Re-importing revises, it does not duplicate.** A question whose prompt is
already in the bank keeps its qid and its half; if the file gives it
different metadata, that record is *updated* (new `meta`, new reference, the
new `source` and `accepted_at`, the original `accepted_by`) and the import
prints how many it updated. Identical metadata is still skipped. The prompt
is the identity; everything else is the author's to revise.

That line IS the ground truth the rubric asks the judge to check against —
above all the acuity. The order is fixed rather than the file's, because the
reference is part of what the item is and the same item must always read the
same way. An item that has its own `reference` keeps it, with the metadata
line appended.

Everything else is unchanged: the `qid` is the hash of the normalised prompt,
the split is the same function with the same salt, and a report-half imported
question is as withheld as any other — never shown, never exported, never in
a request except the judge's.

## A rubric per topic, and criteria files

`rubrics/<slug>.md` grades the topic whose task name carries that slug
(`Medicine & Clinical Health` → `exam_medicine_clinical_health` →
`rubrics/medicine_clinical_health.md`);
a topic without one is graded by `rubrics/exam.md`. The control set keeps
`factual_accuracy.md`. Which rubric graded a task, with its sha and version,
is recorded per task in every `judge.json`.

A rubric heading that still says **DRAFT** is recorded as such and shown on
the page beside the score: a rubric its author has not signed off grades, but
it does not settle anything. Sign-off is deleting the word — which changes
the sha, which is correct.

**A criteria file** beside it, `rubrics/<slug>.criteria.json`, changes how the
topic is graded. The shape below is what the platform reads; an author's file
may be written differently and the loader normalises it —
`docs/CRITERIA-SCHEMA.md` lists every variant that has arrived and what each
maps onto, and the rule is that the loader learns a new one rather than the
file being rewritten. the judge scores each criterion 0–1 and answers true or
false for each flag, and the 0–4 the rest of the system reads is folded from
those numbers **in code**, never asked of the model. The schema is the
author's:

```json
{"topic": "law",
 "weights": "equal",
 "criteria": [{"id": "relevance", "definition": "…", "weight": 1.0, "conditional": false}],
 "flags": [{"id": "critical_legal_error", "condition": "…", "effect": "zero_score"},
           {"id": "fabricated_authority", "condition": "…", "effect": "cap_at_1_of_4"}]}
```

- `topic` is informational. The **file's name** decides which task it grades.
- `weights: "equal"` or no key at all means every weight is 1; a `weight` on
  a criterion is honoured when it carries one.
- `criteria`: `id` (lower-case, exactly what the model must repeat) and
  `definition`. `label` is optional — without one the id becomes the words
  the page shows. `conditional: true` lets the judge answer `null`, and that
  criterion is then excluded from the fold.
- `flags`: `id`, `condition`, and an `effect` the judge knows how to apply —
  `zero_score` or `cap_at_N_of_4`. Effects are applied after the fold, in
  file order. An unknown effect **refuses to load**: a file whose rule cannot
  be applied must not quietly grade as if the flag did nothing.
- A flag id may repeat a criterion id (law scores `fabricated_authority` and
  flags it). They are two marks under two headings in the prompt.
- `breakdowns` is optional: the metadata fields to tabulate the topic by.
  Without it, whichever of `acuity`, `difficulty`, `jurisdiction_required`
  and `intent` the topic's items actually carry.
- `audience` is optional, and is one sentence about the **register** the
  training documents for this topic should be written in. It replaces the
  default ("guidance a layperson can read and act on … not clinical notes,
  case files, legal memoranda or textbook exposition"). The audience
  *counts* beside it are always the bank's own — styles, subjects and
  intents as labels and percentages — and travel with both the proposal and
  the generation request, because a generator that is told only the skill
  writes for whoever it imagines.

The 0–4 is `round_half_up(4 × Σ w·c / Σ w)` over the applicable criteria,
then each true flag's effect. The prompt sent to the judge is generated from
this file, so it cannot ask for criteria the fold does not know about. Both
the rubric's sha and the criteria file's sha ride in `judge.json`: **change
either and scores before and after are not comparable**, exactly as for the
prose rubric alone.

## The control set

`fr_control_mmlu` is not authored. `exam_build.py build` builds it from the
diagnosis half of MMLU on disk: the question only, no options, graded
against the gold option's text, about ten per topic. It answers one question
per model: of the items it got wrong as multiple choice, how many did it get
right when asked openly — "knew it, couldn't pick it" versus "didn't know it
either way".

## Calibration before anything counts

Nothing judged enters any average until a person has graded a sample and
Cohen's κ against the judge clears 0.60 (`scripts/judge_calibrate.py`). Below
that the suite is shown as preliminary and never ranked. Plan the calibration
pass alongside each round of curation.
