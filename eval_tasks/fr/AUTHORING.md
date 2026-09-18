# The exam — how it is written, curated and split

The exam is the instrument. One question bank across the **topics in
`scripts/categories.yaml`** — economics, law, medicine & health, mathematics,
computer science, physics & engineering, chemistry & biology, history,
philosophy & religion, politics & government, psychology & sociology, business
& accounting, geography & world facts, language & logic, other — because
"which topic are we weak in" is the question the loop exists to answer. Each
checkpoint sits the whole exam; a judge grades every answer 0–4 against
`rubrics/exam.md`; the per-topic score is what picks the next thing to train.

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
in this directory, ten items each) are migrated into the bank under `other`,
as they are, with their skill on the record. Nothing was thrown away; they
are simply not topics.

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
