# Brief for Claude Code — Improve as one pipeline (12g)

Start **after 12h.2**. masein is away and may not have merged 12h yet:
- If 12h.2 is merged, branch from main.
- If it isn't, branch from 12h.2's branch and open the PR against it. Say at
  the top of the PR that it's stacked on 12h.2.

Don't change 12h's files beyond what this brief needs.

**No model runs on the Mac**, as in 12h: test with fixtures and fake models.
Real runs happen on the server after deploy.

Two PRs:

- **12g.1** — Improve becomes one pipeline for one model, fed by the
  Knowledge exam, with the Standard benchmarks as a before → after **watch**.
  Plus two small fixes from the last live check.
- **12g.2** — Everyday tasks join the pipeline: a practice/hidden split, a
  group opens only when it has enough questions, and training data made as
  checked chat examples.

Each is done when `scripts/check.sh` passes on its branch head, its summary
line is in the PR under "Local check", and deploy step 4 passes on the server.

**The rules that do not bend are unchanged**, and so are the display rules:
one number first; nothing empty; no system words; a caveat once per page;
one main action; numbers in mono, words in sans.

---

## The decision behind this brief

masein decided on 2026-09-25:

- **Knowledge exam and Everyday tasks are what Improve trains toward.**
- **Standard benchmarks are never a training target.** That includes 12h's
  IFEval, MMLU-Pro and MATH-500. They are the outside check. Our own exam and everyday questions are what the loop learns from;
  the public benchmarks are the one scoreboard it never aims at, so they can
  tell us whether training made a model better or only better at our own
  questions. In Improve they appear only as a **watch**: before → after for
  each retrained checkpoint, with a warning when a score drops.

Nothing in Improve may put a Standard benchmark into a proposal, a generator
request or a training dataset. Add a test that asserts it over the recorded
request bodies, the same way the split is asserted.

---

# 12g.1 — the pipeline

## 1. One model, four stages

Improve today is three tabs moved in by 12b: By topic (the old Loop), Review
and Training runs. Replace **By topic** and **Review** with one view:

```
Improving: SmolLM2-360M-Instruct ▾                         [ Propose ]

Weak spots (3)      Proposals (1)       Training data (2)    Retests (1)
Economics 0.79/4    Sociology           Arts · 20 of 20      Arts 1.2 → 1.7
  Propose             waiting for you     Use in training      Standard: no drop
Arts 1.21/4           Review            Economics · 20 of 20
  Propose                                 Use in training
```

- **A model picker at the top.** Everything below is for that model, so no
  list needs a model column. It remembers the viewer's last model, and opens
  on the model with the most judged topics the first time.
- **Four stages, left to right**, each a count and a short list of items,
  each item one line with one action. Five items per stage, then **+ n more**
  opening the full list in place.
- The stages read from what exists today:
  - **Weak spots** — the model's judged topics, weakest first, that have no
    open proposal. Action: **Propose** (the existing New proposal dialog,
    filled in).
  - **Proposals** — To review and Ready to generate from today's Review.
    Actions: **Review** / **Generate**.
  - **Training data** — datasets made for this model. Action: **Use in
    training** (copies the `--gap-dataset N` line, as today).
  - **Retests** — checkpoints trained on this model's datasets, with the
    topic score before → after. Action: **Compare**.
- A stage with nothing in it says one line — "No proposals waiting" — and
  doesn't draw an empty box.
- The one filled button is **Propose**, top right. It opens the dialog on the
  weakest topic.
- **Training runs** stays as Improve's second tab. It's the team's training
  telemetry, not one model's pipeline.

Every old address keeps working: `#tab=improve&sub=topics` and
`&sub=review` (and its views) land on the pipeline with the same model;
`#tab=loop` and `#tab=review` still land here through 12b's redirects.

## 2. Retests need to know which model a checkpoint came from

The Retests stage pairs a checkpoint with the model it was trained from.
First find out what the board records today — the taint marks, the
`bench.init()` training runs, the `local/` artifact metadata — and **say in
the PR what links a checkpoint to its base model now**.

If nothing does reliably, add one field, **Trained from**, on the checkpoint's
model page: set once by a person, from a picker of models on the board, and
filled in automatically when a training run records its base. Retests uses
it. A checkpoint with no Trained from doesn't appear in Retests, and its
model page says, in one line, "Set what this was trained from to see it in
Improve".

## 3. The Standard watch

In each Retests item, under the topic before → after, one line:

- **Standard 30.5 → 30.9 · no drop**, or
- **Standard 30.5 → 28.1 · dropped** in the warning colour, with a link to
  the model page's Standard block.

"Dropped" means a difference the board's existing z-test calls real. A
difference inside the noise reads **no drop**. When the checkpoint has no
Standard result yet, the line reads **Standard: not tested · Test**.

**Which Standard number:**
- The average over the benchmarks **both** the base and the checkpoint were
  tested on, computed the way 12h.2's "Avg of N" is.
- The line names N: "Standard (6) 30.5 → 30.9".
- A single benchmark that dropped by a real margin makes the line "dropped",
  even when the average held. Say which one: "dropped · MMLU-Pro 41.2 → 36.0".
- Instruct models include 12h's three benchmarks when both sides have them.

This line is the only place Standard appears in Improve.

## 4. Two fixes from the 12b.3 live check

- **The checks panel doesn't close.** The status dot opens a `<details>`
  panel that stays open through page changes, Escape and clicks outside it.
  Use the 11a popover component, as the run counter does: it closes on
  Escape, on a click outside and on a page change.
- **Home's Knowledge exam card** shows the weakest topic *across the board*,
  which today is pythia-31m's 0 / 4 — a 31M model that scores zero
  everywhere, so the card says nothing. It shows the weakest topic **of the
  model with the most judged topics** instead, and names that model:
  "SmolLM2-360M-Instruct · weakest: Economics 0.79 / 4". Its link opens
  Improve on that model.

## 12g.1 tests

- The pipeline renders the four stages for a fixture model; a stage with no
  items shows its one line, not a box.
- Old Improve and Review addresses land on the pipeline with the model kept.
- A Retests item shows the Standard line, and reads "dropped" only when the
  z-test says so.
- Nothing from a Standard benchmark appears in any proposal, generator
  request or dataset — asserted over the recorded request bodies.
- The checks popover closes on Escape, an outside click and a page change.

---

# 12g.2 — Everyday tasks in Improve

## 5. The split comes to the Everyday bank

Until now every Everyday question has been readable, and the bank never
reached Propose or a generator. To train on Everyday tasks, each group needs a
**hidden half** that scores the model and a **practice half** the loop may
read — the same rule as the exam.

- Every Everyday question gets a `qid` and a half by the **same function and
  salt** as the exam (`diagnose.split_of`). Nothing is assigned by hand.
- The **hidden half is never shown** — not on the questions list, not in
  answers, not in the Playground later — exactly as for the exam. The
  questions list and the results show practice questions only, and say how
  many hidden ones there are.
- The published Everyday result per group becomes the **hidden-half** score.
  The badge stays as 12a.4 left it, *not ranked · provisional judge*, until
  the judge is calibrated.
- **The split changes the bank version** (12a.4 § 4). The split is part of
  what a score means.
  - Results from before the split go under History, as 12a.4 does for
    "earlier wording", labelled **"all questions, before the split"**.
  - They're never mixed with hidden-half scores.
- **Say the real hidden count per group in the PR.** Honesty has 41
  questions, so about 20 hidden, right at the bar. If any group lands below
  20, say which. masein will add questions.

## 6. A group opens only when it's big enough

A group joins Improve only when its **hidden half has at least 20
questions**. Below that, a score from so few items is noise, and training
toward it would chase noise.

- In the pipeline, Everyday groups appear in **Weak spots** beside the exam
  topics, marked **Everyday**.
- A group below the line appears once, greyed, as one line: "**Honesty ·
  needs 2 more hidden questions** to improve on." No Propose.
- Make the threshold one setting.

## 7. Everyday training data is checked chat examples

Exam gaps are filled with prose documents. Everyday gaps need something
else: examples of a request and a good assistant reply.

- A new dataset format, **`chat`**: each item is one user message and one
  assistant reply, plus the checks the reply must pass — the same check
  vocabulary as the bank.
- **Every generated example is marked by its own checks before it is kept**,
  with 12a.4's checker (`admits_limit`, `any`, …).
  A reply that fails its checks is dropped and counted as missing, with the
  reason, as the `doc` format does for its gate. So the training data can't
  contain the very mistake it's meant to fix.
- The 13-gram contamination gate runs against **the whole Everyday bank,
  both halves**, and the exam, as for documents.
- The proposal's focus comes from the group's **failed practice questions**,
  one example batch per failed skill, spread as 11e spreads documents over
  concepts.
- The generator is told the group, the failed skills and the practice
  questions it failed, **never a hidden question**.
- `chat` is offered only for Everyday groups; `doc` only for exam topics.
  `free` stays as #60 left it.

## 8. What masein sees

- **Weak spots** mixes exam topics and Everyday groups, weakest first within
  each, with a small label on each item: *Exam* or *Everyday*.
- **Retests** shows the group before → after on the hidden half ("Instructions
  6 of 22 → 13 of 22"), then the Standard watch line. Both runs must be on
  the same bank version. If they aren't, the line says "retest on the current
  questions · Test", not a before → after.
- The model page's Improve tab shows the same, for that model.

## 12g.2 tests

- The split assigns every Everyday question by `split_of`, and no hidden
  question text reaches the page, a proposal or a generator request —
  asserted over the recorded request bodies.
- A group with 19 hidden questions shows "needs … more" and no Propose; with
  20 it offers Propose.
- A `chat` example whose reply fails its checks is dropped with its reason;
  one that passes is kept.
- The contamination gate drops an example that copies 13 words from any
  Everyday question, hidden or practice.
- The old all-questions results stay visible and labelled, and aren't mixed
  with hidden-half scores.
- A retest across two bank versions shows "retest on the current questions",
  not a before → after.
- The Standard watch averages only the benchmarks both sides have, and a
  single real drop reads "dropped · <benchmark>".

---

## Done when

On the live server:

1. **Improve** opens on one model with four stages, and the Standard watch
   line under each retest.
2. No Standard benchmark reaches a proposal, request or dataset.
3. The **checks popover** closes on Escape, a click outside and a page change.
4. **Home's Knowledge exam card** names the most-judged model's weakest topic.
5. **Everyday groups** appear in Weak spots, each saying how many more
   questions it needs, until it has 20 hidden.
6. A `chat` dataset keeps only replies that pass their own checks.
7. Each PR has its Local check line, and deploy step 4 passes on the server.

## Deploy steps, for masein, after each merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests and
the task-discovery check inside the container.
