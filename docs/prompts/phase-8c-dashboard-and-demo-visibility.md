# Brief for Claude Code — see the demo, fix the page, take a bank from a person (phase 8c)

Phase 8 P0–P4c are merged. On 2026-09-20 the medicine demo ran start to
finish on the box against vLLM in 103 s: 50 imported questions, sat by
SmolLM2-360M-Instruct, judged criterion by criterion, a spec, 20 documents,
both safety checks passed. Then the dashboard was opened at
`#model=HuggingFaceTB/SmolLM2-360M-Instruct` to look at the judged section —
and it showed the model's **August standard-suite page, with no judged
section at all**. The demo's `judge.json` is under `$BENCH_ROOT/demo/`, the
service reads `$BENCH_ROOT/results/full`, and nothing links the two. That is
correct isolation and a broken demo: "shown greyed on the page" is true of a
code path nobody can open.

This brief fixes that first, then the page problems found on the same visit,
then adds the one thing the exam's author needs next — he is producing
**~100 questions per topic with a criteria file per topic**, and today the
only way in is SSH and a CLI.

**Read first:** `scripts/demo_loop.py` (the summary step and what it
prints), `service/app.py` (`_WATCH`, the payload cache, how the page is
served), `scripts/report_lm_eval.py` (`build_payload`, the judged section,
the P4b criteria row), `scripts/exam_build.py::import_bank` (or whatever P4a
named it), `HANDOFF.md` §13 (the two cache bugs that already happened).

Do P5a as its own PR, then P5b, then P5c.

---

## P5a — the demo is visible, and still not on the leaderboard

Two things, both required.

### 1. The demo ends by building its own report

At the end of a run (with or without `--keep`; the report is small and is
the point), `demo_loop.py` builds the **static** report of the demo tree —
the same `report_lm_eval.py` and the same `build_payload`, pointed at
`$BENCH_ROOT/demo/results/full` and the demo's exam bank — and writes it to
`$BENCH_ROOT/demo/report.html`. It prints the path and the URL below. If
`--keep` is off, the results tree is cleaned as now but `report.html`
stays; say so in the summary.

The report carries a banner at the top, not dismissible: **"DEMO RUN —
<timestamp> — every number here is provisional and stamped demo; nothing on
this page is on the leaderboard."** Same words in the page `<title>`.

### 2. The service serves it, clearly apart

`GET /demo` serves `$BENCH_ROOT/demo/report.html` if it exists, else a
plain page saying no demo has run and the command to run one. Add the file
to `_WATCH` so a new run replaces it without a restart (HANDOFF §13: both
cache bugs were exactly this). The live dashboard header gets one link,
"demo run from <date>", only when the file exists. The demo's own report
links back to the live dashboard. The live payload never reads anything
under `demo/`; add a test that `build_payload` on the live root does not
open a path containing `/demo/`.

Not this: a `?root=demo` switch on the live page, or merging demo results
into the live tree greyed. Two pages, two trees, one link each way.

### 3. Then look at it

Once `/demo` renders, review the judged section against P4b's spec and fix
what is missing — this is the first time a person has been able to see it.
Specifically confirm on the page: the folded score greyed with the
provisional and draft-rubric labels in words; the per-criterion row (zero-
anchored bars, labelled from the criteria file, the conditional one with
its `n`); critical-failure count and share in text next to the score; the
by-acuity table; and one diagnose-half graded item. Take a screenshot into
the PR.

**Tests:** the demo (fake backend, CI) ends with `demo/report.html` present
and containing the DEMO banner and the medicine judged section; `/demo`
serves it and 404s to the explanatory page without it; `_WATCH` includes
it; the live payload opens nothing under `demo/`.

---

## P5b — the page, as found on 2026-09-20

In order of how much they got in the way.

1. **Failure to load is silent and hot.** From a browser that blocked the
   API, the page showed "Loading results…" forever while requesting
   `/api/submissions?limit=100` **27 times in a few seconds**. Add
   exponential backoff (1 s → 30 s cap) on any failed poll, and after the
   second failure replace "Loading results…" with a visible line: what URL
   failed, the status or error, and that it will keep retrying. The same
   backoff for the live poll once loaded: a service restart must not turn
   every open tab into a hammer.

2. **Two "Check:" banners sit above every tab, permanently.** (Chat-template
   applied to some models; 18 of 33 preliminary.) They are right and they
   are in the way on the Exam and Review tabs, where they are irrelevant.
   Show them on Overview and Leaderboard only; elsewhere collapse to one
   line "2 checks on the leaderboard" that expands. Never dismiss them
   permanently — they are findings, not notifications.

3. **The Exam tab cannot take a bank.** It says "Draft more: python3
   scripts/exam_build.py draft …". The person who writes the exam does not
   have a shell on the box. P5c fixes this; here, at least make the Exam
   tab say that a human-written bank is imported with `exam_build.py
   import` and show the command, beside the draft one, so the two paths are
   both visible.

4. **The live bank is empty (0 of 15 topics) and the 40 legacy items were
   never migrated.** Not a page bug — nobody has run `migrate` or `import`
   on the live tree. But the page should say which, plainly: when the bank
   is empty, the Exam tab's summary line reads "no questions yet — run
   `exam_build.py migrate` for the 40 seed items, `import` for a
   human-written bank, or `draft` for LLM candidates". The demo's summary
   step should print the live-import command for the bank it just used, so
   the next step after a demo is one paste.

5. **Duplicate rows at the top of the leaderboard** —
   `qwen35-delta-moe-7d560104-step945` and `-v2`, byte-identical on 8 tasks,
   #1 and #2. Known since HANDOFF §11. Detect identical result sets
   (same task scores and same item counts) and show the second as
   "duplicate of #1" un-ranked, with both names. Do not delete anything.

6. **Timezone.** The header says "generated 2026-09-20 10:54 +03"; the
   team is in Dubai (+04). `TZ` in compose defaults to `Asia/Qatar`. Change
   the default to `Asia/Dubai` and say in `.env.example` that it is the
   display timezone. One line; ask Omar if he wants otherwise.

7. **Model page header.** The line "ranking 7th of 15 ranked models here"
   and the `#7` badge come from an evaluation dated 2026-08-18. Put the
   date beside the rank, not two sentences later: "#7 · evaluated
   2026-08-18". A rank without a date is a claim about now.

8. **Demo wording in the log.** "canary: … None from the previous run"
   when there is no previous run. Say "first run — no previous canary to
   compare". And step 6 says "6 of 21 diagnosis-half answers fell short; 6
   go into the request" then prints 3; print all six or say "3 of 6
   shown".

**Tests:** backoff timing and the visible error line (browser test with
the API stubbed to fail); banners present on Overview, collapsed on Exam;
duplicate detection on the fixture with a copied model dir; the empty-bank
line.

---

## P5c — a person can deliver a bank and a criteria file from the page

Dr. Hossein is producing about 100 questions per topic and a criteria file
per topic. Every topic needs: the questions (JSON array, his shape), a
prose rubric (`rubrics/<slug>.md`) and a criteria file
(`rubrics/<slug>.criteria.json`). Today the questions go in by CLI and the
rubrics by git commit. Give him a page.

### Import on the Exam tab

An "Import a bank" panel on the Exam tab: a JSON file (his array shape;
the same parser as `exam_build.py import`), a topic (drop-down from
`categories.yaml`), the approver name (the existing recorded-name field,
required), an optional source label. **Two steps, not one:** first a
**preview** — items parsed, duplicates against the live bank by qid,
items missing `prompt`, the report/diagnose split the import would
produce, the per-acuity and per-intent counts from `meta` — and only then
a **commit** button. Nothing is written by the preview. The preview shows
prompts **only for items that would land in the diagnose half**; a
report-half prompt is shown as its qid and its metadata line, exactly as
`public_bank` would. (He wrote them; the page still does not echo them
back — the rule is the rule.)

Endpoints: `POST /api/exam/import/preview` and `POST /api/exam/import`,
both taking the file, topic, approver, source. The commit is the same code
path as the CLI, so a bank imported from the page and one imported from a
shell are byte-identical records. Size cap 2 MB; refuse anything that is
not a JSON array of objects.

### Rubric and criteria per topic, from the page

On the same tab, per topic: which rubric file and criteria file the judge
would use right now (`exam.md` fallback or the topic's own), their shas
and versions, and whether the rubric is DRAFT. A **download** of both
files and an **upload** of new ones, with the same preview/commit shape:
the preview validates the criteria JSON (ids unique, weights positive,
fold method known, conditional ones have `applies_when`), shows what
changed against the current file line by line, and warns in words that
committing changes the sha and makes earlier judged runs on this topic
non-comparable. Commit writes the file under `eval_tasks/fr/rubrics/` in
the checkout the service runs from **and records who and when in the
service database** — this is the one place the page writes to the repo
tree, so log it. Do not commit to git from the service.

If writing to the checkout is refused on the box (the container's
`/app` is the image, not the bind-mounted checkout), fall back to
`$BENCH_ROOT/rubrics/` and have `judge.rubric_for` look there first, then
in the repo. Say which one is live on the page.

### Tests

Preview with a good file, a duplicate, a missing-prompt item, a non-array;
commit produces the same records as the CLI on the same input (compare
the bank files); report-half prompts absent from the preview response
body; criteria upload validation for each rule; sha shown changes after
an upload; the database row for a rubric change; the request-body tests
still pass with a page-imported bank.

---

## Definition of done

CI green. On the box: the medicine demo ends with a printed `/demo` URL;
that page shows the judged section with criteria, critical failures and
acuity; the live leaderboard is unchanged and still lists no demo model. On
the Exam tab, a 100-question JSON and a criteria file for one topic go in
through the page with a preview first, under a typed name, and the judge
uses them on the next `suite=judged` run.
