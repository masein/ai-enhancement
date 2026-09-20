# Brief for Claude Code — the loop as a page you can drive (phase 8e)

Do this **after** 8c (demo visibility, load errors, banners, bank import
from the page) and 8d (law, flags list, breakdowns). Do not redo anything
those briefs cover; build on it.

## Why

The whole loop — sit the exam, judge, choose the weak topic, propose,
approve, generate — runs, but on the page it is scattered across five tabs
with no thread between them, the person has to know which tab is next,
and two steps are not on the page at all. Omar's request, verbatim: *"do I
have a button in the dashboard to press and say do the evaluation on a
specific topic, then see the answers, then be able to generate the dataset
with that in the dashboard?"* Today the honest answer is: sit (whole exam
only, and the `judged` suite is not even offered when the bank is empty),
judge (automatic), see the answers (no), propose/approve/generate (yes, on
Review). This brief makes the answer yes end to end, and makes the rest of
the page navigable.

Walked on 2026-09-20 with 33 models on the board. Findings that shaped
this brief are in §4.

**Read first:** `scripts/report_lm_eval.py` (`navigate()`, hash routing,
the tab registry, `el()`), `service/app.py` (submissions API, the
`suite` values, `judged_blocked`), `service/worker.py` (how a judged job
becomes a judge batch), `scripts/judge.py` (`judge.json` item shape after
P4b/8d), `service/proposals.py` (propose → approve → generate).

Three PRs: P6a the Loop tab, P6b the Models section, P6c navigation. Each
green on its own.

---

## P6a — a Loop tab: one place, one topic, next button always visible

### The topic board

A new tab **Loop**, first after Overview. It opens on a table, one row per
topic in `categories.yaml`:

| topic | bank (report / diagnose) | rubric | last judged model → score | open proposal | datasets | **next step** |

- **bank** — counts, and "under 30 report-half" in words when it is.
- **rubric** — `exam.md` fallback / own rubric, DRAFT or signed, criteria
  file yes/no. Links to the 8c rubric panel.
- **last judged** — the most recent `judge.json` that contains this topic:
  model, folded score, provisional/single-provider/draft badges in words.
- **next step** — a single button, computed: *Import a bank* (bank empty)
  → *Sit the exam* (bank ≥ 1, no judged run) → *Read the results* (judged,
  no proposal) → *Propose* (results read, gate clear) → *Review the spec*
  (proposal pending) → *Generate* (approved) → *Hand to training* (dataset
  ready, shows the id and the `--gap-dataset` line). When the gate blocks
  (provisional judge, under 30, κ not yet measured), the button is
  disabled and the reason is beside it in words. This is the same
  `judged_blocked` / propose-gate logic the API already enforces; the page
  reads it, never re-implements it.

Clicking a row opens the **topic page** (`#topic=<slug>`): the same
information in full, plus the step panels below, in loop order.

### Sit the exam on one topic

`POST /api/submissions` gains `tasks: ["exam_law"]` (a list of built exam
task names) alongside `suite: "judged"`. The worker runs only those tasks
and the judge batch covers only those answers. `suite=judged` with no
`tasks` keeps meaning "every built exam task". The topic page's *Sit the
exam* panel: model id (Hub or `local/`), kind, the recorded name, and the
one topic pre-selected with the others as checkboxes. Submit & Queue's
suite drop-down gains `judged` and, when the exam is not built, shows it
**disabled with the reason** rather than omitting it — today it is simply
absent and nobody can tell why.

The queue row for a judged job says which topics, and the judge batch's
progress ("judging 40/80") shows on the row, from the batch state the
poller already tracks.

### See the answers

On the topic page and on the model page's judged section: an **Answers**
panel for the diagnose half. One row per item: acuity (and difficulty
where present), the question, the model's answer (collapsed to two lines,
expand), the folded score, the flags raised, the per-criterion scores as
a compact strip (bars, not numbers, with numbers on hover), the judge's
justification. Filter by acuity, by flag raised, by score, by criterion
below a threshold; sort by score. Report-half items appear **only** as the
aggregate line above the table ("29 report-half items: mean 2.52, 3
critical failures") — never as rows, never in the DOM. This is the rule;
add a browser test that the report-half prompt from the fixture is
absent from the served HTML and from every API response the page makes.

The model's *answer* to a report-half question is also not shown — an
answer quotes the question often enough to leak it.

### Propose, approve, generate — on the topic page

Move nothing; **link** the existing Review-tab panels from the topic page
and pre-filter them to the topic. The Review tab stays as the cross-topic
queue for the person whose job is approving. The topic page's next-step
button opens the right panel with the topic filled in. Generation's
result — dataset id, kept/dropped, the `--gap-dataset` line, provenance
link, one document preview — shows on the topic page as the last panel.

### Tests

Board computes the right next step for each state (fixture per state);
disabled reasons match `judged_blocked`; `tasks` narrows the job and the
judge batch; Answers table renders diagnose rows only and the report-half
leak test; the topic page routes and deep-links.

---

## P6b — a Models section you can browse

Today the only ways to a model page are the Top-5 on Overview, the
leaderboard row, or knowing the hash. 33 models, 19 of them checkpoints,
and no list.

- **Models** tab (replaces the two filter rows now floating above the
  tabs — "All/Base/Instruct" and "All/Models/Checkpoints" — which apply
  to something but the page does not say what). A table: name, kind,
  family, params, official avg or "preliminary n/7", judged topics and
  scores (badged), taint badge, last evaluated. Search box, the two
  existing filters plus family and "has judged run" and "tainted", sort
  on every column, and a "compare" tick that feeds the radar the
  Leaderboard tab already has (up to 5, same slots).
- **Model page**: a sticky sub-nav — Results · Diagnose · Judged ·
  Answers · Provenance · Runs — because the page is long and the judged
  section is now the interesting part. The rank badge carries its date
  (8c item 7). A **history** strip: every submission of this model in
  the queue, newest first, with suite and outcome, so "why is this
  preliminary" is answered on the page.
- A checkpoint's page links its training run on the Training tab and
  back; a model trained on a generated dataset (taint) links the dataset
  and the proposal it came from — the loop's audit trail, one click each.

Tests: filters and sort on the fixture; compare feeds the radar;
sub-nav anchors; taint → dataset → proposal links resolve.

---

## P6c — navigation and naming

Small, all found on the walk:

1. **Deep links.** `#tab=submit` lands on Overview; the real hash is
   `#tab=queue`. The tab labelled **Evals** routes to `#tab=runs` and its
   content is "Run provenance"; the footer says "provenance is in the
   Runs tab". Pick one name per tab, make the hash equal the label's
   slug, accept the old hashes as aliases, and list the hashes in
   `SERVICE.md`.
2. **The "Check:" banners appear three times** — twice in the header and
   twice again as "Warnings" on the Evals tab. 8c collapses them off the
   loop tabs; also drop the duplicate on Evals.
3. **Leaderboard tab leads with the radar**, table second. Table first;
   the radar under it, opened by the compare ticks.
4. **Tab order** by how often each is opened: Overview · Loop · Models ·
   Leaderboard · Submit & Queue · Exam · Review · Training · Tasks ·
   Perplexity · Runs. Exam and Review are steps of the loop and should
   read as such (the Loop tab links to them; they keep their hashes).
5. **Header** — "generated 10:54 +03" reads like a stale static page on a
   live one. "live · refreshed 10:54" and the timezone from 8c.
6. **Theme button** says "Theme: dim" — a state, not an action. "Theme ▾"
   with the three options.
7. The "guide for new users" link should open on the Loop tab's
   explanation for someone whose job is the loop, not only the
   submitter's guide — one more section in FRIENDS.md, linked from Loop.

Tests: every tab reachable by its documented hash and its alias; banner
count per tab; tab order.

---

## §4 — what was found on 2026-09-20, for the record

- Model page for the demo model showed August's standard suite, no judged
  section (8c P5a covers it).
- Submit & Queue: suite options full / quick / control — no `judged`,
  no explanation.
- Leaderboard: #1 and #2 are byte-identical rows (`…step945` failed on
  `ppl_code`, `-v2` is its resubmission); the failed one is ranked #1 and
  shows no perplexity. 8c item 5.
- Exam tab: bank 0 of 15 topics; the only instruction is a shell
  command. 8c P5c.
- Review tab: correct empty states, but "Pick a topic" is the loop's
  most important control and it lives on the fourth tab.
- Evals tab: labelled Evals, hash `runs`, content "Run provenance",
  warnings repeated.
- `#tab=submit` deep link fails silently.
- Two filter rows above the tabs with no stated scope.
- Footer text on every tab; fine.

## Definition of done

From a fresh browser on the tailnet, with the law bank imported (8d): open
Loop → law row says *Sit the exam* → submit SmolLM2-360M-Instruct on law
only → queue shows the job and then "judging n/100" → the row flips to
*Read the results* → the Answers panel lists the diagnose-half items with
criteria strips and flags, filterable by acuity → *Propose* is disabled
with "judge is provisional" in words (local judge) — and enabled on a
fixture where it is not → Review → Generate → dataset id and the
`--gap-dataset` line on the topic page. Models tab lists 33 models,
filters to 3 instruct, opens one, sub-nav jumps to Judged.
