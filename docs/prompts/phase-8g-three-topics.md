# Brief for Claude Code — three more topics, the author's third schema, and the loop's first-use findings (phase 8g)

Do after 879b88e. One PR for A–C (topics), one for D (loop UX). D can
start first if A–C are waiting on anything.

## A — Three topics, verbatim

Dr. Hossein delivered computer science, economics and physics &
engineering: 100 questions each, a criteria file each, and — new — a
prose rubric each **with his own 0–4 anchors**. Files in this commit,
all byte-for-byte as delivered:

| File | Note |
|---|---|
| `eval_tasks/fr/computer_science_v1.json` | bare array |
| `eval_tasks/fr/economics_v1.json` | bare array |
| `eval_tasks/fr/physics_engineering_v1.json` | **wrapped**: `{"questions": [...]}` |
| `eval_tasks/fr/rubrics/computer_science.criteria.json`, `economics.criteria.json`, `physics_engineering.criteria.json` | his third schema, below |
| `eval_tasks/fr/rubrics/computer_science.md`, `economics.md`, `physics_engineering.md` | his prose rubrics — score anchors, principles, critical-error examples. **Not DRAFT**: the author wrote them. |

Slugs must resolve from `categories.yaml` topics `computer science`,
`economics`, `physics & engineering` to these file names — check
`topic_task` agrees; rename the files only if the slug function says
otherwise, and say so.

## B — The loader accepts his third schema; one canonical schema going forward

Observed shapes, all three files:

```json
{"criteria": [{"id": "relevance", "name": "Relevance", "definition": "...", "weight": 0.05, "conditional": false}, ...],
 "critical_flag":       {"id": "critical_technical_error", "condition": "...", "effect": "score=0",
                         "examples": [...], "not_critical": [...]},          // CS, physics
 "critical_error_flag": {"id": "...", ..., "do_not_classify_as_critical": [...]},   // economics
 "evaluation_principles": ["...", ...]}                                        // physics only
```

and physics has **numeric `id`s with the slug in `name`** (`{"id": 1,
"name": "relevance"}`) where CS and economics have the slug in `id` and a
label in `name`.

Loader rules, added to the ones from 8d (his second schema stays
accepted — law and medicine are on it):

- Criterion id: `id` if it is a non-numeric string; else `name`. Label:
  `name` when it is not the id, else derived from the id as now.
- Flags: `flags` (list) **or** `critical_flag` **or** `critical_error_flag`
  (single object) → one internal list. `effect` accepts `zero_score`,
  `cap_at_N_of_4`, `score=N`, `cap=N`; anything else refuses to load,
  naming the file and the string.
- `examples`, `not_critical` / `do_not_classify_as_critical` on a flag,
  and top-level `evaluation_principles`, go **into the judge prompt**
  under the flag's definition ("counts as critical: … / does not count:
  …") and above the criteria ("principles: …"). They are the author's
  calibration of the judge and they materially change what "critical"
  means; they belong in the request and their text is part of the sha
  (it already is — whole-file sha).
- Weights: numeric per criterion as before; `0.05 × 20` is just equal.
- Conditional criteria: 10–13 of 20 in each of these topics. The
  prompt must say for each conditional criterion "return null if this
  does not apply to the question", as P4b specified, and the page's
  per-criterion row must show `n` applicable beside each mean (it does
  for medicine's conditional; confirm it renders when half the row is
  conditional).

Then **write `docs/CRITERIA-SCHEMA.md`** — for us, not for the author.
Decision (Omar, 2026-09-20): we will not ask him to change how he writes
files; whatever shape arrives, the loader reads it. The document
describes the internal normalised form (`criteria[]` with `id` slug,
`name`, `definition`, `weight`, `conditional`, optional `applies_when`;
`flags[]` with `id`, `condition`, `effect`, `examples[]`,
`not_critical[]`; optional `evaluation_principles[]`), lists every
input variant seen so far and how each maps onto it, and says that a
new variant is handled by extending the loader, never by editing his
file. Two paragraphs and a table. Add a test that loads all five real
criteria files and asserts they normalise to the same internal shape.

## C — Import and breakdowns

- `import` accepts a bare array **or** an object with a single list
  value (`questions`, `items`, …); the preview says which it found.
- Breakdown tables are chosen **by cardinality, not by name**: from the
  topic's `meta`, every field with 2–12 distinct values, in a fixed
  preference order when present (`acuity`, `difficulty`, `domain`,
  `style`, `intent`, `jurisdiction_required`, then any other), capped at
  four tables. A constant field (these three have `acuity: routine` ×
  100 and `jurisdiction_required: false` × 100) is skipped and named in
  one line ("acuity is routine on every item — no table"). A free-text
  field (CS and economics `intent`: ~100 distinct) is skipped silently.
  Medicine and law keep exactly the tables they have now (acuity,
  difficulty, intent; law also jurisdiction_required) — test that this
  rule reproduces them.
- The reference line built from metadata includes `Domain:` and
  `Style:` (it already does) — for these topics, `Intent:` is a sentence
  on CS and economics; include it anyway, it is the author's statement
  of what the question tests and the judge should see it.
- `DEMO.md` gains the three runs. All three clear `PROPOSE_MIN_N`.

Tests: loader on all three real files (numeric ids → slugs; flag
object → list; `score=0`; principles and examples present in the
recorded judge request body); wrapped import; breakdown selection on
each real bank and on medicine/law unchanged; the reference line with a
sentence-valued intent; the recorded-request-body leak tests still pass
with five banks.

## D — The Loop tab, after its first live use (2026-09-20)

Sitting law on one topic from the page, watching the queue, reading the
answers and hitting the disabled Propose all worked. Four things did
not, plus the duplicate pair:

1. **The board re-renders on every poll and detaches its elements.** A
   reference taken to the name input was gone before it could be
   clicked. A person typing during the 5-second poll loses focus. Same
   class as the tab-bar bug fixed in #21: build once, update text and
   attributes in place; never replace an input the user may be in.
   Browser test: focus the name field, wait 6 s, type — the text lands.
2. **Queue row says "judging 130 answers", not "judging 40/130".** The
   poller tracks the batch's done count; put it on the row and on the
   topic page's next-step line, as the 8e brief asked.
3. **The recorded name does not carry across pages.** Typed on the Loop
   board, empty on the topic page after a reload. Keep it for the
   session (localStorage is fine for this — it is a per-viewer
   convenience, not state) and pre-fill every name field from it.
4. **"5 checks on the leaderboard"** after a judged run — say what kind:
   "5 checks · 3 about the judged suite".
5. **The duplicate pair still ranks #1 and #2, and the board names no
   differing field.** The two directories are
   `~/benchmarks/results/full/local__qwen35-delta-moe-7d560104-step945`
   and `…-step945-v2` on the server; Omar will attach their
   `results_*.json`. With the real files: reproduce, then fix whichever
   it is — detector or ranking.

Definition of done for D: on the live server, type a name on the Loop
board, wait ten seconds, keep typing; open a topic page — the name is
there; the queue row counts up; the leaderboard shows one qwen35 row
ranked and the other marked as its duplicate.
