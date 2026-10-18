# Reports

Reports is a place on the board, after Benchmarks (`#tab=reports`). Each
report is one file in `eval_tasks/reports/`, and a new file is a new report.
A report opens at its own link (`#tab=reports&report=<id>`).

Every number on a report's page comes from the board's own tables, read when
the page opens:

- **Ours:** what the board shows for each row (score, standard error, answers,
  how many ran out of room). A benchmark that isn't finished shows a dash,
  never an older number. Each row has a status line: "Complete", "18 of 21
  chunks in", "HLE waits for grading".
- **Epoch AI:** from the daily import (AI models ▸ Outside data), only at the
  versions the file names. The import keeps every version a report names, even
  when it is outside the per-maker limit or below the model's highest effort.
  Those rows are read by the report and never shown on Outside data.
- **Model cards:** entered through the board's card entry, each with its
  card's link and date (below).
- **Thinking tokens and loop examples:** worked out from the stored answers by
  the stats worker (below), and kept.

No Artificial Analysis number is read: a report must stay shareable. A loop
example from the hidden half is refused. A report never starts a run, an
import or a grading batch.

## The file

`eval_tasks/reports/step-checkpoints.json` holds:

- `ours`: each of our rows (`served/… · thinking`), its label, its colour slot,
  whether it is in the picker, whether Models ▸ Latest shows it (`glance`),
  and a `note` saying what it is where it isn't one of the team's (22:
  Ornith-1.5 Q4_K_M, "measured here, on its card's settings; not the team's
  model");
- `benchmarks`: our task for each benchmark, Epoch's name for it, and whether
  it is compared;
- `theirs`: each other model, with:
  - the Epoch version to read;
  - HLE from Epoch's collection, by its version (`"from": "epoch_external"`);
  - the card cells (`"from": "card"`), each with the card's link pinned to the
    commit that was read, its date, the model and benchmark as the card names
    them, and its setting;
  - `check_14oct`, the values Epoch's data had on 14 Oct, used by the tests;
- `loops`: the MMLU-Pro questions of the loop examples, each with a note;
- `reference`: the model the token line compares against;
- `page`: the full page (below), its file in `eval_tasks/reports/pages/`, its
  starting pick and its datasets; and on a reference row, `page_status`, the
  status the page shows for it ("7–8 Oct, Qwen's card settings");
- `glance: true` on the rows Models ▸ Latest shows, ours and the popular
  models alike, and `glance_benchmarks`, its benchmarks (below);
- on a card cell, `note`: a word the cell's source says with it ("medium
  effort" on o4-mini's MMLU-Pro).

## Models ▸ Latest

Models opens on Latest when the link names no other view. It is one table:
the report's rows marked `glance: true` — ours (step4000, step1000, Q4, BF16,
Q2, and Ornith-1.5 Q4_K_M) and the popular models — on the report's
`glance_benchmarks` (GPQA Diamond, OTIS Mock AIME, MMLU-Pro, HLE).

- Every number is read from the report, so the two always agree.
- Our rows are bold; each published row says where it is from, a model card
  with its link. One of ours with a note says it in its Source column
  (Ornith-1.5: "Measured here, on its card's settings; not the team's model").
- It is sorted by the first benchmark, with no rank and no average: published
  numbers are never ranked or averaged with ours.

Change the models by editing `glance` in the report file. Latest is shown
only while one of our rows on it has a score; until then Models opens on
Standard.

## Ornith-1.5-35B-A3B (22)

A third-party model a colleague asked about, measured here with a runtime of
its own (`card-ornith`: upstream llama.cpp and its card's settings, never the
lda runtime; docs/ORNITH-RUN.md). On the step-checkpoints report:

- **Ours:** "Ornith-1.5 Q4_K_M" (`served/Ornith-1.5-35B-A3B-Q4_K_M · thinking`),
  colour slot 5, in the picker, on Latest, with its note;
- **Its card:** "Ornith-1.5-35B-A3B, its card", a published row with its card's
  GPQA Diamond 89.2 and HLE 25.6 (no tools), each the card's average of five
  runs (the cell's note) and linked to the card at commit `12393612`, on
  Latest so the claim sits beside the measurement. Like every published row it
  is never ranked or averaged with ours.

On the ran-out report it is in the picker beside step4000 and step1000.

The published page (`pages/step-checkpoints.html`) is masein's, unchanged. Its
picker and colours name its own five models, so Ornith-1.5 isn't one of its
picks; its tables read the report's rows, so Ornith-1.5's row is in its table
of our scores (grey, as "Q4, earlier run" is) and its card's row among the
other models.

## What the numbers mean

- **A score** is the average over every answer, in percent. Its line says how
  many questions and how many times each is asked: GPQA Diamond 198 × 4, OTIS
  Mock AIME 45 × 8, ARC-AGI-2 167 tasks × 2, the rest once.
- **±** is its standard error: how far the score could move with a different
  draw of questions of the same kind.
- **The token columns** are named as the published page names them, and their
  definitions are the page's own words, read from its file.
- **The thinking tokens** are counted exactly as `~/tokens.py` counts them
  (its copy is `tests/fixtures/tokens_py/tokens.py`, and the tests hold the
  board to it):
  - an answer's thinking share leaves the think tags out;
  - think_share is all the thinking characters over all the characters;
  - the median and p90 are the values at n/2 and 0.9·n of the sorted list;
  - each answer's thinking tokens are rounded before they are averaged.
  A benchmark whose answers are in but not scored yet is read from its
  answers, as `tokens.py` reads it.
- **A loop example's length** is its thinking alone, as the page's "thought for
  N characters" says (21d.1): Q4's answer to MMLU-Pro 486 is 7,247 characters
  whole, 6,485 of thinking and 742 of reply.

## The full page

The scorecard as masein publishes it is at `/reports/step-checkpoints/page`.
"Open the full page" on the report's card in Reports, and in its report, opens
it in a tab of its own. 21's report, in the board's own style, stays as it is.

- **The page:** `eval_tasks/reports/pages/step-checkpoints.html`, exactly as
  published. When the published page changes, replace the file whole; any fix
  goes in the runtime or the shell, never in the page.
- **Its runtime:** `service/static/reports/dash.js` gives the page what it was
  written for (`dash.data`, `dash.calc`, `dash.onData`, the params, the link,
  the colours). `theme.css` holds the theme tokens, light and dark; the page
  follows the viewer's setting and the board's own. d3 7.9.0 is kept beside
  them (ISC licence, `LICENSE-d3.txt`). Nothing is fetched from another host:
  the page's Content-Security-Policy says so to the browser too.
- **Its four datasets,** live, at `/api/reports/step-checkpoints/datasets/<id>`,
  built from the tables 21's report reads and rounded as it rounds:
  - `ours`: a row per model and benchmark, with its status;
  - `others`: every other model, with where its numbers come from;
  - `tokens`: the token stats;
  - `loops`: the loop examples, from the shown half, with the file's notes.
- **The header line** says when each source last changed.

The page works out where each model stands from the numbers as shown, to a
tenth of a point, and 21's report now does the same. Both show the same gap and
the same verdict in every head-to-head cell. Two things differ by design:

- the page sets ours against our lda-runtime runs and the popular models;
  21's report sets it against "Q4, earlier run" too;
- the page's summary line counts the popular models only.

## The stats worker

The token stats, the loop examples and the ran-out lists are worked out by a
process of their own (`python -m service.report_stats`):

- it runs at the lowest CPU and disk priority, one at a time;
- it starts after each import or scoring, and when a page finds something not
  worked out yet;
- each stat is kept in `BENCH_ROOT/reports/stats.json`, with the files it read
  and when it was worked out.

A request only reads what is kept. A stat whose answers changed since is shown
with its time until the new one is in. A first open, before any is kept, says
"Working these out, back in a minute." The worker's lines are in
`BENCH_ROOT/reports/stats.log`.

To time it on the server, working every stat out again:

```bash
cd ~/benchmarks/aienh && time sudo docker compose exec -T bench python -m service.report_stats --again
```

## The card numbers

Check every number on its card first. Then enter the file's card cells through
the board's card entry. Running it again changes nothing:

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/enter_report_cards.py step-checkpoints --by masein
```

## The token stats beside `~/tokens.py`

This prints "same", or the fields that differ, for each row and benchmark:

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/report_token_stats.py step-checkpoints --against - < ~/tokens-2026-10-14.jsonl
```

## Where step4000 runs out of room

`eval_tasks/reports/ran-out-of-room.json` (`"kind": "ran_out"`) lists every
answer step4000 and step1000 ran out of room on. Q4 is compared on the same
questions. For each model and benchmark it shows:

- how many ran out, and their share;
- for MMLU-Pro by subject, and for HLE by category;
- how Q4 did on the same questions;
- whether the right answer was clearly stated before the limit, and at which
  token;
- how long the loops are.

The list filters and sorts, and one answer opens at a time, its thinking in
full.

**For insight only, never for training.** The warning stands above the report
and above every question. The page has no export, and nothing from it reaches
a training, generation or Improve path.

**Both halves, in this report only.** masein decided on 14 Oct 2026 that this
report shows the hidden half too. Only a report file that says
`"show_hidden": true` does; any other is blind to it, as the rest of the board
is.

**Question text the board never shows stays withheld here too:** GPQA
Diamond's and HLE's (their authors ask), and MATH's (disputed copyright). Their
answers show their stats and their thinking, with each line that quotes the
question withheld.
