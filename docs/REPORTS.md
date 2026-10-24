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
- `reference`: the model near the ceiling is read from (it scores 90% or
  more there) and the loop examples are set beside;
- `baselines` (27e): the builds the viewer can compare against, the first the
  default — Q2 (UD-Q2_K_XL), the same size as our checkpoints (12.6 GB, to
  step4000's 13.6 and step8804's 15.8), then Q4 (22.9 GB). A switch in the
  report's head picks one (`&base=` in the link). The lines that name a
  baseline follow it: instruction following's question-by-question sentence,
  Across training's line against it and its dashed line (BF16 stays beside
  it), and the thinking-tokens line. Every number stays as it is; the server
  works the lines out for each baseline from what the stats worker keeps;
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
  draw of questions of the same kind. It comes from how much the score varies
  between questions; a question asked several times counts once, at the
  average of its runs (23b: the published page and the board say so, where
  they said "from one run").
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

## Two builds, question by question (23)

`eval_tasks/reports/paired.json` (`"kind": "paired"`) compares two of our
builds on the questions both answered, at
`#tab=reports&report=paired&a=<build>&b=<build>`: a link a colleague can open
on the same pair. The scorecard links to it under Our runs ("Compare two of
these question by question ▸", its first two picks).

- **The builds** are every row of the scorecard's `ours` (`"builds": {"from":
  "step-checkpoints"}`), "Q4, earlier run" among them; a new checkpoint added
  there is here too. The picker offers those with scored answers. `default`
  is the pair it opens on.
- **The numbers** come from each row's newest scored samples (`frontier[].ok`,
  the board's own verdict), lined up by question id. A question only one build
  answered, or one not graded on either side yet, is left out and the row says
  how many. One run a question (HLE, MATH Level 5, MMLU-Pro, SimpleQA): the
  difference, its 95% range and the exact McNemar test. Several runs (GPQA ×4,
  OTIS ×8) and ARC-AGI-2 by task: each question at its average, the paired
  standard error and the range. The sums are `scripts/paired_stats.py`, which
  `score_bundle.py` and the parity check use too.
- **Churn:** under the table, for GPQA and OTIS, how often each build gives a
  different result between two of its own runs, beside how often the two
  builds differ.
- **The worker** works out all 21 pairs, each benchmark, and keeps them in
  `stats.json`; a page reads what is kept, and swaps A and B by flipping signs
  and counts.
- **The reader's rules:** the hidden half is counted, never listed or opened;
  GPQA, HLE and MATH never show their question text, and an answer's lines
  quoting it are withheld. No Epoch, card or Artificial Analysis number.

After a deploy, the pairs are worked out once by the worker:

```bash
cd ~/benchmarks/aienh && time sudo docker compose exec -T bench python -m service.report_stats --again
```

## Beyond the score (23b)

Four measures from the answers and grades the board keeps, worked out by the
stats worker for every row with scored Frontier answers (`service/beyond.py`)
and kept in `stats.json`; totals over both halves, no question listed.

- **Reliability** (GPQA Diamond ×4, OTIS Mock AIME ×8): the share of
  questions right every time, never, and sometimes. A question with a run not
  graded yet is left out and counted.
- **Calibration** (HLE): CAIS's calibration error, ported from
  `hle_eval/run_judge_results.py` at `a59048963171` with its quirks (bins of
  100 answers sorted by the judge's confidence; the last bin never counted, so
  100 to 199 answers give 0; fewer than 100 give none), over the graded
  answers with a confidence; answers that ran out have no grade and are
  counted. Answers sharing a confidence keep question order, where numpy's
  sort gives an order of its own. Ten bins of stated confidence for the
  chart, the mean stated confidence beside the share right, and how many
  replies stated none (the judge counts those as 100%).
- **SimpleQA Verified:** right, wrong, not attempted and ran out; tries
  ((A + B) of all), right when it tries (A of A + B), and F, the SimpleQA
  paper's harmonic mean of the two. The board's score (A of all) doesn't
  change. Our prompt carries Epoch AI's "guessing game" line, so tries are high
  by design.
- **Loops:** the share of answers ending in a loop by the board's rule, of
  all and of those that ran out of room. 27f: the rule reads the part where
  the answer ended, alone (`fb.ended_in_loop`), decided by the reply: an
  answer that finished, its reply; one that ran out with no reply (nothing
  but whitespace after its thinking), its thinking; one that ran out partway
  through its reply, that reply. A served answer's thinking comes back closed
  even when the limit cut it mid-thought (`service/served.ask`), so the
  closing tag can't tell where it ended. Read on the whole text, a
  reply drafted in the thinking and written out once repeated its end three
  times and was counted: 81 of the 82 step8804 answers it flagged on IFEval
  and IFBench (the questions the reader shows) were that.

They show on a model page's Scores tab ("Beyond the score", closed unless
opened, its header the HLE calibration error), and on the step-checkpoints
report: a section after "Scores by benchmark", a column "Ended in a loop" in
"Thinking tokens" and its chart "Score against thinking".

After a deploy, the worker works them out once:

```bash
cd ~/benchmarks/aienh && time sudo docker compose exec -T bench python -m service.report_stats --again
```

## On-device models (25c)

`eval_tasks/reports/on-device.json` (`"kind": "on_device"`, listed after the
others): of the models that can run on a phone, which are best — and where
are ours.

- **Rows.** Ours, measured here: the step checkpoints, the Q4 baseline and
  (27e) Q2, "the original, squeezed to 2 bits — our same-size baseline" (a
  row's `line`), and every phone build with a score on the report's
  benchmarks. The others,
  published: Edge0-35B and -8B, Bonsai 27B (1-bit), Ternary Bonsai 2 2B, 4B
  and 9B, Gemma 4 E4B and E2B, and (27f) NVIDIA-Nemotron-3.5-Bolt-E4B. Nothing
  is measured here for those, and nothing is estimated.
- **Nemotron-3.5-Bolt-E4B (27f).** Gemma 4 E4B-it trained further by NVIDIA
  (8B total, 4B effective), released 23 Oct 2026. Its card (pinned to the
  commit read on 24 Oct) gives IFBench alone of our four. The cell is the
  Q4_K_M file's, the one its memory column is: 64.00, NVIDIA's own run under
  NeMo Gym with reasoning on, beside Gemma 4 E4B-it's Q4_K_M 42.67 on the same
  setup (its base row, `base_only`), so it keeps 150.0%; the BF16 pair, 63.79
  and 42.84, is in the cells' setting, on hover. Its memory is the 4-bit file's
  size from NVIDIA's file list for NVIDIA-Nemotron-3.5-Bolt-E4B-4GB (6.18 GB,
  though the repo's name says 4GB). BFCL-v3 and Tau2-Bench are under "Also
  published". Its row's line, `line`, says it was trained further, not
  compressed; `line_tip`, on hover, that its training seed data includes
  IFEval prompts (the card's data table). NVIDIA's letter is N, its family
  colour its own.
- **Columns.** Base model, size on disk, the memory it needs, phone speed
  (the device named, or "no phone numbers yet"), what our testers' phones
  measured (26: our phone build's decode and engine memory, per phone),
  GPQA Diamond, MMLU-Pro, IFEval and IFBench. Every cell says "measured
  here", "measured on our testers' phones", or "published by …" with its
  link.
- **Retention.** Each row's score over its own base's, from the same
  source. For a published model, that is its maker's base row
  (`base_only`), entered like the others but never a row on Models (the
  card's `report_only`). For ours, it is our BF16.
- **A stand-in base (27e).** On a benchmark our BF16 hasn't run, the file's
  `ours_stand_in` (our Q4, on IFEval and IFBench) stands in for it. Each row
  it serves says so — "of our Q4, standing in for BF16 (Q4 keeps 100.0–100.3%
  of BF16 elsewhere)", the range read from the other benchmarks both have —
  and the panel's note says it once. When BF16's own score comes in, it
  replaces the stand-in by itself and the line goes. A build of ours that
  hasn't run a benchmark is a "–" in its panel, "not run yet" on hover.
- **Where the numbers live.** Scores come from the board's own tables
  (measured) and its card entries (published). The other facts (size,
  memory, speeds, what else was published) are in the file, each with its
  link. A reported model's page on Models reads them too.

Enter its card numbers once, after checking each on its source. Gemma 4's
are 25b's and are already there:

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/enter_report_cards.py on-device --by masein
```

## On our testers' phones (26)

`eval_tasks/reports/testers.json` (`kind: "phone_tests"`, `service/telemetry.py`).
Our phone app ("Etachat") checks itself once on each tester's phone and
uploads one JSON a test to Cloudflare R2 (Kevin's format, schema 1). The
board reads the same files. It is read-only, and it never asks his app to change.

- **The sync.** Every 10 minutes (`TELEMETRY_SYNC_S`), while the R2 keys are
  in `.env`, it lists `R2_TELEMETRY_PREFIX` in `R2_BUCKET`. It downloads only
  the tests it doesn't have yet, by test id. Each is kept as it came in
  `BENCH_ROOT/phone/telemetry/<install_id>/<test_id>.json`. Nothing is ever
  written to R2.
  - It uses no boto3: two signed GETs (AWS Signature V4, `service/r2.py`).
  - Without the keys the place says "Phone checks: not set up — R2 keys
    missing in .env". A failed sync says why, and when the last good one was.
  - To sync now: `python -m service.telemetry sync`.
- **Reading.** A test is read as Kevin's Testers dashboard reads it, so the
  two never disagree:
  - answers are the checks done over those run (`not_run` never counts);
  - prefill is speed-cold → speed-warm;
  - decode is the sustain check's, or think-math's when sustain didn't run;
  - time to first word is speed-warm's;
  - sustained is sustain's last third over its first.

  A field we don't know stays in the raw file. A higher schema is kept raw
  and said ("schema 2: not read yet").
- **Names.** These are ours: `eval_tasks/phone/names.json` maps the app's
  ids to names:
  - `models` maps a `model_id` to `{name, row, what, sha256}`; `row` is the
    board row it belongs to (`served/...`);
  - `tiers` and `engines` map an id to a short description;
  - `devices` maps a model code to its marketing name.

  An id not in it shows as itself with "(not named yet)", never guessed.
  Plain names only: the file goes to the public mirror.
- **Flags.** A flagged test is greyed and kept out of the medians it would
  skew, with the reason on hover. The thresholds are in one constant,
  `telemetry.IMPOSSIBLE`. A test is flagged when:
  - it shows a speed no phone can do: decode above 60 tok/s, or prefill
    above 2,000, for a model named as 30B-class (25B to 40B). This test
    leaves the medians entirely;
  - it ran while charging: its battery temperatures are left out;
  - the app left the foreground (`stayed_in_app: false`): its speeds are
    left out.
- **On the page.**
  - Tiles: tests, phones, testers, didn't finish.
  - Every test, newest first. Click one to open every check with its reply.
  - The medians per phone and model, over its finished tests.
  - What failed, by phone and app version.
- **Elsewhere.** The phone build that `names.json` maps a model to shows the
  measured medians per phone on its model page ("Mobile · on the phone"),
  above the numbers typed in by hand. Those keep their label, "Reported by
  hand". The On-device report has a column for them too.

Tester labels and install ids are shown on the board, which is internal.
The tests use made-up ones.

