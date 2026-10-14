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
- **Thinking tokens and loop examples:** worked out from the stored answers in
  a thread of their own, and kept until a file they read changes.

No Artificial Analysis number is read: a report must stay shareable. A loop
example from the hidden half is refused. A report never starts a run, an
import or a grading batch.

## The file

`eval_tasks/reports/step-checkpoints.json` holds:

- `ours`: each of our rows (`served/… · thinking`), its label, its colour slot,
  and whether it is in the picker;
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
- `reference`: the model the token line compares against.

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
