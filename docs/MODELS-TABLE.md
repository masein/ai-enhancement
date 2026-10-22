# One table for every Models tab (27)

Latest, Standard, Mobile (DeviceMark's table too), Everyday and Frontier, and
the full Mobile-MMLU card, all draw through one function, `modelTable(spec)`
in `scripts/report_lm_eval.py`. A view says only which rows, which columns,
the groups and the default sort. It works out no numbers: every cell is the
view's, drawn one way. Compare (models across, benchmarks down) and Language
modelling's cross-entropy table (a row per task and model) keep their own.

## The row

- **Mark.**
  - A filled 10 px dot in the model's own colour: measured here. The colour
    is the one it has on the reports' charts (`DATA.rowColours`, from the
    report files), else its family's.
  - A hollow grey ring: published by someone else.
  - A model with both gets two rows, one with each mark.
- **Name: `displayName(row)`.** The same on every tab and on the model page.
  - Format: family · variant. The family is the maker's full name.
  - `nameParts()` splits a board name:
    - " · thinking" goes to the Thinking filter;
    - "(lda runtime)", "(phone build)" and "(card settings)" go to the line
      under the name;
    - our quant is the variant ("Qwen3.6-35B-A3B · Q4");
    - an older run's tag stays ("· Q4 (original, k=8)", "· BF16 (unsloth)");
    - a published model's setting is its variant ("o4-mini · high"), and its
      quant goes to the line ("Bonsai 27B", 1-bit);
    - "Qwen3.6 k4-LDA", based on Qwen3.6-35B-A3B, is written out in full.
  - Two rows never share a name in one table. A published twin of a model
    measured here says "(published)"; any other pair gets the end of its id.
  - A raw run id is the name when nothing better exists, cut in the middle
    with the full id on hover.
- **Line.** One muted 12 px line: maker (for others' models) · size or
  params · quant · build (phone build, served, GGUF, uploaded) · thinking.
  Only a status that matters (prelim, provisional) stays a badge.

## The cell

- **Number.** Monospace, tabular figures, right-aligned. Bold is the best in
  the column, or within its noise. Measured cells are compared with each
  other, and each source's published numbers with their own; never across.
- **±.** 11 px, muted, after the number.
- **Bar or tint.**
  - A 3 px bar of the score out of 100, in the row's colour (grey when
    published). Only 0–100 score columns get one.
  - With Tint on (one switch on every tab, remembered, off by default), each
    score column is tinted `--heat-1…5` from its own lowest to its highest,
    with no bars.
- **Published numbers.** Italics, with a raised letter for the source: E
  Epoch AI, Q Qwen, G Google, P PrismML, X Edge0, C model card, A Artificial
  Analysis, D DeviceMark. The key is one line under the table, and the whole
  source is on hover. A card's letter links the card.
- **No score.** "–", never a blank.
- **Everyday.** The share passed, a whole number, is the big figure ("81"),
  with "22 of 27" under it. How many ran out of room is in the Total's hover.

## The frame

- **Groups.** Ours, then Open models, then Closed models. Inside each group,
  rows are sorted by the chosen column, highest first; the default is the
  view's headline.
  - Our team's builds are Ours: `local/` checkpoints, and served or GGUF
    builds that aren't someone else's model.
  - Ornith and Bonsai measured here are open models with a filled dot.
  - Whether a published model is open comes from one rule on its name
    (`MT_OPEN_RE`). No source carries it.
- **Header.** The short name on the first line, never cut (GPQA, MATH L5,
  ARC-AGI-2 never wrap); a long name of several words ("Mathematics &
  Science") may wrap between its words. What it tests and how many questions
  sit muted under it, the full name is on hover, and groups sit above. The
  group row's empty head is pinned over # and Model only, so a scrolled
  table never hides a group's name.
- **Rank.** Standard and DeviceMark keep their # column and its interval rule.
- **Thinking: on / off / both.** One row a model.
  - The default is each model's own setting (a served model as registered,
    else its switch's way).
  - "Both" shows the other setting as an indented sub-row.
- **Columns ▾.** Says "Columns: N of M ▾". Each tab remembers its choice
  (`bench-mt-cols`).
  - Frontier has one column a benchmark: `MT_BENCH` holds every name others
    give it (HLE is Humanity's Last Exam, MMLU-Pro is MMLU Pro).
  - When two sources give a number for one benchmark, a measured number
    wins; otherwise the newer published one shows, the other on hover.
- **Empty rows.** A row with no score in the columns shown is hidden and
  counted: "N more models have no score in these columns · show them".
- **Scrolling and phones.**
  - The header stays at the top and the names at the left. The table
    scrolls inside its own box; the page never scrolls sideways.
  - Under 640 px there is one card a model: its mark, name and line, and
    four headline scores with their bars. Tapping a card opens the model
    page. Sort ▾ sits above the cards, and the tabs scroll sideways. Tint
    has nothing to tint on a card, so its switch shows on the wide view only.

## Tests

- `tests/test_27_values_browser.py` checks that no number changed.
  - `tests/fixtures/models27/before.json` is every (view, model, column)
    value the old renderers drew on its world.
  - Each cell's own number must be in sight now, and every other number it
    showed must be in the cell or its hover.
- `tests/test_27_models_table_browser.py` covers the one renderer, the same
  name everywhere, the naming rules, Frontier's columns, no blanks,
  Everyday, Tint, the Thinking filter, empty rows, and every tab at 1280 and
  400, light and dark, scrolled both ways.
