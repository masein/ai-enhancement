# Brief for Claude Code — phase 11 live-check fixes (11e, 11f)

Start from `1a947bf`. There are **two PRs**. Do them in order, and each must
be green on its own:

- **11e**: the live-check fixes. These are items 1–12 below.
- **11f**: masein's seven. A simpler table header, a smoother opened row,
  one motion system, every new page starting at the top, one pattern for
  row actions, and better dropdowns. See the second half of this file.

These are the findings from masein's live check of phase 11 on the server,
on 2026-09-22. The check used a real Propose → Approve → Generate on
Mathematics & Statistics and on Physics & Astronomy, and was run at 1,512 px
(a MacBook's default width) and at 400 px.

## What passed

Keep all of this working:

- **More ▾** opens fully, and Esc returns focus to the button.
- **The search** greys out `local/qwen35-delta-moe-7d560104-step945` as
  "results only — weights not on the server".
- **The API refusal** (422) uses the runner's own wording.
- **Physics & Astronomy dataset #4** spread its 20 documents over 12
  sub-areas: projectile motion, the Bohr model, Darcy flow, induction,
  damped oscillators, and so on. Dataset #2 before it had put 11 of its 18
  documents on relativistic momentum.
- **The Judged topics chip** and the **Judged by area** radar source are
  disabled with their reason.
- **Opened rows** survive a poll and come back from the hash.
- **The model page**: its hero, the sticky section chips, topics grouped by
  area with no mean while the judge is provisional, the preliminary note said
  once, and the one-row-per-topic length table.
- **The dark theme** reads well.

Two deviations from the brief are accepted:
- the `kept + len(missing) == requested` invariant;
- the toolbar on two rows at 1,280 px.

**The rules that do not bend are unchanged from phase 11.** In particular,
only diagnose-half items are ever read to build a generation request.

---

## 1. Spread documents on every topic (masein's decision, 2026-09-22)

### What was seen

Mathematics & Statistics (dataset #3) said "this topic's questions carry no
domain labels, so the documents are not spread by area". That is wrong: its
bank has 18 labels (Algebra, Probability, Linear Algebra, …), and they were
refused only by the "at most 15 distinct values" cap. As a result, 13 of
its 20 documents were about conditional probability or Bayesian updating.

Today only **3 of 37 topics** pass `is_domain_label_set()`: Physics & Astronomy,
AI & Machine Learning, and Finance & Accounting.

### The labels in the 37 banks

**8 topics have a small label set as written, 6–25 labels each:**

| Topic | Labels |
|---|---|
| AI & Machine Learning | 6 |
| Finance & Accounting | 9 |
| Physics & Astronomy | 12 |
| Food & Veterinary Sciences | 16 |
| Mathematics & Statistics | 18 |
| Earth & Environmental Sciences | 21 |
| General & Multidisciplinary | 22 |
| Agriculture | 25 |

**10 topics have many labels, but those labels share a prefix** before
` — `, ` – `, ` - `, `: ` or ` / `. Law's `Legal Method – Precedent` and
Manufacturing's `Manufacturing Processes - Casting` are examples. Grouping
by prefix gives 7–11 groups:

| Topic | Labels → groups |
|---|---|
| Architecture & Built Environment | 86 → 7 |
| Design | 97 → 11 |
| Education | 100 → 11 |
| Ethics & Religion | 95 → 8 |
| History & Archaeology | 97 → 11 |
| Law | 52 → 8 |
| Manufacturing & Applied Sciences | 88 → 10 |
| Media & Communication | 100 → 10 |
| Philosophy | 81 → 9 |
| Political Science & International Relations | 100 → 9 |

**19 topics have nearly one label per question.** For example:
- Sociology: 100 labels for 100 questions;
- Arts, Business & Management and Anthropology & Human Geography: 85 each.

### The rule

1. **A label's group** is the text before the first ` — `, ` – `, ` - `,
   `: ` or ` / `, each with a space on both sides except `: `.
   - A label with none of these is its own group.
   - A bare hyphen inside a word, as in "Evidence-Based", never splits a
     label.
2. **By area**: this covers the 18 topics above.
   - Use the labels as written when the bank has at most **25 distinct
     labels**. Otherwise use their groups, when there are at most 25
     groups.
   - Allocate over those areas exactly as today: by the count of failing
     diagnose-half items, largest remainder, at least one each while the
     count allows, in round-robin order.
   - Drop the "each value on at least 2 items" requirement. Food &
     Veterinary Sciences has single-item labels.
3. **By concept**: when the bank has more than 25 groups. This covers the 19
   topics above.
   - Each document targets the **full label of one failed diagnose-half
     item**.
   - Items are taken weakest first (lowest score), with ties broken by qid.
   - One document per distinct label. If the count is larger than the
     number of distinct failed labels, go round again.
4. **Label hygiene, in both modes.**
   - The text sent is at most 64 characters and at most 10 words, and has
     no `.`, `?` or `!`.
   - If it fails, use its group instead. If the group fails too, skip that
     item.
   - Never send `intent`, question text, a count or a score.
   - The request still gets exactly one extra line: `Focus: <label>`.
5. **Show the plan before Approve.** This is masein's condition.
   - The proposal card's **Decide** section shows the plan beside the spec.
     Two examples:
     - "20 documents, one per concept the model missed in the practice
       half: Structure and Agency · Social Construction · …"
     - "Documents will cover 8 areas: Algebra 3 · Probability 3 · …"
   - A tick box, **Spread the documents over these**, is on by default.
   - **Approve freezes the plan** in the proposal as `approved_focus`. This
     is the ordered list of labels, long enough for 100 documents.
   - Generate with count N uses the first N entries of that frozen list.
   - The Generate section repeats the plan read-only. The dataset's
     provenance records `focus_mode` (`area`, `concept` or `off`) and the
     labels actually used.
   - A proposal approved before 11e keeps today's behaviour. Its card says
     "approved before plans were shown".
6. **Say the true reason when there is no plan:**
   - "no sub-area labels on this topic's questions";
   - "every label is too long to send";
   - "the approver turned spreading off".

   Never say "carry no domain labels" when labels exist.

**Tests:**
- A fixture shaped like Mathematics (18 labels) → by area, spread.
- Fixtures shaped like Law (`Legal Method – X`) and like Manufacturing
  (`Manufacturing Processes - X`) → grouped into their prefixes.
- Over the 37 real banks: 18 topics are by area and 19 are by concept, and
  the counts in the tables above match.
- A fixture shaped like Sociology (unique labels) → by concept. Each
  document is a distinct failed label, weakest first.
- A report-half item that fails and has a unique label → its label never
  appears in any request.
- A label over 64 characters → its group is sent.
- The plan is visible on the card **before** Approve.
- The dataset's labels equal the first N entries of `approved_focus`.
- Unticking the box → no `Focus:` lines, and `focus_mode: off`.

---

## 2. The header at 1,512 px: the checks pill covers More ▾

**What was seen.**
- The pill "6 checks · 3 about the judged suite — show" is so wide that it
  overlaps the tab strip. At the centre of More ▾, `elementFromPoint`
  returns the pill's `span.dot`.
- The strip then scrolls sideways: `scrollWidth` 476 against `clientWidth`
  418. "Overview" shows as "ew".

masein flagged the same pill separately. In his screenshot it is also taller
than `masein ▾` and `Theme ▾` beside it, and its text runs to "— show".

**Fix.**
- The pill reads only `● 6 checks ▾`. "3 about the judged suite" moves into
  the panel it opens.
- It uses the same button component as `masein ▾` and `Theme ▾`: the same
  height, font, border and radius.
- Above 1,100 px, the tabs never scroll, and nothing in the bar overlaps
  anything else.

**Test.** At 1,280, 1,440, 1,512 and 1,920 px:
- every tab's centre hit-tests to the tab itself;
- `#tabs.scrollWidth === #tabs.clientWidth`.

## 3. The LIVE badge shows the timezone, not the time

**What was seen.** The badge reads `LIVE · +04`, and the status line reads
`● live +04`. The refresh string is "refreshed 2026-09-22 15:17 +04", and
the code took its last token.

**Fix.** The badge reads `LIVE · 15:17`, and so does the status line.

**Test.** With a fixed refresh time, both show `15:17`.

## 4. The "1 duplicate" link overlaps the Params cell

**What was seen.** On the qwen35-delta-moe-…-v2 row, "1 duplicate ▸" paints
over "2.3B · 908M act", and the two render as garbled text.

**Fix.**
- The model cell clips with an ellipsis and never paints into the next
  cell.
- The duplicate link moves inside the cell's own width, after the badges,
  or into the opened row.

**Test.** In every row, the bounding boxes of the model cell's children
stay inside that cell.

## 5. Opened row: the TASKS block runs into MMLU BY AREA

**What was seen.** The text "#6/15 · tied with best" overlaps the
"Humanities & Arts" and "Business & Law" rows of the next block.

**Fix.** Give the blocks grid `minmax()` widths, and let the tasks block
wrap its last column.

**Test.** No two blocks' boxes intersect, at 1,280 px and at 1,512 px.

## 6. MMLU by area is empty for every model

**What was seen.**
- Every model's stored `diagnose.json` still carries the **old 15
  categories** (economics, law, medicine & health, …).
- As a result, the Knowledge chip shows 8 empty area columns, and every
  opened row shows 8 empty bars.
- masein will re-run `scripts/diagnose.py` on the server. That is a data
  step, not a code change.

**Fix, in the code.**
- When a model's stored categories aren't the current `categories.yaml`
  set, say so once: "MMLU by area needs a fresh diagnosis — this one was
  made with the old 15 categories". Don't show empty bars.
- Hide area columns that are empty for every model on the page.

**Test.** A fixture with old-category diagnoses shows the sentence, and no
empty bars or columns.

## 7. A poll drops focus inside an open popover

**What was seen.** With Columns ▾ open and a control focused, `render()`
keeps the popover open but moves focus to `<body>`.

**Fix.** Put focus back on the same control, found by its data key, like
the search field already does.

**Test.** Focus a checkbox in the popover and call `render()`.
`document.activeElement` is still that checkbox.

## 8. The frontier counts models of the same size

**What was seen.**
- Live: both SmolLM2-135M and SmolLM2-135M-Instruct sit on the frontier,
  and so do two 362M models.
- In the test board, three 750M models do.
- The dashed line runs straight up between them.

My brief's rule said "no *smaller* model beats it". It should say **no
model of the same size or smaller** beats it by a z-tested gap.

**Fix.**
- Apply that rule.
- Draw the line through the frontier points sorted by parameters. At any
  one size, the line goes through the best point only. Points tied within
  the noise are still marked, but the line doesn't jump between them.

**Also:** the caption "0 models here are bigger and score lower than X"
becomes "No model here is bigger and scores lower than X."

**Test.** Two fixture models of the same size, 10 points apart with a real
gap → only the better one is on the frontier.

## 9. Highlight values are too long

**What was seen.** At 1,512 px, BEST MODEL wraps onto four lines
("qwen35-delta-moe-7d560104-step945-v2 · 39.3"), and WEAKEST TOPIC puts
"/ 4" on a line of its own.

**Fix.**
- **The value is the number only**, in 28 px mono: `39.3`, `0.79 / 4`,
  `7 / 37`, `30 / 30`.
- **The name** goes on its own 16 px line, with an ellipsis and a tooltip
  giving the full id.
- **The canary numbers** are rounded to 2 decimals: "0.17 from the human
  marks, 0.03 from the last run".

**Test.** At 1,280 px and at 1,512 px, every card's value fits on one line.

## 10. 400 px

- **The Leaderboard shows no scores.** The pinned model column is 280 px
  inside a 315 px scroller, which leaves 3 px for the scores.
  - Cap the pinned columns at 45% of the scroller, with an ellipsis on the
    name.
  - Keep only the `prelim` badge.
  - Show a fade on the right edge, plus a small "scroll →" hint, while
    there's more to the right.
  - **Test:** at least one score cell is fully visible without scrolling.
- **The sticky header is about 145 px tall**, over three rows.
  - Row 1: the title, LIVE, and one `⋯` menu holding checks, name and
    theme.
  - Row 2: the tabs.
  - **Test:** the header is at most 96 px tall.
- **Chart text shrinks to about 5 px.**
  - Below 600 px, draw Weakest topics as HTML rows, not SVG.
  - Give the scatter a minimum width of 520 px inside its own sideways
    scroller.
  - **Test:** no chart text is under 12 px.

## 11. Datasets made before 11a

**What was seen.** Dataset #2 shows "18 kept of 20" and says nothing about
the missing two.

**Fix.** Add "2 missing — reasons not recorded (made before 11a)", as the
11a brief asked.

## 12. Polish

- **Spacing.**
  - The Insights intro paragraph touches "SCORE AGAINST SIZE". Add the
    standard 16 px gap.
  - "Submit a model" touches the stats line under it. Add a 12 px gap.
- **The "provisional" badge** beside Weakest topics' model chip renders as
  a tall 16 px box. Use the standard 12 px badge.
- **The opened row's LINKS block.** "Add to radar" and "Run exam" are
  indented, unlike the links above them. Make them link-styled buttons
  with no padding.
- **The disabled Judged topics chip.** Its reason sits as a permanent line
  under the toolbar. Move it into the chip:
  - the tooltip;
  - `aria-describedby`;
  - on click, a one-line note under the chip (use `aria-disabled` so the
    chip still takes the click).

---

## Definition of done — 11e

On the live server:

1. **A new Propose on Sociology** shows, before Approve, a plan of 20
   concepts. After Generate, the dataset's labels are those 20.
2. **A new Propose on Mathematics & Statistics** shows a by-area plan over
   its labels, and the documents are no longer mostly conditional
   probability.
3. **At 1,512 px**:
   - every tab is fully visible and none overlaps;
   - the badge reads `LIVE · <time>`;
   - the highlight values sit on one line each.
4. **The duplicate row and the opened row** have no overlapping text.
5. **After masein re-runs the diagnosis**, the Knowledge chip shows 8 area
   columns with numbers. Before it, the page says the diagnosis is stale.
6. **At 400 px**:
   - the Leaderboard shows at least one score column;
   - the header is at most 96 px tall;
   - chart text is readable.
7. **All tests are green**, with screenshots in `tests/_screens/phase11e/`.

## Deploy steps, for masein, after each PR merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in each PR description and in
`HANDOFF.md`.

---
---

# 11f — masein's seven (after 11e is merged)

masein went through the board himself and asked for these. Each comes with
the screenshot he pointed at. The goal is his: *a board people like to work
in — simple, calm and smooth, not strict.*

**One rule changes here, at masein's request.** The ± error no longer sits
on every cell. It stays one hover or one click away, as described in 1b.

Everything else in "Rules that do not bend" still holds. In particular, the
tie marks keep their statistical meaning: bold means "best in the column or
within the noise of it".

## 1. The table header and the top rows are too busy

**Seen, in his screenshot of the Leaderboard.**
- The header is four lines tall:
  - "AVG ▼ / above / chance · %";
  - "MMLU / 5-shot · % / ⓘ";
  - "ARC_ / CHALLENGE";
  - "JUDGED / AVG / rubric / 0–4", in italics.
- Every cell carries a `±0.6`, plus a `●` or a `≈`.
- The top rows are one solid blue slab. With only 14 ranked models, the
  top rank-fifth and every tie share the strongest tint.

### 1a. A header one line tall, under a quiet group row

**Column names are one word each**, in the phase-11 mono uppercase 12 px.

| Name | Column |
|---|---|
| `#` | rank |
| `MODEL` | model |
| `PARAMS` | parameters |
| `AVG ▼` | average |
| `MMLU` | mmlu |
| `HELLASWAG` | hellaswag |
| `WINOGRANDE` | winogrande |
| `PIQA` | piqa |
| `ARC-C` | arc_challenge |
| `ARC-E` | arc_easy |
| `GSM8K` | gsm8k |
| `TRUTHFULQA` | truthfulqa_mc2 |
| `JUDGED` | judged average |
| `UPDATED` | last eval |

**Everything else leaves the header:**
- The n-shot and unit lines, and the ⓘ icons.
- The unit moves into the name's tooltip, e.g. "mmlu — 5-shot, % above
  chance". The whole header cell carries that tooltip, and is focusable,
  with a dotted underline on hover.
- "above chance" already lives in the `Scale:` pill, so it goes too.

**The group row** (KNOWLEDGE, COMMONSENSE, …):
- It is muted, not accent blue.
- A hairline bracket spans each group.
- It shows only on the `All tasks` chip.

**The JUDGED column** is hidden while no model on the page has a value in
it. Today every row is "—".

**Dates** read "15 Sep". The year appears only when it isn't the current
year.

**One caption line** sits under the table, in mono 12 px, muted: "Bold =
best in the column or within its noise · hover a score for its ± error ·
hover a column name for its setup".

### 1b. Cells: the number, plus bold for the leaders

- **Each cell holds the number only.**
- **The ± error** appears:
  - in the cell's tooltip, on hover or focus;
  - in the opened row, where it already is;
  - on every cell when **Show ± errors** is switched on. This switch lives
    in the Columns popover, is off by default, and is remembered in this
    browser.
- **Remove the `●` and `≈` glyphs from cells.** Instead, the column's best
  score, and every score within its noise, is **bold** with the tint below.
  This is more honest than today: a `●` on one of three statistically tied
  scores claims a winner the data doesn't show.
- **The tint marks the leaders only.**
  - Only the bold group gets a tint: `--heat-3` in light (11.5%), and the
    matching step in dark and dim. Every other cell is plain.
  - This replaces the rank-fifth steps, which painted a slab at 14 models.
  - `Tint: on / off` in Columns keeps working.
- **The Avg column** follows the same rule.

**Tests for 1:**
- The header row is at most 2 lines tall, counting the group row: at most
  64 px at 1,280 px.
- No ⓘ in any `th`.
- Every `th` has a tooltip naming its n-shot.
- No `±` in a cell by default. It appears with the switch on, and in the
  tooltip.
- The bold cells in each column are exactly the z-test's best-or-tied set.
- Only bold cells are tinted.
- The JUDGED column is absent when every value is "—".

## 2. Opening a row should feel smooth

**Seen, in his screenshot of Qwen3-0.6B opened.** The detail appears
instantly and looks like a separate table stuck under the row. The Links
block's buttons are indented unlike the links above them. That indentation
is fixed in 11e item 12; this item is about the motion and the look.

**The motion:**
- The detail opens with a **200 ms** height and fade:
  - the wrapper uses `display: grid` with `grid-template-rows` going from
    `0fr` to `1fr`;
  - the content fades from opacity 0 to 1 and moves from `translateY(-4px)`
    to `0`;
  - easing is `var(--ease)`.
- The ▸ turns to ▾ by rotating 90° over 150 ms.
- Closing plays the same in reverse. The detail row leaves the DOM after
  `transitionend`.
- **It animates only when a person toggles it.** A row that is already
  open, after a poll, a page change, a sort or a hash load, renders open
  with no animation.
- If the opened panel ends below the screen, scroll just enough to show it:
  `scrollIntoView({block: 'nearest', behavior: 'smooth'})`.
- On row hover: the `--accent-soft` background, a pointer cursor, and the ▸
  turns ink. It should be obvious the row opens.

**The look:**
- The panel reads as the row's own continuation:
  - the row's 3 px accent bar runs down the panel's left edge;
  - a faint `--accent-soft` background;
  - 16 px padding and rounded bottom corners;
  - no hairline between the row and its panel.
- A small **✕ Close** sits at the top right.
- **TASKS** becomes a compact list. Each line reads name · score · a thin
  inline bar on the column's scale · rank. The n-shot and ± error are
  muted.
- **LINKS** is a vertical list of equal link-styled buttons.

**Tests for 2:**
- A toggle adds the transition class. A poll re-render does not.
- The panel's height ends at its content height.
- Under `prefers-reduced-motion` there is no transition, and the panel
  opens at once.
- Esc on a focused row, or ✕, closes the panel and returns focus to the ▸.

## 3. The checks pill

This is folded into 11e item 2, as the same button component as its
neighbours. There's nothing more to do here.

## 4. A new page starts at the top

**Seen.** masein scrolls down a page and clicks a link to another page (a
model, a topic, a tab). The new page opens at the old scroll position,
halfway down.

**Fix.**
- **When the view changes** (a different tab, model or topic), the new view
  starts at `scrollY = 0`.
- **The exception:** a button that targets a section still lands on it.
  `state.after.scroll` and the settle logic are unchanged.
- **Back and Forward** (`popstate`) restore the scroll position the person
  left that view at. Save `scrollY` in `history.state` via `replaceState`
  on scroll, debounced.
- **"← Back to …" links** behave like Back: they restore the old position.
- **Changes within a page don't move the scroll:** sorting, paging,
  filters, chips, opening a row, and polls.

**Tests:**
- Scroll the Leaderboard to 2,000 px and click a model → `scrollY` is 0.
- Browser Back → the Leaderboard is at 2,000 ± 50 px.
- A "Read the results" button still lands on its section.
- A sort or page change doesn't move `scrollY`.

## 5. Smooth transitions everywhere: one motion system

**Seen.** masein: *"I feel a bit strictness in the whole system."*
Everything snaps: tabs, popovers, toasts and theme changes.

**Tokens**, beside the phase-11 ones:
- `--dur-1: 120ms` for hover and press;
- `--dur-2: 180ms` for popovers, expanding and the tab underline;
- `--dur-3: 240ms` for toasts and view changes;
- `--ease: cubic-bezier(.2, .8, .2, 1)`.

Under `prefers-reduced-motion: reduce`, all three durations are `0ms`.

**Where they apply:**

| Thing | Motion |
|---|---|
| Buttons, chips, pills, tabs | background and border colour over `--dur-1`; on press, `scale(.98)` |
| Tab underline | slides from the old tab to the new one (a transform on one underline element) |
| Popovers and menus | fade in, and move from 4 px towards their anchor to 0, over `--dur-2`; leaving takes half the time |
| Toasts | slide up 8 px and fade in over `--dur-3`, then fade out |
| A new view, after navigation only | fade in and rise from 6 px over `--dur-3` |
| A value a poll changed (a score, a status, a count) | a soft `--accent-soft` highlight that fades out over 1.2 s |
| Theme change | `.theme-fade` on the root for 250 ms: background, colour and border colours transition, then the class is removed |
| Skeletons | a slow shimmer; none under reduced motion |

**The limits:**
- Animate only `opacity` and `transform`, plus `grid-template-rows` for
  the opened row.
- Nothing lasts longer than 300 ms, except the 1.2 s value highlight.
- **Nothing replays on a poll.** A poll never fades the view, never reopens
  a popover, and never re-animates an open row.
- A click works mid-animation. No animation ever delays an action.

**Tests:**
- The tokens exist, and read as 0 under emulated reduced motion.
- A navigation adds the view-enter class. A poll does not.
- A changed value gets the highlight class. An unchanged one does not.
- Every transition in the stylesheet lists only `opacity`, `transform`,
  `grid-template-rows` or colours.

## 6. One pattern for row actions (the Queue)

**Seen, in his screenshot of the Queue's right edge.**
- **Four button styles in one column:**
  - "Open results" is an outlined blue button;
  - "Log" is an underlined link;
  - "Resubmit" is a grey outlined button;
  - "Retry grading" is a filled blue button.
- **The order changes by row:** Log comes first on some rows and last on
  others.
- **The GPU cell wraps** "1 / min" onto two lines.

masein: *"this ui is not pretty enough and disordered."*

**Fix: one action cell, used by every table with actions.** That means the
Queue, the Loop board, the dataset list, and the Review cards' decide row.

- **Layout.** The cell is right-aligned, with a fixed width per table and
  an 8 px gap. Every button in it is 32 px tall with radius 6.
- **One visible action per row: its next step.**
  - Open results, when done.
  - Retry grading, when grading failed. This is the only **filled** style,
    because it needs a person.
  - Resubmit, when the run failed.
  - Cancel, while queued.
  - These are ghost buttons in the same style, except Retry grading.
- **The rest go in a `⋯` row menu** built on the 11a popover:
  - Log;
  - Resubmit, when it isn't the main action;
  - Copy id;
  - Open model page.
- **The GPU cell never wraps.** It reads `1 min`, `6 min` or `—`, with
  `white-space: nowrap`.
- **Long failure text in Progress** is clamped to 2 lines, with "details ▸"
  to expand. Today it pushes rows to five lines.

**Tests:**
- Every Queue row has at most one visible button plus `⋯`.
- All visible action buttons share height and radius.
- Only Retry grading is filled.
- The GPU cell is one line.
- The `⋯` menu passes the 11a popover checks.

## 7. Better dropdowns

**Seen, in his screenshot of the model page's topic picker.** "Economics —
0.79 / 4" is a native `<select>`, with a separate "find a topic" box beside
it. The other native selects on the board look the same:
- the answers model picker on the topic page;
- `25 per page`;
- `kind: auto-detect`;
- `status: all`;
- the documents format;
- the sort pickers.

**Fix: two components**, both built on the 11a popover. After this, there
is **no native `<select>` in `#view`**.

- **`Select`**, for short fixed lists (up to 12 options): per page, kind,
  status, format.
  - It is a button that shows the current value and a chevron, and opens a
    listbox.
  - Keyboard: ↑ and ↓, Home and End, typeahead, Enter, and Esc.
  - ARIA: the listbox pattern.
  - It is 36 px tall, with radius 6, `--border`, and the accent focus ring.
- **`Combobox`**, for long lists: topics and models.
  - A search field that opens a grouped list as you type.
  - **Topics** are grouped by the 8 areas, weakest first within each area.
    Each option shows the topic name and its score in mono, right-aligned,
    e.g. `0.79 / 4`. A tiny bar follows, and a `provisional` mark where it
    applies.
  - **Models** show their id, params and judged count, like the phase-9
    search.
  - It replaces the select *and* the "find a topic" box: one control, not
    two.
  - ARIA: the combobox pattern with `aria-activedescendant`.

**Tests:**
- No `select` element inside `#view` on any tab or page.
- Each component's keyboard path works.
- Typing "eco" in the topic combobox leaves Economics as the only option.
- The chosen value survives a poll, via the existing `data-keep` mechanism.

---

## Definition of done — 11f

On the live server, at 1,512 px and at 400 px, in light and dark:

1. **The Leaderboard header** is one line of names under a quiet group row.
   Cells hold only numbers, the best-or-tied cells are bold and tinted, and
   there's no blue slab.
2. **Opening a row** glides open in about 0.2 s and closes the same way. A
   poll never replays it.
3. **Scrolling down any page and opening another** starts the new page at
   the top, and Back returns to where you were.
4. **Motion:** tabs, popovers, toasts and theme changes move smoothly. With
   reduced motion switched on in macOS (System Settings › Accessibility ›
   Display), nothing moves.
5. **Every Queue row** shows one tidy button and a `⋯` menu, and no GPU
   cell wraps.
6. **There are no native dropdowns.** The topic picker is a searchable list
   grouped by area.
7. **All tests are green.** Screenshots are in `tests/_screens/phase11f/`,
   plus a short screen recording (GIF or MP4) of a row opening and closing,
   attached to the PR.
