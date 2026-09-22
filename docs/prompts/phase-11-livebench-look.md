# Brief for Claude Code — the LiveBench look (phase 11)

Start from 1263b2e. There are four PRs. Do them in order, and each must be
green on its own:

- **11a**: five findings from the 37-topic live check (2026-09-22). These
  are small, functional fixes. The shared popover component lands here.
- **11b**: the visual system. Tokens, type, the sticky bar, numbered
  sections, and one table component.
- **11c**: the Leaderboard. Topic-group chips, grouped headers, one-line
  cells with a rank tint, rows that open in place, the popovers, and the
  Insights section.
- **11d**: the Overview highlight cards and a tidy-up of the model page.

## What masein asked for

masein pointed at two public leaderboards:

- LiveBench: https://livebench.ai. This is the reference for the look.
  *"I really liked the style and the feeling of the page"*, especially the
  table with a row opened in place.
- Artificial Analysis: https://artificialanalysis.ai/leaderboards/models.
  Take a few ideas from it.

We implement the ideas ourselves:
- **Copy nothing** from either site: no code, stylesheets, fonts, logos,
  names or wording.
- Take only the general feel and the interaction ideas listed below.

### What gives the reference page its feel (what to reproduce)

1. **A calm page.** A pale blue-grey page, white cards with a hairline
   border, and no shadows.
2. **One accent colour**, a saturated blue. It's used for the active chip,
   links, the section index and the strongest cell tint. Nothing else is
   coloured, except status.
3. **Monospace for data and labels**: column headers, section eyebrows,
   numbers, chips, the status line and the footer. Prose stays in the sans
   face.
4. **A small uppercase, letter-spaced mono eyebrow** above each section,
   with a numbered index (`01`, `02`, …).
5. **A heavy, tight title**: weight 800, with negative letter-spacing.
6. **Numbers in bold mono**, right-aligned, one line per row.
7. **A pale tint by rank in every column**, so the leaders show without
   reading.
8. **A sticky, translucent top bar** with a green `LIVE` badge, plus a
   one-line mono status line: "Showing … · ● live".
9. **Filters as pills.** Secondary controls are small popovers anchored to
   their button.
10. **Rows that open in place.** Click a row and its detail opens right
    under it, on the same page.
11. **An Insights section** under the table:
    - a quality-against-size scatter with a frontier;
    - a ranked bar list;
    - a radar with model chips.

From Artificial Analysis, take these:
- highlight cards, each with a one-sentence verdict;
- filter pills that read "Label: Value ▾";
- grouped two-level column headers with the unit on the second line;
- a thin colour bar per row showing the model's family;
- a back-to-top button.

## Rules that do not bend (all unchanged)

**Report-half isolation.**
- Every judged number on the board, in a chart or in a card is a
  report-half number.
- The diagnose half is never shown as a score.

**A local or uncalibrated judge.** `provisional_reason()` is the test. Its
scores are:
- shown on one model's own page and in that model's opened row, stamped
  *provisional, not ranked*;
- **never** put in any average, whether a judged average or an area mean.
  `judged_avg()` stays as it is;
- **never** a column on the Leaderboard. Judged columns still appear only
  once a person has agreed with the judge (κ over the line), as today;
- **never** tinted by rank, sortable against other models, or put on a
  frontier.
- Sorting one model's own topics weakest-first is allowed; the model page
  already does it.

**Taint, preliminary models, error bars and ties** keep their meaning:
- A tainted topic's score is shown but not ranked.
- A model missing some of the 7 required tasks has no Avg and is not
  ranked. Its per-task cells are valid.
- Every score keeps its standard error.
- The `●` best mark and the `=` too-close-to-call mark come from the same
  z-test as now.

**Nothing silently disappears.**
- Every control in the phase-9 appendix keeps a way to reach it.
- A control removed here is removed in the same PR as the test that used
  it, and the PR description says so.
- `test_every_action_answers.py` is updated to the new controls.

**Themes.**
- auto, light, dark and dim all stay.
- Text contrast is 4.5:1 or better in every theme
  (`test_the_text_reads_in_every_theme`).

**The page stays one file.**
- No web fonts, no CDN and no outside images: it has to work on the team
  server with no internet.
- System font stacks only.

**Polls never close what is open.** `render()` rebuilds the view every
few seconds. Across a poll, all of these stay exactly as they were:
- an opened row;
- an open popover;
- a focused checkbox inside a popover;
- the scroll position.

This is the same class of bug as the tab bar's and the search field's. The
same cure applies: state lives in `state`, not in the DOM.

---

## 11a — five findings from the live check

### 1. The More ▾ menu is clipped

**Seen.** `div.morewrap` sits inside `#tabs`, which has `overflow: auto` so
the tabs can scroll on a phone. The dropdown is cut off to 37 px, so almost
nothing in it can be seen or clicked.

**Fix: one popover component.** It serves More ▾, Theme ▾ and the name menu
(`masein ▾`), and later 11c's popovers.

`popover(anchor, panel, { key, placement: 'bottom-start' })` behaves as
follows:
- **Where it lives.** The panel is appended to `document.body` with
  `position: fixed`.
- **Placement.** It is placed from `anchor.getBoundingClientRect()`:
  - it flips above the button when there's no room below;
  - it stays at least 8 px inside the viewport;
  - it is placed again on scroll and resize;
  - its z-index is above the sticky bar.
- **Opening and closing.**
  - Only one popover is open at a time; opening one closes the others.
  - It closes on an outside `mousedown`, on Tab, and when its button
    scrolls out of view.
- **Keyboard: the menu-button ARIA pattern.**
  - The button carries `aria-haspopup`, `aria-expanded` and
    `aria-controls`.
  - ↓, Enter and Space open the menu and focus the current or first item.
  - ↑ and ↓ wrap; Home and End go to the first and last item.
  - Esc closes the menu and returns focus to the button.
- **Surviving a render.** The panel is keyed by `data-pop`. After every
  `render()`:
  - if a popover is open, it is anchored again to the element with the
    same key, and focus stays where it was;
  - if that element is gone, the popover closes.

**Tests**, at 1,280 px and at 400 px:
- Open More ▾. For every item, `elementFromPoint` at the item's centre is
  the item itself.
- The keyboard path works end to end.
- An outside click closes the menu.
- The menu is still open, with the same item focused, after two polls.
- Theme ▾ and the name menu pass the same checks.

### 2. Generated documents cluster on one sub-area

**Seen.** Dataset #2 (Physics & Astronomy, from SmolLM2-360M-Instruct) has
18 documents. About 11 of the 18 are about relativistic momentum and
collisions, for example:
- "Relativistic Momentum Transfer in Particle Collisions"
- "Relativistic Momentum in a Simple Collision Scenario"
- "Modeling Relativistic Momentum Transfer in Collisions"
- …

Every request in `generation_requests()` is identical except for the style
seed, so a small generator writes the same document again.

The bank already labels every question with a `domain`. Physics & Astronomy
has 12 of them:
- Classical Mechanics (14 items)
- Electricity and Magnetism (13)
- Thermodynamics and Statistical Mechanics (10)
- Quantum Mechanics (10)
- Relativity (8)
- … down to Observational Astronomy, Experimental Physics, and Measurement
  (4)

**Fix: spread the documents over the domains where answers failed.**

- **Count the failures.** For each `domain`, count the graded
  **diagnose-half** items that scored below `WEAK_SCORE`. Look up each
  item's domain in the bank by its qid. The report half is never read, by
  the same rule as `justifications_for()`.
- **Allocate the documents.** Split the requested count across those
  domains in proportion to their failures, by largest remainder. While the
  count allows, every failing domain gets at least one document.
- **Order the requests round-robin** (A, B, C, A, B, C, …), so a batch cut
  short still covers the spread.
- **Add one line to each request:** `Focus: <domain label>`.
  - The line carries the label only.
  - It carries no count, no score, no qid and no question text.
- **A new check, `is_domain_label_set()`.** A topic's `domain` values are
  used only when all of these hold:
  - the topic's bank has **at most 15 distinct values**;
  - each value appears on **at least 2 items**;
  - each value is at most 64 characters and at most 8 words;
  - no value contains `.`, `?` or `!`.

  That makes them a closed set of labels, not a sentence per question.
  Otherwise there is no `Focus:` line, which is today's behaviour.
  - Don't loosen `is_label_set()`. Its no-whitespace rule is right for the
    audience line.
- **Provenance** gets
  `focus_plan: [{"domain", "failing_diagnose", "documents"}]`.
  - The plan appears on the proposal's Review card *before* Generate, for
    example "Documents will cover 9 areas: Relativity 3 · Classical
    Mechanics 3 · …".
  - The dataset card shows it too.

**Tests:**
- Failures in 3 domains with a count of 7 → 3 / 2 / 2 documents, in
  round-robin order.
- A report-half failure in a fourth domain changes nothing.
- A sentence-valued `domain` produces no `Focus:` line.
- No request text contains a failure count.
- Every other promise in the `generation_requests()` docstring still holds.

### 3. Missing documents are not explained

**Seen.** Dataset #2 asked for 20 documents and holds 18:
- `items: {generated: 18, kept: 18}`;
- the gate's `items_in` is 18.

The other two never parsed, and nothing says why:
- `_finish_generation()` collects `res.error`, but reports it only when
  *every* request failed.
- `parse_items()` drops an item silently when:
  - it has no title;
  - its body is under `DOC_MIN_WORDS` (120) words;
  - the reply isn't JSON.

**Fix: account for every request.**

- **Give every request an outcome**, one of:
  - `kept`
  - `error: <message>`
  - `reply not JSON`
  - `no title`
  - `too short (<n> words)`
  - `empty reply`
  - `dropped by the gate: benchmark | exam | duplicate`
- **Provenance** gets:
  - `items.requested`;
  - `items.missing: [{"request": k, "focus": <domain or null>, "why"}]`.

  The invariant: `generated + len(missing) == requested`.
- **The dataset card** says, for example:
  "18 of 20 documents · 2 missing — 1 too short (87 words), 1 reply not
  JSON". A disclosure lists each missing document, and the domain it was
  meant to cover, so the gap in the focus plan shows.
- **The toast** when generation finishes says the same line.
- **No automatic retry** in this PR. The local GPU is shared.

**Tests:**
- The invariant holds across fixtures covering each outcome above.
- The card and the toast carry the line.
- A dataset from before this PR (with no `missing` field) shows "reasons
  not recorded (made before 11a)", not a blank.

### 4. The model search offers `local/` models with no weights here

**Seen.** In #53, `local/qwen35-delta-moe-7d560104-step945` was picked
from the search. It is on the board, because its results came from
elsewhere, but its weights aren't on this server:
- the search offered it like any other model;
- the API accepted it into the queue;
- it then failed at start with "no uploaded artifact named
  'qwen35-delta-moe-7d560104-step945' — upload it first …".

The error is clear, but it arrives after the person has already queued the
run. `suggest.local_candidates()` marks every board model `on_board`
whether or not its weights are here.

**Fix:**
- **Know which ones have weights.** For a `local/` id, set `weights: bool`
  to whether a server artifact of that name exists. The `artifacts` list
  is already passed in.
- **In the search box**, an id with no weights is:
  - listed but greyed, with `aria-disabled="true"`;
  - labelled "results only — weights not on the server".

  Arrow keys skip it.
- **Refuse at submit.** The API refuses such a submission before anything
  is queued, with a 422: "`local/<name>` has results on this board but no
  weights on this server — upload it first (POST /api/artifacts/<name>) to
  run it here." Reuse the runner's wording so the two never disagree.
- **Picked anyway.** If a greyed item is picked from the list, the submit
  form shows that same sentence beside the field, and **Queue this run** is
  disabled.

**Tests:**
- The suggestion carries `weights: false` and is not selectable.
- The API returns 422 with that sentence.
- A `local/` model that does have an artifact is still offered normally.

### 5. "rubrics v?" on the model page

In `report_lm_eval.py` (around line 3510), the judged header builds its
rubric list with `` `v${r.version}` ``. That prints "v?" for rubrics with
no version marker, which is every author-written one.

**Fix.** Use the existing `rubricVersion()` helper (around line 8139). The
header then reads "rubrics: no version", or for a mix, "v2, no version".

**Test.** A model judged with an unversioned rubric shows "no version" and
never "v?".

---

## 11b — the visual system

### Tokens

Light theme, old → new. Contrast ratios are measured against the page
(#F6F8FC) and against white.

| Token | Now | New | Note |
|---|---|---|---|
| `--plane` (page) | #f9f9f7 | **#F6F8FC** | pale blue-grey |
| `--surface-1` (cards, tables) | #fcfcfb | **#FFFFFF** | |
| `--text-primary` (ink) | #0b0b0b | **#14213D** | 15.0 : 1 on the page |
| `--text-secondary` | #52514e | **#3D4B66** | 8.8 : 1 |
| `--muted` | #6f6d68 | **#5A6B85** | 5.1 : 1 on the page, 5.4 : 1 on white |
| `--border`, `--grid` | rgba(11,11,11,.10), #e1e0d9 | **#E4E9F2** | hairline |
| `--axis` | #c3c2b7 | **#C9D2E3** | |
| `--accent` (and links) | #236bc4 | **#2F54EB** | 5.9 : 1 on white, 5.5 : 1 on the page |
| `--accent-soft` | rgba(35,107,196,.10) | **rgba(47,84,235,.08)** | row hover |
| `--live-text` *(new)* | — | **#0B7A5A** | 5.3 : 1 |
| `--live-dot` *(new)* | — | **#12B886** | dot and border only, **never text** (2.6 : 1) |
| `--bar-bg` *(new)* | — | **rgba(246,248,252,.86)** | the sticky bar, with `backdrop-filter: blur(8px) saturate(1.4)` |
| `--heat-1` … `--heat-5` *(new)* | — | `color-mix(in srgb, var(--accent) P%, var(--surface-1))` with P = **3.5, 7, 11.5, 17, 24** | the rank tint |

- **Text in a tinted cell is always ink, never the accent.** Ink on the
  strongest step is 11.1 : 1; the accent on it would be 4.1 : 1.
- **Unchanged:**
  - `--good`, `--critical`, `--warning` and their `-text` variants;
  - the series palette `--s1` to `--s8`, which is checked for colour-blind
    readers.

**Dark theme.**
- Keep the surfaces: #0d0d0d page, #1a1a19 cards.
- `--accent` becomes **#7C9BFF** (6.6 : 1 on the cards).
- Heat steps P = **6, 11, 17, 25, 34**, mixed with the card surface. White
  text on the strongest step is 9.6 : 1.
- `--live-text` #2FD39A; `--bar-bg` rgba(13,13,13,.80).

**Dim theme.**
- Keep the surfaces: #141a26 page, #1c2333 cards.
- `--accent` #7C9BFF (6.0 : 1); heat steps as dark.
- `--live-text` #2FD39A; `--bar-bg` rgba(20,26,38,.82).

### Type

Everything stays on the 12 / 14 / 16 / 20 / 28 scale
(`test_every_piece_of_text_is_on_the_type_scale`).

Two new font stacks:
- `--font-sans`: `system-ui, -apple-system, "Segoe UI", sans-serif`, as now.
- `--font-mono`: `ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace`.

| Piece | Style |
|---|---|
| Page title (h1) | 28 px, weight 800, letter-spacing −0.025em, ink |
| Section title (h2) | 20 px, weight 700, −0.01em, after a mono index (`01`) at 12 px in the accent |
| Eyebrow | mono 12 px, weight 600, uppercase, letter-spacing 0.16em, accent |
| Column header | mono 12 px, weight 600, uppercase, letter-spacing 0.03em, muted, hairline under it |
| Number | mono 14 px, `tabular-nums`; a column's leader in weight 700 |
| Error bar, unit, secondary token | mono 12 px, muted |
| Chip, badge, pill, status line, footer | mono 12 px |
| Prose | sans 14 px, line-height 1.5, at most 72ch |

The reference page's title is 44 px. Ours is a working tool, so the title
stays at 28 px; weight 800 gives the same feel without the banner.

### The pieces

1. **The sticky bar**: 56 px tall, `--bar-bg` with blur, hairline border
   underneath.
   - **Left: the title and badge.**
     - "Team model benchmark" at 16 px, weight 800.
     - The **status badge**: mono 12 px, weight 600, uppercase, a 1 px
       border in `--live-dot`, radius 6, with a dot.
       - It reads `● LIVE · 10:40`: the time of the last refresh.
       - It takes over every state the "live · refreshed" chip has today,
         each with its own tone and words.
       - The dot pulses gently; under `prefers-reduced-motion`, it doesn't.
       - A static report has no LIVE badge.
   - **Middle: the tabs**, as today.
     - The active tab has a 2 px accent underline.
     - The tabs scroll sideways on a narrow screen. With 11a's popover,
       More ▾ is no longer clipped by that.
   - **Right, in order:**
     - **The checks pill**, e.g. "● 6 checks" with an amber dot. It opens
       the same checks list as today, just under the bar. The line is
       still on every tab and still never dismissed; it is now in the bar.
     - `masein ▾`
     - `Theme ▾`
   - **At 720 px or less**, the title shortens to "Benchmark" and the tabs
     move to a second row inside the same sticky block.
2. **The hero, on Overview only.**
   - An eyebrow: `TEAM BENCHMARK · 37 TOPICS · 33 MODELS`.
   - The h1, the one-line description, and the two guide links as mono
     pills.
   - When the page is live, **Submit a model** is the primary button here.
   - Every other tab starts straight at its first numbered section.
3. **Section headers.** Each has:
   - the mono index and the h2;
   - a one-line muted description;
   - its actions, aligned right.

   The cards on each tab are numbered in order. On Overview: `01`
   Highlights, `02` Top models, `03` The loop, and so on.
4. **Cards.** White, 1 px border, radius 10, padding 24 (16 at 720 px or
   less), no shadow.
5. **A status line** above every big table: mono 12 px, muted. For example:
   "Showing 1–25 of 32 models · 15 ranked · sorted by Avg ▼ · ● live 10:40".
6. **The footer**: mono 12 px one-liners, e.g. "Every score carries its
   standard error · gaps are z-tested before they are called wins · build
   1263b2e".
7. **Back to top.** A mono pill, "↑ Top", bottom right.
   - It appears after two screens of scrolling.
   - It moves focus to the bar.
   - It is hidden when printing.
8. **One tooltip component**, used for the ⓘ column help and for cell
   hovers.
   - Colours: background `--text-primary`, text `--surface-1`, so it
     inverts in every theme.
   - Mono 12 px, radius 6, at most 280 px wide.
   - It shows on hover **and on focus**, and is wired with
     `aria-describedby`.
   - Native `title` attributes stay everywhere else.

### One table component

It covers every table: Leaderboard, Models, Queue, Top models, the topic
tables and the answers tables.

**Header row.**
- Mono uppercase, as in the type table.
- Sticky inside the table's own scroller, as `test_one_table_component`
  requires.

**Rows.**
- At least 44 px tall (the test requires at least 39.5), padding 10 × 12.
- Hairline separators.
- The hover background is `--accent-soft`.
- An opened or selected row has a 3 px accent bar on its left.

**Cells.**
- Numbers are right-aligned, mono, `tabular-nums`. Text is left-aligned.
- At 400 px, the table scrolls sideways inside its own scroller, with the
  model column pinned on the left.

**Removed: the comfortable / compact density toggle.** The one-line row is
the compact one. Update `test_every_action_answers.py` (line ~312), which
clicks "compact".

**Tests for 11b:**
- **Tokens.** Every token's computed value is correct in each theme.
- **Unchanged tests still pass:**
  - the contrast test, with the new tokens;
  - the type-scale test;
  - `test_one_table_component`.
- **Mono** is used on `th`, number cells, eyebrows and chips.
- **The sticky bar** is at most 56 px tall at 1,280 px and stays at the top
  after scrolling 3,000 px.
- **The badge** is there only when the page is live.
- **Reduced motion** stops the pulse.
- **Section indices** run in order on every tab.
- **Back to top** appears after scrolling and moves focus to the bar.
- **Screenshots** of Overview, the Leaderboard and a model page:
  - at 1,280 px and at 400 px;
  - in light, dark and dim;
  - saved to `tests/_screens/phase11/` and attached to the PR.

---

## 11c — the Leaderboard

### One toolbar row instead of four

Today there are four stacked control rows: Avg scale, Cells, View, and
Columns / Rows. They become one row.

**Left: topic-group chips.** These are mono pills. The active one is filled
with the accent and has white text.

`All tasks` · `Knowledge` · `Commonsense` · `Reasoning` · `Math` · `Truthfulness` · `Judged topics`

- **The groups** are the existing ones (around line 4520):

  | Group | Tasks |
  |---|---|
  | Knowledge | mmlu |
  | Commonsense | hellaswag, piqa, winogrande |
  | Reasoning | arc_challenge, arc_easy |
  | Math | gsm8k |
  | Truthfulness | truthfulqa_mc2 |

  **All tasks** is today's view.
- **A chip shows its group's columns.**
- **Knowledge** also shows **MMLU by area**: eight columns, one per topic
  area (see *Areas* below).
  - Each column is the mean over the MMLU subjects mapped to that area's
    topics, weighted by item count.
  - All eight areas have at least one subject.
  - The 24 per-topic MMLU columns stay reachable under Choose columns.
  - This replaces today's "View: tasks | MMLU by category" switch.
- **Judged topics** shows the judged columns grouped by area. It follows
  exactly today's rule: **only once the judge is calibrated.**
  - Until then, the chip is disabled and says why, e.g.: "judged columns
    appear once a person has agreed with the judge — today: not
    calibrated, local judge". It follows
    `test_a_disabled_button_looks_disabled_and_says_why`.
  - Once the judge is calibrated, the area columns show area means of the
    judged topics. They are tinted and sortable like any other column.
  - **An area mean** follows `judged_avg()`'s rules:
    - tainted topics are left out;
    - it needs at least half the area's topics judged. Otherwise the cell
      is "—", with "k of n topics judged" in its tooltip.
  - Test both states with fixtures.

**Right: filter pills** in the "Label: Value ▾" form.
- `Kind: All ▾` (base / instruct / checkpoint)
- `Size: All ▾` (< 200M, 200M–1B, 1–3B, > 3B)
- `Status: All ▾` (ranked / preliminary / tainted)
- Then:
  - `Columns · 4 hidden ▾`
  - `Models ▾`
  - `Scale: above chance ▾`

The Models tab's filters become the same pills.

**The popovers** all use 11a's component.
- **Choose columns**
  - A checklist of the current chip's columns, with **Show all**.
  - Changes apply at once.
  - It also holds a **Tint: on / off** switch, remembered in this browser.
- **Models**
  - A search box and a checklist of models, each with its family colour
    dot. This list is the legend for the family bar.
  - **Select all** / **Clear**.
  - A footer that reads "All models shown" or "12 of 32 shown".
  - **Apply**.
- **Scale**: above chance or raw accuracy, as today.

**The view lives in the URL hash**, so a pasted link reproduces it, e.g.
`#tab=leaderboard&chip=knowledge&kind=instruct&open=<id>`. Old hashes keep
working.

**Removed: the "Cells: numbers | heat" switch.** The rank tint below
replaces the old heat. Update any test that clicks it.

### Grouped headers

There are two header rows, both sticky:
1. The group names (Knowledge, Commonsense, …), each spanning its columns.
2. The task names. The unit sits on a muted second line inside the same
   cell, e.g. `5-shot · %`.

### Cells

- **One line per cell.** `52.3` in bold mono 14 px, then `±0.4` in mono
  12 px, muted. The `%` lives in the header, not in every cell. The row is
  one line, about 44 px, instead of today's 73.
- **The rank tint.**
  - In each column, the model's rank among all rows that have that cell
    sets its step, from `--heat-5` (top fifth) down to `--heat-1`.
  - The rank is taken over the **whole board**, like `#n/of`, so filtering
    never changes a colour.
  - Cells tied with the column's best by the existing z-test get
    `--heat-5` and bold.
  - Judged cells are tinted only when the judge is calibrated.
- **The marks.** `●` (best) and `=` (too close to call) stay, to the right
  of the number.
- **A new first column, `#`**, in mono and muted:
  - 1–15 for ranked models;
  - "—" for preliminary ones.

  The "#1/14" under Avg goes.
- **Params** stays one line, e.g. `2.3B · 908M act` for a mixture-of-experts
  model.
- **The family bar**: a 3 px bar on the left edge of the model cell,
  coloured by family.
  - The family is the Hub organisation for Hub ids. For local checkpoints,
    it is the run name without its `-step<N>` suffix.
  - Families take `--s1` to `--s8` in alphabetical order. Any beyond eight
    get `--axis` grey.
  - Colour is never the only carrier: the family name is in the tooltip,
    in the opened row and in the Models popover.
- **Badges** stay, in mono 12 px: base, instruct, ckpt, prelim 2/7, and
  duplicates.

### Rows that open in place

This is the reference screenshot.

**How a row opens.**
- Clicking anywhere on a row opens a detail row directly below it,
  spanning every column. Links, checkboxes and badges with their own
  action keep their own clicks.
- A ▸ disclosure button sits in the `#` cell, with `aria-expanded` and
  `aria-controls`. Enter and Space toggle it.
- Several rows may be open at once.
- Open rows are kept in `state` by model id and in the hash (`open=`).
  They survive polls, paging and re-sorting.
- An open row has the accent bar and an `--accent-soft` background.

**What the detail row holds:** a grid of small blocks, each with a mono
eyebrow.
- **TASKS**
  - Every task, with its score ± error, its n-shot and its rank ("#3/14").
  - "tied with best" where the z-test says so.
- **MMLU BY AREA**, when per-subject results exist.
  - Eight mini bars, above chance.
  - Hovering a bar lists its topics.
- **JUDGED TOPICS**, when the model has been judged.
  - The topics as mono chips, grouped by area, weakest first, e.g.
    `Economics 0.79`. Report half only.
  - While the judge is provisional:
    - the chips are neutral grey;
    - the line "provisional — local judge, not calibrated · not ranked"
      appears **once**;
    - **no area means** are shown. `judged_avg()`'s rule covers every
      average.
- **LINKS**
  - Open model page →
  - Provenance →
  - Add to radar
  - Run exam (only when the page is live and the weights are on the
    server)

At 400 px, the blocks stack in one column.

### The long paragraph

The paragraph at the top of the Leaderboard today goes behind **"How to
read this table ▾"**.
- It is collapsed by default, and its state is remembered.
- It merges with "About these benchmarks".

The status line from 11b takes its place.

### Areas

The 37 topics are grouped into 8 areas.
- They live in a new file, `scripts/areas.yaml`.
- The file has the same shape rules as `categories.yaml`: `Area:` lines
  and `  - Topic` lines only, read without pyyaml.
- `tests/test_areas.py` checks that:
  - every one of the 37 topics is in exactly one area;
  - every name matches `categories.yaml` exactly.

The grouping below is my proposal, and masein can change it.

| Area | Topics |
|---|---|
| Mathematics & Science | Mathematics & Statistics · Physics & Astronomy · Chemistry & Materials Science · Biology & Life Sciences · Earth & Environmental Sciences |
| Computing & Technology | Computer Science · Software Engineering & Programming · AI & Machine Learning · Systems & Cybersecurity · IT · Technology · Data & Information Science |
| Engineering & Applied | Engineering · Manufacturing & Applied Sciences · Architecture & Built Environment · Design · Agriculture · Food & Veterinary Sciences |
| Health & Mind | Medicine & Clinical Health · Public Health & Wellness · Psychology & Cognitive Sciences |
| Society | Economics · Sociology · Anthropology & Human Geography · Political Science & International Relations · Government & Public Policy · Education |
| Humanities & Arts | History & Archaeology · Philosophy · Ethics & Religion · Language & Literature · Arts · Media & Communication |
| Business & Law | Law · Business & Management · Finance & Accounting |
| General | General & Multidisciplinary |

That is 5 + 7 + 6 + 3 + 6 + 6 + 3 + 1 = 37.

### Insights (section 02 on the Leaderboard tab, under the table)

There are three charts, drawn as inline SVG like the existing charts, in the
series palette.
- Every point and bar can be reached by keyboard and shows the dark
  tooltip.
- Each chart has **Show as table** as its text alternative.

**1. Score against size.**
- Ranked models only. x is parameters on a log scale; y is Avg above
  chance.
- A model is **on the frontier** when no smaller model beats it by a
  z-tested gap: the same test as the rest of the board, so noise never
  draws the line.
- The frontier is drawn as a dashed line and labelled.
- **Clicking a point** highlights it and shades the region of models that
  are bigger and score lower, captioned e.g. "4 models here are bigger and
  score lower than Qwen3.5-0.8B-Base". Esc, or a click on empty space,
  clears it.
- Preliminary models aren't plotted. The caption says how many, e.g. "18
  preliminary not shown".

**2. Weakest topics.**
- One judged model's topics, weakest first, as bars on the 0–4 scale. This
  orders one model's own topics; it doesn't rank models.
- By default, the model is the one the loop is on. A model chip with
  **Change model…** opens a search of judged models.
- While provisional, the bars are grey-hatched and the caption says
  "provisional".
- Each bar links to "Open the topic →".
- With nothing judged, the chart says: "No model has been judged yet —
  Loop ▸ Sit the exam".

**3. The radar.**
- Up to 5 model chips, added with **Add a model…**.
- A source switch:
  - `Tasks`: 7 axes, today's capability profile;
  - `MMLU by area`: 8 axes;
  - `Judged by area`: 8 axes. This option is disabled, with its reason,
    until the judge is calibrated, because it would be an average.
- This **replaces** both the "Capability profile" card and the "compare"
  checkbox column: the chips are the comparison. Update
  `test_the_leaderboards_compare_ticks_are_the_only_ones` in
  `test_dashboard.py` (line ~659) to the chips.

**Tests for 11c:**
- **Chips:**
  - each chip shows exactly its group's columns;
  - Judged topics is disabled with its reason while provisional, and live
    with a calibrated fixture.
- **Rank tint:**
  - the step follows the whole-board rank, and filtering doesn't change it;
  - ties with the best share the top step;
  - no provisional cell is ever tinted.
- **Rows:**
  - one line, at least 39.5 and at most 48 px tall;
  - an opened row survives two polls, a page change and a re-sort, and
    comes back from a pasted hash;
  - opening works by keyboard.
- **Popovers** (Columns, Models, the filter pills) stay open across a poll.
- **The frontier** ignores gaps inside the noise. The fixture: two models,
  the smaller one ahead within the noise → the bigger one is still on the
  frontier.
- **The radar's** judged source is disabled while provisional.
- **areas.yaml** passes its test.
- At 400 px, the model column is pinned and nothing overflows the page.

---

## 11d — Overview and the model page

### Overview

- **The hero** is as in 11b.
- **The four stat cards** (Models compared, Tasks, Real differences, Eval
  wall-clock) become one mono line under the hero, e.g. "33 models · 10
  tasks · 1,091 of 1,538 gaps are real · 16.4 h of evaluation ·
  2026-08-18 → 2026-09-22".
- **`01` Highlights**: four cards. Each has an eyebrow, a value in mono at
  28 px, a **one-sentence verdict** and a link. Cards 2–4 appear only when
  the page is live.

  | Card | Value | Verdict (example) | Links to |
  |---|---|---|---|
  | BEST MODEL | qwen35-delta-moe…-v2 · 39.3 | "Leads Qwen3.5-0.8B-Base by 0.7 points — within noise." The verdict comes from the z-test. | Leaderboard |
  | WEAKEST TOPIC | Economics · 0.79 / 4 | "SmolLM2-360M-Instruct, 7 of 37 topics judged — provisional, local judge." One model's own topic, so it isn't a ranking; no area mean. | the topic page |
  | THE LOOP | 7 / 37 topics judged | "Last judged SmolLM2-360M-Instruct 20 min ago." | Open the Loop |
  | JUDGE STEADINESS | 30 / 30 steady | "30 fixed scripts re-graded: 0.17 from the human marks, 0.03 from the last run (limit 0.5). Not calibrated yet." | Provenance |

- **"Best official average"** folds into the BEST MODEL card.
- **"Biggest statistically real gap"** becomes a one-line mono callout
  under Top models.
- **`02` Top models** uses the table component, with the tint and `#`, and
  "See the leaderboard →".
- **`03` The loop** is as today, in the new style.

### The model page

**The hero.**
- An eyebrow, e.g. `MODEL · INSTRUCT · #6 OF 14`.
- The name as h1, at 28 px, weight 800.
- The Hub id in mono.
- Three highlight cards: Parameters, Average above chance (with its
  verdict), and Tasks 7/7.
- The prose paragraph stays underneath, muted, at 14 px.

**The section nav.** Judged · Earlier exams · Results · Diagnose ·
Provenance · Runs become sticky mono chips under the bar.
- They are numbered `01` to `06`.
- The section in view is highlighted as you scroll, using
  IntersectionObserver.

**Judged, by topic.**
- Topics are grouped under the 8 areas. Each area header shows its name
  and "k of 6 topics judged".
- There is **no area mean while provisional**.
- Inside each area, topics run weakest first.
- The note "the judged suite is preliminary — the topic page says what
  proposing would mean" appears **once**, above the table. Today it
  appears on every row.
- The row's action becomes a small **Propose →** text button. Its behaviour
  and dialog are unchanged.

**Score against answer length.**
- Today this table has one row per topic *per length bucket*: 28 rows for
  7 topics, which would be 148 rows for 37.
- It becomes one row per topic, with four columns: ≤ 20, 21–50, 51–120 and
  > 120 words.
- Each cell reads `0.87 · 86` (mean · items).
- Cells get a neutral grey tint by the mean on the absolute 0–4 scale. It
  is not a rank tint, because these scores are provisional.
- Cells with fewer than 5 items are muted and marked "few".
- The sentence above the table stays.
- With more than 10 topics, the table starts collapsed behind **Show the
  table**.
- Apply the same change to the topic page's matching table, if it has one.

**Tests for 11d:**
- Each highlight card's verdict is derived from the data, checked on
  fixtures: a lead within the noise, a lead outside it, nothing judged, and
  not live.
- The preliminary note appears exactly once on the model page.
- The length table has one row per topic.
- No average of provisional scores appears anywhere on the page. Assert
  this on a provisional fixture.
- The section nav highlights the section in view.

---

## Testing notes

- **No test depends on timing**, or on which of several same-second writes
  wins:
  - fixtures use fixed timestamps;
  - pollers are paused in browser tests unless the test is *about* a poll;
  - a test about a poll triggers the poll itself and waits for its
    completion event, never for a number of seconds.
- **Screenshots** are named `<tab>-<width>-<theme>.png` in
  `tests/_screens/phase11/`. Attach them to each PR from 11b on.

## Definition of done

On the live server, after the deploy steps below:

1. More ▾ opens fully at 1,280 px and at 400 px, and Esc returns focus to
   the button.
2. A new Generate, on the proposal masein makes live in the demo
   (Economics is kept free for it):
   - shows its focus plan on the Review card;
   - spreads the documents over at least 5 domains;
   - explains any missing document on the card and in the toast.
3. The search shows `local/qwen35-delta-moe-7d560104-step945` greyed as
   "results only", and submitting it by API returns the 422.
4. The Leaderboard has:
   - the pale page, a white card, mono headers, tinted cells, and one-line
     rows of about 44 px;
   - one toolbar row;
   - rows that open in place;
   - popovers that survive a poll.
5. The Judged topics chip is disabled and says why. SmolLM2-360M-Instruct's
   opened row shows its 7 topics as grey chips, with the provisional line
   once and no area mean.
6. Insights shows the frontier chart, Weakest topics and the radar.
7. Overview shows the hero, the one-line stats and four highlight cards
   with verdicts.
8. Light, dark and dim all look right (screenshots in the PR), and every
   test is green.

**Deploy steps**, for masein, after each PR merges. This phase has no data
migration: it changes code and adds `scripts/areas.yaml`.

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.

## Where each idea came from (for the PR description)

| Idea | From | Ours |
|---|---|---|
| Pale page, white hairline cards, one accent | LiveBench | 11b tokens |
| Mono headers, eyebrows, numbered sections | LiveBench | 11b type and section headers |
| Sticky translucent bar with LIVE | LiveBench | 11b bar; the badge takes over today's states |
| Rank tint per column | LiveBench | 11c; z-test ties share the top step; never on provisional scores |
| Category chips | LiveBench | 11c chips over our task groups and areas |
| A row that opens in place | LiveBench | 11c detail row |
| Choose-columns and compare popovers | LiveBench | 11c Columns and Models popovers; the radar chips |
| Scatter with a frontier; "bigger and worse" zone | LiveBench | 11c, z-tested; captioned in plain words |
| Radar with model chips | LiveBench | 11c, replacing the capability profile |
| Highlight cards with verdicts | Artificial Analysis | 11d Overview and model hero |
| "Label: Value ▾" filter pills | Artificial Analysis | 11c toolbar and the Models tab |
| Two-level headers with units | Artificial Analysis | 11c |
| A colour bar per model family | Artificial Analysis (maker bars) | 11c family bar; never colour alone |
| Back to top | Artificial Analysis | 11b |
