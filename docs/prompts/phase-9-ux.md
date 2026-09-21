# Brief for Claude Code — a dashboard people like using (phase 9)

Do after 20b68b9. Four PRs, in this order, each green on its own:

- **9a — the five things Omar asked for**: model search, the answers table,
  pages that outlive a deploy, Propose with a warning, pagination
- **9b — the first screen and the way around**
- **9c — every action answers, every table fits**
- **9d — one visual system**

## Why

Omar, 2026-09-21: *"the ui and ux i still don't like it … i want the users to
love to work with this dashboard."* I walked every tab on the live server
(472d5a5: 49 submissions, 33 models, five topics judged for two models) and
clicked every control. The page is **correct**: the numbers, the split and the
gates all hold. But it reads like an audit log:

- the first screen of Overview and Leaderboard is six paragraphs of warnings;
- the answers table is 2,474 px wide in a 1,234 px container;
- a disabled button looks exactly like an enabled one;
- the same "your name" box appears on five panels, and is empty on one of them;
- four different places start a proposal;
- 49 queue rows, 46–59 answers and 33 models all come in one long table.

The appendix lists every control, what it does now and what it should do.
Each numbered item below is small on its own.

**Rules that do not bend**, unchanged from every brief before this one:
- the report half never appears as a row, a question, an answer or a qid,
  anywhere;
- provisional and preliminary stamps stay on everything they stamp;
- taint follows data into the models trained on it.

The Propose override in 9a keeps all three.

**Read first:** `scripts/report_lm_eval.py`:
- rendering: `render()`, `renderTabs()`, `renderWarnings()`, `mkSel()`
- the topic page: `vTopic()`, `loopSitPanel()`, `loopAnswersPanel()` /
  `ansRow()` / `ansStrip()`
- polling and identity: `loadQueue()` / `refreshResults()`, `rvNameInput()` /
  `rememberedName()`
- the gate: `topic_gate()`, `provisional_reason()`

Then `service/app.py`: `POST /api/proposals` and `/api/proposals/{pid}/generate`.
Then `service/proposals.py`.

---

## 9a — the five asks

### 1. Model search in every model-id box

Where: the topic page's **Sit the exam** and Submit & Queue's **Submit a
model**.

- **When it opens.** After 2 characters, debounced 200 ms, a suggestion list
  opens under the box.
- **What it lists, in order:**
  1. models this board already knows: board rows, every `hf_id` in the
     queue, uploaded artifacts (`local/…`);
  2. then Hugging Face Hub matches.
- **Hub search goes through the service**, never from the browser:
  - `GET /api/models/suggest?q=` calls
    `HfApi().list_models(search=q, sort="downloads", limit=10)`, restricted
    to text-generation;
  - results are cached 10 minutes per query, with a 2-second timeout;
  - if the Hub doesn't answer, return the local matches alone and put
    "Hub search unavailable" in the list's footer;
  - typing is never blocked.
- **What each suggestion shows:** the id, plus what is known:
  - params;
  - kind (from the board, or guessed from `-Instruct` / `-it` in the id);
  - "on the board";
  - "judged on N topics";
  - "over the size cap" when the Hub reports a safetensors size above it.
    Show it greyed but still pickable, so preflight explains why.
- **Matching:** case-insensitive substring on the whole id. "hugg" matches
  `HuggingFaceTB/…`, "smol" matches SmolLM2, "360" matches
  SmolLM2-360M-Instruct.
- **Keyboard:** ↑/↓ moves, Enter picks, Esc closes, Tab accepts the
  highlighted item. Use the ARIA combobox pattern (`role="combobox"`,
  `listbox`, `aria-activedescendant`).
- **Picking** fills the field and sets **kind** when it is known.
- **Polls:** the field keeps `data-keep`, so a 5-second poll mid-typing
  loses neither the text nor the open list.

**Tests:**
- "hugg" lists the board's HuggingFaceTB models before any Hub result;
- a keyboard pick fills the box and sets kind;
- with the Hub mocked to fail, local matches still return, with the footer;
- a poll re-render mid-typing keeps text, caret and the open list.

### 2. The answers table fits: no sideways scrolling

**What's wrong.** On the medicine topic page at a 1,512 px viewport, the
answers table (`data-answers-table`) is 2,474 px wide in a 1,234 px
container. The criteria strip's bars render about 170 px wide each (the
`dxbar` class overrides the inline 8 px), so 15 bars take the width of two
screens. That pushes **what the judge wrote**, the most useful text on the
page, off to the right.

**The fix.** Replace the table with a list of answer cards. Keep
`data-answers-table` on the list, and `data-answer` and
`data-half="diagnose"` on each card, so the leak tests still find them.

```
┌──────────────────────────────────────────────────────────────┬───────────┐
│ urgent · difficulty 3                                          │   0 / 4   │
│ Q  I accidentally took two doses of my blood pressure …        │ Critical  │
│ A  To take two doses of your blood pressure medicine …         │ safety    │
│    (3 lines, "Show the whole answer")                          │ failure   │
│ Judge  The answer does not tell the person to … (in full)      │           │
│ Criteria ■■■■■■■■■■■■■■■  weakest: safety 0.0 · red flag       │           │
│          coverage 0.0 · triage 0.3                             │           │
└──────────────────────────────────────────────────────────────┴───────────┘
```

- **Columns.** The left column is flexible. The right column is a fixed
  140 px, with the score large and flags as chips under it.
- **Criteria.**
  - One row of fixed 14 × 14 px cells: at most 23, so at most about 350 px.
  - Colour on a sequential 0 → 1 scale from the theme's tokens, colour-blind
    safe.
  - Each cell's `title` and `aria-label` give the criterion and its value.
  - After the cells, the three weakest criteria **named in words**. The
    words are what people read; the cells are the overview.
- **The judge's note** is shown in full, labelled, under the answer.
- **Filters.** Acuity, flags, score and sort stay. Add **criterion below
  0.5**: it was in the 8e spec and never built.
- **Widths.**
  - At 1,280 and 1,512 px: `scrollWidth <= clientWidth` for the list and for
    the page.
  - Under 800 px, the right column drops below the text.

**Tests:**
- the width assertion at both widths on medicine (15 criteria) and law (23);
- per-cell tooltip text;
- the report-half leak test, unchanged.

### 3. A page left open across a deploy runs the old code forever

**What happened.**
- Omar ran #49 (SmolLM2-135M-Instruct on medicine), and the answers panel's
  model select on his open tab listed only SmolLM2-360M-Instruct.
- The server had the result: at 12:33, `/api/results` listed medicine for
  both models.
- A tab reloaded after the 11:30 deploy picked #49 up on its own and listed
  both.
- His tab had been loaded before #33 shipped the judge-finished refresh, so
  it was running the old page code. A page like this never reloads its own
  JavaScript.

**The fix:**
- Stamp the served page with the build: the git short sha, known to the
  image or computed at startup, in `<meta name="evalboard-build">`. Return
  the same value on every `/api/*` response as `X-Evalboard-Build`.
- When `api()` sees a header that differs from the page's build, show a bar
  across the top: **"The dashboard was updated. Reload to get the new
  version."** with a **Reload** button.
- If nobody is typing and no dialog is open, reload by itself after 60 s,
  keeping the hash. Never reload while a text field has focus.
- The answers panel's model select says what each model has on this topic:
  "SmolLM2-135M-Instruct — 2.06 / 4". When a new judged model appears on an
  open page, the select gets a one-time highlight: "new:
  SmolLM2-135M-Instruct".

**Tests:**
- a page built at sha A talking to an API at sha B shows the bar;
- no automatic reload while a field has focus;
- extend the #33 browser test: after a judged run lands on an open topic
  page, the select lists both models.

### 4. Propose works, with a warning, when the judge is provisional

Omar wants to demo the whole loop on the live board with today's local judge.
The gate stays, but it splits in two:
- reasons about the **judge** become overridable, with a warning;
- reasons about the **data** stay hard.

**The gate:**
- Today `topic_gate()` returns on the first preliminary reason and **never
  evaluates the data checks**. Evaluate both, always.
- **Soft** reasons (overridable): not calibrated against a person (κ);
  provisional (local judge); single-provider loop; draft rubric.
- **Hard** reasons (never overridable):
  - under the report-half floor (`PROPOSE_MIN_N`);
  - the model wrote nothing usable;
  - collapsed output;
  - the judge file is `skipped`;
  - no judged answers on file;
  - a proposal for this model and topic is already active;
  - the judge wrote no justification to propose from.
- The gate's shape becomes `{ok, overridable, soft: [...], hard: [...], why,
  short, caution}`:
  - `ok` means no soft and no hard reasons;
  - `overridable` means soft reasons only.

**The button:**
- `ok`: **Propose**, a primary button.
- `overridable`: **Propose…**, a secondary button, enabled, with an amber
  dot. Clicking opens an in-page dialog, not `window.confirm`:
  - Title: **This judge is a small local model**
  - Body: the soft reasons as a list, in words. Then: "Anything proposed from
    these grades is marked **proposed over a provisional judge**: the spec,
    the dataset, and any model trained on it. Use it for demos and trials,
    not for results."
  - The name, pre-filled from the header (see 9b-4).
  - A checkbox, **I understand these grades are not evidence**.
  - **Cancel** and **Propose anyway**; the second stays disabled until the
    box is ticked.
  - Esc cancels. Focus stays inside the dialog, and returns to the button
    when it closes.
- `hard`: disabled, visibly (see 9d), with the reason beside it in words.

**The API:**
- `POST /api/proposals` accepts `override_preliminary: true`.
- The server recomputes the gate every time:
  - any hard reason: 409 with the hard reasons;
  - soft reasons and no override: 409, as today;
  - soft reasons with the override: accepted. Record
    `override = {"by", "at", "reasons"}` on the proposal row (a nullable JSON
    column, via an additive migration) and log one line.

**The mark travels with everything made from it:**
- the proposal (Review tab card, topic page panel);
- the approved spec;
- the dataset's `provenance.json`:
  `"proposed_over_provisional_judge": {"by", "at", "reasons"}`;
- the taint trail of any model trained on that dataset.

Each of those places shows an amber badge, **over a provisional judge**, with
the reasons in its tooltip.

**Setting:** `ALLOW_PRELIMINARY_OVERRIDE` in `.env`:
- default `1`, so it works today;
- `0` restores the hard gate exactly as it is now: the button disabled, the
  old words.

Document it in `SERVICE.md` and `.env.example`.

**Approve and Generate** work as they do today, and carry the mark.

**One entry point.** Propose lives on the **topic page**, and the Loop board's
next-step button opens it. The model page's per-topic **Propose a skill spec**
and the Review tab's per-model cards become links to the topic page. Today four
places start the same action, with four copies of the gate logic in the page.

**Tests, on a fixture with a local judge:**
- soft reasons only → the dialog → a proposal is created, carrying
  `override`;
- soft reasons plus under the floor → a hard refusal, and no dialog;
- `ALLOW_PRELIMINARY_OVERRIDE=0` → 409, as today;
- the mark is present in a generated dataset's `provenance.json`;
- the recorded-request-body leak tests are unchanged.

### 5. Pagination: one component, every long table

- **The control.** One shared `pager()` renders "1–25 of 49" · a page-size
  select (10 / 25 / 50 / 100) · **‹ Prev** · page numbers (at most 7, with an
  ellipsis) · **Next ›**.
- **State.**
  - The page size is remembered per table in `localStorage`. It's a
    convenience: without storage it defaults to 25.
  - A filter or sort change goes back to page 1.
  - The current page lives in `state`, so a 5-second poll never throws you
    back to page 1.
- **Where, and page size:**
  - the answers list: 25;
  - the Submit & Queue table: 25;
  - Models: 25;
  - Leaderboard: 25, ranked rows first;
  - Provenance: 25;
  - the Training runs list: 20.
- **Accessibility.** Page controls are real buttons, and the current page has
  `aria-current="page"`.

**Tests:**
- 49 queue rows make two pages;
- filtering to done returns to page 1;
- a poll while on page 2 stays on page 2.

---

## 9b — the first screen and the way around

1. **The checks become one line, on every tab.**
   - Today they are expanded on Overview and Leaderboard and fill the first
     screen.
   - The new bar: **6 checks · 3 about the judged suite [Show]**.
   - Expanded, each check is one line with a severity dot (warning or info),
     a **Show me** link to the rows it concerns, and its long text behind a
     disclosure.
2. **Overview.**
   - (a) The hero, Top models and "Biggest statistically real gap" skip rows
     marked duplicate. Today the hero features `…step945`, the run the
     Leaderboard marks "duplicate of …-v2", with a sentence that contradicts
     itself: *"It has 7 of 7 required tasks, so it is preliminary and carries
     no overall rank."* Top models lists both qwen35 rows as #1 and #2.
   - (b) Replace the run-on line *"18 preliminary models (pythia-31m 2/7,
     demo2-lr3e4-step150 2/7, …)"* with **18 preliminary models — see
     them**, linking to Models filtered to preliminary.
   - (c) Add a **Loop** card: topics judged, each judged model's weakest
     topic with a button to it, and the last judged run and when. The
     newest and most important work has no presence on the first tab.
   - (d) Put a **Submit a model** button in the hero row.
3. **Tabs: from 11 to 6.** **Overview · Loop · Models · Leaderboard · Queue ·
   More ▾**. More holds Exam, Review, Training, Tasks, Perplexity & Loss and
   Provenance. Exam and Review are steps of the loop and are reached from
   it. Every old hash keeps working as an alias, as 8e did.
4. **Who you are, once.**
   - The "your name" box is on the Loop board, the topic page, Review, the
     Exam import and Submit. **Submit's is empty** although the others are
     filled; 8g-D #3 missed it.
   - Move it into the header as **masein ▾** (click to change). Every action
     reads it, and the per-panel boxes go.
   - On a first visit with no name, a small prompt appears in the header,
     rather than a refusal after the first click.
5. **The Loop board shows one model.**
   - Today the "last judged" column shows whichever model ran last. After
     #48 and #49, economics reads 0.44 and medicine 2.06 (135M) while law,
     computer science and physics show 360M: one column, two models.
   - Add **Results for: [model ▾]** at the top of the board. The default is
     the model with the most judged topics.
   - Every score on the board is that model's. A topic it hasn't sat says
     **Not sat — Sit the exam**.
6. **The next step is the same for everyone.**
   - Today **Read the results** versus **Propose** depends on a per-browser
     "read" flag in `localStorage`: physics shows Read the results, the
     other four show Propose. Two people get two different next steps.
   - Drop the read flag from the step. The steps are Import → Sit → Propose
     → Review → Generate → Train.
7. **Empty topics fold away.**
   - Ten of the fifteen topics have no questions, and they fill two thirds of
     the Loop board, the Exam rubrics table and **By topic**.
   - Show the topics that have questions, then one row: **10 topics without
     questions: mathematics, chemistry & biology, … [Import a bank]**, which
     expands to the full list.
8. **Buttons carry their context.**
   - **Import a bank** on the mathematics row opens the Exam tab with the
     topic empty. It must open the import panel with mathematics selected
     and the file picker focused.
   - **Read the results** opens the topic page scrolled to the answers.
   - **Sit the exam** opens the sit panel with that topic ticked and focus
     in the model box.
9. **The product's words, not the shell's.** Remove from the page:
   - shell commands, e.g. the Exam tab's `python3 scripts/exam_build.py
     --root … import …`;
   - container paths ("Uploads are written to /app/eval_tasks/fr/rubrics");
   - sha prefixes in tables: keep them in Provenance and in tooltips;
   - "exam.md (fallback) single score", ten times over;
   - "Rebuild the harness tasks from the bank" (see 9c-4).

   Keep the long explanations, but behind an **ⓘ** popover on each card's
   title, not as a paragraph above every card.
10. **Freshness you can see.** "live · refreshed 12:33" gets:
    - a green dot while polls succeed;
    - amber with "last update 3 min ago — retrying" when they fail.

---

## 9c — every action answers, every table fits

1. **Feedback.**
   - Every action that changes something shows a toast: bottom-right, 4 s, a
     polite live region, with a link to where the thing went. Examples:
     "Queued #49 — medicine & health", "Imported 100 questions", "Proposal #3
     requested".
   - Errors stay inline, next to the control that caused them.
   - The "#49 queued" line under Sit the exam goes away once the toast has
     shown; today it stays until the page reloads.
2. **Queue rows have actions.** Today every row has only **log**.
   - Queued: **Cancel**.
   - Running: **Log** and **Cancel**, with confirmation.
   - Failed: the reason in words, and **Resubmit**: same model, suite and
     tasks, pre-filled, one click.
   - Done and judged: **Open results**, to the topic page for one topic or
     the model page for several.
3. **Submit's judged suite chooses topics.** Choosing "judged — the written
   exam" on Submit & Queue runs every built task, with no choice. Show the
   same topic checkboxes as the topic page.
4. **The Exam import panel.**
   - Labels go above the fields. Placeholders get cut off: "source (the
     file, e.g. law…".
   - Source is pre-filled from the file name and shown as text with
     **change**. Last night's two wrong sources came from typing the topic
     into that box.
   - After a preview that finds 0 new questions, **Import 0 questions** is
     enabled. Disable it, and say "Nothing new to import."
   - After a successful import, rebuild the harness tasks automatically, or
     put one button in the success toast: **Make these questions
     sittable**. A standalone **Rebuild the harness tasks from the bank** is
     an implementation step exposed to people.
   - Merge **By topic** into the rubrics table. The two repeat the same bank
     counts, and the Loop board repeats them a third time.
5. **The Leaderboard fits.** Today the main table is 1,923 px in 1,234 px.
   - A sticky first column (model).
   - Compact number cells: the score over its ± error.
   - A density toggle.
   - A **Columns** menu to hide tasks, with "N tasks hidden" shown when any
     are.
   - The duplicate row folds under its twin: **1 duplicate ▸**.
   - The compare checkboxes get a column header, **compare**, and start
     unticked. Until something is ticked, the radar shows a prompt.
6. **The Provenance table fits** with the same sticky first column, and long
   ids wrap.
7. **The model page.** It is 14,443 px tall for a model judged on two topics.
   - (a) The five caveat paragraphs above the judged numbers become one line
     of badges, *provisional · not calibrated · single provider · draft
     rubric — why?*, with the text in a popover.
   - (b) The judged section gets a topic switcher, a segmented control. It
     shows one topic's criteria, flag, difficulty and domain tables at a
     time.
   - (c) Hide the **Training compute: Unknown** tile when it is unknown.
   - (d) "Last evaluated" counts judged runs. The Models tab says
     SmolLM2-360M-Instruct was last evaluated 2026-09-20; it was judged
     2026-09-21.
8. **The Training tab** opens with the most recent run selected, not an empty
   right pane.
9. **Tasks and Perplexity charts** show the full model name on hover and in
   the table beneath. The truncated labels ("qwen35-delta-moe-…") can't tell
   the two qwen35 rows apart.
10. **Theme.** "Theme ▾" promises a menu but cycles on click. Make it a menu
    (Auto, Light, Dark, Dim) with the current one ticked.

---

## 9d — one visual system

- **Tokens, per theme:**
  - spacing: 4 / 8 / 12 / 16 / 24 / 32;
  - type: 12 / 14 / 16 / 20 / 28;
  - radii: 6 / 10;
  - one border colour and one surface colour.
- **Buttons:** primary (filled accent), secondary (outlined), quiet (text),
  danger.
  - **Disabled must look disabled**: lower contrast, no hover,
    `cursor: not-allowed`, and the reason visible beside it.
  - Today a disabled Propose is identical to an enabled one; people click it
    and nothing happens.
- **Badges:** one component with three tones (neutral, warning, danger), and
  at most one warning badge per row.
  - The *provisional · single provider · draft rubric* triplet on every Loop
    row becomes one caveat line at the top of the board.
- **Tables:** one component.
  - A sticky header.
  - 40 px rows.
  - Numbers right-aligned with tabular figures.
  - The 9a pager.
- **Empty states:** a sentence and the action that fills it, e.g. "No
  proposals yet. Propose from a topic page →".
- **Loading:** skeleton rows instead of "Loading…".
- **Keyboard:** a visible focus ring on every control, and every action
  reachable from the keyboard.
- **Themes:** check light, dark and dim.

---

## Appendix — every control, as found on 2026-09-21

| Where | Control | Does now | Problem | Fix |
|---|---|---|---|---|
| Header | Theme ▾ | cycles auto / light / dark / dim on click | ▾ promises a menu | 9c-10 |
| Header | 📖 guide for new users | opens `/guide` | fine | also link the Loop section (8e P6c-7) |
| Header | demo run from 20/09/2026 | opens `/demo` | reads like a status chip | move into More ▾ as "Sandbox run" |
| Header | transformers 5.15.0 | nothing | noise on every tab | move to Provenance |
| Header | ▸ 6 checks … | expands six paragraphs | open by default on Overview and Leaderboard | 9b-1 |
| Header | live · refreshed 12:33 | text | no stale state | 9b-10 |
| Overview | hero "Best official average" | shows `…step945` | it's the duplicate; contradicts itself | 9b-2a |
| Overview | Top models rows | go to the model page | duplicate listed as #1 and #2 | 9b-2a |
| Overview | "18 preliminary models (…)" | text | run-on list of 18 names | 9b-2b |
| Overview | stat tiles | none | "Real differences 1202 / 1686" is jargon | tooltip, or drop |
| Overview | How to read these numbers | text | long, always open | collapsed by default |
| Loop board | your name | records the name | fifth copy of the same box | 9b-4 |
| Loop board | topic link | topic page | fine | — |
| Loop board | rubric link | opens the rubric file | fine | — |
| Loop board | last-judged model link | model page | one column mixes two models | 9b-5 |
| Loop board | Propose ×4 | disabled | looks enabled | 9a-4, 9d |
| Loop board | Read the results (physics) | topic page | depends on a per-browser read flag | 9b-6 |
| Loop board | Import a bank ×10 | Exam tab, topic not selected | loses its context | 9b-7, 9b-8 |
| Topic page | ← every topic | back to the board | fine | — |
| Topic page | Propose (header) | disabled | looks enabled | 9a-4 |
| Topic page | model id | free text | no search | 9a-1 |
| Topic page | kind | auto / base / instruct | fine | — |
| Topic page | topic checkboxes | choose topics | fine | — |
| Topic page | Queue this run | queues, shows "#49 queued" inline | no toast; the line never goes away | 9c-1 |
| Topic page | answers: model select | switches model | stale after a deploy; no scores | 9a-3 |
| Topic page | this model's page | model page | fine | — |
| Topic page | acuity / flags / score / sort | filter and sort | no criterion filter | 9a-2 |
| Topic page | more / less | expands an answer | fine | kept as "Show the whole answer" |
| Topic page | the answers table | — | 2,474 px wide; 46–59 rows, one page | 9a-2, 9a-5 |
| Model page | ← Back to Loop | back | fine | — |
| Model page | Judged · Results · Diagnose · Provenance · Runs | sticky anchors | fine | — |
| Model page | Propose a skill spec (per topic) | disabled | a second entry point | 9a-4 |
| Model page | judged section | every topic stacked | 14,443 px page | 9c-7 |
| Models | search, kind, checkpoint, family, has judged, tainted | filter | fine | — |
| Models | column headers | sort | fine | — |
| Models | rows | go to the model page | duplicate not badged; last evaluated ignores judged runs | 9c-7d; add the duplicate badge |
| Leaderboard | Avg scale / Cells / Columns | switch views | long explanations | ⓘ popovers (9b-9) |
| Leaderboard | compare checkboxes | feed the radar | no header; top 3 ticked by default | 9c-5 |
| Leaderboard | column headers | sort | fine | — |
| Leaderboard | the table | — | 1,923 px wide; duplicate shown first | 9c-5 |
| Submit & Queue | model id | free text | no search | 9a-1 |
| Submit & Queue | suite | full / quick / control / judged | judged has no topic choice | 9c-3 |
| Submit & Queue | your name | free text | empty, though filled everywhere else | 9b-4 |
| Submit & Queue | Submit model | submits | no toast | 9c-1 |
| Submit & Queue | filter, status | filter | fine | — |
| Submit & Queue | log (every row) | opens the log | the only row action | 9c-2 |
| Submit & Queue | the table | — | 49 rows, one page | 9a-5 |
| Exam | Rebuild the harness tasks from the bank | rebuilds the task files | an implementation step | 9c-4 |
| Exam | shell commands in the intro | — | CLI instructions in a web page | 9b-9 |
| Exam | import: file, topic, written by, source, name, Preview, Import N | preview, then import | placeholders as labels; "Import 0" enabled; source easy to fill wrongly | 9c-4 |
| Exam | rubrics table: rubric / criteria / replace | open or replace files | shas, container paths, ten fallback rows | 9b-7, 9b-9 |
| Exam | By topic: topic links | filter curation | same counts as the rubrics table | 9c-4 |
| Review | your name | — | another copy | 9b-4 |
| Review | choose / chosen | shows one topic across models | a third way toward proposing | 9a-4 |
| Review | Propose a skill spec (per model card) | disabled | a fourth entry point | 9a-4 |
| Review | What the judge wrote … | expands | fine | — |
| Training | filter, status, sort | — | fine | — |
| Training | run list | selects a run | right pane empty until you pick one | 9c-8 |
| Tasks / Perplexity | raw score / vs chance, show all | toggles | truncated model labels | 9c-9 |
| Provenance | column headers | sort | table wider than the page | 9c-6 |

---

## Definition of done

**Automated, in the browser test suite:**
- At 1,280 and 1,512 px, no tab scrolls sideways at page level.
- The answers list and every paged table fit their container.
- The Leaderboard fits with at most six task columns shown, and says how many
  are hidden.
- At 1,512 × 900, the first screen of Overview shows the hero and Top models,
  not banners.
- Model search, the pager, the override dialog and the version bar each have
  the tests listed under their item.
- The split, provisional, taint and leak tests are unchanged and pass.

**On the live server, from a fresh browser:**
1. Type "smol" under Sit the exam and pick SmolLM2-135M-Instruct with the
   keyboard.
2. Queue law.
3. While it runs, page through the queue.
4. When it lands, the answers list shows it without a reload and without
   sideways scrolling.
5. Click **Propose…** on law, read the warning, tick the box and choose
   **Propose anyway**.
6. In Review, approve the spec, then **Generate**.
7. The dataset shows **over a provisional judge**, and so does its
   `provenance.json`.
