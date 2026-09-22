# Brief for Claude Code — read files, the first page, sitting the exam from the model page, and the Review tab (11g–11j)

**Start from main once 11f has merged.** There are four PRs. Do them in
order, and each must be green on its own:

- **11g**: read every file in the page, not only download it. This is
  masein's first ask.
- **11h**: the first page (Overview) and the findings from the 11e live
  check. This is masein's second ask, plus mine.
- **11i**: sit the exam from the model page, and warn about checkpoints that
  ship their own code *before* queueing them. This is masein's third ask.
- **11j**: the Review tab, rebuilt around what a person comes to do. It
  brings simpler proposal and dataset cards, shows reviewers the questions,
  fixes the blank failure count, and removes "Sandbox run". This is
  masein's fourth ask.

**Plain words.** masein: *"the texts are not simple enough — even I got
confused."* The writing rules in 11h §7 apply to every PR in this brief,
not only to 11h.

**The rules that do not bend are unchanged.** One matters most here:
**report-half text never reaches the page.** This brief adds several new
ways to read files, and none of them may show a report-half question,
answer or qid.

---

## 11g — read files in the page

### What masein asked

> "I want to be able to see items, not only download them. Like the
> items.jsonl in a dataset, be able to preview them appropriately; or for
> rubric and criteria I want to be able to read them in the dashboard — or
> wherever else we are."

### What the page does today (checked in the code)

| File | Today |
|---|---|
| Dataset documents | an `items.jsonl` link that downloads (`/api/datasets/{id}/items.jsonl`) |
| Rubric `.md` and `.criteria.json` | "rubric · criteria" download links on the Exam tab. On the topic page, `arts.md` and `20 criteria` are plain text. |
| Run log | "Log" opens the raw text in a new browser tab (`/api/runs/{id}/log`, the last 200 lines) |
| Exam questions | `/api/exam/bank` already withholds report-half text (`public_bank`), but nothing lets a person read the bank |
| Provenance | JSON in the API. The page shows parts of it, scattered. |

### One Reader

Every file opens in one component, the **Reader**. It is a side sheet over
the page, from the right.

**Size.**
- Its width is `min(760px, 92vw)`.
- Below 720 px it covers the whole screen.

**The header holds:**
- a title;
- where the file comes from, e.g. "dataset #6 · Arts · 20 documents" or
  "`arts.md` · no version · sha `ab12…`";
- the actions **Copy**, **Download** and **Open raw ↗**;
- **✕ Close**.

**Behaviour.**
- It opens and closes with the 11f motion tokens.
- Esc closes it and returns focus to what opened it. Focus is trapped
  while it's open.
- **It has an address**: `#…&read=<kind>:<id>[:<n>]`, e.g.
  `read=dataset:6:3`. A pasted link opens it, and Back closes it.
- It survives polls. Like the opened rows, it lives in `state`, not in the
  DOM.

**Text is always safe.**
- Everything is escaped. Markdown goes through a small renderer that
  allows no raw HTML.
- Links in files open with `rel="noopener"`.
- Nothing from a file is ever run.

**Size limits.**
- Lists are paged: 50 per page from the API.
- Long logs render in chunks.
- A 20-document dataset opens in under 300 ms.

### What each kind shows

**1. Dataset documents** (`read=dataset:<id>`)

- **The head of the reader:**
  - topic, model, the approved spec (collapsible) and focus mode;
  - "requested 20 · kept 18 · missing 2", each missing entry with its
    reason;
  - the over-a-provisional-judge mark, when it applies.
- **On the left, a list.** Each entry has its number, title, `Focus:` label
  and word count.
  - Missing documents appear greyed in their place, with their reason.
- **On the right, the document**, set as readable prose:
  - 16 px sans, at most 68 characters per line;
  - its paragraphs kept;
  - its title as a heading.
- **Getting around:**
  - "3 of 20" beside the title;
  - ← and → for the previous and next document;
  - a search box that searches titles and text, and highlights matches.
- **The "free" format** (question and answer) shows each item as a card:
  Question, Answer, Rationale.
- **New API:** `GET /api/datasets/{id}/items?offset=&limit=&q=` returns
  JSON. The `items.jsonl` download stays as it is.

**2. Rubric** (`read=rubric:<name>`)

- The `.md` rendered: headings, lists, tables, emphasis and code.
- A contents list built from its headings.
- The version marker, or "no version", and the sha in the header.

**3. Criteria** (`read=criteria:<name>`)

- **Parse the file with the judge's own `normalise_criteria()`**, so the
  page shows exactly what the judge reads. Both layouts are handled:
  `critical_error` and `critical_flags`.
- **A table of criteria:**
  - #;
  - the criterion;
  - its weight or maximum;
  - **when it applies**, for conditional criteria;
  - what counts as a pass.
- **Then the flags.** For each flag, show:
  - its name;
  - what it does (e.g. "score becomes 0");
  - its examples;
  - its "not critical" examples.
- **Top-level fields** such as `domain` and `scoring_scale`.
- **Show raw JSON**, pretty-printed.

**4. Exam questions** (`read=bank:<topic>`)

- **Diagnose half only**, from `/api/exam/bank?topic=…&half=diagnose`.
  *(Corrected 22 Sep, masein's answer: the plain `/api/exam/bank` lists the
  hidden questions by qid, so the reader asks for the practice half and a
  count; without the parameter the endpoint is unchanged.)*
- The report half appears as one line: "56 report-half questions — never
  shown, by design".
- Each question shows:
  - its prompt;
  - its difficulty (1–5);
  - its `domain` label;
  - its style;
  - "written by masein".
- Filters: difficulty, domain and style. A search box too.

**5. Run log** (`read=log:<run id>`)

- Mono text, with line numbers and a wrap toggle.
- A search box that highlights matches, with **next ↓**.
- Lines with `ERROR`, `Traceback` or `failed` are tinted with the warning
  tone.
- **Follow.** While the run is active, it polls the tail and keeps
  scrolling to the end until the person scrolls up.
- **Load earlier lines**, up to the API's 2,000.
- Queue's "Log" opens this reader. "Open raw ↗" stays in the row's `⋯`
  menu.

**6. Provenance** (`read=provenance:dataset:<id>`, and
`read=provenance:judge:<model id>` for how a model was graded)

- A tree of keys and values, collapsible.
- Hashes shortened, e.g. `ab12…`, with a copy button.
- Timestamps in local time.
- The provisional and override marks as badges.

### Where "Read" appears

Everywhere a file is linked or named today. **Read** is the main action;
**Download** stays beside it.

| Place | Reads |
|---|---|
| The topic page's datasets table | dataset documents; provenance in each row's `⋯` menu |
| The Review card's "datasets: #6 ready" | dataset documents |
| The Exam tab's rubric table | rubric, and criteria |
| The topic header's `arts.md` and `20 criteria` | rubric, and criteria |
| The topic page's bank line, `bank 100 (56 report / 44 diagnose)` | exam questions, diagnose half |
| Queue rows | the run log |
| The dataset reader's header, and each dataset row's `⋯` menu (the topic page, and 11j's Datasets view) | dataset provenance |
| The model page's Judged section: **How this was graded ▸** | judge-run provenance |

*(Corrected 22 Sep, masein's answer: the Provenance tab has no dataset or
judge-run rows, and gets no new tables.)*

### Tests for 11g

- **Every kind:**
  - opens from its link;
  - opens from a pasted `read=` address;
  - closes on Esc and returns focus;
  - survives a poll.
- **Dataset reader:**
  - the list count equals `kept`, and missing documents show with their
    reasons;
  - ← and → move between documents;
  - search narrows the list.
- **Criteria reader:** the table's rows match `normalise_criteria()` on
  the Arts file and on one file of each layout.
- **Report-half isolation.** Record every network response while each
  reader opens, then assert that none contains:
  - the text of a report-half question from the fixture bank;
  - a report-half qid.
- **Safe rendering.** A rubric containing `<script>` and
  `<img onerror>` renders as text, and nothing runs.
- **The log reader:**
  - follows a growing log;
  - stops following after the person scrolls up.
- **At 400 px**, the reader is full screen and nothing overflows.
- **Screenshots** of each kind, in light and dark, go in
  `tests/_screens/phase11g/`.

---

## 11h — the first page, and the 11e findings

### 1. The Overview "seems a bit wrong"

masein said: *"it seems a bit wrong — maybe it's the font, maybe the sizes,
I'm not sure."*

**What I measured on the live page at 1,512 px:**
- **Monospace is everywhere.** It sets 47 of the 72 pieces of text on the
  Overview. That includes:
  - the card values;
  - the underlined card links, which read like code;
  - the stats line;
  - the guide pills;
  - the captions.

  The reference site uses mono only for small labels and numbers.
- **The title appears twice**: in the sticky bar, and again as the 28 px
  hero heading.
- **The hero is a narrow column.** It sits on the left, with about 60% of
  the width empty, and **Submit a model** touches the stats line under it.
- **The hierarchy is flat.** The card values (28 px), the hero title
  (28 px) and the section titles (20 px) compete with each other.
- **Top models:**
  - the Avg column is a solid blue block, which 11f should fix;
  - the hover background covers only part of a row: it stops before
    PARAMS.

**The fix.**

**One font rule, for the whole board.**
- **Mono** is for:
  - numbers in tables and cards;
  - model ids;
  - eyebrows and section indices;
  - table column headers;
  - the LIVE badge;
  - the log reader.
- **Sans** is for everything read as words:
  - model names in cards;
  - verdicts and links;
  - pills and buttons;
  - the stats line, captions and the footer.

**A compact hero, on one row.**
- **Left:** the eyebrow, then a one-line description, then the two guide
  links in sans.
- **Right:** **Submit a model**.
- **Drop the hero `h1`.** The bar already carries the title.
- **Below it:** the stats line, in sans 14 px, muted.
- The hero is at most 120 px tall at 1,280 px and above.

**Highlight cards.**
- **The value** is **28 px sans, weight 700, `tabular-nums`**, not mono.
- **The name line** is 14 px, weight 600, with an ellipsis.
- **The verdict** is 14 px and muted.
- **The link** is sans, in the accent colour, with no underline until
  hover.
- All four cards are the same height, with their links on one baseline.

**Top models.** The hover background covers the whole row.

**Test: the font rule.** On Overview, the Leaderboard and a model page,
every text node in mono is inside one of these:
- a `td.num`;
- a `th`;
- `.eyebrow`, `.secidx` or `.badge`;
- a model-id element;
- the log reader.

**Test: the layout.** At 1,280 px and at 1,512 px:
- the hero is at most 120 px tall;
- the four cards and the first row of Top models are visible without
  scrolling, in an 868 px-tall window.

### 2. Two scales for the same number (from the 11e check)

**What was seen.**
- On the **Knowledge** chip, the eight area columns show plain accuracy
  ("report half · %"), even with `Scale: above chance` selected.
- The opened row's MMLU-by-area bars show above chance.
- So Qwen3-0.6B's Mathematics & Science reads **44.7** in the table and
  **26** in its own row.

**Fix.**
- Every MMLU-by-area number, in columns, bars and the radar, follows the
  `Scale:` pill.
- The column tooltip and the bars' caption say which scale is shown.

**Test.** Qwen3-0.6B shows the same value in its Math & Science column
and its bar, under both scales.

### 3. The page scrolls sideways on Knowledge

**What was seen.** At 1,512 px, the Knowledge table (mmlu plus 8 area
columns) is wider than its card. The whole page scrolls sideways:
`scrollWidth` is 1,591 against a 1,512 px window, and "General" is cut off.

**Fix.** The table scrolls inside its own scroller at **every** width,
not only on a phone:
- the # and Model columns stay pinned;
- a fade and the "scroll →" hint show on the right while there's more.

**Test.** On every chip, `document.documentElement.scrollWidth` is at most
the window width, at 1,280, 1,512 and 1,920 px.

### 4. Filters take half a phone screen

**What was seen.** At 400 px, the table starts about 550 px down the
page:
- the chips take 3 rows;
- the filter pills take 3;
- the pager takes 2.

**Fix.**
- Below 720 px, fold Kind, Size, Status, Columns, Models and Scale into
  one **Filters ▾** button.
- That button opens a sheet, showing a count when any filter is set, e.g.
  "Filters · 2".
- The chips become one sideways-scrolling row.
- The pager fits on one line.

**Test.** At 400 px, the table's first row starts within 320 px of the
top of the Leaderboard card.

### 5. The LIVE time looks frozen

**What was seen.** At 17:45, the badge still read `LIVE · 17:29`. That
was when the data last changed, not when the page last checked. It looks
stuck.

**Fix.**
- The badge shows the time of the **last successful check**.
- Its tooltip says "data last changed 17:29 · checked 12 s ago".
- If checks fail for more than two intervals, the badge turns amber and
  reads `STALE · 17:29`.

**Test.** With a fake clock, the badge's time advances on every
successful poll, and turns stale when polls fail.

### 6. The opened row's Tasks list (check 11f first)

**What was seen.** On the All tasks chip, the opened row's Tasks block is
narrow. It wraps `47.4 / ±0.4` and `5- / shot` onto two lines each.

**Fix.** If 11f's compact list already fixed this, do nothing. Otherwise,
every task line fits on one line at 1,280 px.

### 7. Plain words, across the board

masein: *"The texts are not simple enough. As you see, even I got
confused. Let's make it simpler so the user understands better."*

**The writing rules** apply to everything a person reads on the page:
headings, cards, buttons, toasts, empty states and dialogs.
- **One idea per sentence.** At most 20 words.
- **Say what it is, then what to do.** Explanations go behind **How this
  works ▸**, **Why? ▸** or a tooltip. They are never paragraphs above the
  thing itself.
- **One warning per screen.** The same caveat is never repeated on one
  screen, as a badge and then as three boxes.
- **No internal names on the surface.** These appear only in **Details
  ▸**, tooltips and the Provenance tab:
  - task ids like `exam_arts`;
  - backend ids like `local/chat`;
  - the prompt and sha fingerprints;
  - `judge.json`;
  - batch counts;
  - "13-gram".

**The glossary.** This changes page text only. Code, API fields and file
names stay as they are.

| Today | Say instead |
|---|---|
| diagnosis half / DIAGNOSIS-half / diagnose half | **practice questions** (or "the practice half") |
| report half / report-half questions | **hidden questions**. A score is "score (hidden questions)" |
| over a provisional judge / provisional / preliminary (for judged work) | a **Demo only** badge. Its tooltip gives the reason in words, e.g. "the judge is a small local AI no person has checked yet" |
| skill spec / spec | **the missing skill** |
| LLM / proposer / generator | **the AI** / **the AI that writes documents** |
| 13-gram contamination gate | **copy check**: "no document may copy 13 words in a row from any exam or benchmark question" |
| κ / kappa | **agreement with a person** |
| fell short | **scored below 3 of 4** |
| taint | **trained on this topic's practice data** |

**Test.** Keep a list of banned surface phrases: "diagnosis half",
"DIAGNOSIS", "report half", "report-half", "over a provisional judge",
"LLM", "13-gram", "batch items", "exam_", "local/chat", "judge.json" and
"κ".
- Assert none of them appears in the visible text of Overview, Loop,
  Models, Leaderboard, Queue, Review, a topic page or a model page.
- Text inside Details, tooltips, the Reader's raw views and the
  Provenance tab is excluded.

### 8. Remove "Sandbox run"

**What it is.** A link in More ▾ opens `/demo`. That page reports a
practice run of the whole loop, made by `scripts/demo_loop.py` in a
separate `$BENCH_ROOT/demo/` folder, not on the real board. The loop now
runs live, so the link only confuses people.

**Remove from the dashboard:**
- the `Sandbox run` menu item;
- the `GET /demo` route and its page;
- `payload["demo"]`;
- the demo notes on the startup page (`service/app.py`, around line 1994);
- the tests that check them.

**Keep `scripts/demo_loop.py` and `DEMO.md`** as a command-line tool for
now. Add one line to `DEMO.md`: "the dashboard no longer links to the demo
tree". masein may ask to delete them later.

**Test.** No "Sandbox run" in the menu, and `GET /demo` returns 404.

### Tests for 11h

These are listed under each item above. Add screenshots of Overview, the
Leaderboard (All tasks and Knowledge) and a model page:
- at 1,280, 1,512 and 400 px;
- in light and dark;
- saved in `tests/_screens/phase11h/`.

---

## 11i — sit the exam from the model page

### What masein asked

> "When I want to evaluate a model using our exams, it's not a good user
> experience to go from the topic page. It should be from the model page,
> so I can click to start on all or some of the topics for a model."

Today there are two ways to run the exam:
- the topic page's **Sit the exam** panel, which covers one topic and asks
  which model;
- the Queue's submit form with the `judged` suite, which has a flat list of
  37 tick boxes.

The model page, where a person is thinking about *this model*, has no way
at all.

### 1. The Sit the exam panel on the model page

**Where it opens from:**
- a **Sit the exam** button in the model page's hero, beside the three
  cards;
- the opened Leaderboard row's **Run exam** link;
- the Loop board's model rows.

Every one of these opens the panel on the model page, and scrolls to it.

**What the panel holds:**
- **All 37 topics, grouped under the 8 areas.**
  - Each area header has a tick box that selects the whole area.
  - Each topic row has a tick box, the topic name and **its status for this
    model**, which is one of:
    - `not sat`;
    - `0.79 / 4 · 22 Sep` (judged; provisional marked as today);
    - `on older questions` (its answers are under Earlier exams, after
      10b);
    - `in the queue` or `running…`, with that row's tick box disabled.
- **Quick picks:**
  - **Not sat yet (30)**;
  - **All 37**;
  - **Weakest 5**, which uses only this model's own judged topics;
  - **None**.
- **MMLU control (open-ended)**: its own tick box, off by default, with a
  one-line explanation.
- **A live summary line**, e.g. "12 topics · about 1,200 answers · about 25
  min of GPU, then grading · the GPU is shared".
  - Over 10 topics, add "consider a quiet time".
- **Queue this run.** It is disabled, with the reason beside it, when:
  - the judge is offline;
  - the weights aren't on the server (11a);
  - the checkpoint ships its own code and it isn't allowed (item 2 below).

**What happens after Queue this run:**
- a toast;
- each ticked topic's status turns to `in the queue`, then `running…`;
- each score fills in as the judge finishes.

**One component.** The Queue's submit form (for a model not on the board
yet) uses the **same** grouped picker when the `judged` suite is chosen.
The topic page keeps its single-topic panel.

**A topic already judged on the same questions** re-grades without
re-answering, via the 10b fingerprint. The row says so: "same questions —
re-grade only, no GPU".

### 2. Checkpoints that ship their own code: say so before queueing

**What was seen.** #56 queued `local/qwen35-delta-moe-7d560104-step945-v2`
on all 37 topics, and it failed at start:

> "artifact … carries an auto_map, so loading it executes the Python shipped
> in the upload. That is off by default…"

The reason is right, but it arrives after the person has queued the run.
And the page has no way to send `allow_remote_code`, so the row's
**Resubmit** would fail again.

**Fix.**
- **Look before queueing.** Preflight reads the artifact's `config.json`
  before queueing. It does this in the search suggestion, the panel above
  and the Queue form.
- **When `auto_map` is present** and the server allows remote code
  (`ALLOW_REMOTE_CODE=1` and `EVAL_USER` set):
  - show an **unticked** box: "Run this checkpoint's own model code
    (`modeling_*.py`, sha `ab12…`) — as the unprivileged `benchjob` user".
  - Queue this run needs it ticked.
  - Send `allow_remote_code: true`.
- **When the server doesn't allow it:** Queue this run is disabled. The
  reason names the two settings and `SERVICE.md § custom model code`.
- **The API** refuses such a submission with 422 before queueing, in the
  same words. It never fails at start.
- **Resubmit**, on a row that failed for this reason, offers the same box.
- If `REMOTE_CODE_SHAS` is set and the file's sha isn't in it, say so, and
  give the sha to add.

### 3. A judged row lists 37 topic names

**What was seen.** #56's Suite cell lists all 37 topics, which makes one
row about 650 px tall.

**Fix.**
- The cell reads `judged · 37 topics + MMLU control ▸`.
- The ▸ opens the list, grouped by area.
- One topic stays spelled out, e.g. `judged · Arts`.

### Tests for 11i

**The model page panel:**
- it shows 37 topics under 8 areas;
- statuses come from the model's `judge.json`, `history` and the queue;
- each quick pick ticks exactly its set;
- the summary line's counts follow the ticks;
- Queue this run sends exactly the ticked `exam_*` tasks (and
  `fr_control_mmlu` only when ticked);
- a topic already in the queue can't be ticked.

**The shared picker.** The Queue form's `judged` suite uses the same
picker component.

**Remote code** (fixture artifacts):
- one with `auto_map` on a server that doesn't allow it → disabled with
  the reason, and the API returns 422;
- allowed → the box is required, and `allow_remote_code` is sent;
- a sha not on the allowlist → the message names the sha;
- Resubmit on #56's kind of row offers the box.

**The Suite cell** is one line for a 37-topic row.

Screenshots go in `tests/_screens/phase11i/`.

---

## 11j — the Review tab, rebuilt

### What masein said

> "Check the Review tab. Don't you think the tab is confusing? If I want to
> start a new review, or want to review the pending ones, the approved
> ones and the generated datasets — I think the UX is bad."

He also asked about the proposal card:
> "Show the questions — the human wants to see them to review."

> "The '—' needs to get fixed."

### What the tab is today (live, 22 Sep)

One long page, **7,084 px tall at 1,512 px**. In order:

1. A paragraph of explanation, plus lines about the AI's usage.
2. **Pick a topic**: a table of judged topics with "choose" buttons. It
   repeats the Loop board.
3. **Awaiting review.**
4. **Approved — ready to generate.** Five cards, each about 1,300 px tall,
   and each repeating the same warning four times.
5. **Datasets**, at the very bottom, under all five cards.

**Rejected proposals** have no place.

**The failure count is blank.** "failures — of 44 diagnosis-half items" is a
bug. The page reads `ev.diagnose_wrong`, but the service sends
`diagnose_weak` (`service/proposals.py`). The number should be 36.

### The new tab

**At the top, one row:**
- **four views, each with a count:**
  - **To review (0)**;
  - **Ready to generate (1)**;
  - **Datasets (8)**;
  - **History**, which holds rejected proposals and older ones;
- on the right, a primary button: **+ New proposal**.

The view lives in the hash, e.g. `#tab=review&view=datasets`. The tab
opens on **To review** when something is waiting, and on **Datasets**
otherwise.

**+ New proposal** opens a small dialog, so a person no longer has to go
through a topic page.
1. Pick a model: the combobox of judged models.
2. Pick a topic: grouped by area, weakest first, each showing its score for
   that model. A topic is disabled, with its reason, when:
   - it isn't judged for that model;
   - it already has an open proposal;
   - it has fewer than 30 hidden questions.
3. The Demo-only note, in one sentence, with the tick box from 9a.
4. **Propose.**

The topic page's **Propose…** opens the same dialog, with the model and
topic already filled in.

**Each view is a compact list**, using the table component. A row is one
line.

| View | Columns |
|---|---|
| To review / Ready to generate / History | topic · model · status · asked by · when · Demo-only badge |
| Datasets | # · topic · model · documents ("20 of 20", or "18 of 20 · 2 missing") · made by · when · Demo only · **Read** · **Use in training ⧉** |

- **Clicking a proposal row** opens it in a side sheet, the 11g Reader's
  frame, with the simple card described below.
- **Read** opens the dataset reader from 11g.
- **Use in training ⧉** copies `--gap-dataset 8`. Its tooltip says: "Pass
  this to your training run. The run records the dataset, and the
  checkpoints it makes are marked as trained on this topic's practice
  data."
- The "Pick a topic" table goes; the Loop board already covers it.
- The explanation paragraph becomes one line plus **How this works ▸**.
- The usage lines move into **How this works ▸**.

### The simple proposal card

This is the side sheet. Everything in it follows the §7 writing rules.

**Header.**
- "**Arts** · SmolLM2-360M-Instruct" in large type. `#5` is small and
  muted.
- One status chip: To review, Approved or Rejected.
- **One** `Demo only` badge. Its tooltip gives the three reasons in plain
  words:
  - "no person has checked the judge yet";
  - "the judge is a small local AI";
  - "the same AI did more than one step".

**What's missing.** The missing skill, in large text. It is editable while
the proposal is To review.

**Why.** One line: "36 of 44 practice answers scored below 3 of 4 · Arts
score 1.46 / 4 (hidden questions)". This fixes the blank count.

**The answers it read (33) ▸**, which opens the full list, not 8. Each item
shows:
- **the practice question**, which the reviewer may read;
- the model's answer;
- its score;
- the judge's comment.

One muted line explains the difference: "The AI that wrote the missing
skill saw only the judge's comments, with the question wording taken out,
so no exam wording can reach the training data. You see the questions so
you can check its reading."
- The questions come from the practice half only.
- **Hidden questions never appear**, and no qid of theirs.

**Documents will cover.**
- The focus plan as chips, with the first 8 shown and "+12 more".
- The "Spread the documents over these" tick box.

**Actions, by status:**
- **To review:** **Approve** or **Reject**, with a reason required for
  Reject.
- **Approved:** count [20], format [Documents ▾], then **Generate**. The
  format menu offers "Q&A (for comparison)".

**Datasets from this proposal.** "#6 · #7", each opening the reader.

**Details ▸** holds:
- who asked and approved, and when;
- which AI proposed it, and the prompt fingerprint;
- the copy-check explanation;
- today's usage;
- the full technical reasons.

### Dataset rows elsewhere

The topic page's datasets table uses the same one-line row as the Datasets
view. Today, each row carries a sentence about training runs; that sentence
moves into the **Use in training ⧉** tooltip.

### Tests for 11j

- **The views:**
  - the four views show the right proposals and datasets, and their counts
    match;
  - the chosen view survives a poll and comes back from a pasted hash.
- **New proposal:**
  - the dialog disables each blocked topic with its reason;
  - Propose sends the chosen model and topic;
  - the topic page's Propose… opens the same dialog, pre-filled.
- **The card:**
  - "36 of 44" shows for the Arts fixture, and no "—" appears when the
    number exists;
  - The answers it read lists all of them, not 8, each with its practice
    question, answer, score and comment.
- **Report-half isolation.** Record every network response while the card
  and the list open, then assert that none contains the text or qid of a
  hidden question.
- **One Demo-only badge** per card, and no repeated warning box.
- **Dataset rows:** Read opens the reader, and Use in training copies
  `--gap-dataset N`.
- **The tab is short.** At 1,512 px, with 5 proposals and 8 datasets, the
  tab is at most 1,600 px tall.

Screenshots go in `tests/_screens/phase11j/`.

---

## Definition of done

On the live server:

1. **A dataset reads in the page.** From the Arts topic page, dataset #6
   opens in the Reader:
   - with 20 documents, each with its focus label;
   - with ← and → working;
   - with search working.

   Dataset #2 (Physics) shows its 2 missing documents with "reasons not
   recorded".
2. **Rubric and criteria.** `arts.md` reads as formatted text. The Arts
   criteria read as a table of 20 criteria, then the flag with its
   examples.
3. **The Arts bank** reads its 44 diagnose-half questions. The 56
   report-half questions appear only as a count.
4. **A Queue row's Log** opens in the Reader, and follows while a run is
   active.
5. **Every Reader has an address.** Pasting its link opens it, and Back
   closes it.
6. **The Overview has one font rule**, a compact hero with no repeated
   title, and cards with a sans number and one-line names.
7. **No page scrolls sideways** at 1,280, 1,512 or 1,920 px, on any chip.
8. **One scale.** The area numbers match between the table and the opened
   row, under both scales.
9. **At 400 px**, the Leaderboard table starts near the top, behind a
   Filters ▾ button.
10. **The LIVE badge** moves with each check.
11. **Sitting the exam from the model page.** From
    SmolLM2-360M-Instruct's page:
    - Sit the exam shows 37 topics by area, with 7 judged and 30 not sat;
    - picking **Weakest 5** or an area and clicking Queue this run queues
      exactly those topics.
12. **Remote code.** `local/qwen35-delta-moe-7d560104-step945-v2` shows
    before queueing that it ships its own code:
    - with the box, when the server allows it;
    - disabled with the reason, when it doesn't.
13. **The Queue's #56 row** is one line tall.
14. **Plain words.** None of the banned surface phrases appears on the main
    tabs. Every card has one Demo-only badge instead of four warnings.
15. **No "Sandbox run"** in the menu.
16. **The Review tab** opens on To review or Datasets.
    - **+ New proposal** works from the tab itself.
    - Proposal #5 (Arts) opens as a short card showing "36 of 44 practice
      answers scored below 3 of 4".
    - Its answers list shows all 33, with their practice questions.
    - The Datasets view lists #1–#8, each with Read and Use in training.
17. **All tests are green**, with screenshots in `tests/_screens/phase11g/`,
    `phase11h/`, `phase11i/` and `phase11j/`.

## Deploy steps, for masein, after each PR merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in each PR description and in
`HANDOFF.md`.
