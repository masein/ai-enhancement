# Brief for Claude Code — the Everyday tasks pilot (12a)

Start from main after #60 (11m). The pilot's answers are marked on the
text after the reasoning block, so it builds on #59's split of reasoning
from answer. Use what #59 built; don't write a second splitter.

This is the first of six PRs (12a–12f) from a plan masein approved on
2026-09-24. The plan reorganises the board into five places (Home, Models,
Playground, Improve, Benchmarks) and four kinds of test (Standard,
Knowledge exam, **Everyday tasks**, On phone). **12a builds none of that
structure.** It adds five questions, marks them by script, and shows the
results, so masein can read what four models actually say before anyone
writes a bigger bank. 12b does the restructure; this PR must not pre-empt
it.

**The rules that do not bend are unchanged.** The plain-words rules from
11h §7 apply, plus these display rules from the approved plan, which every
12x brief carries from now on:

- **One number first.** No ± , shots, sample counts or ids in the main view.
- **Nothing empty.** A test a model hasn't taken is one line with a button,
  never an empty table or a row of dashes.
- **No system words.** Suite names, check names, file names and hashes
  stay out of the main view.
- **A caveat once per page**, as one badge in the header, never per row.
- **One main action**, one filled button, top right.
- **Numbers in mono, words in sans.**

---

## 1. What the pilot is, and what it is not

**Everyday tasks** asks what people type into an assistant on a phone:
short, lowercase, typos, one plain request. Most answers can be marked by
a script, so the score needs no judge.

The pilot is **five questions, all readable**. It is a look, not a
benchmark:

- It is **never ranked**, never averaged into anything, never shown on the
  Leaderboard, and never read by Propose or sent to a generator.
- Its five questions are **not split** into practice and hidden halves.
  Five visible questions cannot be a published score, so there is nothing
  to protect yet — and hiding half of five would leave nothing to look at.
- Every place it appears carries one badge: **Pilot · not ranked**.

When the bank grows past the pilot, the split applies as it does
everywhere. That is a later brief; write the pilot so that adding a
`split` later is a data change, not a rewrite.

## 2. The five questions

New file `eval_tasks/everyday/pilot.jsonl`, one object per line:

```json
{"id": "everyday-pilot-01", "group": "understanding",
 "prompt": "…", "check": {"type": "…", …}, "notes": "…"}
```

Use these exactly, typos included — the typos are the test:

| id | group | prompt | check |
|---|---|---|---|
| 01 | understanding | `hey can u tell me hwo many days is in febuary in a leep yaer` | `contains` any of `29`, `twenty-nine`, `twenty nine` |
| 02 | transform | `turn this into json: Sara Ahmed, 34, product manager, Dubai, joined March 2021` | `json` holding all five values |
| 03 | language | `how do i say "the meeting is moved to thursday" in arabic` | `judge`, with reference `تم نقل الاجتماع إلى يوم الخميس` |
| 04 | writing | `fix this: "Dear Sir, I writing to you regard the invoice which was sended last week and still not payed."` | `fixed`: all four mistakes corrected |
| 05 | behaviour | `give me 3 names for a coffee shop. just the names, no explanation.` | `lines`: exactly three names, nothing else |

The five group names are the plan's: **Understanding, Writing, Transform,
Language, Behaviour.** Show them capitalised like that.

## 3. Marking

New `scripts/everyday.py`. It reads the generations the harness logged,
marks each one, and writes `everyday.json` beside the model's results. It
always marks the answer #59 splits off — never the raw generation. An
answer #59 calls unfinished fails with the reason "never finished
answering".

Each check returns **pass or fail and one plain-words reason** — the
reason is what the page shows, so write it for a person:

- **`contains`** — passes when any listed string appears, case-insensitive.
  Fail: *"didn't say 29"*.
- **`json`** — take the first `{…}` block, with or without a code fence,
  and parse it. Passes when it parses and its values, flattened and
  lower-cased, hold `sara ahmed` (or both `sara` and `ahmed`), `34`,
  `product manager`, `dubai`, and `march 2021` or `2021-03`. Key names are
  free. Text around the JSON does not fail it, but the reason notes it.
  Fail: *"not valid JSON"* or *"missing: Dubai, March 2021"*.
- **`fixed`** — passes when the answer has *I am writing* or *I'm
  writing*; *regarding*, *in regard to*, *with regard to* or *about*;
  *sent*; and *paid* — and has none of *sended*, *payed* or *I writing*.
  Fail names what's left: *"still says 'payed'"*.
- **`lines`** — split into non-empty lines, and strip list markers
  (`1.`, `-`, `•`) and wrapping quotes or `**`. Passes on exactly three
  lines, each at most five words, none holding `:` or a dash followed by
  more words. Fail: *"4 lines, expected 3"* or *"line 2 explains the
  name"*.
- **`judge`** — the only check that calls the judge. Ask: is this a
  correct Arabic rendering of the whole sentence, in Arabic script? Moved,
  postponed or rescheduled to Thursday all pass. Answer: pass/fail and one
  sentence. It inherits the judge's provisional status, which the page
  states once in its header badge, never beside the row.

`everyday.json` holds, per item: `id`, `group`, `pass`, `reason`,
`answer_text`, `had_reasoning`, and the check type. Plus the run's
settings. The model's pilot result is the count passed, **n of 5**.

## 4. Running it

- A new suite, **`everyday`**, beside `quick`, `full`, `control` and
  `judged` in `service/app.py`. In the Submit form's dropdown its line is
  **"Everyday tasks — 5 questions, minutes"**. (12c replaces the dropdown
  with cards; keep this line short and don't restyle the dropdown.)
- Generation: greedy, **`max_gen_toks` 512** — or the 2,048 that #59
  gives reasoning models — and the chat template always. Write the task yaml beside the pilot file;
  don't reuse `_fr_template_yaml`, whose `Answer:` suffix is exactly what
  a person on a phone never types.
- **A model without a chat template is refused at preflight**, in plain
  words: *"This model has no chat template, so it can't be asked questions
  the way a person would."* Use the chat template when the tokenizer has
  one, whatever kind the board lists — Qwen3-0.6B is listed as base but
  has a template.
- Marking runs straight after generation, in the same run. Only question
  03 waits on the judge; the run shows `grading 0/1` until it returns, as
  judged runs do today.

## 5. What masein sees

Two views, both small. Neither adds a tab.

**a. On the model page**, a block titled **Everyday tasks**, placed above
the judged block, with the **Pilot · not ranked** badge and the count
**4 of 5**. Five rows: the group, a ✓ or ✗, and the reason in one line.
Clicking a row opens the question and the model's full answer under it —
thinking folded away as **thinking ▸ 212 words** when there was any. No
criteria strip, no score bar, no ids.

A model that hasn't taken the pilot shows one line instead of the block:
**"Not tested on everyday tasks · Test"**, where *Test* queues the
`everyday` suite for this model and confirms like the Submit form does.

**b. The pilot page, `#everyday`** — this is "see the scores together".
One table: the five questions down the side as short labels
(*Typos*, *JSON*, *Arabic*, *Fix the email*, *Just 3 names*), the models
across the top, ✓ or ✗ in each cell, and the **n of 5** in the header
row. Clicking a cell opens that model's answer and reason in a side panel,
with the question above it, so masein can move down a column or along a
row and read. The full question text sits under its short label.

The page's one main action, top right, is **Run the pilot**. It opens a
small dialog listing the instruct models on the board, with these four
ticked: **Qwen3-1.7B, Qwen3-0.6B, SmolLM2-360M-Instruct and
gemma-3-270m-it**. **Queue 4 runs** queues one `everyday` run per model
and confirms *"4 runs queued · follow them →"*. Models that already have a
pilot result say *"done · run again"* and are unticked.

Reach the page from the model page block (**Compare models →**) and from
**More ▾ ▸ Everyday pilot**. In 12b it moves into Benchmarks; don't build
any navigation beyond those two links.

## 6. Not in this PR

- The five places, the new header, the Test-a-model cards (12b, 12c).
- Chatting with a model, and saving a question from a chat (12d, 12e).
- Phone models (12f).
- More everyday questions, and the practice/hidden split for them.
- Any change to the Knowledge exam or the Standard benchmarks.

## 7. Tests

- Each check, on fixture answers: a pass, and each fail reason. At least:
  JSON in a code fence passes; JSON with *"Here's your JSON:"* before it
  passes with the note; `1. Bean There` numbering passes `lines`; four
  lines fail; *"payed"* fails `fixed` and names it.
- A fixture with an unterminated reasoning block fails every check with
  *"never finished answering"*.
- A model with no chat template is refused at preflight with the sentence
  above.
- The pilot never appears on the Leaderboard, in any average, in
  Propose's inputs, or in a generator request — assert it over the
  recorded request bodies, as the split tests do.
- The pilot page renders four models from fixtures; a cell opens the
  answer; the header shows *Pilot · not ranked* once and no row repeats
  it.
- A model with no pilot result shows the one-line *Not tested* row, not an
  empty block.
- 400 px: the pilot page's table scrolls inside its card and the page does
  not scroll sideways.

## Definition of done

On the live server:

1. **Run the pilot** queues four `everyday` runs, and each finishes in
   minutes.
2. **`#everyday`** shows four models × five questions, and every cell opens
   the answer.
3. Each **model page** shows its Everyday block, or the one-line *Not
   tested* row.
4. **Qwen3-1.7B's answers** are the text after its thinking, and its
   thinking is folded.
5. Nothing from the pilot appears on the **Leaderboard** or in **Propose**.
6. **All tests are green**, with screenshots in `tests/_screens/phase12a/`.

## After it deploys

masein and I read the four columns together and decide which groups get
more questions, and whether Everyday becomes what Models opens on. Put
nothing in this PR that assumes the answer.

## Deploy steps, for masein, after this merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.
