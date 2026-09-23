# Brief for Claude Code — thinking models, reads and three fixes (11l)

Start from main after 11k. **§1 is a correctness bug that invalidates a
whole run.** If it turns out to be large, ship §1 alone as 11l and the rest
as 11m — say which you are doing in the PR description.

**The rules that do not bend are unchanged.** The plain-words rules from
11h §7 apply here too.

**Not in this PR: the exam questions themselves.** masein thinks the banks
are weak. He and I are working that out separately and it will get its own
brief. Change nothing about question content here.

---

## 1. Thinking models are being scored on their thinking

**Seen.** masein submitted `Qwen/Qwen3-1.7B` (run #60, judged, 37 topics,
3,730 answers). Qwen3 is a reasoning model: it emits a `<think>…</think>`
block and *then* the answer. The board stores, shows and grades the raw
generation.

I measured **1,271 answers across 25 of the 37 topics**, straight from
`/api/answers`:

| | count |
|---|---|
| answers measured | 1,271 |
| open `<think>` | **1,271** |
| ever close `</think>` | **0** |
| contain any text after the reasoning block | **0** |

Not one answer in the run got past thinking. Lengths sit between 1,267 and
1,507 characters with a hard ceiling near 1,500 — the generation budget.
`SmolLM2-360M-Instruct` on the same topic: median 865 characters, no
reasoning blocks, finishes comfortably. Qwen3 spends the whole budget
reasoning and is cut off mid-sentence before it answers.

So the judge read 3,730 truncated monologues and scored them as answers.
Qwen3-1.7B's model page now shows Physics & Astronomy 1.26 / 4,
Mathematics & Statistics 1.5 / 4, and so on for all 37 topics. **Every one
of those numbers is meaningless.** Worse, they are the numbers Propose
would read when deciding what to teach the model.

This is not specific to Qwen3. Any reasoning model — DeepSeek-R1
distillations, gpt-oss, later Qwens — hits it, and the board silently
reports a bad score instead of failing.

**Fix, in four parts.**

**a. Separate the reasoning from the answer.** When a generation contains a
known reasoning wrapper, split it. Keep the raw generation for provenance,
and add:
- `answer_text` — the generation with the reasoning block removed. This is
  what the judge grades and what the page shows.
- `reasoning_text` — what was removed.
- `had_reasoning: true`.
- `reasoning_unterminated: true` when the block opens and never closes.

Handle at minimum `<think>…</think>`. Put the wrapper list in config rather
than hard-coding one tag, so the next model's tag is a config change.

**b. An answer that never left the reasoning block is not an answer.** When
`reasoning_unterminated` is true, or `answer_text` is empty after
stripping, the item does **not** get a score. Mark it `no answer`, exclude
it from the topic mean and from every criterion mean, and count it. A topic
where most items have no answer reports **no score at all** — not a low
one. Reuse the "0 unreadable" counter that the by-criterion header already
had: the model page should say "47 answers, 47 no answer" and show a dash
where the score was.

**c. Give reasoning models room to answer.** The current budget truncates
them. Either raise the per-answer token limit when `had_reasoning` is seen,
or send the chat template's no-thinking switch (Qwen3 takes
`enable_thinking=False`; other families differ). Whichever you choose, the
run's settings must be visible on the model page's "How this was graded ▸"
and in provenance, because it changes what the score means. Say which one
you picked and why in the PR description.

**d. Void the bad results.** Run #60's judged results must stop being shown
as scores. Mark them void with a reason the page states in plain words —
"the model never finished answering: these questions were not scored" —
rather than deleting them silently. masein can re-run once the fix lands.

**A new check.** Add to the `N checks` bar: **answers that never finished**.
It is amber when any judged run has items with no answer, and names the
model and the count. This class of failure must never again be discovered
by a person reading one answer.

**Tests.**
- A fixture answer of `<think>reasoning…` with no close: no score, counted
  as no answer, excluded from the topic mean and every criterion mean.
- A fixture answer of `<think>reasoning</think>The answer is 6.`: the judge
  receives only `The answer is 6.`, and the page shows only that.
- A topic whose items are all no-answer reports a dash, not a number.
- A model with no reasoning blocks scores exactly as it does today —
  confirm SmolLM2-360M-Instruct's existing numbers do not move.
- The checks bar goes amber with the model named.

## 2. Read a topic's questions from the Exam tab

**Seen.** masein went to the Exam tab to read the questions, which is the
obvious place, and could not. The **Rubrics and criteria** table has
`rubric ↓ · criteria ↓ · replace` in its FILES column and a BANK cell
reading `100 — 55 report / 45 diagnose` as **plain text**. The bank reader
exists only on the topic page, behind **Read the practice questions**.

**Fix.**
- The BANK cell's count opens the same reader (`read=bank:<topic>`), and so
  does a **questions** entry in the FILES column beside `rubric` and
  `criteria`.
- The cell says which half opens: `100 — 55 report / 45 practice · read the
  practice half`. The report half stays unreadable in the page, as always.

**Test.** From the Exam tab, the bank reader opens for three topics and
shows only practice items.

## 3. Remove the per-criterion breakdown

**Seen.** The model page's **`<TOPIC>` — BY CRITERION (0–1), WEAKEST
FIRST** block. masein does not want it.

**Fix.** Remove the block and its topic picker from the model page. Keep
the per-answer criteria strip inside answer cards (the squares and
`weakest: … 0.0 · … 0.0`) — that one is useful and stays.

Nothing else should depend on the removed view; if something does, say so
rather than working around it.

## 4. The checks pill has no horizontal padding

**Seen.** Measured live at 1,512 px:

| pill | padding | width |
|---|---|---|
| `7 checks ▾` | **3px 0px** | 84 px |
| `masein ▾` | 4px 12px | 77 px |
| `Theme ▾` | 4px 12px | 75 px |

The text touches its own border. Its two neighbours in the same bar are
correct.

**Fix.** `padding: 4px 12px`, matching its siblings. One rule should style
all three; they should not be able to drift apart again.

**Test.** All three pills in the bar report the same computed padding.

## 5. Conditional criteria may not be gating

**Seen.** In the by-criterion view being removed in §3, every criterion
marked `conditional` in `criteria.json` — "only when the question calls for
it" — reported **100 of 100** for Mathematics & Statistics. All twelve of
them, on all one hundred answers.

If the judge is not actually deciding whether a criterion applies, then
every answer is marked down for not doing error analysis or proof reasoning
on questions that never asked for either. That drags topic scores down and
corrupts the weakest-criterion list — which is exactly what Propose reads.

**Fix.** Check `judge.py`: does it ask whether each conditional criterion
applies before scoring it, or score all twenty every time? Report what you
find across several topics and models before changing anything. If
conditionals are working and the banks really are that uniform, say so in
the PR — the finding then belongs with masein's separate question about
bank quality.

Removing the view in §3 does not remove this problem; it only hides it.

## 6. Submitting a model gives no sign that anything happened

**Seen.** masein pressed **Submit model**, the form did not change, and he
could not tell whether it had worked. It had — run #60 was queued.

**Fix.** On success: clear the model field, reset the topic ticks to their
default, and show a confirmation that links to the new row — **"Run #60
queued — follow it →"**. The button is disabled while the request is in
flight. On failure: keep every field exactly as typed and show the error.

**Test.** A stubbed success clears the form and links the run; a stubbed
failure keeps every field.

## 7. The question-and-answer dataset format cannot work

**Seen.** Dataset #9 (Government & Public Policy, pythia-31m) failed with
`the generator returned no parseable items`, keeping 0 of 26. Every item
was dropped for "no question or no answer". The dataset was requested as
`fmt: free`, but the generator was handed the **document** register, which
ends: *"Prose, never question-and-answer pairs."* The generator wrote prose,
as instructed, and the Q&A parser found none.

Datasets #1–#8 all used `fmt: doc` and all succeeded. #9 is the only
`fmt: free` and the only failure. All nine carry the same never-Q&A line.

**Fix.** Give `fmt: free` a register of its own that asks for
question-and-answer items, or take it out of the picker until it has one.
An option that cannot succeed must not be selectable.

**Test.** Either a `free` fixture produces parseable items, or `free` is
absent from the picker.

## 8. A failed dataset explains nothing

**Seen.** The row shows `Failed` in the DOCUMENTS MADE BY column, where a
count belongs, and the error is the unhelpful sentence above. The real
per-item reasons are already stored in `provenance.items.missing`.

**Fix.**
- The DOCUMENTS MADE BY cell reads **0 of 26**; the status carries the
  failure.
- The message says what happened: **"0 of 26 kept — asked for
  question-and-answer items, but the generator was given the prose
  register."**
- **Details ▸** lists the per-item reasons already in provenance, paged,
  instead of leaving them only in the JSON.

## 9. The suite options are not discoverable

**Seen.** masein believed the `full` option had been removed, because the
dropdown showed `judged` and nothing says what the five options are for.

**Fix.** Each option carries one plain line saying what it gets you, and
the panel says that **`full` and `judged` are separate runs, not one
inside the other** — a model needs both to have an average and a judged
score. Per-task resume already makes re-submitting free; say that too.

---

## Definition of done

On the live server:

1. A reasoning model's answers are judged on the answer, never the
   reasoning, and an unterminated block scores nothing and is counted.
2. **Qwen3-1.7B's 37 judged topics** no longer show scores, and the page
   says why in plain words.
3. The **checks bar** names any run with unfinished answers.
4. The **Exam tab** opens the practice questions for any topic.
5. The model page has **no by-criterion block**.
6. All three **header pills** have the same padding.
7. `judge.py`'s handling of conditional criteria is **reported on**.
8. **Submit** clears the form and links the queued run.
9. `fmt: free` either works or is gone.
10. A **failed dataset** shows `0 of n` and says what went wrong.
11. **All tests are green**, with screenshots in `tests/_screens/phase11l/`.

## Deploy steps, for masein, after this merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.
