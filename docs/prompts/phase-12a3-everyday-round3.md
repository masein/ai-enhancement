# Brief for Claude Code — Everyday tasks round 3 (12a.3)

Start from main **after the everyday-runs fix** ("Selected Tasks: []", #62–#65)
is merged. The import itself doesn't need it, but its done-when does: a real
run on the server.

One PR. Done when `scripts/check.sh` passes on its branch head, its summary
line is in the PR under "Local check", and deploy step 4 passes on the server.

The rules that do not bend, and the display rules, are unchanged.

---

## What this is

Round 3 grows the Everyday bank from 111 to **333 questions**, so every group
can have a hidden half big enough to score — 12g.2's bar is 20 hidden per
group. Every question was checked against its own sample answer, a right
answer worded differently, and a typical wrong one.

| Group | Round 2 + pilot | Round 3 | Total | Hidden, about |
|---|---|---|---|---|
| Understanding | 16 | 29 | 45 | 22 |
| Writing | 16 | 32 | 48 | 24 |
| Summarising | 17 | 46 | 63 | 32 |
| Transform | 16 | 30 | 46 | 23 |
| Quick maths | 15 | 30 | 45 | 22 |
| Instructions | 16 | 29 | 45 | 22 |
| Honesty | 15 | 26 | 41 | 20 |

The files are beside this brief in `docs/prompts/phase-12a3/`:

| File | What it is |
|---|---|
| `everyday_round3.jsonl` | the 222 new questions, same schema as round 2 |
| `checks.py` | the reference checker — **replaces the one from 12a.2** |
| `everyday_probes_all.jsonl` | 366 answers with the verdict each must get: round 2's 180 and round 3's 186 |

## 1. Import the questions

Add the 222 to the bank beside round 2 and the pilot. Same rules as 12a.2's
import: refuse an unknown check type, a duplicate id or an invalid line, and
name the line. Keep `written_by`.

## 2. Port the checker changes

`checks.py` changed since 12a.2. Port each change into `scripts/everyday.py`
so the two agree on **all 366 probes**:

- **Time ranges.** "7–11 am", "2–3pm" and "9:30–11:30 am" read as both
  times: "7 am-11 am". Apply it everywhere times are matched — contains
  checks, `in_order`, `facts`, `first_mention`.
- **`in_order` reads am/pm the same way as the contains checks:** "8am
  pickup" matches "8 am pickup".
- **`no_invented` ignores what the question itself contains.** Repeating the
  caller's number back ("I can't tell who 050 712 8841 is") isn't inventing a
  phone number. For phone, price and distance, compare digits; for url and
  email, compare the text.
- **Two new check types:**
  - **`facts`** — `{"type":"facts","n":4,"values":[[…],[…],…]}`: at least n
    of the listed facts appear, each fact a list of ways to say it. Its
    reason names how many were kept and the first missing ones: *"kept 3 of
    6 key facts, needs 4 (missing: 7 am, Mirdif)"*. Summaries and rewritten
    messages use it: a good one keeps most key facts, not every one.
  - **`first_mention`** — `{"type":"first_mention","right":[…],"wrong":[…]}`:
    the right choice is named before any wrong one. So *"Message Nadia, not
    Nabil"* passes and *"Nabil"* fails. Reasons: *"never names Nadia"*,
    *"names Nabil first"*.

In the question list, say each new check in plain words, as the others are:
*"keeps at least 4 of: the date, the time, the place, …"* and *"picks Nadia,
not Nabil"*.

## 3. The split, when 12g.2 lands

Nothing here assigns halves by hand. When 12g.2's split runs on this bank,
check the real hidden counts per group and say them in that PR. Honesty is
the one at risk: 41 questions give about 20 hidden, right at the bar. If it
lands below, say so, and masein will add a few more.

## Tests

- **All 366 probes get their expected verdict.** A probe on a question with a
  judge check expects the script checks' verdict.
- Every question's `reference` passes its own script checks.
- Summarising: pasting the original message back as the answer fails every
  round-3 summary. Assert it for all of them; the word limit is what makes it
  fail.
- Writing: repeating the request back as the answer fails every round-3
  writing question.
- The new checks' plain-words descriptions appear in the question list.

## Done when

1. The bank holds 333 questions in seven groups, and the question list shows
   each group's count.
2. All 366 probes pass in the local check.
3. **Run everyday tasks** runs the whole bank on the five instruct models on
   the live server.
4. The PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after this merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests and
the task-discovery check inside the container.
