# Brief for Claude Code — the 12b live check, and Everyday round 2 (12b.3, 12a.2)

Start from main at `571545b` (#63). Two parts:

- **Part A (12b.3)** — what the live check of 12a, 12b.1 and 12b.2 found on
  2026-09-24 at 17:30. Small, and §A1 is urgent.
- **Part B (12a.2)** — Everyday tasks round 2: 106 questions in seven groups,
  marked by a general check vocabulary. Bigger.

Ship them as two PRs, A first. Each is done when `scripts/check.sh` passes
on its branch head and its summary line is in the PR under "Local check".

**The rules that do not bend are unchanged**, and so are the display rules
every 12x brief carries: one number first; nothing empty; no system words;
a caveat once per page; one main action; numbers in mono, words in sans.

---

# Part A — the live check (12b.3)

Checked on the live server at 1,512 px. What works: the header reads Home ·
Models · Improve · Benchmarks with Runs, Test a model, the status dot and
masein ▾; **all eleven old `#tab=` addresses land where 12b §8 says**,
including `review` → Improve ▸ Review with its view kept and `provenance` →
Data & sources; Home shows its three blocks; the model page has its tiles and
Scores · Answers · Improve · History; Models has its switch; the pilot page
sits under Benchmarks ▸ Everyday tasks.

## A1. The board's own address shows an empty page — urgent

**Seen.** `http://100.74.89.105:8899/` and
`http://100.74.89.105:8899/?token=…`, with no `#`, render the page title and
the footer line and nothing else: no navigation, no content, no console
error. It stays empty. `#tab=home` renders Home correctly.

This is the address the team was sent this morning.

**Fix.** An empty hash, or any hash the router doesn't know, lands on Home
and rewrites the address to `#tab=home`. The bare `#home`, `#models` and
`#benchmarks` should land on their places too — people will type them.

**Test.** A browser test that loads `/` and `/?token=…` with no hash and
finds the four places in the header and Home's three blocks. Load it cold,
not by changing the hash of a page that has already rendered, since that's
the case that broke.

## A2. The pilot shipped with the Arabic question

**Seen.** `eval_tasks/everyday/pilot.jsonl` still has 03 as the Arabic
translation, and the groups as `language` and `behaviour`. The bank is
**English only**, decided on 2026-09-24. No model has taken the pilot yet,
so nothing is lost.

**Fix.** Replace 03 and rename the two groups:

| id | group | prompt | check |
|---|---|---|---|
| 03 | summarising | `tldr pls: "Dear parents, on Thursday 1 October school will close early at 11:30 for staff training. Buses will leave at 11:15. If you are collecting your child yourself please come by 11:45 at the latest. Friday is a normal day. Thank you, Ms Rahman"` | `judge`, rubric below |
| 05 | **instructions** (was behaviour) | unchanged | unchanged |

Rubric for 03, passed to the judge with the question and the answer: *"Passes
if the answer is at most two sentences and says school closes early at 11:30
on Thursday, with buses at 11:15 or pickup by 11:45. Fails if it changes a
time or adds anything the notice doesn't say."*

The judge takes its rubric from the question; nothing is hard-coded to
Arabic. The pilot page's short label for 03 is **TL;DR**. The pilot's groups
are Understanding, Writing, Transform, Summarising, Instructions.

## A3. A test that has never run against the real harness fails

**Seen.** In the container, `tests/test_categories.py::
test_matches_the_installed_harness` fails:

```
subjects = sorted(name[5:] for name, meta in tm.task_index.items()
                  if name.startswith("mmlu_")
                  and "tasks/mmlu/default/" in str(meta.get("yaml_path", ""))
E   AttributeError: 'Entry' object has no attribute 'get'
```

The installed lm_eval returns `Entry` objects in `TaskManager.task_index`,
not dicts. The test was written against the old shape and has always been
skipped where lm_eval isn't installed.

**Fix.**
- Read `task_index` in a way that works for both shapes.
- **Search `scripts/` and `service/` for every other use of `task_index`**,
  or any other lm_eval internal read as a dict. If the app's MMLU-by-area
  code reads it the same way, it is broken on the server too. Say what you
  found in the PR.
- Deploy step 4 then ends with no failures.

## A4. The status dot is always amber, so it will be ignored

**Seen.** The dot reads **● 7**, and Home's Needs you says *"7 checks are not
green"*. The popover opens with "3 of 7 are about the judged suite", then:

1. Chat template on some models, not others
2. Models evaluated with different chat templates
3. 1 duplicate row: the same run submitted twice
4. 19 of 34 models are preliminary
5. The judge is not calibrated against a person
6. 5 models graded by a local judge: provisional
7. Single-provider loop

Most of these are **standing limits**, true until the setup changes: an
uncalibrated judge, a single provider, preliminary models. They will be
amber for weeks. A dot that is always amber stops meaning anything, and then
nobody sees the duplicate row or a failed run. Also, 1 and 2 are the same
check shown twice.

**Fix.**
- Split the checks into **problems** — something happened that a person
  should act on (a duplicate row, a failed run, answers that never finished,
  a judge that drifted) — and **known limits** — a standing condition of the
  setup (judge not calibrated, provisional local judge, single provider,
  preliminary models, mixed chat templates).
- **The dot counts problems only.** Green when there are none.
- The popover lists problems first, then one line **Known limits (5) ▸**,
  collapsed.
- **Home's Needs you lists problems only.**
- Merge the two chat-template checks into one.

**Test.** With only known limits present, the dot is green and Needs you
doesn't mention checks; add a duplicate row and the dot shows **● 1**.

## A5. Needs you counts datasets nobody will train on

**Seen.** *"8 datasets made but not used in training."* All nine datasets
are **Demo only**, made over a provisional judge. Training happens outside
the board, so they will sit in Needs you permanently.

**Fix.** Needs you counts only datasets that are **not** Demo only. A Demo
only dataset is still listed in Improve ▸ Review as it is today.

## A6. A score exists but the tile shows a dash

**Seen.** SmolLM2-360M-Instruct's **Knowledge exam** tile reads **—**, *out
of 4 · 8 of 37 topics · not ranked*, and so does its Scores block header. It
has eight judged topics. A dash reads as "nothing here". Home's Best in each
kind has no Knowledge exam card at all, for the same reason.

The rule behind it stays: a provisional judged score never enters an average
or a ranking. So don't show an average. Show what is allowed, a topic score:

- The tile reads **8 of 37 topics judged**, then the model's weakest topic
  and its score (**weakest: Economics 0.79 / 4**), with the provisional
  badge once in the tile.
- The block header shows the same, not a dash.
- Home's Knowledge exam card shows **the weakest topic across the board**,
  as the old Overview's "Weakest topic" card did, with one provisional
  badge.

## A7. The model page has two main buttons

**Seen.** The header's **Test a model** and the page's **Test this model**
are both filled buttons, side by side on one screen.

**Fix.** On a model page, the header button reads **Test this model** and
opens the dialog with the model filled in. The page's own button goes.
Everywhere else the header button reads Test a model, as now.

## A8. Small things

- **Home's section numbers.** "01 Needs you · 02 Running now · 03 Best in
  each kind" aren't steps in a sequence. Drop the numbers.
- **Home's card link** reads *Compare models →*, but comparing is 12h and
  doesn't exist yet. It reads **See all models →**.
- **System words on the model page.** *"Canary steady. 30 of 30 fixed
  scripts re-graded: mean absolute deviation 0.1667 from the human marks…"*
  sits in the main view. The main view says **Judge steady**; the rest goes
  under How this works ▸.

Not checked by me: 400 px. My browser wouldn't resize the window. Keep your
400 px screenshots in the PR.

## Part A — done when

1. `http://100.74.89.105:8899/` with no `#` opens Home.
2. The pilot's 03 is the English TL;DR and its groups are the five above.
3. Deploy step 4 ends with no failures, and the PR says where else
   `task_index` is read.
4. The dot is green when only known limits remain.
5. Needs you ignores Demo only datasets.
6. No tile or block header shows a bare dash for a model with judged
   topics.
7. One filled button per page.
8. The local check's summary line is in the PR.

---

# Part B — Everyday tasks round 2 (12a.2)

The pilot proved the plumbing. Round 2 is the first real Everyday bank:
**106 questions in seven groups**, written by an AI from a prompt masein and
I iterated on, and then every check was tested against a right and a wrong
answer.

The three files are beside this brief in `docs/prompts/phase-12b3/`:

| File | What it is |
|---|---|
| `everyday_round2.jsonl` | the 106 questions |
| `checks.py` | the reference checker: **copy its behaviour exactly** |
| `everyday_round2_probes.jsonl` | 180 answers with the verdict each must get |

## B1. The seven groups

**Understanding, Writing, Summarising, Transform, Quick maths,
Instructions, Honesty** — in that order, everywhere. English only. Show them
capitalised like that. The group key in the file is lower-case
(`quick_maths`).

## B2. One check vocabulary

Every question carries a `checks` list; **all must pass**. `checks.py` is the
reference; port it into `scripts/everyday.py` so the two agree on every
probe. What matters in its behaviour:

- **Whole-word matching, case-insensitive** unless `"case_sensitive": true`.
  `"2 november"` must not match `"22 november"`.
- **am/pm is normalised:** `12:30pm` reads as `12:30 pm`, on both sides.
- **Numbers** accept thousands commas (`1,284.50`).
- **`line_count`**, **`word_count`** and **`sentence_count`** ignore
  code-fence lines and one lead-in line ending in `:` ("Here you go:").
- **`json`** takes the first object *or array*, fenced or not; each inner
  list of `required_values` is alternatives, matched inside any value.
- **`in_order`** accepts a list of alternatives in any position.
- **`no_invented`** has `phone`, `url`, `email`, `price` (both `AED 80`
  and `80 AED`, and `1.2 million AED`) and `distance`.
- **`judge`** sends the question, the answer and the question's `rubric`.
  Five questions use it; each also has script checks, and the judge only
  decides if the script checks pass.

The types: `contains_any`, `contains_all`, `not_contains`, `number`,
`json`, `line_count`, `max_words`, `word_count`, `sentence_count`,
`in_order`, `no_emoji`, `no_digits`, `numbers_from_source`, `no_invented`,
`asks_back`, `judge`. An unknown type fails loudly at import.

Each check returns pass or fail and a plain reason, as the pilot does. The
reason is what the page shows.

## B3. The pilot joins the bank

Convert the pilot's five questions to this vocabulary and add them to the
bank as `everyday-pilot-01…05`. That makes 111 questions and **one Everyday
bank**, not a pilot beside a round. Keep the pilot's result history: a
model that sat the pilot keeps its five marks.

## B4. What it is — still not a benchmark

Like the pilot: **every question readable, never ranked, never averaged into
anything, never read by Propose, never sent to a generator.** The header
badge reads **Round 2 · not ranked**. The practice/hidden split starts in
round 3, when each group reaches about 40. Write it so that adding a `split`
field later is a data change.

## B5. Running it

- The `everyday` suite now runs the whole bank. Its line in the Test a
  model dialog: **"Everyday tasks — 111 questions, a few minutes"**.
- Generation as in 12a: greedy, the chat template, `max_gen_toks` 512, or
  the 2,048 that #59 gives reasoning models; answers marked after the
  reasoning is split off.
- The pilot page's **Run the pilot** becomes **Run everyday tasks**, with
  the same model picker.

## B6. What masein sees

**Benchmarks ▸ Everyday tasks** — the bank, grouped by the seven groups,
each question readable with its checks in plain words ("says 29", "valid
JSON with the date, time and place", "no invented phone number").

**The results table** — models across the top, the seven groups down the
side, each cell **n of k**, where k is the group's size (15 to 17 once the
pilot's five have joined). Clicking a cell opens that model's answers in that group,
each with ✓ or ✗ and its reason. One row per question inside it, so masein
can read down a group.

**On the model page**, the Everyday block shows the seven groups as **n of
k** and opens to the answers the same way.

**On Models**, the Everyday tasks switch appears once any model has results,
as 12b.1 already does: one column per group plus the total, the Round 2 badge
once.

## B7. Tests

- **Every probe in `everyday_round2_probes.jsonl` gets its expected
  verdict.** A probe whose question carries a judge check expects the script
  checks' verdict; a probe expecting a pass must not fail on a script check.
- Every question's `reference` passes its own script checks.
- The import refuses a file with an unknown check type, a duplicate id or an
  invalid line, and names the line.
- The bank never reaches Propose, a generator request or any average —
  asserted over the recorded request bodies, as for the pilot.
- A cell on the results table opens the right model's answers in the right
  group.

## Part B — done when

1. **Run everyday tasks** runs 111 questions on four models in minutes each.
2. Benchmarks ▸ Everyday tasks shows the bank by group and the results by
   model and group, and every cell opens its answers.
3. All 180 probes pass.
4. Nothing from the bank appears in any ranking or in Propose.
5. The local check's summary line is in the PR.

## Deploy steps, for masein, after each merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests
inside the container. Put the expected output of each in the PR.
