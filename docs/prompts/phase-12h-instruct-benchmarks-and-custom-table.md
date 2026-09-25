# Brief for Claude Code — IFEval, MMLU-Pro, MATH-500, nine new models, and a table you build (12h)

Start from main **after 12a.4** is merged and deployed. If it isn't, stop and
say so.

Two PRs:

- **12h.1** — three generative benchmarks, and nine small instruct models
  that can run them.
- **12h.2** — on Models → Standard, a table where you choose the benchmarks
  and the models, with the average of the chosen benchmarks.

Each is done when `scripts/check.sh` passes on its branch head, its summary
line is in the PR under "Local check", and deploy step 4 passes on the server.

**No model runs on the Mac.** Don't run IFEval, MMLU-Pro or MATH-500, or load
any of the nine models, on the Mac or in the local check. Write the code and
test it against saved example answers (fixtures), with no model. Every real
run happens on the server after deploy. masein does those runs from the
commands under "Commands for masein, after deploy" and pastes you the output.

The rules that do not bend, and the display rules, are unchanged:
- one number first;
- nothing empty;
- no system words;
- a caveat once per page;
- one main action;
- numbers in mono, words in sans.

---

# 12h.1 — the benchmarks and the models

## 1. Three benchmarks

| Shown as | lm_eval task | What it measures | Scored by |
|---|---|---|---|
| **IFEval** | `ifeval` | follows format instructions ("no commas", "3 bullets", "under 100 words") | prompt-level strict accuracy, shown first; instruction-level beside it |
| **MMLU-Pro** | `mmlu_pro` | knowledge and reasoning, 10 options, 12,032 questions, chain of thought | exact match on the extracted letter |
| **MATH-500** | `hendrycks_math500` | competition maths, 500 problems | exact match on the final answer |

- **Check the task names against lm_eval 0.4.12 first.** `hendrycks_math500`
  is on lm_eval's main branch, and it may be newer than 0.4.12. If a task
  isn't there, say so in the PR and propose the smallest fix: a pinned task
  YAML in our repo, or a newer lm_eval. Don't upgrade lm_eval quietly.
- **Scoring must work on chat answers.** lm_eval has an open issue about
  `hendrycks_math` scoring 0 on chat models, because it can't find the answer
  in free text.
  - Build the answer-reading layer, and test it on fixtures: realistic chat
    answers for each benchmark, written by you, with the answer each must
    yield. Include the awkward cases: "The answer is (C).", "**C**", "I think
    it's C because…", `\boxed{\frac{1}{2}}` vs 0.5, and an answer with a
    `<think>` block.
  - For MATH-500, compare answers with `math-verify` or an equivalent, not
    raw string equality.
  - Say what our layer does beyond lm_eval. The tooltip on the column says
    the scorer, as it does for every setup today.
  - Add **`scripts/trial_generative.py`** (see "Commands for masein, after deploy"). It runs N items of the three
    benchmarks on one model inside the container and prints, per item, the
    raw answer, what we extracted, the correct answer and the verdict. On the
    server, it's the check that extraction works before any full run.
- **Instruct models only.** These run with the chat template, so a base model
  can't be scored fairly. In the table, a base model shows "instruct only" in
  these columns once, in the "Not tested on this" list. It never shows an
  empty cell.
- **Group them** in the column chips as a new group, **Instruction & maths**:
  IFEval, MMLU-Pro and MATH-500. The existing Math chip keeps whatever it has
  today.

## 2. What a run costs, and how to keep it sane

- MMLU-Pro is 12,032 chain-of-thought answers per model. The current
  Standard tasks score likelihoods and finish in minutes; this one generates
  text.
  - Use the vLLM backend from phase 8 for these three tasks wherever the
    model loads in vLLM. Fall back to HF only when it doesn't, and say which
    models fell back.
  - `trial_generative.py` prints the seconds per item. From that it
    estimates a full MMLU-Pro run for that model: "about 2 h 40 min for
    12,032". If the estimate is over 3 hours, masein decides between a full
    run and a fixed, seeded subset **labelled as a subset**. Build the subset
    option, off by default. Only a full run is comparable to published
    numbers.
- Answer budget: 2,048 tokens in non-thinking mode and 8,192 with thinking on.
  Answers that ran out of room are counted and shown, as 12a.4 does for
  Everyday tasks.

## 3. Thinking on or off

Several of these models can think before answering, and it changes the score
a lot. Qwen3.5-2B publishes IFEval 61.2 without thinking and 78.6 with it.

- **Default: thinking off** for every model that can turn it off. That's how
  a phone assistant runs, and it keeps the comparison fair.
- The mode is **part of the result**. It's recorded on the run and shown in
  the column tooltip and on the model page. A model that can't turn thinking
  off runs with it on, and its row says "thinking" in the same small label
  style as "instruct".
- A thinking-on run of a model that has an off switch is a **separate row**:
  "Qwen3.5-2B · thinking". It's never averaged with the same model's
  thinking-off run. The submit form gets a "Think before answering" switch,
  off by default. It appears only for models that support it.

## 4. Nine models

Add these to the model list and the submit form's suggestions. The Hugging Face
names are checked; the notes are from each model card, so check them against
the real load.

| Model | Hugging Face | Size | What to watch |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 0.8B | needs the newest transformers; the card loads it as multimodal; thinking off by default |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | 2B | same |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4B | same |
| LFM2.5-1.2B | `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B | convolution + attention hybrid; LFM 1.0 licence |
| Granite-4.0-H-1B | `ibm-granite/granite-4.0-h-1b` | 1.5B | Mamba2 hybrid; install `mamba_ssm` and `causal-conv1d` if the image can build them, or it runs slowly |
| Gemma 4 E2B | `google/gemma-4-E2B-it` | 2B effective | multimodal; the card loads it with `AutoModelForMultimodalLM`; newest transformers |
| Nemotron-3-Nano-4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | Mamba2 hybrid; **ships its own code**; thinking on by default, so turn it off |
| Youtu-LLM-2B | `tencent/Youtu-LLM-2B` | 2B | **ships its own code**; needs transformers **4.56–4.57.1** |
| Nanbeige4.1-3B | `Nanbeige/Nanbeige4.1-3B` | 3B | the card says it ships its own code, and uses the slow tokenizer; its tags say Llama, so try the standard Llama loader first; a reasoning model |

**Two conflicts to resolve, and to say how you resolved them in the PR:**

1. **transformers versions.** Qwen3.5 and Gemma 4 need the newest
   transformers. Youtu-LLM needs 4.57.1 or older.
   - First, find the one version that loads the most of the nine, and move
     the bench image to it. Re-run the unit tests and deploy step 4 on it,
     because every existing task must still be found.
   - If Youtu-LLM can't load on that version, don't hold the other eight
     back. Say so, and propose the smallest way to run it: a second pinned
     environment, or vLLM if vLLM supports it. Wait for masein's answer.
2. **Models that ship their own code.** The board refuses it by default (the
   #56 failure). Don't turn it on for everything.
   - Add an **approved list**: repo name plus a pinned commit, kept in the
     repo and reviewed in the PR. A model on the list may run its own code,
     at that commit only. Everything else is still refused, with today's
     message.
   - Put the commits for Nemotron-3-Nano-4B and Youtu-LLM-2B on the list, and
     Nanbeige4.1-3B's only if the standard loader fails.

You can't load the models from here, so write the loading code from the model
cards and test what can be tested without weights: the approved list, the
thinking switch in each chat template, and the transformers version choice.
The real loads happen on the server, with `trial_generative.py`.

## 5. Sanity against published numbers

Published scores for these models use their own setups, so ours won't
match. They should land in the same neighbourhood, though, so a big gap
points to a scoring bug. **Add this table to the new columns' tooltip data,
and have the board flag any of our scores more than 15 points below the
published one** with a small "far below published, check extraction" note on
the cell, for masein to look into:

| Model | Published IFEval | Published MMLU-Pro | Published MATH-500 |
|---|---|---|---|
| Qwen3.5-2B | 61.2 (no thinking) | 55.3 (no thinking) | — |
| LFM2.5-1.2B | 86.2 | 44.4 | — |
| Granite-4.0-H-1B | 78.5 (average) | 32.9 (5-shot CoT) | — |
| Nemotron-3-Nano-4B | 82.8 (prompt, no thinking) | — | 95.4 (thinking) |
| Youtu-LLM-2B | 81.2 | 61.6 | 93.7 |
| Gemma 4 E2B | — | 60.0 | — |


## Commands for masein, after deploy

Put these in HANDOFF § 5b as copy-pasteable lines, and in the PR:

```
# 1. check that extraction works: 20 items per benchmark, one model
sudo docker compose exec -T bench python scripts/trial_generative.py --model Qwen/Qwen3-1.7B --limit 20 2>&1 | tail -60

# 2. check each new model loads, and estimate full-run time
sudo docker compose exec -T bench python scripts/trial_generative.py --model Qwen/Qwen3.5-2B --limit 5 2>&1 | tail -30
```

Then the full runs are queued from the dashboard, as today.

## 12h.1 tests

- The three tasks are in deploy step 4's list. (`check_tasks.py` itself runs
  on the server.)
- The answer-reading fixtures: every chat answer yields its expected answer
  and verdict, for all three benchmarks.
- `trial_generative.py --help` works, and its output format is tested on a
  fake model that returns fixed text.
- The approved list refuses a repo at a different commit, and one that isn't
  on it.
- Thinking on and off produce separate rows, never averaged together.
- A base model shows "instruct only" in the new columns and is never counted
  in their average.

---

# 12h.2 — a table you build

Today, Models → Standard averages every column in view, over every model. The
chips (Knowledge, Commonsense, …) narrow the columns, but the average is
always "all of these", and you can't pick models.

## 6. Choose benchmarks and models

On Models → Standard, beside **Filters**, add two pickers:

```
[All tasks] [Knowledge] … [Instruction & maths]   Benchmarks: 3 ▾   Models: 6 ▾   Filters ▾

Custom · IFEval, MMLU-Pro, MATH-500 · 6 models                     Save view · Reset

#  MODEL              PARAMS   AVG OF 3 ▼   IFEVAL   MMLU-PRO   MATH-500
1  Qwen3.5-4B         4B       …            …        …          …
```

- **Benchmarks ▾**: a checklist of every Standard benchmark, in its chip
  groups, with a search box. Choosing any chip still works as today, and it
  fills the checklist.
- **Models ▾**: a checklist of the board's models with a search box, grouped
  as today (instruct, base, checkpoints). The top has "All ranked" (today's
  default) and "Clear". One click on a row's name in the table still opens
  the model.
- **The average is over the chosen benchmarks only**, and its header says so:
  **"Avg of 3"**. Its ± error combines the chosen columns' errors, and it's
  z-tested for bold as today.
- **A chosen model missing a chosen benchmark** isn't averaged. It sits under
  the line in the existing "Not tested on this" list, which says what's
  missing: "Qwen3.5-0.8B · no MATH-500 · Test". Test opens the submit form
  filled in.
- When anything is custom, one line above the table says what's shown, with
  **Reset**. Nothing else on the page changes.

## 7. Keep it and share it

- **The address holds the view.** `#tab=models&cols=ifeval,mmlu_pro,math500&models=…`
  opens exactly this table. Copying the URL is enough to share it.
- **Save view** names it ("Phone shortlist") and saves it for the whole team,
  with who saved it. Saved views appear as chips after the groups, with a
  small divider. The person who saved a view can rename or delete it from a ⋯
  on the chip. Nobody else can.
- **Copy as CSV** under ⋯ on the custom line: the rows and columns as shown,
  with the average and ± errors.

Everyday tasks and the Knowledge exam aren't offered as columns here. Their
scores are provisional or judged, and they're never averaged with Standard
numbers. Their own tabs keep their own tables.

## 12h.2 tests

- Choosing 3 benchmarks makes the average over those 3, labelled "Avg of 3".
  A model missing one of them isn't averaged, and it's listed with what's
  missing.
- A model subset shows only those rows. "All ranked" brings the default back.
- The URL round-trips: open it and the same table appears.
- A saved view appears for another user. Only its owner can rename or delete
  it.
- CSV matches the table, including the average and errors.
- At phone width, the pickers stack and the table scrolls sideways inside its
  own box, not the page.

---

## Done when

On the live server:

1. **In the PR:** the fixtures pass in the local check, and no model ran on
   the Mac.
2. **On the server, after deploy:** deploy step 4 finds the three tasks, and
   `trial_generative.py` on Qwen3-1.7B shows extraction working. If it
   doesn't, masein pastes you the output and you fix it in a follow-up PR.
3. The nine models load on the server, or a follow-up says which didn't and
   why. A model that ships its own code runs only from the approved list.
4. The instruct models have the three scores with thinking off, and a score
   far below its published number is flagged on its cell.
5. Models → Standard lets masein pick benchmarks and models. The average
   follows the choice. A view can be saved, shared by URL and copied as CSV.
6. Each PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after each merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests and
the task-discovery check inside the container. 12h.1 changes the image, so
the rebuild will take longer than usual.
