# Brief for Claude Code — AI models, the judge test, and the question builder (12i)

Start from main at **3b4918e** (#74) or later. #75 (12a.5b) may be merged or not; nothing here depends on it.

Three PRs, in order:
- **12i.0**: polish from the 2026-09-25 live check, and one HANDOFF rule.
- **12i.1**: an **AI models** page (who judges, who writes questions, who writes training data, who checks), using OpenRouter or the local model, plus a **judge test** where masein marks answers and sees which judge agrees with him.
- **12i.2**: a **question builder** that writes Knowledge exam or Everyday questions step by step, instead of uploading a bank.

Each PR is done when CI on the mirror is green (its run link in the PR, as now) and deploy step 4 passes on the server. **No model and no paid API call runs on the Mac or in CI.** Test with fixtures and a fake OpenRouter.

The rules that do not bend, and the display rules, are unchanged:
- one number first;
- nothing empty;
- no system words;
- a caveat once per page;
- one main action;
- numbers in mono, words in sans.

---

# 12i.0 — polish

From the live check of #71–#73:

1. **"Avg of N" says what it is.** The header reads **"Avg above chance"**, with a one-line tooltip: "0 = guessing, 100 = perfect, so a 25% guess on a 4-option test counts as 0". Today a reader who averages the columns by hand gets a different number and thinks it's wrong.
2. **Display names only in the benchmark picker:** MMLU, HellaSwag, PIQA, WinoGrande, ARC-Challenge, ARC-Easy, GSM8K, TruthfulQA, IFEval, MMLU-Pro, MATH-500. Today it shows "mmlu" and "ARC-C arc_challenge".
3. **Both pickers apply instantly.** Today Benchmarks applies instantly and Models needs Apply. Drop the Apply button.
4. **A custom table numbers its rows 1, 2, 3,** not with their overall ranks. The line says "3 chosen · 2 tested" when a chosen model has no scores.
5. **The picker panel stays inside the page card.** Today it runs past the right edge.
6. **List benchmarks that haven't run yet**, greyed with "not run yet". Today IFEval, MMLU-Pro and MATH-500 are missing from the picker until something has run them.
7. **Improve's intro** says "What the Knowledge exam **and Everyday tasks** say this model is missing…".
8. **Improve's header line** ("the AI local/chat · today local (generator, judge, exam writer): 47 items, no daily limit") is replaced by one plain line from 12i.1: **"AI: local gemma · change"**. Until 12i.1 lands, just drop the system words.

**HANDOFF § 5b rule, and each PR says which applies:**
- **Code only (the image doesn't change):** skip deploy step 3. CI already ran the unit tests on that commit.
- **The image changes:** run step 3. It's the only test of the real image.
- **Step 4 always.**

---

# 12i.1 — AI models, and the judge test

## 1. The AI models page

Put it under the user menu (masein ▾ → **AI models**), with one row per job:

| Job | What it does today | Suggested default |
|---|---|---|
| **Judge** | marks Knowledge exam answers 0–4, and the few Everyday questions with a judge check | `deepseek/deepseek-v4.1-flash` |
| **Question writer** | writes new questions (12i.2) | `z-ai/glm-5.3` |
| **Training-data writer** | writes Improve's documents and chat examples | `z-ai/glm-5.3` |
| **Checker** | answers generated questions without seeing the reference (12i.2) | `openai/gpt-6-luna` |

Each row shows the current model, its price per million tokens in and out, and **change ▾**:
- **Choices:** "Local (gemma on this server)", plus OpenRouter's text models, with a search box.
- **The list is fetched live** from OpenRouter's models endpoint, with name, price and context, and cached for a day.
- **The suggested model sits at the top**, marked "suggested", with one line why ("cheap and strong as a judge when given a reference answer").

**Warnings,** each one line, shown only when true:
- the same family is doing two jobs (the judge and the writer both GLM, say): "a judge tends to favour its own family's style";
- a job's family matches a model on the board being improved (a Qwen writer while improving a Qwen model);
- the **training-data writer** is a model whose terms restrict training on its output (OpenAI, Google, Anthropic): "check the terms — its output becomes training data. Open-weight models (GLM, DeepSeek, Qwen) avoid this." Keep this as a small list in the code, easy to edit.

**Pinning,** so a model can't change quietly:
- save the exact slug OpenRouter returns;
- if it offers a dated version, save that, never a "latest" alias;
- pin the **provider**: pick the first provider OpenRouter lists for the model, save it, and send it with `allow_fallbacks: false`. A different provider can run a different precision, and that shifts marks. The row shows the provider in small text.

**Spend:**
- A **monthly limit in $**, default $20.
- The page shows this month's spend so far, from OpenRouter's usage figures or our own count of tokens times price.
- At the limit, AI jobs wait with a plain message; they never silently fall back to another model.

**The key:**
- `OPENROUTER_API_KEY` lives in the server's environment (docker-compose `.env`), never in the repo or the page.
- With no key, the page says so in one line, and only Local is offered.

The line on Improve and on the Knowledge exam (12i.0 #8) reads, for example, **"AI: judge DeepSeek V4.1 Flash · writer GLM 5.3 · change"**.

## 2. Changing the judge starts a new judge version

A score is only comparable with scores from the same judge.

- **A judge version** = model slug + provider + judge prompt hash. It's recorded on every judged answer.
- Model pages, tables, Improve and Home show judged scores from **the current judge version only**. Older ones go to the model's History, labelled "judged by <model>".
- **Changing the judge asks first:** "Re-judge the N answers on file with the new judge? About $X." Choosing Yes queues a re-judge, which uses no GPU; Later keeps the old scores in History until then.

## 3. The judge test

This is how masein picks a judge, and how the board earns the right to drop "provisional judge".

**Step 1, masein marks:**
- The page shows answers already on file: a mix of exam topics and models, plus the Everyday questions that use a judge. 100 by default; the number is a setting.
- One at a time, it shows the question, the criteria and the answer. masein gives 0–4 (for Everyday, pass or fail), with keys **0–4** or **P/F**, and **S** to skip.
- The judges' marks are hidden while he marks.
- Progress saves itself, so he can stop and come back.

**Step 2, candidates judge the same answers:**
- masein ticks up to 4 candidates from the dropdown. Suggest DeepSeek V4.1 Flash, GPT-6 Luna, GLM 5.3 Flash, and the local gemma.
- The cost is shown before running.
- Each candidate marks the same answers with the board's judge prompt.

**Step 3, the result, one table:**

| Judge | Same mark as you | Within 1 point | Agreement (weighted κ) | Cost per 1,000 answers |
|---|---|---|---|---|

- The best row is marked. **"Use this judge"** makes it the judge, which starts a new judge version (§2).
- **Removing "provisional":** when the current judge has a weighted κ of **0.7 or more** against masein on at least 100 answers, "provisional judge" comes off. Judged scores can then be ranked, per the existing rules.
- The badge says what it rests on: "judge checked against masein on 120 answers · κ 0.78".
- The threshold is one setting.

## 12i.1 tests

Use a fake OpenRouter in all of them.
- The model list loads, shows prices, and marks the suggested model.
- Saving a model saves the exact slug and provider, and requests send `allow_fallbacks: false`.
- Each warning shows only when its condition holds.
- At the spend limit, jobs wait and the page says why.
- A new judge version hides old judged scores from current views and keeps them in History.
- The judge test computes exact, within-1 and weighted κ correctly on a fixture (hand-checked numbers). "Provisional" comes off only at κ ≥ 0.7 on ≥ 100 answers.
- With no API key, only Local is offered, and nothing crashes.

---

# 12i.2 — the question builder

## 4. Where it lives, and what it makes

- **Knowledge exam → Questions**, and **Everyday tasks → Questions**, each get **"Build questions"**, next to the existing **"Upload a bank"**, which stays for Dr. Hossein's files.
- The output is **the same format as each bank today**:
  - Knowledge: a question plus criteria in the existing exam format;
  - Everyday: a question, reference and checks in the Everyday format.
- Published questions join the bank as a **new bank version** and split into practice and hidden halves as usual.

## 5. Three steps on one page

**Step 1 — What:**
- **Kind:** Knowledge or Everyday.
- **Topic:**
  - Knowledge: a topic, with suggested subtopics masein edits.
  - Everyday: a group, from the eight, or "new group".
- **Level** (Knowledge only): general public or specialist.
- **How many:** any number, typed. Under about 40, one note: "fewer than 40 won't give this topic its own score in Improve". It's advice, not a block.
- **Writer and checker:** from the AI models page, and changeable here for this batch.
- **Check for duplicates:** on by default, with a toggle. Against: this bank, earlier generated batches, and each other.
- **Edit the writing instructions ▸:** closed by default. It opens pre-filled with the default prompt:
  - Everyday: `docs/prompts/phase-12a4/everyday-question-prompt.md`;
  - Knowledge: a new default, below.

  The **output-format section is locked**, so edits can't break reading the result. There's "Reset to default", and the prompt used is saved with the batch.
- **Estimated cost**, shown before anything runs: "about $0.40".
- One main button: **Try 10**.

**Step 2 — Try 10:**
- The writer makes 10. masein sees one at a time and chooses **A**ccept, **E**dit or **R**eject, with an optional reason chip: "too easy", "two right answers", "trivia", "unclear", "wrong answer" or "other".
- The reasons and edits go into the prompt for the rest, as a short "avoid / do more of" list.
- **Make the rest** (the main button) is enabled after at least 5 have been reviewed.

**Step 3 — Make the rest, checked:**
- **Everyday, every question:**
  - its reference passes its own checks;
  - for summarising and shortening, pasting the source back fails;
  - **the checker writes its own answer without seeing the checks, and that answer must pass.** A failure is flagged: "the checker's answer failed: 'high' vs 'urgent'".
- **Knowledge, every question:**
  - the checker answers without the reference;
  - the judge marks that answer against the criteria;
  - under 3/4 is flagged: "the checker answered X; the criteria expect Y";
  - the checker also flags questions it finds ambiguous, time-sensitive or trivia.
- **Duplicates:** near-duplicates are flagged side by side ("looks like #214"), with keep new, keep old, or keep both. Use an embeddings model through OpenRouter with a cosine threshold of 0.9 (one setting), plus the existing 13-gram check. With no API key, use the 13-gram check alone.
- **Review:** all flagged questions must be cleared, plus a random 10% of the unflagged. Same keys as step 2.
- **Publish** makes the new bank version. Every question records: written by <writer>, checked by <checker>, approved by masein, batch id, prompt hash.

**Throughout:**
- Drafts save themselves, so masein can leave and come back.
- Progress shows while generating: "34 of 60 written · 5 flagged".
- A batch can be cancelled. What's written so far is kept as a draft.

## 6. The default Knowledge prompt

Write it into `docs/prompts/phase-12i/knowledge-question-prompt.md`, beside the Everyday one. It asks for:
- questions a curious adult would ask, answerable in 2–5 sentences;
- **one clear correct answer**, stable over time, with no news, prices or "current" facts;
- no trivia (dates, names for their own sake) and no textbook drills;
- criteria that say what a full answer contains, in 3–5 points, so the 0–4 judge has something to check;
- at the level chosen; English; neutral (no one country's institutions unless the topic is about them).

It also shows two good and two bad examples, in the style of the Everyday prompt.

## 12i.2 tests

Use a fake writer, checker, judge and embeddings.
- **Everyday:** a question whose blind answer fails its checks is flagged. One whose reference fails its own checks is rejected before review.
- **Knowledge:** a checker answer the judge marks 2/4 flags the question.
- **Duplicates:** a near-duplicate of a bank question is flagged, and "keep old" drops the new one. With the toggle off, nothing is flagged.
- The locked output section can't be edited, while the rest can, and "Reset to default" restores it.
- The count is free, with the under-40 note.
- Publish makes a new bank version with provenance on every question.
- A draft survives a page reload.

---

## Done when

1. **12i.0:** the eight polish items are live, and HANDOFF says when step 3 is needed.
2. **12i.1:** masein can pick each job's model from a live OpenRouter list, with prices and warnings. A changed judge starts a new judge version. The judge test runs, shows agreement, and can make a judge current.
3. **12i.2:** masein can build 60 Everyday or Knowledge questions in three steps, review only what's flagged plus a sample, and publish a new bank version.
4. Each PR has its CI run link, and deploy step 4 passes.

## Deploy steps, for masein

As HANDOFF § 5b. **Before 12i.1**, add the key on the server once:

```
cd ~/benchmarks/aienh && echo 'OPENROUTER_API_KEY=sk-or-...' | sudo tee -a .env >/dev/null && sudo docker compose up -d
```

Put the exact line in the PR. The key must never appear in logs or pages.
