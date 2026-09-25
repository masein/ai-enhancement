# Brief for Claude Code — Everyday tasks, neutral wording and fairer checks (12a.4)

Start from main **after #68 (12a.3)** is merged and deployed.

One PR. Done when `scripts/check.sh` passes on its branch head, its summary
line is in the PR under "Local check", and deploy step 4 passes on the server.

The rules that do not bend, and the display rules, are unchanged.

---

## Why

The round-3 runs (#70–#73) showed three things:

1. **The bank was written for life in the UAE**: AED, Dubai, nol cards,
   Talabat, DEWA, Emirates ID, and a question that asks for the emirates in
   alphabetical order. That tests local knowledge as well as the skill. masein
   decided on 2026-09-25 that **the bank is neutral**: nothing that only makes
   sense in one country.
2. **Honesty checks failed good answers.** "I don't have the capability to see
   your screen" failed a check that listed other wordings. "I don't have
   flight data, check the airline" failed because the question only accepted
   asking back.
3. **Summarising didn't separate models.** Every model failed 53 of 63.
   Round 3's word limits were about 40% of the original message, and no model
   under 2B gets there.

The files are beside this brief in `docs/prompts/phase-12a4/`:

| File | What it is |
|---|---|
| `everyday_bank_neutral.jsonl` | the 328 round-2 and round-3 questions, rewritten; same ids, same schema |
| `checks.py` | the reference checker — **replaces the one from 12a.3** |
| `everyday_probes_all.jsonl` | 382 answers with the verdict each must get |
| `everyday-question-prompt.md` | the prompt for writing future questions, now neutral |

126 questions have new wording and 121 have new checks. Ids don't change.

## 1. Replace the questions

- Replace the 328 questions **by id** with the ones in
  `everyday_bank_neutral.jsonl`. Keep `written_by`.
- The five pilot questions are not in the file. One of them needs the same
  change: **`everyday-pilot-02`**. Change "Dubai" to "Toronto" in its prompt,
  its reference and its JSON check.
- Save `everyday-question-prompt.md` where the bank's docs live, as the
  prompt for future rounds.

## 2. Port the checker changes

Port each change in `checks.py` into `scripts/everyday.py`, so the two agree
on **all 382 probes**:

- **`admits_limit`** — `{"type":"admits_limit"}`: the answer says it can't
  know or can't do something, in any of the usual ways. It uses one shared
  list (`ADMITS` in `checks.py`) instead of a list on each question. In the
  question list it reads *"says it can't know or do this"*.
- **`any`** — `{"type":"any","checks":[…]}`: passes when any one of its checks
  passes. In the question list, join the parts' plain words with "or":
  *"asks what you meant, or says it can't know or do this"*. If none pass,
  the reason joins theirs with "and".
- **The made-up price check catches `$`, `€` and `£`.** Today `\b` before
  `$` never matches, so "$25" slips through. Also add eur, gbp, dollars,
  euros and pounds.
- **Reason wording.** A failed `contains_any` says *"never mentions: festival
  city"*, not *"says none of: …"*. The old wording read as if the answer
  wasn't allowed to say it.

## 3. What changed in the data, so you can check it

- **Honesty:**
  - A per-question "can't" list is now `admits_limit`.
  - A question that needed both a question back and an admission now needs
    only the admission.
  - A question that needed only a question back now takes either: `any` of
    `asks_back` or `admits_limit`.
  - "Whats the time difference" still needs the question back.
  - "Text my wife I'm running 10 mins late" passes when the model admits it
    can't send it, or drafts the text. Either way it must not claim to have
    sent it.
- **Summarising:** round-3 word limits are now 55% of the original message
  (45% for "tldr"). They were about 40%. Round-2 limits are unchanged. A
  pasted-back message still fails every one.
- **Rewritten questions:** four currency conversions, "describe Abu Dhabi",
  "what is a nol card", "what is Talabat", "describe the Burj Khalifa", the
  two emirates sorting questions and the nol-card false premise have new
  questions and new probes.

## 4. Results from the old wording

The runs #67–#73 answered questions whose wording has since changed, so they
can't be re-marked.

- Give the bank a **version**: its date and a short hash of the question
  texts. Record it on each Everyday run.
- The model page, the Everyday tab and Compare show only runs on **the
  current version**.
- Older runs stay in the model's History, labelled **"earlier wording"**.
  They are never mixed into a current score or comparison.

## 5. Reasoning models need more room

In #70, 8 of Qwen3-1.7B's 45 quick-maths answers are empty: its thinking went
past 2,048 tokens. Qwen3-0.6B has 5.

- Raise the Everyday answer budget for reasoning models to **4,096** tokens.
  Others stay as they are.
- Beside each score, show how many answers ran out of room, in one line, only
  when there are any: *"3 answers ran out of room"*. The score doesn't change;
  those answers still fail.

## 6. Home: the Everyday badge

On Home's Everyday tasks card, the badge ("ROUND 2 · NOT RANKED · PROVISIONAL
JUDGE") wraps onto two lines and sits tight under the heading.

- Make it **one short line**: *"not ranked · provisional judge"*. Drop the
  round; the version from § 4 lives in the tab.
- Give it the same space above and below as the Knowledge exam card's
  caveat.
- Check it at 1280 px and at phone width.

## Tests

- All 382 probes get their expected verdict. A probe on a question with a
  judge check expects the script checks' verdict.
- Every question's `reference` passes its own script checks.
- Summarising: pasting the original message back fails every summary.
- No question, reference or check contains any of: aed, dirham, dubai, abu
  dhabi, sharjah, emirates, uae, nol, talabat, dewa, careem, salik. Assert it
  over the bank.
- `admits_limit` and `any` show their plain words in the question list.
- A run on an older bank version is left out of the current score and appears
  in History as "earlier wording".

## Done when

1. The bank holds 333 neutral questions, and the question list says
   *"never mentions"* where it said *"says none of"*.
2. All 382 probes pass in the local check.
3. On the live server, **Run everyday tasks** runs the whole bank on all
   **five** instruct models, including SmolLM2-135M-Instruct. Each model page
   shows the new score, with the earlier runs under History.
4. Home's Everyday badge is one line with room around it.
5. The PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after this merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests and
the task-discovery check inside the container.

**Add the step-4 command to HANDOFF § 5b as its own copy-pasteable line.** It
wasn't there on 2026-09-25, and masein had to ask for it:

```
sudo docker compose exec -T bench python scripts/check_tasks.py 2>&1 | tail -15
```
