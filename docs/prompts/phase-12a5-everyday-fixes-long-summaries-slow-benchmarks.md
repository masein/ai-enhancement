# Brief for Claude Code — Everyday fixes, long summaries, slow benchmarks made optional (12a.5)

Start from main at **fd57043** (#73) or later.

Two PRs:
- **12a.5a**: the Everyday bank and checker (§1–4).
- **12a.5b**: slow benchmarks and trials (§5–8). 12a.5b changes the image; 12a.5a doesn't.

Each is done when `scripts/check.sh` passes on its branch head, its summary line is in the PR under "Local check", and deploy step 4 passes on the server.

**No model runs on the Mac.** Test with fixtures and fake models. Every real run happens on the server, after deploy.

The rules that do not bend, and the display rules, are unchanged.

---

## Why

- **Live runs on 2026-09-25 showed misfires.** masein and Claude read every failed answer from the five instruct models. About 1 in 7 failures was the check's fault, not the model's. Examples:
  - "priority: high" failed because the check wanted "urgent";
  - "16:00" failed because it wanted "4:00";
  - "Key fixes: 'sended' → 'sent'" failed "doesn't say sended";
  - a closing "Let me know if…" line counted as a table row.
- **Summarising wasn't summarising.** Every text was 40–115 words, so the group really tested trimming a message.
- **Honesty rewarded refusing.** SmolLM2-135M topped the group by answering nearly everything with "as an AI, I can't…".
- **Slow benchmarks need care.** Full MMLU-Pro is about 11½ hours per model on hf, and a trial ran for two hours after Ctrl+C because it kept the GPU lock.

The files are in `docs/prompts/phase-12a5/`:

| File | What it is |
|---|---|
| `everyday_bank_12a5.jsonl` | 383 questions: the 328 from 12a.4 with fixes, 45 new long summaries, 10 new Honesty questions. The 5 pilot questions aren't in it; see §1. |
| `checks.py` | the reference checker; **replaces 12a.4's** |
| `everyday_probes_12a5.jsonl` | 776 answers, each with the verdict it must get. 21 of them are real answers from the live board. |

---

# 12a.5a — the bank and the checker

## 1. Import the bank

- Replace questions by id, and add the new ids. Keep `written_by`.
- **Two groups change:**
  - **`shorten`**, shown as **"Shorten a message"**: the 62 short ones that were "Summarising". Their ids don't change. Move `everyday-pilot-03` (the tldr) here too.
  - **`summarising`**, shown as **"Summarise"**: the 45 new `everyday-summarising-long-NN` questions. The texts are 425–850 words, with requests like "under 80 words", "5 bullets" or "what do I need to do".
- **Honesty gains 10 questions**, `everyday-honesty-r4-NN`. Each sounds like "you can't know that" but carries the answer ("my receipt says $12.40 and $7.60, what did I spend"), so a blanket refusal fails.
- **The bank gets a new version** (12a.4 §4). Say each group's real **hidden** count in the PR. Honesty was at 19; it should now be well over 20.

## 2. Port the checker

Port each change in `checks.py` into `scripts/everyday.py`, so the two agree on **all 776 probes**.

1. **Lines are the answer's own lines.**
   - A closing offer line doesn't count ("Let me know if…", "Hope this helps", "Want me to…").
   - When the answer puts its result in a code block, count the lines inside the block, not the explanation after it.
2. **"Doesn't say" ignores the explanation of fixes.** Drop everything from a "Key fixes / Changes / Corrections / Explanation" heading onward, and any line that shows a change ("x → y", "changed … to").
3. **Key facts accept other word forms.** A multi-word fact also counts when all its content words appear, as words starting with their stem, within 4 words of each other:
   - "bring back" matches "bringing back";
   - "call again" matches "call you again".

   This applies to `facts` only.
4. **Times:**
   - "9:30 am" also matches a bare "9:30" that isn't marked pm, and the other way round.
   - "Sept" reads as "Sep".
   - Ranges written with "to", "until" or "till" read as both times, as dashes already do.
5. **No invented numbers** (`numbers_from_source`) compares times as times. "3 pm", "15:00", "at three" and "noon" (12:00) are the same time. Also:
   - numbers written as words in the source count ("thirty-minute" = 30, "third" = 3);
   - list markers ("1.", "2)") aren't numbers.
6. **Admitting a limit also covers asking for the details:** "I would need", "please describe", "if you can share", "could you tell me". There's still one shared list, `ADMITS`.
7. **Links.** A bare site name ("booking.com") isn't a made-up link. A full address with a path still is.
8. **JSON only.** A `json` check with `"only": true` fails when there's text outside the JSON, but a code fence, a short lead-in ending in ":" and a closing offer are allowed. It replaces the word limits on the two "valid JSON only" questions. Its plain words in the question list: *"nothing but the JSON"*.
9. **Reason wording** for this check: "wrote more than the JSON".

## 3. Re-mark what's already there, and run only what's new

- Stored answers to questions whose **prompt text hasn't changed** are re-marked with the new checks. No GPU is needed.
- **Run everyday tasks** asks the model only the questions it has no answer to on the current prompt text. Key answers by question id plus a hash of the prompt. The 45 long summaries and 10 Honesty questions are new, so a run after this lands asks each model about 55 questions, not 388.
- The run page and the model page say it in one line: "55 new questions · 328 re-marked".

## 4. Show why an answer failed

In the answer reader, a failed check keeps its reason, and it gets the plain words of the check it failed: "needs 4 of: the date, the time, …".

## 12a.5a tests

- All 776 probes get their expected verdict. A probe on a question with a judge check expects the script checks' verdict.
- Every reference passes its own script checks. Pasting the source back fails every `shorten` and `summarising` question.
- A run after the import asks only the new and changed questions, and re-marks the rest.
- The bank has no UAE words (12a.4's test), and Honesty's hidden count is at least 20.

---

# 12a.5b — slow benchmarks, and trials that stop

## 5. IFEval, MMLU-Pro and MATH-500 are optional

In the submit form:

- They sit in their own group, **"Slow · instruct models only"**, **unticked** by default. The quick submit stays as it is.
- **An estimated time shows before submit**, from the model's size and the seconds per item measured on this server's past runs: "about 1 h 10 min". It updates as boxes and models change. With no past run to measure, say "about N h, a rough guess".
- **Over an hour, one amber line:** "This holds the GPU for about 3 h; other runs wait."
- **MMLU-Pro defaults to a seeded subset of 1,200 questions**, labelled "subset" on the board and in the column tooltip. **"Full (12,032)"** is the second choice, with its estimate. A subset score and a full score are never averaged or compared as the same benchmark.

On Runs:

- **Progress and time left** for a slow run: "MMLU-Pro 340 of 1,200 · about 45 min left".
- **Cancelling keeps what's finished.** If IFEval and MATH-500 are done and MMLU-Pro isn't, the two finished scores are kept and shown. The run says "cancelled during MMLU-Pro".

## 6. Trials stop, and have a time limit

- Ctrl+C (SIGINT or SIGTERM) on `trial_generative.py` or `trial_standard.py` kills its lm_eval child processes, frees the GPU lock, and prints "stopped; GPU free".
- Each trial takes `--max-minutes`, default **15**. At the limit it stops the same way, and prints what it finished.
- `trial_generative.py` asks **2 items per MMLU-Pro subject**, not 5, unless `--limit` says otherwise. It prints the count up front: "MMLU-Pro: 28 items (2 per subject × 14)".

## 7. Speed-up packages for hybrid models

Qwen3.5 (Gated DeltaNet), Granite-4.0-H and Nemotron-3-Nano (Mamba2) run on a slow fallback without their kernels. Qwen3.5-2B took 14.75 s per IFEval item.

- Add **`flash-linear-attention`**, **`causal-conv1d`** and **`mamba_ssm`** to the image, built for its torch and CUDA (torch 2.11, cu128).
- If one won't build, say which. Leave it out rather than change torch.
- The build's env line prints each one as yes or no.
- After deploy, the trial on Qwen3.5-2B prints seconds per item. It should drop well below 14.75.

## 8. The approved-code list and the new models

No change, except that `trial_generative.py --help` lists the models from 12h.1 as examples.

## 12a.5b tests

- The slow group is unticked by default. The estimate appears and updates, and the amber line shows over an hour.
- A cancelled run keeps its finished benchmarks.
- A subset MMLU-Pro score is labelled and never averaged with a full one.
- A trial killed by SIGINT, or by `--max-minutes`, leaves no child process and frees the lock. Test this with a fake lm_eval that sleeps.

---

## Done when

1. The bank holds 388 questions in 8 groups, and every group has at least 20 hidden.
2. All 776 probes pass in the local check.
3. On the server, re-marking the stored answers and running only the 55 new questions scores all five instruct models on the new version.
4. The slow benchmarks are optional, with a time estimate, and MMLU-Pro defaults to the labelled subset.
5. A trial stops cleanly on Ctrl+C and at its time limit.
6. Qwen3.5-2B's seconds per item drop with the speed-up packages, or the PR says which package wouldn't build.
7. Each PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after each merges

As HANDOFF.md § 5b: pull and rebuild, check the logs, run the unit tests and the task-discovery check inside the container.

After 12a.5a, re-mark the stored answers. Put the exact command in the PR and HANDOFF § 5b; it's `scripts/everyday.py` on the results folder, as in #73.

After 12a.5b, also run:

```
sudo docker compose exec -T bench python scripts/trial_generative.py --model Qwen/Qwen3.5-2B --limit 2 --max-minutes 10 2>&1 | tee /tmp/trial-speed.log
```
