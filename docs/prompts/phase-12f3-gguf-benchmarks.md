# Brief for Claude Code — Benchmarks measured on the GGUF (12f.3), plus small fixes (12i.4)

Start from main at **c8db23b** (#87) or later. There are two PRs:

- **12i.4**: small fixes found live on 2026-09-27 (§1–3). Do this one first; it's quick.
- **12f.3**: MMLU, HellaSwag, Winogrande, ARC and TruthfulQA measured directly on GGUF files with the fork's `llama-perplexity` (§4–9).

Each PR merges only on a **green `ci.yml` run on the mirror** (masein/ai-enhancement) for its rebased head, with the run's link in the PR. Also run `scripts/check.sh` locally if you want, but the mirror's CI is what counts.

**No model runs on the Mac or in CI.** For 12f.3, test with a fake `llama-perplexity`: a small script that prints canned output in the real format. **Nothing may call OpenRouter.**

The display rules (12b) are unchanged.

---

# 12i.4 — small fixes

## 1. Build questions: past batches

- **Under the three steps on the Build page,** add a short list of past batches: topic or group, date, writer, "7 of 10 written · 3 flagged · 7 published", and who ran it. Newest first, 10 shown, with **Show all** after that.
- **Each row opens the batch** (`#tab=build&draft=<id>`).
- **Today a batch's page shows only its summary.** Show **every question the batch wrote**, in the same reader style as the bank. For each question show:
  - the prompt;
  - the reference;
  - the checks or rubric;
  - your A/E/R review;
  - the checker's blind answer and whether it matched;
  - where it went: practice, hidden, or not published.
- **This page may show hidden questions,** because the person reviewing wrote and approved them. Everywhere else, the hidden rule is unchanged. The page is reachable only from the Build page and says so in one grey line: "Includes the hidden half — this batch's author view."
- **Knowledge exam readers:** in `read=bank:<topic>`, add a **written by** filter, so the questions one batch published to the practice half can be found.

## 2. Served models: answer length and running out

For a served model (and every other model that thinks), the Everyday and Knowledge blocks on the model page get two small mono numbers under the score:
- **median answer length** in tokens, thinking included;
- **ran out while thinking**: how many answers hit the reply budget with no answer written.

The **Compare** line against the base model shows both, for example: "Everyday 171 vs 176 · median 990 vs 400 tokens · ran out 0 vs 3".

Why: on 2026-09-27 the phone build (k=4 + LDA) answered a simple question correctly every time but thought about 2.5× longer than the original. The original once thought for 6,000 tokens and never answered. For a phone, both matter as much as the score.

## 3. The two chat settings, as measured

Set `CHAT_CPU_MAX_PARAMS_B` and `CHAT_GPU_MARGIN_GB` to the values masein reports after measuring on the server (Claude will send them). If they aren't in by the time this PR is ready, leave the defaults and say so.

## 3b. Found live on 2026-09-27 after #87

1. **Served models are hard to find.** A registered served model with no results isn't in Models ▸ "Not tested on this", and `#model=served/<id>` lands on Home instead of its page. List served models in "Not tested on this" with **Test**, and make their page open like any model's.
2. **The time estimate is wrong for served thinking models.** Test a model says "Everyday tasks — 388 questions, a few minutes"; the phone build measured 5.2 s an answer, about 34 min. For served models, use their measured seconds per answer once one run exists, and "about N min, a rough guess" before that.
3. **The "How it's served" placeholder mentions "lookahead 1".** That's not how either model is served, and it invites copying. Make it a neutral example: "llama.cpp build, quantisation, offload flags, routing".
4. **Everyday tasks table:** the note "3 answers ran out of room" runs into the TOTAL column ("99 of 2003 answers…"). Put it on its own line under the total, or in the row's tooltip.

## 12i.4 tests

- The Build page lists past batches, and a batch opens with all its questions, including hidden ones, with their review and checker result.
- The hidden half is still not reachable from any bank reader or API outside the batch's author view.
- The bank reader's written-by filter works.
- Median length and ran-out counts are computed from stored answers. The Compare line shows them.
- §3b: a served model with no results is listed with Test and its page opens; the estimate uses measured speed; the total and the ran-out note don't overlap at 1280 px.

---

# 12f.3 — measured on the GGUF

## 4. Why

masein wants MMLU and HellaSwag (and later others) for the phone build. The board measures those with lm_eval by comparing each answer choice's probability. A served chat endpoint can't give those probabilities, so #86 correctly leaves them out.

The team's llama.cpp fork includes **`llama-perplexity`**. It scores exactly these benchmarks straight from a GGUF file:
- `--hellaswag`;
- `--winogrande`;
- `--multiple-choice`, for MMLU, ARC and TruthfulQA in its binary format.

Its prompts and shot count differ from lm_eval's. So its numbers go in their **own column group** and are only compared with other GGUFs measured the same way. That's the comparison that matters here:
- the phone build: Qwen3.6-35B-A3B **k=4 + LDA**, UD-Q4_K_XL;
- the same file without the change: unsloth **MTP-GGUF UD-Q4_K_XL**, k=8.

The two are 0.7 MB apart, which is exactly the LDA tensors.

**Read the fork's source for the exact flags and output format.** It's on masein's Mac at `~/Developer/llama.cpp-teraformer`, tools/perplexity (branch `teraformer/lda-2026-09-22`, commit 91428471f). Don't guess the output format: parse what the source prints.

## 5. Where it runs

`llama-perplexity` is built on the host, outside Docker, in `~/llama.cpp-teraformer/build-lda`. It uses the CUDA runtime from `~/lda-env`. The board's container can't run it. So:

- **Add `scripts/gguf_worker.py`,** using the standard library only. masein starts it on the host with `nohup`, as he does llama-server (HANDOFF § 5d).
- **It watches `results/gguf_requests/`.** The results folder is bind-mounted, so the host and the container see the same files.
- **For each request file, it:**
  1. takes **the same run lock** as every run (`results/.run.lock`, mkdir-atomic, with the pid file), so it never overlaps a board run;
  2. runs `llama-perplexity` with the requested benchmark and file;
  3. writes progress and the result to `results/gguf_results/<id>.json`, with the full command, the binary's build string, the model file's name, size and sha256 (hashed once and cached by size + mtime), and the dataset file's sha256;
  4. releases the lock.
- **The board queues a request and shows its progress** like any run. It shows "The GGUF worker isn't running" (one line, with the HANDOFF command) when no heartbeat file has been written in the last 60 s.
- **Ctrl+C and a time limit behave as in the trials (12a.5 §6).** A stopped job keeps the benchmarks it finished.
- **GPU use:** use the same `-ngl 99 --cpu-moe` as the servers, as a per-model setting, so it fits next to the judge.

## 6. The data

- **Use the same questions as the board's lm_eval columns where possible:**
  - MMLU test (14,042);
  - HellaSwag validation (10,042);
  - Winogrande validation;
  - ARC-Challenge and ARC-Easy test;
  - TruthfulQA MC.
- **Converting lm_eval's datasets into `llama-perplexity`'s formats is the preferred source.** The multiple-choice binary format is documented in the fork's perplexity source. The HellaSwag and Winogrande text formats are in its README. A converter in `scripts/` keeps the questions identical to the other columns.
- **If a converter can't reproduce one exactly,** use the community validation files llama.cpp's docs point to, and record their source URL and hash.
- **The dataset files live in the results folder** (`results/gguf_data/`), with their hashes pinned. A changed file is a new dataset version, as for the Everyday bank.
- **The default is the full sets,** with a time estimate shown before starting (as for the slow benchmarks, 12a.5b §5). A **seeded subset** option (for example 2,000 per benchmark) is the second choice, labelled "subset", and never compared with a full score.

## 7. Registering a GGUF

- **A served model (#86) gets an optional field, "GGUF file on the server".** It's an absolute host path, such as `/home/masein/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf`.
- **A GGUF can also be added with no server,** in **Test a model → A GGUF file**. It asks for:
  - name;
  - path;
  - based on;
  - "how it's built", free text and required, as for served models;
  - the per-model flags.
- **The worker checks the path exists and records its hash before the first job.**

## 8. How it shows

- **On Models, under Standard,** a new column group sits after the others: **"Measured on the GGUF · llama.cpp, 0-shot"**, with MMLU, HellaSwag, Winogrande, ARC-C, ARC-E and TruthfulQA.
  - It's shown only for models that have a GGUF result.
  - Its tooltip says: "Scored by llama.cpp's llama-perplexity on the quantised file. Not comparable with the lm_eval columns to its left: different prompts and no examples."
- **These numbers never enter any average,** or the custom table's Avg, together with lm_eval columns. The custom table can pick them, but only against other GGUF columns, and it says so.
- **The model page's Scores → Standard** shows the GGUF group under its own heading, with each run's pinned details in History.
- **The On phone card (#87)** shows the measured GGUF MMLU next to the colleague's reported 81.98%, each labelled: "measured here (llama.cpp, 0-shot, full 14,042)" and "reported by <name>". They're never merged.
- **When two GGUFs share a "based on",** the model page shows the difference per benchmark, for example "k=4 + LDA vs original: MMLU −0.4, HellaSwag −0.9", with the usual significance test (the board's z-test on the two proportions). It says "not a clear difference" when it isn't one.

## 9. Later benchmarks

Keep the benchmark list in one table (name → mode, dataset file, flags), so adding one later (for example a Persian MC set, or MMLU-Pro as multiple choice) is a new row plus its dataset converter, not new code.

## 12f.3 tests

**With a fake `llama-perplexity`:**
- the worker takes and releases the run lock, and skips while a board run holds it;
- output is parsed into scores for each benchmark;
- Ctrl+C and the time limit keep finished benchmarks;
- a changed model file or dataset hash stops the job with a plain line;
- the heartbeat line shows when the worker is down.

**Converter tests on a few fixture questions:** the binary round-trips, and the questions match lm_eval's.

**Display tests:**
- GGUF columns never enter an lm_eval average;
- the custom table refuses to mix them;
- the reported and measured MMLU stay separate;
- the "not a clear difference" wording appears under the significance threshold.

---

## Done when

1. Past batches are listed on the Build page, and each batch shows all its questions to its author.
2. Served models show median answer length and ran-out counts, and the Compare line shows both.
3. The GGUF worker runs on the host under the run lock and measures the six benchmarks on both Qwen3.6 GGUFs. The full sets are the default.
4. Models shows the "Measured on the GGUF" group for those two, never averaged with lm_eval columns. The phone build's page shows the difference against the original, tested for significance.
5. Each PR merged on a green mirror CI run for its rebased head.

## Deploy steps, for masein, after each merges

As in HANDOFF § 5b. Say in the PR whether the image changed (step 3).

After 12f.3, add HANDOFF § 5d (the worker's start and stop commands) and start it:

```
cd ~/benchmarks/aienh && nohup python3 scripts/gguf_worker.py --results <the results folder> \
  --binary ~/llama.cpp-teraformer/build-lda/bin/llama-perplexity \
  --ld-library-path ~/lda-env/lib:~/llama.cpp-teraformer/build-lda/bin > ~/gguf-worker.log 2>&1 &
```

The PR states the exact results path and the converter command that fills `results/gguf_data/` once.
