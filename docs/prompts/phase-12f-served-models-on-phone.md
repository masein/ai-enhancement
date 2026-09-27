# Brief for Claude Code — Served models and the On phone kind (12f), plus two small fixes

Start from main after the Playground (12d) PRs, or after #81 if 12d is still in review. There are three PRs, in this order:

- **12f.0**: two small fixes found on 2026-09-26 (§1–2).
- **12f.1**: served models, which the board tests over an OpenAI-compatible endpoint it doesn't start itself (§3–7).
- **12f.2**: the **On phone** kind: the phone build's card and the numbers reported from the phone (§8–9).

Each PR is done when:
- `scripts/check.sh` passes on its branch head;
- its summary line is in the PR under "Local check";
- deploy step 4 passes on the server.

**No model runs on the Mac or in CI.** Test with a fake OpenAI-compatible server (a small FastAPI app in the tests that returns canned replies). **Nothing here spends OpenRouter money.**

The rules that do not bend and the display rules (12b) are unchanged.

---

## Why

The team's colleague delivered a phone model: **Qwen3.6-35B-A3B k=4 + LDA, UD-Q4_K_XL** (a 22.9 GB GGUF). It runs only on the team's llama.cpp fork (`teraformer/lda-2026-09-22`). A stock loader rejects the file, because the fork adds 82 `ffn_moe_out_{scale,bias}` tensors. So the board can't load it through transformers or vLLM the way it loads everything else.

On the GPU server, masein now runs it with the fork's `llama-server`, outside Docker:
- on port 8090 (8081 is taken by soft-label-explorer on the tailnet address);
- attention layers on the GPU and experts in RAM (`--cpu-moe`), next to the judge's vLLM;
- the phone's routing settings as environment variables.

The board needs to **test a model that something else serves**, and say clearly that it was tested that way.

---

# 12f.0 — two small fixes

## 1. The judge-test sample is mostly base-model loops

**The problem.** Of the 86 Knowledge exam answers in the current judge-test sample, about 52 are base models repeating one phrase ("The following is the following:…"). Any judge scores those 0, so they inflate agreement and leave few answers that are actually hard to mark.

**The fix:**
- Draw the sample mostly from instruct and chat models' answers.
- Cap degenerate answers at 10% of the sample. Degenerate means over 60% of the text repeats one 8-word n-gram, or the answer has no words beyond the question's.
- Keep the Everyday share as it is.
- **A new sample is a new judge-test version.** Keep the old marks and results under History, and say in one line: "the sample changed; mark the new one".
- **Don't delete anyone's marks.** masein's 100 marks (recorded as `claude`) stay with the old version.

## 2. A disk check on the status dot

**Why.** On 2026-09-26 the server's system disk hit 100% (Docker's images), and nothing on the board said so.

- **Add a check to the status dot.** Look at the free space of the filesystem that holds the results folder, and of `/` as the container sees it.
- Show **amber under 10 GB** and **red under 3 GB**, with one line: "The server's disk has 2.1 GB free. Runs may fail to save."
- Put the thresholds in config.
- **A run doesn't start below the red threshold.** It says why, in the same words.

## 12f.0 tests

- Sample builder: a fixture with 60% degenerate answers gives a sample with ≤10% of them. Old marks stay with the old version.
- Disk check with a faked `shutil.disk_usage`: it goes amber, then red, and a red disk blocks a run start with the line.

---

# 12f.1 — served models

## 3. Adding a served model

In **Test a model**, a fourth way sits next to the others: **"A model served elsewhere"**. The fields:

| Field | Example | Notes |
|---|---|---|
| Name | Qwen3.6-35B-A3B k4-LDA (phone build) | shown everywhere |
| Address | `http://host.docker.internal:8090/v1` | OpenAI-compatible base URL |
| Key | optional | stored like the OpenRouter key, never shown again |
| Based on | Qwen/Qwen3.6-35B-A3B | links it to the base model's page, if the board has it |
| How it's served | "llama.cpp fork teraformer/lda-2026-09-22 @ 91428471f, --cpu-moe, lookahead 1, fusion off" | free text, **required** |
| Thinking | on / off / the model decides | as the board's other reasoning models |

- **Check before saving.** Call `GET /v1/models`, plus `/props` and `/health` if they answer. Show what the server reports: model file, context size, and build if it's given. If nothing answers, say so in one line and don't save.
- **Pin what the server reports.** Save the model file name and size, the context size and the build string with the model. At the start of every run, compare them. If any changed, the run stops: "The server now serves a different file than the one registered. Register it again if that's intended."
- **The board never starts, stops or restarts the server.** HANDOFF.md § 5c holds masein's start and stop commands (below).

## 4. What it can be tested on

- **Everyday tasks: yes.** Send the questions as chat messages with the same settings as a local run, from the one shared function (12d §4). Use the same reply budgets, including `EVERYDAY_REASONING_MAX_GEN_TOKS` when thinking is on.
- **Knowledge exam: yes.** Same as above; the judge marks the answers as usual.
- **Standard, generative only: yes.** IFEval, MMLU-Pro (the seeded subset by default) and MATH-500, through lm_eval's `local-chat-completions`. GSM8K too, if it's asked as generation.
- **Standard, log-likelihood tasks (MMLU, HellaSwag, ARC, …): no.** Served chat endpoints don't return the prompt's log-probabilities. The model page says so once, in one line: "Multiple-choice benchmarks need the model loaded here; this one is served elsewhere." Don't show a column of dashes.
- **Improve: no.** Training needs the weights. The Improve tab isn't shown for a served model.

## 5. Runs

- **A served-model run takes the same run lock as any run.** It's one run at a time, because the server shares the GPU.
- **Requests go one or two at a time,** as a config value. llama-server is started with one slot by default.
- **Progress and time left work as for slow runs.** Time left comes from the measured seconds per answer.
- **If the server stops answering mid-run:** wait and retry for 2 minutes, then stop. Keep what's finished. Say "the server stopped answering at 140 of 200".
- **Each answer records** the served model's pinned details and the request settings, so History shows exactly how it was asked.

## 6. How it shows

- **On Models, a served model is a normal row.** A small grey tag, **served**, sits next to its name, and its tooltip carries the "How it's served" text.
- **The model page's History** shows the pinned server details with each run.
- **Scores from a served model are never averaged into another model's scores.** They sit next to the base model's scores for comparison. If the base model is on the board, the page shows: "Compared with Qwen3.6-35B-A3B loaded here: Everyday 171 vs 176."

## 7. Reaching the server from the container

- **Add `extra_hosts: ["host.docker.internal:host-gateway"]`** to the bench service in `docker-compose.yml`.
- **llama-server must listen where the container can reach it,** which is not only 127.0.0.1. HANDOFF § 5c starts it on the Docker bridge address with `--api-key`. Never on 0.0.0.0 without a key: the server is on the tailnet.

HANDOFF § 5c, to add verbatim (masein's working setup; paths are his):

```
# start (from ~/llama.cpp-teraformer; build-lda built with ~/lda-env)
GGML_CPU_DISABLE_FUSION=1 LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1 \
LD_LIBRARY_PATH=$HOME/lda-env/lib:$PWD/build-lda/bin \
nohup build-lda/bin/llama-server -m ~/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf \
  -ngl 99 --cpu-moe -c 16384 --jinja --host 172.17.0.1 --port 8090 --api-key "$LDA_KEY" \
  > ~/lda-server.log 2>&1 &
# stop
pkill -f build-lda/bin/llama-server
```

## 12f.1 tests

**With the fake server:**
- adding a served model reads and pins its details;
- a changed model file stops the next run with the line;
- a server that stops mid-run gives a partial, labelled result;
- Everyday and Knowledge runs go through the shared settings function;
- log-likelihood tasks aren't offered, and the one line shows;
- the Improve tab is hidden;
- the key is never returned by any endpoint.

**Elsewhere:**
- the compose file has the `extra_hosts` line;
- a served model's score is never in another model's average.

---

# 12f.2 — On phone

## 8. The phone build's card

- **When a served model's "How it's served" or a flag says it's a phone build,** the **On phone** kind appears, as 12b set aside: a fourth switch after Everyday tasks.
- **It's a card of numbers someone measured on the phone, typed in:**
  - device;
  - chip and RAM;
  - decode tokens per second (median and best);
  - the settings (streaming, lookahead, MTP);
  - the date;
  - who measured it.
- **Every number is labelled "reported by <name>".** The board never computes or estimates phone speed. The server's own speed isn't shown here: the fork's README says CUDA throughput doesn't represent the phone.
- **Next to it: the quality scores the board measured through the served model** (Everyday, Knowledge exam, generative Standard), each with the base model beside it.

The first card's data, from the fork's README (masein will confirm with the colleague):

| Field | Value |
|---|---|
| Device | OnePlus 15, Snapdragon 8 Elite Gen 5, 16 GB RAM |
| Settings | experts streamed from flash, lookahead 1, MTP n_max 3, fusion off |
| Decode | 13.5 tok/s median, 16.0 best (3 cold repeats, ≤65 °C) |
| Quality reported | MMLU 81.98% (all 14,042), measured on Metal |

## 9. Reported and measured are never mixed

- **The reported MMLU is shown as reported, with its source.** It's not put in the Standard column, which the board measured itself, and it's not averaged.

## 12f.2 tests

- The On phone switch appears only when a phone build exists.
- Reported numbers carry their name and date, and never enter an average or a board-measured column.

---

## Done when

1. The judge-test sample is mostly instruct answers, with ≤10% loops. The old marks are kept.
2. A full disk turns the status dot red and stops new runs, with one plain line.
3. The LDA phone model, served by masein's llama-server, is on Models as **served**, scored on Everyday tasks, the Knowledge exam and the generative Standard benchmarks, next to its base model.
4. On phone shows the phone's reported speed and quality, labelled, beside what the board measured.
5. Each PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after each merges

As in HANDOFF.md § 5b. 12f.1 changes `docker-compose.yml` (`extra_hosts`), so run `docker compose up -d` again. The image itself doesn't change, so step 3 can be skipped. Before adding the served model on the board, start llama-server as in § 5c.
