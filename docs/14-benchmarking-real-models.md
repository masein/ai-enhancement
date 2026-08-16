# 14 — Benchmarking real models on a GPU server

Downloading real open models, running MMLU / HellaSwag / friends against them, and
producing a report your team can read.

Two scripts do the work: `scripts/run_benchmarks.sh` (runs the harness with settings
held constant) and `scripts/report_lm_eval.py` (turns the output into a self-contained
HTML report with standard errors and significance testing).

**What I verified, and what I didn't.** The CLI flags, model IDs and install commands
below were checked against current documentation and the HuggingFace API in August
2026. Timings are estimates from typical hardware — I have not run this on your
server, so treat the clock figures as rough.

---

## Step 0 — find out what you're working with

SSH in, then:

```bash
nvidia-smi
```

The three things that matter, all on the first screen:

```
| NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4 |
|   0  NVIDIA A100-SXM4-80GB    ...    0MiB / 81920MiB  |     0%   Default |
```

- **Memory** (`81920MiB` → 80 GB) — decides which models fit.
- **How many GPUs** — one block per GPU. Count them.
- **CUDA Version** — needs to be ≥ 12.1 for current PyTorch wheels.

More detail in one line:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
```

What your memory buys you, in bf16 (2 bytes per parameter, plus ~20-30% for
activations and the KV cache):

| VRAM | Comfortable up to | Notes |
|---|---|---|
| 16 GB | ~7B | 4B is comfortable; 8B is tight |
| 24 GB | ~13B | Everything in this guide fits easily |
| 32 GB (RTX 5090) | ~14B | Comfortable for everything here — **if the card is free** |
| 40–48 GB | ~24B | vLLM really pays off here |
| 80 GB | ~35B dense, or a 35B-A3B MoE | Room to run several models without reloading |
| 2× 80 GB | ~70B | Set `TP=2` in the run script |

Also check:

```bash
df -h ~                    # model weights land in ~/.cache/huggingface — want 100GB+ free
free -g                    # system RAM; vLLM likes 32GB+
python3 --version          # 3.10+ required
nproc                      # CPU count, affects dataset preprocessing speed
```

**Run everything under `tmux`** so an SSH drop doesn't kill a six-hour job:

```bash
tmux new -s bench          # later: tmux attach -t bench
```

### If the GPU is shared — read the Processes table

The bottom half of `nvidia-smi` is the part people skip, and on a shared box it is
the part that decides whether you can work at all:

```
|    0   N/A  N/A   3537376      C   VLLM::EngineCore              13238MiB |
|    0   N/A  N/A   3635598      C   ...conda3/envs/olmo/bin/python3  16132MiB |
```

Type `C` means *compute* — a real job. `G` means graphics (an X server, a few MiB,
ignore it). Add the `C` rows up and subtract from total: that is what is genuinely
taken. Then:

```bash
# who owns those processes, and how long have they been running?
ps -o user:16,pid,etime,cmd -p 3537376,3635598

# free memory in one line
nvidia-smi --query-gpu=memory.free,memory.total,utilization.gpu --format=csv

# watch until it frees up
watch -n 30 'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader'
```

**Two rules on a shared card.** First, `gpu_memory_utilization` in vLLM is a fraction
of **total** GPU memory, not free memory — set it to 0.85 while a colleague holds
29 GB and you either fail to start or start a fight over the same memory. Derive it
from what is actually free: `gpu_memory_utilization ≈ (free_MiB × 0.88) / total_MiB`.
`run_benchmarks.sh` does this automatically in its preflight.

Second, **talk to whoever owns the other process before you launch.** An OOM in your
eval costs you an hour. An OOM in their training run costs them a day, and it will be
your fault. The script refuses to start below `MIN_FREE_MIB` (default 10 GiB) for
exactly this reason, and `./scripts/run_benchmarks.sh full --wait` will block and
poll until the card frees up so you can queue behind them politely.

### Put the model cache on a big volume, before the first download

`df -h` usually shows a small root filesystem and a large data volume. HuggingFace
caches to `~/.cache/huggingface` by default, and a handful of models is tens of GB.
Point it somewhere with room, **before** you download anything:

```bash
export HF_HOME=/big/volume/$USER/hf-cache     # put this in ~/.bashrc
mkdir -p "$HF_HOME"
df -h "$HF_HOME"
```

Pick the volume with the most free space that you can actually write to — test with
`touch "$HF_HOME/.probe" && rm "$HF_HOME/.probe"`. Filling the root filesystem on a
shared machine breaks everyone's jobs, not just yours.

---

## Step 1 — environment

```bash
mkdir -p ~/benchmarks && cd ~/benchmarks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

### Pick the PyTorch build by GPU ARCHITECTURE, not just driver version

This is where RTX 50-series (Blackwell) trips everyone up. Blackwell consumer cards
report compute capability **sm_120**, and PyTorch wheels built for CUDA 12.6 or
earlier contain **no kernels for sm_120**. They install cleanly, `torch.cuda` reports
the GPU correctly, and then the first real operation dies with:

```
CUDA error: no kernel image is available for execution on the device
```

You need a **CUDA 12.8 or newer** build:

```bash
# RTX 50-series / Blackwell (sm_120) — REQUIRED
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Ampere / Ada (RTX 30xx, 40xx, A100, L40S) — cu126 or cu128 both fine
# Hopper (H100) — cu126 or newer
```

Your *driver's* CUDA version (the number `nvidia-smi` prints top-right) only needs to
be ≥ the wheel's. It is not the thing to match on. A driver reporting CUDA 13.2 will
happily run a cu128 build; it will not rescue a cu124 build on a Blackwell card.

Then the harness:

```bash
pip install "lm_eval[vllm]"
```

If `[vllm]` fails, install separately (`pip install lm_eval vllm`). vLLM has had a
long tail of Blackwell issues — if it errors on sm_120, use `BACKEND=hf`, which is
slower but has no kernel requirements beyond PyTorch's.

### Verify with a real operation, not `is_available()`

```bash
python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))   # (12, 0) on a 5090
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul OK:", float((x @ x).float().mean()))
EOF
```

**`torch.cuda.is_available()` returns True even when no kernels match your
architecture.** That is exactly why the wrong wheel is so confusing: every check
passes until you do arithmetic. The matmul above is the check that actually fails.

Finally confirm the task names:

```bash
lm_eval --tasks list | grep -iE '^\s*(mmlu|hellaswag|arc_|winogrande|piqa|gsm8k|truthfulqa)' | head -30
```

Not optional ceremony — **task names move between harness versions**, and it takes
two seconds to confirm the ones you are about to depend on exist in your install.

---

## Step 2 — HuggingFace access

```bash
pip install -U huggingface_hub
hf auth login          # older versions: huggingface-cli login
```

Paste a token from <https://huggingface.co/settings/tokens> (read scope is enough).

**Gated repos will block you.** Gemma and Llama both require accepting a licence on
the model page while logged in, in a browser, before the download works. Gemma is
usually instant; Llama can take longer to approve. Qwen models are Apache 2.0 and
ungated.

Practical consequence: **start with Qwen** so you're running within minutes, and add
Gemma once access clears.

Test one download:

```bash
hf download Qwen/Qwen3-0.6B --quiet && echo "download works"
```

---

## Step 3 — pick the models

These IDs were confirmed against the HuggingFace API in August 2026, with download
counts as a proxy for "well-supported, unlikely to have surprises".

**Your 270M ask:**

| Model | Params | Gated | Note |
|---|---|---|---|
| `google/gemma-3-270m` | 270M | yes | Base model. Expect near-chance on MMLU — that is the correct result, not a bug. |
| `google/gemma-3-270m-it` | 270M | yes | Instruction-tuned variant of the same. |

**Your 4-5B ask:**

| Model | Params | Gated | Note |
|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | no | Strong, ungated, widely used. Best default. |
| `Qwen/Qwen3.5-4B` | 4B | no | Newer generation. |
| `google/gemma-3-4b-it` | 4B | yes | Good contrast with Qwen. |
| `microsoft/Phi-4-mini-instruct` | 3.8B | no | MIT licence. |
| `HuggingFaceTB/SmolLM3-3B` | 3B | no | Fully open training recipe — unusual and useful. |

**Worth adding for the shape of the curve** — a 270M-to-4B jump with nothing between
makes a boring chart:

`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`.

**If you have 80GB and want something directly relevant to your team's MoE
questions:** `Qwen/Qwen3.6-35B-A3B` is a mixture-of-experts model — ~35B total
parameters, ~3B active per token. Benchmarking it next to a dense model of similar
*active* size is the FLOP-matched comparison from `docs/04`, on real models.

Verify anything before you queue it:

```bash
hf download <model-id> --quiet && echo OK
```

---

## Step 4 — the smoke run (do this first, always)

```bash
cd ~/benchmarks
git clone <your-repo> aienh   # or just copy the two scripts across
./aienh/scripts/run_benchmarks.sh smoke
```

20 items per task, ~10 minutes total. The scores are meaningless at that sample size
and the script says so. What you are testing is the **plumbing**: downloads work,
the GPU is used, the task names resolve, chat templates apply, output files land
where you expect.

Watch `nvidia-smi -l 5` in a second pane. If GPU utilisation sits near zero, the
model loaded on CPU and you'd have discovered that six hours in.

Then build a report from the smoke output just to confirm that step works too:

```bash
python aienh/scripts/report_lm_eval.py results/smoke -o /tmp/smoke_report.html
```

---

## Step 5 — the real run

Edit the `MODELS` array at the top of `run_benchmarks.sh`, then:

```bash
./aienh/scripts/run_benchmarks.sh full
```

Rough timings per model on one modern GPU with vLLM (**estimates**):

| Task | Items | 270M | 4B |
|---|---|---|---|
| MMLU (5-shot) | 14,042 | ~5 min | ~15–25 min |
| HellaSwag (10-shot) | 10,042 | ~4 min | ~12–20 min |
| ARC-Challenge (25-shot) | 1,172 | ~1 min | ~3–5 min |
| Winogrande (5-shot) | 1,267 | ~1 min | ~2–4 min |

So five models across those tasks is roughly 2–4 hours. On the HF backend rather than
vLLM, multiply by 3–5.

The script is **resumable** — it skips any model whose results already exist, so
killing it and restarting costs you only the model in flight.

### The settings that are held constant, and why

Everything in the script is explicit because every one of these changes the score:

- **`--seed 1234`** on every run. Few-shot example selection is random; a different
  seed is a different exam.
- **Few-shot counts** pinned per task (MMLU 5, HellaSwag 10, ARC-C 25, Winogrande 5).
  These are the Open LLM Leaderboard v1 conventions — using them is what makes your
  numbers comparable to published ones. If your team prefers different values, change
  them in one place and re-run *everything*.
- **`--apply_chat_template` on instruct models only.** This is the big one. Applying
  it to a base model, or omitting it on an instruct model, moves scores by tens of
  points. The report flags it if you were inconsistent.
- **`--batch_size auto`** — the harness finds the largest batch that fits.
- **`--log_samples`** — writes every prompt and completion. Costs disk; it is the
  first thing you will want when a number looks wrong.
- **`dtype=bfloat16`** on every model, so you aren't comparing an fp32 run against a
  bf16 one.

---

## Step 6 — the report

```bash
python aienh/scripts/report_lm_eval.py results \
    -o artifacts/benchmark_report.html \
    --csv artifacts/benchmark.csv \
    --title "Small model benchmark — August 2026"
```

One self-contained HTML file. No server, no build step — attach it to an email or a
PR and it still works. It contains:

- **Scores by task**, with ±1 standard error whiskers on every bar.
- **"Is the difference real?"** — a two-proportion z-test on every pair, and a
  headline count of how many comparisons are *inside the noise*. This is the section
  that stops "our model beat theirs by 1.2 points" from becoming a slide.
- **Every metric including sub-tasks** — MMLU's 57 subjects, plus `acc` and
  `acc_norm` wherever both were reported.
- **Run provenance** — dtype, batch size, chat template, seed, harness git hash, wall
  clock, per model. Publish this next to the numbers or the numbers are hearsay.
- **Warnings at the top** if the runs aren't actually comparable — different few-shot
  counts, inconsistent chat templates, a `--limit` left in, or results from two
  different harness versions.

The CSV is there so someone can pull it into a spreadsheet without asking you.

---

## Step 7 — the things that will bite you

**Qwen3 thinking mode.** Qwen3 models are hybrid reasoning models whose chat template
can emit a long `<think>` block. That changes generation length, latency and scores.
The harness exposes `enable_thinking` and `think_end_token` in `--model_args`; if you
use one you must set the other. Decide deliberately which mode you're benchmarking
and *say which* in the report — a thinking-mode number and a non-thinking number are
different measurements of different things.

**Your numbers won't exactly match published ones.** Different harness version,
different few-shot seed, different prompt format, different metric (`acc` vs
`acc_norm`) — any one of these moves the number by a point or three. This is normal.
What matters is that your numbers are internally consistent, which is what the fixed
settings buy you. Say "measured with lm-eval `<hash>`, 5-shot, acc_norm" and nobody
can misread you.

**MMLU is 14k questions and HellaSwag is 10k**, so the standard errors are small
(~0.4%) and small differences *are* detectable. On ARC-Challenge (1,172 items) the
standard error is ~1.4%, so a 2-point gap there is not a result. The report does this
arithmetic for you.

**Out of memory.** Lower `GPU_UTIL` (0.85 → 0.7), lower `max_model_len` (4096 → 2048),
or switch to `BACKEND=hf` with a fixed small `--batch_size`. If a 4B model OOMs on a
24GB card something else is using the GPU — check `nvidia-smi` for other processes.

**Base models score near chance on MMLU and that is correct.** `gemma-3-270m` will
land around 25% (random is 25% on 4-way multiple choice). A 270M model does not know
much. The interesting thing about including it is the *shape* of the curve across
scale, not its absolute score.

**Disk fills up.** `--log_samples` writes every prompt and completion — hundreds of MB
per model across these tasks. Worth it, but watch `df -h`.

---

## Step 8 — what to actually tell people

The report gives you the numbers. The framing is yours, and this is where you show
the judgement:

1. **Lead with the comparison that answers a decision.** "Qwen3-4B matches
   Gemma-3-4b-it on HellaSwag and beats it by 10 points on MMLU, at the same active
   parameter count" is useful. A wall of numbers is not.
2. **Name what is inside the noise.** The pairs your report flags as
   indistinguishable are the ones people would otherwise argue about.
3. **State the settings inline.** "5-shot, `acc_norm`, greedy, lm-eval `a1b2c3d`."
   One line, and it pre-empts the entire class of "but I got a different number".
4. **Say what these benchmarks don't measure.** MMLU and HellaSwag are
   multiple-choice: the model picks between given options and never has to *produce*
   anything. They say nothing about instruction-following, tool use, long context or
   generation quality. If someone is about to pick a model for a product on the
   strength of an MMLU score, that caveat is the most valuable thing you'll say all
   week.

---

## Where to go after this

- **Add a generative task** — GSM8K (`--tasks gsm8k`) requires producing an answer,
  not picking one, so it exercises a completely different capability and a parser.
  The gap between a model's MC scores and its generative scores is itself a finding.
- **Add IFEval** for instruction-following, which is checked programmatically.
- **Measure throughput**, not just accuracy — tokens/sec per model at a fixed batch
  size. "Points per GPU-second" is a ranking almost nobody computes and everybody
  wants once they see it.
- **Wire it into the registry** in this repo (`src/aienh/registry.py`) so real-model
  results and your own training runs live in one leaderboard with the same lineage
  fields.
- **Run it on a schedule** against your team's checkpoints, and you have the
  regression gate from `docs/11`.
