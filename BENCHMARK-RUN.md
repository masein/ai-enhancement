# Benchmark run — step by step

Written for **your** server: `teraformer-5090-3`, one RTX 5090 (32 GB, shared), Python
3.12.3, 32 cores, 186 GB RAM. Eleven models, all under 1B parameters, all of which fit
in the ~3 GB currently free on the card.

Copy-paste each block in order. After each one there's a **"you should see"** —
if you don't see it, stop there rather than continuing.

Time: ~20 min of setup, then a smoke run of ~15 min, then the real run.

---

## 1 · Set up the environment (~10 min, once)

```bash
mkdir -p ~/benchmarks && cd ~/benchmarks
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Now PyTorch. **This must be `cu128` or newer** — your RTX 5090 is Blackwell
(compute capability sm_120), and wheels built for CUDA 12.6 or earlier contain no
kernels for it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lm_eval
```

**You should see:** `Successfully installed torch-... lm_eval-...`

### Verify the GPU actually works

Not with `torch.cuda.is_available()` — that returns `True` even when no kernels match
your architecture, which is exactly why the wrong wheel is so confusing. Do real
arithmetic:

```bash
python - <<'EOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul OK:", float((x @ x).float().mean()))
EOF
```

**You should see:** `compute capability: (12, 0)`, `NVIDIA GeForce RTX 5090`, and a
number from the matmul.

**If instead you get** `CUDA error: no kernel image is available for execution on the
device` — you have the wrong wheel. `pip uninstall torch` and reinstall with `cu128`.

---

## 2 · Point the model cache at a volume with room

Your root filesystem has 17 GB free at 82% used, and `/data` sits on it. Filling root
on a shared machine breaks everyone's jobs. Your home is on `/home` with 488 GB free,
which is far more than these models need (all eleven together are under 5 GB):

```bash
# remove any earlier HF_HOME line first, so you don't end up with two
sed -i '/HF_HOME/d' ~/.bashrc
echo 'export HF_HOME=$HOME/hf-cache' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$HF_HOME"

# PROVE it is writable before anything tries to use it
touch "$HF_HOME/.probe" && rm "$HF_HOME/.probe" && echo "WRITABLE: $HF_HOME"
df -h "$HF_HOME"
```

**You should see:** `WRITABLE: /home/masein/hf-cache`, then a `/home` line with plenty
available.

That two-second probe is worth doing every time you point `HF_HOME` somewhere new. Skip
it and the failure surfaces later, from inside a library, as a stack trace ending in
`PermissionError: [Errno 13]` — see Troubleshooting.

*(If you later benchmark 4B+ models you may want `/data-03`, your 15 TB RAID array. Run
the probe against it first: `mkdir -p /data-03/$USER && touch /data-03/$USER/.probe`.
It's a shared volume, so you may well not have write access — check `ls -ld /data-03`.)*

---

## 3 · Hugging Face login

```bash
pip install -U huggingface_hub
hf auth login
```

Paste a token from <https://huggingface.co/settings/tokens> — read scope is enough.

**Nine of the eleven models are ungated** and will download immediately. The two Gemma
ones are gated: open <https://huggingface.co/google/gemma-3-270m> in a browser while
logged in and accept the licence. It's usually instant.

Test:

```bash
hf download EleutherAI/pythia-160m --quiet && echo "ungated OK"
hf download google/gemma-3-270m --quiet && echo "gated OK too"
```

**If the second one fails** with a 401 or `GatedRepo` — the licence isn't accepted yet.
Don't wait on it; the script will run the other nine and report the two Gemma models as
failures you can pick up later.

---

## 4 · Get the scripts

```bash
cd ~/benchmarks
# copy from your Mac, or clone if you've pushed it somewhere:
# scp -r ~/Developer/ai-enhancement masein@teraformer-5090-3:~/benchmarks/aienh
ls aienh/scripts/run_benchmarks.sh aienh/scripts/report_lm_eval.py
```

Confirm the task names exist in your harness version — they move between releases:

```bash
lm_eval --tasks list | grep -iE '^\s*(mmlu|hellaswag|arc_challenge|winogrande|piqa)\s*$'
```

**You should see** all five. If `mmlu` isn't listed under that exact name, check what
it's called (`lm_eval --tasks list | grep -i mmlu`) and set
`TASKS_OVERRIDE="..."` when you run.

---

## 5 · Smoke run first — always

```bash
tmux new -s bench            # so an SSH drop doesn't kill anything
cd ~/benchmarks
source .venv/bin/activate
./aienh/scripts/run_benchmarks.sh smoke
```

20 items per task. The scores are meaningless at that sample size and the script says
so — what you're testing is the plumbing.

**You should see** a header like:

```
GPU     : 3182 MiB free of 32607 MiB   (29425 MiB in use, 87% busy)
Sharing with:
          <user>  3537376  02:14:07  pt_main_thread
Backend : hf   batch=8   seed=1234
WILL RUN  (11): EleutherAI/pythia-14m EleutherAI/pythia-70m ...
```

then per model:

```
RUN   EleutherAI/pythia-14m  (base, hf, batch=8, 3182 MiB free)
  mmlu           5-shot ... 41s
  hellaswag      5-shot ... 22s
  ...
DONE     EleutherAI/pythia-14m in 2 min
```

**In a second tmux pane, watch the GPU:**

```bash
nvidia-smi -l 5
```

You want to see *your* python process appear in the Processes table with a few hundred
MiB. If your process never appears, the model loaded on CPU and you'd have discovered
that hours into the real run.

**Then confirm the report side works too:**

```bash
python aienh/scripts/report_lm_eval.py results/smoke -o /tmp/smoke_report.html
```

**You should see** `found N result file(s)` and a written HTML path.

---

## 6 · The real run

```bash
./aienh/scripts/run_benchmarks.sh full
```

**Estimate your own timing rather than trusting mine:** watch the first model's
`DONE ... in N min` line and multiply by 11. The models differ in size but they're all
tiny, so the total is dominated by dataset iteration (~26,500 items across the five
tasks), not by model forward passes. Expect a few hours, longer because you're sharing
the card at 87% utilisation.

If you need to stop it: Ctrl-C. It resumes **per model *and* per task**, so restarting
loses at most the one task in flight.

### While it runs, the four things worth watching

| Watch | Healthy | Not healthy |
|---|---|---|
| your PID in `nvidia-smi` | present, a few hundred MiB | absent → running on CPU |
| the per-task times | consistent per model | growing → GPU contention |
| `PARTIAL` lines | none | a model failed; check its log |
| free VRAM | stable | dropping → neighbour growing; the script will skip rather than crash |

---

## 7 · Build the report

```bash
python aienh/scripts/report_lm_eval.py results/full \
    -o artifacts/benchmark_report.html \
    --csv artifacts/benchmark.csv \
    --title "Sub-1B model benchmark — teraformer-5090-3, August 2026"
```

Copy it to your Mac to look at it:

```bash
# from your Mac:
scp masein@teraformer-5090-3:~/benchmarks/artifacts/benchmark_report.html .
open benchmark_report.html
```

One self-contained HTML file — no server, no assets. It contains scores with error
bars, a pairwise significance test, every metric including MMLU's 57 subjects, and a
provenance table. Send it to anyone.

---

## 8 · What the results will look like — read this BEFORE you see them

**Chance baselines.** A score at chance means the model knows nothing about the task.
These are arithmetic, not estimates:

| Task | Options | Chance |
|---|---|---|
| MMLU | 4 | **25%** |
| HellaSwag | 4 | **25%** |
| ARC-Challenge | 4 | **25%** |
| Winogrande | 2 | **50%** |
| PIQA | 2 | **50%** |

That Winogrande row is the one that catches people: **50% on Winogrande is a total
failure**, not a pass. Someone will misread it in a meeting. Have the number ready.

**What to expect from this particular set.** Two things, and both are findings rather
than problems:

1. **MMLU will be flat at ~25% across all eleven models.** MMLU requires broad
   knowledge that doesn't appear until models are in the low billions of parameters.
   Eleven flat bars is the correct result, and being able to say *why* before anyone
   asks is worth more than the numbers.
2. **HellaSwag will rise with size.** Commonsense sentence completion emerges much
   earlier than knowledge does. So you should get a rising HellaSwag curve next to a
   flat MMLU one — different capabilities emerging at different scales, in one chart.

**The Pythia models are the valuable part.** `pythia-14m` → `70m` → `160m` → `410m`
were trained by EleutherAI on **identical data in identical order**, differing only in
size. That makes those four a controlled experiment: parameter count is the only
variable, so the curve is a real scaling curve. Every other model on the list confounds
size with training data, recipe and vintage.

Given your team is running MoE *scaling* experiments (that 5 TB
`/data/moe-scaling-run-logs` mount), a clean scaling curve is the shape of result they
already care about.

**The SmolLM2 base/instruct pairs** are there deliberately: `SmolLM2-135M` and
`SmolLM2-135M-Instruct` share a lineage, one scored without a chat template and one
with. The gap between them is the instruction-tuning delta, measured rather than
asserted.

---

## 9 · One decision I made for you, and how to reverse it

I set few-shot counts to **5 for everything** (0 for PIQA), rather than the Open LLM
Leaderboard v1 conventions (MMLU 5, HellaSwag 10, ARC-Challenge 25).

Why: **Pythia's context window is 2048 tokens, and a 25-shot ARC-Challenge prompt
doesn't fit in it.** A prompt that gets silently truncated is a different question from
the one everyone else asked. So the choice was:

- **leaderboard counts** → comparable to published numbers for the long-context models,
  quietly wrong for the short-context ones
- **counts that fit every model** → internally consistent, not directly comparable to
  published leaderboards

For a scaling study the second is right, because the whole point is comparing these
models *to each other*. To switch:

```bash
LEADERBOARD_SHOTS=1 ./aienh/scripts/run_benchmarks.sh full
```

Either way, **state which you used** when you report. One line — "5-shot uniform,
`acc_norm`, greedy, lm-eval `<git hash>`" — pre-empts the entire class of "but I got a
different number". The report's provenance table has all of it.

---

## 10 · When the GPU frees up

Open `aienh/scripts/run_benchmarks.sh`, uncomment the billion-parameter models at the
bottom of the `MODELS` list, and run the **same command**:

```bash
BACKEND=vllm BATCH=32 ./aienh/scripts/run_benchmarks.sh full
```

Everything already measured is skipped; only the new models run. `BACKEND=vllm` is much
faster on larger models — I defaulted to `hf` because vLLM grabs a big KV-cache block up
front, which is antisocial while someone else is mid-training.

To queue politely behind the running job instead of checking manually:

```bash
MIN_FREE_MIB=12000 ./aienh/scripts/run_benchmarks.sh full --wait
```

It polls once a minute and starts the moment 12 GB is free.

---

## Troubleshooting

**`no kernel image is available for execution on the device`**
Wrong PyTorch wheel for Blackwell. Reinstall with `--index-url .../whl/cu128`.

**`PermissionError: [Errno 13] Permission denied: '/data-03/.../stored_tokens'`**
`HF_HOME` points at a directory you cannot write to. `/data-03` is a shared volume and
you may not own a directory under it. Check with `ls -ld /data-03 /data-03/$USER`, then
repoint `HF_HOME` at `$HOME/hf-cache` (Step 2) — remembering to **delete the old line**
from `~/.bashrc` rather than appending a second one — and run `hf auth login` again.
Nothing is lost; no token was stored.

**`401 Client Error` / `GatedRepoError`**
Accept the licence on the model page in a browser while logged in. Affects only the
two Gemma models here.

**`CUDA out of memory`**
The neighbouring job grew. Lower `BATCH=4`, or wait. The script re-checks free memory
before each model, so it usually skips rather than crashes.

**A model shows `PARTIAL`**
One task failed, the rest succeeded. Check `logs/<model>_full.log`. Re-running picks up
only the missing task.

**Scores don't match published numbers**
Expected. Different harness version, few-shot count, prompt format, or metric (`acc` vs
`acc_norm`) each move the number by a point or three. Internal consistency is what
matters, and that's what the fixed settings buy you.

**MMLU takes far longer than the others**
It's 14,042 questions versus ~1,200 for ARC-Challenge. Normal.

**Disk filling up**
`--log_samples` writes every prompt and completion. Worth it — it's the first thing
you'll want when a number looks wrong — but watch `df -h "$HF_HOME"`.

---

## What to say when you share it

1. **Lead with the scaling curve**, not the leaderboard. "Across a controlled 14M→410M
   ladder, HellaSwag rises from chance to X% while MMLU stays at chance throughout" is
   a finding. A table of eleven models is data.
2. **Name what's inside the noise.** The report tells you which pairs aren't
   distinguishable at this sample size. Those are the ones people would otherwise argue
   about.
3. **State the settings in one line.** 5-shot uniform, `acc_norm`, greedy, harness hash.
4. **Say what these benchmarks don't measure.** All five are multiple-choice: the model
   picks between given options and never produces anything. They say nothing about
   instruction-following, tool use, long context or generation quality. If anyone is
   about to choose a model on the strength of an MMLU score, that caveat is the most
   useful thing you'll say all week.
