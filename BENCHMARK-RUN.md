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
pip install lm_eval transformers accelerate sentencepiece
```

**Why transformers is spelled out:** `pip install lm_eval` alone does NOT install it —
the HF backend's dependencies are behind an extra, and without them every single
invocation dies instantly with `ModuleNotFoundError: No module named 'transformers'`
(exit 1 on every task, no download ever starts). `accelerate` is needed for device
placement; `sentencepiece` prevents a tokenizer failure on some model families.

**You should see:** `Successfully installed ...`, then verify the import chain:

```bash
python -c "import lm_eval, transformers, accelerate; print('harness deps OK')"
```

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

**Nine of the eleven models are ungated** and download immediately:

```bash
hf download EleutherAI/pythia-160m --quiet && echo "ungated OK"
```

### The two Gemma models are gated — and SSH-only is not a blocker

`Error: Access denied. This repository requires approval.`

**The gate is on your Hugging Face ACCOUNT, not on the machine.** You do not need a
browser on the server. Open this on your laptop, logged into the same account you made
the token with, and click **Acknowledge license**:

<https://huggingface.co/google/gemma-3-270m>

Google's model card says *"Requests are processed immediately."* Then, back on the
server, the token you already have works:

```bash
hf download google/gemma-3-270m --quiet && echo "gated OK too"
```

Do the same for <https://huggingface.co/google/gemma-3-270m-it> if you want the
instruction-tuned one (it is a separate repo and a separate click).

**If approval doesn't come through**, there are ungated re-uploads of the same weights.
I verified `unsloth/gemma-3-270m-it`: `gated: false`, architecture `Gemma3ForCausalLM`,
**268,098,176 parameters in bf16** — a plain mirror, not a quantization. Uncomment the
`unsloth/...` lines in `run_benchmarks.sh` and comment out the `google/...` ones.

One caveat: I can confirm the metadata matches, not that the weights are byte-identical
to Google's upload. For internal comparison it makes no practical difference; if you
publish numbers, say which repo they came from. The report's provenance table records
the exact `pretrained=` id, so this is handled for you.

**Or just skip Gemma entirely for now.** Nine models is a perfectly good first run, and
the four Pythia models — the controlled scaling ladder, which is the most valuable part
— are all ungated.

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
lm_eval --tasks list | grep -iE '^\s*(mmlu|hellaswag|arc_challenge|arc_easy|winogrande|piqa|truthfulqa_mc2|gsm8k)\s*$'
```

**You should see** all eight (verified present under exactly these names in
lm_eval 0.4.12). If one isn't listed under that exact name, check what it's called
(`lm_eval --tasks list | grep -i <name>`) and set `TASKS_OVERRIDE="..."` when you run.

---

## 4b · Optional but recommended: add corpus perplexity

The datasets people list when they talk about LLM data — FineWeb, C4, Dolma,
RedPajama, Common Crawl — are **pretraining corpora**: raw text, no labels, trillions
of tokens. You cannot score accuracy on them, because there is no right answer. They
are not benchmarks.

But they give you the one measurement MMLU and HellaSwag cannot, and it matters
especially for **this** run:

- **MMLU can't distinguish your models.** It sits at chance until the low billions, so
  all eleven will look identical. Bits-per-byte is continuous and will separate
  pythia-14m from pythia-410m cleanly.
- **It works on base models.** No chat template, no format to follow, nothing to parse.
- **Domain slices are the real product question.** Perplexity on code vs. legal vs.
  medical text tells you what a model is *for*. No public benchmark tells you that
  about your team's data.

`scripts/make_ppl_task.py` turns any corpus into a pinned lm-eval task:

```bash
cd ~/benchmarks

# a slice of a real pretraining corpus (streamed — no full download)
python aienh/scripts/make_ppl_task.py --hf HuggingFaceFW/fineweb-edu \
    --config sample-10BT --split train --field text \
    --n 300 --max-chars 8000 --name ppl_fineweb_edu

# a code slice, for a domain contrast (ungated, streams, field is "content")
python aienh/scripts/make_ppl_task.py --hf codeparrot/codeparrot-clean-valid \
    --split train --field content --n 200 --max-chars 8000 --name ppl_code

# and your team's own data — this is the one they'll actually care about
python aienh/scripts/make_ppl_task.py --local /path/to/corpus.jsonl \
    --field text --n 300 --name ppl_internal
```

Why `--n 300` and not thousands: rolling perplexity feeds **every byte of every
document** through the model, for **all eleven models**. 300 pinned documents is
minutes per model and plenty to rank them cleanly; the harness reports no standard
error for perplexity anyway, so extra documents buy less than they cost. The number
that matters for comparability is the **sha256 it prints**, not the sample size. (If
the code dataset errors on the field name, the script prints the fields it does
have — use that.)

Each writes two files into `eval_tasks/`: the sampled documents as JSONL (**the pinned
item set** — keep it; a perplexity number from a fresh random sample isn't comparable
to last week's) and the task YAML. It prints a sha256 of the item set; record it.

`run_benchmarks.sh` auto-discovers anything in `eval_tasks/` and runs it 0-shot, so
there's nothing to wire up. Verify one cheaply first:

```bash
lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m \
    --include_path eval_tasks --tasks ppl_fineweb_edu --limit 5 --device cuda:0
```

**You should see** three metrics: `word_perplexity`, `byte_perplexity`,
`bits_per_byte`.

**Quote `bits_per_byte`.** It's per byte, so it's tokenizer-independent — the only one
of the three you can compare across model families. Per-token perplexity is on a
different scale for every tokenizer, which makes cross-family comparison meaningless.
The report puts these in their own chart, labelled lower-is-better, and excludes them
from the significance test (they aren't proportions, and the harness reports no
standard error for them).

---

## 5 · Smoke run first — always

```bash
cd ~/benchmarks
source .venv/bin/activate
bash aienh/scripts/run_benchmarks.sh smoke
```

(`bash script.sh`, not `./script.sh` — the Mac→server sync does not preserve the
execute bit, and `bash` works either way. No tmux on this box; for anything long,
`nohup ... > run.log 2>&1 &` in step 6 survives an SSH drop the same way.)

20 items per task. The scores are meaningless at that sample size and the script says
so — what you're testing is the plumbing.

**You should see** a header like:

```
GPU     : 3182 MiB free of 32607 MiB   (29425 MiB in use, 87% busy)
Sharing with:
          <user>  3537376  02:14:07  pt_main_thread
Backend : hf   batch=8   seed=1234
Lock    : results/.run.lock (pid 12345 — a second run will refuse to start ...)
Tasks   : mmlu,hellaswag,arc_challenge,arc_easy,winogrande,piqa,truthfulqa_mc2,gsm8k
WILL RUN  (11): EleutherAI/pythia-14m EleutherAI/pythia-70m ...
```

That `Lock` line is new and it is load-bearing: **only one run at a time**. A second
invocation while one is alive prints `REFUSING TO START` with the owner's PID —
that's the script working, not breaking. (Two concurrent runs is exactly how the
first full run here produced duplicate DONE lines and OOM-killed its own tasks.) If
a run died hard (`kill -9`, reboot) the next invocation says `Stale lock ... taking
over` and proceeds by itself.

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
nohup bash aienh/scripts/run_benchmarks.sh full --wait > run.log 2>&1 &
tail -f run.log            # Ctrl-C stops the tail, NOT the run
```

`nohup ... &` keeps it alive when SSH drops; `--wait` makes it queue politely until
enough VRAM is free. To stop the actual run: `kill <pid>` (the pid is in the `Lock`
line and in `pgrep -af run_benchmarks`) — the lock cleans itself up and the in-flight
lm_eval is killed with it.

**Estimate your own timing rather than trusting mine:** watch the first model's
`DONE ... in N min` line and multiply by 11. Two cost notes: MMLU dominates the
multiple-choice tasks (14,042 items of the ~33,000 total), and **gsm8k is the one
generative task** — the model writes out a solution per item instead of scoring four
options, so on the batch-1 big-vocab models (gemma, Qwen) it can take longer than
every other task combined. It runs last per model, so the cheap results land first.

However it stops — `kill`, Ctrl-C on a foreground run, reboot — it resumes **per
model *and* per task**, so restarting loses at most the one task in flight. A model
counts as DONE only when **every** task has results; anything less shows up as
RESUMING with a `k/8 tasks done` count in the next run's header.

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

Copy it to your Mac to look at it (over Tailscale the hostname just works):

```bash
# from your Mac:
scp masein@teraformer-5090-3:~/benchmarks/artifacts/benchmark_report.html ~/Desktop/
open ~/Desktop/benchmark_report.html
```

Or, if you'll be regenerating it repeatedly, serve it over your tailnet — bind to the
Tailscale IP specifically, so it is reachable from your devices but not the server's LAN:

```bash
# on the server:
cd ~/benchmarks/artifacts
python3 -m http.server 8899 --bind "$(tailscale ip -4)"
# then on your Mac:  http://teraformer-5090-3:8899/benchmark_report.html
# Ctrl-C stops it. If `tailscale ip` needs sudo, get the IP from the Tailscale admin page.
```

One self-contained HTML file — no server, no assets, works from a `file://` open, an
email attachment, or `python -m http.server`. It's a full interactive dashboard:
tabs for **Overview / Leaderboard / Tasks / Perplexity & Loss / Runs**, a sortable
leaderboard, model search and base/instruct filters, per-task panels with chance
lines, cross-entropy loss derived from bits-per-byte, a query bar over every raw
metric (field filters + numeric comparisons, with filtered CSV export),
light/dark theme, and JSON export. The data is embedded in the file, so
regenerating after more models finish is the same one command.

---

## 8 · What the results will look like — read this BEFORE you see them

**Chance baselines.** A score at chance means the model knows nothing about the task.
These are arithmetic, not estimates:

| Task | Options | Chance |
|---|---|---|
| MMLU | 4 | **25%** |
| HellaSwag | 4 | **25%** |
| ARC-Challenge | 4 | **25%** |
| ARC-Easy | 4 | **25%** |
| Winogrande | 2 | **50%** |
| PIQA | 2 | **50%** |
| TruthfulQA (mc2) | weighted, multi-true | no clean chance level |
| GSM8K | generative | **0%** |

That Winogrande row is the one that catches people: **50% on Winogrande is a total
failure**, not a pass. Someone will misread it in a meeting. Have the number ready.

**What to expect from this particular set.** All of these are findings rather than
problems:

1. **MMLU will be flat at ~25% across all eleven models.** MMLU requires broad
   knowledge that doesn't appear until models are in the low billions of parameters.
   Eleven flat bars is the correct result, and being able to say *why* before anyone
   asks is worth more than the numbers.
2. **HellaSwag will rise with size.** Commonsense sentence completion emerges much
   earlier than knowledge does. So you should get a rising HellaSwag curve next to a
   flat MMLU one — different capabilities emerging at different scales, in one chart.
3. **ARC-Easy is the early mover.** Same format as ARC-Challenge, easier questions —
   it climbs well above chance even for these sizes, which is exactly why it's here:
   it separates the small models where ARC-Challenge still can't.
4. **TruthfulQA may go DOWN as models get better.** mc2 rewards not-endorsing common
   misconceptions; bigger models imitate popular text more fluently, misconceptions
   included. A flat or falling TruthfulQA next to rising everything-else is the
   famous inverse-scaling result, live on your own hardware. (It's also 0-shot **by
   design** — the task ships its own primer — so the report's few-shot warning
   ignores it.)
5. **GSM8K will be ~0–5% everywhere.** It's the one generative task: the model must
   write a worked solution and the harness extracts the final number
   (`exact_match`, flexible extraction). Written arithmetic essentially does not
   exist below a billion parameters. Zero is the informative, correct answer.

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

I set few-shot counts to **5 for everything** (0 for PIQA and TruthfulQA — the
latter is 0-shot by construction), rather than the Open LLM Leaderboard v1
conventions (MMLU 5, HellaSwag 10, ARC 25, TruthfulQA 0, GSM8K 5).

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
LEADERBOARD_SHOTS=1 bash aienh/scripts/run_benchmarks.sh full
```

Either way, **state which you used** when you report. One line — "5-shot uniform,
`acc_norm`, greedy, lm-eval `<git hash>`" — pre-empts the entire class of "but I got a
different number". The report's provenance table has all of it.

---

## 10 · When the GPU frees up

Open `aienh/scripts/run_benchmarks.sh`, uncomment the billion-parameter models at the
bottom of the `MODELS` list, and run the **same command**:

```bash
BACKEND=vllm BATCH=32 bash aienh/scripts/run_benchmarks.sh full
```

Everything already measured is skipped; only the new models run. `BACKEND=vllm` is much
faster on larger models — I defaulted to `hf` because vLLM grabs a big KV-cache block up
front, which is antisocial while someone else is mid-training.

To queue politely behind the running job instead of checking manually:

```bash
MIN_FREE_MIB=12000 bash aienh/scripts/run_benchmarks.sh full --wait
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

**`CUDA out of memory` on a *tiny* model (gemma-270m, Qwen3-0.6B)**
Not the weights — the logits tensor. For loglikelihood evals, memory ≈
`batch × seq_len × vocab × 4 bytes × ~2.5`. gemma-3's vocab is 262K, so batch 8 on
5-shot MMLU tries to allocate ~12 GiB from a model whose weights are 0.6 GB; Qwen3's
152K vocab tries ~6.5 GiB. Pythia and SmolLM2 (49–50K vocabs) are unaffected at
batch 8. The model list carries per-model batch sizes for exactly this; the blunt
override is `BATCH=1 bash aienh/scripts/run_benchmarks.sh smoke` — resume means only
the missing models re-run.

**`CUDA out of memory` on a model that fit before**
The neighbouring job grew between the check and the load (it's a race; one smoke run
here lost pythia-14m to exactly this). Just re-run — resume picks up only what's
missing. The script re-checks free memory before each model, so it usually skips
rather than crashes.

**`REFUSING TO START: another run of this script (PID ...) holds results/.run.lock`**
Working as intended — one run at a time. `ps -o pid,etime,cmd -p <PID>` to see what it
is; if it's a run you forgot about, either let it finish (`tail -f run.log`) or
`kill <PID>` (which also kills its in-flight lm_eval and releases the lock). Only if
the PID is genuinely dead and the message persists: `rm -rf results/.run.lock`.

**Duplicate `DONE` lines / `DONE ... in 0 min` / DONE then SKIP for the same model**
The signature of two runs racing each other (each one resuming past work the other
just finished) — possible only before the lockfile existed. Kill the strays and see
what actually exists on disk; results are per-(model, task) directories, so nothing
raced is corrupt, at worst incomplete:

```bash
pgrep -af "run_benchmarks|lm_eval"        # any stray runs still alive?
cd ~/benchmarks
for d in results/full/*/; do
  printf '%-45s %s tasks done\n' "$(basename "$d")" \
    "$(find "$d" -name 'results*.json' | wc -l)"
done
```

Then one fresh `run_benchmarks.sh full` fills every gap — the header's RESUMING line
shows exactly what it thinks is missing.

**A model shows `PARTIAL`**
One task failed, the rest succeeded. Check `logs/<model>_full.log` (the run prints the
failing lines inline too). Re-running picks up only the missing task.

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

1. **Lead with the trend**, not the leaderboard. "Across a controlled 14M→410M
   ladder, HellaSwag rises from chance to X% while MMLU stays at chance throughout" is
   a finding. A table of eleven models is data.
2. **Name what's inside the noise.** The report tells you which pairs aren't
   distinguishable at this sample size. Those are the ones people would otherwise argue
   about.
3. **State the settings in one line.** 5-shot uniform, `acc_norm`, greedy, harness hash.
4. **Say what these benchmarks don't measure.** Seven of the eight are
   multiple-choice: the model picks between given options and never produces
   anything. GSM8K is the lone generative task and the perplexity slices measure raw
   language modelling — between them you still have nothing on
   instruction-following, tool use, long context or open-ended generation quality.
   If anyone is about to choose a model on the strength of an MMLU score, that
   caveat is the most useful thing you'll say all week.
