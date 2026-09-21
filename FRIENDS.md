# The team benchmark — 5-minute guide

One place for the whole loop: **benchmark any model, track your training runs,
store your checkpoints** — on our shared GPU, on the tailnet.

Dashboard: **http://100.74.89.105:8899/** (you need to be on the tailnet —
ask Masein for an invite). This guide lives at `/guide` on the same host.

---

## 1 · Benchmark a model in 30 seconds (no code)

Open the dashboard → **Submit & Queue** → paste a model id → Submit.

- Any public Hugging Face id works: `HuggingFaceTB/SmolLM2-135M`
- `suite`: **quick** = hellaswag + arc-easy + perplexity, minutes — use while
  iterating. **full** = all 8 benchmarks + perplexity, up to an hour+ — use when
  a checkpoint matters. A quick run upgrades to full later by running only the
  missing tasks.
- `kind`: leave on **auto** (it detects chat templates; wrong templates move
  scores by tens of points).
- Put **your name** in the submitter field. The queue shows live progress; when
  it's done your model is on the Leaderboard with everyone else's.
- Already uploaded a checkpoint (§2)? Its `local/<name>` id goes in the same
  box.

Re-submitting an already-benchmarked model is free (results are cached per
task), and submitting something already in the queue just joins that run.

## 2 · Benchmark your own model — straight from your disk, no Hugging Face

This is the main path for most of us. The service has its **own checkpoint
storage**: upload any `save_pretrained()` directory under a name you pick, and
it becomes `local/<name>` — a first-class model on the leaderboard, no Hugging
Face account anywhere.

First grab the client (one stdlib-only file, no pip installs) straight from
the service — no GitHub access needed:

```bash
curl -o bench_client.py http://100.74.89.105:8899/client
```

(It's also `clients/bench_client.py` in the repo.)

Then one shell line uploads **and** benchmarks:

```bash
python bench_client.py --base http://100.74.89.105:8899 \
    upload my-model-v1 ./my_checkpoint_dir --submit --suite quick --submitter yourname
```

Or, from Python:

```python
from bench_client import Bench
bench = Bench("http://100.74.89.105:8899")
mid = bench.upload_artifact("my-model-v1", "./my_checkpoint_dir")   # -> "local/my-model-v1"
bench.submit(mid, suite="quick", submitter="yourname")
print(bench.scores(mid))     # once it's done — or just watch the dashboard
```

**Custom architecture?** If your checkpoint ships its own `modeling_*.py`
(an `auto_map` in `config.json`), add `--allow-remote-code`. Loading it runs
your Python, so it is opt-in per submission, uploads only (never Hub models),
and the server has to be configured for it. Those runs execute as an unprivileged
user with the Hub offline, and every `.py` is hashed into the run's provenance.
MoE models get both parameter counts on the leaderboard: total, and active per
token beside it.

Three rules, all enforced with readable errors: **one name per checkpoint**
(names are immutable — `my-model-v2` next time, no re-uploads); weights must be
**safetensors** (anything modern `save_pretrained()` writes is); storage is a
shared quota — `python bench_client.py --base … artifacts` shows who's using
what, `… delete <name>` frees space and your scores stay.

## 3 · Track your training run (2 lines in your loop)

With the same `bench_client.py` from §2, in your training code:

```python
from bench_client import Bench
bench = Bench("http://100.74.89.105:8899")

run = bench.init("my-run7", submitter="yourname",
                 config={"lr": 3e-4, "batch": 32})     # config shows + diffs in the UI
...
run.log({"loss": loss.item(), "lr": lr}, step=step)     # every step or every N — your call
...
run.finish()
```

Two useful conventions: put `batch_size` / `micro_batch_size` / `grad_accum` in
`config=` (they show and diff in the UI), and log a cumulative `tokens` metric —
the run list then shows total trained tokens.

Your curves appear **live** on the **Training** tab: overlay runs to compare,
smoothing slider, log scale, and a config diff that highlights exactly what you
changed between two runs. Logging is buffered and can never crash your training —
if the service is unreachable it warns once and your loop keeps going. Anything
is a metric: `grad_norm`, `tokens_per_s`, `gpu_mem_gb`, whatever you log.

## 4 · Checkpoints during training — scores on your loss curve's step axis

Combine §2 and §3: when you save a checkpoint, upload it (same storage as §2)
and mark the step:

```python
model.save_pretrained("ckpt"); tokenizer.save_pretrained("ckpt")
model_id = bench.upload_artifact("my-run7-step4000", "ckpt")   # -> local/my-run7-step4000
run.log_checkpoint(4000, model_id)     # marks the step AND queues the benchmark
```

Now the Training tab shows **benchmark scores plotted against your training
steps**, right under your loss curve — select your run and look for "Benchmarks
along this run". (If you prefer the Hub, push there and pass the repo id to
`log_checkpoint` instead — both work.)

Want the whole thing as working code? The repo has a runnable sample that
trains a tiny model and does all of the above:

```bash
git clone https://github.com/Teraformer-LIMITED/evalboard && cd evalboard
python examples/train_and_benchmark.py --bench http://100.74.89.105:8899 --dry-run   # 1-minute check
python examples/train_and_benchmark.py --bench http://100.74.89.105:8899 --steps 200 --checkpoint-every 100
```

## The loop

Most of this guide is about benchmarking a model. The **Loop** tab is the
other job: improving one. It runs on one topic at a time, and every step has
an owner.

| Step | What happens | Whose job |
|---|---|---|
| **Import a bank** | Questions written by a person arrive whole, under their name — Exam tab, import panel. An LLM can draft candidates instead, but nothing reaches the bank unread. | whoever owns the subject |
| **Sit the exam** | One model answers that topic's questions. The judge grades every answer against the topic's rubric and criteria file, and folds a 0–4 in code. | anyone with a model |
| **Read the results** | The answers, one card each: what the model wrote, the score, which flags fired, each criterion as a small cell with the three weakest named, and what the judge wrote about it. Diagnosis half only. | whoever wants to know why |
| **Propose** | On the topic page, and only there. An LLM reads the judge's *written assessments* — never the questions — and says what skill is missing. When the only thing wrong is the judge (a local model, not yet checked against a person), the button reads **Propose…** and asks first; what it makes is marked "proposed over a provisional judge", all the way to any model trained on it. | anyone, once the gate is clear |
| **Review the spec** | A person approves, edits or rejects that sentence. This is the airlock: only approved text reaches a generator. | the reviewer |
| **Generate** | A generator that has seen only the spec writes prose documents. A 13-gram gate drops anything that overlaps an exam question or a benchmark item. | the reviewer |
| **Hand to training** | The dataset id and the `--gap-dataset` line. A run that consumes it registers it, and its checkpoints carry a taint badge on that topic. | whoever trains |

Two rules hold at every step. **The report half of each topic is never
shown** — not on a page, not in an export, not in any request except the
judge's; it is the published score, and it stays unseen so it stays
meaningful. And **a number that cannot be trusted is not ranked**: an
uncalibrated judge, a judge that has moved, a topic with fewer than 30
report-half questions, or a model that trained on the topic's diagnostics
all leave the score visible and out of every average, with the reason in
words beside it.

The single button on each Loop row is the next step for that topic. When it
is disabled, the sentence under it is the server's own refusal — the same
words the API would answer with.

## 5 · Reading the dashboard

**Overview** — best model, how many differences are statistically real.
Uploaded checkpoints appear everywhere — nothing is hidden. They're just
marked: hollow bars in the panels, a dashed `ckpt` badge in the tables. Since
everything ranks by score and long panels open on their best 12 (a button
shows the rest), a chance-level checkpoint tail stays out of the way on its
own. The **All / Models / Checkpoints** switch next to Base/Instruct filters
when you want only one kind, and the Training tab plots each run's
checkpoints against its training steps.
**Training** — your live curves, run compare, config diff, benchmark-vs-step.
The runs list has a search box (name, project or person), a status filter and a
sort menu — "best loss" and "recently updated" are the two you'll live in.
**Submit & Queue** — submit models, watch progress, read failure logs. The
queue is searchable and filterable the same way (find *your* jobs, failures
first); click any column header to sort.
**Leaderboard** — every model × every task, ± standard error, sortable, with a
"last eval" date per model. **Avg only exists for models that finished all
seven required tasks** (mmlu, hellaswag, arc_challenge, arc_easy, winogrande,
piqa, truthfulqa_mc2). A quick run shows `— 2/7` and a `prelim` badge instead
of an average: its per-task numbers are real and shown everywhere, it just
can't hold an overall rank, because a mean over two easy tasks isn't
comparable to a mean over seven. Run `suite=full` to make a model official.
Avg is scaled so chance = 0 by default (raw accuracy is one click away) —
otherwise a 2-option task like PIQA hands every model a free 50%. Above it, the **Capability profile** radar: tick up
to three models in the table to compare their shape across benchmarks. Its axes
are scaled *above chance* by default (25% on a 4-way task = 0), so read the
shape there and the numbers in the table.
**Tasks** — one panel per benchmark; the dashed line is chance. Distance from
chance is the real score: **50% on Winogrande/PIQA is a coin flip, not a pass**,
and MMLU sits at ~25% for every model under ~1B — that's expected, not a bug.
**Perplexity & Loss** — bits/byte on pinned corpora, plus cross-entropy in
nats/byte (same quantity as your training loss, per byte, so it's comparable
across tokenizers — line it up with your loss curve).
**Evals** — full provenance (sorted newest-eval-first; click a column to
re-sort) and a query bar over every raw number (`pythia task:mmlu_ value>0.3`,
`metric:cross_entropy`), with CSV export. The query bar suggests completions
as you type — model, task and metric names straight from the data, so you
never have to remember what a task is called: ↑↓ to pick, Enter to insert.

Two honest-statistics habits the dashboard enforces: every score carries its
standard error, and if two error bars overlap, treat the models as tied. On
perplexity the harness gives no standard error at all, so the dashboard marks
values within ~1% of the best with `≈` rather than crowning one — a 0.001 lead
on bits/byte is not a win.

## The protocol (what makes two numbers comparable)

Scores can only be compared when the task, metric, n-shot count, prompt format
and **chat-template policy** all match. The service enforces the last one:

- `kind: auto` applies a chat template when the repo ships one and the model's
  name says it is instruction-tuned (`instruct`, `-it`, `chat`, `sft`…).
- For an **uploaded checkpoint** (`local/<name>`) that ships a template with no
  such evidence in its name, **preflight refuses the submission** and asks you
  to say `base` or `instruct` explicitly. This is the case that bites: a
  checkpoint saved from an instruct model's tokenizer inherits its chat template
  even though the weights are a base model, and applying it moves
  multiple-choice scores by tens of points — it cost one of our own models
  three to seven points before anyone noticed.
- For a **Hub repo**, shipping a template usually does mean the model is a chat
  model (Qwen3-0.6B ships one and says nothing in its name), so it is applied
  and the run is flagged as resting on detection alone — visible as a `?` in
  provenance and named in the warnings. If those weights are a pretrained
  checkpoint, resubmit with `kind: base`.
- Every run records which template it used (a short hash), where it came from,
  and why the decision was made — visible in Evals → Run provenance, and in
  every CSV export.

If you change the policy for a model, its old scores are not comparable to the
new ones. Move the old results out of `results/full/<model>/` before re-running,
or the finished tasks are served from cache.

## 6 · House rules

The GPU is shared with real training jobs, so: one evaluation runs at a time
(your submission queues — that's normal); use **quick** while iterating and save
**full** for checkpoints that matter; models are capped at **4B params**;
uploaded checkpoints must be **safetensors** (pickle `.bin` files are refused —
they execute code on load) and count against a shared storage quota, so delete
old artifacts you don't need (`DELETE /api/artifacts/<name>`, scores stay).
Always set your name as submitter — the queue should answer "whose job is this"
without archaeology.

## FAQ

**My model scored below 25% on MMLU.** Below-chance on a 4-way task means the
prompt format is fighting the model (usually a chat-template mismatch), not that
it "knows negative things". Try `kind: base` vs `instruct`, or ask Masein.

**GSM8K is ~0%.** Correct and expected under ~1B params — written math barely
exists at this scale. It's there so you can see it emerge.

**A gated model (gemma, llama) fails preflight.** The server's HF account must
accept that model's license once — ask Masein, or submit an ungated mirror.

**My run shows "idle 42m".** It is still marked *running* (nothing called
`run.finish()`) but has been silent for a long time — longer than 5× its own
usual gap between updates, and at least 30 minutes. Usually a crash or Ctrl-C.
Display-only: logging again clears it, and hovering the row shows this run's
normal reporting rhythm.

**My run says "finished" but there are no benchmark scores yet.** "Finished"
means the *training* finished; your checkpoints may still be in the eval queue
(one runs at a time). The run's page shows "benchmarks: X/Y done" and the
Submit & Queue tab shows live progress — scores appear as each one lands.

**Something else broke.** Every failed submission has a `log` link with the raw
error, and the error messages are written to be actionable. If they aren't: Masein.

*Everything here is also an HTTP API — see
[API.md](https://github.com/Teraformer-LIMITED/evalboard/blob/main/API.md) if you'd
rather curl.*
