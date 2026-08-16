# START HERE — the practical path

Every command below was run and verified. Run them **from the repo root**, in order.
Times are for a MacBook; the first one is the only slow one.

Each step says what to look for, because the output is the lesson.

---

## Step 0 — setup (2 minutes, once)

```bash
cd ~/Developer/ai-enhancement
rm -rf _to_delete                       # the tarball I used to deliver this

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt         # torch, numpy, pyyaml (~2 min)

export PYTHONPATH=src                   # every command below needs this
```

Put `export PYTHONPATH=src` in every new terminal, or run `source .venv/bin/activate`
and add it to the venv's `activate` script once.

Check it worked:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

**Look for:** `True` at the end — that's your Mac's GPU. The code picks it up
automatically (`utils.pick_device`). If it says `False`, everything still runs on CPU,
just slower.

---

# Session 1 — see the pieces (10 minutes)

## 1.1 The data pipeline

```bash
python -m aienh data --corpus dirty --n-docs 4000
```

**Look for:** the stage table. Each row is a cleaning stage and how many documents it
dropped:

```
stage                 in      out  dropped   drop%
load                4000     4000        0    0.0%
normalise           4000     3920       80    2.0%
quality_filter      3920     3438      482   12.3%
exact_dedup         3438     2770      668   19.4%
near_dedup          2770     2416      354   12.8%
```

**What it teaches:** this report is the most important output of any data pipeline. A
filter that silently eats 90% of a source is the most common data bug in the field and
looks identical, from outside, to "the model just didn't learn much".

Now run it on a *clean*, structured corpus and compare:

```bash
python -m aienh data --corpus arithmetic --n-docs 4000
```

**Look for:** `quality_filter` drops **0%**. That is not luck — `data.py:FILTER_PRESETS`
gives the arithmetic corpus its own thresholds. With prose defaults it drops 36%, and
all of the dropped documents are the small-operand ones. Read the comment above
`FILTER_PRESETS`; it's the whole lesson in 15 lines.

📖 Read now: `docs/02-tokenization-and-data.md`

## 1.2 The model

```bash
python -m aienh.model
```

**Look for:**

```
moe=False params=426,624 active=426,624 loss=4.511
moe=True  params=2,263,680 active=690,816 loss=4.142
```

**What it teaches:** total vs active parameters. For the dense model they're the same
number; for MoE they're wildly different, and *active* is the one that predicts your
inference bill. Also note the untrained loss ≈ `ln(vocab_size)` — a model that starts
anywhere else has a broken initialisation.

## 1.3 The tokenizer

```bash
python -m aienh.tokenizer
```

**Look for:** `char vocab=97 len=23` vs `bpe vocab=277 len=7`. Same string, one third
the tokens. Tokens are the unit of compute cost, so that ratio is a compute bill.

## 1.4 The tests

```bash
python tests/test_smoke.py
```

**Look for:** `39/39 passed`. Then open the file and read three of them:
`test_causality`, `test_pack_is_a_shifted_pair`, `test_grpo_update_moves_probabilities_the_right_way`.
These are the "is the maths right" tests, not the "does it run" tests. Each one
corresponds to a bug that produces plausible wrong numbers rather than a crash.

📖 Read now: `docs/00-orientation.md` (the map and the vocabulary), then
`docs/01-how-llms-work.md` with `src/aienh/model.py` open beside it.

---

# Session 2 — the whole pipeline, fast (5 minutes)

```bash
AIENH_TRACKER=local python -m aienh pipeline --scale smoke
```

This runs *everything* — pretrain dense, pretrain MoE, SFT, GRPO, two distillations,
eval after every stage, registry rows, dashboard — at CI scale.

**Look for**, in order:

1. The header for each run: params, tokens/step, `tokens_per_param`.
2. `[dense vs MoE]` — total params up ~3×, **active/token 1.00×**. That's a genuinely
   FLOP-matched comparison; most published MoE comparisons aren't.
3. The eval block after each stage, with `+/- stderr` on every accuracy.
4. The GRPO run printing `held-out exact_match` *during* training.
5. The final leaderboard, with a **warning** that two different suite hashes are in the
   table and must not be compared.

Then open the dashboard:

```bash
open artifacts/dashboard.html
```

**What it teaches:** this is the deliverable your job produces. Note the suite selector
(runs from different suites are never mixed), the "Show data table" button under every
chart, and the stage-over-stage delta table.

> The `artifacts/dashboard.html` that shipped with the repo is from *my* 15-minute run,
> so you can see a fuller version before yours exists. Running the pipeline overwrites it.

📖 Read now: `docs/11-the-pipeline-you-will-build.md`

---

# Session 3 — do it stage by stage, by hand (~35 minutes)

This is the one that actually teaches you. Same work as the pipeline, but you drive it,
and you see each artifact appear.

## 3.1 Pretrain a dense model (~5 min)

```bash
python -m aienh train --config configs/pretrain_dense.yaml
```

Open `configs/pretrain_dense.yaml` in another window while it runs — every value is
commented with *why*.

**Look for:**

- `budget: 800 steps = 2.46M tokens (1.37 tokens/param, 4.93 epochs over the data)` —
  Chinchilla-optimal is ~20 tokens/param, so this model is deliberately undertrained.
- `contamination train->val: 0.0% exact, 35.7% 13-gram` — measured on every build.
- `gnorm` in the log line. Watch it. A spike precedes a loss spike.
- The run name it prints, e.g. `pre-dense-quick-onyx-1e1c`. **Copy it** — you need it next.

```bash
BASE=runs/pre-dense-quick-onyx-1e1c/model.pt      # ← your name here
```

Talk to it:

```bash
python -m aienh sample $BASE --prompt "the dog ran to" --max-new-tokens 40
python -m aienh sample $BASE --prompt "17 + 25 = " --greedy --max-new-tokens 4
```

**Look for:** fluent-ish story text, and a *wrong* sum (mine said 47). This is a base
model: it continues text, it does not answer.

## 3.2 Score it

```bash
python -m aienh eval $BASE --tasks ppl_stories,arith_exact,arith_mc4,format_ok --n 200
```

**Look for:** `arith_exact ≈ 0.005` but `arith_mc4 ≈ 0.30`. Same capability, two ways of
measuring, and 25% of the MC number is free (random choice among 4). That gap is why
"our model scores 82% on <MC benchmark>" and "our model is useful" are different claims.

📖 Read now: `docs/09-evaluation-and-benchmarking.md` — this is your job.

## 3.3 Pretrain an MoE at the same budget (~8 min)

```bash
python -m aienh train --moe --config configs/pretrain_moe.yaml
```

**Look for:** `params=5.34M (active/token=1.80M)` — 3× the parameters, the same active
per token, and **worse wall-clock throughput** than the dense run. That inversion is
real at this scale: per-expert matmuls are too small to amortise the routing overhead.
MoE's advantage is asymptotic.

📖 Read now: `docs/04-mixture-of-experts.md`

## 3.4 Fine-tune it (~1 min)

```bash
python -m aienh sft $BASE --config configs/sft.yaml
```

**Look for:** `loss_tokens 22%` in each log line — only 22% of each batch contributes to
the loss, because the prompt is masked out. That's the *entire* mechanical difference
between SFT and pretraining.

```bash
SFT=runs/sft-terse-anchor-e8e2/model.pt          # ← your name here

python -m aienh sample $SFT --greedy --max-new-tokens 4 \
  --prompt "$(printf 'Q: 17 + 25 =\nA:')"
```

**Look for:** `A: 42`. It can add now. Note the prompt has to be *exactly* the training
template, newline and all.

📖 Read now: `docs/05-post-training.md`

## 3.5 The 30-second demo that matters most

Same checkpoint, same questions, two prompt formats:

```bash
python -m aienh eval $SFT --tasks arith_exact --template chat --n 100 --no-record
python -m aienh eval $SFT --tasks arith_exact --template raw  --n 100 --no-record
```

**Look for:** ~0.31 vs **0.00**. Nothing about the model changed. This is the #1 false
alarm in evaluation work, and it is why `evaluate.suite_hash()` includes the prompt
template — so two scores from different formats can never end up in the same table.

## 3.6 Measure what contamination is worth

```bash
python -m aienh eval $SFT --tasks arith_exact,arith_exact_seen --template chat --n 200 --no-record
```

**Look for:** two rows. The first is problems the model has never seen; the second is
problems it trained on (mine: 0.31 vs 0.43). That 12-point gap is memorisation — exactly
what a contaminated public benchmark rewards. `arith_exact_seen` is deliberately kept
out of the default suite and out of the score.

## 3.7 Should you even run RL? Measure the headroom first

```bash
python -m aienh eval $SFT --tasks arith_exact,arith_pass@8 --template chat --n 60 --no-record
```

**Look for:** pass@1 well below pass@8. A large gap means the capability is present but
unreliable — the regime RL is for. A gap near zero means RL can only add noise. Handing
an RL engineer this one measurement before they burn a week of GPU time is the highest-
leverage thing an eval owner does.

## 3.8 Run GRPO anyway, and watch it fail honestly (~2 min)

```bash
python -m aienh grpo $SFT --config configs/grpo.yaml
```

**Look for**, in this order:

- `skipped 5/8` — five of eight groups had zero reward variance, so they contributed an
  identically-zero gradient while still costing you the generations. Your effective
  batch size is not what your config says.
- `clip 0.00%` — correct, not a bug: with `mu=1` the importance ratio is exactly 1, so
  clipping never fires. Try `--set mu=4` and watch it become nonzero.
- `held-out exact_match` printed every 25 iterations, and a warning if it drops more
  than 3 standard errors below the run's own best.
- `ent` (entropy) falling — the policy going deterministic.

Mine got *worse*. That's a real result, reproduced in four configurations, and
`docs/07-grpo.md` has the table and the diagnosis rather than a tuned-until-it-looked-good
number.

📖 Read now: `docs/07-grpo.md` and `docs/06-rlhf.md`

## 3.9 Distil a small student three ways (~2 min)

```bash
python -m aienh distill $SFT --mode offline --config configs/distill_offline.yaml
python -m aienh distill $SFT --mode online  --set steps=300
python -m aienh distill $SFT --mode on_policy --set steps=300
```

**Look for:**

- The offline run prints the storage arithmetic: your cache size, and what full logits
  would cost at production scale (256 TB vs 96 GB — that ratio is why top-k caching exists).
- Run the offline command **twice**. The second time prints
  `reusing teacher cache — teacher not run`. That reuse *is* the economic argument for
  offline distillation.
- `soft_kl` and `hard_ce` logged separately, so you can see the two halves of the loss.
- The student is ~13% of the teacher's size — check the header line.

📖 Read now: `docs/08-labels-and-distillation.md`

## 3.10 Look at what you built

```bash
python -m aienh leaderboard
python -m aienh dashboard && open artifacts/dashboard.html
```

**Look for:** the lineage column (`pre-… → sft-… → grpo-…`), the suite-hash warning, and
in the dashboard the stage-over-stage delta table. Then:

```bash
cat runs/registry.jsonl | python -m json.tool --json-lines | head -60
```

That file is the whole database. Diffable, greppable, mergeable in git.

---

# Session 4 — break it on purpose (~15 minutes)

Each of these demonstrates a class of bug that makes teams distrust their own numbers.

```bash
python scripts/demo_data_bias.py
```

Trains two identical models where the *only* difference is one filter threshold, then
prints overall accuracy next to accuracy sliced by operand size. **Look for:** the
aggregate barely moves while one slice collapses. That is why you slice every metric.

```bash
python scripts/demo_template_mismatch.py
```

The full version of 3.5, with training included.

```bash
python scripts/wandb_example.py
```

The five W&B calls that matter, running against the local backend. Then look at what it
wrote:

```bash
ls runs/wandb-example/
cat runs/wandb-example/summary.json
```

To use the real thing: `pip install wandb && wandb login`, then re-run anything. Same
call sites, no code change — that's what `tracking.py` is for.

📖 Read now: `docs/10-experiment-tracking-wandb.md`

---

# Session 5 — now change things and measure

This is where it stops being a tutorial. Each of these is one command and a real
question. Run the eval afterwards and compare.

**Does prompt masking actually matter?**
```bash
python -m aienh sft $BASE --config configs/sft.yaml --set mask_prompt=false
```

**What does the load-balancing loss actually do?** (watch `moe/balance` collapse)
```bash
python -m aienh train --moe --config configs/pretrain_moe.yaml --set aux_loss_coef=0.0
```

**Does the data mixture change the answer?**
```bash
python -m aienh train --config configs/pretrain_dense.yaml --set corpus='{"arithmetic":1.0}'
python -m aienh train --config configs/pretrain_dense.yaml --set corpus='{"arithmetic":0.4,"stories":0.6}'
```

**Does the tokenizer change the answer?**
```bash
python -m aienh train --config configs/pretrain_dense.yaml --set tokenizer=bpe --set vocab_size=512
```

**Do the GRPO variants behave differently?**
```bash
python -m aienh grpo $SFT --set scale_rewards=false     # Dr. GRPO
python -m aienh grpo $SFT --set ratio_mode=sequence     # GSPO — what you'd use on an MoE
python -m aienh grpo $SFT --set mu=4                    # now clipping actually fires
python -m aienh grpo $SFT --set beta=0.05               # add the KL leash back
```

**Does SFT length trade off against RL headroom?**
```bash
python -m aienh sft $BASE --config configs/sft.yaml --set epochs=4    # ~0.09 exact-match
python -m aienh sft $BASE --config configs/sft.yaml --set epochs=12   # ~0.81, no headroom left
```

**A hyperparameter sweep, no account needed:**
```bash
python scripts/sweep_local.py --steps 300
```
**Look for:** the warning it prints at the end about the spread across trials versus the
standard error on the metric. Sweeps are very good at producing a confident-looking
winner out of noise.

After any of these:
```bash
python -m aienh eval runs/<new-run>/model.pt --tasks arith_exact,arith_mc4 --template chat --n 200
python -m aienh dashboard && open artifacts/dashboard.html
```

---

## Reading order, if you'd rather read first

`docs/README.md` has the full table. The short version:

**One evening:** 00 (the map) → 09 (your job) → 03 (the training loop) → skim 07.
**A weekend:** all thirteen in order, running the commands as you go.

Keep `artifacts/cheatsheet.html` open in a tab. It's the one-page version of everything.

---

## Two things that will bite you

**`export PYTHONPATH=src`** — every command needs it, and the error if you forget is
`No module named aienh`.

**Run names are unique per config.** If you re-run with the same config you get the same
name and the registry's last-write-wins on read. Change anything real and the name
changes with it — that's `utils.run_name()` putting the config hash in the name on
purpose.
