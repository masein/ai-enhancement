# 09 — Evaluation and benchmarking

Code: **`src/aienh/evaluate.py`**, **`src/aienh/registry.py`**, **`src/aienh/dashboard.py`**.

This is your job. Everything else in these docs is context for reading this one
correctly.

---

## An eval harness is two primitives and a lot of discipline

### Primitive 1: `loglikelihood(context, continuation)`

Returns `(sum log P(continuation | context), n_tokens, is_greedy)`. Answers "how much
does the model like this exact string?" without generating anything.

**Every multiple-choice benchmark is built on this.** MMLU, ARC, HellaSwag,
WinoGrande: score each option, take the argmax. No parsing, no judge, zero sampling
noise, and cheap. It is also why MC benchmarks measure something narrower than they
appear to — the model never has to *produce* anything.

Two derived metrics you must be able to distinguish:

- **`acc`** — argmax over the raw summed logprob.
- **`acc_norm`** — argmax over logprob normalised by length (per character, usually).

Longer continuations accumulate more negative logprob simply by being longer, so raw
sums systematically favour short options. **When a paper reports both and they
disagree, the benchmark is partly measuring option length.** Know which one you are
quoting; a lot of leaderboard confusion is exactly this.

`evaluate.py:loglikelihood` returns the same triple lm-evaluation-harness does, and
`tests/test_smoke.py` verifies it against a hand-computed value *and* checks that
batching does not change it.

### Primitive 2: `generate(context)`

Answers "what does the model actually do?" Needed for GSM8K, HumanEval, IFEval,
anything agentic. Requires a parser or a judge, and is sensitive to decoding
parameters — which is where most irreproducible numbers come from.

---

## The discipline: seven rules

**1. Fix the items.** Same seed, same questions, every model, forever. If the item
set drifts, cross-model comparisons are noise. `evaluate.py:arith_items` takes a seed
and is documented as schema-like: changing it invalidates every historical
comparison.

**2. Fix the decoding.** Greedy (temperature 0) when measuring, unless the metric is
explicitly sampling-based (pass@k). Record the decoding parameters in the result.

**3. Report `n` and stderr.** For a proportion, `stderr = sqrt(p(1-p)/n)`. At
`p=0.5, n=200` that is **±3.5 points**. Two runs differing by 2 points are
indistinguishable. This one habit will save your team more wasted work than
everything else in this doc.

**4. Log raw samples.** Persist the generations. Every "the model got worse"
investigation starts by reading twenty of them, and roughly half end at "the parser
broke", not "the model broke".

**5. Version the suite.** A benchmark whose code changed is a different benchmark.
`evaluate.py:suite_hash` hashes the task list, the item seed **and the prompt
template**, and `registry.render_table` refuses to silently mix suite hashes in one
table.

**6. Slice every metric.** One aggregate hides everything.
`task_arith_exact` reports overall accuracy *and* accuracy by operand size, which is
how a data-filtering bias becomes visible instead of being inferred six weeks later
(doc 02).

**7. Separate the problem space, not just the samples.** See "contamination" below.

---

## The taxonomy of benchmarks, and what each is good for

| Kind | Example | Measures | Fails at |
|---|---|---|---|
| MC loglikelihood | MMLU, ARC, HellaSwag | knowledge, cheap + stable | not generation; saturates; length bias |
| Generative + exact match | GSM8K, MATH | producing an answer | parser-sensitive, format-sensitive |
| Code execution | HumanEval, MBPP, SWE-bench | actually works | needs a sandbox; slow |
| Constraint checking | IFEval | instruction following | only verifiable constraints |
| LLM-as-judge | MT-Bench, AlpacaEval | open-ended quality | judge bias, drift, cost |
| Human preference | Arena-style | what people prefer | slow, expensive, noisy |
| Perplexity | any held-out corpus | training health | not capability; tokenizer-dependent |

**Perplexity deserves a specific warning.** It is only comparable across models
sharing a tokenizer *and* an eval corpus. Different tokenizers → different token
counts for the same text → the numbers are on different scales. Use **bits per byte**
when you must compare across tokenizers (same text, same denominator).
`task_perplexity` reports both.

And a measured example of why perplexity is not capability: on this repo's arithmetic
corpus, most of the loss sits in the *operand* tokens, which are random by
construction and irreducible. A model can sit at a respectable loss while getting the
answer wrong most of the time, because the answer is 2 of ~12 tokens. Loss is a
training-health signal. Task metrics are the product signal.

### MC vs generative, quantified

Run both on the same capability and compare. Measured in this repo, on the same
checkpoint: `arith_exact` (generate the answer) **10.5%**, `arith_mc4` (pick from 4)
**51.5%**. A quarter of that MC number is free — random choice among 4 is 25%.

"Our model scores 82% on \<MC benchmark\>" and "our model is useful" are different
claims. A serious suite contains both kinds of task, and the gap between them is
itself a metric worth tracking.

### pass@k

Sample `k` times, count the item solved if **any** sample is right. Answers "does the
model *know*" rather than "what does the model *say*".

The pass@1 → pass@k gap is the single most useful diagnostic you can hand an RL
engineer (doc 07): a large gap means the capability is present but unreliable, which
is exactly what RL fixes; a gap near zero means RL can only add noise. Measured here:
a checkpoint at pass@1 = 0.105 had pass@8 = 0.45 — enormous headroom.

### LLM-as-judge, if you must

Sometimes there is no verifier. Then: pin the judge model *version* (a silent judge
upgrade rewrites your history), randomise option order (judges have strong position
bias), and — the part teams skip — **measure agreement with human raters on a sample
before trusting it**. A judge you have not validated is a number-generator, not a
metric. Prefer verifiable constraints wherever you can construct them; they cost
nothing and never drift.

---

## Contamination

The first question anyone asks about a benchmark number. The only defensible answer
is a measurement.

**Measure it:** `data.contamination_report` reports exact-document overlap and
13-gram overlap (the GPT-3 convention, followed by most work since). Some overlap is
normal for common phrasings; a high rate on a specific task means that task's score
is inflated.

**Prevent it where you can:** this repo partitions the *problem space*, not the
samples. `data.arith_split` deterministically assigns every `(a, b)` pair to train or
test via a stable md5 hash; corpora, SFT data and RL prompts draw from `train`, eval
items from `test`. Sampling more examples would not have helped — with only 2,500
possible problems, a few thousand training examples covers essentially all of them,
so "held-out" items would have been items the model was trained on and the score
would have measured recall.

**Measure what contamination is worth on your task**, once, so you know the stakes:

```bash
python -m aienh eval runs/<run>/model.pt --tasks arith_exact,arith_exact_seen --template chat
```

`arith_exact_seen` is the same task on problems the model *was* trained on. It is
deliberately excluded from the default suite and from the points formula, because it
is a contaminated number by construction. The gap between the two is the model's
memorisation — which is precisely what a contaminated public benchmark rewards.

---

## The composite score ("points")

Teams want one number. One number hides regressions. Both are true, so:

- `evaluate.POINTS_WEIGHTS` is a published dict. The weights are visible in code and
  rendered in the dashboard next to the score.
- Accuracy metrics map linearly; perplexity is squashed with `1/(1+log(ppl))` —
  arbitrary but monotone and **fixed**. Any such mapping is arbitrary; what matters
  is that changing it invalidates every historical score, so it is a schema change.
- The component breakdown is always one click away. A composite whose components are
  hidden is a way of hiding regressions, and eventually someone will notice.

---

## The registry

`registry.py` is an append-only JSONL file. Deliberately: diffable, greppable,
mergeable in git, impossible to corrupt with a bad migration. Move to Postgres when
you have a reason.

Four fields do the real work:

- **`name`** — stable, human-sayable, never reused. People say these out loud in
  standups and type them into filters.
- **`config_hash`** — hash of everything that could change the numbers. Two rows with
  the same name and different config hashes means someone overwrote a result, which
  is the most expensive class of mistake on an eval team.
- **`suite_hash`** — which exam this score came from. Rows with different suite
  hashes are not comparable, and the tooling should refuse to pretend otherwise.
- **`parent`** — the checkpoint this run started from. `pre-lucid-ridge →
  sft-keen-onyx → grpo-warm-beacon`. Without lineage, "why did the RL run regress?"
  is unanswerable, because you cannot find the model it regressed from.

Plus `git_sha`: record the code version alongside the numbers, or the numbers are
hearsay.

---

## The debugging checklist: "the model regressed"

In priority order, because this is what you will be doing on a Tuesday afternoon:

1. **Did the prompt template change?** Most common cause by a wide margin. Check the
   suite hash first.
2. **Did the parser change?** Read 20 raw samples. A stricter regex looks exactly
   like a capability drop.
3. **Did the decoding change?** Temperature, top-k, max tokens. Greedy or not.
4. **Did the item set change?** Compare suite hashes. Compare `n`.
5. **Is the difference inside the standard error?** At n=200, a 3-point move is
   noise. Compute it before escalating.
6. **Did the batch size change?** If your harness cannot left-pad correctly, batching
   changes results. This repo groups prompts by length instead — see
   `generate_batch`'s docstring for why.
7. **Did the checkpoint change to one with a different tokenizer?** Perplexity
   comparisons silently become meaningless.
8. **Only then**: did the model change?

Wire the first five into the harness so they answer themselves. A harness that
reports its own suite hash, `n`, stderr, template and decoding parameters in every
result has pre-answered most of a postmortem.

---

## Production tools you will actually use

- **lm-evaluation-harness** (EleutherAI) — the de facto standard for MC and
  generative academic benchmarks; the thing Open LLM Leaderboard numbers come from.
  Its `loglikelihood` / `generate_until` interface is the model I copied here. Adding
  a task is a YAML file. *(I was unable to fetch its README in this session, so treat
  specific CLI flags as unverified — check `lm_eval --help` before scripting against
  it.)*
- **lighteval** (Hugging Face) — same space, tighter integration with the HF stack.
- **vLLM / SGLang** — you will not evaluate at any scale without a fast inference
  server. Both expose an OpenAI-compatible API that the harnesses can target.
- **W&B / MLflow** — run tracking and the shared run table (doc 10).

**Do not write your own harness for standard academic benchmarks.** Reimplementing
MMLU guarantees your numbers disagree with everyone else's for reasons nobody can
find. Write your own for *your team's* tasks, where no standard exists — which is
exactly the pipeline you are being hired to build.
