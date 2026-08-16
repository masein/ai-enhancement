# 13 — Review: "The Assembly Line of LLM Training" (training_LLM.md, 2026-07-26)

Notes on the internal doc, written from the point of view of the person who will own
the eval pipeline. Three things: what to trust, what to fix, and what it doesn't cover
that your job depends on.

---

## What this document is

A **landscape survey**: which datasets exist, which tokenizers exist, which
post-training methods exist, which metrics exist. As an orientation map for someone
new to the field it is genuinely useful, and the datasets section is the strongest
part — nine corpora with "who it's best for" and "strengths/limitations" for each is
real work and mostly accurate.

What it is not is a **method**. It tells you what the options are; it does not tell
you how to choose between them, what breaks, or how you would know. That distinction
matters most in the section closest to your job.

The framing table on page 1 (layer → what it optimises → what users perceive) is the
best thing in the document. Keep it.

---

## Errors worth fixing

### 1. C4's licence is wrong, and it is a licensing claim

> "Its **open license (CC BY-SA)** and large size make it ideal for building models…"

C4 is released under **ODC-BY**, not CC BY-SA. The dataset card is explicit:

> "We are releasing this dataset under the terms of ODC-BY. By using this, you are
> also bound by the Common Crawl terms of use in respect of the content contained in
> the dataset."

Two separate corrections in there: the licence name, and the fact that Common Crawl's
terms of use ride along with it. CC BY-SA is *share-alike* — materially different
obligations from ODC-BY, and share-alike is exactly the kind of term a legal review
reacts to. Worth fixing before anyone plans around it.
([source](https://huggingface.co/datasets/allenai/c4/blob/main/README.md))

### 2. The data recipe contradicts itself

The prose says:

> "A standard proportional mix allocates roughly **60% web data, 20%
> scientific/educational texts, and 20% code and structured data**."

The table immediately below says: web 40–50%, encyclopedic 10–15%, books 10–15%,
scientific ~10%, code 5–10%, multilingual 5–10%, government 5–10%.

Grouped the same way, the table is roughly **45% web / 35% scientific+educational /
20% code+structured**. That is a 15-point disagreement on the largest category, in
two adjacent paragraphs. Only the prose version carries a citation.

Also: the table's ranges sum to 85% at the low end and 120% at the high end, so it
cannot be used as written — you can't take the top of every range. If it is meant to
be a recipe it needs to add to 100%.

### 3. Two tables are explicitly "recommended by ChatGPT"

The commercial-vendor ranking (with ⭐⭐⭐⭐⭐ "Excellent" ratings) and the data recipe
table both say so, which is honest of the author. But it means neither carries any
evidence, and the vendor table in particular reads like a procurement recommendation.
Nobody has benchmarked Defined.ai against Nexis Data+ for LLM training suitability;
the stars are a language model's impression of brand reputation.

Suggested fix: keep them, relabel them "starting point for research, not a
recommendation", and strip the star ratings. A five-star rating with no methodology
behind it will eventually be quoted back at someone in a meeting.

### 4. MLM is listed as an LLM training methodology

The "Training Methodologies" section lists Masked Language Modeling. Decoder-only LLMs
— which is what your team trains — are trained on **next-token prediction**, not
masked-token prediction. MLM is BERT-family / encoder pretraining.

In the multimodal context where the section places it, MLM does appear legitimately.
But a reader skimming the section headings will come away thinking MLM is an option
for their language model, and it isn't. One clarifying sentence fixes it.

### 5. Facts I checked that hold up

Worth saying explicitly, so the corrections above don't cast doubt on the rest:

- **Common Crawl, March 2026: 344.64 TiB across 1.97 billion pages.** Exactly right,
  matching the crawl announcement to two decimal places.
  ([source](https://commoncrawl.org/blog/march-2026-crawl-archive-now-available))
- **PleIAs Common Corpus**, ~2T tokens, released mid-November 2024, largest open
  multilingual pretraining dataset. Correct.
- **RefinedWeb** public release ~600B tokens, **C4** ~750 GB English, **Dolma**
  multi-trillion token. All correct.

Whoever wrote the datasets section did the reading.

---

## The one structural question: is this doc about text models or multimodal ones?

Section 5, "Training Methodologies and Data Requirements", is entirely about
**multimodal** models — contrastive learning, image-text matching, masked image
modeling, visual grounding. It opens with "The pretraining phase is crucial for
MLLMs".

Nothing else in the document is about multimodal. The datasets section lists no
image-text corpora (no LAION, no DataComp, no OBELICS). The tokenization section
covers no image tokenizers or patch encoders. The eval section has no multimodal
benchmarks (no MMMU, no VQA).

So either:

- the team trains a text model, and this section belongs in a different document; or
- the team trains a multimodal model, and the other four sections are missing their
  multimodal half.

This is worth asking about directly, because the answer changes what your eval
pipeline has to measure. It is a real question, not a nitpick.

---

## What the document doesn't cover

Not criticism — a survey can't cover everything. But three of these are named in the
doc's own assembly-line table on page 1 and then never returned to, which is worth
flagging because a reader will expect them.

| Missing | Named in the page-1 table? | Why it matters |
|---|---|---|
| **Training mechanics** — batch size, optimizer, LR schedule, gradient accumulation, precision, distributed strategy | yes, "System and architecture" | This is the actual *assembly line*. The doc has no forward pass, no backward pass, no optimizer. |
| **Mixture of Experts** | "active parameters", once | If the team's model is MoE this is a large hole — and MoE changes how RL behaves. |
| **RLVR / GRPO** | no | See below. |
| **Distillation** | yes, "Distillation and deployment" | Named as a layer, never explained. |
| **Evaluation *method*** | yes, "Eval and reward" | The section is a metric list. See below. |

### RLVR is the significant one

The post-training section covers RLHF, DPO and RFT. RFT is described as "a more
productionizable interface that packages task definition, grader design, and reward
signal into a deployable pipeline" — that is a fair description of the *product*
framing.

What's missing is the open training methodology behind that whole direction:
**RL with Verifiable Rewards** — GRPO and its successors (DAPO, Dr. GRPO, GSPO). That
is where the reasoning-model step change came from, it is what most teams training
open models are actually running, and it is the part of post-training that most
depends on good evaluation, because the reward function *is* an eval and the whole
run is steered by it.

Given that post-training evaluation is your remit, this is the gap most worth
volunteering to fill.

---

## The eval section, specifically — this is your job

The metric taxonomy is comprehensive and well-organised: pretraining, generation,
diversity, reasoning, coding, instruction-following, safety, RAG, agent, efficiency.
As a checklist of *what can be measured* it is good.

What is absent is everything that makes a measured number **trustworthy or
comparable**. Every item below is a thing that silently invalidates comparisons:

- **No `n`, no standard error.** At n=200 and p≈0.5 the standard error is ±3.5 points.
  Without it, teams treat 2-point moves as results. This is the single highest-value
  addition.
- **No prompt-template sensitivity.** The same checkpoint scored under a mismatched
  template can go to zero. It looks exactly like a capability regression.
- **No contamination / decontamination.** The first question anyone asks about a
  benchmark number, and the doc doesn't mention it — even though the *datasets*
  section discusses deduplication at length. Same technique, different purpose.
- **No decoding parameters.** Greedy vs sampled, temperature, top-k, max tokens. Two
  runs with different decoding are not comparable, and this is where most
  irreproducible numbers come from.
- **acc vs acc_norm is not distinguished.** MMLU and friends are scored by comparing
  option log-probabilities; length normalisation changes the ranking. When a paper's
  two numbers disagree, the benchmark is partly measuring option length.
- **pass@k is listed without its estimator.** "HumanEval (Pass@k): percentage of
  programming problems solved correctly" — pass@k has an unbiased estimator for
  k < n_samples, and most reimplementations get it wrong.
- **LLM-as-a-Judge gets one line.** No judge-version pinning, no position bias, no
  measurement of agreement with human raters. An unvalidated judge is a
  number-generator, not a metric.
- **BLEU / ROUGE / METEOR / chrF** are listed without the caveat that they are largely
  obsolete for evaluating modern LLMs. They need reference texts and were designed for
  MT and summarisation. Reporting BLEU on an LLM mostly signals that nobody chose the
  metric deliberately.
- **No suite versioning.** A benchmark whose task list, item seed, or prompt format
  changed is a different benchmark, and scores across the change are not comparable.

None of that is exotic. It is the difference between a metrics glossary and an eval
harness, and it maps one-to-one onto `docs/09-evaluation-and-benchmarking.md` in this
repo.

---

## Suggested next move

The doc is a good map with a thin spot exactly where your role sits. Rather than
sending back a list of corrections, the higher-value move is to offer the missing
section: a two-page addendum on **evaluation discipline** — the reporting contract
(every number carries n, stderr, prompt template, decoding params, suite hash), plus
contamination measurement and the pass@1-vs-pass@k question for anything being RL'd.

You have a working implementation of all of it, so the addendum can point at running
code rather than assert good practice. That is a strong first contribution: it fills a
real gap, it doesn't step on anyone's section, and it makes the case by demonstration.

The corrections above are worth sending too — just separately, and briefly. The C4
licence one is the only one that could cost anyone anything.

---

## Questions worth asking your colleague

1. **Text or multimodal?** (See the structural question above. Changes everything
   downstream for eval.)
2. **Is the model MoE?** If so, is GSPO or routing replay being used for RL? MoE
   routing makes per-token importance ratios noisy, and that gets misdiagnosed as a
   learning-rate problem.
3. **Is RLVR/GRPO in the plan, or is post-training stopping at DPO?** Different eval
   requirements: RLVR needs verifiable graders and pass@k headroom measurement.
4. **Which of the listed datasets are actually in the mix, at what weights?** The
   recipe table is a suggestion; the real one is a config file somewhere.
5. **Where does the current eval suite live, and is its item set pinned?** If the
   answer is "a script on someone's machine", that is your first project.
6. **Has contamination ever been measured?** Most likely answer is no. Measuring it
   once is the most valuable — and least welcome — thing you can do in month one.
