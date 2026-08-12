# 12 — Reading list

Ordered by what to read first, with what to take from each. Skip anything you already
know; the annotations are the point, not the citations.

I have listed these from memory of the literature rather than re-fetching each one in
this session. Titles and author groups are reliable; if you need to cite exact
numbers, pull the paper.

---

## Read these first (a weekend)

**Attention Is All You Need** (Vaswani et al., 2017)
The transformer. Read sections 3.1–3.3 and skip the machine-translation results.
Note it is encoder-decoder; you work with decoder-only, so mentally delete the
encoder and the cross-attention.

**Language Models are Unsupervised Multitask Learners** (GPT-2, Radford et al., 2019)
Where decoder-only + "just predict the next token" became the whole recipe. The
architecture in `src/aienh/model.py` is essentially this plus RMSNorm and RoPE.

**Training Compute-Optimal Large Language Models** (Chinchilla, Hoffmann et al., 2022)
The ~20-tokens-per-parameter result. Read it, then internalise why almost nobody
follows it: inference cost dominates over a model's lifetime, so you overtrain a
smaller model on purpose.

**Training language models to follow instructions with human feedback**
(InstructGPT, Ouyang et al., 2022)
The SFT → reward model → PPO pipeline, and the moment "post-training" became a
distinct discipline. The clearest single source for what RLHF actually is.

---

## Post-training

**Direct Preference Optimization** (Rafailov et al., 2023)
Preference optimisation without RL. Read for the derivation — understanding *why* the
closed form exists is what lets you evaluate the variant of the month.

**DeepSeekMath** (Shao et al., 2024)
Where GRPO is introduced. Section on GRPO is short and worth reading in the original,
because most secondary explanations garble the normalisation.

**DeepSeek-R1** (2025)
RLVR at scale, and the result that pushed the whole field toward verifiable rewards.
Also notable for what it reports about pure-RL-from-base.

**DAPO** (ByteDance/Tsinghua, 2025)
The four fixes: clip-higher, dynamic sampling, token-level loss, overlong shaping.
Read this one for the *diagnostics* — it is a good model of how to debug an RL run.

**GSPO** (Qwen team, 2025)
Sequence-level importance ratios, motivated by MoE instability. Directly relevant if
your team's model is MoE.

**Constitutional AI** (Bai et al., 2022)
RLAIF: replacing the human labeller with a model and a written set of principles.

---

## Mixture of Experts

**Outrageously Large Neural Networks** (Shazeer et al., 2017)
The sparsely-gated MoE layer and the load-balancing problem, stated once and for all.

**Switch Transformers** (Fedus et al., 2021)
Top-1 routing, the auxiliary loss form used in `model.py`, and an honest account of
the stability problems.

**Mixtral of Experts** (Mistral, 2024)
The open model that made MoE mainstream. 8 experts, top-2, and clean reporting of
total vs active parameters.

**DeepSeek-V3** (2024)
Fine-grained experts, shared experts, and auxiliary-loss-free load balancing. The
current-generation reference architecture.

---

## Distillation

**Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
Soft labels, temperature, "dark knowledge". Nine pages; still the best explanation.

**DistilBERT** (Sanh et al., 2019)
Distillation applied end to end with real ablations. Good template for how to report a
distillation result.

**On-Policy Distillation of Language Models** (GKD, Agarwal et al., 2023)
Why sampling from the *student* matters: the exposure-bias argument, made properly.

---

## Data

**The Pile** (Gao et al., 2020) and **RefinedWeb** (Penedo et al., 2023)
Read for the pipelines, not the datasets. RefinedWeb's argument that
well-filtered web text alone can beat curated corpora is the important claim.

**Deduplicating Training Data Makes Language Models Better** (Lee et al., 2021)
The empirical case for the dedup stages in `data.py`, including the effect on
memorisation.

**Scaling Data-Constrained Language Models** (Muennighoff et al., 2023)
How many epochs you can repeat data before it stops helping (roughly: up to ~4 is
nearly free). Directly useful when your high-quality source is small.

---

## Evaluation — the ones your job depends on

**Holistic Evaluation of Language Models (HELM)** (Liang et al., 2022)
Long, and worth skimming for the taxonomy: which axes exist, and how a benchmark
suite is *designed* rather than accumulated.

**lm-evaluation-harness** (EleutherAI, software)
Read the code, specifically the `loglikelihood` / `generate_until` interface and one
task YAML. That interface is the design I copied in `evaluate.py`, and knowing it
means you can read anyone's benchmark implementation.

**Evaluating Large Language Models Trained on Code** (HumanEval, Chen et al., 2021)
Introduces pass@k, including the unbiased estimator for `k < n_samples` — which is
the correct way to compute it and is not what most reimplementations do.

**Judging LLM-as-a-Judge** (MT-Bench, Zheng et al., 2023)
Position bias, verbosity bias, self-enhancement bias, and measured
judge–human agreement. Read before you ship a judge-based metric.

**Documenting the English Colossal Clean Crawled Corpus** (Dodge et al., 2021)
Contamination and filtering artefacts, measured. The paper that makes contamination
concrete rather than theoretical.

---

## Practitioner writing (often more useful than papers)

**nanoGPT** (Karpathy, code) — the reference minimal training loop. This repo's
`train.py` is a descendant. Read `model.py` and `train.py`, both short.

**Let's build GPT / Let's build the GPT Tokenizer** (Karpathy, video) — the fastest
path from "I can read the code" to "I could have written it". The tokenizer one in
particular saves you a week.

**The Hugging Face TRL docs** — the practical reference for GRPO/DPO/SFT as actually
implemented, including the current defaults (which move; check them rather than
trusting any doc, including mine).

**Interconnects** (Nathan Lambert) — the most reliable running commentary on
post-training, with enough technical detail to be actionable.

**Cameron Wolfe's Deep Learning Focus** — long-form explainers; the GRPO tricks post
is a good example of practitioner-level detail that papers omit.

---

## How to read a paper, given your job

You do not need to reproduce results. You need to answer three questions:

1. **What is being claimed, and against what baseline?** A 3-point gain over a weak
   baseline is not a result. Check whether the baseline was tuned as hard as the
   method.
2. **What is the eval setup?** Prompt template, few-shot count, decoding parameters,
   `n`. If any is unstated, treat the number as approximate — and notice that "we
   report the best of 3 runs" is a different metric from "we report the mean".
3. **What would this change in our pipeline?** Usually nothing, and saying so is
   valuable. Occasionally it is one flag — as with `ratio_mode="sequence"` for an MoE
   policy — and then it is worth a week.
