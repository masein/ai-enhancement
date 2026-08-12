"""
The evaluation harness. This is the part of the stack you are being hired to own.

An eval harness is a small number of primitives plus a lot of discipline.

THE TWO PRIMITIVES
    loglikelihood(context, continuation) -> (sum log P(continuation | context), n_tokens)
        Answers "how much does the model like this exact string?" without
        generating anything. Every multiple-choice benchmark (MMLU, ARC, HellaSwag,
        WinoGrande) is built on this: score each option, pick the argmax. No
        parsing, no judge, zero sampling noise. This is why MC benchmarks are
        cheap and stable — and why they measure something narrower than they
        appear to.
    generate(context) -> string
        Answers "what does the model actually do?" Needed for GSM8K, HumanEval,
        IFEval, anything agentic. Requires a parser (or a judge) and is sensitive
        to decoding parameters, which is where most irreproducible numbers come
        from.

THE DISCIPLINE — the parts that make numbers trustworthy
    1. Fixed items.       Same seed, same questions, every model, forever. If your
                          item set drifts, cross-model comparisons are noise.
    2. Fixed decoding.    Greedy (temperature=0) when measuring, unless the metric
                          is explicitly sampling-based (pass@k). Record it.
    3. Report stderr.     A 2-point gap on 200 items is not a result:
                          stderr = sqrt(p(1-p)/n) ~= 3.5 points at p=0.5, n=200.
                          Report n and stderr next to every number, always.
    4. Log samples.       Persist the raw generations. Every "the model got worse"
                          investigation starts by reading 20 of them, and half of
                          them end at "the parser broke", not "the model broke".
    5. Version the suite. A benchmark whose code changed is a different benchmark.
                          Hash the item set and the harness config into the result.
    6. Slice the metric.  One aggregate hides everything. `arith_exact` below is
                          reported overall AND split by operand size, which is how
                          you catch a data-filtering bias that a single number
                          would have hidden completely.

WHAT THIS DOES NOT DO, on purpose: no LLM-as-judge. It needs its own doc (judge
model version, prompt version, position bias, agreement with humans). Programmatic
metrics first; add a judge only when you have measured its agreement with people.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F

from .data import ARITH_MAX_OPERAND, arith_split, corpus_arithmetic, load_corpus, pack


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task: str
    metric: str
    value: float
    n: int
    stderr: float = 0.0
    extra: dict = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["samples"] = self.samples[:10]   # keep results files small
        return d

    def __str__(self) -> str:
        pm = f" +/- {self.stderr:.3f}" if self.stderr else ""
        return f"{self.task:>18s}  {self.metric:<10s} {self.value:8.4f}{pm}  (n={self.n})"


def binom_stderr(p: float, n: int) -> float:
    """Standard error of a proportion. Print it or your comparisons are vibes."""
    if n <= 1:
        return 0.0
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


# ---------------------------------------------------------------------------
# primitive 1: loglikelihood
# ---------------------------------------------------------------------------

@torch.no_grad()
def loglikelihood(model, tok, context: str, continuations: list[str], device):
    """
    Score each continuation given the same context, in one batch.

    Returns a list of (sum_logprob, n_tokens, is_greedy) — the same triple
    lm-evaluation-harness returns, because these are the three things every MC
    metric is built from:
        acc       argmax over sum_logprob
        acc_norm  argmax over sum_logprob / n_chars   (length-normalised)
        is_greedy whether the model would have produced it under greedy decoding

    Why acc_norm exists: longer continuations accumulate more negative logprob
    just by being longer, so raw sums systematically favour short options. When a
    paper reports acc and acc_norm and they disagree, the benchmark is partly
    measuring option length. Know which one you are quoting.

    Right-padding is safe here: attention is causal, so a pad token at position j
    cannot influence any position < j, and we only ever read positions inside the
    real sequence.
    """
    ctx_ids = tok.encode(context)
    seqs, spans = [], []
    for cont in continuations:
        cont_ids = tok.encode(cont)
        seqs.append(ctx_ids + cont_ids)
        spans.append((len(ctx_ids), len(cont_ids)))

    max_len = max(len(s) for s in seqs)
    if max_len > model.cfg.block_size:
        raise ValueError(f"item longer ({max_len}) than block_size ({model.cfg.block_size})")
    pad = getattr(tok, "pad_id", 0)
    batch = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    logits, _, _ = model(batch)
    logprobs = F.log_softmax(logits.float(), dim=-1)

    out = []
    for i, (start, n_cont) in enumerate(spans):
        if n_cont == 0:
            out.append((float("-inf"), 0, False))
            continue
        total, greedy = 0.0, True
        for j in range(n_cont):
            # position start+j-1 predicts token at start+j
            pos = start + j - 1
            if pos < 0:
                # Empty context: there is no position that predicts the very first
                # token, so it contributes nothing. Without this guard Python's
                # negative indexing silently reads the LAST position and returns a
                # plausible number computed from the wrong place.
                continue
            target = int(batch[i, start + j])
            total += float(logprobs[i, pos, target])
            greedy &= int(logprobs[i, pos].argmax()) == target
        out.append((total, n_cont, greedy))
    return out


# ---------------------------------------------------------------------------
# primitive 2: batched greedy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int, device,
                   greedy: bool = True, temperature: float = 1.0) -> list[str]:
    """
    Generate a continuation for each prompt.

    Prompts are grouped by token length and batched within groups. Reason: this
    model (like a raw nn.Module without an attention mask) cannot left-pad — pad
    tokens on the left would be attended to as real context and quietly change the
    output. Production harnesses solve this with an attention mask; grouping is the
    honest 10-line version. If you ever see eval scores change when you change
    --batch_size, this is the first thing to check.
    """
    encoded = [tok.encode(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    completions: list[str | None] = [None] * len(prompts)

    i = 0
    while i < len(order):
        L = len(encoded[order[i]])
        group = [j for j in order[i:] if len(encoded[j]) == L]
        i += len(group)
        x = torch.tensor([encoded[j] for j in group], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens, greedy=greedy, temperature=temperature,
                             eos_id=getattr(tok, "eos_id", None))
        for row, j in zip(out, group):
            completions[j] = tok.decode(row[L:].tolist())
    return [c or "" for c in completions]


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"-?\d+")


def parse_int(text: str) -> int | None:
    """The parser is part of the benchmark. A stricter parser measures format
    compliance as well as capability; a looser one measures capability alone.
    Neither is wrong — but you must know which one you shipped, because switching
    it silently moves every score you have ever reported."""
    text = text.split("<|endoftext|>")[0]
    m = _ANSWER_RE.search(text)
    return int(m.group()) if m else None


def task_perplexity(model, tok, device, corpus: str = "stories", n_docs: int = 400,
                    block_size: int | None = None, seed: int = 1234) -> TaskResult:
    """
    Perplexity = exp(mean token cross-entropy) = "how many equally-likely options
    was the model effectively choosing between at each token". Lower is better;
    ~1.0 is perfect.

    It is the cheapest signal you have and the one most often misused: perplexity
    is only comparable across models that share a tokenizer and an evaluation
    corpus. Different tokenizer => different number of tokens for the same text =>
    the numbers are not on the same scale. If you must compare across tokenizers,
    use bits per byte, which this task also reports.
    """
    block_size = block_size or model.cfg.block_size
    # For the arithmetic corpus this pulls the TEST side of the problem space, so
    # perplexity is measured on unseen problems like every other metric here.
    docs = (corpus_arithmetic(n_docs=n_docs, seed=seed, split="test")
            if corpus == "arithmetic" else load_corpus(corpus, n_docs=n_docs, seed=seed))
    blocks = pack(docs, tok, block_size)

    # Bytes of exactly the text we score, so bits-per-byte is honest. (EOS tokens
    # contribute loss but no bytes; that is the standard convention and the bias is
    # tiny at block_size >> 1.)
    scored_ids = blocks[:, 1:].reshape(-1).tolist()
    n_bytes = len(tok.decode([i for i in scored_ids if i != tok.eos_id]).encode("utf-8"))

    model.eval()
    total_nll, total_tokens = 0.0, 0
    bs = 16
    for s in range(0, len(blocks), bs):
        chunk = torch.from_numpy(blocks[s:s + bs].astype("int64")).to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        with torch.no_grad():
            logits, _, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tokens += y.numel()

    mean_nll = total_nll / max(1, total_tokens)
    # bits per byte is tokenizer-independent: same text, same denominator, so it IS
    # comparable across tokenizers where perplexity is not.
    bpb = (total_nll / math.log(2)) / max(1, n_bytes)
    return TaskResult(
        task=f"ppl_{corpus}", metric="perplexity", value=math.exp(min(20, mean_nll)),
        n=total_tokens,
        extra={"mean_nll": mean_nll, "bits_per_byte": bpb, "tokens": total_tokens},
    )


# The prompt format is part of the benchmark, not a detail.
#
# A base (pretrained-only) model expects raw continuation: "17 + 45 =".
# A model that was SFT'd on a template expects that template: "Q: 17 + 45 =\nA:".
# Score either one under the other's format and you measure format mismatch, not
# capability — and the drop is large enough to look like a real regression.
# scripts/demo_template_mismatch.py measures exactly this. Every result in this
# repo records which template produced it.
RAW_TEMPLATE = "{q}"
CHAT_TEMPLATE = "Q: {q}\nA:"


def arith_items(n: int = 200, seed: int = 1234,
                max_operand: int = ARITH_MAX_OPERAND,
                template: str = RAW_TEMPLATE, split: str = "test") -> list[dict]:
    """
    Fixed evaluation items, drawn from the TEST side of the problem space.

    Two separate guarantees, and you need both:

      * `split="test"` means no training stage has seen these problems (see
        data.arith_split). This is decontamination enforced by construction rather
        than measured after the fact.
      * the fixed `seed` means every model, forever, is asked the same questions.
        Changing it invalidates every historical comparison — treat it like a schema
        migration, not a tweak.

    Pass `split="train"` to deliberately measure on problems the model was trained
    on. The train-minus-test gap is the model's memorisation, and it is worth
    looking at once so you know what contamination is worth on your task.
    """
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
        if split and arith_split(a, b) != split:
            continue
        items.append({"prompt": template.format(q=f"{a} + {b} ="),
                      "answer": a + b, "a": a, "b": b})
    return items


def task_arith_exact(model, tok, device, n: int = 200, seed: int = 1234,
                     max_new_tokens: int = 6, template: str = RAW_TEMPLATE,
                     split: str = "test") -> TaskResult:
    """
    Generative exact-match on held-out addition. Greedy decoding, so the number is
    deterministic given the checkpoint.

    Reported overall AND sliced by operand size. The slice is the point: a single
    aggregate cannot distinguish "the model is 60% correct" from "the model is 95%
    correct on large operands and 20% on small ones because a length filter
    removed short examples from the training data".
    """
    items = arith_items(n, seed, template=template, split=split)
    outs = generate_batch(model, tok, [it["prompt"] for it in items],
                          max_new_tokens, device, greedy=True)
    correct, samples = [], []
    buckets = {"both_small(<=9)": [], "mixed": [], "both_large(>=10)": []}
    for it, out in zip(items, outs):
        pred = parse_int(out)
        ok = pred == it["answer"]
        correct.append(ok)
        small = (it["a"] <= 9) + (it["b"] <= 9)
        buckets["both_small(<=9)" if small == 2 else "mixed" if small == 1
                else "both_large(>=10)"].append(ok)
        if len(samples) < 20:
            samples.append({"prompt": it["prompt"], "raw": out,
                            "pred": pred, "gold": it["answer"], "correct": ok})
    acc = sum(correct) / len(correct)
    return TaskResult(
        task="arith_exact", metric="exact_match", value=acc, n=len(correct),
        stderr=binom_stderr(acc, len(correct)),
        extra={"template": template, "split": split,
               "parse_failures": sum(1 for s in samples if s["pred"] is None),
               "by_operand_size": {k: (round(sum(v) / len(v), 4) if v else None, len(v))
                                   for k, v in buckets.items()}},
        samples=samples,
    )


def task_arith_mc(model, tok, device, n: int = 200, seed: int = 1234,
                  n_choices: int = 4, template: str = RAW_TEMPLATE,
                  split: str = "test") -> TaskResult:
    """
    The same capability, measured multiple-choice style via loglikelihood.

    Run this next to `arith_exact` and compare. The MC number is almost always
    higher, because picking the best of 4 given options is an easier task than
    producing the answer — 25% is free. This gap is exactly why "our model scores
    82% on <MC benchmark>" and "our model is useful" are different claims, and why
    a serious eval suite contains both kinds of task.
    """
    items = arith_items(n, seed, template=template, split=split)
    rng = random.Random(seed + 1)
    n_correct_raw, n_correct_norm, samples = 0, 0, []
    for it in items:
        gold = it["answer"]
        distractors: set[int] = set()
        while len(distractors) < n_choices - 1:
            d = gold + rng.choice([-10, -9, -2, -1, 1, 2, 9, 10, 11, 100])
            if d != gold and d >= 0:
                distractors.add(d)
        choices = [gold, *distractors]
        rng.shuffle(choices)
        conts = [f" {c}" for c in choices]
        scored = loglikelihood(model, tok, it["prompt"], conts, device)
        raw = max(range(len(choices)), key=lambda i: scored[i][0])
        norm = max(range(len(choices)), key=lambda i: scored[i][0] / max(1, len(conts[i])))
        n_correct_raw += choices[raw] == gold
        n_correct_norm += choices[norm] == gold
        if len(samples) < 20:
            samples.append({"prompt": it["prompt"], "choices": choices, "gold": gold,
                            "pred": choices[raw],
                            "logprobs": [round(s[0], 3) for s in scored]})
    acc = n_correct_raw / len(items)
    return TaskResult(
        task="arith_mc4", metric="acc", value=acc, n=len(items),
        stderr=binom_stderr(acc, len(items)),
        extra={"acc_norm": n_correct_norm / len(items),
               "random_baseline": 1.0 / n_choices, "template": template,
               "split": split},
        samples=samples,
    )


def task_arith_pass_at_k(model, tok, device, n: int = 60, k: int = 8,
                         temperature: float = 0.9, seed: int = 1234,
                         template: str = RAW_TEMPLATE, split: str = "test") -> TaskResult:
    """
    pass@k: sample k times, count the item as solved if ANY sample is right.

    This is the metric family HumanEval popularised, and it answers a different
    question from greedy accuracy: not "what does the model say" but "does the
    model know". A big pass@1 -> pass@8 gap means the capability is present but
    unreliable — which is precisely the gap RL post-training (GRPO) is good at
    closing, because it reinforces the samples that already happen to be correct.
    Measure this BEFORE and AFTER any RL run; it is the cleanest evidence of what
    RL actually did.
    """
    items = arith_items(n, seed + 7, template=template, split=split)
    prompts = [it["prompt"] for it in items]
    hits = [False] * n
    first_try = [False] * n
    for attempt in range(k):
        outs = generate_batch(model, tok, prompts, 6, device,
                              greedy=False, temperature=temperature)
        for i, (it, out) in enumerate(zip(items, outs)):
            ok = parse_int(out) == it["answer"]
            hits[i] |= ok
            if attempt == 0:
                first_try[i] = ok
    p = sum(hits) / n
    return TaskResult(
        task=f"arith_pass@{k}", metric=f"pass@{k}", value=p, n=n,
        stderr=binom_stderr(p, n),
        extra={"pass@1_sampled": sum(first_try) / n, "temperature": temperature,
               "template": template},
    )


def task_format_compliance(model, tok, device, n: int = 100, seed: int = 1234,
                           template: str = RAW_TEMPLATE, split: str = "test") -> TaskResult:
    """
    A programmatic rubric judge: does the output obey a required format, ignoring
    whether the content is right?

    Trivially cheap, and it catches the failure mode that breaks downstream
    parsers. Instruction-following benchmarks (IFEval and friends) are this idea
    scaled up: a list of verifiable constraints, checked with code rather than a
    model. Prefer verifiable constraints over judge opinions wherever you can
    construct them — they cost nothing and never drift.
    """
    items = arith_items(n, seed + 13, template=template, split=split)
    outs = generate_batch(model, tok, [it["prompt"] for it in items], 8, device, greedy=True)
    pattern = re.compile(r"^ ?\d+(<\|endoftext\|>|$)")
    ok = [bool(pattern.match(o.split("\n")[0])) for o in outs]
    p = sum(ok) / len(ok)
    return TaskResult(
        task="format_ok", metric="rubric_pass", value=p, n=len(ok),
        stderr=binom_stderr(p, len(ok)),
        extra={"rule": r"^ ?\d+ then EOS or end", "template": template},
        samples=[{"prompt": it["prompt"], "raw": o, "ok": k}
                 for it, o, k in list(zip(items, outs, ok))[:20]],
    )


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------

TASKS = {
    "ppl_stories": lambda m, t, d, **kw: task_perplexity(m, t, d, corpus="stories", **kw),
    "ppl_arithmetic": lambda m, t, d, **kw: task_perplexity(m, t, d, corpus="arithmetic", **kw),
    "ppl_code": lambda m, t, d, **kw: task_perplexity(m, t, d, corpus="code", **kw),
    "arith_exact": task_arith_exact,
    # Same task, measured on problems the model WAS trained on. Not in the default
    # suite — it is a contaminated number by construction and must never be mixed
    # into a headline score. Run it deliberately: the gap between it and
    # `arith_exact` is the model's memorisation, i.e. exactly what a contaminated
    # public benchmark would have been rewarding.
    "arith_exact_seen": lambda m, t, d, **kw: task_arith_exact(
        m, t, d, **{**kw, "split": "train"}),
    "arith_mc4": task_arith_mc,
    "arith_pass@8": task_arith_pass_at_k,
    "format_ok": task_format_compliance,
}

DEFAULT_SUITE = ["ppl_stories", "ppl_arithmetic", "arith_exact", "arith_mc4", "format_ok"]

# Weights for the single composite "points" number (0-100). A composite is a
# political object as much as a technical one: it decides which regressions are
# allowed to be invisible. Publish the weights next to the score, always, and keep
# the component table one click away.
POINTS_WEIGHTS = {
    "arith_exact": 40.0,
    "arith_mc4": 20.0,
    "format_ok": 15.0,
    "ppl_arithmetic": 15.0,   # scored on a squashed scale, see points_from_results
    "ppl_stories": 10.0,
}


def points_from_results(results: dict[str, TaskResult]) -> tuple[float, dict]:
    """
    Collapse a result set into one 0-100 number.

    Accuracy-style metrics map directly (0-1 -> 0-100 of their weight).
    Perplexity is unbounded and lower-is-better, so it is squashed with
    1/(1+log(ppl)) — arbitrary but monotone and stable. Any such mapping is
    arbitrary; what matters is that it is FIXED and written down, because the
    moment you change it every historical score becomes incomparable.
    """
    breakdown, earned, available = {}, 0.0, 0.0
    for task, weight in POINTS_WEIGHTS.items():
        r = results.get(task)
        if r is None:
            continue
        available += weight
        if r.metric == "perplexity":
            frac = 1.0 / (1.0 + math.log(max(r.value, 1.0 + 1e-9)))
        else:
            frac = max(0.0, min(1.0, r.value))
        got = weight * frac
        earned += got
        breakdown[task] = {"metric": r.metric, "value": round(r.value, 4),
                           "weight": weight, "points": round(got, 2)}
    total = 100.0 * earned / available if available else 0.0
    return round(total, 2), breakdown


def suite_hash(tasks: list[str], seed: int, template: str = RAW_TEMPLATE) -> str:
    """Version the suite. Changing tasks, the seed, or the prompt template changes
    this hash — and any leaderboard that mixes two hashes is comparing different
    exams. The template is included because it demonstrably moves scores."""
    return hashlib.sha1(json.dumps(
        {"tasks": sorted(tasks), "seed": seed, "template": template}
    ).encode()).hexdigest()[:8]


def run_suite(model, tok, device, tasks: list[str] | None = None, seed: int = 1234,
              verbose: bool = True, task_kwargs: dict | None = None,
              template: str = RAW_TEMPLATE) -> dict:
    """`template` applies to every generative/MC task. Evaluate each model with the
    format it was trained for, and record which one you used."""
    tasks = tasks or DEFAULT_SUITE
    task_kwargs = task_kwargs or {}
    model.eval()
    results: dict[str, TaskResult] = {}
    for name in tasks:
        if name not in TASKS:
            raise ValueError(f"unknown task {name!r}; have {sorted(TASKS)}")
        kw = dict(task_kwargs.get(name, {}))
        if not name.startswith("ppl_"):
            kw.setdefault("template", template)
        r = TASKS[name](model, tok, device, seed=seed, **kw)
        results[name] = r
        if verbose:
            print("   ", r)
    points, breakdown = points_from_results(results)
    if verbose:
        print(f"    {'POINTS':>18s}  {points:.2f} / 100")
    return {
        "points": points,
        "breakdown": breakdown,
        "results": {k: v.to_dict() for k, v in results.items()},
        "suite_hash": suite_hash(tasks, seed, template),
        "suite": tasks,
        "seed": seed,
        "template": template,
    }
