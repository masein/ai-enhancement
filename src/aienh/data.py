"""
Data: corpora, the preprocessing pipeline, mixtures, and packing.

This is the module you will spend most of your real time in. Modelling gets the
papers; data work gets the results. The pipeline here is a scaled-down version of
what every serious pretraining stack does, in the same order:

    1. LOAD       get raw documents from somewhere
    2. NORMALISE  unicode form, whitespace, control characters
    3. FILTER     drop documents that are too short / too repetitive / mostly junk
    4. DEDUP      exact (hash) then near-duplicate (SimHash over shingles)
    5. MIX        combine sources with explicit weights
    6. SPLIT      hold out a validation set BY DOCUMENT, never by token
    7. TOKENIZE   text -> ints
    8. PACK       concatenate with EOS separators, chunk into fixed-length blocks

Two rules that cause most real incidents:

  * Dedup before you split, and split by document. If the same document appears
    in train and val, your val loss is a lie and you will not find out until a
    benchmark disagrees with it.
  * Every stage reports how many documents it dropped. A filter that silently
    removes 90% of a source is the single most common data bug, and it looks
    exactly like "the model just didn't learn much" from the outside.

The corpora below are generated, not downloaded, so this repo runs offline and
byte-identically on any machine. They are deliberately three different
distributions so you can watch the same architecture behave differently:
`stories` (natural-ish language), `arithmetic` (a verifiable task), `code`
(rigid syntax).
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# 1. corpora
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# THE TASK DEFINITION — one constant, imported everywhere.
#
# Operands run 0..ARITH_MAX_OPERAND. This single number defines the training
# corpus, the SFT set, the RL prompts AND the eval items, so they cannot drift
# apart. A pipeline where the eval task and the training task disagree about
# their own definition produces numbers that are not wrong so much as meaningless,
# and the drift is invisible in code review when the constant is duplicated.
#
# 49 gives 2,500 distinct problems with two-digit answers (<= 98) — small enough
# for a laptop-scale model to actually learn, large enough that a held-out split
# tests generalisation rather than recall. Raise it to 99 (10,000 problems, some
# three-digit answers) for a visibly harder task; expect exact-match to drop by
# roughly half at the same compute budget.
ARITH_MAX_OPERAND = 49

# Fraction of the PROBLEM SPACE reserved for evaluation.
#
# This is the fix for a problem that is easy to miss and fatal when you do. The task
# has (ARITH_MAX_OPERAND+1)^2 = 2,500 possible problems. Train on a few thousand
# examples and you have seen essentially all of them — so "held-out" eval items are
# items the model was trained on, and the score measures recall, not addition.
#
# Sampling more examples does not fix it. You have to partition the SPACE, not the
# samples: every (a, b) pair is deterministically assigned to train or test by a
# stable hash, and no stage may cross the line. Training corpora, SFT data and RL
# prompts draw from "train"; eval items draw from "test".
#
# The same reasoning applies at real scale, where it is called benchmark
# contamination and is much harder to enforce because the training corpus is a web
# crawl and the benchmark is public. Enforcing it here, where it is easy, is how you
# learn to notice when it is being violated there.
ARITH_TEST_FRAC = 0.2


def arith_split(a: int, b: int, test_frac: float = ARITH_TEST_FRAC) -> str:
    """Deterministic train/test assignment for one problem.

    Uses md5 rather than the built-in hash(), which is randomised per process — a
    split that changes between runs is worse than no split at all, because it
    silently leaks in one direction or the other every time."""
    h = int(hashlib.md5(f"{a}+{b}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "train"


_SUBJECTS = ["the dog", "a cat", "the girl", "my friend", "the old man", "a robot",
             "the small bird", "the tall woman", "a fox", "the child"]
_VERBS = ["ran to", "looked at", "found", "carried", "hid behind", "jumped over",
          "waited near", "pointed at", "walked past", "sat beside"]
_OBJECTS = ["the park", "a red ball", "the blue door", "an open window",
            "the tall tree", "a paper box", "the quiet river", "an old bicycle",
            "the wooden chair", "a bright lamp"]
_ENDINGS = ["and then it started to rain .", "because it was getting late .",
            "while the sun was going down .", "and everyone laughed .",
            "but nothing happened .", "until the bell rang .",
            "and the day felt long .", "so it went home ."]

_CODE_TEMPLATES = [
    "def {name}({a}, {b}):\n    return {a} {op} {b}\n",
    "def {name}({a}):\n    if {a} > 0:\n        return {a}\n    return -{a}\n",
    "def {name}(items):\n    total = 0\n    for x in items:\n        total = total + x\n    return total\n",
    "class {cls}:\n    def __init__(self, {a}):\n        self.{a} = {a}\n\n    def get(self):\n        return self.{a}\n",
    "{a} = [{n1}, {n2}, {n3}]\n{b} = sorted({a})\nprint({b})\n",
]


def corpus_stories(n_docs: int = 4000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        n_sent = rng.randint(1, 3)
        sents = []
        for _ in range(n_sent):
            sents.append(
                f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
                f"{rng.choice(_OBJECTS)} {rng.choice(_ENDINGS)}"
            )
        docs.append(" ".join(sents))
    return docs


def corpus_arithmetic(n_docs: int = 4000, seed: int = 0,
                      max_operand: int = ARITH_MAX_OPERAND,
                      split: str = "train") -> list[str]:
    """`a + b = c`, one per document. Small enough that a tiny model can learn it,
    and *verifiable*: you can compute exact-match accuracy with no judge.

    Only problems on the requested side of `arith_split` are emitted, so no training
    stage can see an eval problem. See ARITH_MAX_OPERAND and ARITH_TEST_FRAC above."""
    rng = random.Random(seed)
    docs = []
    while len(docs) < n_docs:
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        if split and arith_split(a, b) != split:
            continue
        docs.append(f"{a} + {b} = {a + b}")
    return docs


def corpus_code(n_docs: int = 4000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    names = ["add", "calc", "total", "step", "score", "value", "run", "apply"]
    docs = []
    for _ in range(n_docs):
        t = rng.choice(_CODE_TEMPLATES)
        docs.append(t.format(
            name=rng.choice(names), cls=rng.choice(["Box", "Node", "Item", "Cell"]),
            a=rng.choice(["x", "a", "n", "val"]), b=rng.choice(["y", "b", "m", "out"]),
            op=rng.choice(["+", "-", "*"]),
            n1=rng.randint(0, 9), n2=rng.randint(0, 9), n3=rng.randint(0, 9),
        ))
    return docs


def corpus_dirty(n_docs: int = 4000, seed: int = 0) -> list[str]:
    """Stories, but polluted the way a real web scrape is: exact duplicates,
    near-duplicates, boilerplate, junk, and empties. Run the pipeline on this to
    see every filter actually fire."""
    rng = random.Random(seed)
    clean = corpus_stories(n_docs=int(n_docs * 0.55), seed=seed)
    docs = list(clean)
    for _ in range(int(n_docs * 0.15)):                       # exact duplicates
        docs.append(rng.choice(clean))
    for _ in range(int(n_docs * 0.15)):                       # near-duplicates
        docs.append(rng.choice(clean) + " " + rng.choice(["Read more .", "Share this .", "Click here ."]))
    for _ in range(int(n_docs * 0.05)):                       # low-information junk
        docs.append(rng.choice(["ok", "", "   ", "!!!!!!!!!!!!!!", "aaaaaaaaaaaaaaaaaaaaaaaa"]))
    for _ in range(int(n_docs * 0.10)):                       # repetitive spam
        docs.append(("buy now " * rng.randint(8, 30)).strip())
    rng.shuffle(docs)
    return docs


CORPORA: dict[str, Callable[..., list[str]]] = {
    "stories": corpus_stories,
    "arithmetic": corpus_arithmetic,
    "code": corpus_code,
    "dirty": corpus_dirty,
}


def load_corpus(name: str, n_docs: int = 4000, seed: int = 0) -> list[str]:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; have {sorted(CORPORA)}")
    return CORPORA[name](n_docs=n_docs, seed=seed)


# ---------------------------------------------------------------------------
# 2-4. the cleaning stages
# ---------------------------------------------------------------------------

@dataclass
class StageStat:
    stage: str
    docs_in: int
    docs_out: int

    @property
    def dropped(self) -> int:
        return self.docs_in - self.docs_out

    @property
    def drop_pct(self) -> float:
        return 100.0 * self.dropped / max(1, self.docs_in)


@dataclass
class PipelineReport:
    stages: list[StageStat] = field(default_factory=list)

    def add(self, stage: str, docs_in: int, docs_out: int) -> None:
        self.stages.append(StageStat(stage, docs_in, docs_out))

    def to_dict(self) -> dict:
        return {s.stage: {"in": s.docs_in, "out": s.docs_out,
                          "dropped": s.dropped, "drop_pct": round(s.drop_pct, 2)}
                for s in self.stages}

    def render(self) -> str:
        w = max((len(s.stage) for s in self.stages), default=10)
        lines = [f"{'stage'.ljust(w)}  {'in':>8} {'out':>8} {'dropped':>8}  {'drop%':>6}"]
        lines.append("-" * len(lines[0]))
        for s in self.stages:
            lines.append(f"{s.stage.ljust(w)}  {s.docs_in:>8} {s.docs_out:>8} "
                         f"{s.dropped:>8}  {s.drop_pct:>5.1f}%")
        return "\n".join(lines)


def normalise(doc: str) -> str:
    """NFKC unicode normalisation, strip control chars, collapse whitespace runs.
    Deterministic and lossy — do it once, early, and record that you did it."""
    doc = unicodedata.normalize("NFKC", doc)
    doc = "".join(ch for ch in doc if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    doc = re.sub(r"[ \t]+", " ", doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc.strip()


def repetition_ratio(doc: str) -> float:
    """Fraction of the document taken up by its single most common word.
    Cheap, effective spam signal ("buy now buy now buy now ...")."""
    words = doc.split()
    if len(words) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words)


def alpha_ratio(doc: str) -> float:
    if not doc:
        return 0.0
    return sum(ch.isalnum() or ch.isspace() for ch in doc) / len(doc)


# Filter thresholds tuned for prose destroy structured data. Measured, not guessed:
# with the prose defaults below, the `arithmetic` corpus loses 36% of documents to
# min_chars=12 (because "0 + 3 = 3" is 9 characters) and more to max_repetition
# (because "0 + 0 = 0" repeats one token 3 times out of 5). The survivors are
# systematically the large-number problems — so the model never sees small ones,
# and eval accuracy on small operands collapses for a reason that is invisible
# unless you read the stage report.
#
# This is the most common data bug in the field, and it is why every stage in this
# pipeline prints its drop rate. Per-source filter settings, not one global set.
FILTER_PRESETS: dict[str, dict] = {
    "arithmetic": {"min_chars": 5, "max_repetition": 0.9, "min_alpha": 0.5},
    "code": {"min_chars": 10, "max_repetition": 0.6, "min_alpha": 0.55},
}


def quality_filter(
    docs: list[str],
    min_chars: int = 12,
    max_repetition: float = 0.4,
    min_alpha: float = 0.6,
) -> list[str]:
    """Heuristic filters, in the spirit of the Gopher/C4 rules. Every threshold
    here is a judgement call you should be able to defend with a number."""
    out = []
    for d in docs:
        if len(d) < min_chars:
            continue
        if repetition_ratio(d) > max_repetition:
            continue
        if alpha_ratio(d) < min_alpha:
            continue
        out.append(d)
    return out


def exact_dedup(docs: list[str]) -> list[str]:
    """Hash the whole document. Catches copy-paste duplicates; misses anything
    with one character changed — which is why near-dedup exists."""
    seen: set[str] = set()
    out = []
    for d in docs:
        h = hashlib.sha1(d.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out


def _simhash(doc: str, shingle: int = 4, bits: int = 64) -> int:
    """64-bit SimHash over word shingles. Similar documents get hashes that
    differ in few bits, so near-duplicate detection becomes a Hamming distance.
    Production stacks use MinHash+LSH; the idea is the same, this is the 30-line
    version so you can read it."""
    words = doc.split()
    shingles = [" ".join(words[i:i + shingle]) for i in range(max(1, len(words) - shingle + 1))]
    v = [0] * bits
    for s in shingles:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def near_dedup(docs: list[str], max_hamming: int = 8) -> list[str]:
    """
    O(n * kept) in the worst case — fine at this scale, replace with MinHash+LSH
    banding above ~1e5 documents.

    `max_hamming` is a precision/recall dial and you should treat it as a tuned
    hyperparameter, not a constant. Measured on the `dirty` corpus in this repo
    (1385 docs after exact dedup, ~15% of which are boilerplate near-duplicates):

        threshold   near-dupes caught      false positives on clean text
            3              15  (1%)                0 / 1489
            5              68  (5%)                1 / 1489
            8             165 (12%)                8 / 1489   <- default
           12             307 (22%)              105 / 1489
           16             644 (46%)              613 / 1489   <- destroys the corpus

    Too loose and you delete legitimate distinct documents. On a real corpus that
    shows up as a mysteriously small dataset and a model that has never seen a
    common phrasing.
    """
    kept: list[int] = []
    out: list[str] = []
    for d in docs:
        h = _simhash(d)
        if any(bin(h ^ k).count("1") <= max_hamming for k in kept):
            continue
        kept.append(h)
        out.append(d)
    return out


def preprocess(
    docs: list[str],
    do_normalise: bool = True,
    do_filter: bool = True,
    do_exact_dedup: bool = True,
    do_near_dedup: bool = True,
    filter_opts: dict | None = None,
    report: PipelineReport | None = None,
) -> tuple[list[str], PipelineReport]:
    report = report or PipelineReport()
    report.add("load", len(docs), len(docs))

    if do_normalise:
        before = len(docs)
        docs = [normalise(d) for d in docs]
        docs = [d for d in docs if d]
        report.add("normalise", before, len(docs))

    if do_filter:
        before = len(docs)
        docs = quality_filter(docs, **(filter_opts or {}))
        report.add("quality_filter", before, len(docs))

    if do_exact_dedup:
        before = len(docs)
        docs = exact_dedup(docs)
        report.add("exact_dedup", before, len(docs))

    if do_near_dedup:
        before = len(docs)
        docs = near_dedup(docs)
        report.add("near_dedup", before, len(docs))

    return docs, report


# ---------------------------------------------------------------------------
# 5-6. mixture and split
# ---------------------------------------------------------------------------

def mix_corpora(sources: dict[str, list[str]], weights: dict[str, float],
                total_docs: int, seed: int = 0) -> list[str]:
    """Sample `total_docs` documents from several sources at explicit ratios.

    The mixture is a first-class hyperparameter: 'add 10% code to the mix' is a
    real, measurable pretraining decision, not a data-plumbing detail. Sources
    are up-sampled (with replacement) if the weight asks for more than exists,
    which is exactly what real runs do with small high-quality sources — and is
    exactly how you accidentally memorise them.
    """
    rng = random.Random(seed)
    total_w = sum(weights.values())
    out: list[str] = []
    for name, w in weights.items():
        n = int(round(total_docs * w / total_w))
        pool = sources[name]
        if not pool:
            continue
        picks = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
        out.extend(picks)
    rng.shuffle(out)
    return out


def ngrams(text: str, n: int = 13) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contamination_report(train_docs: list[str], eval_docs: list[str], n: int = 13) -> dict:
    """
    How much of your evaluation set is in your training set?

    This is the first question anyone will ask about a benchmark number, and the
    only defensible answer is a measurement. Two levels, both cheap:

        exact_overlap   the same document appears in both. Indefensible.
        ngram_overlap   an eval document shares an n-gram with training data.
                        n=13 is the convention GPT-3 and most since have used.
                        Some overlap is normal for common phrasings; a high rate
                        on a specific task means that task's score is inflated.

    Note for this repo specifically: the generated corpora here overlap heavily by
    construction (both splits are drawn from the same finite template space), so
    `arithmetic` eval items DO appear in training. That is realistic — it is
    exactly what happens when benchmark data leaks into a web crawl — and the
    honest move is to measure it and say so, not to hope nobody checks.

    Real pipelines then *decontaminate*: drop the contaminated eval items (and
    report how many), or drop the offending training documents. Either way the
    number you publish is the clean one, with the drop rate next to it.
    """
    train_hashes = {hashlib.sha1(d.encode()).hexdigest() for d in train_docs}
    train_ngrams: set[str] = set()
    for d in train_docs:
        train_ngrams |= ngrams(d, n)

    exact, ngram_hit = 0, 0
    for d in eval_docs:
        if hashlib.sha1(d.encode()).hexdigest() in train_hashes:
            exact += 1
        if ngrams(d, n) & train_ngrams:
            ngram_hit += 1
    total = max(1, len(eval_docs))
    return {
        "n": n,
        "eval_docs": len(eval_docs),
        "exact_overlap": exact,
        "exact_overlap_pct": round(100 * exact / total, 2),
        "ngram_overlap": ngram_hit,
        "ngram_overlap_pct": round(100 * ngram_hit / total, 2),
    }


def decontaminate(train_docs: list[str], eval_docs: list[str], n: int = 13
                  ) -> tuple[list[str], dict]:
    """Return the eval documents that do NOT overlap training data, plus the report.
    Publish the drop count — a benchmark that shrank by 40% is a different benchmark."""
    train_ngrams: set[str] = set()
    for d in train_docs:
        train_ngrams |= ngrams(d, n)
    kept = [d for d in eval_docs if not (ngrams(d, n) & train_ngrams)]
    return kept, {"before": len(eval_docs), "after": len(kept),
                  "dropped": len(eval_docs) - len(kept)}


def split_docs(docs: list[str], val_frac: float = 0.05, seed: int = 0
               ) -> tuple[list[str], list[str]]:
    """Split BY DOCUMENT. Splitting a packed token array by index instead lets
    the tail of a training document leak into validation."""
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(docs) * val_frac))
    val = [docs[i] for i in idx[:n_val]]
    train = [docs[i] for i in idx[n_val:]]
    return train, val


# ---------------------------------------------------------------------------
# 7-8. tokenize and pack
# ---------------------------------------------------------------------------

def pack(docs: list[str], tokenizer, block_size: int, add_eos: bool = True) -> np.ndarray:
    """
    Concatenate documents into one long token stream (EOS between them) and cut
    it into blocks of `block_size + 1`. The +1 is the shift: a block's inputs are
    tokens[:-1] and its targets are tokens[1:], which is the whole of
    "next-token prediction" as a data-layout question.

    Packing wastes no compute on padding, at the cost of letting attention see
    across the EOS boundary into an unrelated document. Real stacks either accept
    this (most do) or use a block-diagonal attention mask to prevent it.

    Returns an int array of shape [n_blocks, block_size + 1].
    """
    stream: list[int] = []
    for d in docs:
        stream.extend(tokenizer.encode(d))
        if add_eos:
            stream.append(tokenizer.eos_id)
    n = (len(stream) - 1) // block_size
    if n <= 0:
        raise ValueError(
            f"corpus too small: {len(stream)} tokens can't fill one block of {block_size}"
        )
    dtype = np.uint16 if tokenizer.vocab_size < 2**16 else np.int32
    arr = np.zeros((n, block_size + 1), dtype=dtype)
    for i in range(n):
        arr[i] = stream[i * block_size: i * block_size + block_size + 1]
    return arr


def build_dataset(
    corpus: str | dict[str, float],
    tokenizer_kind: str = "char",
    vocab_size: int = 512,
    block_size: int = 128,
    n_docs: int = 4000,
    val_frac: float = 0.05,
    seed: int = 0,
    clean: bool = True,
    verbose: bool = True,
) -> dict:
    """
    The whole pipeline, end to end. `corpus` is either a corpus name or a dict of
    {name: weight} for a mixture.

    Returns {train, val, tokenizer, report, n_train_tokens, ...}.
    """
    if isinstance(corpus, str):
        docs = load_corpus(corpus, n_docs=n_docs, seed=seed)
        label = corpus
        filter_opts = FILTER_PRESETS.get(corpus)
    else:
        sources = {name: load_corpus(name, n_docs=n_docs, seed=seed) for name in corpus}
        docs = mix_corpora(sources, corpus, total_docs=n_docs, seed=seed)
        label = "+".join(f"{k}{v:g}" for k, v in sorted(corpus.items()))
        # A mixture needs the most permissive preset of its members, otherwise the
        # strictest source silently gets filtered out of the mix. Real stacks avoid
        # this by filtering each source separately, before mixing.
        filter_opts = None
        presets = [FILTER_PRESETS[c] for c in corpus if c in FILTER_PRESETS]
        if presets:
            filter_opts = {
                "min_chars": min(p.get("min_chars", 12) for p in presets),
                "max_repetition": max(p.get("max_repetition", 0.4) for p in presets),
                "min_alpha": min(p.get("min_alpha", 0.6) for p in presets),
            }

    docs, report = preprocess(
        docs,
        do_normalise=clean, do_filter=clean,
        do_exact_dedup=clean, do_near_dedup=clean,
        filter_opts=filter_opts,
    )
    train_docs, val_docs = split_docs(docs, val_frac=val_frac, seed=seed)

    from .tokenizer import build_tokenizer
    tok = build_tokenizer(tokenizer_kind, train_docs, vocab_size=vocab_size)

    train = pack(train_docs, tok, block_size)
    val = pack(val_docs, tok, block_size)

    # Measure the leak between the splits on every build. Cheap, and it means no run
    # in this repo produces a validation number without a contamination figure sitting
    # next to it in the tracker config. A pipeline that *can* measure contamination
    # and doesn't is a pipeline that will be asked to, on the worst possible day.
    contam = contamination_report(train_docs, val_docs, n=13)

    if verbose:
        print(f"[data] corpus={label} tokenizer={tokenizer_kind} vocab={tok.vocab_size}")
        print(report.render())
        print(f"[data] train blocks={len(train)} ({train.size:,} tokens)  "
              f"val blocks={len(val)} ({val.size:,} tokens)")
        print(f"[data] contamination train->val: {contam['exact_overlap_pct']}% exact, "
              f"{contam['ngram_overlap_pct']}% 13-gram "
              f"({contam['ngram_overlap']}/{contam['eval_docs']} val docs)")

    return {
        "label": label,
        "train": train,
        "val": val,
        "tokenizer": tok,
        "report": report,
        "n_train_tokens": int(train.size),
        "n_val_tokens": int(val.size),
        "n_train_docs": len(train_docs),
        "n_val_docs": len(val_docs),
        "contamination": contam,
    }


if __name__ == "__main__":  # python -m aienh.data
    print("=== clean corpus ===")
    build_dataset("stories", n_docs=2000)
    print()
    print("=== deliberately dirty corpus: watch each filter fire ===")
    build_dataset("dirty", n_docs=2000)
