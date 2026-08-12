# 02 — Tokenization and the data pipeline

Code: **`src/aienh/tokenizer.py`**, **`src/aienh/data.py`**.
Run: `python -m aienh data --corpus dirty`

Modelling gets the papers. Data work gets the results. This is also the part of the
stack where mistakes are invisible: a bad filter does not raise an exception, it
just quietly produces a worse model, and the symptom looks exactly like "the
architecture didn't help".

---

## Tokenization

### Why not characters or words

**Characters**: tiny vocabulary, but sequences are 4–5× longer, so you pay 4–5× the
compute for the same text and burn context window on nothing.

**Words**: short sequences, but the vocabulary is unbounded — every typo, every
proper noun, every compound is a new entry — and anything unseen becomes `<unk>`,
which is unrecoverable information loss.

**Subwords (BPE)**: common words get one token, rare words split into pieces,
nothing is ever unrepresentable. This is what everyone uses.

### Byte-level BPE, the algorithm

Start with the 256 raw bytes as the vocabulary. Then repeat: count every adjacent
pair of symbols in the corpus, merge the most frequent pair into a new symbol,
record the merge. Stop at your target vocabulary size. Encoding replays the merges
in learned order.

Two details that matter in practice:

1. **Starting from bytes** means every possible input is encodable. No `<unk>`,
   ever. `tests/test_smoke.py:test_bpe_handles_unseen_bytes` demonstrates this on
   an emoji the tokenizer never saw.
2. **Pre-tokenization** splits text on a regex before merging, so no single token
   can span " the cat sat". Without it, BPE learns whole common phrases and
   generalises worse.

`tokenizer.py:ByteBPETokenizer.train` is the whole algorithm in ~40 lines. Read it
once and tokenization stops being magic.

### Why the tokenizer is your problem, not someone else's

- **It sets your compute bill.** Tokens are the unit of cost. A tokenizer that
  needs 20% more tokens for the same text makes training and inference 20% more
  expensive, forever.
- **It decides what the model finds hard.** If "1234" tokenizes as "123"+"4",
  arithmetic is harder than if it splits into digits. This is a real, documented
  effect on math benchmarks, and it is pure tokenizer choice.
- **It breaks perplexity comparisons.** Perplexity is per *token*. Two models with
  different tokenizers producing different token counts for the same text have
  perplexities on different scales. Comparing them is meaningless. Use **bits per
  byte** when you must compare across tokenizers — same text, same denominator.
  `evaluate.py:task_perplexity` reports both.
- **It is frozen at pretraining.** You cannot change it later without retraining.
  Every downstream decision inherits it.

### One trap this repo defends against explicitly

A vocabulary built only from the pretraining corpus cannot represent anything
outside it. Pretrain on digits, then fine-tune on a prompt template containing
"Q:", and a char-level tokenizer silently drops every letter — no error, just a
broken model. `tokenizer.py:BASE_ALPHABET` forces printable ASCII into the
vocabulary for exactly this reason. Byte-level tokenizers make the problem
impossible by construction, which is one more argument for them.

---

## The pipeline, stage by stage

`data.py:preprocess` runs these in order, and **prints how many documents each one
dropped**. That report is the most important output of the whole module.

### 1. Normalise

Unicode NFKC, strip control characters, collapse whitespace runs. Deterministic and
lossy — do it once, early, and record that you did it.

### 2. Quality filter

Heuristics in the spirit of the C4 and Gopher rules:

- minimum length — a 3-character document teaches nothing
- **repetition ratio** — the fraction of the document taken by its most common
  word. Catches `buy now buy now buy now …`
- **alphanumeric ratio** — catches `!!!!!!!!!!!!!!!` and binary garbage

Real pipelines add: language identification, a small classifier trained to
recognise "reference-quality" text, perplexity filtering under a cheap model, PII
scrubbing, and toxicity filtering. Same shape, more stages.

**Every threshold here is a judgement call you should be able to defend with a
number.** Which brings us to the incident this repo is built around.

### The min_chars incident (a real bug, measured)

With prose defaults (`min_chars=12`) the `arithmetic` corpus loses **36% of its
documents** (measured: 7,239 of 20,000), because `"0 + 3 = 3"` is nine characters.
The survivors are systematically the *large-operand* problems.

A detail worth noticing, because it is how filters actually interact: with
`min_chars=12` in place, the repetition and alphanumeric filters drop **zero**
additional documents — every high-repetition case (`"0 + 0 = 0"`, ratio 0.6) is
already shorter than 12 characters and was removed by the length filter first. Filter
stages are not independent, so "which filter is responsible" is a question about their
*order*, and the per-stage drop report is the only way to answer it.

Downstream, the model never sees small operands, and accuracy on them collapses.
The aggregate accuracy barely moves, so nothing looks wrong.

Two defences, both implemented:

- `FILTER_PRESETS` — per-source filter settings, because prose thresholds destroy
  structured data. Real stacks filter each source separately, *before* mixing.
- `evaluate.py:task_arith_exact` reports accuracy **sliced by operand size**, so a
  bias like this is visible in the eval output rather than inferred six weeks later.

Run `python scripts/demo_data_bias.py` to watch it happen: same model, same steps,
one changed constant, and the slices diverge while the aggregate does not.

### 3. Exact dedup

Hash each document; drop repeats. Catches copy-paste duplication. Duplicated
training data is actively harmful: it wastes budget, encourages memorisation, and
inflates validation scores when the same document lands on both sides of a split.

### 4. Near dedup

Documents that differ by a boilerplate footer are still duplicates. `data.py`
implements **SimHash** over word shingles: similar documents get hashes differing
in few bits, so "near-duplicate" becomes a Hamming distance. Production stacks use
MinHash + LSH banding (better recall at scale); the idea is identical.

The threshold is a precision/recall dial, and here are actual measurements from
this repo's `dirty` corpus (1,385 docs after exact dedup, ~15% boilerplate
near-duplicates):

| max_hamming | near-dupes caught | false positives on clean text |
|---|---|---|
| 3 | 15 (1%) | 0 / 1489 |
| 5 | 68 (5%) | 1 / 1489 |
| **8** (default) | **165 (12%)** | **8 / 1489** |
| 12 | 307 (22%) | 105 / 1489 |
| 16 | 644 (46%) | 613 / 1489 |

Too loose and you delete legitimate distinct documents — on a real corpus that
shows up as a mysteriously small dataset. Also note: SimHash distance is a
*fraction-of-content* signal, so it is unreliable on very short documents
(appending three words to an 8-word document measures Hamming 13; to a 34-word
document, 4). That is a property of the method, not something to tune away.

### 5. Mix

`data.py:mix_corpora` samples from several sources at explicit ratios. The mixture
is a **first-class hyperparameter**: "add 10% code" is a real, measurable
pretraining decision with known effects (code in the mix improves reasoning
benchmarks — reported repeatedly, mechanism still debated).

Sources whose weight exceeds their size get up-sampled with replacement, which is
what real runs do with small high-quality sources, and is also exactly how you
accidentally memorise them.

**A warning about loss comparisons across mixtures.** Measured in this repo: the
same architecture and step count reaches val loss ≈0.94 on pure `arithmetic` and
≈0.28 on a 70/20/10 arithmetic/stories/code mixture. The second model is not
better — the mixture contains highly templated, highly predictable text that drags
the average down. **Loss is only comparable between runs on the identical data
distribution.** Task metrics are what survive a mixture change.

### 6. Split — by document, before packing

Hold out validation *documents*, not token ranges. Split a packed token array by
index and the tail of a training document lands in validation, so your val loss is
optimistic and you will not find out until a benchmark disagrees with it.

### 7. Pack

Concatenate everything into one token stream with EOS between documents, then cut
into blocks of `block_size + 1`. The `+1` is the input/target shift: inputs are
`block[:-1]`, targets `block[1:]`.

Packing wastes no compute on padding. The cost: attention can see across an EOS
boundary into an unrelated document. Most stacks accept this; the rigorous fix is a
block-diagonal attention mask per document.

---

## Contamination — this one is yours

The first question anyone asks about a benchmark number is whether the eval data
was in the training data. The only defensible answer is a measurement, and it will
be your measurement.

`data.py:contamination_report` computes two levels:

- **exact overlap** — the same document in both. Indefensible.
- **13-gram overlap** — an eval document shares a 13-word span with training data.
  13-grams is the convention GPT-3 used and most work since has followed. Some
  overlap is normal for common phrasings; a high rate on a specific task means that
  task's score is inflated.

`decontaminate()` then drops the contaminated eval items and reports how many —
because a benchmark that silently shrank by 40% is a different benchmark.

**Be honest about this repo:** the generated corpora overlap heavily by construction
(both splits are drawn from the same finite template space), so arithmetic eval
items *do* appear in training. That is realistic — it is what happens when benchmark
data leaks into a web crawl — and the correct response is to measure it and say so.

---

## What to do first

```bash
python -m aienh data --corpus dirty --n-docs 4000     # watch every stage fire
python -m aienh data --corpus arithmetic              # note the 0% filter drop
python scripts/demo_data_bias.py                      # break it on purpose
python -m aienh.tokenizer                             # char vs BPE sequence length
```
