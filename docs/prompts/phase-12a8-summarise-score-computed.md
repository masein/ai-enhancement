# Brief for Claude Code — Summarise: the judge reports findings, the code computes the score (12a.8)

One small PR, before the hidden-half PRs. It merges on a **green `ci.yml` run on the mirror** for its rebased head. There are no model runs: this re-marks stored answers only.

## Why

After 12a.7 and #111, on Summarise's practice half for the 5 served setups, **18 of 145 answers fail**, down from 67. Reading all 18:

- **5 are right to fail:**
  - 3 ran out of room;
  - 1 is missing the account balance (bank SMS);
  - 1 is "2x faster" for "half the time". The number rule rejects it, which is acceptable.
- **10 are "several versions instead of one"** (school notice ×5, leave request ×4, email ×1). The rubric makes that **−1** (3 of 4, a pass), but the judge (gemma) gives **0 to 2**. One leave request with every fact right got **0 of 4**. The judge doesn't follow the rubric's point values.
- **1 is a judge mistake:** school run plan, lookahead + MTP. The judge says the answer "misses the fact that the mother is coming", but the answer includes "Your mom arrives" at 7:00 PM.

So the small judge is reliable at spotting *what* is wrong, but not at *adding up points*. Let the judge report findings, and compute the score in code.

## 1. The judge returns findings only

The judge replies with JSON, and **no score**:

```json
{"missing_facts":      ["<each quoted from the reference's fact list>"],
 "invented_or_wrong":  ["<each quoting the answer>"],
 "several_versions":   true,
 "length_ok":          "yes" | "no" | "not asked",
 "note":               "<one short sentence>"}
```

- **Missing facts** come only from the reference's fact list, quoted.
- **Invented or wrong** items must quote the answer.
- **Style is never a finding:** a lead-in, a closing offer, headings, bullets, bold or emoji.
- Keep the two worked examples from 12a.7 in the prompt, rewritten as findings:
  - a bulleted summary with a lead-in and a closing offer, and every fact → no findings;
  - the same with one fact missing → one missing fact.

## 2. The code computes the score

```
score = 4
      − min(2, number of missing facts)
      − 2 if any invented or wrong
      − 1 if several versions
      − 1 if a length was asked and not met
floored at 0; passing at 3
```

## 3. The code checks the judge's claims

- **A claimed missing fact the answer contains is dropped.** Take that fact's key words (numbers, names, times and nouns from the reference's fact) and look for them in the answer, normalised (case, 7:00 PM = 7 pm = 7, mum = mom = mother). If they're all present, drop the claim and note it in the verdict.
- **A claimed invented or wrong item that doesn't quote the answer verbatim is dropped,** and noted.

## 4. What's shown

- The reason is built from the findings, for example "missing: the balance · several versions (−1)", or "all key facts, one version".
- Dropped claims show greyed: "the judge said it missed 'mum arrives at 7'; the answer has it".
- The verdict keeps the judge's raw JSON, for the audit.

## 5. Re-mark

Re-mark every stored Summarise answer, both halves, with no model runs. Paste `--compare` in the PR.

## Tests (the exact cases)

- A three-option leave request with every fact → **3, pass**, with "several versions (−1)".
- The school run plan answer that includes "Your mom arrives at 7:00 PM", where the judge claims it's missing → the claim is dropped, and the answer passes.
- A missing balance (the bank SMS) → **3**, with "missing: the balance".
- An invented number → −2.
- Style only (a lead-in and a closing offer) → **4**.
- The judge's JSON is malformed → retried once, then "the judge's reply couldn't be read", neither pass nor fail, and counted as waiting.
