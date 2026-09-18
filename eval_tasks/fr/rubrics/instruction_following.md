# Rubric — instruction following (version 1)

Score the single answer against the constraints listed in the reference.
Judge compliance first, sense second. Do not reward length: a complete answer
that meets every constraint in one line is a 4; an answer that meets them all
and then keeps going, repeats itself, or adds unrequested material loses a
point.

- **4** — every constraint met exactly; the content is sensible; nothing extra.
- **3** — every constraint met, with one small slip (an extra word over a
  limit, one formatting deviation) or noticeable padding.
- **2** — most constraints met, one clearly violated (wrong count, wrong
  format, a forbidden element present), content otherwise sensible.
- **1** — the answer engages with the task but ignores most constraints, or
  meets the constraints with content that does not make sense.
- **0** — no answer, off-topic, a refusal without cause, or text unrelated to
  the prompt.

Length: the reference states the expected size when it matters. Over the
stated limit costs a point at 4 and 3. Under it is fine if complete.
