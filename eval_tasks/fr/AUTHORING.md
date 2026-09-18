# Free-response tasks — how to write an item

Four suites, one per category, answered open-ended and graded by a local,
pinned judge against a written rubric (`rubrics/<category>.md`). These are
for models at 1B+ and mainly instruct-tuned; below that, multiple choice is
the only thing that extracts signal, and this suite will read near zero.
That is the honest result, not a bug — see `docs/design-diagnose-and-generate.md` §5.

**The items are human-authored.** The ten per category shipped here are
marked `"seed": true` and exist so the plumbing can be tested end to end. The
target is about fifty per category, chosen by a person who has read the
rubric, because the questions must be chosen well and that is a human job.
Do not generate them with an LLM and do not pad the files to fifty. A judged
number over ten seed items is a smoke test, and the dashboard says so.

## The shape of a good item

One JSON object per line in `fr_<category>.jsonl`:

```json
{"id": "fr_reasoning-0011", "category": "reasoning",
 "prompt": "<what the model is asked — complete, self-contained, one ask>",
 "reference": "<what a full-marks answer must contain — facts, steps, or the constraints met>",
 "notes": "<for authors and calibrators: why this item, what a 2 looks like>"}
```

- **One ask.** A prompt that asks two things gets answers that half-comply
  and the judge cannot score them consistently.
- **A reference that names the substance, not the wording.** The judge grades
  against the rubric with the reference as the answer key; it must be able to
  recognise a correct answer phrased differently. Write "names the three
  primary colours of light: red, green, blue" rather than a model sentence.
- **Length-neutral.** Every rubric puts length in writing: a complete short
  answer scores fully, padding never scores higher. Do not write prompts that
  reward volume ("list everything you know about…").
- **Unambiguous ground truth**, or a clearly bounded one. For `cultural`, say
  whose culture and when; for `factual_accuracy`, pick facts that do not move.
- **Not on the internet verbatim.** Write it yourself. An item lifted from a
  known set is a contamination risk and gets caught by the same 13-gram gate
  the generated datasets go through.
- **Answerable in under 120 words.** The judge's prompt budget and the length
  clause both assume this.

## Per category

- `instruction_following` — the prompt carries explicit, checkable
  constraints (count, format, exclusions, order). The reference lists the
  constraints; the rubric scores how many were met and whether the content
  is sensible.
- `factual_accuracy` — a question with a stable, verifiable answer; the
  reference is the answer plus any acceptable variants.
- `reasoning` — a short multi-step problem where the steps can be checked;
  the reference gives the answer AND the chain, so a right answer with a wrong
  chain scores a 2, not a 4.
- `cultural` — knowledge of conventions, practices or references bounded by
  place and time; the reference says what a well-informed local would say.

## The control set

`fr_control_mmlu.jsonl` is **not authored**. `scripts/fr_build.py` builds it
from the diagnosis half of MMLU on disk: the question only, no options,
graded against the gold option's text. About ten items per category,
stratified. It exists to answer one question per model: of the items it got
wrong as multiple choice, how many did it get right when asked openly —
"knew it, couldn't pick it" versus "didn't know it either way". Every item's
source hash splits to `diagnose` and a test enforces it.

## Calibration before anything counts

Nothing judged enters any average until a person has graded a hundred
sampled answers and Cohen's κ against the judge clears 0.60
(`scripts/judge_calibrate.py export … / import …`). Below that the suite is
shown as preliminary and never ranked. When you add items, plan the
calibration pass with them.
