# The criteria file, as this platform reads it

*Internal. Decided 2026-09-20 (Omar): we do not ask the author to change how
he writes these files. Three deliveries have arrived in three shapes and all
three grade; whatever arrives next, the loader learns it.*

A criteria file sits beside a topic's prose rubric —
`eval_tasks/fr/rubrics/<slug>.criteria.json` — and turns the judge's 0–4 into
a fold of per-criterion scores. `scripts/judge.py::normalise_criteria` reads
whatever the file says and produces **one internal shape**, which is what the
prompt builder, the fold, `judge.json`, the page and the demo all see:

```json
{"criteria": [{"id": "relevance",            // the slug the model repeats back
               "name": "Relevance",          // what a person reads
               "definition": "…",
               "weight": 1.0,
               "conditional": false,
               "applies_when": "…"}],        // optional; conditional criteria may return null
 "flags":    [{"id": "critical_technical_error",
               "condition": "…",
               "effect": "zero_score",       // or cap_at_N_of_4, or set_at_N_of_4
               "examples": ["…"],            // his calibration: what counts
               "not_critical": ["…"]}],      //                  what does not
 "evaluation_principles": ["…"]}             // optional, topic-wide, goes in the prompt
```

The whole file's sha256 is recorded in every `judge.json`, so the *delivered*
bytes are the provenance — normalising changes what we read, never what we
record. Anything the file carries that this table does not name is passed
through untouched.

## Every input variant seen so far, and what it maps onto

| In the file | Seen in | Read as |
|---|---|---|
| `criteria[].id` is a slug, `name` is a label | medicine, law, computer science, economics | `id` = the slug, `name` = the label |
| `criteria[].id` is a row number (`1`, `2`, …) and `name` is the slug | physics & engineering | `id` = `name`, label derived from it ("Relevance") |
| `criteria[].name` absent | medicine, law | label derived from the id |
| `flags` as a list | medicine, law | the list, unchanged |
| `critical_flag` as a single object | computer science, physics & engineering | a one-item list |
| `critical_error_flag` as a single object | economics | a one-item list |
| `effect: "zero_score"` / `"score=0"` | law, medicine / the three new topics | `zero_score` — the score becomes 0 |
| `effect: "cap_at_1_of_4"` / `"cap=1"` | law | `cap_at_1_of_4` — `min(score, 1)` |
| `effect: "score=2"` | (not yet) | `set_at_2_of_4` — the score becomes 2 |
| `examples` on a flag | the three new topics | prompt: "counts as `<flag>`: …" |
| `not_critical` / `do_not_classify_as_critical` | physics & engineering / economics | prompt: "does NOT count as `<flag>`: …" |
| `evaluation_principles` at the top level | physics & engineering | prompt: a "how this topic is graded" block above the criteria |
| `weights: "equal"`, or a `weight` per criterion, or neither | all | equal unless a criterion says otherwise; `0.05 × 20` is equal |
| `breakdowns` naming metadata fields | (not yet) | those tables instead of the chosen ones |

An effect the judge cannot apply **refuses to load**, naming the file and the
string: a file whose rule cannot be carried out must not quietly grade as if
the flag did nothing. That is the one thing a delivery can get wrong that we
will not paper over — and the fix is a loader that learns the new spelling,
not a file the author has to rewrite.

## Adding the next variant

Extend `normalise_criteria` (and, for a new effect, `normalise_effect` and
`fold`), add the row above, and add the file to the loader test that asserts
every real criteria file in the repo normalises to this shape. The test is
`tests/test_criteria_schema.py::test_every_delivered_file_normalises_to_one_shape`.
