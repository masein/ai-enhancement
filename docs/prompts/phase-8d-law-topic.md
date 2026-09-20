# Brief for Claude Code — the law topic, the author's criteria schema, two generalisations (phase 8d)

Do this after 8c. Supersedes the earlier 8d text in full.

Dr. Hossein has delivered, in one go: **100 law questions**, **100 medicine
questions** (the original 50 unchanged, 50 new, each with a `difficulty`),
and **criteria files for both topics in a schema of his own** — clean,
consistent, and simpler than the one P4b invented. Decision: **his schema is
the platform's criteria-file format from now on.** His files go into the
repo verbatim, so the sha recorded in provenance is the author's file, not
a transcription of it. The medicine work built exactly one flag by name
(`critical_safety_failure`) and one breakdown by name (`by_acuity`); law
needs a list of flags with different effects and breakdowns by whatever
`meta` fields a topic has. Generalise; do not special-case law.

Files in this commit:

| File | What |
|---|---|
| `eval_tasks/fr/medicine_v2.json` | 100 medicine questions. Ids 1–50 are byte-identical prompts to v1 (same qids, same split); 51–100 new; every item has `difficulty` from the author. **Replaces** `medicine_v1.json` — delete v1 (git keeps it) and point `DEMO.md` and any test fixture at v2. |
| `eval_tasks/fr/law_v1.json` | unchanged from the earlier 8d: his 100 law questions plus `difficulty` from his id-range table (1–20 → 1, 21–45 → 2, 46–65 → 3, 66–85 → 4, 86–100 → 5), pending his confirmation |
| `eval_tasks/fr/rubrics/medicine_health.criteria.json` | **the author's file, verbatim** — replaces P4b's draft. 15 criteria, no conditional ones (medication safety is folded into safety and accuracy), one flag |
| `eval_tasks/fr/rubrics/law.criteria.json` | **the author's file, verbatim** — 23 criteria, no conditional ones, two flags |
| `docs/medicine-criteria-v2.md`, `docs/law-criteria-v2.md` | his scoring notes for each, as delivered |
| `eval_tasks/fr/rubrics/medicine_health.md`, `law.md` | the prose rubrics (0–4 anchors and flag rules), aligned to his v2 ids; still DRAFT — he has not reviewed the anchors |

**Read first:** `scripts/judge.py` (the P4b criteria path: loader, prompt,
parse, fold, `judge.json` blocks), both `.criteria.json` files, `scripts/
report_lm_eval.py` (criteria row, CSF text, by-acuity table).

One PR.

---

## 1. The criteria-file schema is the author's

```json
{"topic": "law",
 "weights": "equal",
 "criteria": [{"id": "relevance", "definition": "...", "weight": 1.0, "conditional": false}, ...],
 "flags":    [{"id": "critical_legal_error", "condition": "...", "effect": "zero_score"},
              {"id": "fabricated_authority", "condition": "...", "effect": "cap_at_1_of_4"}]}
```

The loader reads exactly this. Rules:

- `topic` is informational; the file's name decides which task it grades
  (`rubrics/<slug>.criteria.json`, as P4a set up). Do not match on it.
- `weights: "equal"` or absent → every weight is 1; a `weight` on a
  criterion is honoured when present. Both his files say equal and carry
  1.0 on every row; the loader must not require either.
- `conditional: true` means the judge may return `null` for that criterion
  and it is excluded from the fold. Neither file has one today; keep the
  path and its tests from P4b. If a conditional criterion has no
  `applies_when` text, the prompt says "return null when it does not
  apply to this question".
- `effect` is a string: `zero_score` → the folded score becomes 0;
  `cap_at_N_of_4` → `min(score, N)`. Applied after the fold, in file
  order. Unknown effect → refuse to load, name the file and the string.
- `label` is optional; when absent, derive it from the id (`red_flag_coverage`
  → "Red flag coverage") for the page and the demo.
- `breakdowns` is optional; when absent, tabulate every field in this
  list that the topic's items actually carry in `meta`: `acuity`,
  `difficulty`, `intent`. Both topics carry all three.
- The flag list replaces the single `critical_safety_failure` object.
  **Do not keep a compatibility path for the old draft schema** — the only
  file that used it was the medicine draft, and this PR replaces it. One
  loader, one schema, tests against both real files.
- A criterion id may also be a flag id (law: `fabricated_authority` is
  both — the criterion scores any invention at all, the flag fires only
  when it carries the conclusion). Allowed; the reply has separate
  `criteria` and `flags` objects, and the prompt lists them under two
  headings, "CRITERIA (0.0–1.0 each)" and "FLAGS (true/false each)", each
  with its own definition, so the judge cannot conflate them. Test it.

## 2. The judge prompt and reply

The prompt lists each criterion as `id — definition`, each flag as `id —
condition — what it does to the score`, and asks for:

```json
{"criteria": {"relevance": 0.9, ...}, "flags": {"critical_legal_error": false, ...},
 "justification": "one or two sentences about the ANSWER, never quoting the question"}
```

Parse: every criterion present and numeric in [0, 1] (clamp; missing →
parse failure unless conditional); every flag present and boolean (missing
→ parse failure — the flags are the fields that matter most). Unknown keys
counted in `extra_keys` as now.

## 3. `judge.json` and the page

Per item: `criteria`, `flags: {id: bool}`, `score` (folded), `fold:
{"applicable": n, "effects_applied": [ids]}`. Per task: `criteria_mean`,
`criteria_n`, and per flag `{"n", "share", "qids"}` with **diagnose-half
qids only**; breakdown tables per field, values in natural order (acuity
emergency → routine; difficulty 1 → 5; others alphabetical), each cell
`{"n", "mean", "flags": {id: n}}`.

Page: one line per flag in words beside the score; one table per
breakdown field, in the order acuity, difficulty, intent. Put his §5
sentence beside the difficulty table: does the model do well only on basic
questions, or on ambiguous and high-risk ones too. Demo step 5 prints the
same.

## 4. The medicine numbers move, and that is correct

The demo run of 2026-09-20 was graded with P4b's draft criteria file. This
PR replaces it with the author's, so that run is no longer comparable to
the next one — by the existing rule (rubric sha in provenance). Say so in
the PR description. The **fold function** is what must not change: keep
the P4b fold tests (boundaries, null exclusion, effect application) and
add a fixture-level test that the medicine task on the stub judge produces
a `judge.json` whose per-item `score` equals `fold(criteria, flags, spec)`
recomputed in the test.

## 5. Import and run

```
python scripts/exam_build.py import eval_tasks/fr/medicine_v2.json \
    --topic "medicine & health" --approver "Dr. Hossein" --source medicine_v2 --root $BENCH_ROOT/exam
python scripts/exam_build.py import eval_tasks/fr/law_v1.json \
    --topic law --approver "Dr. Hossein" --source medicine_v1 --root $BENCH_ROOT/exam
```

Medicine v2 on a bank that already holds v1: 50 skipped, 50 imported —
the idempotence P4a promised, and a test for it. The reference line gains
`Difficulty: n.` after the domain when the item has one. `DEMO.md` gets
both runs beside each other; both topics now clear `PROPOSE_MIN_N`, so the
demo's shortfall line says so.

## 6. Tests

Loader on both real files; equal weights with and without the key; effect
strings including an unknown one; the criterion/flag id collision; the
prompt's two headings; parse with a missing flag; fold + effects in order;
`judge.json` per-flag blocks carry only diagnose-half qids; breakdown
tables for all three fields on law and medicine; v1 → v2 medicine import
skips 50; the medicine `score` = recomputed fold on the fixture; the
recorded-request-body tests still pass with both banks.

## 7. Still open with the author (HANDOFF §8)

- Confirm the law `difficulty` mapping by id range, or resend law with his
  own values as he did for medicine.
- `jurisdiction_required` per law item — his notes say the criteria use it
  "where present"; it is present nowhere yet.
- Review of the two prose rubrics' 0–4 anchors; the criteria themselves
  are now his and need no sign-off.
