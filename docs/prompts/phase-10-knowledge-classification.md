# Brief for Claude Code — the 37-topic exam (phase 10)

Do after 8a540d2. Three PRs, in order, each green on its own:

- **10a** — the files, the topic list, the loader, retiring the old five
- **10b** — answers and results tied to the questions they belong to, and
  the daily limit
- **10c** — a board with 36 topics, plus the five findings from the phase-9
  live check

## Decisions (Omar, 2026-09-21)

- Omar added `docs/Knowledge Classification/`: 37 folders, one per topic.
- **All 37 become the exam's topics.** The folder names are the topic names,
  exactly as written.
- 36 folders hold three files each:
  - `<slug>_questions.json`: 100 questions, a bare array;
  - `<slug>_criteria.json`: 20 criteria;
  - `<slug>_criteria.md`: the prose rubric with its 0–4 anchors.
- **Arts is empty.** It's a topic with no questions yet.
- **The current five topics are retired**: medicine & health, law,
  economics, computer science, physics & engineering (500 questions).
  - Their files and judged runs stay, as history.
  - The live exam is the new bank only.
- **Written by: `masein`** on every imported question, as for the current
  banks.

Checked against the files:
- Every item has the same fields as the current banks: `id`, `prompt`,
  `acuity`, `difficulty` 1–5, `domain`, `intent`, `jurisdiction_required`,
  `style`.
- **No new question repeats an old one.** The new Law, Economics and Computer
  Science questions are different questions from the retired ones.
- `topic_task()` turns each of the 37 names into exactly the slug the files
  already use (e.g. `Medicine & Clinical Health` → `medicine_clinical_health`,
  `IT` → `it`), so nothing needs renaming.

**Rules that do not bend**, unchanged:
- the split, and qids that come from content;
- report-half isolation;
- provisional stamps;
- taint;
- files go in byte-for-byte as delivered;
- a new file shape is handled by extending the loader, never by editing the
  file.

---

## 10a — files, topics, loader, retirement

### 1. The files move, verbatim

`git mv`, so every byte and sha stays the same:

| From `docs/Knowledge Classification/<Name>/` | To |
|---|---|
| `<slug>_questions.json` | `eval_tasks/fr/banks/<slug>_v1.json` |
| `<slug>_criteria.json` | `eval_tasks/fr/rubrics/<slug>.criteria.json` |
| `<slug>_criteria.md` | `eval_tasks/fr/rubrics/<slug>.md` |

**The retired five move first**, because three of their names collide with
new files (`law.*`, `economics.*`, `computer_science.*` in `rubrics/`):

- their banks, `eval_tasks/fr/{medicine_v2,law_v2,economics_v1,computer_science_v1,physics_engineering_v1}.json`,
  go to `eval_tasks/fr/retired/`;
- their five rubric pairs go to `eval_tasks/fr/retired/rubrics/`;
- `retired/` is outside every path the service reads rubrics or banks from.

Also:
- The now-empty `docs/Knowledge Classification/` is removed (Arts included;
  it held nothing). Future files for Arts arrive through the page's import
  and rubric panels.
- `service/startup.py`'s list and `tests/test_image_contents.py` must cover
  all 108 new files.
- Update `DEMO.md` and any fixture that points at a moved file.

### 2. `scripts/categories.yaml`: 37 topics, MMLU remapped

- The 37 folder names replace the 15 categories.
- MMLU's 57 subjects are remapped as below: my proposal, which Omar can
  change.
- `tests/test_categories.py` still enforces every subject exactly once.
- 13 of the topics have no MMLU counterpart. That's fine: the MMLU caution
  line simply doesn't appear for them.
- **`General & Multidisciplinary` replaces `other`** as the fallback for any
  unmapped subject. `other` stops being a topic.

| Topic | MMLU subjects |
|---|---|
| AI & Machine Learning | machine_learning |
| Anthropology & Human Geography | high_school_geography |
| Biology & Life Sciences | college_biology, high_school_biology |
| Business & Management | management, marketing |
| Chemistry & Materials Science | college_chemistry, high_school_chemistry |
| Computer Science | college_computer_science, high_school_computer_science |
| Economics | econometrics, high_school_macroeconomics, high_school_microeconomics |
| Engineering | electrical_engineering |
| Ethics & Religion | business_ethics, moral_disputes, moral_scenarios, world_religions |
| Finance & Accounting | professional_accounting |
| General & Multidisciplinary | global_facts, miscellaneous |
| Government & Public Policy | high_school_government_and_politics |
| History & Archaeology | high_school_european_history, high_school_us_history, high_school_world_history, prehistory |
| Law | international_law, jurisprudence, professional_law |
| Mathematics & Statistics | abstract_algebra, college_mathematics, elementary_mathematics, high_school_mathematics, high_school_statistics |
| Media & Communication | public_relations |
| Medicine & Clinical Health | anatomy, clinical_knowledge, college_medicine, medical_genetics, professional_medicine, virology |
| Philosophy | philosophy, formal_logic, logical_fallacies |
| Physics & Astronomy | astronomy, college_physics, conceptual_physics, high_school_physics |
| Political Science & International Relations | security_studies, us_foreign_policy |
| Psychology & Cognitive Sciences | high_school_psychology, professional_psychology |
| Public Health & Wellness | human_aging, human_sexuality, nutrition |
| Sociology | sociology |
| Systems & Cybersecurity | computer_security |

**No MMLU subjects:** Agriculture, Architecture & Built Environment, Arts,
Data & Information Science, Design, Earth & Environmental Sciences,
Education, Food & Veterinary Sciences, IT, Language & Literature,
Manufacturing & Applied Sciences, Software Engineering & Programming,
Technology.

The MMLU control set is built at `CONTROL_PER_CATEGORY` = 10 per category
that has subjects. It grows from 150 to about 240 items. Say so in the build
output.

### 3. The loader reads two more criteria layouts

The 36 files use four ways of stating the critical flag:

| Key | Shape | Files |
|---|---|---|
| `critical_error_flag` | one object | 30 (already read) |
| `critical_error` | one object | Architecture & Built Environment, Language & Literature, Mathematics & Statistics, Political Science & International Relations |
| `critical_flags` | a list of objects | Biology & Life Sciences, Public Health & Wellness |

- All of these normalise into the one internal `flags[]`.
- Top-level `benchmark`, `task`, `score_scale` and `score_range` are
  informational: ignored by the loader, but part of the file's sha as now.
- Add both layouts to `docs/CRITERIA-SCHEMA.md`.
- Test: all 36 new files and the 5 retired ones load and normalise to the
  same internal shape.

**Acuity** now also takes `critical` and `high`:
- `critical` and `urgent`: Engineering, Government & Public Policy, IT,
  Manufacturing & Applied Sciences;
- `high`: Architecture & Built Environment, Law, Systems & Cybersecurity.

The natural order for breakdown tables becomes emergency → critical → urgent
→ high → moderate → mild → routine. The constant-field rule from 8g still
skips `acuity` where it is `routine` on every item.

### 4. Import all 36, retire the old five

Two `exam_build.py` commands, used in the deploy steps.

**`import-dir <dir> --approver <name>`** imports every `<slug>_v1.json` in the
folder into the topic whose slug matches:
- source = the file stem;
- written by = `--approver`;
- it prints one line per topic, then a total;
- a file whose slug matches no topic is refused, by name;
- it's idempotent, like `import`.

**`retire --topic <name> --reason <text>`** marks every row of that topic's
bank as retired (`retired_at`, `retired_reason`) **without deleting or
re-splitting anything**:
- retired rows are left out of the built tasks, the Loop board, the counts
  and imports;
- **matching is by the stored topic string, not the slug.** The retired
  `law` and the new `Law` share the slug `law`, so a slug match would retire
  both, or merge 100 old questions into the new topic;
- it refuses a topic name that isn't in the bank, and prints the rows
  changed.

Tests:
- `import-dir` on the real folder: 36 topics × 100;
- retiring `law` leaves `Law` untouched, and the reverse;
- after both, a build contains exactly the 36 new banks plus the MMLU
  control.

---

## 10b — results belong to the questions they were given on

### 5. Resume must compare the questions, not the task name

On 2026-09-21, #47 ran SmolLM2-360M-Instruct, judged, on economics: **0 GPU
seconds**. `runner._task_done()` saw `results/full/<model>/exam_economics_0shot/`
and skipped the task. The judge then re-graded the answers saved by #46.

That's the right economy for an unchanged exam. After 10a it's a silent
error: `exam_law`, `exam_economics` and `exam_computer_science` keep their
names but hold **different questions**, so every model's retired answers
would be reused and graded as if they answered the new ones.

- **Fingerprint the question set.**
  - `exam_build.build` writes each exam task's `bank_sha256`: sha256 over
    the task's sorted qids, which are themselves content hashes. It goes
    into the task's yaml metadata, or a sidecar the runner reads.
  - The runner writes the same value beside the results
    (`<task>_<n>shot/bank.sha256`).
  - `_task_done()` treats an exam task as done only when the fingerprints
    match.
  - Harness tasks (mmlu, hellaswag…) are unchanged.
- **The row says what happened.** A judged run that reuses answers says so:
  "answers reused from #46 (same questions) · re-graded". A person reading
  the queue should not have to infer it from "GPU —".
- **`judge.json` records `bank_sha256` per task.** A model's topic result
  counts only when its fingerprint equals the current task's. That applies
  to the Loop board, the answers, the propose gate, averages and the
  Leaderboard's judged columns.
  - Other results are **history**: shown on the model page under **Earlier
    exams (retired question sets)** with their date, and nowhere else.
  - Entries without a fingerprint (everything judged before this PR) are
    history.
  - `/api/answers` applies the same rule. It must never show a retired
    answer against a new topic's question.

Tests:
- run → rebuild the same bank → reuse (0 GPU), and the row says so;
- run → add one question → rebuild → re-answer;
- a pre-10b `judge.json` entry appears as history, not as the current Law
  result;
- `/api/answers` for the new Law returns nothing for a model that only sat
  the retired law.

### 6. The daily limit must not block the loop

- `LLM_DAILY_ITEM_CAP` (default 2,000) is compared with
  `db.llm_items_today()`, which sums **every** batch.
  `judge.py:1403` records judge batches in the same table.
- One full judged run is 3,600 answers + about 240 control + 30 canary, or
  about 3,870 items. After it, every Propose and Generate that day is
  refused with "would pass LLM_DAILY_ITEM_CAP".
- The limit exists for paid providers. On `local` it limits nothing.

The fix:
- The limit is per provider.
- `local` identities are unlimited unless `LOCAL_DAILY_ITEM_CAP` is set.
- A paid provider keeps `LLM_DAILY_ITEM_CAP`, and its judge batches count
  against **the judge's** provider.
- The Review tab shows usage per provider.

Test: after a 3,870-item local judge batch, Propose on the same day is
accepted.

---

## 10c — a board with 36 topics, and the phase-9 findings

### 7. 36 topics on every screen that lists topics

- **Loop board.**
  - Default sort: weakest first, for the model chosen in "Results for".
  - A **search box**.
  - The shared pager, at 25 per page.
  - Arts folds into the "topics without questions" line, as in 9b-7.
- **Sit the exam and Queue → judged.**
  - The topic checkboxes get a filter box, **All / None**, and a count
    ("12 of 36 topics · about 1,200 answers · about 25 min").
  - The time estimate comes from the last judged runs, not a constant.
- **Model page → Judged.** The 9c-7b topic switcher becomes a searchable
  select. 36 segments don't fit.
- **Leaderboard.**
  - The 36 judged-topic columns and the ~24 "MMLU by category" columns start
    hidden, behind the Columns menu.
  - Add one **Judged average** column (report half, over current-bank topics
    only), shown by default.
- **Exam tab.**
  - The rubrics table gets the shared pager and a search box.
  - Its bank column says "100 — 5x report / 4x diagnose" per topic, as now.

### 8. What the phase-9 live check found

1. **A judged run spends GPU before checking the judge.** #50 answered law
   (37 GPU-seconds) and then failed: nothing was answering on port 8000,
   because vLLM had crashed three hours earlier.
   - The service checks the judge (`GET /v1/models`, 2-second timeout,
     result cached 30 s) when a judged submission is created.
   - If the judge is down: refuse, in plain words.
   - The Loop board and the topic page show a **judge offline** chip, and
     *Queue this run* is disabled with the reason.
   - The header's live dot also covers the judge.
2. **A failed row's message is written for developers** (container
   hostnames, an ssh tunnel).
   - Lead with one plain line: "The grading model isn't running. Start it,
     then Retry grading — the answers are kept."
   - The raw text goes behind **details**.
   - When the failure was the judge, the row's button is **Retry grading**:
     with 10b's resume it re-grades the saved answers and uses 0 GPU.
3. **A toast with a link disappears after 4 s.** Make it 8 s, paused while
   hovered or focused.
4. **Coming back to Queue keeps page 2**, so a new row, even a failure, is
   out of sight. Reset to page 1 on entering the tab, and when a row you
   queued changes status. Highlight that row.
5. **The queue's pager is rebuilt on every refresh while a run is live.** A
   click can land on a node that has just been replaced. Build the pager
   once and update it in place, as the tab bar is (#21).

---

## Definition of done

**On the live server, after the deploy steps below:**
1. Exam shows 37 topics: 36 with 100 questions (split about 50/50),
   written by masein, and Arts with none.
2. The retired five appear nowhere on the Loop board. Their judged runs
   appear on SmolLM2-360M-Instruct's page under **Earlier exams**.
3. SmolLM2-360M-Instruct, judged on **Law**, re-answers (GPU > 0). It does
   not reuse the retired law answers.
4. A judged run of all 36 topics completes, and Propose still works the same
   day.
5. With vLLM stopped, *Queue this run* is disabled with the reason, and the
   API refuses a judged submission in plain words.

**Deploy steps**, for Omar, after this merges:

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
for t in "medicine & health" law economics "computer science" "physics & engineering"; do
  sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam retire --topic "$t" --reason "replaced by the 37-topic exam (2026-09-21)"
done
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam import-dir eval_tasks/fr/banks --approver masein
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam build results/full
sudo docker compose exec -T bench python3 scripts/exam_build.py --root /home/masein/benchmarks/exam summary
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.
