# ai-enhancement — model benchmarking on a shared GPU

Submit a Hugging Face model id — or upload a checkpoint directly, no HF account
needed — and get it evaluated on a fixed lm-eval suite and placed on a live,
interactive leaderboard the whole team can read. Training code can also stream
its metrics here (wandb-style `run.log()`), giving live loss curves, run
comparison with config diffs, and benchmark scores joined to training steps on
Improve ▸ Training runs. Built for one shared GPU and a handful of trusted users on a
tailnet.

```
browser / training script ──► FastAPI + SQLite queue ──► one worker at a time
                                                            │  preflight (Hub metadata:
                                                            │  gated? size? vocab→batch→VRAM?
                                                            │  refuses trust_remote_code)
                                                            ▼
                                             lm_eval per (model, task) ──► results tree
                                                            │
        live dashboard  ◄── /api/results ◄─────────────────┘
```

## The suite

Eight benchmarks at fixed few-shot counts — MMLU, HellaSwag, ARC-Challenge,
ARC-Easy, Winogrande, PIQA, TruthfulQA (mc2), GSM8K — plus corpus perplexity
(bits per byte) on pinned text slices you create with
`scripts/make_ppl_task.py`. Same seed, dtype, and harness version (lm_eval
0.4.12, pinned) for every run: scores are comparable to each other by
construction. The dashboard shows standard errors everywhere, counts which
pairwise gaps clear a z-test before anyone calls them wins, derives
cross-entropy loss (nats/byte) from the perplexity slices so checkpoints can be
read against training curves, and gives the Runs tab a query bar over every raw
metric.

## Run it (Docker)

```bash
cp .env.example .env          # set BENCH_ROOT, HF_HOME, BIND (your tailscale IP)
docker compose up -d --build
# open http://<host>:8899/
```

Details, knobs and troubleshooting: [`SERVICE.md`](SERVICE.md). Manual
(venv/nohup) mode is in there too.

## Use it

- **Browser:** open the dashboard → **Test a model** (top right) → paste an
  `org/model` id. `quick` = minutes (iteration); `full` = the comparable
  number. Progress is on the run counter beside it (**● n running**), errors
  and logs on **All runs**; results land on **Models** automatically.
- **From code / training loops:** the JSON API is the whole product —
  [`API.md`](API.md) documents every endpoint and the checkpoint→submit→collect
  pattern. [`clients/bench_client.py`](clients/bench_client.py) is a
  zero-dependency client you can vendor into any repo.
- **Manual CLI runs:** [`BENCHMARK-RUN.md`](BENCHMARK-RUN.md) is the operator
  runbook — same pipeline without the service, plus what the numbers should look
  like before you see them (chance levels, expected curves, known artifacts).

## Layout

```
service/              FastAPI app, SQLite queue, worker, HF preflight
scripts/
  run_benchmarks.sh   the same pipeline as a standalone CLI (lockfile, resume)
  report_lm_eval.py   results tree -> interactive dashboard (live or frozen single file)
  diagnose.py         per-item diagnosis from --log_samples (held-out split; DIAGNOSE.md)
  categories.yaml     the 37 exam topics; MMLU's 57 subjects mapped onto 24 of them (+ categories.py)
service/
  llm.py, proposals.py, contamination.py, llm_poller.py
                      find the gap: LLM proposals from the diagnosis half, human review,
                      generation from the approved spec only, 13-gram gate, provenance, taint
  make_ppl_task.py    any corpus -> pinned perplexity task (records a sha256)
eval_tasks/
  mmlu_perm/          the permutation control: MMLU with the options rotated (suite=control)
  fr/                 the exam: banks/ (37 topics x 100), rubrics/, retired/ (the old five),
                      the migrated skill items, AUTHORING.md (suite=judged)
scripts/
  demo_loop.py        the whole loop in one narrated command, against the local model (DEMO.md)
  exam_build.py       draft (LLM) -> curate (human, Benchmarks ▸ Knowledge exam) -> split by qid -> build harness tasks
  judge.py            the local, pinned judge -> judge.json beside each model's results
  judge_calibrate.py  human vs judge: export a CSV, import it, Cohen's kappa gates the suite
clients/
  bench_client.py     stdlib-only API client + CLI
examples/
  train_and_benchmark.py   real mini training run -> push checkpoints -> benchmark each
Dockerfile / docker-compose.yml / .env.example
SERVICE.md            operate it     API.md  integrate it     BENCHMARK-RUN.md  run it by hand
DEMO.md               watch the whole loop turn once, and what a green run does not prove
```

## Tests and CI

Nothing in the suite touches the GPU or the Hub. It runs against a synthetic
`results/full` tree (`tests/fixtures/make_fixture.py`) in the harness's real
0.4.12 on-disk shape, with one model per failure mode `scripts/diagnose.py`
detects — so every later feature is tested against the same eight models,
and a test can say "this model, this finding" and mean it.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && playwright install chromium
pytest -q            # under a minute; the dashboard smoke drives Chromium
ruff check .
```

`scripts/check.sh` is the check before every merge: ruff, a compile pass over
`scripts/ service/ clients/`, the unit and API tests, then the Playwright
browser suite, with tests marked `gpu` or `network` deselected. It runs them in
a container (`python:3.12-slim` with the dev requirements and Chromium,
`scripts/check.Dockerfile`), so it needs Docker and never the host's Python. It
ends with one line, `lint ok · unit 516/516 · browser 398/398 · 19 min`, which
goes in the PR under **Local check** (HANDOFF.md § 5b).
`.github/workflows/ci.yml` runs the same steps, and the Docker image build, but
only when started by hand: the Actions minutes ran out on 2026-09-24.

## Guarantees worth knowing

One evaluation at a time (an atomic lock shared with the CLI script — a service
run and a manual run can never race); free-VRAM checks before every model with
batch sizes derived from vocabulary size; per-(model, task) resume, so re-submits
are free and interrupted runs lose at most one task; submitted repos are never
executed as code; and every run's provenance (dtype, batch, seed, template,
versions, wall-clock) is recorded and displayed, because a score without its
settings is hearsay.

---

*History note: this repo started as a learning codebase (toy training pipeline +
study docs). That material was removed when the service became the product — it
lives in git history before the `service-only` restructure commit.*
