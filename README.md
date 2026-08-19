# ai-enhancement — model benchmarking on a shared GPU

Submit a Hugging Face model id; get it evaluated on a fixed lm-eval suite and
placed on a live, interactive leaderboard the whole team can read. Built for one
shared GPU and a handful of trusted users on a tailnet.

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
construction. The dashboard shows standard errors everywhere and z-tests every
pairwise gap before anyone calls it a win.

## Run it (Docker)

```bash
cp .env.example .env          # set BENCH_ROOT, HF_HOME, BIND (your tailscale IP)
docker compose up -d --build
# open http://<host>:8899/
```

Details, knobs and troubleshooting: [`SERVICE.md`](SERVICE.md). Manual
(venv/nohup) mode is in there too.

## Use it

- **Browser:** open the dashboard → *Submit & Queue* tab → paste an `org/model`
  id. `quick` = minutes (iteration); `full` = the comparable number. Progress,
  errors and logs are on the same tab; results land on the leaderboard
  automatically.
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
  make_ppl_task.py    any corpus -> pinned perplexity task (records a sha256)
clients/
  bench_client.py     stdlib-only API client + CLI
Dockerfile / docker-compose.yml / .env.example
SERVICE.md            operate it     API.md  integrate it     BENCHMARK-RUN.md  run it by hand
```

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
