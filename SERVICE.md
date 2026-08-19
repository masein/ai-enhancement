# The benchmark service — submit a model id, get it on the leaderboard

The self-serve version of `BENCHMARK-RUN.md`: teammates on the tailnet open one
URL, submit a Hugging Face model id, and watch it move through the queue onto the
live dashboard. Same lm-eval pipeline, same results tree, same report — the
service is a queue and a live view wrapped around what already works.

```
friend's browser (tailnet)
        │  submit org/model
        ▼
FastAPI  ──► SQLite queue ──► worker (ONE at a time)
   │                             │  preflight (HF metadata: exists? gated? size?
   │                             │  vocab→batch→VRAM need? trust_remote_code? kind?)
   │                             │  wait for results/.run.lock  ← shared with the CLI script
   │                             │  wait for free VRAM
   │                             ▼
   │                          lm_eval per task → results/full/<model>/<task>_<n>shot/
   ▼
GET /  = the dashboard, live (fetches /api/results, polls the queue)
```

Guardrails it enforces so a shared GPU stays shared: one run at a time (same lock
as the CLI — service and manual runs can never race); free-VRAM wait before every
model; per-model batch size from the vocab logits law; a parameter cap
(`MAX_PARAMS_B`, default 4B); and it **refuses models that require
`trust_remote_code`** — nobody's repo executes code on this box.

## Run it — Docker (preferred)

Requires docker + the NVIDIA container toolkit. Prove GPU passthrough once:

```bash
docker run --rm --gpus all pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime nvidia-smi
```

Then:

```bash
cd ~/benchmarks/aienh
cp .env.example .env && sed -i "s|^BIND=.*|BIND=$(tailscale ip -4)|" .env
nano .env                     # confirm BENCH_ROOT and HF_HOME match your paths
docker compose up -d --build
docker compose logs -f        # Ctrl-C stops the log view, not the service
```

Update after a `git pull`: `docker compose up -d --build` again. Stop:
`docker compose down` (an in-flight run is killed; it re-queues on next start and
per-task resume repeats only the interrupted task).

Three container decisions are load-bearing — the comments in
`docker-compose.yml` explain each, but in short: `pid: "host"` keeps the
run-lock's liveness check truthful across container/host (without it a
containerized service would misread a live manual CLI run as a stale lock and
race it); `BENCH_ROOT` and `HF_HOME` are mounted **path-identical** (the sqlite
DB, your HF auth/licenses, and `eval_tasks/*.yaml` absolute corpus paths all keep
working); and the port publishes on the Tailscale IP only.

## Run it — manual (no Docker)

```bash
cd ~/benchmarks
source .venv/bin/activate
pip install -r aienh/requirements.txt
git -C aienh pull

nohup bash aienh/service/run.sh > service.log 2>&1 &
tail -f service.log        # Ctrl-C stops the tail, not the service
```

**You should see** `benchmark service on http://<tailscale-ip>:8899`. Check it:

```bash
curl -s "http://$(tailscale ip -4):8899/healthz"
```

Friends open `http://teraformer-5090-3:8899/` (tailnet hostname works from any
device on the tailnet). To stop the service: `pkill -f "uvicorn service.app"` —
a run in flight is killed with it; on restart the interrupted submission is
re-queued automatically and per-task resume repeats only the interrupted task.

## Knobs (environment variables, all optional)

| var | default | meaning |
|---|---|---|
| `PORT` / `BIND` | 8899 / tailscale IP | where to listen — keep it off 0.0.0.0 |
| `TITLE` | Team model benchmark | dashboard heading |
| `MAX_PARAMS_B` | 4 | reject models bigger than this (bf16 weights ≈ 2 GB/B) |
| `MAX_JOB_GB` | 10 | VRAM budget a job may plan for (drives batch choice) |
| `SUBMIT_TOKEN` | *(unset)* | if set, submits need it — friends use `http://…/?token=<value>` |
| `TASK_TIMEOUT_S` | 10800 | kill a single task after this |
| `BENCH_ROOT` | cwd | the directory holding results/, eval_tasks/, logs/ |

## How a submission behaves

`queued → preflight → waiting_lock/waiting_gpu (if needed) → running (k/n · task)
→ done | failed`. Failures carry a plain-language reason (gated repo, OOM, size
cap, custom-code refusal) plus a `log` link with the raw output. Canceling is
only possible while `queued` — a running job finishes its current task.

Two properties worth telling teammates: submitting a model that is already on
the leaderboard costs nothing (per-task resume sees the results and finishes in
seconds), and a `quick` run (hellaswag + arc_easy + perplexity slices) later
upgrades to `full` by running only the missing tasks.

## API (everything the page does, scriptable)

```bash
B="http://$(tailscale ip -4):8899"
curl -s $B/api/results | jq '.models | length'
curl -s -X POST $B/api/submissions -H 'Content-Type: application/json' \
     -d '{"hf_id":"EleutherAI/pythia-31m","suite":"quick","submitter":"omar"}'
curl -s $B/api/submissions | jq '.[0]'
curl -s $B/api/runs/1/log
curl -s -X POST $B/api/submissions/2/cancel
```

## Coexistence with manual runs

`bash aienh/scripts/run_benchmarks.sh full` still works and still owns the same
lock: whichever starts first runs; the other waits (the service shows
`waiting_lock`, the CLI prints `REFUSING TO START`). Both write the same tree, so
the dashboard shows the union either way. The frozen single-file report
(`report_lm_eval.py results/full -o …`) also still works and is the right thing
to email outside the tailnet — it's the same page with the data baked in.

## Troubleshooting

**`no tailscale IP found`** — `tailscale ip -4` printed nothing; set `BIND=<ip>`
explicitly.
**Port already in use** — the old `python -m http.server 8899` or a previous
service instance is still up: `pkill -f http.server; pkill -f "uvicorn service.app"`.
**Submission stuck in `waiting_lock`** — a manual CLI run holds the GPU;
`cat results/.run.lock/pid` and decide whose job it is.
**Stuck in `waiting_gpu`** — the card genuinely lacks the head-room the model
needs (`need_gb` on the queue row); it starts the moment memory frees, and gives
up after `GPU_WAIT_MAX_S` with a resubmit-later message.
**Everything fails with the same error** — read one `log` link; a
`ModuleNotFoundError` means the venv changed under the service.

## Backup

Two things hold all state: the `results/` tree and `service.sqlite3`. Copy those,
and a fresh checkout of this repo reproduces the rest.
