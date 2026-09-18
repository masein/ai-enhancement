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

Friends open `http://100.74.89.105:8899/` (tailnet hostname works from any
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
| `ARTIFACT_MAX_GB` | 8 | per-upload cap for checkpoint artifacts |
| `ARTIFACT_QUOTA_GB` | 150 | total artifact storage before uploads are refused |
| `BENCH_ROOT` | cwd | the directory holding results/, eval_tasks/, logs/ |
| `ALLOW_REMOTE_CODE` | 0 | permit uploads that carry their own modeling code — read the section below first |
| `EVAL_USER` | benchjob | unprivileged account those jobs run as; required when the above is on |
| `REMOTE_CODE_SHAS` | *(unset)* | if set, an allowlist: only these .py hashes may run |
| `CONTROL_TASKS_DIR` | the repo's `eval_tasks/mmlu_perm` | where a `suite=control` run finds the permutation control's task yaml |

## Custom model code (`trust_remote_code`)

A checkpoint with a custom architecture ships `modeling_*.py` and points at it
through `auto_map` in `config.json`. Loading it **executes that Python** — there
is no way to evaluate such a model without running the uploader's code.

Off by default. Two things must line up before any of it runs, and the service
refuses with a specific reason when one is missing:

```bash
ALLOW_REMOTE_CODE=1          # the operator turned it on
EVAL_USER=benchjob           # and there is a non-root account to run it as
```

There is deliberately **no token gate** on top. The tailnet already is the
authentication boundary — the service binds to the Tailscale IP, so anyone who
can upload an artifact was invited onto that network by hand. A shared secret
would separate "on the tailnet" from "on the tailnet and knows a string", which
is not a separation shared secrets keep (they end up in shell history and chat),
while charging every friend a `--token` on every call. If you ever put a device
on the tailnet you don't control, set `SUBMIT_TOKEN` — it gates every submission,
remote code included.

`EVAL_USER` is not optional and is not a formality: the service itself runs as
root (it writes the shared results tree), so without the drop, uploaded code
would execute as root beside everyone's results and the server's Hugging Face
token. If the name doesn't resolve to a real account the job **fails** rather
than running as root. The Docker image creates `benchjob` for this.

### Two setup steps the drop requires

Dropping privileges means the job user needs somewhere to write. The service
handles the first automatically and the second needs one command from you.

**Per-job scratch (automatic).** Torch builds its inductor cache under
`tempfile.gettempdir()` at *import* time, before any model is loaded, so a
`/tmp` the job user cannot write fails instantly with a traceback that names
none of your code. Each remote-code job therefore gets `HOME`, `TMPDIR`,
`XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` pointed at
`$BENCH_ROOT/.jobscratch/<id>/`, created and chowned to `EVAL_USER` and removed
when the job ends. That also keeps those writes on the mounted volume rather
than the container's own filesystem.

**The Hugging Face cache (one command).** `datasets` takes lock files inside
the cache while loading a task, and that cache is root-owned because the
service created it. Give the job user write access, while keeping the token
unreadable:

```bash
sudo chmod -R a+rwX "$HF_HOME"          # the cache is shared, not secret
sudo chmod 600 "$HF_HOME/token"         # the token is
```

A remote-code job runs a canary first — it imports the stack and write-tests
the cache — and reports an environment fault as a *service* error rather than
letting the submitter see four identical tracebacks blamed on their model.

**Watch the disk.** These jobs are also the first thing to break when the box
fills up: ext4 keeps 5% of blocks in reserve for root, so at ~100% usage the
root-run stock jobs keep working while every dropped-privilege job fails with
`ENOSPC`. If custom-code submissions start failing and stock ones don't, run
`df -h` before anything else.

What the gate actually buys, stated honestly:

- the job runs as an unprivileged user, so a stray `rmtree` in someone's
  modeling file cannot touch what that user does not own;
- `HF_TOKEN` and friends are stripped from its environment and the hub is put
  offline — a local artifact needs neither, so nothing is lost. Make the token
  file root-owned and mode 600 (`chmod 600 $HF_HOME/token`) and the drop does
  the rest;
- every `.py` in the artifact is sha256'd into the run's provenance, so what
  code produced a score is on the record and a changed file is visible;
- `REMOTE_CODE_SHAS` turns that into an allowlist when you want review-then-run.

What it is **not**: a sandbox. The code still runs on the host GPU, in the host
PID namespace, can read the results tree, and can burn CPU, RAM and disk.
Anyone who can upload an artifact and holds the token can still ruin your day if
they set out to. The threat this design addresses is a teammate's code
misbehaving, not an adversary. If you ever need the stronger property, the shape
is a separate ephemeral container per remote-code job, without `pid: host` and
without `HF_HOME` mounted.

Submissions from the Hub are refused whatever the settings — "a teammate
uploaded this to our box" is the only trust signal there is, and a Hub id
carries none. Upload it as an artifact instead.

```bash
# submitting one, once the server is configured
python bench_client.py --base http://…:8899 \
    submit local/my-moe-step4000 --suite full --kind base --allow-remote-code
```

## How a submission behaves

`queued → preflight → waiting_lock/waiting_gpu (if needed) → running (k/n · task)
→ done | failed`. Failures carry a plain-language reason (gated repo, OOM, size
cap, custom-code refusal) plus a `log` link with the raw output. Canceling is
only possible while `queued` — a running job finishes its current task.

A third suite, `control`, runs only `mmlu_perm` — MMLU with the answer options
rotated, the experiment DIAGNOSE.md describes. It is a control, never part of
the average, and it queues like anything else: one job at a time, same lock.

Two properties worth telling teammates: submitting a model that is already on
the leaderboard costs nothing (per-task resume sees the results and finishes in
seconds), and a `quick` run (hellaswag + arc_easy + perplexity slices) later
upgrades to `full` by running only the missing tasks.

## API (everything the page does, scriptable)

```bash
B="http://$(tailscale ip -4):8899"
curl -s $B/api/results | jq '.models | length'
curl -s -X POST $B/api/submissions -H 'Content-Type: application/json' \
     -d '{"hf_id":"EleutherAI/pythia-31m","suite":"quick","submitter":"masein"}'
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

Three things hold all state: the `results/` tree, `service.sqlite3` (queue +
training runs + metrics), and `artifacts/` (uploaded checkpoints). Copy those,
and a fresh checkout of this repo reproduces the rest.
