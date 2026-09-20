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
| `LLM_PROVIDER` | *(unset — off)* | `anthropic` / `openai` / `local` / `fake`: the model behind skill-spec proposals and data generation (batch API only; `local` is vLLM on this box — see *The local model*) |
| `LLM_MODEL` | *(unset)* | the generator model id, pinned; recorded in every dataset's provenance |
| `LLM_API_KEY` | *(unset)* | in `.env` only — see *The LLM key* below |
| `LLM_MAX_ITEMS_PER_BATCH` / `LLM_DAILY_ITEM_CAP` | 200 / 2000 | spend guard, in batch requests (one per proposal, one per ten generated items); the Review tab shows today's use |
| `DATASET_QUOTA_GB` | 20 | total generated-dataset storage under `$BENCH_ROOT/datasets` |
| `EXAM_PROVIDER` / `EXAM_MODEL` / `EXAM_API_KEY` | *(unset — off)* | the exam writer behind `scripts/exam_build.py draft`. A different identity from the generator and the judge, on purpose |
| `EXAM_DIR` | `$BENCH_ROOT/exam` | `candidates/` awaiting curation, `bank/` accepted questions (split by qid), `tasks/` what the harness runs |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` / `JUDGE_API_KEY` | *(unset — off)* | the judge: an API call, batch mode. A **dated** model id, never an alias; a **different provider** from the exam writer and the generator; `stub` for a dry run. The page states the reason when any rule fails. `local` is the one exception to the dated id: it runs, **provisional** |
| `JUDGE_CANARY_MAX_DRIFT` | 0.5 | thirty fixed scripts are re-graded every run; if their grades move more than this from the previous run the run is preliminary |
| `ALLOW_SINGLE_PROVIDER_LOOP` | 0 | the documented override for a one-provider trial; every judged score is then stamped "single-provider loop" |
| `JUDGED_TASKS_DIR` | `$EXAM_DIR/tasks` | where `scripts/exam_build.py build` put the exam tasks |
| `LOCAL_BASE_URL` | `http://localhost:8000/v1` | the `local` provider's OpenAI-compatible server (vLLM). Loopback only on the deploy box |
| `LOCAL_CONCURRENCY` | 2 | `local` requests in flight at once — the card is shared |
| `LOCAL_MAX_TOKENS_LLM` | 1536 | the cap on a `local` generation or proposal reply. A ~600-word training document is ~1,400 tokens and the request already asks for one, so there is no smaller request to make; two of these in flight is well inside what the shared card has left |
| `LOCAL_MAX_TOKENS_JUDGE` | 1024 | the cap on a `local` judge reply — a 23-criterion JSON answer is about 350 tokens |
| `LOCAL_MAX_TOKENS_EXAM` | 1024 | the cap on a `local` exam-writer reply |
| `LOCAL_MAX_TOKENS` | *(unset)* | the fallback for any role above that has no knob of its own. A truncated reply names the knob that capped it |
| `LOCAL_TIMEOUT_S` | 180 | per `local` request; a timeout is retried like a 5xx |

## The LLM key

Phase 4 of `DIAGNOSE.md` (find the gap, generate data) uses an external LLM
through its batch API. The key is a secret, and this is exactly where it is
and is not protected:

- It lives in `.env`, which is gitignored, and reaches the container through
  docker-compose `${LLM_API_KEY}` interpolation. It is **never a literal in
  `docker-compose.yml`** and never in git — a test greps for both.
- It is **stripped from every evaluation subprocess** (`SECRET_ENV_VARS` in
  `service/runner.py`), so a submitted model's own code cannot read it. Add
  any new secret's variable name to that list in the same commit that adds
  the secret.
- It is **not hidden from anyone with docker access on the box**:
  `docker inspect aienh-bench-1` prints `Config.Env`, key included. That is
  the trust model here — the tailnet and the docker group are the boundary —
  and it is worth saying plainly: the key is protected from submitted code,
  not from colleagues. Use a key with a spend limit set at the provider.

Generation is off unless `LLM_PROVIDER` is set; the Review tab says so. A
provider that is set with a missing model or key fails the container at
startup, where the operator is looking, rather than at the first click.
Everything the LLM produces goes through the contamination gate
(`service/contamination.py`) before it can be stored, and every dataset
carries a full provenance record. Who approved a spec is recorded as a
typed name, the same way submissions record a submitter — there is no login.

## The local model (vLLM)

There are no cloud keys yet; there is a vLLM server on the deploy box. The
`local` provider lets any of the three roles — generator, exam writer, judge —
run against it, so the loop can be driven end to end today.

- **Endpoint.** `http://localhost:8000/v1`, **loopback only**. Served model id
  `chat` (weights `google/gemma-4-E4B-it`); `curl -s localhost:8000/v1/models`
  lists what is served. From another machine, tunnel it first:

  ```bash
  ssh -L 8000:localhost:8000 <box>
  ```

- **From inside the container, through the host gateway.** In the service
  container `localhost` is the container, not the box, so compose maps the
  box's loopback in as `host.docker.internal` (`extra_hosts:
  host.docker.internal:host-gateway`) and defaults `LOCAL_BASE_URL` to
  `http://host.docker.internal:8000/v1`. vLLM stays bound to 127.0.0.1 and
  nothing new is exposed on the tailnet or the LAN. `network_mode: host`
  would also reach it and would throw away the `${BIND}` publish line that
  keeps the service off the LAN, so it is the gateway, not host networking.
  On the host — the demo script, `exam_build.py draft`, `judge.py --wait` —
  the code default `http://localhost:8000/v1` is the right one. If the URL is
  wrong the client says so plainly instead of failing at the first click.
- **No key.** vLLM ignores one unless it was launched with `--api-key`; if it
  was, the role's `*_API_KEY` is sent. A missing key is fine for `local` and
  still fatal for `anthropic` and `openai`. A missing model id is fatal for all.
- **No batch API.** vLLM has no `/v1/batches`, so the client keeps the batch
  *interface* and fulfils it itself: `submit()` writes the batch under
  `$BENCH_ROOT/llm_batches/local/<id>/` and returns at once; a worker thread
  runs the requests, `LOCAL_CONCURRENCY` at a time, appending each result to
  `results.jsonl` as it lands. A restart resumes from that file instead of
  re-running. The work happens in whichever process is polling the batch —
  the service's poller, or `exam_build.py fetch` / `judge.py --wait` on the
  command line.
- **One bad request never fails the batch.** A 5xx, a timeout, a refused
  connection or anything that reads like CUDA OOM is retried after 2, 8 and
  30 s, then recorded as that request's error. A 4xx is recorded at once — a
  malformed request or an unknown model id does not fix itself. The batch is
  `failed` only when every request failed.
- **One item per request.** Asked for several, a small model answers in
  shapes nothing can use: two documents at once overrun `LOCAL_MAX_TOKENS_LLM`
  and the JSON is cut off, and four exam questions come back as one question,
  or as a list of question texts with no reference answers. So a `local`
  generator is asked for one document and a `local` exam writer for one
  question; the batch simply gets a row each, which the on-disk resume does
  not care about. Both parsers also read the shapes JSON mode actually
  produces — the object itself, or the array wrapped in one — and the exam
  drafter reports any reply it could not read, per request, rather than
  leaving a round with no candidates and no reason.
- **Checked at start.** The first use asks `/v1/models` and refuses a model id
  the server does not serve, by name: *vLLM is up but serves `['chat']`, not
  `'gemma'`*. If nothing answers, the poller waits and tries again next tick
  rather than failing pending batches — vLLM takes a while to load after a
  reboot.

**Everything a local identity produces is provisional, and there is no flag
to turn that off.** A local server's model id is whatever someone typed at
launch: not dated, and changeable with no version to check. A 4B model is
also not a judge anyone publishes scores from. So:

- `judge.json` from a local judge records `"provisional": true`, the reason
  (*graded by a local model — not a pinned benchmark*), the base URL, the
  served id and the weights. The reason joins the preliminary reasons, so the
  page shows those scores **greyed, labelled, never ranked and in no
  average**. No calibration lifts it, and nothing a local judge wrote can pick
  a topic to train. The same-family rule uses the *weights'* family (`gemma`),
  not the served id's; the canary still shows whether the weights behind
  `chat` changed between runs.
- Exam candidates a local writer drafted, a spec a local proposer wrote and a
  dataset a local generator made all carry the same stamp. Dataset
  provenance lists which roles were local under `local_models`.
- All three roles on `local` is a provider clash and still needs
  `ALLOW_SINGLE_PROVIDER_LOOP=1`. Someone has to type it, and the
  single-provider caveat stays on the page beside the provisional one.

A green run against the local model exercises the callers, the split, the
airlock, the gate and the dashboard. It does **not** exercise the Anthropic or
OpenAI batch clients, which stay untested until real keys exist.

To watch the whole loop turn once against it, in one command:

```bash
python3 scripts/demo_loop.py --topics economics --model EleutherAI/pythia-160m
```

That is `scripts/demo_loop.py`, and [`DEMO.md`](DEMO.md) is its page: the
`.env` block to paste, what each of its nine steps shows, and — plainly —
what a green run does not prove.

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

A fourth suite, `judged`, runs the exam (`exam_<topic>` tasks built from the
curated bank, plus `fr_control_mmlu`) inside the lock, then **submits** the
answers to the API judge as one batch (seconds, no GPU) and releases the
card; the service's poller writes `judge.json` when the provider completes
the batch. A judge that waits on an API never holds the card. Files written
by the earlier local judge are kept, labelled `local`, and shown as their own
series — never merged with API-judged scores.
Nothing judged is ranked until `scripts/judge_calibrate.py` has a human
sample with Cohen's κ ≥ 0.60 on file; see DIAGNOSE.md.

The rubric and criteria file that grade each topic can be replaced from the
Exam tab (preview, then commit, under a name that is recorded in the
`rubric_changes` table). They are written to `eval_tasks/fr/rubrics/` in the
checkout when the service can write there, and to `$BENCH_ROOT/rubrics/`
when it cannot — in the container `/app` is the image, not the bind mount,
so on the box the uploads land beside the bank. The judge reads
`$BENCH_ROOT/rubrics/` first and the repo second, and the page says which
directory is live. Changing either file changes its sha256, which every
`judge.json` records: scores from before and after are not comparable, so
re-run `suite=judged` for that topic.

```bash
# once: the four skill suites' items into the bank, under 'other'
sudo docker compose exec -T bench python3 scripts/exam_build.py migrate --root /home/masein/benchmarks/exam
# each round: draft candidates (needs EXAM_*), curate them on the Exam tab, then build
sudo docker compose exec -T bench python3 scripts/exam_build.py draft --root /home/masein/benchmarks/exam --per-topic 8
sudo docker compose exec -T bench python3 scripts/exam_build.py build results/full --root /home/masein/benchmarks/exam
#   (the Exam tab's "Rebuild the harness tasks" button does the last step too)
# a bank someone wrote by hand — or the Exam tab's "Import a bank" panel,
# which runs this same code path, previews it first and records the name
sudo docker compose exec -T bench python3 scripts/exam_build.py import eval_tasks/fr/medicine_v2.json \
    --root /home/masein/benchmarks/exam --topic 'medicine & health' --approver 'Dr. Hossein'
# then per model:
python clients/bench_client.py --base http://<ip>:8899 submit <model> --suite judged --submitter you
```

A third suite, `control`, runs only `mmlu_perm` — MMLU with the answer options
rotated, the experiment DIAGNOSE.md describes. It is a control, never part of
the average, and it queues like anything else: one job at a time, same lock.

Two properties worth telling teammates: submitting a model that is already on
the leaderboard costs nothing (per-task resume sees the results and finishes in
seconds), and a `quick` run (hellaswag + arc_easy + perplexity slices) later
upgrades to `full` by running only the missing tasks.

## The page's addresses

Every view is in the URL, so a link to one is a link someone else can open.
The hash is the tab's own label, lower-cased:

| Hash | What |
|---|---|
| `#tab=overview` | the board's summary |
| `#tab=loop` | the loop, one row per topic |
| `#topic=<slug>` | one topic's page: bank, rubric, sit, answers, propose (`#topic=law`) |
| `#tab=models` | every model, with the filters that narrow it |
| `#tab=leaderboard` | the table, then the radar of whatever compare holds |
| `#tab=queue` | Submit & Queue |
| `#tab=exam` | the bank, the import panel, the rubric and criteria files |
| `#tab=review` | proposals awaiting a person, and the datasets |
| `#tab=training` | training runs and their curves |
| `#tab=tasks`, `#tab=perplexity` | per-task panels |
| `#tab=provenance` | how every number was produced |
| `#model=<id>` | one model's page (`#model=fx%2Fgood-750m`) |

Older hashes still resolve: `#tab=runs` and `#tab=evals` land on Provenance,
`#tab=submit` on Submit & Queue, `#tab=ppl` on Perplexity & Loss. A hash the
page does not know leaves you where you were rather than silently on
Overview.

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
**`this build cannot serve requests: …`** and the container exits at `up` —
the image is missing a file the service reads at run time (the harness task
template, a rubric, the canary). The message names the paths. It means the
Dockerfile's COPY list or `.dockerignore` dropped them; rebuild with
`docker compose up -d --build` after fixing, and note that
`tests/test_image_contents.py` checks this without building. The check runs
at startup on purpose: a missing file used to surface as a 500 on whichever
button needed it first.

## Backup

Four things hold all state: the `results/` tree, `service.sqlite3` (queue +
training runs + metrics + proposals and dataset records), `artifacts/`
(uploaded checkpoints), and `datasets/` (generated data, each with its
`provenance.json`). Copy those, and a fresh checkout of this repo reproduces
the rest.
