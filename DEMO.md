# The loop, end to end, against the local model

One command on the deploy box runs the whole cycle — write an exam, curate
it, sit it, judge it, find the weakest topic, propose a skill spec, generate
training documents — and narrates every step, printing what it is about to
do, what came back, and where it landed.

```bash
python3 scripts/demo_loop.py --topics economics --model EleutherAI/pythia-160m
```

It is for comprehension. Nothing it writes goes near the live board: it works
in `$BENCH_ROOT/demo/` — its own exam bank, results tree, datasets, database
and batch state — everything it records is stamped `demo`, and the tree is
removed at the end unless you pass `--keep`.

**Read [What this does not test](#what-this-does-not-test) before you draw any
conclusion from a green run.**

---

## The endpoint

vLLM's OpenAI-compatible server on the box:

- `http://localhost:8000/v1` — **loopback only**
- served model id `chat`, weights `google/gemma-4-E4B-it`
- `GET /v1/models` lists what is served; the client asks once at startup and
  refuses a model id the server does not serve, by name
- no key needed: vLLM ignores one unless it was launched with `--api-key`

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "chat", "temperature": 0, "max_tokens": 300,
  "messages": [{"role":"user","content":"Say hello."}]}'
```

From your laptop, tunnel it first — it is not on the tailnet:

```bash
ssh -L 8000:localhost:8000 <box>
```

From inside the service container, `localhost` is the container: compose maps
the box's loopback in as `host.docker.internal` and points `LOCAL_BASE_URL`
there. The demo script runs on the host, where the default
`http://localhost:8000/v1` is right.

## The `.env` block

All three identities on the local model, which is a provider clash and needs
the documented override — someone has to type it:

```bash
LLM_PROVIDER=local
LLM_MODEL=chat
EXAM_PROVIDER=local
EXAM_MODEL=chat
JUDGE_PROVIDER=local
JUDGE_MODEL=chat
ALLOW_SINGLE_PROVIDER_LOOP=1
# LOCAL_BASE_URL=http://localhost:8000/v1   # the default on the host
# LOCAL_CONCURRENCY=2                       # the card is shared
# LOCAL_MAX_TOKENS=1024                     # every reply is capped
# LOCAL_TIMEOUT_S=180
```

The demo reads the same environment the service does, so exporting those in a
shell works as well as putting them in `.env`.

## What each step shows

| step | what it does | what to look for |
|---|---|---|
| 1 preflight | resolves the three identities, asks vLLM what it serves | every role `local/chat`, the weights, and the two caveats: single-provider loop, and provisional |
| 2 draft | the exam writer drafts candidates per topic (`--per-topic`, default 12), one question per request on `local` | a batch id, two candidate questions printed in full with their reference answers, and how many replies could not be read |
| 3 accept | `--auto-accept` accepts them unread as approver `demo` | it says plainly that this is **not** curation, and the report/diagnose split per topic |
| 4 build | writes the harness task yamls from the bank | the split counts, and the paths the harness will run |
| 5 sit | the model answers, through the normal runner, free-VRAM gate and shared lock | the submission id and the status; `--sit stub` writes answers without a model and says so |
| 6 judge | the judge grades the canary, then every answer | the per-topic table (report half is the score), the canary drift, and the **provisional** stamp |
| 7 propose | picks the weakest topic and turns the judge's words into a spec | the justifications that went in, the spec that came out, and the safety check below |
| 8 generate | the approved spec becomes documents, through the contamination gate | kept/dropped, what the gate indexed, and one document in full |
| 9 summary | every path written | what a real cycle would do next: fine-tune, resubmit, compare halves |

Useful flags: `--dry-run` prints the plan and calls nothing; `--keep` leaves
the tree; `--count` sets how many documents; `--no-auto-accept` stops after
drafting so you can curate on the Exam tab; `--sit stub` skips the GPU.

### The medicine run — a human-written bank

Dr. Hossein's 50 consumer health questions, with his 15-criterion framework
behind the grading:

```bash
python3 scripts/demo_loop.py --topic "medicine & health" \
    --import eval_tasks/fr/hossein_medicine_v1.json --approver "Dr. Hossein" \
    --model HuggingFaceTB/SmolLM2-360M-Instruct --keep
```

`--import` replaces steps 2 and 3 with one import step: his name is the
approver on every item, the metadata is the reference (acuity first), and the
step prints the report/diagnose split, the per-acuity counts and the
shortfall against the 30-question floor as a concrete ask back to him. The
run needs **no exam writer** configured — nobody is drafting anything.

Use an **instruct** model for this topic. A base model answers a triage
question with word salad and scores 0 on everything, which teaches nothing;
the demo's preflight says so when the model id does not look instruction
tuned.

Step 6 then shows what per-criterion grading looks like: each criterion's
mean weakest first, the critical-failure count with the acuities it fell on,
the by-acuity table, and one graded diagnosis-half answer in full — its
criteria, its flag and the fold that turned them into a 0–4.

### The safety property, shown rather than claimed

Step 7 reads the request bodies the backend actually wrote to disk and checks
every question in the demo bank — **both halves** — against them with the
contamination gate's own 13-gram rule, then prints `PASSED` or `FAILED` with
the offending question. The report half is never shown to a person or a model
anywhere in the loop, and the judge's justifications are about the *answers*;
this is the step that proves it for the run you just watched. Step 8 repeats
the check for the generation request, which is built from the approved spec
string and nothing else.

### Everything a local model made is provisional

A local server's model id is whatever someone typed at launch: not dated, and
changeable with no version to check. A 4B model is also not a judge anyone
publishes scores from. So every artefact of a local identity is marked, and no
flag turns it off:

- `judge.json` carries `provisional`, the reason, the base URL, the served id
  and the weights, and the reason joins the preliminary reasons — the
  dashboard greys those scores, never ranks them and leaves them out of every
  average. No calibration lifts it.
- Candidates, the bank rows accepted from them, the spec and the dataset
  provenance carry the same mark, and the demo prints each one.
- Because the scores are provisional, the live service would **refuse** the
  proposal in step 7 (the gate is enforced in the API, not just on the
  button). The demo goes around it deliberately, to show the shape of the
  step, and says so on the line above it.

## What this does not test

Driving the loop against a local synchronous server exercises the callers, the
split, the airlock, the gate and the dashboard. It does **not** exercise the
Anthropic or OpenAI batch clients, which remain the untested glue until real
keys exist.

In detail, a green demo says nothing about:

- `AnthropicBatches` and `OpenAIBatches` — file upload, `/v1/batches`,
  polling, the results and error files. The `local` backend has no batch API
  behind it at all; it keeps the same interface and runs ordinary chat
  completions itself.
- Real batch latency, 24-hour completion windows, partial results, expiry,
  cancellation or spend.
- Whether `LLM_MAX_ITEMS_PER_BATCH` and `LLM_DAILY_ITEM_CAP` are set sensibly
  for a paid provider.
- The judge as a *benchmark*: the scores from a local judge are provisional by
  construction, there is no calibration against a person behind them, and a
  4B model's grades are not evidence about a model's ability.
- Question quality. `--auto-accept` accepts whatever was drafted; a real bank
  is read question by question by a person whose name is recorded.
- **The medicine rubric and its criteria file are drafts**, pending Dr.
  Hossein's sign-off. They grade, and they are marked DRAFT on the page and
  in every `judge.json` until he signs them off by removing the word — which
  changes their sha, so scores from before and after do not compare.
- **The 0–4 for a criteria topic is a deterministic fold of the 15 criteria**
  (the rule is in `rubrics/medicine_health.criteria.json`), not a number the
  judge chose. Per-criterion agreement with a human has **not** been measured
  yet: `judge_calibrate.py` exports the columns for it and reports the
  differences, but nothing gates on them, and κ is still computed on the
  folded score alone.

So: a green demo is not a green production path.

## When it stops

The script stops at the first problem and says which one.

| it says | what to do |
|---|---|
| `nothing is answering at http://localhost:8000/v1` | vLLM is down, or you are off the box — start it, or `ssh -L 8000:localhost:8000 <box>` |
| `vLLM is up ... but serves ['chat'], not 'gemma'` | set `*_MODEL` to an id from `GET /v1/models` |
| `One identity is not usable` | the line under it names the variable; the block above is what a local trial needs |
| `the judge's provider (local) is the same as the exam writer's` | set `ALLOW_SINGLE_PROVIDER_LOOP=1` — deliberately not implicit |
| `the evaluation did not finish` | a real run failed: the log path is printed, and `--sit stub` gets you past the GPU while you look |
| `N cut off at LOCAL_MAX_TOKENS` | replies are being truncated — lower the items per request or raise the cap, knowing the card is shared |
| `replies unread — N of M` | the model answered in a shape nothing can curate (commonly question texts with no reference answers). The batch is on disk under `$BENCH_ROOT/demo/llm_batches/local/<id>/results.jsonl`: read what it actually sent. A whole round of this usually means the prompt needs the shape spelled out again, not that anything is broken |

## Related

[`SERVICE.md`](SERVICE.md) § The local model — the provider itself, its
config and its on-disk batches. [`DIAGNOSE.md`](DIAGNOSE.md) — the split, the
gate and the taint rules the loop is built on.
