# Brief for Claude Code — run the whole loop against a local model

Phase 7 is merged and green (185 tests). The loop is built but has never run
against anything real: the Anthropic and OpenAI clients and the judge have only
ever seen the `fake` backend and stubs. There are no cloud API keys yet.

There *is* a local model on the box we deploy to. This brief adds it as a
provider so the loop can be driven end to end today, and adds a narrated demo
script so a person can watch each step and understand what was built.

**Read first:** `service/llm.py` (the `Backend` interface, `ROLES`,
`backend_for`, `blocked`), `scripts/judge.py::blocked`, `service/config.py`
lines around `JUDGE_*` and `ALLOW_SINGLE_PROVIDER_LOOP`,
`docs/prompts/phase-7-exam-driven-loop.md`.

Do **P0** first as its own small PR, then P1, then P2, then P3.

---

## The endpoint

vLLM's OpenAI-compatible server, on the deploy box:

- `http://localhost:8000/v1` — **loopback only**
- served model id: `chat` (weights `google/gemma-4-E4B-it`)
- `api_key` is required by the OpenAI client library and **ignored** by vLLM
- `GET /v1/models` lists what is served
- honours `temperature`, `max_tokens`, and `response_format={"type":"json_object"}`

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "chat", "temperature": 0, "max_tokens": 300,
  "messages": [{"role":"user","content":"Say hello."}]}'
```

**The shared card.** 32 GB total; vLLM holds 13.3 GB, and with Qwen loaded in
Ollama the card sits at ~94%. A long-context request from either side can push
it over. This constrains the design — see P1.

---

## P0 — Two bits of cleanup

1. `rm -rf logs/_testenv logs/_repo_snapshot.tgz` — a throwaway venv and a repo
   tarball left behind by an earlier test run. Both are under the gitignored
   `logs/`, so this is housekeeping on disk, not a commit.
2. Add `.claude/` to `.gitignore` — it is showing as untracked.

One commit for the `.gitignore` line. Nothing else.

---

## P1 — A `local` provider for `service/llm.py`

### The thing that decides the design

**vLLM does not implement the Batch API.** There is no `/v1/batches`. So this
is *not* `OpenAIBatches` pointed at a different base URL — that would 404 on
the first submit. The new backend must present the same batch *interface*
while fulfilling it with ordinary synchronous chat completions.

```python
class LocalOpenAI(Backend):
    name = "local"
```

- `submit(requests)` mints a batch id, writes the pending batch to disk under
  `config.BENCH_ROOT/llm_batches/local/<id>/`, starts a worker thread, and
  **returns immediately**. Never block in `submit()`: exam drafting is a couple
  of hundred requests, and the callers and `service/llm_poller.py` already
  expect to poll.
- The worker walks the requests with **bounded concurrency** (`LOCAL_CONCURRENCY`,
  default **2**) and writes each `Result` to disk as it lands, so a restart
  resumes instead of re-running. Follow `FakeBatches`' on-disk layout — it
  already solves this shape.
- `status()` reports `pending` with a progress detail (`"41/225 done"`), then
  `done`, or `failed` with the first error if every request failed.
- `fetch()` returns `{custom_id: Result}` exactly as the others do.
- HTTP through the existing `_http` helper. On a 5xx, a timeout, or anything
  that reads like CUDA OOM: back off (2s, 8s, 30s) and retry up to three times,
  then record that one request's `Result.error` and carry on. **One failed
  request must never fail the batch** — the card is shared and transient
  pressure is expected.
- `api_key` is optional for this provider. `blocked()` must stop demanding a
  key when the provider is `local`; it must still demand a model id.
- At client construction, `GET /v1/models` once: confirm the configured model
  is served, and record the served list. A clear "vLLM is up but serves
  `['chat']`, not `'gemma'`" beats a 404 on the first click.

### JSON output

Gemma-4-E4B is small and will wrap JSON in prose more often than a frontier
model. Add an optional `json: bool = False` field to `llm.Request`; the `local`
backend turns it into `response_format={"type":"json_object"}`, and the other
backends ignore it. Set it `True` in the callers that parse JSON — the proposal
parser, `parse_items`, and the exam drafter. `extract_json` stays as the
belt-and-braces fallback.

### Config

`LOCAL_BASE_URL` (default `http://localhost:8000/v1`), `LOCAL_CONCURRENCY`
(default 2), `LOCAL_MAX_TOKENS` (default 1024, a deliberate cap for the shared
card), `LOCAL_TIMEOUT_S` (default 180). Add `"local"` to `PROVIDERS` and to
`backend_for`. Document all of it in `.env.example` and SERVICE.md, including
the loopback-only note and the `ssh -L 8000:localhost:8000` tunnel.

### The stamp — this is the part that matters

A local server's model id is whatever someone typed at launch. It is not a
dated id and it can change under you with no version to check — the pinning
problem the phase-7 brief was built to avoid, only worse. And a 4B model is not
a judge you would publish scores from.

So: **any artefact produced by a `local` identity is marked, always, with no
flag to turn it off.**

- `judge.json` from a `local` judge records `"provisional": true` with the
  reason `"graded by a local model — not a pinned benchmark"`, plus the base
  URL, the served model id, and the weights id if `/v1/models` gives one.
- The dashboard shows those scores **greyed, labelled, never ranked**, and
  never in any average — reuse the existing preliminary mechanism rather than
  inventing a second one.
- Dataset provenance records the same for a `local` generator or exam writer.
- `scripts/judge.py::blocked` stops refusing a `local` judge for the
  unpinned-id reason, and instead lets it run *provisional*. The refusal
  becomes a stamp. Every other refusal stays.
- Running all three roles on `local` is a provider clash, so it needs the
  existing `ALLOW_SINGLE_PROVIDER_LOOP=1`. Keep that requirement — someone
  should have to type it — and keep its "single-provider loop" caveat on the
  page alongside the provisional one.

**Tests.** A stub HTTP server in the test (never the real vLLM, and CI has no
GPU): the batch interface round-trips; concurrency is bounded; a 500 on one
request retries then degrades to that one `Result.error` while the batch still
completes; `json: True` puts `response_format` on the wire; a missing key is
fine for `local` and still fatal for `openai`; `/v1/models` mismatch gives a
readable error; `judge.json` from a `local` judge carries `provisional` and the
dashboard greys it. Mark none of these `gpu` or `network` — they must run in CI.

---

## P2 — `scripts/demo_loop.py`, a narrated end-to-end run

The point is comprehension, not coverage. Someone runs one command on the box
and watches the loop turn, with every artefact's path printed.

```
python scripts/demo_loop.py --topics economics,law --model EleutherAI/pythia-160m
```

Each step prints a short heading, what it is about to do, what came back, and
where it landed:

1. **Preflight** — vLLM reachable, served model, `BENCH_ROOT`, the three role
   identities and whether each is blocked and why. Stop here on any problem.
2. **Draft the exam** — a handful of candidates per topic (default 12, not 90),
   through the `local` backend. Print two verbatim, with their reference
   answers.
3. **Accept** — `--auto-accept` for the demo, recording the approver as
   `demo` so nobody mistakes it for curation. Print the report/diagnose split
   counts per topic and say plainly that a real run needs 30 report-half
   questions per topic before anything can be proposed.
4. **Build the tasks** and print the yaml paths.
5. **Sit the exam** — the named model, through the normal queued path and the
   normal lock. Default to a small model; say why.
6. **Judge** — print the per-topic table, the canary drift, and the
   `provisional` stamp.
7. **Pick the weakest topic**, then **propose** — print the judge's
   justifications that went into the request and the spec that came out.
   **Assert, in the script, that no exam question text is in the request
   body**, and say so in the output. This is the safety property; show it
   working rather than asserting it in prose.
8. **Generate** a small dataset (default 20 documents), run the contamination
   gate, print kept/dropped and one document in full.
9. **Summary** — every path written, and the one-line "what would happen next"
   (fine-tune with `--gap-dataset`, resubmit, compare halves).

`--dry-run` prints the plan without calling anything. `--keep` leaves the
artefacts; default cleans up the demo bank so it cannot be mistaken for the
real one. Everything it creates is stamped `demo` and lands under
`$BENCH_ROOT/demo/`, never in the live exam bank.

**Test** it end to end against the `fake` backend in CI, so the script itself
cannot rot.

---

## P3 — Docs

A `DEMO.md`: the endpoint, the `.env` block to paste, the one command, what
each step shows, and — plainly — **what this does not test**. That last section
matters: driving the loop against a local synchronous server exercises the
callers, the split, the airlock, the gate and the dashboard. It does **not**
exercise the Anthropic or OpenAI batch clients, which remain the untested glue
until real keys exist. Say so in those words, so a green demo is not mistaken
for a green production path.

---

## Definition of done

CI green. `pytest -q` still under a minute for the non-browser suite. On the
box: `python scripts/demo_loop.py --topics economics --model <small>` runs
start to finish against vLLM and prints a judged score, a spec, and a generated
document — every one of them visibly stamped provisional, and none of them on
the leaderboard.
