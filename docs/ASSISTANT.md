# The assistant (28)

A chat panel on the board that answers questions about what the board holds
and opens its pages. It is for the team: only the names in `ASSISTANT_USERS`
see it. The brief is `docs/prompts/phase-28-chat-assistant.md`.

## How it works

The model picks, the board does. For every message the model chooses one
action and fills in its fields, as JSON forced into the board's schema (the
provider's JSON-schema mode). The board resolves every name against its own
data, refuses an id it doesn't know (a clarify with the nearest few), runs
the action and draws the reply with its own parts. The model never writes a
number.

| Action | What the board does |
|---|---|
| `open_page` | Builds the board's own link and opens it: a Models tab with its view, group, chart, Show (`only`), search (`q`), columns (`cols`), sort, Thinking, When (7d, 30d, today, this month, since the newest checkpoint, since a model was added, a range) and its models or the phone builds; a model's page or its answers on a benchmark; a report — the two-builds reader on a benchmark and side, Step checkpoints' baseline (`base`), a ran-out pick, a testers' test panel (the newest of a phone and app model); Runs, Benchmarks or a benchmark's questions, Playground, Improve, Home. |
| `compare` | 27's cells for 1 to 8 models on benchmarks: the score, what each keeps of its own base, how many ran out of room, or how many ended in a loop. Published numbers are shown as published, never ranked with ours. |
| `best_on` | The top few measured here, within a group (ours, open, closed, phone), and the published ones listed apart. "Phone" is the On-device report's published models and our builds our testers have run on a phone. |
| `chart` | Scores by benchmark's dots, Across training's panel, the file-size chart or On-device's memory chart. |
| `phone` | Our testers' medians per phone and app model: decode, prefill, time to first token, engine memory, temperature. "The oneplus" is every OnePlus. |
| `boxes` | What is running on rented boxes and on the server, and when a rented run should finish. |
| `explain` | The board's own help text: a report's notes, a benchmark's description, or a legend the page keeps. |
| `clarify` | One question with the choices as buttons; a click asks again. |
| `cant` | What it can't do (start, stop, delete, change, spend, grade, upload, or anything off the board) and where the button is. |

Read-only by construction: every action is a read or a navigation. The
module calls none of the board's writes, and its one outgoing request is the
model's chat completion (`tests/test_28_assistant.py` checks both).

**The prompt** is the rules, then the board's lists: each model with its id,
name, other names and group; the benchmarks; the reports; our testers'
phones; the phone app's models; the topics; a few examples (never the test
bank's). It stays the same from message to message while the board does, so a
llama-server's prompt cache keeps it; today's date, the previous question and
its action (for a follow-up, "and on HLE?") come with the question.

**Names** people type are in `eval_tasks/assistant/names.json`, with the
benchmarks' names, the topics and the examples. The board matches loosely:
case, spaces, hyphens and dots don't count; a name inside another or a close
one counts when only one model has it.

## The panel

"Ask" at the top right (in Menu ▾ on a phone), and ⌘K / Ctrl+K. A panel on
the right, 380 px wide, over the page; a phone's whole width. Messages stay
for the session, in memory; Clear empties them. Under each reply 👍 and 👎; 👎
asks "What did you want?". Three example questions from the test bank when it
is empty. While the model answers: "Thinking…", or "Slower now: a run is using
the model…"; with the model down, the panel says so and nothing else breaks.

## Only us

- `ASSISTANT_USERS` in `.env`: the names from "Who are you?", comma-separated.
  Empty or missing (the default, and the mirror's): the assistant doesn't
  exist — the payload says so and the page asks nothing.
- The API refuses everyone else (403); with nobody named, it is 404.
- Every question, its action, its reply's kind and time, and every thumb go
  to `BENCH_ROOT/assistant/log.jsonl` — internal, never in the mirror — so the
  test bank grows from real misses.

## The model

`ASSISTANT_PROVIDER` in `.env`:

- `served:<a served model's id>` — the default is the Playground's
  `served:served/Qwen3.6-35B-A3B-Q4-original-k-8`;
- `llama:<a llama-server's /v1 URL>` — a model of its own, e.g. the small one
  in the GPU's free memory;
- `openrouter:<model>` — for comparison only, with the board's OpenRouter key.

Thinking off, temperature 0, 600 tokens, the JSON schema on; one request a
message.

## The tests

- **The bank**, `tests/fixtures/assistant/queries.jsonl`: 256 questions in 12
  scenarios (15 or more each), each with the action it should produce.
- **Fast** (CI), `tests/test_28_assistant.py`: a fake model answers each bank
  question with its expected action; the board must run it — the link, the
  table's values, the chart, the phones, the boxes, the help text, the
  clarify, the refusal. Also an unknown id, a reply that isn't the JSON, a
  model down, the request's schema and settings, the people allowed, the log,
  and read-only. `tests/test_28_browser.py` draws the panel at 1280 and 400,
  light and dark.
- **Real-model** (on the server, never in CI): `scripts/assistant_eval.py`
  asks a model the whole bank as the board asks it and prints the pass rate per
  scenario, the reply times and every miss with what the model returned. The
  target: 95% overall, no scenario under 85%.

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/assistant_eval.py --provider served:served/Qwen3.6-35B-A3B-Q4-original-k-8
```
