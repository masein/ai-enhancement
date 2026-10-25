# The assistant (28)

A chat panel on the board that answers questions about what the board holds
and opens its pages. It is for the team: only the names in `ASSISTANT_USERS`
see it. The brief is `docs/prompts/phase-28-chat-assistant.md`.

## How it works

The model picks, the board does. For every message the model chooses one
action and fills in its fields, as JSON forced into the board's schema (the
provider's JSON-schema mode). The schema is strict: one branch an action — for
`open_page` one a page, `models_table` (the Models tabs) or `model_page` (one
model) — each with its own fields alone, the closed lists as enums. The reply
is `{"do": an action}`, or `{"do": the first, "then": the second}` when a
message asks two things: a second request is a whole action of the same
branches, never free text. Every name — model, benchmark, report, phone, app —
— and (28d) a Models group or a place on the top bar — is a string as the
person wrote it ("step one", "the arc one", "the report about phones", "the
galaxy", "mobile-mmlu", "the catalogue"): the board matches it against its
names and asks back when several match (step1000, step4000, step8804;
ARC-AGI-2, ARC-Challenge, ARC-Easy; the testers' and the On-device report).
When none does, it says so — "I couldn't find “gpt-9”" — with a search for
it, never a question without choices. The model clarifies only when the
message names nothing ("it", "the two") and there is no previous action. A
tab or a thinking setting that neither the message nor the one before it
names is left out — "only ours" stays on the tab you're on. The board then
runs the action and draws the reply with its own parts. The model never
writes a number.

**The person's words, checked (28d).** A name the model wrote that the
messages don't hold is the model's own choice, and the board checks it
against what the person did write: kept when some of their words mean that
one alone ("Q4 (UD-Q4_K_XL)" for "the q4"); their words instead when those
mean it among others, so the board asks back ("Qwen3.6-35B-A3B" for "the
qwen3.6 model", "testers" for "the report about phones"); and a model or app
nothing they wrote means is left out — the board asks for it when the action
needs one ("compare the two on GPQA"). A report, benchmark or phone can be
what the message asks for without its name ("what step8804 got wrong that Q4
got right" is the two-builds reader), so those stay. Several names the same
words mean ("the three bonsai 2 models") are the model spelling them out, and
stay. A date, "none" or a page's name is never a name. A list the check cuts
— some of its models stay, some go — is never shown as what's left: the board
asks back between all of them (28f; "compare the two checkpoints" kept only
step8804, whose other names say "checkpoint").

**A model referred to, never named (28f).** "They", "them", "it", "the two"
or "the other one" with no earlier turn and no model named in the message:
the board asks which model, whatever action the model chose — it answered
anyway, with a ranking or a guess. A refusal and a word explained ("had it,
then lost it") are left alone.

**One model.** Rows that are settings of one model — thinking on or off,
"lookahead 1" — are one model, never asked between; its row is the one with
the most scores, a tie to thinking on (27c's rule), then to the fewest
settings.

| Action | What the board does |
|---|---|
| `open_page` | Builds the board's own link and opens it. `models_table`: a Models tab with its view, group, chart, Show (`only`), `search`, `columns`, sort, Thinking, When (7d, 30d, today, this month, since the newest checkpoint, since a model was added, a range) and its models or the phone builds. `model_page`: a model's page, or its answers on a benchmark beside another's, with thinking on or off (28e: its row, its link and the number in the reply are that setting's). `report`: the two-builds reader on a benchmark and side, Step checkpoints' baseline (`base`), a ran-out pick, a testers' test panel (the newest of a phone and app model). Runs, Benchmarks or a benchmark's questions, Playground, Improve, Home. |
| `compare` | 27's cells for 1 to 8 models on benchmarks: the score, what each keeps of its own base, how many ran out of room, or how many ended in a loop. Published numbers are shown as published, never ranked with ours. One model on one benchmark — here, or its page opened on that benchmark — and the reply says the number, its ± and the thinking setting ("step8804 on GPQA Diamond: 70.0 ± 2.0, thinking on"), or who published it. |
| `best_on` | The top few measured here, within a group (ours, open, closed, phone), and the published ones listed apart. "Phone" is the On-device report's published models and our builds our testers have run on a phone. |
| `chart` | Scores by benchmark's dots, Across training's panel, the file-size chart or On-device's memory chart. |
| `phone` | Our testers' medians per phone and app model: decode, prefill, time to first token, engine memory, temperature. "The oneplus" is every OnePlus. |
| `boxes` | What is running on rented boxes and on the server, and when a rented run should finish. |
| `explain` | The board's own help text: a report's notes, a benchmark's description, or a legend the page keeps. "What's churn in the two-builds report" is explain, the report its second action. |
| `clarify` | One question with the choices as buttons; a click asks again. |
| `cant` | What it can't do (start, stop, delete, change, spend, grade, upload, or anything off the board) and where the button is. |

Read-only by construction: every action is a read or a navigation. The
module calls none of the board's writes, and its one outgoing request is the
model's chat completion (`tests/test_28_assistant.py` checks both).

**The prompt** is the rules, then the board's lists: each model with its
name, other names and group; the benchmarks; the reports; our testers' phones
with their other names; the phone app's models with theirs; the topics; a few
examples (never the test bank's). It stays the same from message to message
while the board does, so a llama-server's prompt cache keeps it; today's date
and the message come after.

**A follow-up** ("and on HLE?") gets the previous action, its models by name,
with the rule to keep it and change only what the new message changes.

**Two requests in one message**: the first is done, and the second — a whole
action, worked out with the first — is a button under the reply ("Also: Open
On-device models") that shows it. "What's X in the report" is the
explanation first and the report second, whichever way the model ordered
them; a follow-up whose first action is the turn before's again ("and as a
chart" after "best on gpqa") shows only the new one (28e).

**Names** people type are in `eval_tasks/assistant/names.json`, with the
benchmarks' names, the places' other names ("catalogue" is Benchmarks), the
topics and the examples. The board matches loosely: case, spaces, hyphens and
dots don't count; a name inside another or a close one counts when only one
model has it — inside another only from one of its words' starts, never
running on into part of a later word ("then" is not in "the newest
checkpoint"), and a name with an "s" stuck on is tried without it ("q4s",
"step4000s"; 28e). A benchmark is matched by its names, not its key, so "arc" is
every ARC; and by its Frontier group's name (28f), so "the
instruction-following one" asks between IFEval and IFBench and "the maths
one" between OTIS and MATH Level 5 ("math" alone is MATH Level 5). A Models
group is matched among the groups of the tab asked for, by its name exactly
or closely first ("Maths" is Standard's Math), inside a longer name only when
nothing else matches. A Models group is recognised before a benchmark (28e): written
where a benchmark goes ("mobile devicemark chart" as a chart of
"DeviceMark"), it is that group on its tab — unless a benchmark has that very
name ("math" is MATH Level 5). The fields a report or chart doesn't use are
left out (an app on the ran-out report, "only" on On-device's memory chart) —
after a field written in the wrong place is read as the one it is (28f): the
testers' report takes a named model as its app, so its newest test is that
model's, and a benchmark written as the ran-out report's app is its
benchmark. The ran-out report is "Ran out of room" to the assistant: its
page's title names a model, which the model copied. A phone is matched by
its name, code or other name before its maker: "the oneplus 15" is that
phone, "the oneplus" every OnePlus.

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
  test bank grows from real misses: `scripts/assistant_eval.py --thumbs-down`
  prints the questions whose last thumb is down, with the turn before, the
  board's action and what the person wanted — the next bank's questions.

## The model

`ASSISTANT_PROVIDER` in `.env`:

- `served:<a served model's id>` — the default is the Playground's
  `served:served/Qwen3.6-35B-A3B-Q4-original-k-8`;
- `llama:<a llama-server's /v1 URL>` — a model of its own, e.g. the small one
  in the GPU's free memory;
- `openrouter:<model>` — for comparison only, with the board's OpenRouter key.
  It is sent only the parameters OpenRouter's models API says the model takes,
  read once a run (28e): with `require_parameters`, which keeps the JSON
  schema honoured, a parameter the model doesn't take found no endpoint at
  all ("No endpoints found…", 404). A model that can't take `response_format`
  is refused with a clear message; the evaluation's header says what was left
  out.

Thinking off, temperature 0, 600 tokens, the JSON schema on; one request a
message.

## The tests

- **The bank**, `tests/fixtures/assistant/queries.jsonl`: 266 questions in 12
  scenarios (16 or more each), each with the action it should produce (`also`:
  another as good — "show the qwen3.6 model" may search for it or ask back;
  `then`: the second action it must offer). Graded by what the person sees
  (28d): the actions are compared after the board's defaults and matching,
  and what looks different but shows the same passes — every run for "the
  rented box" (boxes' `which` is all), a refusal of another kind, the base
  (BF16) beside the model whose keep is asked, a follow-up's new models or
  benchmarks alone or beside what was shown, a follow-up's tab left as it is,
  one model's page on a benchmark for its number, "not found" for a clarify.
  Each scenario is split in two:
  a **working half** (`"half": "work"`), the only one the prompt, the names and
  the schema are tuned on, and a **held-out half** (`"held"`), never looked at
  to tune; the evaluation reports both, so a high rate means more than a
  prompt that learned our questions.
- **Fast** (CI), `tests/test_28_assistant.py`: a fake model answers each bank
  question with its expected action; the board must run it — the link, the
  table's values, the chart, the phones, the boxes, the help text, the
  clarify, the refusal. Also an unknown id, a reply that isn't the JSON, a
  model down, the request's schema and settings, the people allowed, the log,
  and read-only. `tests/test_28_browser.py` draws the panel at 1280 and 400,
  light and dark.
- **Real-model** (on the server, never in CI): `scripts/assistant_eval.py`
  asks a model the bank as the board asks it and prints the pass rate per
  scenario for each half, the reply times and every miss with what the model
  wrote, before the board matched it, and what the board made of it. Tune
  with `--half work`; report both halves. The target: 95%
  overall, no scenario under 85%. Every raw answer is kept in
  `BENCH_ROOT/assistant/evals/` (internal, as the log is; the run prints the
  file), and `--rescore <that file>` grades the answers again — after a
  change to the board or the grading — without asking the model. Three
  answers in a row that don't come stop a run, with the error.

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/assistant_eval.py --provider served:served/Qwen3.6-35B-A3B-Q4-original-k-8
```
