# Brief for Claude Code — the five places (12b)

Start from main after 12a. This PR **moves things**. It changes where each
part of the board lives and what the header holds. It does not redesign
tables, readers or dialogs: they keep their code and move into new
containers. The only new compositions are **Home** and the **model page
tabs**, and both are built from pieces that exist today.

If it is too big for one review, ship §1–§5 as **12b.1** and §6–§8 as
**12b.2**, and say so in the PR description.

**The rules that do not bend are unchanged.** The plain-words rules from
11h §7 apply, and so do the display rules every 12x brief carries:

- **One number first.** No ±, shots, sample counts or ids in the main view.
- **Nothing empty.** A test a model hasn't taken is one line with a button.
  A section with nothing in it says so in one line, or isn't shown.
- **No system words.** Suite names, check names, file names and hashes
  stay out of the main view.
- **A caveat once per page**, as one badge in the header, never per row.
- **One main action**, one filled button, top right.
- **Numbers in mono, words in sans.**

---

## Why

The board is gaining three things: Everyday tasks (12a), chatting with a
model (12d) and phone models (12f). Added to today's layout, each would be
another tab. Today there are five tabs, a More menu with six more, a
Theme pill and a checks pill. masein approved a plan on 2026-09-24 that
organises the board around what people come to do:

| Place | The question it answers |
|---|---|
| **Home** | What needs me, and what changed? |
| **Models** | Which model is best at what? |
| **Playground** | What does this model actually say? *(arrives in 12d)* |
| **Improve** | How do we make this model better? |
| **Benchmarks** | What are we testing, with which questions? |

And every score belongs to one of **four kinds of test**, named the same,
in the same order, everywhere: **Standard** (the public benchmarks),
**Knowledge exam** (the 37 judged topics), **Everyday tasks** (12a), and
**On phone** (12f). A kind with no data yet is not shown at all.

## 1. Where everything goes

This table is the contract. **Every row must be reachable after the PR**,
and the tests check each one.

| Today | After |
|---|---|
| Overview | **Home**, rebuilt as three blocks (§6) |
| Leaderboard | **Models**, one table (§5) |
| Leaderboard ▸ Insights (score against size, frontier, weakest topics, compare shapes) | **Models**, below the table, as today |
| Leaderboard ▸ About these benchmarks | **Benchmarks ▸ Standard** |
| Models tab | **Models**, merged into the same table (§5) |
| Loop | **Improve ▸ By topic**, unchanged |
| More ▸ Review | **Improve ▸ Review**, unchanged, its four views kept |
| More ▸ Training | **Improve ▸ Training runs**, unchanged |
| More ▸ Exam | **Benchmarks ▸ Knowledge exam**, unchanged |
| Topic pages (`#topic=…`) | Stay pages; their back link reads **← Knowledge exam** |
| More ▸ Tasks | **Benchmarks ▸ Standard** |
| More ▸ Perplexity & Loss | **Models ▸ Standard ▸ Language modelling** (§5) |
| More ▸ Provenance ▸ Run provenance | The model page's **History** tab |
| More ▸ Provenance ▸ Query every metric, Export | **masein ▾ ▸ Data & sources**, a page |
| Queue | **● n running** in the header (§3), and an **All runs** page |
| Queue ▸ Submit a model | **Test a model** in the header (§3) |
| The `#everyday` pilot page (12a) | **Benchmarks ▸ Everyday tasks** |
| `7 checks ▾` | The **status dot** (§3) |
| Theme ▾ | **masein ▾ ▸ Theme** |
| 📖 guide for new users, the loop explained, How to read these numbers | **masein ▾ ▸ Help** |
| Model page: 01 Judged · 02 Results · 03 Diagnose · 04 Provenance · 05 Runs | Tabs **Scores · Answers · Improve · History** (§7) |

**Nothing is deleted.** If something on the board isn't in this table,
place it by the same logic and list it in the PR description.

## 2. The header

```
Team model benchmark   Home  Models  Improve  Benchmarks        ● 1 running   [Test a model]   ●   masein ▾
```

- Four places for now. **Playground** joins in 12d, between Models and
  Improve. Leave room; don't add a placeholder.
- **No More menu.** Nothing in the header opens a list of more places.
- The LIVE time badge stays beside the name.
- Below 720 px, the four places collapse into one **Menu ▾** button on the
  left, and the right side keeps the run counter, the main button and
  masein ▾. The header stays one line at 400 px.

## 3. The header's right side

**● n running** — a pill with a pulsing dot while anything runs; **Runs**
with a plain dot when nothing does. Clicking it opens a popover, the 11a
component, listing the running and queued runs with progress, then the
last five that finished, each linking to its model. At the bottom:
**All runs →**, a full page holding today's Queue table unchanged, with
its filters. The All runs page is not in the nav; it is reached from here
and from any "follow it →" link.

**Test a model** — the one filled button on every page. It opens today's
Submit form in a dialog, unchanged: 12c replaces its insides with the four
cards. From a model page it opens with that model filled in.

**The status dot** — replaces `7 checks ▾`. Green when every check
passes, and nothing else. Amber with a count when any doesn't: **● 2**.
Clicking it opens the checks list as a popover, with the **Show me** and
**why** links they have today.

**masein ▾** — holds **Theme**, **Data & sources** and **Help**, with
the name at the top. Nothing here is a place; it's settings and reference.

## 4. Benchmarks

One page with a switch at the top: **Standard · Knowledge exam · Everyday
tasks**.

- **Standard**: today's Tasks page, with About these benchmarks under it.
- **Knowledge exam**: today's Exam tab, unchanged, including Import a bank,
  the rubrics and criteria table, and Awaiting curation.
- **Everyday tasks**: the 12a pilot page, moved. Its **Run the pilot**
  button stays its one main action.

The switch remembers the last choice for the viewer.

## 5. Models

Leaderboard and the Models tab become **one table**.

- A switch at the top: **Standard · Knowledge exam · Everyday tasks**. It
  opens on **Standard**. It remembers the viewer's last choice.
- **Standard** keeps today's group chips — All, Knowledge, Commonsense,
  Reasoning, Math, Truthfulness — and adds **Language modelling**, which is
  today's Perplexity & Loss table. Its charts go under the table the way
  Insights does.
- **Knowledge exam** is today's Judged topics chip, promoted to the switch.
- **Everyday tasks** shows each model's pilot row from 12a: ✓/✗ per group
  and **n of 5**, with the **Pilot · not ranked** badge once in the header.
- The Models tab's extra facts — family, kind, last evaluated, flags — are
  columns under **Filters ▾**, off by default.
- **Models without a result in the current view** don't get a row of
  dashes. They sit under one collapsed line at the bottom: **Not tested on
  this (12) ▸**, each with a **Test** link. Preliminary models — a result,
  but not every required task — are shown with their result and a
  **preliminary** tag, unranked, exactly as the Leaderboard treats them
  today.
- **Clicking a row opens the model page.** The row expansion goes; the
  model page is where the detail lives.
- Kind, Size, Status, Columns, Models and Scale go into **Filters ▾**, as
  they already do on a phone.

## 6. Home

Three blocks, in this order. No hero paragraph and no stats line.

**Needs you.** One line per thing waiting, each a link: proposals waiting
for review, datasets made but not used in training, runs that failed in
the last seven days, and checks that aren't green. When there is nothing:
**Nothing needs you.**

**Running now.** The same list as the run counter's popover, full width.
When nothing runs: **Nothing running · Test a model**.

**Best in each kind of test.** One card per kind that has data: the best
model's number, its name on one line, and one link. Standard shows the
best average above chance; Knowledge exam the best judged topic average,
with the provisional-judge caveat once in the block header; Everyday tasks
the best **n of 5**, with the pilot badge. A kind with no data has no
card.

The old Overview's pieces that don't fit go where §1 says: Top models to
Models, the guides and "How to read these numbers" to Help, and Judge
steadiness to the checks behind the status dot.

## 7. The model page

**Header:** the name; one line of facts (size · instruct or base ·
family); the main action **Test this model**; and one tile per kind of
test — **Standard** (average above chance, and *7 of 7 tasks* under it),
**Knowledge exam** (judged average), **Everyday tasks** (**n of 5**). A
kind the model hasn't taken is a tile reading **Not tested · Test**.

**Tabs**, remembered per viewer:

| Tab | Holds | From today |
|---|---|---|
| **Scores** | One block per kind the model has taken, most recent open, the others collapsed | 01 Judged, 02 Results, 03 Diagnose, and 12a's Everyday block |
| **Answers** | What the model said, filtered by kind and topic or group; practice questions only | "The answers, topic by topic", and 12a's answers |
| **Improve** | This model's proposals and datasets — the Review lists filtered to it. **Shown only when there is at least one** | new, from existing data |
| **History** | This model's runs, run provenance, and how each was graded | 04 Provenance, 05 Runs |

Inside **Scores**:
- **Standard** holds today's Results. Today's **Diagnose** section is
  benchmark item analysis — it belongs here, folded under **What the score
  can't show ▸**. It is *not* the improvement loop; don't move it to
  Improve.
- **Knowledge exam** holds today's judged block. Score against answer
  length and Earlier exams fold under **More detail ▸**.
- **Everyday tasks** holds 12a's block as it is.

The **Chat** tab joins in 12d, between Answers and Improve.

## 8. Old links keep working

People have bookmarks, the demo run sheet and `HANDOFF.md` full of links.
Every old hash lands on its new home, with its sub-state kept:

| Old | Lands on |
|---|---|
| `#tab=overview` | Home |
| `#tab=leaderboard`, `#tab=models` | Models (Standard) |
| `#tab=loop` | Improve ▸ By topic |
| `#tab=review` (and its view) | Improve ▸ Review, same view |
| `#tab=training` | Improve ▸ Training runs |
| `#tab=queue` | All runs |
| `#tab=exam` | Benchmarks ▸ Knowledge exam |
| `#tab=tasks` | Benchmarks ▸ Standard |
| `#tab=perplexity` | Models ▸ Standard ▸ Language modelling |
| `#tab=provenance` | Data & sources |
| `#everyday` | Benchmarks ▸ Everyday tasks |

`#model=…`, `#topic=…` and every `read=…` keep working unchanged.

Update the links in `HANDOFF.md`, `DEMO.md` and `README.md` to the new
places.

## Not in this PR

- The Test-a-model cards (12c). The dialog holds today's form for now.
- The Playground and the Chat tab (12d), and saving a question (12e).
- Phone models and the **On phone** kind (12f).
- **Improve as one pipeline** — weak spots, proposals, training data and
  retests as four stages for one model. That is a later brief. Here,
  Improve is today's Loop, Review and Training, moved.
- **Comparing models side by side** from the Models table. Also later.

## Tests

- **The contract:** every "Today" row in §1 is reached from the new
  navigation, and every old hash in §8 lands where the table says.
- The header has exactly four places, no More menu, one filled button,
  and no `checks` pill. At 400 px it is one line.
- The status dot is green with no number when all checks pass, and amber
  with the count when two fail.
- **Nothing empty:** a model with no Knowledge exam result shows a *Not
  tested · Test* tile and no empty block; a view where twelve models have
  no result shows *Not tested on this (12) ▸*, not twelve rows of dashes.
- The model page's Improve tab is absent for a model with no proposals or
  datasets, and present for SmolLM2-360M-Instruct's fixtures.
- Home with nothing waiting says *Nothing needs you*.
- Clicking a Models row opens the model page; no row expands.
- Every existing test that navigates by the old tab names still passes
  through the redirects, or is updated to the new names; say which in the
  PR.
- 400 px and 1,512 px: no page scrolls sideways.

## Definition of done

On the live server:

1. The header reads **Home · Models · Improve · Benchmarks**, with the run
   counter, **Test a model**, the status dot and masein ▾.
2. Every item in §1 is reachable, and every old link in §8 lands.
3. **Models** is one table with the three-way switch, and a row opens the
   model page.
4. The **model page** has Scores · Answers · History, and Improve where
   there's something in it.
5. **Home** shows Needs you, Running now, and the best in each kind.
6. Training runs, Tasks, Perplexity & Loss and Insights are all where §1
   puts them.
7. **All tests are green**, with screenshots of each place at 1,512 and
   400 px in `tests/_screens/phase12b/`.

## Deploy steps, for masein, after this merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.
