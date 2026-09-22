# Brief for Claude Code — the 11f–11j live check (11k)

Start from main after 11j. This is **one PR, 11k**, and it must be green on
its own.

masein and I went through the board together on the evening of 2026-09-22,
after 11f–11j were deployed. Everything in those four PRs works. These are
the nine things we found, in the order they matter.

**The rules that do not bend are unchanged.** The plain-words rules from
11h §7 apply here too.

---

## 1. The Propose button leads nowhere

**Seen.** On the model page's judged table, every row has **Propose →**. It
links to the topic page, with the tooltip "proposing happens on the topic
page, where the answers are". But 11j moved proposing into the Review tab's
**+ New proposal** dialog, so the topic page has no Propose button any
more. The person lands on a page with nothing to do.

masein: *"when I tap on the propose in the models page it goes to the topic
page but I don't see how to propose here. Maybe there should not be a
propose button."*

**Fix.**
- **Propose → opens the New proposal dialog**, with this model and this
  topic already filled in, from wherever it is: the model page, the Loop
  board and the topic page.
- The dialog opens over the page the person is on. It does not navigate.
- When a proposal for that model and topic is already open, the row shows
  **Review it →** instead, linking to that proposal.
- Drop the old tooltip.

**Also: tooltips run off the screen.** That tooltip was cut off at the right
edge of the window. Every tooltip flips side, and is clamped 8 px inside the
viewport, like the popovers in 11a.

**Tests.**
- Propose → opens the dialog with both fields set, and the hash still points
  at the model page.
- With an open proposal, the row reads Review it → and opens that proposal.
- A tooltip on the right-hand edge stays inside the window.

## 2. A run says `done` while it is still being graded

**Seen.** Run #58 (pythia-31m, 3 topics + MMLU control): the status chip said
**done** while the Progress cell said "judge batch … submitted (570
answers) · judging 240/570", and the action cell was empty. It looks as if
nothing happened. The GPU part had taken 66 seconds; grading took several
minutes more.

**Fix.**
- **The status chip tells the true stage:**
  - `queued` → `running 2/4` → **`grading 240/570`** → `done` | `failed`.
  - `done` appears only once `judge.json` is written, or for a run with no
    judged half.
- **The action cell is never empty.** While grading, it shows a quiet
  **Grading… 240/570** chip in place of the button. **Open results**
  replaces it when the grades land.
- The `⋯` menu keeps Log throughout.

**Tests.**
- A fixture mid-grading shows `grading k/n` and the chip, not `done` and an
  empty cell.
- When judge.json lands, the row shows `done` and Open results.

## 3. The dataset reader's "The missing skill" box is empty

**Seen.** In the Reader for dataset #6, **The missing skill ▸** is an empty
grey box. The spec is in the data — `provenance.approved_spec`, 341
characters for that dataset — but nothing is rendered. The box is about
127 px tall whether it is open or closed.

**Fix.**
- Render the approved spec inside it. Fall back to `spec_text`, and when
  neither exists say "no spec recorded".
- Closed, the disclosure is only as tall as its summary.

**Test.** The Arts fixture's reader shows the spec's first words, and the
closed disclosure is under 40 px tall.

## 4. The Queue table spills over its card

**Seen.** At 1,512 px: the card is 1,276 px wide, its scroller 1,226 px, and
the table 1,232 px, with `overflow-x: visible`. The action buttons and their
white background paint across the card's right edge.

**Fix.** The scroller clips and scrolls (`overflow-x: auto`), as the
Leaderboard's does, and the pinned columns' backgrounds stay inside it.

**Test.** At 1,280, 1,440, 1,512 and 1,920 px, the table's right edge is
inside its scroller on the Queue, the Models tab and the Review lists.

## 5. The page still scrolls sideways on a phone

**Seen.** At 400 px, the document is 414 px wide, so the whole page slides
about 14 px.

**Fix.** Nothing overflows the window at 400 px.

**Test.** `document.documentElement.scrollWidth <= window.innerWidth` at 400
px, on Overview, Loop, Models, Leaderboard, Queue, Review, a topic page and
a model page.

## 6. "0 of 5" in Sit the exam

**Seen.** In the Sit-the-exam panel, each area heading reads "0 of 5", which
reads as "0 of 5 judged" but means "0 of 5 ticked".

**Fix.** The heading says both, in plain words: **"0 ticked · 2 of 5
judged"**.

**Test.** The heading's numbers follow the tick boxes and the model's judged
topics.

## 7. The answers list is one endless scroll

**Seen.** In the proposal card, **The answers it read (33) ▸** opens to
about 14,000 px in one scroll, because every answer is shown in full.

**Fix.**
- Page it: 10 at a time, with the pager already used elsewhere.
- Each answer is clamped to about 3 lines, with **Show the whole answer**,
  as the topic page already does.
- The question and the judge's comment stay in full.

**Test.** The list shows 10 with a pager, and each answer is clamped until
asked.

## 8. The checks pill keeps a heavy focus ring

**Seen.** After clicking `● 6 checks ▾`, a thick blue ring stays around it,
and sits awkwardly beside **Sit the exam**.

**Fix.** The standard ring: 2 px accent, 2 px offset, and only on
`:focus-visible`, so a mouse click leaves no ring. Every pill and button in
the bar behaves the same way.

**Test.** A mouse click leaves no ring; a Tab focus shows one.

## 9. The dropdown's selected row looks like a text field

**Seen.** In the Weakest-topics model list, the highlighted option carries a
heavy blue bottom border.

**Fix.**
- The selected option: `--accent-soft` background and a tick at its right.
- The keyboard-active option: the same background plus the focus ring.
- No borders or underlines inside a list.

**Test.** No option has a border, and the selected one has a tick.

---

## Not in this PR

**Reading the hidden questions.** masein asked to read every question of a
topic, not only the practice half. The hidden half stays hidden for now: it
is what makes the published score mean anything. The files are in the repo
for an author who needs them. If masein later wants an "author view" in the
page, it will be its own brief, and it will record who read what, and mark
those scores.

## Definition of done

On the live server:

1. **Propose →** on the model page opens the New proposal dialog, filled in.
   Where a proposal already exists, the row says Review it →.
2. **A judged run** shows `grading k/n` while the judge works, and a
   **Grading…** chip in place of its button.
3. **Dataset #6's reader** shows the missing skill.
4. **No table** paints over its card at 1,280–1,920 px.
5. **No page** scrolls sideways at 400 px.
6. **Sit the exam** headings read "0 ticked · 2 of 5 judged".
7. **The answers list** comes in pages of 10, with answers clamped.
8. **A mouse click** leaves no focus ring in the bar.
9. **Dropdown options** have no underline, and the selected one has a tick.
10. **All tests are green**, with screenshots in `tests/_screens/phase11k/`.

## Deploy steps, for masein, after this merges

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
sudo docker compose logs --since 2m bench | grep -iE "error|traceback" || echo "no errors"
```

Put these, with the expected output of each, in the PR description and in
`HANDOFF.md`.
