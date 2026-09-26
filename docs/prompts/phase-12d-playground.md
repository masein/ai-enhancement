# Brief for Claude Code — the Playground and the Chat tab (12d)

Start from main after **#81**, or later.

There are two PRs:
- **12d.1** covers the chat engine, the Playground page and practice questions (§1–§6).
- **12d.2** covers comparing two models, the model page's Chat tab and trained models (§7–§9).

A PR is done when:
- `scripts/check.sh` passes on its branch head;
- its summary line is in the PR under "Local check";
- deploy step 4 passes on the server.

**No model runs on the Mac or in CI.** Test with a fake model backend that streams canned text. Every real chat happens on the server, after deploy.

**The Playground never spends money.** It chats with the models on this server only. OpenRouter models aren't offered here: no picker entry, no API call.

The plain-words rules and the display rules from 12b apply unchanged:
- one number first;
- nothing empty;
- no system words;
- a caveat once per page;
- one main action;
- numbers in mono, words in sans.

---

## Why

The board scores models but never lets you *see* one talk. Most questions about a small model can only be answered by reading it: "is Qwen3-0.6B really that bad at summaries?", "what does the trained one do differently?", "does it refuse this?". Today that means SSH, a notebook and a GPU fight.

12b set the place aside. **Playground** answers *"What does this model actually say?"* It sits in the header between Models and Improve. The model page gets a **Chat** tab between Answers and Improve.

---

# 12d.1: the engine, the page, practice questions

## 1. The chat engine

This is a chat worker inside the bench container, next to the service.

- **The loader is the one the runs use.** It uses the same code path as runs, so the same things apply:
  - the approved-code rule: code a model runs itself is used only at its approved commit;
  - the kernels from 12a.5b;
  - the dtype choice;
  - the chat template.

  Don't write a second loader.
- **Replies stream.** Stream them token by token to the browser over server-sent events (or a WebSocket if it fits the app better). Use transformers with a streamer. vLLM's start-up time is wrong for "pick a model and say hi".
- **Only one reply is generated per loaded model at a time.** A second person's message waits, and their page says "answering another message, yours is next".
- **Stop** ends generation at once and frees the worker. Closing the tab does the same.
- **Loaded models unload after 10 idle minutes.** Make this a config value. The header's run popover lists a loaded chat model as one quiet line: "Playground: Qwen3-1.7B loaded".
- **Thinking models:**
  - the thinking streams into a folded **Thinking ▸** block above the reply, closed by default;
  - the reply's own text is what's shown and copied;
  - the reply budget matches the scored runs: `EVERYDAY_REASONING_MAX_GEN_TOKS` for a reasoning model, and the Everyday budget otherwise.

## 2. Runs come first

The GPU is shared by three things: runs (the lock in `service/runner.py`), the local judge's vLLM, and now chat. **A run is never slowed or broken by a chat.**

- **While a run holds the lock, chat doesn't load a model on the GPU.** The page says, in one line: "The GPU is running Qwen3.5-2B's Standard tests, about 40 min left. Chat starts when it's done." It uses the run's own time-left estimate. With no estimate it says "when it's done". The page retries on its own and notices when the GPU is free.
- **Small models can use the CPU in the meantime.** A model under 1B parameters (the SmolLM2s, gemma-3-270m, Qwen3-0.6B) can answer on the CPU while the GPU is busy. It's slower, and the page says so once in a grey line: "on the CPU while a run uses the GPU — slower". Measure it on the server. If a 0.6B model is under about 3 words a second there, drop the threshold and say so in the PR.
- **When a run is about to take the lock:**
  - the runner asks the chat worker to unload its GPU models, and waits up to 60 s;
  - an open chat shows "Paused: a run started. Your conversation is kept." and continues on the CPU if the model qualifies;
  - a reply in progress is stopped, and keeps what it wrote, marked "cut short".
- **Before loading on the GPU, check free memory.** Use `torch.cuda.mem_get_info`, and require the weights' size plus a margin. The margin is a config value: measure it and put the number in the PR. If it doesn't fit, say so in one line. Don't crash.
- The chat worker **never takes the run lock**, and a run never waits for chat.

## 3. The Playground page

`#tab=playground`, in the header between **Models** and **Improve**. Below 720 px it is in the Menu ▾ list too.

```
┌ Chats ─────────────┐ ┌──────────────────────────────────────────────────────┐
│ + New chat          │ │ Qwen3-1.7B ▾                         Settings ▸      │
│                     │ │                                                      │
│ Today               │ │   (empty: "Ask anything, or try a practice           │
│  shorten the park…  │ │    question ▾")                                      │
│  json from text     │ │                                                      │
│ Earlier             │ │   you  ›  can u make this shorter: "…"               │
│  …                  │ │   Qwen3-1.7B  ›  The picnic is at 4 pm Saturday…     │
│                     │ │                   Thinking ▸   copy · again           │
│                     │ │                                                      │
│                     │ │ [ message…                                   ] Send  │
└─────────────────────┘ └──────────────────────────────────────────────────────┘
```

- **The model picker** lists the models on the board with a chat template, by name, grouped as on Models. Base models without a template are left out, and a line under the list says so: "Base models aren't listed: they have no chat format."
- **Chats are kept on the server**, per person, by the name in masein ▾.
  - Each chat stores the model and its pinned commit, the settings, and every message with its time.
  - A chat's title is the first few words of its first message.
  - The list shows the person's last 50 chats. There's a **Delete** on each, which asks once.
  - Other people's chats are never listed.
- **The empty state is one line and one button,** not a wall of tips.
- **Under each reply,** in small grey mono: its length, time and words per second. Also **copy** and **again**. "Again" re-asks the same message and keeps both replies, shown as ‹ 1 of 2 ›.
- **Long input:** if a message plus the reply budget won't fit the model's context, say so before sending: "Too long for this model: about 1,900 words fits."
- On a phone, the chat list folds into a **Chats ▾** button at the top.

## 4. Settings

**Settings ▸** is folded, top right of the chat.

- **By default the chat uses exactly the settings the Everyday run uses:**
  - the chat template;
  - thinking on or off, as the model was scored;
  - the reply budget;
  - sampling;
  - no system message.

  Read them from **one shared function** that the Everyday run also calls, so the two can't drift. If there isn't one today, make one and have the run use it. "What you see is what was scored" is the point of the page.
- **The fields you can change:**
  - a **system message**;
  - **thinking** on or off (reasoning models only);
  - **temperature**;
  - **longest reply**.
- **Once anything differs from the scored settings,** one amber line shows above the input: "Not the scored settings." Next to it, **reset** puts them back.
- The settings are stored with the chat. A chat reopened later uses its own settings.

## 5. Try a practice question

**Try a practice question ▾** is next to the input, and in the empty state.

- **It lists practice questions only**, from Everyday tasks by group and from the Knowledge exam by topic. **Hidden questions are never offered, sent to the page or reachable through the API.** A test checks this at the API level.
- Picking one puts its prompt in the input, unsent, so it can be edited.
- **If it's sent unedited, on the scored settings, as the chat's first message,** the reply is marked the way the run marks it:
  - Everyday questions get the script checks, in the same plain words as 12a.5 §4: "✓ passes" or "✗ needs 4 of: the date, the time, …";
  - Knowledge questions show **Reference answer ▸**, folded.

  This one is **not judged**: judging calls the judge, and the Playground stays free. Say so in the fold: "not marked here; the exam's judge marks it in runs".
- If the prompt was edited or the settings changed, nothing is marked, and the reply says why in grey: "edited, so not marked".
- A marked reply here is **never stored as a score** and never touches the board's numbers.

## 6. 12d.1 tests

- **Fake backend:** it streams, Stop cuts it off, and closing the connection frees the worker.
- **Lock:**
  - with the run lock held, a GPU load is refused with the one-line reason;
  - a model under the CPU threshold answers on the CPU;
  - a run taking the lock unloads chat within 60 s (use a fake clock);
  - a run is never delayed by chat.
- Idle unload after the configured time (fake clock).
- **Hidden questions** are never returned by any Playground endpoint, whatever the parameters.
- **Settings:**
  - defaults equal the Everyday run's (same function);
  - a changed setting shows the amber line;
  - a chat reopens with its own settings.
- **Chats:** per person; delete asks once; one person can't read or delete another's by id.
- A reasoning model's thinking goes into the fold, and copy gives the reply without it.
- A base model isn't in the picker, and the line says why.
- Header order is Home · Models · Playground · Improve · Benchmarks. It stays one line at 400 px.

---

# 12d.2: side by side, the Chat tab, trained models

## 7. Compare two models

- **+ Compare** next to the model picker adds a second model. One message goes to both, and the replies sit side by side (stacked on a phone), each labelled with its model.
- **If both fit in free memory they answer at once; otherwise one after the other.** The second one's column says "waiting for the GPU".
- Two models at most. Stop stops both.
- In the chat list a compared chat shows both names: "Qwen3-0.6B vs SmolLM2-360M".
- **Try a practice question** marks both replies, on the same terms as §5.

## 8. The model page's Chat tab

- **Tabs:** Scores · Answers · **Chat** · Improve · History.
- **It is the same component** as the Playground, with the model fixed. It shows this person's recent chats with this model, and a link, **Open in Playground →**.
- Base models show one line instead of a chat: "This is a base model: it has no chat format, so there's nothing to chat with." Show the tab anyway, so the tabs don't move between models.
- In **Answers**, each practice answer gets **Ask it again ▸**. It opens the Chat tab with that question in the input, unsent.

## 9. Trained models

- **Models trained in Improve** (with "Trained from" set) are in the picker, under their base model, labelled "trained · <date>".
- **Compare** suggests the base model as the second one: "Compare with Qwen3-1.7B (before training)". This is the fastest way to see what training changed.
- They load through the same loader, adapter included.

## 12d.2 tests

- **Compare with the fake backend:** both columns stream; the sequential fallback when memory is short; Stop stops both.
- The Chat tab shows on every model page, with the base-model line where it applies.
- Ask it again pre-fills the question and doesn't send it.
- A trained model is under its base, and Compare suggests that base.

---

## Not in these PRs

- **Saving a question from a chat into a bank (12e).** Leave a place for it under each reply, but don't add a button.
- **Chatting with OpenRouter models.** They cost money; the AI models page is where they live.
- **Phone models and On phone (12f).**
- **Sharing a chat by link.** This comes later if people ask.

## Done when

1. On the server, Playground chats with every instruct model on the board, streaming, with thinking folded for the reasoning ones.
2. During a run, the GPU stays the run's. The small models answer on the CPU, and the rest say when they'll be back.
3. The default settings are the scored settings, from one shared function.
4. A practice Everyday question sent unedited shows the same pass or fail the run would give. No hidden question can be reached.
5. Two models answer side by side, and a trained model can be compared with its base.
6. Each PR has its Local check line, and deploy step 4 passes.

## Deploy steps, for masein, after each merges

As in HANDOFF.md § 5b. If the image doesn't change (no new package), skip step 3 and say so in the PR.

After 12d.1, put the number from §2 in the PR: the CPU speed of the 0.6B model and the memory margin you measured. Then Claude will live-check the page.
