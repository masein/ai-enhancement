# Agent runs — the design note (18, part 1)

What the three coding-agent benchmarks need, before any code: SWE-bench
Multilingual, DeepSWE 1.1 and SWE-bench Pro, on our server, with local Docker
and our own Qwen3.6-35B-A3B. Everything below was read on 8 Oct 2026. The
sources are at the end. Nothing was installed on this Mac: Harbor was only
installed in throwaway containers to read `--help`. No model was run and no
task, verifier or reference solution was downloaded.

## What the brief had that has changed

- **mini-swe-agent 2.x calls the model with tool calls**, one `bash` tool,
  not a command read from the reply's text. Reading a command from the
  text is the legacy mode.
- **Harbor's built-in mini-swe-agent runs inside the task's container.** It
  installs itself there from the internet, and the model's address and key
  are given to it there.
- **Harbor has no `LLM_BASE_URL`.** Its mini-swe-agent reads
  `OPENAI_BASE_URL` / `OPENAI_API_BASE`, and the key from `MSWEA_API_KEY` or
  the provider's key.
- **SWE-bench Pro V2 isn't on Harbor's Hub.** Its 642 tasks are Harbor task
  folders in Scale's repository; the Hub's `scale-ai/swe-bench-pro` is V1's
  731.
- **SWE-bench Multilingual is `swebench_multilingual@1.0`** in Harbor's older
  registry. It isn't on the Hub.
- **DeepSWE's agent limit is now 10,800 s a task** (since 26 Aug). Its paper
  says 9,000 s.
- **swebench.com's `multilingual-leaderboard.html` is gone.** The board is a
  tab of swebench.com.

## 1. One runner for all three?

**Yes: Harbor 0.24.0, local Docker (`-e docker`, its default).**

| Benchmark | How Harbor gets it | Tasks | Pin |
|---|---|---|---|
| SWE-bench Multilingual | `-d swebench_multilingual@1.0` (older registry) | 300 | the registry pins its tasks to `laude-institute/harbor-datasets` at commit `61366953…`; the HF dataset is `SWE-bench/SWE-bench_Multilingual` at `846e647b` (17 Aug 2026) |
| DeepSWE 1.1 | `-d datacurve/deep-swe-1-1@<digest>` (Hub), or `--repo datacurve-ai/deep-swe@<commit> -p tasks` | 113 | the digest or commit, read at install (the Hub page shows neither) |
| SWE-bench Pro V2 | `--repo scaleapi/SWE-bench_Pro-os@<v2.0.0 commit> -p v2/tasks` | 642 | the repo's commit; the HF dataset is `ScaleAI/SWE-bench_Pro` at `2d52cb3d` (22 Sep 2026) |

**Versions pinned:**
- Harbor 0.24.0 (PyPI, 5 Oct 2026; Python 3.12 or newer).
- mini-swe-agent 2.4.6 (PyPI, 23 Jul 2026), with the litellm it resolves to, recorded exactly.
- For a cross-check of Multilingual only: the SWE-bench harness, `swebench` 5.0.2 (18 Aug 2026).
- Every image is recorded by its digest when it is pulled.

All three are installed in one venv on the server; nothing is installed system-wide.

Multilingual could instead use mini-swe-agent's own batch runner
(`mini-extra swebench --subset multilingual --split test`) and the SWE-bench
harness to score it. That is exactly how the board's numbers were made. Harbor's
adapter scores with the same FAIL_TO_PASS / PASS_TO_PASS rule. **Harbor is the
one runner.** The harness is kept as a one-off cross-check on the pilot's ten
tasks: the same patches, scored both ways, must agree.

## 2. The agent

**mini-swe-agent 2.4.6, one `bash` tool, each benchmark's own prompt:**
- **Multilingual:** mini-swe-agent's `swebench.yaml`, the board's: 250 steps, 60 s a command, output over 10,000 characters cut to the first and last 5,000.
- **DeepSWE:** Datacurve's shared prompt and settings, from mini-swe-agent at the commit its board pins (`adfe2023`): no step or cost limit; the run ends when it submits, outgrows the window or times out.
- **Pro V2:** Scale's reference mini-swe-agent config, from its `v2/tooling`: no step limit.

**Where it runs (proposed): on the host, beside the task's container.** It is
driven by a thin Harbor agent of ours (an "external" agent, as Harbor's own
Terminus-2 is). That agent runs mini-swe-agent's loop and sends each command
into the container through Harbor's exec, as `docker exec … bash -c`. This is
how the Multilingual board ran it (mini-swe-agent's own runner, with
`docker exec`), and how Scale's V2 tooling runs it (`host_mini_swe.py`). Harbor's
built-in mini-swe-agent would instead need internet inside every container to
install itself, and the model's address and a key inside it.

**What it differs in:**

| | Qwen's number | The boards' numbers | Ours |
|---|---|---|---|
| Scaffold | Qwen's own, bash **and** a file-edit tool | Multilingual: mini-swe-agent v2.0, bash only. DeepSWE: mini-swe-agent `adfe2023` through Pier on Modal. Pro V1: SWE-agent, 250 turns. Pro V2: mixed agents, "locked protocol" | mini-swe-agent 2.4.6, bash only (files edited with `sed`, heredocs, `python`) |
| Window | 200K | each API's own | the one our server can hold (point 3) |
| Sampling | temperature 1.0, top_p 0.95 | each vendor's default | 1.0, 0.95, top_k 20, thinking on: the card's |
| Pro's set | Qwen's own corrected copy of V1's 731 | V1 731 / V2 642 | V2 642 |
| Attempts | not stated | Multilingual 1; DeepSWE 4; Pro 1 | 1 first |

## 3. How the agent reaches the model

**The address.**
- mini-swe-agent calls litellm with `model: openai/<the served name>` and
  `model_kwargs.api_base` set to the **relay** on the host (point 4). The relay
  then calls the team's llama-server at its registered address, with its key.
- The agent holds no key: it sends a dummy one. Because a model that costs
  nothing makes mini-swe-agent fail, it needs `cost_tracking: ignore_errors`.
- The agent runs beside the container, not in it, so the container has no
  address to reach.

**Tool calls or text.**
- It uses tool calls, which needs `--jinja`. masein's llama-server has it.
- The first check before a run sends one chat request with the `bash` tool
  and refuses the run unless it comes back as a parsed `tool_calls`. llama.cpp
  must parse Qwen3.6's tool-call format, and the team's fork is built from
  b6500.
- Text mode (`litellm_textbased`) is not used. When thinking arrives in the
  text without its opening tag, a command written while thinking can be run.

**Thinking between steps.**
- llama-server returns the thinking apart, as `reasoning_content`.
  mini-swe-agent keeps the whole message and **sends it back on every later
  step**.
- Whether it reaches the prompt is the chat template's choice. Qwen3's
  templates keep the thinking of every assistant turn after the last user
  message. In a tool loop that is every step, so the thinking stays in the
  context and the context grows fast.
- To be checked once, in step A: render a three-step conversation with
  llama-server's `/apply-template`.

**When the conversation outgrows the window.**
- llama-server refuses the request ("exceeds the available context size").
  litellm reads that as `ContextWindowExceededError`, and mini-swe-agent
  stops at once: no retry, nothing cut.
- The task ends with no patch: a failure, as DeepSWE and Epoch count it.
- **The window the runs need:** as large as the card and the server's memory
  allow.
  - Qwen ran 200K. DeepSWE's open models averaged about 80,000 tokens out over
    100–124 steps a task, with the thinking kept.
  - So 16,384, `lda-serve.sh`'s default, is far too small. 262,144 is the
    model's own maximum.
  - An estimate to check, not to rely on: this model keeps attention in about
    one layer in four. If its GGUF header says so, a 262,144 window costs a
    few GB of cache, which fits beside the judge's 13 GB.
- Step A reads the header and step B measures the card. The runner refuses a
  window smaller than the benchmark's minimum and names it. My proposal for
  that minimum is 131,072, with 200K preferred.

## 4. What a task's container can reach

**No route out** (as built).
- Each task's container runs with `network_mode: none` and `pids_limit:
  4096`, merged over the task's own compose file by Harbor
  (`--extra-docker-compose`, the run's `no-network.yaml`): it has no network
  interface but loopback, so no route to the host's other ports, the tailnet,
  the LAN or the internet. Harbor respects a network set on `main`.
- The agent is on the host, so the container needs no route during its run.
  Building an image (Multilingual's Dockerfile installs `curl` and `uv`) keeps
  Docker's build network: the benchmark's own set-up. Neither benchmark's
  tests need a network.
- **Before the model is asked anything**, the agent checks from inside the
  task's container: the model's server, the board, each of the host's
  addresses (the tailnet's among them), the LAN gateway, 1.1.1.1 and
  8.8.8.8 — the task is an error of ours, not run, if any answers, if the
  Docker socket is there, or if there is no process limit.
- **Step A** runs the same check in a container started as a task's
  (`agent_run.py … --check-reach`), with the server's own targets.
- **Test** (`tests/test_18_reach.py`, in CI): a host listener, the tailnet's
  own DNS address, the internet and a host folder, tried from inside — none
  answers; without the override the same container does reach the host.

**No host folder, no Docker socket.**
- Harbor mounts no Docker socket.
- It does bind-mount three empty folders of the trial's own into the
  container: `/logs/agent`, `/logs/verifier` and `/logs/artifacts`, under the
  job's folder, never `$HOME` or the repository.
- There is one risk in that: the model's commands could write
  `/logs/verifier/reward.txt` before the verifier runs. When the agent stops,
  every process it left is killed and any reward file it wrote is removed,
  before the verifier starts.

**Limits.**
- CPU and memory come from each task: Multilingual 4 CPUs and 8 GB, DeepSWE 2
  and 8 GB, Pro V2 1 and 4 GB.
- Harbor sets no process limit, so we add `pids_limit: 4096` through the extra
  compose file.
- **Test:** starting 5,000 processes is stopped.

**The key never inside a task's container.**
- A relay on the host listens on `127.0.0.1` only. It is the agent's one door
  to the model.
- It takes only `POST /v1/chat/completions`, with a size cap, and refuses
  everything else (`/v1/models`, `/props`, `/slots`, `/completion`).
- It adds llama-server's key, which it reads from the board's record of the
  served model when it starts and never prints.
- The agent, Harbor's `config.json`, the trajectories and the logs never see
  the key.
- **Test:** the key never reaches the agent's side, anything but the chat
  request is refused, and a search of the job folder for the key finds
  nothing.

## 5. Disk

**Where Docker keeps images on the server:**

```bash
sudo sh -c 'd=$(docker info -f "{{.DockerRootDir}}"); echo "$d"; df -h "$d"; docker system df'
```

The brief's 303 GB free is `/home`. Docker's root is often `/var/lib/docker`,
on another disk: this command says which.

**An image's size, compressed as the registries give it:**

| Benchmark | Size of one image | Source |
|---|---|---|
| Multilingual | 0.3–0.9 GB (four sampled) | Docker Hub |
| Pro | 0.5–2.9 GB (ten V1 images sampled); V2 not given | Docker Hub, ghcr |
| DeepSWE | not given | public.ecr.aws |

Unpacked, an image is usually two to three times its compressed size. Step A
measures the real size of three of each.

**The peak, for a run that deletes as it goes.** Each task's image is removed
after its verifier, unless a queued task uses it. So the peak is about the
tasks at once × the largest image, plus Harbor's job folder: about 5–10 GB
for Multilingual with one or two at a time. Kept, Multilingual's 300 images
would be about 0.5 TB, so they can't be kept.

**The guard.**
- Before each task, and every minute during one: if Docker's root would have
  less than 50 GB free after the next image, nothing more is pulled.
- The task running finishes, and the run stops with one line: what is free
  and what to remove.

## 6. Time

**Each benchmark's own limits for a task:**

| Benchmark | Agent | Verifier |
|---|---|---|
| Multilingual (Harbor's adapter) | 3,000 s | 3,000 s |
| DeepSWE | 10,800 s | 1,800 s (and 1,800 s to build) |
| Pro V2 | 3,000 s ("a 50-minute budget") | 3,000 s |

**What to expect on our server.**
- The model decodes about 85 tokens a second, one conversation at a time.
- DeepSWE's open models wrote about 80,000 tokens a task, which is about 16
  minutes of decoding alone, plus reading each step's new tokens.
- So a hard task may run into Multilingual's and Pro's 50 minutes, where the
  APIs those limits were set for don't.
- **The pilot keeps each benchmark's own limit and counts how many tasks hit
  it.** Harbor can scale a limit (`--agent-timeout-multiplier`), but that
  changes the protocol: masein decides after B, and the run records it.

**A second conversation at once** (`-np 2`, a window each) holds two caches
beside the judge's 13 GB. Most of the model's experts sit on the CPU, so two
conversations may share its speed rather than double it. B measures it: some
tasks one at a time, then the same number two at a time. Tokens a second,
minutes a task and the card's memory each way decide it, not a guess.

## 7. Scoring

**Each benchmark's own rule.**
- **Multilingual:** resolved means every FAIL_TO_PASS and PASS_TO_PASS test
  passes; the score is the share resolved, one attempt.
- **DeepSWE:** the verifier passes or fails a task. Its board runs the suite
  four times and gives the mean ± the run-to-run standard error (shown as a
  95% interval). Outgrowing the window and timing out count as failures;
  provider errors are left out.
- **Pro V2:** resolved, one attempt. The patch is graded again in a fresh
  image.

**What we run first:** one attempt (`-k 1`). DeepSWE's four runs come later,
if masein wants its board's ± (four times the time).

**What counts.**
- **A failure, counted against the model:** a wrong patch; the agent's time
  limit; the window outgrown; the step limit; three malformed replies in a
  row; stopping without submitting.
- **Ours, run again and never counted:**
  - a Docker failure (pull, build, start);
  - the relay or llama-server down, after mini-swe-agent's ten retries;
  - the disk guard's stop;
  - the verifier crashing or timing out;
  - a reward written before the verifier.

  They are run again with `harbor job resume -f <error types>`, and the
  runner says how many of each there were.

## 8. What our number can be put beside

| Benchmark | Model | Score | Who ran it | Agent | Set | Published | Source |
|---|---|---|---|---|---|---|---|
| Multilingual | **Qwen3.6-35B-A3B** | 67.2 | Qwen (vendor) | Qwen's own scaffold, bash and a file-edit tool, 200K, temperature 1.0 | 300 | 15 Apr 2026 (card last changed 24 Apr) | huggingface.co/Qwen/Qwen3.6-35B-A3B |
| Multilingual | Qwen3.5-35B-A3B / Qwen3.5-27B | 60.3 / 69.3 | Qwen (vendor) | the same | 300 | the same | the same |
| Multilingual | Gemini 3 Flash / Claude Opus 4.6 / GLM-5 / Kimi K2.5 / GPT-5.2 / DeepSeek V3.2 | 72.7 / 72.0 / 69.7 / 67.3 / 66.7 / 59.0 | the SWE-agent team (owner) | mini-swe-agent v2.0, bash only, 1 attempt | 300 | 13–20 Feb 2026 | swebench.com, Multilingual tab |
| DeepSWE 1.1 | Claude Opus 5 / GPT-6-astra / Gemini 3.8 Flash | 74±4 / 74±3 / 74±1 | Datacurve (owner) | mini-swe-agent `adfe2023` through Pier on Modal, 4 runs | 113 | updated 22 Sep 2026 | deepswe.datacurve.ai |
| DeepSWE 1.1 | GLM-5.3 / Kimi-K3 / DeepSeek-v4-flash / Qwen3.8-max (API) | 69±3 / 69±5 / 53±4 / 57±3 | Datacurve (owner) | the same | 113 | the same | the same |
| Pro | **Qwen3.6-35B-A3B** | 49.5 | Qwen (vendor) | Qwen's own scaffold | **Qwen's corrected copy of V1's 731** | 15 Apr 2026 | the model card |
| Pro (V1) | GPT-5.4 xHigh / Claude Opus 4.6 / Qwen3-Coder-480B / Qwen3-235B / DeepSeek V3.2 | 59.1 / 51.9 / 38.7±3.6 / 21.4 / 15.6 | Scale (owner) | SWE-agent, 250 turns (* mini-swe-agent) | 731 | Jan–Apr 2026 | labs.scale.com/leaderboard/swe_bench_pro_public |

**What this means.**
- **Nobody but Qwen has published a number for Qwen3.6-35B-A3B** on any of
  the three. Ours would be the first independent one.
- **DeepSWE has no number of Qwen's at all.**
- **Multilingual is the cleanest comparison.** Our agent and settings are
  the board's, beside Qwen's 67.2 on Qwen's own scaffold.
- **Pro is the weakest.**
  - Qwen's 49.5 is on its own corrected V1 set, which is neither board's.
  - V2's board (22 Sep, 642 tasks) shows 89.9–99.4 for every model: a
    measure I couldn't confirm, so not to be put beside ours until it is.
  - masein's choice: V2, as the brief says, or V1's 731, which Scale's V1
    board and Qwen's number are closer to.

## What masein decided (8 Oct)

1. **The agent on the host**, as proposed (§ 2): `scripts/agent_host_mini.py`.
2. **SWE-bench Multilingual and DeepSWE now; SWE-bench Pro once both have
   run** — on the set the numbers we compare with were run on (below).
3. **Each benchmark's own time limits.** The pilot counts the tasks that hit
   them.
4. **A 262,144 window** if it fits beside the judge with one conversation;
   the runner refuses anything under 131,072.
5. **Temperature 1.0, top_p 0.95, top_k 20, no presence penalty**: Qwen's own
   settings for its SWE-bench numbers (a presence penalty punishes code for
   repeating its own names). Thinking on.

Built since (docs/AGENT-RUNS.md): DeepSWE's tasks are taken at a commit of
their repository (`datacurve-ai/deep-swe@0b9fabbb63b9`, 26 Aug 2026), since
Harbor's Hub shows no revision to pin; DeepSWE's board runs mini-swe-agent's
default `mini.yaml` (Pier's mini-swe-agent: no cost limit, the task's
instruction as the task, `sh -c` and 30 s a command), and the commit it pins
(2026-05-21) differs from 2.4.6 in one error template only.

## SWE-bench Pro: which set the published numbers are on (read 8 Oct)

- **Scale's public board** (V1, the 731-task public set; the board's own
  page, as archived on 14 Sep 2026): GPT-5.4 (xHigh) 59.1 ± 3.6, Claude Opus
  4.6 (thinking) 51.9 ± 3.6, Gemini 3.1 Pro (thinking) 46.1 ± 3.6, all added 8
  Apr 2026, all marked * — run with **mini-swe-agent**, uncapped cost and a
  **250-turn limit** (the grey rows of the old board had 50 turns and a cost
  cap). Not stated: the mini-swe-agent version and config (Scale's repository
  pins its fork at v1.15.0, Nov 2025, whose SWE-bench config has 250 steps),
  the number of attempts (pass@1 by every other sign), whether 731 or 730
  tasks were scored, and who ran them (no vendor mark; likely Scale).
- **Qwen's 49.5** for Qwen3.6-35B-A3B: Qwen's own scaffold (bash and a
  file-edit tool), temperature 1.0, top_p 0.95, 200K window, on **Qwen's own
  corrected copy of the public set** — which tasks it corrected isn't
  published (asked in QwenLM/Qwen3.6 #179, no answer). On that copy Qwen gives
  Claude 4.5 Opus 57.1, where Scale's V1 board gives 45.9: Qwen's setup reads
  about 11 points higher.
- **V2** (642 tasks, 22 Sep 2026) is % resolved on a locked protocol, near
  its ceiling (89.9–99.4); none of the three models above has a V2 number.

**So the set that matches is V1's public 731, with mini-swe-agent at 250
steps**: Scale's three numbers are then beside ours on the same set and the
same agent family. Qwen's 49.5 matches neither set nor scaffold, and is shown
as Qwen's, labelled so. When Pro is built: V1 (`ScaleAI/SWE-bench_Pro`,
config `v1`), Harbor's `swebenchpro@1.0`, 250 steps.

## Sources (read 8 Oct 2026)

- **Harbor:**
  - Source code, tag v0.24.0: `src/harbor/agents/installed/mini_swe_agent.py`, `agents/model_connection.py`, `environments/docker/`, `models/task/config.py`, `models/trial/paths.py`, `job.py`, `cli/jobs.py`, `registry.json`.
  - Docs: `docs-mintlify/agents/custom-agents.mdx`, `tasks/network-policies.mdx`.
  - pypi.org/pypi/harbor/json; hub.harborframework.com/datasets.
- **mini-swe-agent:**
  - v2.4.6: `src/minisweagent/models/litellm_model.py`, `config/benchmarks/swebench.yaml`, `agents/default.py`, `environments/docker.py`, `run/benchmarks/swebench.py`.
  - mini-swe-agent.com/latest/models/local_models.
- **SWE-bench Multilingual:**
  - swebench.com/multilingual.html; swebench.com (leaderboard data).
  - huggingface.co/api/datasets/SWE-bench/SWE-bench_Multilingual.
  - github.com/SWE-bench/SWE-bench; pypi.org/project/swebench.
  - hub.docker.com/v2/repositories/swebench/….
- **DeepSWE:**
  - deepswe.datacurve.ai (with the blog posts for 1.0 and 1.1).
  - github.com/datacurve-ai/deep-swe (a task's `task.toml`).
  - arxiv.org/html/2607.07946; epoch.ai/benchmarks/deepswe.
- **SWE-bench Pro:**
  - github.com/scaleapi/SWE-bench_Pro-os (README, `v2/README.md`, `v2/GATE.md`).
  - huggingface.co/api/datasets/ScaleAI/SWE-bench_Pro; arxiv.org/html/2509.16941.
  - labs.scale.com/leaderboard/swe_bench_pro_public and `…_public_v2`, read through a reader: Scale's site refused a direct read.
- **Qwen:** huggingface.co/Qwen/Qwen3.6-35B-A3B (README); huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF.
- **Others:** vals.ai; Epoch's data export; Artificial Analysis. None has a Qwen3.6-35B-A3B number on these three.

**Not confirmed:**
- the Hub digests of `datacurve/deep-swe-1-1` and `scale-ai/swe-bench-pro`;
- whether Harbor clears a reward file the agent wrote;
- how Qwen3.6's template renders earlier thinking;
- whether the team's llama-server parses Qwen3.6's tool calls;
- the sizes of Pro V2's and DeepSWE's images;
- Scale V2's metric.

Each is settled in step A or in the build, before it is relied on.
