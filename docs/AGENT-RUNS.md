# Agent runs on this server (18)

SWE-bench Multilingual and DeepSWE 1.1, run by Harbor 0.24.0 on local Docker,
with mini-swe-agent 2.4.6 as the agent and our own model through llama-server.
The design, the decisions and the sources: [AGENT-RUNS-design.md](AGENT-RUNS-design.md).
SWE-bench Pro follows once both have run.

## How a run works

- `scripts/agent_run.py` runs on the host, in the agent venv, under tmux. It
  never starts or stops a model server.
- Each task is one Harbor job.
  - The agent is ours: `scripts/agent_host_mini.py`, mini-swe-agent's own
    loop and config, run **beside** the task's container. It sends each
    command in with Harbor's exec.
  - The task's container has no network (`network_mode: none`), at most
    4,096 processes, and no address or key of the model. The agent checks
    this from inside before the model is asked anything.
- The agent reaches the model through the **relay**
  (`scripts/agent_relay.py`), in the runner's process:
  - it listens on 127.0.0.1 only and takes only the chat request;
  - it adds the server's key (read from the board's record, never written
    anywhere);
  - it sets the run's sampling: temperature 1.0, top_p 0.95, top_k 20, no
    presence penalty, thinking on.
- **Before the first task**, the runner refuses in one line, with what to do,
  if any of these fails:
  - the pinned versions;
  - Docker, as you without sudo;
  - 50 GB free where Docker keeps its images;
  - the board serving the model with the registered file;
  - a window of at least 131,072 tokens;
  - a tool call coming back through the relay.
- **A pilot is fixed:** `--tasks 10` is the same ten every time, spread over
  the benchmark's languages.
- **It carries on:**
  - a task with a result is never asked again;
  - an error of ours (Docker, the relay, the server down) is asked again,
    three times at most;
  - after a kill, the same command goes on.
- **Disk:** each task's image is removed once no waiting task needs it, its
  digest recorded first. Nothing more is pulled when Docker's disk would fall
  under 50 GB free: the task running finishes, and the run stops in one line.
- **On the board:**
  - the run is a Runs row, "this server", its line every 3 minutes ("37 of
    300 · 21 resolved · 28 min a task · about 5 days left"), imported at the
    end;
  - Benchmarks ▸ Agent tasks has a card each, with our score ("% resolved ±
    its error · N tasks"; a pilot says "pilot: 10 of 300") and the published
    numbers beside it, as reference;
  - a run's page lists its tasks, with the failures one click away; a task
    opens to its conversation, step by step, then its patch and the
    verifier's output.
- **Never exported:** conversations, patches and task text stay on the board,
  read from the run's folder under `$BENCH_ROOT/agent-runs/`, and are in no
  export.

## A. No model (masein, on the server)

1. Bring the checkout and the board up to date (the board shows agent runs
   from this build):
   ```bash
   cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
   ```
2. Make the agent venv. Harbor needs Python 3.12 or newer: if `python3
   --version` says older, make the venv with `uv venv --python 3.12
   ~/agent-venv` instead of the first command.
   ```bash
   python3 -m venv ~/agent-venv && ~/agent-venv/bin/pip install -q -r ~/benchmarks/aienh/docs/agent-requirements.txt
   ```
3. Let your user run Docker without sudo: Harbor runs `docker` as you. Then
   log out and in again.
   ```bash
   sudo usermod -aG docker $USER
   ```
4. See where Docker keeps images, the space there, and what Docker uses:
   ```bash
   sudo sh -c 'd=$(docker info -f "{{.DockerRootDir}}"); echo "$d"; df -h "$d"; docker system df'
   ```
5. Check a task's container reaches nothing. It must end "reaches nothing
   of: …":
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --check-reach
   ```
6. Three Multilingual tasks with Harbor's oracle agent, which applies each
   reference solution, no model. Each must read Resolved:
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --tasks 3
   ```
7. The same for DeepSWE. If a task errors with "Docker egress control was not
   enabled", the kernel lacks nftables' `fib` rules: say so before going on.
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py deepswe --oracle --tasks 3
   ```
8. Run step 4 again: the images are gone, and the disk is back.

## B. With the model (masein, after A is reported)

**What counts as broken.** Stop and report if any of these happens:
- fewer than 3 of the 10 resolved;
- any error of ours;
- a task's container reaching anything;
- no tool call coming back;
- thinking typed as a command;
- the window outgrown on more than 2 of the 10;
- a command left running past its limit.

1. Start the original with a 262,144 window. If the card can't hold it beside
   the judge, start it with the largest that fits, at least 131,072.
   ```bash
   CTX=262144 ~/lda-serve.sh
   ```
2. On the model's page, press "Use the new window" (17j). Then check the
   card's memory, with the judge's beside it:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```
3. The checks only, nothing asked. It must end "ready — 10 task(s)":
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --as served/Qwen3.6-35B-A3B-Q4-original-k-8 --tasks 10 --check
   ```
4. Ten SWE-bench Multilingual tasks, one attempt, in tmux:
   ```bash
   tmux new -s agent
   ```
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --as served/Qwen3.6-35B-A3B-Q4-original-k-8 --tasks 10
   ```
5. Report from the run's last line and its page on the board:
   - minutes a task, tokens, how many resolved, any error;
   - the disk used (step A4's command);
   - the card's memory (step B2's command);
   - anything odd in the conversations: thinking typed as a command, the
     window outgrown, a command that hung.

**C.** masein decides what runs next from B's time a task.
