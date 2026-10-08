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
    this from inside before the model is asked anything, and the check
    fails closed: it must show it can try a connection at all, nothing may
    answer (by IPv4, IPv6 or name), and the loopback must be its only
    interface.
  - Harbor mounts three of the trial's folders into the container
    (`agent/`, `verifier/`, `artifacts/`), and the model's commands run
    there as root: anything in them may be a link, a FIFO or a device it
    made. The agent's own files (its trajectory, the patch, its counts) go
    in the trial's `agent-host/` folder instead, which no container mounts,
    each written under a new name and renamed into place, never through a
    link. However the agent stops, every process left in the container is
    killed and the three mounted folders are emptied before the verifier
    runs.
  - The board reads a file only if it is a regular file with one link,
    reached from the run's folder through real folders: never a link, a
    FIFO, a device or a hard link, and at most 2 MB of it.
- The agent reaches the model through the **relay**
  (`scripts/agent_relay.py`), in the runner's process:
  - it listens on 127.0.0.1 only and takes only the chat request;
  - it adds the server's key (read from the board's record, never written
    anywhere);
  - it sets the run's sampling: temperature 1.0, top_p 0.95, top_k 20, no
    presence penalty, thinking on.
- **Before the first task**, the runner refuses in one line, with what to do,
  if any of these fails:
  - the pinned versions, and every package in the venv as the lock
    (`docs/agent-requirements.txt`, with hashes) has it;
  - Docker, as you without sudo;
  - 50 GB free where Docker keeps its images;
  - the board serving the model with the registered file;
  - a window of at least 131,072 tokens;
  - a tool call coming back through the relay.
- **A pilot is fixed:** `--tasks 10` is the same ten every time, spread over
  the benchmark's languages. `--only a,b` asks those tasks; `--only
  fetching` the tasks whose own tests fetch packages.
- **Verification has no network either:** it runs the model's code, and the
  board takes writes from the tailnet without a key. The runner asks each
  task from a copy of it (`agent-tasks/<benchmark>+offline/`) whose
  verification needs none:
  - SWE-bench Multilingual verifies in the agent's container, and its
    `test.sh` has uv fetch the SWE-bench parser's packages and a Python
    3.11+ from the internet. The copy's Dockerfile runs that script's header
    once while the image is built, then tells uv it is offline. The few
    tasks whose own test commands run a package manager have those commands
    run once at build time, in a throwaway copy of the repository, and the
    tools told they are offline;
  - DeepSWE verifies in a separate container, offline by its own design;
    Harbor gives it none of the runner's compose files, so the copy gives it
    one: no network, the agent's limits.
- **It carries on:**
  - a task with the model's result is never asked again, whatever the
    result: a failure is the model's result, and a later success never
    replaces it;
  - an error of ours is asked again, three times at most. Ours is a short
    list: Docker failing to pull, build or start the task's container; the
    container reaching something; the model's server failing its health
    check (the relay asks `/health` when the server doesn't answer). One bad
    reply from a server that is up is the model's;
  - every task asked is in the score's denominator: one given up after three
    errors of ours counts as not resolved, and the score says how many;
  - the run's page says how many tasks were asked again, and why;
  - after a kill, the same command goes on.
- **The board's runs are left alone:** the runner's calls into the board
  (its checks, its line every 3 minutes, the import) touch only its own
  Runs row. Only the board's own start puts runs that were running back in
  the queue.
- **Disk:** each task's image is removed once no waiting task needs it, its
  digest recorded first. Nothing more is pulled when Docker's disk would fall
  under 50 GB free: the task running finishes, and the run stops in one line.
- **On the board:**
  - the run is a Runs row, "this server", its line every 3 minutes ("37 of
    300 · 21 resolved · 28 min a task · about 5 days left"), imported at the
    end;
  - Benchmarks ▸ Agent tasks has a card each, with our score ("% resolved ±
    one standard error · N tasks"; a pilot says "pilot: 10 of 300") and the
    published numbers beside it, as reference. A task counts as resolved
    only when the agent submitted and the tests passed: a working tree it
    never submitted (the step limit, the window outgrown) is not resolved,
    as mini-swe-agent's own numbers count it;
  - a run's page lists its tasks, with the failures one click away; a task
    opens to its conversation, step by step, then its patch and the
    verifier's output.
- **Never exported:** conversations, patches and task text stay on the board,
  read from the run's folder under `$BENCH_ROOT/agent-runs/`, and are in no
  export.

## A. No model (masein, on the server)

Each step is numbered; a step that refers to another gives its number.

1. Bring the checkout and the board up to date (the board shows agent runs
   from this build):
   ```bash
   cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
   ```
2. Install uv 0.12.18 for your user, checked against its published
   checksum. It must print `uv-x86_64-unknown-linux-gnu.tar.gz: OK`, then
   `uv 0.12.18`:
   ```bash
   cd /tmp && curl -fsSLO https://github.com/astral-sh/uv/releases/download/0.12.18/uv-x86_64-unknown-linux-gnu.tar.gz && echo "89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16  uv-x86_64-unknown-linux-gnu.tar.gz" | sha256sum -c && tar xzf uv-x86_64-unknown-linux-gnu.tar.gz && mkdir -p ~/.local/bin && install -m 755 uv-x86_64-unknown-linux-gnu/uv ~/.local/bin/uv && ~/.local/bin/uv --version
   ```
3. Make the agent venv with Python 3.12 — uv fetches Python 3.12 if the
   server has none — and pip in it (`--seed`):
   ```bash
   ~/.local/bin/uv venv --python 3.12 --seed ~/agent-venv
   ```
4. Install the lock, every package checked against its hash. It must end
   `Successfully installed …` with no error:
   ```bash
   ~/agent-venv/bin/pip install --require-hashes --no-deps -r ~/benchmarks/aienh/docs/agent-requirements.txt
   ```
5. **Docker for the runner: you choose.** Harbor runs `docker` as the user
   who runs the runner, and whoever can run `docker` is root on this server
   (a container can mount `/` and write anywhere). Both ways make the
   runner, and every package it imports, root:
   - **(a) The docker group.** Your account is root without a password from
     then on, for everything you run — every package in every venv you use,
     this one's 103 included, and any script you run. It stays until you
     take it away (`sudo gpasswd -d $USER docker`). Then log out and in:
     ```bash
     sudo usermod -aG docker $USER
     ```
   - **(b) sudo, each time.** Nothing changes for your account: only the
     runner is root, and only while it runs. Put `sudo` in front of
     `~/agent-venv/bin/python` in steps 7–10 and in § B. What that changes:
     the run's files under `$BENCH_ROOT/agent-runs/` and `agent-tasks/` are
     owned by root (the board reads them as before); Harbor's cache is
     under `/root/.cache`; the Runs row still says masein (the runner takes
     the name sudo was run by; `--by masein` says it outright).
6. See where Docker keeps images, the space there, and what Docker uses:
   ```bash
   sudo sh -c 'd=$(docker info -f "{{.DockerRootDir}}"); echo "$d"; df -h "$d"; docker system df'
   ```
7. Check a task's container reaches nothing: the model's server, the board,
   the relay and ssh on every address of this server (the tailnet's and
   IPv6 among them) and on Docker's gateway, the LAN, Docker's DNS, the
   tailnet's DNS, the internet over IPv4 and IPv6, any name. It must end
   "… no host folder; at most 4,096 processes"; anything else is a refusal
   that says what answered or what the check couldn't do:
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --check-reach
   ```
8. The ten Multilingual tasks of step B, with Harbor's oracle agent, which
   applies each reference solution: no model. Verification runs with no
   network, as it will with the model; what it would have fetched (the
   SWE-bench parser's packages and Python) is installed while each image
   is built. Each must read Resolved. About 5–10 minutes a task:
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --tasks 10
   ```
9. The Multilingual tasks whose own tests run a package manager — three at
   this pin (npm, composer, cargo); the runner finds them in their tests and
   names them. Their fetches run once while the image is built, and the
   tools are told they are offline. Each should read Resolved; one that
   doesn't can't be verified here without the network — say which:
   ```bash
   cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --only fetching
   ```
10. Three DeepSWE tasks. DeepSWE verifies in a separate container with no
    network by its own design; the runner gives that container the same
    limits as the agent's. If a task errors with "Docker egress control was
    not enabled", the kernel lacks nftables' `fib` rules: say so before
    going on:
    ```bash
    cd ~/benchmarks/aienh && ~/agent-venv/bin/python scripts/agent_run.py deepswe --oracle --tasks 3
    ```
11. Run step 6's command again: the images are gone, and the disk is back.

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
   - the disk used (step A6's command);
   - the card's memory (step B2's command);
   - anything odd in the conversations: thinking typed as a command, the
     window outgrown, a command that hung.

**C.** masein decides what runs next from B's time a task.
