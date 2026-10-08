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
    4,096 processes, no new privileges, every capability dropped but the
    three Harbor's own steps need on the folders it mounts (CHOWN,
    DAC_OVERRIDE, FOWNER; Docker's others come back only for a task that
    proves it needs one), and no address or key of the model. The agent checks
    this from inside before the model is asked anything, and the check
    fails closed: it must show it can try a connection at all, nothing may
    answer (by IPv4, IPv6 or name), the loopback must be its only
    interface, and its capabilities and no-new-privileges must be as set.
  - Harbor mounts three of the trial's folders into the container
    (`agent/`, `verifier/`, `artifacts/`), and the model's commands run
    there as root: anything in them may be a link, a FIFO or a device it
    made. The agent's own files (its trajectory, the patch, its counts) go
    in the trial's `agent-host/` folder instead, which no container mounts,
    each written under a new name and renamed into place, never through a
    link. However the agent stops, every process left in the container is
    killed, the request it had in flight is cut at the relay (the server
    stops a reply nobody waits for), and the three mounted folders are
    emptied before the verifier runs.
  - The board reads a file only if it is a regular file with one link,
    reached from the run's folder through real folders: never a link, a
    FIFO, a device or a hard link, and at most 2 MB of it.
- The agent reaches the model through the **relay**
  (`scripts/agent_relay.py`), in the runner's process:
  - it listens on 127.0.0.1 only and takes only the chat request;
  - it adds the server's key (read from the board's record, never written
    anywhere);
  - it sets the run's sampling: temperature 1.0, top_p 0.95, top_k 20,
    min_p 0, no presence penalty, thinking on;
  - it caps a reply at 32,768 tokens (Qwen's own output length for most
    tasks), inside the smallest window a run takes;
  - it keeps each reply's tokens and what the server took from its cache.
- **Before the first task**, the runner refuses in one line, with what to do,
  if any of these fails:
  - the pinned versions, and the venv exactly as the lock
    (`docs/agent-requirements.txt`, with hashes) has it: every package of
    it at its version, none it doesn't hold, and every installed file as
    its package's record has it;
  - Harbor's command beside the venv's python (the runner runs it from
    there, with the venv's `bin` first on each job's PATH);
  - Docker, for the user running the runner (step A5);
  - 50 GB free where Docker keeps its images;
  - the board serving the model with the registered file;
  - a window of at least 131,072 tokens;
  - a tool call coming back through the relay;
  - a run's folder started with other settings (its window, the model's
    file, its build or flags, the sampling, the agent's prompt, the lock,
    the build's lock, the reply's cap, the container's limits, the runner's
    commit and files): a run's settings never change, and its first are
    kept. `--run 2` starts another run of the same model beside it;
  - a full run (`--all`) before Harbor's oracle has run on every task (§ C).
- **`--check`** also renders a three-step conversation with the server's own
  template (each step must render as it did before the next came, with its
  thinking, and the thinking open), sends two requests that extend one
  conversation as the agent sends them, and refuses when the second read
  more than a fifth of a long prompt again rather than from the server's
  cache, or when the server doesn't say what it took from it. Its requests
  have the agent's own budget (the relay's 32,768 tokens), so a long think
  before the first tool call is no refusal. It says the speeds it measured
  and what they make of the whole benchmark.
- **Which tasks is always said:** `--tasks 10`, `--only a,b` or `--all`.
- **A pilot is fixed:** `--tasks 10` is the same ten every time, spread over
  the benchmark's languages. `--only a,b` asks those tasks; `--only
  fetching` the tasks whose own tests fetch packages.
- **Verification has no network either, and nor do the builds:** it runs
  the model's code, and the board takes writes from the tailnet without a
  key. The runner asks each task from a copy of it
  (`agent-tasks/<benchmark>+offline/`) whose build and verification need
  none:
  - SWE-bench Multilingual verifies in the agent's container, and its
    `test.sh` has uv fetch the SWE-bench parser's packages and a Python
    3.11+ from the internet. The runner fetches those once instead
    (`$BENCH_ROOT/agent-build/`, named by the lock's hash), from
    `docs/agent-build.json` and `docs/agent-build-requirements.txt`: the
    task's own uv (0.7.13), a Python 3.11 and the parser's 78 packages,
    nothing published in the 14 days before the lock's date, each file
    checked against its hash, wheels only (no package's code runs on the
    host). The copy's image takes them from its build folder and installs
    them with Docker's build network **off**; the task's own uv installer
    line is left out (the same uv comes from those files). The lock's date
    and hash are in `run.json`, and a run's settings never change.
  - The few tasks whose own test commands run a package manager (three at
    this pin: npm, composer, cargo) have those commands run once at build
    time, in a throwaway copy of the repository, with Docker's own network
    and no package's own scripts (npm's `--ignore-scripts`, composer's
    `--no-scripts --no-plugins`; cargo runs none to fetch), then the tools
    are told they are offline. npm's is held to the lock's date
    (`--before`); composer and cargo take the newest they're allowed, and
    the runner names those tasks.
  - DeepSWE verifies in a separate container, offline by its own design;
    Harbor gives it none of the runner's compose files, so the copy gives it
    one: no network, the agent's limits. Its images are built from its own
    Dockerfiles, which clone the repository and fetch its packages: Docker's
    own network.
  - Docker can't give a build "the package registries and nothing else"
    here: a build has either no network or Docker's own, which reaches what
    this host reaches (its own ports, the LAN, the tailnet). Step A7 says
    what a build with Docker's own network reaches. To limit it to the
    registries would take a build daemon of its own on an internal network
    whose only way out is a proxy that lets only the registries through
    (`docker buildx create --driver docker-container`, with Harbor's builds
    sent to it), or firewall rules on the host for the builds' bridge.
- **It carries on:**
  - a task with the model's result is never asked again, whatever the
    result: a failure is the model's result, and a later success never
    replaces it;
  - an error of ours is asked again, three times at most, only with
    positive evidence that it came before the agent started (Harbor never
    began the agent's phase, and the agent wrote nothing): Docker failing
    to pull, build or start the task's container; the container reaching
    something (the agent's check, before the model is asked); the run
    stopped while it ran. A failure of Harbor's or Docker's once the agent
    had run (copying the tests in, the clean-up, the teardown) is an error
    of ours that counts as not resolved and is never asked again: asking
    again would let a failure be replaced by a success;
  - the model's server failing the relay's health check (the relay asks
    `/health` when the server doesn't answer) stops the run: nothing more
    starts, that try isn't counted, and the same command, once the server
    is back, asks the task again from its start — said on the run's page.
    One bad reply from a server that is up is the model's;
  - a Harbor job has a time limit: its agent's, verifier's and build's
    limits plus 30 minutes. Past it, the runner kills it and removes its
    containers, said on the run's page; a job that ends with no result
    counts as a try;
  - every task asked is in the score's denominator: one given up after three
    errors of ours counts as not resolved, and the score says how many;
  - the run's page says how many tasks were asked again, and why;
  - after a kill, the same command goes on.
- **The board's runs are left alone:** the runner's calls into the board
  (its checks, its line every 3 minutes, the import) touch only its own
  Runs row. Only the board's own start puts runs that were running back in
  the queue. While a run posts its line, the board's queue holds runs on
  its model, each with a line saying why; runs on other models go on.
- **Disk:** when a task ends, what Harbor built for it is removed (Harbor
  removes it itself unless the job was killed), then the build cache its
  build made, one record at a time — the rest of Docker's cache (the
  board's deploy's) stays. Its base image goes once no waiting task needs
  it, its digest recorded first. What a task takes of Docker's disk at its
  peak is measured while it runs; before each task starts — before its pull
  and build — the runner asks for that much with 50 GB left over, and stops
  in one line when it isn't there: the tasks running finish.
- **On the board:**
  - while it runs, the model's page and the Playground say the model is
    busy with an agent run, until about when: a reply there waits behind
    the agent's requests and slows it;
  - the run is a Runs row, "this server", its line every 3 minutes ("37 of
    300 · 21 resolved · 28 min a task · about 5 days left"), imported at the
    end;
  - Benchmarks ▸ Agent tasks has a card each, with our score ("% resolved ±
    one standard error · N tasks"; a pilot says "pilot: 10 of 300") and the
    published numbers beside it, as reference — with what isn't comparable
    (a Multilingual task past its 50 minutes is not resolved here; the
    published runs had no such limit) and the tasks left out (their
    reference solution doesn't pass here, § C). A task counts as resolved
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

**Every runner line is written in its sudo form**, with the agent venv's
`bin` first on PATH: `sudo env PATH="$HOME/agent-venv/bin:$PATH"
~/agent-venv/bin/python scripts/agent_run.py …`. If you choose the docker
group in step 5, the line is the same without `sudo`.

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
3. Move an existing agent venv aside (uv would reuse it), then make the
   agent venv with Python 3.12 — uv fetches Python 3.12 if the server has
   none. No seed: pip comes from the lock in step 4, pinned and hashed like
   every other package:
   ```bash
   [ -e ~/agent-venv ] && mv ~/agent-venv ~/agent-venv.old.$(date +%s)
   ```
   ```bash
   ~/.local/bin/uv venv --python 3.12 ~/agent-venv
   ```
4. Install the lock with uv, every package checked against its hash, pip
   among them. It must end `Installed 104 packages …` with no error:
   ```bash
   ~/.local/bin/uv pip install --python ~/agent-venv/bin/python --require-hashes --no-deps -r ~/benchmarks/aienh/docs/agent-requirements.txt
   ```
5. **Docker for the runner: you choose.** Harbor runs `docker` as the user
   who runs the runner, and whoever can run `docker` is root on this server
   (a container can mount `/` and write anywhere). Both ways make the
   runner, and every package it imports, root:
   - **(a) The docker group.** Your account is root without a password from
     then on, for everything you run — every package in every venv you use,
     this one's included, and any script you run. It stays until you take
     it away (`sudo gpasswd -d $USER docker`). Then log out and in, and run
     each runner line below without its `sudo`:
     ```bash
     sudo usermod -aG docker $USER
     ```
   - **(b) sudo, each time.** Nothing changes for your account: only the
     runner is root, and only while it runs. Each runner line below is
     written this way. What that changes: the run's files are owned by root
     (under `$BENCH_ROOT/agent-runs/`, `agent-tasks/` and `agent-build/`;
     the board reads them as before); Harbor's and pip's caches are under
     `/root/.cache`; the Runs row still says masein (the runner takes the
     name sudo was run by; `--by masein` says it outright).
6. See where Docker keeps images, the space there, and what Docker uses:
   ```bash
   sudo sh -c 'd=$(docker info -f "{{.DockerRootDir}}"); echo "$d"; df -h "$d"; docker system df'
   ```
7. Check a task's container and a task's build reach nothing: the model's
   server, the board, the relay and ssh on every address of this server
   (the tailnet's and IPv6 among them) and on Docker's gateway, the LAN,
   Docker's DNS, the tailnet's DNS, the internet over IPv4 and IPv6, any
   name. Its first line must end "… no host folder; at most 4,096
   processes", its second "… reaches none of them"; anything else is a
   refusal that says what answered or what the check couldn't do. Its last
   line says what a build with Docker's own network reaches (the few tasks
   whose own tests fetch packages, and DeepSWE's builds): report it.
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --check-reach
   ```
8. The ten Multilingual tasks of step B, with Harbor's oracle agent, which
   applies each reference solution: no model. The runner first fetches the
   build's files once (uv, a Python and the parser's packages, about 130
   MB, each checked against its hash) and says where they are; each image
   is then built with no network, and verification runs with none, as it
   will with the model. Each must read Resolved. About 5–10 minutes a task:
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --tasks 10
   ```
9. The Multilingual tasks whose own tests run a package manager — three at
   this pin (npm, composer, cargo); the runner finds them in their tests and
   names them, and names those whose fetch can't be held to the lock's
   date. Their fetches run once while the image is built, with Docker's own
   network and no package's own scripts, and the tools are told they are
   offline after. Each should read Resolved; one that doesn't can't be
   verified here without the network — say which:
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --only fetching
   ```
10. Three DeepSWE tasks. DeepSWE verifies in a separate container with no
    network by its own design; the runner gives that container the same
    limits as the agent's. If a task errors with "Docker egress control was
    not enabled", the kernel lacks nftables' `fib` rules: say so before
    going on:
    ```bash
    cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py deepswe --oracle --tasks 3
    ```
11. Run step 6's command again: the images are gone, and the disk is back.

## B. With the model (masein, after A is reported)

**What counts as broken.** Stop and report if any of these happens:
- `--check` refuses (step 5): the template, the cache, a tool call;
- fewer than 3 of the 10 resolved;
- any error of ours;
- a task's container reaching anything;
- thinking typed as a command;
- the window outgrown on more than 2 of the 10;
- a command left running past its limit.

The original's llama-server (port 8091) is the only server touched. Never
the vLLM judge (`gemma-vllm`), never the phone build's server, never the
other llama-servers (8090, 8092, 8094, 8096). While the run is on, the
model's page and the Playground say it is busy with an agent run, and the
board's queue holds runs on that model, each with a line saying why.

0. Nothing else may use the original's server while the agent runs: one
   slot, and a request from elsewhere throws the agent's cache away. On the
   board, Runs must show nothing queued or running on
   `served/Qwen3.6-35B-A3B-Q4-original-k-8`, and the Reasoning lab no loop
   calibration (it uses the same server). Then nothing may be connected to
   port 8091 — this must print only its header line:
   ```bash
   sudo ss -tnp 'dport = :8091'
   ```
1. Find the original's llama-server by its port. Its PID is in the
   `users:(("llama-server",pid=…` part of the line (here `<PID>`):
   ```bash
   sudo ss -ltnp 'sport = :8091'
   ```
2. Stop it by that PID, wait until 8091 is closed, and check the other
   servers are still listening — the last command must print four
   `LISTEN` lines, for 8090, 8092, 8094 and 8096:
   ```bash
   kill <PID>
   ```
   ```bash
   while sudo ss -ltn 'sport = :8091' | grep -q LISTEN; do sleep 2; done; echo "8091 closed"
   ```
   ```bash
   sudo ss -ltn '( sport = :8090 or sport = :8092 or sport = :8094 or sport = :8096 )'
   ```
3. Start it again with a 262,144 window, the experts of 21 layers on the
   CPU. Its own log is `~/serve-base-la0-mtp0.log`:
   ```bash
   CTX=262144 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0 > ~/lda-orig.log 2>&1 &
   ```
4. When its log says it is listening, check the card's memory: each
   process's, then the card's total, with the judge's beside it:
   ```bash
   tail -n 5 ~/serve-base-la0-mtp0.log
   ```
   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```
   If it didn't fit (the log says it couldn't allocate, or the server
   exited), or less than about 1.5 GB of the card is free (total minus
   used), do steps 1–2 again and start it with 131,072, the same line
   otherwise:
   ```bash
   CTX=131072 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0 > ~/lda-orig.log 2>&1 &
   ```
   Nothing to press on the board: the runner takes a change of the window
   alone, and the run records the window the server reports.
5. The checks only. Besides the versions, Docker, the disk, the model and a
   tool call, `--check` renders a three-step conversation with the
   server's own template (each step must render as it did before the next
   came, with its thinking), then sends two requests that extend one
   conversation, as the agent sends them, and reads what the server took
   from its cache: a second step that reads more than a fifth of its prompt
   again is refused. It says the speeds it measured and what they make of
   300 tasks, and must end "ready — 10 task(s)":
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --as served/Qwen3.6-35B-A3B-Q4-original-k-8 --tasks 10 --check
   ```
6. Ten SWE-bench Multilingual tasks, one attempt, in tmux:
   ```bash
   tmux new -s agent
   ```
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --as served/Qwen3.6-35B-A3B-Q4-original-k-8 --tasks 10
   ```
   If the model's server goes down, the run stops in one line: nothing more
   starts, and that try isn't counted. Start the server again (step 3),
   then run the same command: the task it was on is asked again from its
   start, said on the run's page.
7. Report from the run's last lines and its page on the board:
   - step 5's speeds and estimate;
   - minutes a task, tokens, the last prompt's size, how many resolved, any
     error;
   - how many tasks timed out, and how many steps read more of their prompt
     again than they took from the server's cache (the run's last lines
     say both, from `relay-usage.jsonl`);
   - the disk used (step A6's command);
   - the card's memory (step 4's two `nvidia-smi` commands);
   - anything odd in the conversations: thinking typed as a command, the
     window outgrown, a command that hung.
8. Afterwards, the original back to its usual window: steps 1–2 again,
   then:
   ```bash
   CTX=65536 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0 > ~/lda-orig.log 2>&1 &
   ```

## C. Before the full run (masein)

1. Harbor's oracle on every task of both benchmarks. No network at
   verification, and the capabilities the container drops, can make a
   task's own reference solution fail here — likely among the 24 Maven and
   Gradle tasks, and npm's. Those tasks are left out of every model run's
   score and of its comparison with the published numbers, and the card
   names them ("N tasks left out: their reference solution doesn't pass
   here"). A full run (`--all`) is refused until the oracle has run on every
   task. Each of these takes a day or more:
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py swebench-multilingual --oracle --all
   ```
   ```bash
   cd ~/benchmarks/aienh && sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python scripts/agent_run.py deepswe --oracle --all
   ```
2. masein decides what runs next from B's time a task. A run started after
   a `git pull` that changed the runner, the lock, the container's limits or
   the reply's cap is a run of its own (`--run 2`): the pilot's ten never
   roll into it.
