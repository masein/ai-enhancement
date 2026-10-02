# DeviceMark rows on a rented GPU (vast.ai)

A thinking-on DeviceMark run takes hours, and the board runs one model at a
time. This runs a Hugging Face model's battery on a rented GPU instead, with
the board's own code, and brings the answers back as one file the board
imports and scores. Hugging Face models only: served setups stay on the
server.

What runs where:
- **The rented box** runs `scripts/remote_run.py` inside our runner image (the
  Dockerfile's `runner` stage). It's the board's own runner, with the board's
  torch, lm_eval, transformers and fast kernels. It carries the board's code
  and DeviceMark's battery ids, and none of the board's question banks or
  data. It writes `devicemark-<model>-thinking-<on|off>.tar.gz`.
- **The server** pulls that file over `scp` (it connects out: it has no public
  address) and imports it with `scripts/import_remote.py`. The import refuses a
  file whose battery, protocol or pinned libraries aren't the server's.

No credentials go in this file or in any script. You log in to the registry,
vast.ai and Hugging Face yourself.

## 0. Once: an SSH key for the server

On the server:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519 -N '' -C evalboard-server
cat ~/.ssh/vast_ed25519.pub
```

Paste the printed line into vast.ai → Account → SSH Keys. vast.ai puts it on
every instance you rent, so the server can `scp` from them.

## 1. Push the image from the server to a private registry

After the deploy, on the server. `docker.io/<you>` stands for your registry
and account (Docker Hub here; `ghcr.io/<you>` works the same). Create the
repository as **private** first.

```bash
cd ~/benchmarks/aienh
TAG=$(git rev-parse --short HEAD)
sudo docker build --target runner --build-arg EVALBOARD_BUILD=$TAG -t evalboard-runner:$TAG .
sudo docker login docker.io
sudo docker tag evalboard-runner:$TAG docker.io/<you>/evalboard-runner:$TAG
sudo docker push docker.io/<you>/evalboard-runner:$TAG
```

The build reuses the layers the deploy just built (the `deps` stage is the
board's), so it takes a minute.

Then print the server's battery hash. Each run in step 3 checks it before it
starts, so a battery the server would refuse never costs GPU hours:

```bash
sudo docker compose exec -T bench python scripts/import_remote.py --battery
```

Its first line is `items <hash>`.

## 2. What to rent

One instance a model, so the two runs go in parallel.

In vast.ai → Templates → New template:
- **Image Path:Tag:** `docker.io/<you>/evalboard-runner:<TAG>` from step 1.
- **Docker Repository Authentication:** Server `docker.io` (or `ghcr.io`),
  your Username, and a **read-only** access token as Password/Token. Keep the
  template **private**.
- **Launch mode:** "Interactive shell server, SSH". vast.ai replaces the
  image's start command with its own and logs you in over SSH, which is
  what this image expects: its start command only prints `remote_run.py`'s
  help, and it never starts the board (no database, judge or queue).
- **On-start script:** `echo "EVALBOARD_BUILD=$EVALBOARD_BUILD" >> /etc/environment`
  vast.ai doesn't pass an image's environment into SSH or tmux sessions;
  this line puts the board's commit in the bundle's setup record.
- **Disk space:** 60 GB.
- **Environment variables:** none. `HF_TOKEN` is typed in at step 3, never
  stored in a template.

Then, in the search: 1 GPU with **24 GB or more** (an RTX 5090, the board's
card, gives the times below; a 4090 is slower), **max CUDA 12.8 or newer**
(the image's torch is built for CUDA 12.8), and "direct" SSH where offered.
Rent with this template.

## 3. Run, under tmux

From the server (or your Mac), with the address and port vast.ai shows for
the instance:

```bash
ssh -i ~/.ssh/vast_ed25519 -p <port> root@<host>
```

On the instance, **Gemma 4 E2B** (gated: its token must have accepted
Gemma's licence on Hugging Face):

```bash
tmux new -s dm
cd /app
read -rs HF_TOKEN && export HF_TOKEN
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --out /workspace/gemma --battery <hash from step 1>
```

**Qwen3.5-4B**, its IFEval alone (#167 left its MMLU-Pro and MATH on the
server):

```bash
tmux new -s dm
cd /app
python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on --only dm_ifeval --out /workspace/qwen --battery <hash from step 1>
```

`read -rs` takes the token without showing it or keeping it in the shell's
history. Nothing prints it, and nothing writes it into the bundle.

It prints a line an answer: the task, n of N, the pace, the time left on the
task and on the run. `Ctrl-b d` leaves it running; `tmux attach -t dm` comes
back. If the process dies or the instance is stopped, start the instance
again and run **the same command**: it carries on from the last saved answer.

The last lines name the bundle, its size and its sha256. For example:

```
bundle /workspace/qwen/devicemark-Qwen__Qwen3.5-4B-thinking-on.tar.gz · 2.1 MB · sha256 …
```

## 4. Fetch, from the server

The server connects out to the instance:

```bash
mkdir -p ~/benchmarks/bundles
scp -i ~/.ssh/vast_ed25519 -P <port> root@<host>:/workspace/gemma/devicemark-google__gemma-4-E2B-it-thinking-on.tar.gz ~/benchmarks/bundles/
scp -i ~/.ssh/vast_ed25519 -P <port> root@<host>:/workspace/qwen/devicemark-Qwen__Qwen3.5-4B-thinking-on.tar.gz ~/benchmarks/bundles/
```

## 5. Import

```bash
cd ~/benchmarks/aienh
sudo docker compose exec -T bench sh -c 'python scripts/import_remote.py "$BENCH_ROOT/bundles/devicemark-google__gemma-4-E2B-it-thinking-on.tar.gz" --by masein'
sudo docker compose exec -T bench sh -c 'python scripts/import_remote.py "$BENCH_ROOT/bundles/devicemark-Qwen__Qwen3.5-4B-thinking-on.tar.gz" --by masein'
```

**What the import does:**
- It checks the battery hashes, the protocol version and the pinned libraries against the server's, and refuses the file, saying which, when any differ.
- **Merge:** each task the file holds replaces that task's answers here, with the earlier ones kept under `results/earlier/`. Other tasks stay as they are, so Qwen's IFEval joins #167's MMLU-Pro and MATH.
- **Scoring:** the row is scored by the board's own scoring.
- **Where it ran:** its setup says it ran on a rented GPU, and the model page shows that line.
- **Runs:** the Runs list gets an entry with the file's log.

Importing the same file again changes nothing.

## 6. Delete the instance

On vast.ai → Instances, the trash icon (Destroy), or `vastai destroy instance <id>`. Do this once the import has worked.

**Destroy, don't just stop:** a stopped instance keeps charging for its disk.

## How long, and how much disk

The pace comes from #167, Qwen3.5-4B with thinking on, run on the board's RTX 5090:

| Task | Answers | Time | Per answer |
|---|---|---|---|
| IFEval | 259 (then the 3-hour limit stopped it) | 2:59:47 | 41.6 s |
| MMLU-Pro | 196 | 2:26:20 | 44.8 s |
| MATH | 100 | 1:13:09 | 43.9 s |

On DeviceMark's board, Qwen3.5-4B's answers average about 3,600 tokens, which comes to about 85 tokens a second here.

**Qwen3.5-4B, IFEval only:** 300 answers at 41.6 s is **about 3.5 hours** on an RTX 5090.

**Gemma 4 E2B, the full battery (596 answers):**
- It depends on how long its answers run.
- On DeviceMark's board, Gemma 4 E2B's answers average about 390 tokens on IFEval and 800 on MMLU-Pro and MATH. At #167's 85 tokens a second, that is **about 1.5 hours**.
- If its answers run as long as Qwen3.5-4B's, it is about **7 hours**. Plan for the longer.

**On a slower GPU:** on an RTX 4090, allow about a third more time.

**Disk, per instance:**
- the image;
- the model: about 9 GB for Qwen3.5-4B and about 10 GB for Gemma 4 E2B, in bf16;
- the battery's datasets: under 1 GB;
- the run's answers and cache: a few MB.

60 GB leaves room to spare.
