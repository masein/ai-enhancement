# DeviceMark rows on a rented GPU (vast.ai)

A thinking-on DeviceMark run takes hours, and the board runs one model at a
time. This runs a Hugging Face model's battery on rented GPUs instead, with
the board's own code, and brings the answers back as files the board imports
and scores. Hugging Face models only: served setups stay on the server.

What runs where:
- **The image**, `ghcr.io/masein/evalboard-runner`, is built and pushed by the
  mirror's Actions (`.github/workflows/runner-image.yml`) on every push to
  main. It's the Dockerfile's `runner` stage: the board's runner, with the
  board's torch, lm_eval, transformers and fast kernels. It carries the board's
  code and DeviceMark's battery ids, and none of the board's question banks or
  data: the workflow checks every file in it before it pushes
  (`scripts/check_runner_image.py`). It's public, so vast.ai needs no login.
- **The rented boxes** run `scripts/remote_run.py` in that image. A battery can
  be split across several boxes with `--shard i/n`: box i answers items i,
  i+n, i+2n, … of each task. Each writes one file,
  `devicemark-<model>-thinking-<on|off>-shard-<i>-of-<n>.tar.gz`.
- **The server** pulls the files over `scp` (it connects out: it has no public
  address) and imports each with `scripts/import_remote.py`. The import
  refuses a file whose battery, protocol or pinned libraries aren't the
  server's. It scores a task once every shard of it is in, and until then says
  which shards are missing.

No credentials go in this file or in any script. You log in to vast.ai and
Hugging Face yourself.

## 0. Once

**An SSH key for the server.** On the server:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519 -N '' -C evalboard-server
cat ~/.ssh/vast_ed25519.pub
```

Paste the printed line into vast.ai → Account → SSH Keys. vast.ai puts it on
every instance you rent, so the server can `scp` from them.

**Make the image public.** GitHub creates the package private on its first
push, and Actions can't change that. Once, at
<https://github.com/users/masein/packages/container/evalboard-runner/settings>:
Danger Zone → Change visibility → Public. Every later push stays public.

## 1. After a deploy: the image's tag and the battery

On the server. The image's tag is the commit the server runs, the same value
as its `EVALBOARD_BUILD`. The mirror's Actions push it within the hour after
main moves; check it's there:

```bash
cd ~/benchmarks/aienh
TAG=$(git rev-parse --short HEAD)
echo $TAG
sudo docker manifest inspect ghcr.io/masein/evalboard-runner:$TAG > /dev/null && echo "the image is there"
```

Then print the server's battery hash. Each run in step 3 checks it before it
starts, so a battery the server would refuse never costs GPU hours:

```bash
sudo docker compose exec -T bench python scripts/import_remote.py --battery
```

Its first line is `items <hash>`.

## 2. What to rent

One instance a shard: **five** for the two waiting runs (Qwen3.5-4B's IFEval
in 2 shards, Gemma 4 E2B's battery in 3), all running at once.

In vast.ai → Templates → New template:
- **Image Path:Tag:** `ghcr.io/masein/evalboard-runner:<TAG>`, with the tag
  from step 1.
- **Docker Repository Authentication:** leave it empty. The image is public.
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
Rent five with this template.

## 3. Run, under tmux

From the server (or your Mac), with the address and port vast.ai shows for
each instance:

```bash
ssh -i ~/.ssh/vast_ed25519 -p <port> root@<host>
```

On each instance:

```bash
tmux new -s dm
cd /app
```

then that instance's command. **Qwen3.5-4B**, its IFEval alone (#167 left its
MMLU-Pro and MATH on the server), in 2 shards:

```bash
# instance 1
python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on --only dm_ifeval --shard 1/2 --out /workspace/qwen-1 --battery <hash from step 1>
# instance 2
python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on --only dm_ifeval --shard 2/2 --out /workspace/qwen-2 --battery <hash from step 1>
```

**Gemma 4 E2B**, the whole battery, in 3 shards. It's gated: type a token
that has accepted Gemma's licence on Hugging Face first, on each of the three:

```bash
read -rs HF_TOKEN && export HF_TOKEN
# instance 3
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 1/3 --out /workspace/gemma-1 --battery <hash from step 1>
# instance 4
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 2/3 --out /workspace/gemma-2 --battery <hash from step 1>
# instance 5
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 3/3 --out /workspace/gemma-3 --battery <hash from step 1>
```

`read -rs` takes the token without showing it or keeping it in the shell's
history. Nothing prints it, and nothing writes it into the bundle.

Each says its share first, for example `shard 1 of 2: items 1, 3, 5, … of each
task · dm_ifeval 150 of 300`, then prints a line an answer: the task, n of
the shard's N, the pace, the time left. `Ctrl-b d` leaves it running; `tmux
attach -t dm` comes back. If the process dies or the instance is stopped,
start the instance again and run **the same command**: it carries on from the
last saved answer. A folder holds one shard: another shard in the same `--out`
is refused.

The last lines name the bundle, its size and its sha256. For example:

```
bundle /workspace/qwen-1/devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-1-of-2.tar.gz · 1.1 MB · sha256 …
```

## 4. Fetch, from the server

The server connects out to each instance, with that instance's address and
port:

```bash
mkdir -p ~/benchmarks/bundles
scp -i ~/.ssh/vast_ed25519 -P <port 1> root@<host 1>:/workspace/qwen-1/devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-1-of-2.tar.gz ~/benchmarks/bundles/
scp -i ~/.ssh/vast_ed25519 -P <port 2> root@<host 2>:/workspace/qwen-2/devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-2-of-2.tar.gz ~/benchmarks/bundles/
scp -i ~/.ssh/vast_ed25519 -P <port 3> root@<host 3>:/workspace/gemma-1/devicemark-google__gemma-4-E2B-it-thinking-on-shard-1-of-3.tar.gz ~/benchmarks/bundles/
scp -i ~/.ssh/vast_ed25519 -P <port 4> root@<host 4>:/workspace/gemma-2/devicemark-google__gemma-4-E2B-it-thinking-on-shard-2-of-3.tar.gz ~/benchmarks/bundles/
scp -i ~/.ssh/vast_ed25519 -P <port 5> root@<host 5>:/workspace/gemma-3/devicemark-google__gemma-4-E2B-it-thinking-on-shard-3-of-3.tar.gz ~/benchmarks/bundles/
```

## 5. Import

As each file arrives, in any order:

```bash
cd ~/benchmarks/aienh
for f in ~/benchmarks/bundles/devicemark-*-shard-*.tar.gz; do sudo docker compose exec -T bench python scripts/import_remote.py "$f" --by masein; done
```

`~/benchmarks` is the container's `$BENCH_ROOT`, mounted at the same path, so
the host's path works inside it. Running the loop again is safe: a file
imported already changes nothing.

**What the import does:**
- **Checks:** the battery hashes, the protocol version, the pinned libraries,
  and that a shard holds exactly its own items. It refuses the file, saying
  which, when any differ.
- **Shards:** a shard waits under `results/shards/` until the task's others
  are in, and the import says which are missing, for example `dm_ifeval:
  shard 2 of 2 missing — dm_ifeval is scored once it is imported`. With every
  shard in, they're merged into the task's answers, in the battery's order, as
  one run would have written them. Every shard of a task must be one of the
  same n.
- **Merge:** each task the files complete replaces that task's answers here,
  with the earlier ones kept under `results/earlier/`. Other tasks stay as they
  are, so Qwen's IFEval joins #167's MMLU-Pro and MATH.
- **Scoring:** the row is scored by the board's own scoring once every task is
  in.
- **Where it ran:** its setup says it ran on a rented GPU and in how many
  shards, and the model page shows that line.
- **Runs:** the Runs list gets an entry for each file, with its log.

Importing the same file again changes nothing. A newer file of the same shard
replaces it, with the earlier one kept under `results/earlier/`.

## 6. Delete the instances

On vast.ai → Instances, the trash icon (Destroy), or `vastai destroy instance <id>`. Do this once the imports have worked.

**Destroy, don't just stop:** a stopped instance keeps charging for its disk.

## How long, and how much disk

The pace comes from #167, Qwen3.5-4B with thinking on, run on the board's RTX 5090:

| Task | Answers | Time | Per answer |
|---|---|---|---|
| IFEval | 259 (then the 3-hour limit stopped it) | 2:59:47 | 41.6 s |
| MMLU-Pro | 196 | 2:26:20 | 44.8 s |
| MATH | 100 | 1:13:09 | 43.9 s |

On DeviceMark's board, Qwen3.5-4B's answers average about 3,600 tokens, which comes to about 85 tokens a second here.

**Qwen3.5-4B, IFEval in 2 shards:** 150 answers each at 41.6 s is **about 1.75 hours** on an RTX 5090, both at once.

**Gemma 4 E2B, the full battery in 3 shards** (200, 198 and 198 answers):
- It depends on how long its answers run.
- On DeviceMark's board, Gemma 4 E2B's answers average about 390 tokens on IFEval and 800 on MMLU-Pro and MATH. At #167's 85 tokens a second, that is **about half an hour** a shard.
- If its answers run as long as Qwen3.5-4B's, it is **about 2.5 hours** a shard. Plan for the longer.

**On a slower GPU:** on an RTX 4090, allow about a third more time.

**Before the first answer:** each instance pulls the image (5.0 GB) and the
model, and builds the battery: allow 10 to 20 minutes.

**Disk, per instance:**
- the image: 9.2 GB unpacked;
- the model: about 9 GB for Qwen3.5-4B and about 10 GB for Gemma 4 E2B, in bf16;
- the battery's datasets: under 1 GB;
- the run's answers and cache: a few MB.

About 20 GB in all: 60 GB leaves room to spare.

## Publishing the raw runs

DeviceMark links every row of its board to its raw per-item file on Hugging Face
([devicemark/results](https://huggingface.co/datasets/devicemark/results), `raw/`).
Ours get the same link.

**Export.** Inside the container:

```bash
cd ~/benchmarks/aienh
sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" bench python scripts/export_devicemark_raw.py --all
sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" bench python scripts/export_devicemark_raw.py --run <run id>
```

`SCRUB_HOSTS` gives the export the server's host name to remove: inside the
container, the host name is the container's.

**What it writes.** One folder a row, under `~/benchmarks/raw-export/public/` or `…/private/`:
- `items.jsonl`, `setup.json`, `scores.json` and `log.txt`;
- `recompute.py`, which recomputes `scores.json` from `items.jsonl`;
- a `README.md` with the protocol and the battery's sources and licences.

**Public or private, by default.**
- Public: a public Hugging Face model, DeviceMark's calibration models among them.
- Private: a Qwen3.6 build, a setup served here, or a checkpoint.
- `--public` or `--private` decides instead.

**DeviceMark rows only.** Any other run is refused.

**The scrub.** Every file is scrubbed before it is written:
- keys and tokens, by value and by shape;
- home paths;
- this machine's host name and the tailnet's;
- private and tailnet addresses.

**Upload: your step.** Hugging Face sets privacy per repository, so the public and private rows go to two repositories. Once:

```bash
hf auth login
hf repos create <you>/evalboard-devicemark-raw --repo-type dataset
hf repos create <you>/evalboard-devicemark-raw-private --repo-type dataset --private
```

Each time:

```bash
hf upload <you>/evalboard-devicemark-raw ~/benchmarks/raw-export/public . --repo-type dataset
hf upload <you>/evalboard-devicemark-raw-private ~/benchmarks/raw-export/private . --repo-type dataset
```

**The link.** Give the row its link, either:
- on the model page's DeviceMark card (add its raw run's link); or
- with

  ```bash
  sudo docker compose exec -T bench python scripts/export_devicemark_raw.py --link <run id> https://huggingface.co/datasets/<you>/evalboard-devicemark-raw/tree/main/<folder> --by masein
  ```

The board then shows a "raw" link on the row's card and beside the row in the On-device chart's table.
