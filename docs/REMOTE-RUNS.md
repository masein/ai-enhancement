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

**The image is public.** GitHub made the package public on its first push
from the public mirror (2 Oct), so vast.ai pulls it with no login. Actions
can't change a package's visibility; if it ever reads private, it's at
<https://github.com/users/masein/packages/container/evalboard-runner/settings>:
Danger Zone → Change visibility → Public.

## 1. After a deploy: the image's tag

On the server. The image's tag is the commit the server runs, the same value
as its `EVALBOARD_BUILD`. The mirror's Actions push it within the hour after
main moves; check it's there:

```bash
cd ~/benchmarks/aienh
TAG=$(git rev-parse --short HEAD)
echo $TAG
sudo docker manifest inspect ghcr.io/masein/evalboard-runner:$TAG > /dev/null && echo "the image is there"
```

**The battery.** Its expected items hash is committed beside its ids
(`eval_tasks/devicemark/items-v1.sha256.json`). Each run in step 3 checks its
battery against it before it starts, so a battery the server would refuse
never costs GPU hours; a run here and the import check it too. To see the
server's:

```bash
sudo docker compose exec -T bench python scripts/import_remote.py --battery
```

It ends `the repo's items fe5fe34f… (the same)`.

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
python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on --only dm_ifeval --shard 1/2 --out /workspace/qwen-1
# instance 2
python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on --only dm_ifeval --shard 2/2 --out /workspace/qwen-2
```

**Gemma 4 E2B**, the whole battery, in 3 shards. It's gated: type a token
that has accepted Gemma's licence on Hugging Face first, on each of the three:

```bash
read -rs HF_TOKEN && export HF_TOKEN
# instance 3
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 1/3 --out /workspace/gemma-1
# instance 4
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 2/3 --out /workspace/gemma-2
# instance 5
python scripts/remote_run.py --model google/gemma-4-E2B-it --thinking on --shard 3/3 --out /workspace/gemma-3
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

**A rented RTX 5090 is slower than the board's.** The first one (2 Oct) ran
Qwen3.5-4B with thinking on at about 74 s a full-length answer, against about
47 s on the board's (#167's log): about **1.6 times** as long. The times below
are the server's pace times 1.6.

**Rented hosts vary.** The same card can be power-limited, or sit in a host
with a slower CPU or PCIe link, or share the machine. One may be faster or
slower than this: the pace in `remote_run.py`'s lines after the first dozen
answers tells you which.

**Qwen3.5-4B, IFEval in 2 shards:** 150 answers each, **about 2.8 hours** a shard, both at once.

**Gemma 4 E2B, the full battery in 3 shards** (200, 198 and 198 answers):
- It depends on how long its answers run.
- On DeviceMark's board, Gemma 4 E2B's answers average about 390 tokens on IFEval and 800 on MMLU-Pro and MATH. That is **about 50 minutes** a shard.
- If its answers run as long as Qwen3.5-4B's, it is **about 4 hours** a shard. Plan for the longer.

**On a slower GPU:** on an RTX 4090, allow about a third more again.

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

# A GGUF's Frontier benchmarks on a rented GPU (17)

The Frontier benchmarks — GPQA Diamond, OTIS Mock AIME 2024–2025, MATH Level
5, Humanity's Last Exam (text-only), SimpleQA Verified, MMLU-Pro (all 12,032)
and ARC-AGI-2 (public set), each as Epoch AI or its owners run it
(`scripts/frontier.py` says how) — asked of a GGUF, the original Qwen3.6
build or the LDA one, on rented GPUs, and
brought back onto that served model's row on the board, or its "· thinking"
row. `scripts/remote_gguf.py` does it, in the same runner image:
- it starts llama-server on the box (127.0.0.1 only) from a **tarball** you
  build once from the fork, and waits until it is healthy;
- it asks through the board's own code (`service/frontier.py`): the same
  prompt, the GGUF's own chat template (`--jinja`), the model card's sampling,
  the thinking switch said out loud, one seed a question and run;
- it keeps each answer as it lands: the same command again carries on;
- it writes one file, `frontier-served__<name>-thinking-<on|off>[-shard-<i>-of-<n>].tar.gz`,
  with the GGUF's sha256, the llama.cpp build and its sha256, the launch flags
  and environment and the GPU. Never the GGUF, the server or a key.

The fork, the GGUFs and your token never go in the repo or the image: the
tarball and the GGUF sit in a private Hugging Face repository of yours, and the
box pulls them with a token you type.

## G0. Once

**A private repository, and the token.** With a token that can write:

```bash
hf auth login
hf repos create <you>/evalboard-private --private
```

Then a **read** token for the boxes (Hugging Face → Settings → Access Tokens →
fine-grained: read on `<you>/evalboard-private`, and "read access to contents
of all public gated repos you can access"). On the same account, accept GPQA's
terms at <https://huggingface.co/datasets/Idavidrein/gpqa>: it is gated, and
the box fetches it with this token. The same for OTIS Mock AIME
(<https://huggingface.co/datasets/EpochAI/otis-mock-aime-24-25>) and Humanity's
Last Exam (<https://huggingface.co/datasets/cais/hle>): both gated, both
approved at once. The board's own token (HF_TOKEN in its `.env`) needs the
three too: the import reads the questions to check the answers cover them.

**The tarball**, on the server, from the fork at the commit build-lda was built
from. That commit is the one build-lda prints:

```bash
cd ~/llama.cpp-teraformer
LD_LIBRARY_PATH=$HOME/lda-env/lib:$PWD/build-lda/bin build-lda/bin/llama-server --version
git log -1 --format=%H
```

The two must match (check out build-lda's if they don't). Then:

```bash
~/benchmarks/aienh/scripts/build_llama_tarball.sh ~/llama.cpp-teraformer ~/llama-server-cuda12.8.tar.gz
```

It compiles in Docker (nvidia/cuda 12.8 on Ubuntu 22.04, the runner image's
CUDA and C library) for the A100 to the RTX 5090, about 15 minutes; nothing
runs. The last line gives its size and sha256.

**Upload** the tarball, and each GGUF the boxes will run:

```bash
hf upload <you>/evalboard-private ~/llama-server-cuda12.8.tar.gz llama-server-cuda12.8.tar.gz
hf upload <you>/evalboard-private ~/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf
```

**The file's sha256, on the board.** The import refuses a file that isn't the
one registered for the served model. If its file is registered on its page
(Download ▸ register its file) and the GGUF worker has hashed it, the board has
the sha256 already. Otherwise, on the server:

```bash
sha256sum ~/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf
```

and give it to the first import (`--file-sha256`, step G4): the board keeps it
on the model, with your name.

## G1. What to rent

The template from step 2, with **Disk space: 80 GB** (the GGUF, 23 GB; the
tarball; the datasets, HLE's images among them). In the search: an **RTX 5090**
(32 GB), **max CUDA 12.8 or newer**, and **inet down ≥ 1,000 Mbps** (the GGUF
in about 4 minutes; destroy a box that pulls below 25 MB/s).

A slot holds a benchmark's budget and its prompt; the box sizes the context
from the benchmarks it runs (`--only`). What fits beside the 23 GB file, with
the KV cache in q8_0 (`--flags "-ctk q8_0 -ctv q8_0 --flash-attn on"`):

| Benchmarks on the box | Slot, thinking on | `--slots` |
|---|---|---|
| GPQA, HLE, MMLU-Pro, SimpleQA | 34,816–36,864 | 8 |
| OTIS, MATH Level 5 | 67,584 | 8 |
| ARC-AGI-2 (its prompts run to 30,000 tokens) | 98,304 | 5 |
| any, thinking off | 6,144–40,960 | 8 |

## G1b. The pilot's first step: does the box answer as the server does?

The box isn't the server: the whole model on the card, 8 slots, the KV cache in
q8_0. Before anything else, the same 50 MMLU-Pro questions (`scripts/frontier.py`
`PARITY`), thinking off and greedy, on both, each with its own launch.
**Compared:** the letter each side reads from each reply (TIGER-Lab's
extraction), and whether the replies are identical. **The same:** the same
letter on at least 46 of the 50. Greedy decoding on two machines with other
batch sizes, cache types and kernels isn't bit-for-bit the same, so whole
replies may part: they are counted, not required. One build at a time, the
phone build first.

On the server, with the queue idle and the build served as it serves the
board. The folder first, as you: the container runs as root, and a folder it
made would refuse the box's file later.

```bash
mkdir -p ~/benchmarks/parity
```

```bash
cd ~/benchmarks/aienh
```

```bash
sudo docker compose exec -T bench python scripts/frontier_parity.py ask --as served/<phone-build> --out /home/masein/benchmarks/parity/phone-server.jsonl
```

On the box (G2's first three blocks: tmux, `cd /app`, the token), the same
build with the full run's flags:

```bash
python scripts/remote_gguf.py --as served/<phone-build> --gguf hf://<you>/evalboard-private/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on Qwen/Qwen3.6-35B-A3B --thinking off --slots 8 --flags "-ctk q8_0 -ctv q8_0 --flash-attn on" --parity --out /workspace/parity-phone
```

Then on the server:

```bash
scp -i ~/.ssh/vast_ed25519 -P <port> root@<host>:/workspace/parity-phone/parity.jsonl ~/benchmarks/parity/phone-box.jsonl
```

```bash
sudo docker compose exec -T bench python scripts/frontier_parity.py compare /home/masein/benchmarks/parity/phone-server.jsonl /home/masein/benchmarks/parity/phone-box.jsonl
```

It prints "The same: 48 of 50 read as the same letter …" or "Not the same: …".
Not the same: run the box's parity again with the server's own launch
(`--slots 1`, no `--flags`), into another `--out`. If that is the same, the
box's cache type or slots are the difference; either way, stop and say so
before the full run.

**Then the paces** (30 minutes, the same box): `--only` each benchmark in
turn with `--shard 1/40` (a handful of questions each), thinking on and off.
Its lines give each benchmark's seconds an answer; the shards below come from
them. Its bundles aren't imported: a shard of 40 waits for the other 39.

## G2. Run, under tmux

On each box, one block at a time:

```bash
tmux new -s f
```

```bash
cd /app
```

```bash
read -rs HF_TOKEN && export HF_TOKEN
```

Type the token and press Enter: nothing shows, and nothing keeps it.

Then **one command for each build**, each with that build's own launch — the
import refuses a bundle whose routing variables or speculative-decoding flags
aren't the ones registered for its `--as` (memory, context, slots, the cache
type, flash attention and threads may differ: the parity check covers them).

The **phone build** (k4-LDA), registered as "routing local (no lookahead)":
no routing variables.

```bash
python scripts/remote_gguf.py --as served/<phone-build> --gguf hf://<you>/evalboard-private/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on Qwen/Qwen3.6-35B-A3B --thinking on --slots 8 --flags "-ctk q8_0 -ctv q8_0 --flash-attn on" --out /workspace/phone-on
```

The **original build** (k=8): none either.

```bash
python scripts/remote_gguf.py --as served/<original-build> --gguf hf://<you>/evalboard-private/<original-build-file>.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on Qwen/Qwen3.6-35B-A3B --thinking on --slots 8 --flags "-ctk q8_0 -ctv q8_0 --flash-attn on" --out /workspace/orig-on
```

A setup registered **with lookahead** gets its row only from a box run with
its own variables, `--env "LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"`,
and its own `--as`.

- `--as`: the served model's id on the board (its page's address, `served/…`).
  Its row gets the answers; thinking on goes on its "· thinking" row.
- `--based-on`: what the board says it is based on. It picks the card's
  sampling, and the import checks the box asked as the board would.
- No `--cpu-moe`: the card holds the whole model.
- `--shard i/n` splits each benchmark across n boxes, one `--out` each.

It prints the file's sha256, the build and the GPU, each benchmark's share,
then a line as the run moves: answers, the pace, the time left. If the box
stops, run **the same command** again: it asks only what is not answered, and
refuses another build, other flags or another environment in that `--out`.
Something already answering on its port (an earlier llama-server) is
refused before anything starts.

## G3. Fetch, from the server

```bash
scp -i ~/.ssh/vast_ed25519 -P <port> root@<host>:/workspace/phone-on/frontier-served__<phone-build>-thinking-on.tar.gz ~/benchmarks/bundles/
```

## G4. Import

```bash
cd ~/benchmarks/aienh
sudo docker compose exec -T bench python scripts/import_remote.py ~/benchmarks/bundles/frontier-served__<phone-build>-thinking-on.tar.gz --by masein
```

With `--file-sha256 <the sha256 from G0>` the first time, if the board has no
hash of the file. It checks the served model, the file's sha256, the launch
against the one registered (routing and speculative decoding), each
benchmark's protocol and dataset revision, the budget, sampling and thinking
switch, every answer line, that thinking was on or off as asked, and that the
answers cover their questions (a shard's: exactly its own, made with the same
setup as its task's other shards). Each benchmark is scored apart first: a
bundle that fails a check, or a task that can't be scored, leaves the row as
it was. A shard waits until the others are in. Then each benchmark is scored by
code, its result says "run on a rented GPU (<the GPU>)", and the Runs list gets
the import with the box's log.

## G4b. Grading, on AI models

An import scores what code can: GPQA Diamond, MMLU-Pro and ARC-AGI-2 whole,
and MATH Level 5 and OTIS Mock AIME by code. SimpleQA Verified and Humanity's
Last Exam wait for their graders, and MATH's and OTIS's answers the code marks
wrong wait for Epoch's model check. **Nothing is sent by itself.**

AI models ▸ "Frontier benchmarks: grading" lists each grader with its model
(the owners' where it is still served: gpt-4.1-2025-04-14 for SimpleQA,
o3-mini-2025-01-31 for HLE) and its prompt's sha256, then the dry run — each
benchmark's answers waiting, and what they would cost (and, at most, if
every reply used its grader's cap). "change ▾" picks another model; choosing
one asks each of its providers one paid token first, counted in the month's
spend. **Start grading** pins each grader and sends what waits, once however
often it is pressed; Stop holds it, and the card then says why it waits —
Stop, a run of refusals, the month's limit — beside **Carry on**. As grades
land, each benchmark is scored again, and its cell says who graded it, with
which prompt (MATH and OTIS: the code's number beside it). A reply that is
empty, cut at its cap or not a grade is listed under the card, with a batch
that failed and why, and the next Start asks it again. A score from two
graders or prompts names both and isn't final: choose one and grade again.

## G5. The full run: which box runs what

Each build the same way, one `--out` a box: G2's command for that build — the
phone build's with no routing variables, the original's with none — with the
box's `--only`, `--shard` and its own `--out` (`/workspace/phone-mmlu-1`,
`/workspace/orig-hle-2`, …). Every shard of a benchmark is run with the same
tarball, flags and environment: the import refuses to merge shards that
differ.
Before the pilot's paces, from each benchmark's answers and an assumed length
for each (below), at about 1,400 tokens a second on one 5090:

| Box | Thinking | `--only` | `--shard` | about |
|---|---|---|---|---|
| 1–3 | on | `mmlupro_tiger` | `1/3` … `3/3` | 2.8 h each |
| 4–5 | on | `hle_text_cais` | `1/2`, `2/2` | 2.6 h each |
| 6 | on | `gpqa_diamond_epoch`, `otis_aime_epoch` | — | 2.7 h |
| 7 | on | `math_l5_epoch`, `simpleqa_epoch` | — | 1.9 h |
| 8 | on | `arc_agi2_public` (`--slots 5`) | — | 2.8 h |
| 9 | off | every benchmark but ARC-AGI-2 | — | 2.4 h |

ARC-AGI-2 isn't asked with thinking off: it would score nothing. Nine boxes a
build, eighteen in all, about 3 hours; the bundles import in any order.

## G6. The calibration: Gemma 4 26B A4B

The same pipeline on a model Epoch AI has measured, so a gap between our
number and Epoch's is the method's: GPQA Diamond and OTIS Mock AIME, with the
reasoning Epoch ran it with (the model version in the board's Epoch import
says which; thinking on unless it says otherwise), from the **BF16** GGUF (no
quantisation, 50.5 GB) on one 80–96 GB card (an H100 80 GB, or an RTX PRO
6000), with **100 GB of disk**:

After G2's first three blocks (tmux, `cd /app`, the token):

```bash
python scripts/remote_gguf.py --as served/gemma-4-26b-a4b-bf16 --gguf hf://ggml-org/gemma-4-26B-A4B-it-GGUF@bb4531cda34d1ea09d9814959ed4d5833cf2a4c8/gemma-4-26B-A4B-it-BF16.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on google/gemma-4-26b-a4b-it --thinking on --slots 8 --only gpqa_diamond_epoch --only otis_aime_epoch --out /workspace/gemma-cal
```

The BF16 file is public, in one piece (ggml-org's, pinned to its commit): the
box fetches it itself, nothing to upload. (A split GGUF would be given by its
first part: the box fetches every part, and its identity is the sha256 of the
parts' names and sha256s.) Check its first answers carry their thinking
(`answers.jsonl` holds `<think>`): Gemma's template is told to think with
`chat_template_kwargs`, and a template that ignores it answers without. If
they don't, stop the box and say so before the run. The board doesn't serve
Gemma 4, so its first import registers it, with the file's sha256 as Hugging
Face publishes it (the box's first line prints the one it hashed: they must
agree):

```bash
sudo docker compose exec -T bench python scripts/import_remote.py ~/benchmarks/bundles/frontier-served__gemma-4-26b-a4b-bf16-thinking-on.tar.gz --by masein --register "Gemma 4 26B A4B (BF16, rented GPU)" --file-sha256 463c88dbc5f692e812013e6449253eae4cff0fc10fbbd8d0f038d3690f03eb72
```

Then alias it to Epoch's entry for the model on the Frontier view, so the cell
reads "measured here … · Epoch …".

**What counts as a match**, said before the run: our number and Epoch's
(73.2 on GPQA Diamond and 82.2 on OTIS Mock AIME in the board's import) differ
by less than 1.96 × √(our standard error² + Epoch's²), Epoch's error as its
import gives it. With 198 and 45 questions that is a band of about ±9 points on
GPQA and about ±16 on OTIS: GPQA is the real test, and OTIS a check that
nothing is badly off. A miss is reported with its interval, never adjusted.

## The bill, before the pilot

Assumed mean answer lengths, thinking on (off): GPQA 8,000 (600), OTIS 20,000
(2,000), MATH Level 5 6,000 (1,000), HLE 12,000 (700), SimpleQA 1,500 (60),
MMLU-Pro 3,500 (450), ARC-AGI-2 30,000 (not asked). The pilot replaces them.

| | Tokens written, one build | Card-hours, one build |
|---|---|---|
| Thinking on | about 101 M (MMLU-Pro 42 M, HLE 26 M, ARC 10 M) | about 20.5 |
| Thinking off | about 9.5 M (MMLU-Pro 5.4 M) | about 2.4 |
| MMLU-Pro's prompts (5 examples each, 27.7 M tokens a build and setting) | — | about 0.2 with the examples' prefix cached (questions go in category order), 1.5 without |

Both builds: about **46 card-hours**, eighteen 5090s for about 3 hours,
**about $27** at $0.44–0.63 an hour (within a 2× band, $20–55). The
calibration: about 12 M tokens, about 3.5 hours of an 80–96 GB card, about $4.
Grading (stage 3, after the dry run and a yes): MMLU-Pro is scored by code, so
it adds nothing; SimpleQA Verified and HLE with their owners' graders about
$63 for the four runs.
