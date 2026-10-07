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
sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" -e SCRUB_ACCOUNTS=<your Hugging Face account> bench python scripts/export_devicemark_raw.py --all
sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" -e SCRUB_ACCOUNTS=<your Hugging Face account> bench python scripts/export_devicemark_raw.py --run <run id>
```

`SCRUB_HOSTS` gives the export the server's host name to remove: inside the
container, the host name is the container's. `SCRUB_ACCOUNTS` (17g) gives
the Hugging Face accounts to remove wherever they name a path (the account
the builds and the llama-server tarball are kept under).

**What it writes.** One folder a row, under `~/benchmarks/raw-export/public/` or `…/private/`:
- `items.jsonl`, `setup.json`, `scores.json` and `log.txt`;
- `recompute.py`, which recomputes `scores.json` from `items.jsonl`;
- a `README.md` with the protocol and the battery's sources and licences.

**Public or private, by default.**
- Public: a public Hugging Face model, DeviceMark's calibration models among them.
- Private: a Qwen3.6 build, a setup served here, or a checkpoint.
- `--private` keeps any run private. `--public` never makes one public that
  the board doesn't know as public (17g): a build stays in `private/`,
  whatever the flags, and the export says so.

**DeviceMark rows only.** Any other run is refused.

**The Frontier runs** (17f) — the server's and those imported from rented
boxes — have an exporter of their own, scrubbed the same way:

```bash
sudo docker compose exec -T -e SCRUB_HOSTS="$(hostname)" -e SCRUB_ACCOUNTS=<your Hugging Face account> bench python scripts/export_frontier_raw.py --all --rented
```

One folder a run: `setup.json` (where it ran — "this server", or "rented GPU ·
RTX 5090 · box A3" — the box's times, sessions, image and GGUF, and each
benchmark's setup), `scores.json`, `items.jsonl` (each answer right or wrong,
ran out or not, its tokens and grade), `log.txt` and a `README.md`. An
answer's text only for MMLU-Pro, SimpleQA Verified and ARC-AGI-2: GPQA
Diamond, OTIS Mock AIME and Humanity's Last Exam are gated and ask not to be
redistributed, MATH's problems are withheld, and an answer quotes its
question. Each run exports what it brought (17g): a box's shard its own
questions, this server's run nothing of a benchmark an import replaced (the
README says which). `log.txt` withholds each line that quotes a hidden or
gated question, as the board's log view does. Public only for a model the
board knows as public: a public Hugging Face model, or a served one
registered with `--public-weights` (G6's calibration); a build served here
(the Qwen3.6 builds) goes to `private/` whatever the flags.

**The scrub.** Every file is scrubbed before it is written:
- keys and tokens, by value and by shape;
- home paths;
- this machine's host name and the tailnet's;
- private and tailnet addresses;
- (17g) every IPv4 address, an ssh or scp port and the user before an
  address, a rented GPU host's name (`ssh4.vast.ai`), a key given as a flag
  (`--api-key <value>`) or named `x-token`, `SUBMIT_TOKEN`, `HF_TOKEN` or
  `OPENROUTER_API_KEY` whatever its value, a URL's internal host
  (`*.internal`, `*.lan`, …), and the Hugging Face account a private repo is
  kept under (and those in `SCRUB_ACCOUNTS`).

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
approved at once.

**The board's own token** needs the three too: the import reads the questions
to check the answers cover them. The board has no `HF_TOKEN`: it reads the
token file in its `HF_HOME` (the `.env`'s, mounted at the same path in the
container), the one `hf auth login` writes there. Its account changed on
6 Oct, so check which it is, then accept the three sets' terms on that
account:

```bash
HF_HOME="$(sed -n 's/^HF_HOME=//p' ~/benchmarks/aienh/.env)" hf auth whoami
```

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
the sha256 already. Otherwise, on the server, one line per build's file (each
takes a minute or two):

```bash
sha256sum ~/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf
```

```bash
sha256sum ~/<original-build-file>.gguf
```

Keep both: the parity check compares the box's file with them (G1b), and each
build's first import gives its own (`--file-sha256`, step G4): the board keeps
it on the model, with your name.

## G1. What to rent

The template from step 2, with **Disk space: 80 GB** (the GGUF, 23 GB; the
tarball; the datasets, HLE's images among them). In the search: an **RTX 5090**
(32 GB), **max CUDA 12.8 or newer**, and **inet down ≥ 1,000 Mbps** (the GGUF
in about 4 minutes; destroy a box that pulls below 25 MB/s).

A slot holds a benchmark's budget and its prompt; the box sizes the context
from the benchmarks it runs (`--only`). The pilot measured, on a 5090
(32,607 MiB) beside the phone build's file (22,854,339,808 bytes), the KV
cache in q8_0 (`--flags "-ctk q8_0 -ctv q8_0 --flash-attn on"`): 8 slots of
36,864 used 26,136 MiB after load, 8 of 67,584 used 29,198. That is 12.76 KiB
a token of context (the GGUF's header gives 10.6 of it as KV cache;
llama-server's buffers that grow with the context, the rest) and 665 MiB
above the file. Under the limits of 6 Oct:

| Benchmarks on the box | Slot | `--slots` | Memory used, by the pilot's numbers | The box's own check |
|---|---|---|---|---|
| MMLU-Pro, SimpleQA, thinking on | 34,816–36,864 | 8 | 26,136 MiB, measured | 8 fit |
| MATH Level 5, thinking on | 67,584 | 8 | 29,198 MiB, measured | 8 fit |
| GPQA, OTIS (or MATH with them), thinking on | 83,968 | 8 | about 30,830 MiB | 8 fit, 1,154 MiB spare |
| HLE (or anything with it), thinking on | 86,016 | 8 | about 31,040 MiB | 8 fit, 1,570 MiB spare |
| ARC-AGI-2 (its prompts run to 30,000 tokens), thinking on | 114,688 | 5 | about 29,600 MiB | 6 fit; 5, as the pilot ran it |
| any, thinking off (ARC-AGI-2 isn't asked) | 5,120–20,480 | 8 | about 24,500 MiB at most | 8 fit |

The box works this out from the GGUF's header before it fetches anything,
from the pilot's measured slope — 12.76 KiB a token of context, and 666 MiB
above the file at 8 slots holding their recurrent state — keeping 1,024 MiB
spare, and prints the room left in MiB and the basis either way. (17f's
estimate counted a rounded-up overhead and the recurrent state again, 622 MiB
above both readings, and ran HLE at 7.) A step
planned at 8 slots that doesn't fit runs as many as do, down to 7
(`--min-slots 7`, which the box's line gives), and says so; fewer than that
is refused before the fetch. More slots would fit at the shorter contexts,
but nothing has measured whether they answer faster: the plan keeps 8.
`--slots-fit` runs the slots given whatever the estimate says.

## G1b. The pilot's first step: does the box answer as the server does?

The box isn't the server: the whole model on the card, 8 slots, the KV cache in
q8_0. Before anything else, the same 500 MMLU-Pro questions (`scripts/frontier.py`
`PARITY`; `--n` on both sides for another number), thinking off and greedy, on
both, each with its own launch.

**The same** is decided on accuracy, stated here before the run: the
difference in right answers on the same questions (the mean of the box's two
runs, question by question, minus the server's), with its 90% paired
interval, inside ±5 points. That is the two one-sided tests of equivalence
at 5% each. Two identical setups whose answers flip right and wrong on one
question in ten between runs are called not the same in about 1.5% of checks
of 500 (6% against one run of the box; at 14%, 7% and 18%): compare prints
the rate at the box's own flip rate. A pair of fewer than 500 questions is
refused, and so is a pair of files written before 17d (the pilot's 50).

Letter agreement and identical replies are reported beside it, for
information only. Greedy decoding with other batch sizes, cache types and
kernels isn't bit-for-bit the same: on the pilot (50 questions), two runs on
one box agreed on 43 letters of 50, and the box with any setup against the
server on 42 to 46 of 49. So the box asks each question twice, and its
agreement with itself is printed beside its agreement with the server. One
build at a time, the phone build first.

Each side's file opens with what answered: the served model, the file's name,
size and sha256, and the launch. The server's side is asked only while its
server serves the file registered. `compare` refuses the same file twice, a
question answered twice, two sides with other files, and a box whose routing
or speculative decoding isn't the one registered for the build.

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
sudo docker compose exec -T bench python scripts/frontier_parity.py ask --as served/<phone-build> --out /home/masein/benchmarks/parity/phone-server-500.jsonl
```

Its first line says how long the 500 take at this server's measured pace
with thinking off; its last, how long they took.

On the box (G2's first three blocks: tmux, `cd /app`, the token), the same
build with the full run's flags:

```bash
python scripts/remote_gguf.py --as served/<phone-build> --gguf hf://<you>/evalboard-private/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on Qwen/Qwen3.6-35B-A3B --thinking off --slots 8 --flags "-ctk q8_0 -ctv q8_0 --flash-attn on" --parity --out /workspace/parity-phone
```

It asks each question twice (1,000 answers at 8 slots), the second time for
the box's agreement with itself. In the full run nothing more is needed:
box A3 of each build (B12 in plan B) asks these first, and G3–G4's fetch
compares the box's file with the server's when `--parity` names it.

It fetches the GGUF and the tarball into `/workspace/files`, a folder every
run on the box shares: the full run (G2) after it fetches and hashes nothing
again. It prints the file’s sha256 (its first 16 characters).

Then on the server:

```bash
scp -i ~/.ssh/id_ed25519 -P <port> root@<host>:/workspace/parity-phone/parity.jsonl ~/benchmarks/parity/phone-box.jsonl
```

```bash
sudo docker compose exec -T bench python scripts/frontier_parity.py compare /home/masein/benchmarks/parity/phone-server-500.jsonl /home/masein/benchmarks/parity/phone-box.jsonl --file-sha256 <the phone file's sha256 from G0>
```

It prints "The same: the box answers 42.4% right (the mean of its two runs)
and the server 41.8% on the same 500 questions — a difference of +0.6 points,
90% interval -1.6 to +2.8, inside ±5 points. For information: the same letter
on …; at the box's own flip rate (9% of questions right on one run and wrong
on the other), two identical setups would be called not the same in 0.8% of
checks of 500 questions", "Not the same: …",
or "Not the same setup: …" (another file or launch: fix the box's command
before anything else). The board has no sha256 of either build's
file today, so `--file-sha256` is what compares the files whole; without it
they are compared by name, and by llama-server's count of their weights
(never the server's count against the box's file on disk, about 11 MB larger
for its header), and it says so.
Not the same: run the box's parity again with the server's own launch
(`--slots 1`, no `--flags`), into another `--out`. If that is the same, the
box's cache type or slots are the difference; either way, stop and say so
before the full run.

**The paces** were measured on 6 Oct, one box and the phone build, each
benchmark at `--shard 1/40`: G5's hours and the bill come from them.

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

Then **the box's one line**, G5's, with the build's three values. Its steps
run one after the other (a thinking-on benchmark, then thinking-off ones),
each a `remote_gguf.py` command with its own `--out`, so the box needs no
second visit. The **phone build** (k4-LDA), registered as "routing local (no
lookahead)": no routing variables. G5's box A5, for example:

```bash
python scripts/frontier_box.py A5 --as served/<phone-build> --gguf hf://<you>/evalboard-private/Qwen3.6-35B-A3B-k4-LDA-UD-Q4_K_XL.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz
```

The **original build** (k=8): none either.

```bash
python scripts/frontier_box.py A5 --as served/<original-build> --gguf hf://<you>/evalboard-private/<original-build-file>.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz
```

Box A3 of each build asks the parity questions first (G1b's, 500 of
MMLU-Pro, each twice: about half an hour), then its benchmarks. Each build's
steps go in a folder of its own, `/workspace/<build>/A5-1`, … (the name
after `served/`), so one box can run the other build after the first.

Each step's command is printed before it runs: the box's benchmarks
(`--only`), its shard, its slots, the thinking setting, `--flags "-ctk q8_0
-ctv q8_0 --flash-attn on"` and `--based-on Qwen/Qwen3.6-35B-A3B`, the box's
label (`--label A5`, kept in its bundle and its progress file), into
`/workspace/<build>/<box>-<step>`. The import refuses a bundle whose routing
variables or speculative-decoding flags aren't the ones registered for its
`--as` (memory, context, slots, the cache type, flash attention and threads
may differ: the parity check covers them). A setup registered **with
lookahead** gets its row only from a box run with its own variables, `--env
"LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"`, and its own
`--as`. `python scripts/frontier_box.py --list A` prints a plan.

- `--as`: the served model's id on the board (its page's address, `served/…`).
  Its row gets the answers; thinking on goes on its "· thinking" row.
- `--based-on`: what the board says it is based on. It picks the card's
  sampling, and the import checks the box asked as the board would.
- No `--cpu-moe`: the card holds the whole model.
- `--shard i/n` splits each benchmark across n boxes, one `--out` each.

Each step prints the file's sha256, the build and the GPU, the memory it
expects, each benchmark's share, then a line as the run moves: answers, the
pace, the time left. A step that stops doesn't stop the next; the last lines
say how each ended, and why. A step left short after asking (a question the
server failed on with a 5xx, kept for the next run) runs again by itself, up
to three runs, saying so — the third writes such a question off, and the
step has its bundle. If the box stops, paste **the same line** again: each
step asks only what is not answered, and a step already whole isn't run
again (the parity step's 1,000 answers aren't asked twice). A parity step
that stopped is asked again by the paste. A memory refusal's `--slots N` or
`--slots-fit` go on the box's line as they are. Once
its `--out` holds an answer, it refuses another build, other flags or another
environment there; before that (a typo in `--flags`), the corrected command
runs. Its `--out` keeps the served model, the thinking setting and the shard
it was first started with, answered or not: for another of those, another
`--out`. Something already answering on its port (an earlier llama-server)
is refused before anything starts. llama-server gets its `LLAMA_*` and `GGML_*`
variables from `--env` only, each one recorded: one exported in the shell
isn't passed, and the first lines say so.

A question the server fails on is asked once more at the end of its
benchmark. Failing again with an error of its own — a 4xx (its prompt and
budget too long for the context), or no answer within the timeout from a
server that is up — it is written as no answer, counted wrong, and named in
the log, on the row and on the import's line. A 5xx is the server's, whatever
its health check says: the question is kept, the run carries on to the next
benchmark, the step isn't whole, and pasting the line again asks it again;
only after failing on three separate runs is it written off.
`--ask-written-off` on the box's line asks again what was written off. A
server that fails every question it is asked, or more than one in fifty,
stops the run, and keeps none of them.

The bundle's name says the model, the thinking setting, the benchmarks and
the shard — `frontier-served__<phone-build>-thinking-on-mmlu-pro.tar.gz` (box
A5), `…-thinking-on-hle-shard-1-of-4.tar.gz` (A1), `…-thinking-off-gpqa+otis.tar.gz`
(A1's second step) — and the last line prints it.

## G3–G4. Fetch and import, from the server

One command for every box: each box is asked over SSH, once, for its bundles
(with their sha256), its parity file and each step's progress; its bundles
are copied into `~/benchmarks/bundles` and imported with the build's
`--file-sha256`, a box at a time, as soon as they are here. The boxes as
vast.ai's SSH line gives them (`ssh -p <port> root@<host>`: give
`<host>:<port>`), with your key; and the server's parity files from G1b, so
the boxes' are compared with them:

```bash
cd ~/benchmarks/aienh
python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 --sha served/<phone-build>=<the phone file's sha256 from G0> --sha served/<original-build>=<the original file's sha256 from G0> --parity served/<phone-build>=/home/masein/benchmarks/parity/phone-server-500.jsonl --parity served/<original-build>=/home/masein/benchmarks/parity/orig-server-500.jsonl --every 15m <host 1>:<port 1> <host 2>:<port 2> <host 3>:<port 3>
```

It asks for your password once (`sudo`, for the container) and keeps it alive
while it runs; if it lapses, one line says the next import waits for it. Run
it in a tmux pane of its own. A `--parity` file that isn't there is refused
before anything is fetched. `--every 15m` does it all again every 15 minutes until every
box is done, and each time it sends the boxes' progress to the board: Runs ▸
All runs opens with "On rented boxes", each step's box, model, what it asks,
n of N, when it should finish and when it was last heard from (a box quiet
for 45 minutes, or not reached at the last fetch, says so; one whose bundles
are all home and imported reads "done, safe to destroy"). Only labels and
progress go to the board, never a box's address; `--no-board` sends nothing. Each box's first line says whether it is **safe to destroy**:
every step its plan (G5) gives each build started there whole, and every
bundle it holds here with the same sha256 and imported. Otherwise "NOT safe
to destroy" and why — a step that hasn't started, a step still asking (its
progress beside: what it asks, n of N, its restarts, when it was last
written), a step stopped and why (llama-server dying at load, a gated
question set, a parity question the server failed on: paste its line
again), a bundle NOT copied, an import refused. A bundle of a model with no
`--sha` (G6's calibration) counts once it is home, and its line gives the
import to type. Each build's parity verdict is a line of its own every
round. A copy goes to a temporary name and counts only when its
sha256 is the box's: a failed copy leaves an older one here as it was. A box
that doesn't answer is given up on (a 15-second connect timeout, no prompts)
and said so. The exit code isn't 0 when a box couldn't be reached, a copy
failed, an import was refused, or a build's parity isn't the same or
couldn't be compared.

Run it whenever: a bundle made again (the box's line pasted again) holds the
same answers, and the import says so and changes nothing, its grades kept; a
step's bundle with more answers adds them, the grades of the rest kept; one
where some answers changed is graded again only for those, every unchanged
answer keeping its grade (across a task's shards too). A
step of two benchmarks fetched while only its first is whole brings that one;
the second comes with a later fetch, and the first's answers and grades stay
as they are. A shard waits for the others. A box's parity file is compared
with the server's when `--parity` gives it, and the verdict is printed (G1b);
an import never waits for it. By hand, one bundle:

```bash
scp -i ~/.ssh/id_ed25519 -P <port> root@<host>:/workspace/<phone-build>/A5-1/frontier-served__<phone-build>-thinking-on-mmlu-pro.tar.gz ~/benchmarks/bundles/
```

```bash
sudo docker compose exec -T bench python scripts/import_remote.py ~/benchmarks/bundles/frontier-served__<phone-build>-thinking-on-mmlu-pro.tar.gz --by masein --file-sha256 <the phone file's sha256 from G0>
```

`--file-sha256` on every import of a build, with that build's sha256 from G0
(the original build's bundles with the original file's). The board has no
hash of either build's file today: the first import keeps the one given on
the model, with your name, and every later one is checked against it (a
different one is refused). It checks the served model, the file's sha256, the launch
against the one registered (routing and speculative decoding), each
benchmark's protocol and dataset revision, the budget, sampling and thinking
switch, every answer line, that thinking was on or off as asked, and that the
answers cover their questions (a shard's: exactly its own, made with the same
setup as its task's other shards). Each benchmark is scored apart first: a
bundle that fails a check, or a task that can't be scored, leaves the row as
it was, and isn't counted as imported: the same command imports it once
fixed. A shard of a task whose shards here were made with another setup (a
rebuilt tarball) or asked another way (another token limit, protocol or
sampling: shards 1/2 and 2/2 at the old limits, then 1/2 again at the new) is
refused, saying which; `--set-aside-shards` sets those aside, and this
shard starts the task's shards again. A shard waits until the others are in. Then each benchmark is scored by
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

Choosing another grader moves nothing: the dry run shows what Start would do
with it, and Start does it. Each grader's grades are kept under its own name
— choosing one that graded a benchmark before uses what it graded, and the
dry run says how many, and what Start leaves the score as (Start is there
even when nothing is sent). A run of refusals anywhere in a Start (five of
the same with no reply between them, whatever else sits between) stops it
until Start, counting nothing; a reply that lands after the stop is kept. A
refusal of the answer itself (too long for the grader, flagged) is that
answer's try, never part of a run; a key's spend cap, the account's credit,
a provider's own failure and an id the provider doesn't know never are.
Stopped, then another grader chosen: the dry run prices what is held at the
grader it goes to, and a row graded by two graders is finished by the one
chosen now. When a benchmark's rows end up
scored by different graders (one build's first grader left too many
ungraded, the other's didn't), both rows say so, and the card offers the
other row's regrade by the grader chosen now, with its price, sent at the
next Start.

## G5. The full run: which box runs what

From the pilot's paces (one 5090, the phone build, 8 slots; ARC-AGI-2 at 5),
one build, under the limits of 6 Oct. Each answer that ran out of room on
the pilot is taken to run to the new limit (2 of 20 GPQA, 17 of 54 HLE, 8 of
16 OTIS and 3 of 10 ARC-AGI-2 with thinking on; 4 of 20 GPQA, 20 of 54 HLE, 8
of 16 OTIS and 11 of 301 MMLU-Pro with it off), at the same tokens a second —
the most it could take, more if the card writes slower at long contexts.

| Benchmark | Answers | Thinking on | Thinking off |
|---|---|---|---|
| Humanity's Last Exam | 2,158 | 49.7 h | 8.5 h |
| MMLU-Pro | 12,032 | 17.7 h | 7.4 h |
| MATH Level 5 | 1,324 | 13.0 h | 1.9 h |
| OTIS Mock AIME | 45 × 8 | 12.6 h | 1.7 h |
| ARC-AGI-2 | 167 test grids × 2 | 11.6 h | not asked |
| GPQA Diamond | 198 × 4 | 8.6 h | 2.0 h |
| SimpleQA Verified | 1,000 | not measured; about 1 h | 0.2 h |
| **One build** | | **114.2 h** | **21.7 h** |

135.8 box-hours a build, 272 for both, and half an hour a build for the
parity questions. ARC-AGI-2's 120 tasks hold 167 test grids at the pinned
revision, each asked twice. The paces come from small shards (16 to 301
answers): read them as ±25%. `python scripts/frontier_box.py --list A`
prints a plan with each step's hours.

**Plan A** (chosen on 6 Oct): about 18 hours a box.

| Box | Its steps | about |
|---|---|---|
| A1 | HLE on, shard 1/4; then GPQA and OTIS off | 16.1 h |
| A2 | HLE on, shard 2/4; then MATH and SimpleQA off | 14.5 h |
| A3 | the parity questions; then HLE on, shard 3/4 | 12.9 h |
| A4 | HLE on, shard 4/4 | 12.4 h |
| A5 | MMLU-Pro on | 17.7 h |
| A6 | MATH and SimpleQA on | 14.0 h |
| A7 | OTIS on; then MMLU-Pro off, shard 1/2 | 16.3 h |
| A8 | ARC-AGI-2 on (5 slots); then MMLU-Pro off, shard 2/2 | 15.3 h |
| A9 | GPQA on; then HLE off | 17.1 h |

Nine boxes a build, eighteen in all, the longest A5 at 17.7 hours: about
**20 hours** from the first box to the last bundle, starting a box every ten
minutes or so. (A box started on 308fcf3 runs HLE at 7 slots, about 1.8
hours longer for a quarter: the shards merge either way.)

**Plan B**: about 10 hours a box.

| Box | Its steps | about |
|---|---|---|
| B1–B6 | HLE on, shard 1/6 … 6/6 | 8.3 h each |
| B7 | MMLU-Pro on, shard 1/2; then SimpleQA on | 9.9 h |
| B8 | MMLU-Pro on, shard 2/2; then SimpleQA off | 9.0 h |
| B9 | MATH on, shard 1/2; then MMLU-Pro off, shard 1/2 | 10.2 h |
| B10 | MATH on, shard 2/2; then MMLU-Pro off, shard 2/2 | 10.2 h |
| B11 | OTIS on, shard 1/2; then GPQA and MATH off | 10.2 h |
| B12 | the parity questions; then OTIS on, shard 2/2; then OTIS off | 8.5 h |
| B13 | ARC-AGI-2 on, shard 1/2 (5 slots); then HLE off, shard 1/2 | 10.1 h |
| B14 | ARC-AGI-2 on, shard 2/2 (5 slots); then HLE off, shard 2/2 | 10.1 h |
| B15 | GPQA on | 8.6 h |

Fifteen boxes a build, thirty in all: about **13–14 hours**. A box that stops
loses its hours until its line is pasted again: shorter boxes lose less.

Every shard of a benchmark is run with the same tarball, flags and
environment: the import refuses to merge shards that differ. The bundles
import in any order (G3–G4).

## G6. The calibration: Gemma 4 26B A4B

The same pipeline on a model Epoch AI has measured, so a gap between our
number and Epoch's is the method's: **GPQA Diamond**, with the reasoning Epoch
ran it with (the model version in the board's Epoch import says which;
thinking on unless it says otherwise), from the **BF16** GGUF (no
quantisation, 50.5 GB) on one 80–96 GB card (an H100 80 GB, or an RTX PRO
6000), with **100 GB of disk**.

GPQA alone, as chosen on 6 Oct. Gemma's pace isn't measured: at Qwen's on
the 5090 under the limits of 6 Oct (81,920 with thinking on), up to twice
that for BF16's larger reads a token, GPQA's 792 answers take 9–17 hours,
**$9–34** at $1–2 an hour. (With OTIS Mock AIME's 360 as well it would be
about 21–43 hours: OTIS's band below is ±16 points, so it tests little.)

After G2's first three blocks (tmux, `cd /app`, the token):

```bash
python scripts/remote_gguf.py --as served/gemma-4-26b-a4b-bf16 --gguf hf://ggml-org/gemma-4-26B-A4B-it-GGUF@bb4531cda34d1ea09d9814959ed4d5833cf2a4c8/gemma-4-26B-A4B-it-BF16.gguf --server hf://<you>/evalboard-private/llama-server-cuda12.8.tar.gz --based-on google/gemma-4-26b-a4b-it --thinking on --slots 8 --only gpqa_diamond_epoch --out /workspace/gemma-cal
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
agree), and with `--public-weights`, so its raw runs may be exported public
(17g: nothing else ever is):

```bash
sudo docker compose exec -T bench python scripts/import_remote.py ~/benchmarks/bundles/frontier-served__gemma-4-26b-a4b-bf16-thinking-on-gpqa.tar.gz --by masein --register "Gemma 4 26B A4B (BF16, rented GPU)" --public-weights --file-sha256 463c88dbc5f692e812013e6449253eae4cff0fc10fbbd8d0f038d3690f03eb72
```

Then alias it to Epoch's entry for the model on the Frontier view, so the cell
reads "measured here … · Epoch …".

**What counts as a match**, said before the run: our number and Epoch's
(73.2 on GPQA Diamond and 82.2 on OTIS Mock AIME in the board's import) differ
by less than 1.96 × √(our standard error² + Epoch's²), Epoch's error as its
import gives it. With 198 and 45 questions that is a band of about ±9 points on
GPQA and about ±16 on OTIS: GPQA is the real test, and OTIS a check that
nothing is badly off. A miss is reported with its interval, never adjusted.

## The bill, from the pilot

| | Box-hours, both builds | At $0.44–0.63 an hour | The paces ±25% |
|---|---|---|---|
| **Plan A**: 18 boxes, the longest 17.7 h | about 287 | **$126–181** | $96–224 |
| Plan B: 30 boxes, the longest 10.2 h | about 296 | $130–186 | $100–230 |

G5's 272 box-hours, the parity questions' 1, and for each box about 0.3 hours
to start (the fetch, the load) and 0.5 hours from its last bundle to being
destroyed (the fetch's `--every 15m`, then destroying it). The calibration
(G6): GPQA alone, $9–34. Grading (stage 3, after a yes): MMLU-Pro is scored
by code and adds nothing; SimpleQA Verified's and HLE's graders, and Epoch's
model check on MATH and OTIS, are priced by the dry run on AI models from the
bundles' own answers once they are in.

## Token limits: ours, the model's card's and Epoch's

Epoch AI states no limit: it runs each model at its API's defaults, at the
highest reasoning effort it offers (an exception it names: Grok 4 at 128,000,
as xAI recommends). Qwen3.6-35B-A3B's card recommends 32,768 tokens "for most
queries" and 81,920 "for benchmarking on highly complex problems". On 6 Oct,
after the pilot, masein said yes to 17e's proposal, and these are the limits
now (`scripts/frontier.py`):

| Benchmark | Before, on (off) | Ran out on the pilot, on / off | Now, on (off) |
|---|---|---|---|
| GPQA Diamond | 32,768 (4,096) | 2 of 20 / 4 of 20 | 81,920 (16,384) |
| Humanity's Last Exam | 32,768 (4,096) | 17 of 54 / 20 of 54 | 81,920 (16,384) |
| OTIS Mock AIME | 65,536 (8,192) | 8 of 16 / 8 of 16 | 81,920 (16,384) |
| MATH Level 5 | 65,536 (8,192) | 0 of 34 / 0 of 34 | as before |
| ARC-AGI-2 | 65,536 (not asked) | 3 of 10 / — | 81,920 |
| MMLU-Pro | 32,768 (4,096) | 0 of 174 / 11 of 301 | 32,768 (8,192) |
| SimpleQA Verified | 32,768 (4,096) | not asked / 0 of 25 | as before |

A model served on the board needs as much context a slot to sit them: 83,968
tokens for GPQA or OTIS with thinking on, 86,016 for HLE, 114,688 for
ARC-AGI-2; a smaller one is refused in words before anything is asked. Each
import says how many of the answers that ran out end in a loop (the last
passage repeating), and each cell shows the share that ran out, beside its
score.
