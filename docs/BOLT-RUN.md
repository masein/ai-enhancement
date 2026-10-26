# NVIDIA Nemotron 3.5 Bolt E4B, measured on our own V100s (29)

masein, 25 Oct: dgx-03 runs DGX OS 6 (Ubuntu 22.04.5, driver 580.178.04,
which takes CUDA 12.8), and Docker sees its four V100s (32 GB, sm_70). We
benchmark **NVIDIA Nemotron 3.5 Bolt E4B** there the way our own builds were
benchmarked, so it can be compared with everything in our reports and has its
own measured rows. Its published numbers (27f) stay as they are, apart.

## The files

| | Q4_K_M, the phone build | Q8_0, standing in for BF16 |
|---|---|---|
| Repository | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` at `1db54d5b12533ff1bcaa8e42cf4c667988a1f1be` | `ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` at `5f03edf5d3bdfe6859513573802d509fa46636de` |
| File | `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` | `NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf` |
| Size | 6,180,446,208 bytes | 8,031,245,312 bytes |
| sha256 | `c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612` | `64504effb9a71a991e487ac4e4cf827485e78bf577efb0a6edf67ce7e1ba84a8` |

Both are of NVIDIA's BF16 (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` at
`49843eed`). The Q4_K_M is **NVIDIA's own file**: Unsloth's repository has
no Q4_K_M (its 4-bit files are UD quants), and NVIDIA's is the one its card
benchmarks as the 4 GB build. The Q8_0 is ggml-org's conversion of the same
BF16 (Unsloth has no Q8_0 either). A V100 has no native BF16, so the Q8_0
stands in for it, and the board says so wherever it is a base ("of our
Q8_0, standing in for BF16"). Sizes and sha256 are Hugging Face's own (LFS
metadata at those commits); plan D pins both by name
(`frontier_box.PLAN_PINS["D"]`).

Its architecture is Gemma 4 E4B's (`gemma4`): 42 layers, eight heads and
two KV heads, full attention every sixth layer with 512-wide heads, the others
a 512-token sliding window of 256-wide heads, the last 18 layers sharing
earlier layers' caches. Its context is 65,536 tokens (`max_position_embeddings`,
and the card's 64K in and out).

## The runtime, `card-bolt`

`scripts/frontier.py` RUNTIMES; each step's line says every value:

- **llama.cpp: upstream `b11555`** (`1cf7767765b8`), card-ornith's release,
  which reads `gemma4` (NVIDIA's own instructions use b11459, older). Built for
  **sm_70 under CUDA 12.8** (CUDA 13 has no Volta) in the runner image's
  **V100 tag**, `ghcr.io/masein/evalboard-runner:v100`, at `/opt/llama`
  (`scripts/build_llama_v100.sh`, the Dockerfile's `runner-v100` stage). The
  5090's image is unchanged. A box runs it with `--server-dir /opt/llama`; its
  VERSION and `--version` are checked against the runtime before a GGUF is
  fetched.
- **Sampling:** NVIDIA's card gives temperature 1.0 and top-p 0.95 with
  thinking on or off. The model's shipped default, the BF16 repository's
  `generation_config.json` (at `49843eed`), adds **top-k 64** (as our Gemma 4
  preset has it); what neither sets is neutral — min-p 0, no presence or
  repeat penalty — never llama-server's defaults. Each value's source is kept
  in the run's settings (`sampling_from` in each benchmark's `setup.json`, in
  the bundle).
- **Thinking:** the chat template's `enable_thinking` (it writes `<|think|>`
  into the first system turn; the thinking comes back in
  `<|channel>thought … <channel|>`, which llama.cpp b11555 splits into
  `reasoning_content`). The card says thinking is on by default, but **the
  template's own default is off**, so every request says it, on or off. On
  for every benchmark; off for IFEval and IFBench as well (plan I's rule).
- **The chat template:** the GGUF's own (`--jinja`), the BF16 repository's
  `chat_template.jinja` (sha256 `35b7d0641b6f…`), checked against the server's
  before a question.
- **Its 64K context:** each slot holds 65,536 tokens. Each answer's limit is
  the benchmark's (81,920 on the hard ones), or 65,536 minus that question's
  prompt (as the server counts it) when that is smaller — kept in the answer's
  line (`limit`, `capped`). A prompt that fills the context alone is written as
  no room (`finish: length`, counted wrong, never asked). An answer that runs
  out of room is wrong, as for everyone. On the board Bolt's cells say "limit
  capped by its 64K context" on hover, and the Frontier note says it once.
- **Attention:** `-ctk q8_0 -ctv q8_0 --flash-attn on`, as every model we
  run. Whether a V100 runs flash attention with a q8_0 cache at Bolt's
  512-wide heads is checked on the card before a question (a 4,096-token
  prompt, then short ones side by side); refused, the box runs the runtime's
  next launch — f16 cache with flash attention, then flash attention off —
  says so, and records what was refused and why (`setup.json` ▸
  `server.attention`). The import takes the runtime's launch or one of its
  fallbacks, nothing else.
- No reasoning budget, no speculative decoding, no `LOOKAHEAD_*` variable.

## Our own boxes: dgx-03's GPUs

A box is one GPU, labelled **D3.0 … D3.3** (station D3's GPUs 0–3), in its
own container started with `--gpus device=N`, as **the board's account** on
the station (in the `docker` group, no sudo, its work folder on `/raid`).
The station is colleagues' too:

- **Never:** sudo, a reboot, the driver, Docker's own service, NetBird,
  anything installed outside the work folder, or anyone else's process. The
  launcher only reads `nvidia-smi` and waits.
- **Before a run starts** on a GPU, `nvidia-smi` must show no other process
  on it; if one does, nothing starts, and the GPU is named.
- **While it runs**, the GPU is read before each benchmark and every minute
  while asking. Someone else's process appearing pauses it: the answers in
  flight finish and are kept, nothing else starts, its llama-server stops, Runs
  shows the step **paused** and why, and it carries on by itself once the GPU
  is free.
- The container runs as the account's own user, with no capabilities and no
  new privileges; it sees the station's processes (to name them) but can't
  touch them.
- **No price:** Runs says "own box", never $0. Every speed and energy number
  names its card (V100) and is never compared with the 5090 rows as one
  hardware.
- The station's address is the server's own config, never the repo, the
  mirror or a bundle: `EVALBOARD_OWN_D3` in the board's `.env`, as
  `<account>@<address>:<ssh port>:<work folder>/workspace`.

### On the station (as the board's account)

The launcher comes from the image itself; nothing is installed:

```bash
docker run --rm ghcr.io/masein/evalboard-runner:v100 cat scripts/own_box.sh > own_box.sh
```

```bash
bash own_box.sh check D3.0
```

```bash
bash own_box.sh start D3.0
```

The questions are fetched on the station, into its work folder only, with
masein's token typed by him (never `hf auth login`):

```bash
read -rs HF_TOKEN && export HF_TOKEN
```

### The pilot (D3.0), before anything long

```bash
bash own_box.sh pilot D3.0 -- --as served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M --gguf hf://nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB@1db54d5b12533ff1bcaa8e42cf4c667988a1f1be/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --server-dir /opt/llama
```

It checks the GPU is free, then the card check (its probe line names the
card, sm_70, the attention it runs and its speeds), then 5 questions of every
benchmark with thinking on, then 5 of IFEval and IFBench with thinking off:
a line an answer (its thinking's length, the answer read, its limit when
capped, ran out, a loop), each benchmark's pace and tokens a second, whether
the thinking came back apart (and none with thinking off), and last the
estimate: the slots of 64K that fit, tokens a second at 8 slots, and the hours
for both builds across the four GPUs. Nothing long starts before masein's
yes. Each thinking setting keeps its own folder (29.1):
`<build>/pilot/thinking-on` and `<build>/pilot/thinking-off`, each with its
`pilot-thinking-<on|off>.json` and a line an answer; the fetch never reads
them. A GGUF already in `/workspace/files` with its sha256 is not fetched
again.

### dgx-03's slow link: the Q8_0 fetched ahead

dgx-03 reads Hugging Face at 0.1–0.4 MB/s (the Q4_K_M's 6.2 GB took about
90 minutes), so the Q8_0 is fetched before the full run, in a container of its
own (no GPU; `own_box.sh stop` never touches it), resumable — the same line
again carries on from where it stopped:

```bash
docker run -d --name evalboard-q8-fetch --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges -e HOME=/workspace/home -v /raid/evalboard/workspace:/workspace ghcr.io/masein/evalboard-runner:v100 python -c "from huggingface_hub import hf_hub_download as get; get('ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF', 'NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf', revision='5f03edf5d3bdfe6859513573802d509fa46636de', local_dir='/workspace/files')"
```

`docker logs --tail 3 evalboard-q8-fetch` shows how far it is; it is done when
the file is 8,031,245,312 bytes. Stopped, `docker rm evalboard-q8-fetch` and
the same line carries on. Faster, where the board's server reads Hugging Face
quickly: fetch it there and copy it into the same folder with `rsync
--partial` (resumable too). The full run checks its sha256 against plan D's
pin either way, and Hugging Face's client takes a copied file whose sha256
is the published one without fetching it again.

### The full run (after masein's yes)

Plan D (`python scripts/frontier_box.py --list D`): each GPU asks a quarter of
every benchmark, longest first — HLE, MMLU-Pro, MATH Level 5, ARC-AGI-2,
OTIS (×8), GPQA Diamond (×4), SimpleQA Verified, then IFEval and IFBench with
thinking on and off — so the four end together whatever the V100's pace.
On each GPU, Q4_K_M's line first, then Q8_0's (`--then`):

```bash
bash own_box.sh start D3.1
```

```bash
bash own_box.sh run D3.1 -- --as served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M --gguf hf://nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB@1db54d5b12533ff1bcaa8e42cf4c667988a1f1be/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --server-dir /opt/llama --then --as served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0 --gguf hf://ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF@5f03edf5d3bdfe6859513573802d509fa46636de/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf --server-dir /opt/llama
```

The same for D3.0, D3.2 and D3.3 (`start`, then `run` with its label).
`bash own_box.sh log D3.1` follows a line; `bash own_box.sh stop D3.1` removes
its container (and nothing else).

### Fetch and import, from the server

```bash
python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 --own D3 --sha served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M=c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612 --sha served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0=64504effb9a71a991e487ac4e4cf827485e78bf577efb0a6edf67ce7e1ba84a8 --every 10m
```

The station is named D3 on every line; its steps go to Runs as own boxes. The
first bundle of each build is imported by hand once, registering the row
under its name — exactly "NVIDIA Nemotron 3.5 Bolt E4B · Q4_K_M" (and
"· Q8_0"), which puts it in the Open group — as the fetch's line says
(`--register "NVIDIA Nemotron 3.5 Bolt E4B · Q4_K_M"`). An own station is
done once every step is whole and home; it is never "safe to destroy".

### Mobile and Standard: served to the board

The phone builds also have SimpleQA Verified, MobileAIBench, Mobile-MMLU-Pro
and Mobile-MMLU rows, asked by the board's own runner of a served model.
Bolt is served from one GPU (not one running a line), its port on the
station's loopback alone (each request says its own thinking; `--thinking`
is only what the line must give):

```bash
bash own_box.sh serve D3.3 18090 -- --as served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M --gguf hf://nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB@1db54d5b12533ff1bcaa8e42cf4c667988a1f1be/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --server-dir /opt/llama --runtime card-bolt --thinking on --slots 8
```

On the server, in tmux, an SSH tunnel as the board's account from the docker
bridge to it:

```bash
bash scripts/own_tunnel.sh D3 18090 --key ~/.ssh/id_ed25519
```

Then, on the board, Add a model ▸ served, with the row's own name ("NVIDIA
Nemotron 3.5 Bolt E4B · Q4_K_M": the same row), its address
`http://host.docker.internal:18090/v1`, and Test a model with the Mobile and
Standard sets the phone builds have. Someone else's process on that GPU
stops the server until it is free; the board's run stops with its answers
kept and is started again. The same for the Q8_0 after.

## Grading

As for our builds: by code where the benchmark's code grades (GPQA, OTIS,
MATH, MMLU-Pro, ARC-AGI-2, IFEval, IFBench); HLE and SimpleQA Verified by the
board's judge. A paid judge is never run without masein's yes and an
estimate.

## Part 6: the board's account on dgx-03

Added by masein's Ansible, never by the board: the server's public key (its
`.pub` only), an account for the board in the `docker` group with no sudo,
and its work folder on `/raid`.
