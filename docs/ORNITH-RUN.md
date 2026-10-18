# Ornith-1.5-35B-A3B (Q4_K_M) on the Frontier set (22)

A colleague asked us to benchmark **Ornith-1.5-35B-A3B**, a third-party model
(MIT licence), in its 4-bit GGUF:
<https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF>. We run it the way
step1000 ran: plan L, 21 RTX 5090 boxes, thinking on, our Frontier token
limits. **It isn't the team's model, so it never runs on the lda runtime:**
not the lda build, not `LOOKAHEAD_MOE_ROUTER_NORM`. It runs on upstream
llama.cpp with its own card's settings, as "Q4, earlier run" ran on Qwen's.

Its numbers are ours, measured here. Its card's numbers (GPQA Diamond 89.2,
HLE 25.6 with no tools, each the average of five runs) are published, and
are never ranked or averaged with ours.

## The file

| | |
|---|---|
| Repository | `ornith-ai/Ornith-1.5-35B-A3B-GGUF`, public, commit `12393612fd4f730ff5aadc23e9b8f9648aa49ceb` (24 Aug 2026) |
| File | `Ornith-1.5-35B-Q4_K_M.gguf`, the only 4-bit file |
| Size | 21,713,463,040 bytes (21.7 GB) |
| sha256 | `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` |
| Architecture | `qwen35moe`: 256 experts, 8 a token, 41 blocks, a full-attention layer every 4th (as Qwen3.6-35B-A3B) |

The size and sha256 are Hugging Face's own (its LFS metadata at that
commit). Each box fetches the file straight from the public repository at
that commit, with no token; no copy goes to the private repository. Plan L's
box line pins it by name (`frontier_box.PLAN_PINS`): a file of that name with
another sha256 is refused before anything is asked.

## The runtime, `card-ornith`

`scripts/frontier.py` RUNTIMES, the same on every box; each step's line says
every value:

- **llama.cpp: upstream release `b11555`** (commit `1cf7767765b8`, 11 Oct
  2026), built from that tag. The runner image holds no llama-server of its
  own. Upstream's own Linux CUDA 12.8 build is made on Ubuntu 24.04 and needs
  glibc 2.38; the runner image is Ubuntu 22.04 (glibc 2.35), where it stops at
  "GLIBC_2.38 not found". So the tag is built by our script in CUDA 12.8 on
  Ubuntu 22.04, as the lda tarball was, with the release's own CCCL (v3.4.3)
  and, as the release, no NCCL (one GPU a server), for sm_120 (the RTX 5090
  and the RTX PRO 6000). It loads `qwen35moe`: its
  `libllama` names the architecture, as the lda build's does for Qwen3.6.
- **Sampling:** temperature 1.0, top-p 0.95, top-k 20 (the card's settings
  for its benchmarks), min-p 0.0, presence penalty 0, repeat penalty 1.0 — the
  server's defaults and in every request.
- **The chat template:** the GGUF's own, with `--jinja`. It is the card's
  modified Qwen template (`chat_template.jinja` in `ornith-ai/Ornith-1.5-35B-A3B`
  at `10fbf86f`) with Unsloth's fixes for system and developer messages and
  tool calls; the two render a benchmark's messages (one user turn, or HLE's
  system and user turns, thinking on or off) the same, character for
  character. The box checks the server serves that template (its sha256,
  `f55f52930aa8…`) before a question.
- `-ctk q8_0 -ctv q8_0 --flash-attn on`; thinking on; no reasoning budget, no
  speculative decoding, no `LOOKAHEAD_*` variable.
- **Token limits:** 81,920; 65,536 on MATH Level 5; 32,768 on MMLU-Pro and
  SimpleQA Verified.

Before anything is fetched, the box refuses: another model on this runtime,
this model on any other runtime or none, a `LOOKAHEAD_*` variable, a build
that reads one (the lda build does), and a llama-server whose `--version` or
whose tarball's VERSION isn't b11555 at `1cf776776` with nothing uncommitted.
The import refuses a bundle asked any other way.

## O0. The tarball (once, on the server, about 15 minutes)

**Built and uploaded (18 Oct):** `llama-server-b11555.tar.gz`, 616 MiB, sha256
`79af365a8bd7918a9591fc378221adfdbc3daef3f299788790c867e825b76409`, in the private repository at commit
`33b3ea8c8d33f70975eed7dd50f943e6419c3f3b`. Plan L's box line pins it by name
(`frontier_box.PLAN_PINS`), beside the lda tarball: a file of that name with
another sha256 is refused before anything is asked. The next one, the same
way — a full clone, not a shallow one: the build's number is the count of
its commits (11555).

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-upstream && git -C ~/llama.cpp-upstream checkout b11555
```

```bash
cd ~/benchmarks/aienh && LLAMA_CUDA_ARCHS=120 LLAMA_CMAKE_ARGS="-DGGML_CUDA_CCCL_VERSION=v3.4.3 -DGGML_CUDA_NCCL=OFF -DLLAMA_OPENSSL=OFF" scripts/build_llama_tarball.sh ~/llama.cpp-upstream ~/llama-server-b11555.tar.gz
```

Its last line gives its size and sha256. Then into the private repository,
beside the lda tarball:

```bash
hf upload <you>/evalboard-private ~/llama-server-b11555.tar.gz llama-server-b11555.tar.gz
```

The lines below fetch it at that commit (`@33b3ea8c8d33…`), so every box
fetches the same file, whatever is uploaded later; each box also checks its
build and commit.

## O1. The pilot, on the first box

Rent one box as plan L's (docs/REMOTE-RUNS.md, L2): image
`ghcr.io/masein/evalboard-runner:<this phase's tag>`, **Disk 100 GB**, an
**RTX 5090**, max CUDA 12.8 or newer, inet down ≥ 1,000 Mbps. Then:

```bash
ssh -p <port> root@<host>
```

```bash
tmux new -s f
```

```bash
cd /app
```

The read token: GPQA Diamond's and MMLU-Pro's questions, and the tarball in
the private repository, need it (the Ornith repository doesn't):

```bash
read -rs HF_TOKEN && export HF_TOKEN
```

The pilot — 20 GPQA Diamond and 5 MMLU-Pro questions, thinking on, each
asked once, at plan L's 8 slots:

```bash
python scripts/frontier_box.py L1 --pilot --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
```

It checks the card, prints the runtime's every value, fetches and checks the
tarball (its build) and the GGUF (its sha256), the memory and the template,
then a line an answer, in this shape (the numbers made up):

```
GPQA Diamond 3 of 20 (question <its id>): thinking 21,406 characters · 6,170 tokens in all · read C · right C ✓ · 412 s
```

— the thinking's length, the answer the board read, the right one, its
seconds. Then each benchmark's right answers and pace, plan L's hours at
GPQA Diamond's pace here, and **its last line: whether the thinking and the
answer came back apart** — every answer that finished, with its thinking in
the server's `reasoning_content` and its answer in the reply, no think tags
in it. An answer that ran out of room isn't judged.

The pilot writes no bundle and no progress file (into
`/workspace/Ornith-1.5-35B-A3B-Q4_K_M/pilot`, which the fetch never reads).
Send me its lines; if the last says "separated correctly", rent the other
20. The same box then runs L1, with the files it has already:

## O2. The box lines, L1–L21

One label a box, as step1000's. Each starts as O1's box does (rent, ssh,
tmux, `cd /app`, the token), then its line:

1. **L1** — Humanity's Last Exam · shard 1/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L1 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

2. **L2** — Humanity's Last Exam · shard 2/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L2 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

3. **L3** — Humanity's Last Exam · shard 3/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L3 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

4. **L4** — Humanity's Last Exam · shard 4/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L4 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

5. **L5** — Humanity's Last Exam · shard 5/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L5 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

6. **L6** — Humanity's Last Exam · shard 6/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L6 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

7. **L7** — Humanity's Last Exam · shard 7/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L7 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

8. **L8** — Humanity's Last Exam · shard 8/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L8 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

9. **L9** — Humanity's Last Exam · shard 9/9 · 8 slots · about 5.5 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L9 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

10. **L10** — MMLU-Pro · shard 1/3 · 8 slots · about 5.9 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L10 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

11. **L11** — MMLU-Pro · shard 2/3 · 8 slots · about 5.9 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L11 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

12. **L12** — MMLU-Pro · shard 3/3 · 8 slots · about 5.9 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L12 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

13. **L13** — GPQA Diamond, MATH Level 5 · shard 1/4 · 8 slots · about 5.4 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L13 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

14. **L14** — GPQA Diamond, MATH Level 5 · shard 2/4 · 8 slots · about 5.4 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L14 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

15. **L15** — GPQA Diamond, MATH Level 5 · shard 3/4 · 8 slots · about 5.4 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L15 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

16. **L16** — GPQA Diamond, MATH Level 5 · shard 4/4 · 8 slots · about 5.4 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L16 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

17. **L17** — OTIS Mock AIME 2024–2025 · shard 1/3 · 8 slots; then SimpleQA Verified · 8 slots · about 5.2 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L17 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

18. **L18** — OTIS Mock AIME 2024–2025 · shard 2/3 · 8 slots · about 4.2 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L18 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

19. **L19** — OTIS Mock AIME 2024–2025 · shard 3/3 · 8 slots · about 4.2 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L19 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

20. **L20** — ARC-AGI-2 · shard 1/2 · 5 slots · about 5.8 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L20 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

21. **L21** — ARC-AGI-2 · shard 2/2 · 5 slots · about 5.8 h at plan L's paces

   ```bash
   python scripts/frontier_box.py L21 --as served/Ornith-1.5-35B-A3B-Q4_K_M --gguf hf://ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf --server hf://<you>/evalboard-private@33b3ea8c8d33f70975eed7dd50f943e6419c3f3b/llama-server-b11555.tar.gz --runtime card-ornith
   ```

Plan L at the 6 Oct pilot's paces (the phone build, which routes 4 experts a
token; Ornith-1.5 routes 8): 114.1 box-hours, the longest box 5.9 h — $57.05
at $0.50 an hour. Ornith-1.5's own pace isn't measured until the pilot, which
prints plan L at its GPQA Diamond pace.

Once every step on a box reads whole, its scores by code:

```bash
python scripts/score_bundle.py /workspace/Ornith-1.5-35B-A3B-Q4_K_M
```

## O3. Fetch and import, from the server

The fetch of L3 (docs/REMOTE-RUNS.md), with Ornith's sha256 and each box with
its label:

```bash
cd ~/benchmarks/aienh && python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 --sha served/Ornith-1.5-35B-A3B-Q4_K_M=42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f --every 3m <host 1>:<port 1>=L1 <host 2>:<port 2>=L2
```

The board doesn't serve Ornith-1.5 yet, so its first bundle comes home and the
fetch prints the line to import it by hand. Type it once, with its board name
and the file's sha256; the import registers it with the runtime card-ornith,
and the fetch imports the rest by itself:

```bash
cd ~/benchmarks/aienh && sudo docker compose exec -T bench python scripts/import_remote.py ~/benchmarks/bundles/<the bundle the fetch names> --by masein --register "Ornith-1.5-35B-A3B Q4_K_M (card settings)" --file-sha256 42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f
```

HLE and SimpleQA Verified are graded on the board later, with your yes and an
estimate first. No paid API is called from a box.

## On the reports

- **Step checkpoints:** "Ornith-1.5 Q4_K_M" among ours (colour slot 5, in the
  picker, on Models ▸ Latest), noted "measured here, on its card's settings;
  not the team's model"; and "Ornith-1.5-35B-A3B, its card", a published row
  with the card's GPQA Diamond 89.2 and HLE 25.6, each linked to the card at
  `12393612` and noted as a 5-run average.
- **Where step4000 runs out of room:** in the picker.

docs/REPORTS.md says how they show.
