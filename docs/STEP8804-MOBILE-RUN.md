# step8804 on the Mobile tab and DeviceMark (30)

step8804's Mobile and DeviceMark rows, measured the way
`served/Qwen3.6-35B-A3B-k4-LDA-phone-build` and
`served/Qwen3.6-35B-A3B-Q4-original-k-8` were: on the team's RTX 5090 with the
experts of 21 layers on the CPU, asked by the board over its server, one
question at a time. Each setting is copied from those rows' own run logs
(#153, #173, #180, #184, #186, #187).

## What is asked

| Order | Run | Thinking | Questions | Copied from |
|---|---|---|---|---|
| 1 | DeviceMark's server speed (part `speed`) | – | 3 trials | #160 |
| 2 | Mobile-MMLU-Pro (part `mmlu`) | off | 9,497 asked, 9,462 kept | #186 |
| 3 | Mobile-MMLU, the full set (part `mmlu_full`), non-commercial | off | the 16,186 less those asked in 2 | #187 |
| 4 | MobileAIBench: HotpotQA, SQL, Dolly, CNN/DailyMail, XSum | off | 5 × 1,000 | #184 |
| 5 | DeviceMark replica v1 (part `full`) | off | 596 | #153 |
| 6 | DeviceMark replica v1 (part `full`) | on | 596 | #173 |
| 7 | MobileAIBench: HotpotQA and SQL | on | 2 × 1,000 | #180 |

Every request: temperature 0, at most 4,096 tokens (the thinking counted),
`enable_thinking` said either way. MobileAIBench: stop at four newlines and
each set's own system line. DeviceMark: seed 0, scoring v2, a cap of 4,096.
Trust & safety and the Standard tab aren't asked: neither row it is set beside
has them.

**What differs**, said in its "How it's served" (the hover on its name):

- **The runtime.** step8804's own, lda-2610: the lda branch at `a4a76a751`
  (build 9954) with `LOOKAHEAD_MOE_ROUTER_NORM=1`, as its Frontier rows. The
  two rows it is set beside ran two commits earlier, at `91428471f` (build
  9952), without the router norm.
- **The file**: its own, 15,776,378,720 bytes; theirs UD-Q4_K_XL, 22.8 GB.
- **Thinking on, MobileAIBench**: Q4's run #180 said nothing about thinking
  (its server's default, which thinks) and was stopped after 565 of SQL's
  1,000; step8804's says it on. Step D checks the two make the same prompt.

## A. The file

`step8804.gguf` in the private repository (`<masein>/evalboard-private`), as
its Frontier rows were asked with it: 15,776,378,720 bytes, sha256
`c7a719f8c37d85bcd24f5a0bf103323e0102938be01dea3e0992465923188f3a`. On the
server, beside step4000's and step1000's:

```bash
df -h ~
```

```bash
read -rs HF_TOKEN && export HF_TOKEN
```

```bash
hf download <masein>/evalboard-private step8804.gguf --revision <its commit> --local-dir ~
```

Its bytes and sha256 — both must be the ones above:

```bash
stat -c %s ~/step8804.gguf && sha256sum ~/step8804.gguf
```

## B. The build

The lda tarball, `llama-server-lda-a4a76a7.tar.gz` (sha256
`5b538290644d67139c6c996fdaa3e04cce82fd0b343cce66a4ff234648afb618`, sm_120:
the RTX 5090), unpacked into a folder of its own:

```bash
ls -l ~/llama-server-lda-a4a76a7.tar.gz || hf download <masein>/evalboard-private llama-server-lda-a4a76a7.tar.gz --local-dir ~
```

```bash
sha256sum ~/llama-server-lda-a4a76a7.tar.gz && mkdir -p ~/llama-lda-a4a76a7 && tar -xzf ~/llama-server-lda-a4a76a7.tar.gz -C ~/llama-lda-a4a76a7 && cat ~/llama-lda-a4a76a7/llama/VERSION
```

Its `libllama.so` must read the router norm (a count of 1 or more):

```bash
grep -c LOOKAHEAD_MOE_ROUTER_NORM ~/llama-lda-a4a76a7/llama/lib/libllama.so
```

## C. The card, then its server

What runs on the card first: the judge (gemma-vllm, about 12.9 GiB) and
nothing of ours. Never stop or change the judge:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Port 8095 free (it must print nothing):

```bash
ss -ltn 'sport = :8095' | grep LISTEN
```

On the docker bridge's address, where the board's container reaches it as
`host.docker.internal`, with the launch the k4-LDA and Q4 rows ran with and
no sampling flags (the board's requests carry their own):

```bash
H=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1) && L=~/llama-lda-a4a76a7/llama && LOOKAHEAD_MOE_ROUTER_NORM=1 LD_LIBRARY_PATH=$L/lib nohup $L/bin/llama-server -m ~/step8804.gguf --host "$H" --port 8095 -ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 --parallel 1 --jinja > ~/step8804-serve.log 2>&1 &
```

When its log says it is listening, the card again (the two `nvidia-smi`
lines): about 1.5 GB of it must still be free.

```bash
tail -n 5 ~/step8804-serve.log
```

## D. Its thinking switch

The prompt its template makes with the switch unsaid, said on, and said off.
The first two must be the same, the third another:

```bash
H=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1); for k in '' ',"chat_template_kwargs":{"enable_thinking":true}' ',"chat_template_kwargs":{"enable_thinking":false}'; do curl -s "http://$H:8095/apply-template" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Invented question?"}]'"$k"'}' | python3 -c 'import sys, json; print(repr(json.load(sys.stdin)["prompt"][-48:]))'; done
```

## E. The board

Deploy (30 gives a model registered by its import a server here under its own
id; registering it again by name would make another row):

```bash
cd ~/benchmarks/aienh && git pull origin main && sudo EVALBOARD_BUILD=$(git rev-parse --short HEAD) docker compose up -d --build
```

Its address, the board's own token and address read from `.env`:

```bash
cd ~/benchmarks/aienh && B=$(sudo grep '^BIND=' .env | cut -d= -f2- | tr -d "\"'") && T=$(sudo grep '^SUBMIT_TOKEN=' .env | cut -d= -f2- | tr -d "\"'")
```

```bash
curl -s -X PUT "http://$B:8899/api/served/served/step8804-lda-2610/address" -H "X-Token: $T" -H 'Content-Type: application/json' -d '{"base_url": "http://host.docker.internal:8095/v1", "by": "masein", "flags": "-ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 --parallel 1 --jinja", "env": "LOOKAHEAD_MOE_ROUTER_NORM=1", "how": "step8804.gguf on the team'"'"'s RTX 5090 + CPU experts: the lda branch at a4a76a751 (build 9954) with LOOKAHEAD_MOE_ROUTER_NORM=1 — runtime lda-2610, as its Frontier rows — two commits after the build the k4-LDA and Q4 (original, k=8) rows on Mobile and DeviceMark ran on (91428471f, build 9952, without the router norm) · -ngl 99 --n-cpu-moe 21 --no-mmap -c 16384 --parallel 1 --jinja, the default f16 KV cache, no sampling flags (each request its own greedy settings)"}' | python3 -c 'import sys, json; d = json.load(sys.stdin); m = d.get("model") or {}; print(d.get("detail") or (m.get("id"), m.get("base_url"), m.get("pin")))'
```

## F. The runs, in order

The speed first; the queue asks the rest one after the other:

```bash
q() { curl -s -X POST "http://$B:8899/api/submissions" -H "X-Token: $T" -H 'Content-Type: application/json' -d "{\"hf_id\": \"served/step8804-lda-2610\", \"submitter\": \"masein\", \"note\": \"30: step8804 as the k4-LDA and Q4 rows\", $1}"; echo; }
```

```bash
q '"suite": "devicemark", "part": "speed"'
```

```bash
q '"suite": "mobile", "part": "mmlu"' && q '"suite": "mobile", "part": "mmlu_full"' && q '"suite": "mobile"' && q '"suite": "devicemark", "part": "full"' && q '"suite": "devicemark", "part": "full", "thinking": true' && q '"suite": "mobile", "thinking": true, "tasks": ["mab_hotpotqa", "mab_sql"]'
```

## G. Afterwards

Its server stopped by its own PID (never the judge's):

```bash
sudo ss -ltnp 'sport = :8095'
```

```bash
kill <its PID>
```
