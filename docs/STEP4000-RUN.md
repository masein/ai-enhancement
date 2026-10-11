# step4000 on rented boxes: what the team does

We run the Frontier benchmarks (thinking on) on three models with the same
runtime, and compare them: the UD-Q4_K_XL file, the BF16 file, and your
step4000. The runtime is your `lda` branch's llama-server with
`LOOKAHEAD_MOE_ROUTER_NORM=1`, our sampling (temperature 1.0, top-k 20, top-p
0.95, min-p 0.0, presence penalty 1.5, repeat penalty 1.0, repeat-last-n 64),
no reasoning budget and no speculative decoding.

masein rents each box on vast.ai and adds your public key to it. You put two
things on it: your **build** (on every box, all three models) and, on the
step4000 boxes, **the GGUF**. Then you tell masein the two paths, and he takes
your key off the box before he types anything of his. Nothing else of ours is
shared, and nothing of yours leaves the box: the results hold answers and logs,
the sha256 of your binary, libraries and GGUF, and your build's commit and
branch, never the files themselves.

## 1. Send masein your public key

The `.pub` file's one line (`ssh-ed25519 AAAA… you@machine`). He adds it to
each box and sends you its address, for example:

```bash
ssh -p 41234 root@203.0.113.7
```

## 2. Your build, once, then the same on every box

The results of one model are merged across boxes only when every box ran the
same build: the board refuses to merge shards whose `llama-server` or
libraries differ. So **build once and copy the same directory to every box**.

**What a box needs:** a directory holding `bin/llama-server` and its shared
libraries (`libllama.so`, `libggml*.so`, `libmtmd.so`, …), built for CUDA 12.8
on Ubuntu 22.04 (glibc 2.35) or older, for **sm_120** (the RTX 5090 and the
RTX PRO 6000 are both compute capability 12.0). CUDA's own runtime and cuBLAS
are on the box already (the runner image's, CUDA 12.8): you don't copy them.
If your build carries its own copies, those are used.

Built in CUDA's own image on your machine (Docker, no GPU needed), from your
checkout of the `lda` branch:

```bash
docker run --rm -v ~/llama.cpp-lda:/src:ro -v ~/lda-out:/out nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake git build-essential >/dev/null && cp -a /src /build && cd /build && git config --global --add safe.directory "*" && cmake -S . -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && cmake --build build --target llama-server -j"$(nproc)" && mkdir -p /out/lda-build && cp -a build/bin build/CMakeCache.txt /out/lda-build/ && printf "commit %s\nbranch %s\n" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" > /out/lda-build/VERSION'
```

(The `VERSION` file says the commit and branch of a directory copied without
its `.git`; the box records it.)

(`--allow-shlib-undefined`: the image has no GPU driver; the box's driver
provides `libcuda.so.1`. `GGML_NATIVE=OFF`: never your machine's own CPU
instructions, which a rented box's CPU may lack.)

Then onto each box, with only `ssh` and `tar`:

```bash
tar -C ~/lda-out/lda-build -czf - . | ssh -p <port> root@<host> 'mkdir -p /workspace/lda-build && tar -C /workspace/lda-build -xzf -'
```

**Or build on one box** and copy that directory to the others the same way.
The runner image has no compiler or CUDA toolkit; on the box:

```bash
apt-get update && apt-get install -y --no-install-recommends build-essential cmake git wget ca-certificates
```

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 cuda-driver-dev-12-8 libcublas-dev-12-8
```

Your checkout onto the box (with its `.git`, so the commit and branch are
recorded):

```bash
tar -C ~/llama.cpp-lda --exclude=./build -czf - . | ssh -p <port> root@<host> 'mkdir -p /workspace/llama.cpp && tar -C /workspace/llama.cpp -xzf -'
```

Then, on the box:

```bash
cd /workspace/llama.cpp && cmake -S . -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && cmake --build build --target llama-server -j"$(nproc)"
```

The build directory is then `/workspace/llama.cpp/build`.

**Check it runs on the box** (it prints its version, then the card):

```bash
LD_LIBRARY_PATH=/workspace/lda-build/bin /workspace/lda-build/bin/llama-server --version
```

The box's own run checks the same before anything else, and refuses in one
line when `bin/llama-server` isn't there, when it can't load a library, when
it sees no CUDA card, or when the CMakeCache.txt beside it says it was built
for other cards.

## 3. The GGUF, on the step4000 boxes

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/step4000' && scp -P <port> step4000.gguf root@<host>:/workspace/step4000/
```

A GGUF in parts: every part in that folder, and give masein the first
(`…-00001-of-0000N.gguf`). It must fit a 32 GB card with room for 8 slots of
86,016 tokens: about the size of the UD-Q4_K_XL file (23 GB) at most. The box
works out the memory from the file's header before it starts, and says.

## 4. Tell masein the two paths, and log out

For example: build `/workspace/lda-build`, GGUF
`/workspace/step4000/step4000.gguf`. These are what his line is given:

```
--server-dir /workspace/lda-build --gguf /workspace/step4000/step4000.gguf
```

He then takes your key off the box. Leave nothing running there.

## What is recorded

Every bundle (a box's answers, which go to masein's private Hugging Face
repository and later to the board) records: `llama-server --version`; the
sha256 of `llama-server` and of each library beside it, and one identity for
them all; the commit and branch of the `.git` the build directory is in, or
of its `VERSION` file for a copied directory; the CUDA architectures it was
built for; the GGUF's name, size and
sha256; the runtime's every flag and variable; the GPU. Never your source, the
binary, the libraries or the GGUF.
