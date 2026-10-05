#!/usr/bin/env bash
# 17: llama-server from a llama.cpp checkout, as a tarball a rented GPU runs
# (scripts/remote_gguf.py --server). Built once, by masein, from the fork's
# checkout at the commit build-lda was built from — the source never leaves
# this machine; the tarball goes to the private Hugging Face repository beside
# the GGUFs, never to the repo, the public runner image or a bundle.
#
# Built inside nvidia/cuda's 12.8 devel image on Ubuntu 22.04 (the runner
# image's CUDA and C library), for the A100 to the RTX 5090 (sm 80, 86, 89,
# 90, 120), with shared libraries. The tarball holds llama/bin/llama-server,
# llama/lib/ (the build's libraries and CUDA's runtime, cuBLAS and cuBLASLt)
# and llama/VERSION (the commit, whether the tree had changes, the
# architectures, the image). A box needs an NVIDIA driver for CUDA 12.8.
#
#     scripts/build_llama_tarball.sh ~/llama.cpp-teraformer llama-server-cuda12.8.tar.gz
#
# Docker on this machine, and no GPU: nothing is run, only compiled.
set -euo pipefail

src="${1:?usage: build_llama_tarball.sh <llama.cpp checkout> <out.tar.gz>}"
out="${2:?usage: build_llama_tarball.sh <llama.cpp checkout> <out.tar.gz>}"
image="${LLAMA_BUILD_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
archs="${LLAMA_CUDA_ARCHS:-80;86;89;90;120}"

src="$(cd "$src" && pwd)"
[ -f "$src/CMakeLists.txt" ] || { echo "$src isn't a llama.cpp checkout" >&2; exit 1; }
commit="$(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)"
dirty="$(git -C "$src" status --porcelain 2>/dev/null | head -c1 | wc -c | tr -d ' ')"
[ "$dirty" = "0" ] || echo "note: $src has changes not committed — recorded in VERSION" >&2
echo "building llama-server at ${commit:0:12} for sm ${archs} in ${image}" >&2

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
docker run --rm -v "$src":/src:ro -v "$work":/out -e ARCHS="$archs" "$image" bash -euo pipefail -c '
  apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake git build-essential >/dev/null
  cp -a /src /build && cd /build && rm -rf build-tarball
  cmake -S . -B build-tarball -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" >/dev/null
  cmake --build build-tarball --target llama-server -j"$(nproc)"
  mkdir -p /out/llama/bin /out/llama/lib
  cp -L build-tarball/bin/llama-server /out/llama/bin/
  find build-tarball -name "*.so*" \( -type f -o -type l \) -exec cp -P {} /out/llama/lib/ \;
  for l in libcudart.so libcublas.so libcublasLt.so; do
    cp -P /usr/local/cuda/lib64/${l}* /out/llama/lib/
  done
  nvcc --version | tail -1 > /out/llama/.cuda
  chown -R '"$(id -u):$(id -g)"' /out'
{
  echo "commit $commit"
  echo "uncommitted_changes $([ "$dirty" = "0" ] && echo no || echo yes)"
  echo "cuda_archs $archs"
  echo "image $image"
  echo "cuda $(cat "$work/llama/.cuda")"
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$work/llama/VERSION"
rm -f "$work/llama/.cuda"
tar -C "$work" -czf "$out" llama
echo "$out · $(du -h "$out" | cut -f1) · sha256 $(sha256sum "$out" | cut -d' ' -f1)"
