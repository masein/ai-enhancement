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
# architectures, the image) — 17c: and llama/LICENSE (llama.cpp's) and
# llama/NOTICE (the CUDA libraries packed, and the EULA they come under). A box
# needs an NVIDIA driver for CUDA 12.8.
#
#     scripts/build_llama_tarball.sh ~/llama.cpp-teraformer llama-server-cuda12.8.tar.gz
#
# Docker on this machine (sudo docker when plain docker isn't allowed), and no
# GPU: nothing is run, only compiled. It fails, saying why, when git can't
# read the checkout or the build doesn't know its commit.
set -euo pipefail

src="${1:?usage: build_llama_tarball.sh <llama.cpp checkout> <out.tar.gz>}"
out="${2:?usage: build_llama_tarball.sh <llama.cpp checkout> <out.tar.gz>}"
image="${LLAMA_BUILD_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
archs="${LLAMA_CUDA_ARCHS:-80;86;89;90;120}"

src="$(cd "$src" && pwd)"
[ -f "$src/CMakeLists.txt" ] || { echo "$src isn't a llama.cpp checkout" >&2; exit 1; }
# 17b: a checkout git can't read stops here, in git's own words
if ! commit="$(git -C "$src" rev-parse HEAD 2>&1)"; then
  echo "git can't read $src: $commit" >&2
  echo "(for \"dubious ownership\": git config --global --add safe.directory $src)" >&2
  exit 1
fi
# 17c: read whole, never through head: under pipefail, head closing the pipe
# early killed git with SIGPIPE and the script exited 141 with no word (on a
# tree with many changed or untracked files)
# 17c: a file's mode isn't a change to the source (the fork's checkout on the
# server: 2,934 files, modes only, 0 insertions and 0 deletions)
if ! changes="$(git -C "$src" -c core.fileMode=false status --porcelain 2>&1)"; then
  echo "git can't list $src's changes: $changes" >&2
  exit 1
fi
[ -z "$changes" ] || echo "note: $src has changes not committed — recorded in VERSION" >&2
# 17b: the server's docker needs sudo; this one, when plain docker isn't allowed
DOCKER="${DOCKER:-docker}"
if ! $DOCKER info >/dev/null 2>&1; then DOCKER="sudo docker"; fi
echo "building llama-server at ${commit:0:12} for sm ${archs} in ${image} ($DOCKER)" >&2

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat > "$work/build.sh" <<'IN'
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake git build-essential >/dev/null
cp -a /src /build && cd /build && rm -rf build-tarball
# 17b: a copy owned by another user is "dubious" to git, and llama.cpp then
# builds as "0 (unknown)": marked safe, so the build knows its commit
git config --global --add safe.directory '*'
# 17c: the build container has no driver: libcuda.so.1 is the box's, so the
# link leaves its symbols to it (as llama.cpp's .devops/cuda.Dockerfile does).
# And never this machine's own CPU: GGML_NATIVE would build for the server's,
# and a rented box's could die of "Illegal instruction"
cmake -S . -B build-tarball -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
      -DGGML_NATIVE=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" >/dev/null
cmake --build build-tarball --target llama-server -j"$(nproc)"
info=$(find build-tarball -name build-info.cpp -print -quit)
if [ -z "$info" ] || grep -q 'LLAMA_COMMIT = "unknown"' "$info" || grep -q 'LLAMA_BUILD_NUMBER = 0;' "$info"; then
  echo "the build doesn't know its commit (see $info): nothing was packed" >&2
  exit 1
fi
# 17d: guarded — under set -e a grep that finds nothing ended the build without a word
if ! grep -E 'LLAMA_(BUILD_NUMBER|COMMIT) =' "$info" > /out/build-info; then
  echo "the build's $info names no build number or commit: nothing was packed" >&2
  exit 1
fi
mkdir -p /out/llama/bin /out/llama/lib
cp -L build-tarball/bin/llama-server /out/llama/bin/
# 17d: "+", not ";" — find then says when a copy failed, and the build stops
find build-tarball -name "*.so*" \( -type f -o -type l \) -exec cp -P -t /out/llama/lib/ {} +
for l in libcudart.so libcublas.so libcublasLt.so; do
  cp -P /usr/local/cuda/lib64/${l}* /out/llama/lib/
done
# 17b: and every other library they need but the C library's and the driver's
# (libgomp, libstdc++, libgcc_s) — a box's image may hold other versions
export LD_LIBRARY_PATH=/out/llama/lib:/usr/local/cuda/lib64
# (17c: ldd exits non-zero for a file that isn't a dynamic object; under
# pipefail that would end the script without a word — its list is what counts)
{ ldd /out/llama/bin/llama-server /out/llama/lib/*.so* 2>/dev/null || true; } | awk '/=> \//{print $3}' \
  | sort -u | while read -r so; do
    case "$(basename "$so")" in
      libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libcuda.so*|libnvidia*) ;;
      *) [ -e "/out/llama/lib/$(basename "$so")" ] || cp -L "$so" /out/llama/lib/ \
           || { echo "couldn't pack $so: nothing was packed" >&2; exit 1; } ;;
    esac
  done
nvcc --version | tail -1 > /out/llama/.cuda
# 17c: llama.cpp's licence travels with it
cp /build/LICENSE /out/llama/LICENSE
IN
$DOCKER run --rm -v "$src":/src:ro -v "$work":/out -e ARCHS="$archs" "$image" bash /out/build.sh
$DOCKER run --rm -v "$work":/out "$image" chown -R "$(id -u):$(id -g)" /out
{
  echo "commit $commit"
  echo "uncommitted_changes $([ -z "$changes" ] && echo no || echo yes)"
  sed -e 's/^.*LLAMA_BUILD_NUMBER = \([0-9]*\);.*/build \1/' -e 's/^.*LLAMA_COMMIT = "\([^"]*\)";.*/build_commit \1/' "$work/build-info"
  echo "cuda_archs $archs"
  echo "image $image"
  echo "cuda $(cat "$work/llama/.cuda")"
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$work/llama/VERSION"
rm -f "$work/llama/.cuda"
# 17c: what else it carries, and under what — needed for a copy shared beyond
# the private repository, harmless for private use
names() { for f in "$@"; do [ -e "$f" ] && printf '  %s\n' "$(basename "$f")"; done; return 0; }
lib="$work/llama/lib"
{
  echo "llama-server, built from llama.cpp at the commit in VERSION. llama.cpp is under the MIT"
  echo "licence: LICENSE, beside this file. Its own libraries in lib/:"
  names "$lib"/libllama* "$lib"/libggml* "$lib"/libmtmd*
  echo
  echo "lib/ also holds NVIDIA CUDA libraries, copied from the $image image the build ran in:"
  names "$lib"/libcudart.so* "$lib"/libcublas.so* "$lib"/libcublasLt.so*
  echo "They are NVIDIA's, redistributed under the NVIDIA CUDA Toolkit End User License"
  echo "Agreement (https://docs.nvidia.com/cuda/eula/), whose Attachment A lists them among"
  echo "the files that may be distributed with an application."
  echo
  echo "The other libraries in lib/ are the image's Ubuntu 22.04 packages', each under its own"
  echo "licence (/usr/share/doc/<package>/copyright in that image) — the GCC runtime libraries"
  echo "(libgomp, libstdc++, libgcc_s) under the GPL v3 with the GCC Runtime Library Exception."
} > "$work/llama/NOTICE"
tar -C "$work" -czf "$out" llama
echo "$out · $(du -h "$out" | cut -f1) · sha256 $(sha256sum "$out" | cut -d' ' -f1)"
