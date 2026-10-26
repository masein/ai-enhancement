#!/usr/bin/env bash
# 29: llama-server for the V100 (sm_70), built inside the Dockerfile's
# llama-v100 stage (nvidia/cuda's 12.8 devel image on Ubuntu 22.04, the
# runner image's CUDA and C library) and copied into the runner image's V100
# tag at /opt/llama — upstream llama.cpp at the commit runtime card-bolt
# names (scripts/frontier.py), checked out and checked here, then built and
# packed as scripts/build_llama_tarball.sh packs a box's tarball: shared
# libraries, no NCCL, CCCL v3.4.3, no web page, never this machine's CPU;
# bin/llama-server, lib/ (its libraries, CUDA's runtime, cuBLAS and cuBLASLt,
# and what else they need but the C library's and the driver's), VERSION (the
# commit, the build, the architectures, the image), LICENSE (llama.cpp's) and
# NOTICE (the CUDA libraries packed, and the EULA they come under). CUDA 13
# has no Volta: 12.8 it is.
#
#     LLAMA_COMMIT=<40 hex> LLAMA_CUDA_ARCHS=70 bash scripts/build_llama_v100.sh /out/llama
set -euo pipefail

out="${1:?usage: build_llama_v100.sh <out dir>}"
commit="${LLAMA_COMMIT:?the llama.cpp commit to build}"
archs="${LLAMA_CUDA_ARCHS:-70}"
image="${LLAMA_BUILD_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
extra="-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DGGML_CUDA_NCCL=OFF -DLLAMA_OPENSSL=OFF"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "LLAMA_COMMIT: a full commit, 40 hex" >&2; exit 1; }

src=/tmp/llama.cpp
rm -rf "$src"
# every commit (for the build number llama.cpp counts), the files at this one alone
git clone --quiet --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$src"
git -C "$src" -c advice.detachedHead=false checkout --quiet "$commit"
[ "$(git -C "$src" rev-parse HEAD)" = "$commit" ] || { echo "not at $commit" >&2; exit 1; }
git config --global --add safe.directory '*'

cd "$src"
# the image has no driver: libcuda.so.1 is the box's (as llama.cpp's
# .devops/cuda.Dockerfile builds)
cmake -S . -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
      -DGGML_NATIVE=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$archs" $extra >/dev/null
cmake --build build --target llama-server -j"$(nproc)"
info=$(find build -name build-info.cpp -print -quit)
if [ -z "$info" ] || grep -q 'LLAMA_COMMIT = "unknown"' "$info" || grep -q 'LLAMA_BUILD_NUMBER = 0;' "$info"; then
  echo "the build doesn't know its commit (see $info): nothing was packed" >&2
  exit 1
fi

mkdir -p "$out/bin" "$out/lib"
cp -L build/bin/llama-server "$out/bin/"
find build \( -name "*.so" -o -name "*.so.*" \) \( -type f -o -type l \) -exec cp -P -t "$out/lib/" {} +
for l in libcudart.so libcublas.so libcublasLt.so; do
  cp -P /usr/local/cuda/lib64/${l}* "$out/lib/"
done
export LD_LIBRARY_PATH="$out/lib:/usr/local/cuda/lib64"
{ ldd "$out/bin/llama-server" "$out"/lib/*.so* 2>/dev/null || true; } | awk '/=> \//{print $3}' \
  | sort -u | while read -r so; do
    case "$(basename "$so")" in
      libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libcuda.so*|libnvidia*) ;;
      *) [ -e "$out/lib/$(basename "$so")" ] || cp -L "$so" "$out/lib/" \
           || { echo "couldn't pack $so: nothing was packed" >&2; exit 1; } ;;
    esac
  done
cp LICENSE "$out/LICENSE"
{
  echo "commit $commit"
  echo "uncommitted_changes no"
  sed -n -e 's/^.*LLAMA_BUILD_NUMBER = \([0-9]*\);.*/build \1/p' \
         -e 's/^.*LLAMA_COMMIT = "\([^"]*\)";.*/build_commit \1/p' "$info"
  echo "cuda_archs $archs"
  echo "cmake_args $extra"
  echo "web_ui no"
  echo "image $image"
  echo "cuda $(nvcc --version | tail -1)"
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/VERSION"
names() { for f in "$@"; do [ -e "$f" ] && printf '  %s\n' "$(basename "$f")"; done; return 0; }
{
  echo "llama-server, built from llama.cpp at the commit in VERSION. llama.cpp is under the MIT"
  echo "licence: LICENSE, beside this file. Its own libraries in lib/:"
  names "$out"/lib/libllama* "$out"/lib/libggml* "$out"/lib/libmtmd*
  echo
  echo "lib/ also holds NVIDIA CUDA libraries, copied from the $image image the build ran in:"
  names "$out"/lib/libcudart.so* "$out"/lib/libcublas.so* "$out"/lib/libcublasLt.so*
  echo "They are NVIDIA's, redistributed under the NVIDIA CUDA Toolkit End User License"
  echo "Agreement (https://docs.nvidia.com/cuda/eula/), whose Attachment A lists them among"
  echo "the files that may be distributed with an application."
  echo
  echo "The other libraries in lib/ are the image's Ubuntu 22.04 packages', each under its own"
  echo "licence (/usr/share/doc/<package>/copyright in that image) — the GCC runtime libraries"
  echo "(libgomp, libstdc++, libgcc_s) under the GPL v3 with the GCC Runtime Library Exception."
} > "$out/NOTICE"
rm -rf "$src"
echo "$out: llama-server at ${commit:0:12}, build $(sed -n 's/^build \([0-9]*\)$/\1/p' "$out/VERSION"), sm $archs"
