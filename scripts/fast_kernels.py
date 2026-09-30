#!/usr/bin/env python3
"""12v: the hybrid models' fast paths, in the image.

transformers runs a Mamba2 hybrid (Granite-4.0-H) and a gated-delta-net model
(Qwen3.5) on a slow path of its own when three packages are missing, and says
so in the run's log: "The fast path is not available … Falling back to the
naive implementation". On that path one tensor of each of Granite's Mamba
layers takes 1.5 GiB for every 256 tokens it reads at once, and Granite-4.0-
H-1B ran out of GPU memory on MMLU-Pro at a batch of one (#132, #144, #147).

  mamba_ssm, causal_conv1d   compiled CUDA kernels. Not built here: the image
                             is a runtime one, with no CUDA compiler. Hugging
                             Face's kernels-community publishes them built for
                             this image's torch and CUDA (torch 2.11, cu128,
                             Blackwell among the architectures), and
                             transformers 5.5.3 is written against those same
                             builds. Fetched when the image is built, each
                             file pinned by its repo's commit, its size and
                             its hash (scripts/fast_kernels.json); any other
                             bytes are refused
  fla                        flash-linear-attention's core (fla-core, pure
                             Python over Triton), pinned in
                             requirements-kernels.txt; the Dockerfile installs
                             it beside the two

All three go into one folder, /opt/fast-kernels, which a line in Python's
site-packages puts on the path of every Python in the container — unless
FAST_KERNELS=0 is in its environment. That is the way back with no rebuild:
set it in .env and `docker compose up -d`, and every run is on the slow paths
again, as before 12v.

Triton compiles a small C extension for each kernel the first time it runs.
The base image has the C compiler and Python's headers for that; --check
builds one as Triton does, so an image without them fails its build.

    python scripts/fast_kernels.py                 fetch into /opt/fast-kernels, and the path line
    python scripts/fast_kernels.py --check         what is there, that it imports, and the compiler
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib
import importlib.util
import json
import os
import site
import sys
import tempfile
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("fast_kernels.json")
DEST = Path("/opt/fast-kernels")
SWITCH, OFF = "FAST_KERNELS", ("0", "off", "no", "false")
PTH = "evalboard_fast_kernels.pth"
MODULES = ("mamba_ssm", "causal_conv1d", "fla")
# what transformers 5.5.3 takes from each (modeling_granitemoehybrid.py,
# modeling_qwen3_5.py): a name that is gone is a fast path that is gone
NAMES = {
    "mamba_ssm.ops.triton.selective_state_update": ("selective_state_update",),
    "mamba_ssm.ops.triton.ssd_combined": ("mamba_chunk_scan_combined",
                                          "mamba_split_conv1d_scan_combined"),
    "causal_conv1d": ("causal_conv1d_fn", "causal_conv1d_update"),
    "fla.modules": ("FusedRMSNormGated",),
    "fla.ops.gated_delta_rule": ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
}
CHUNK, TRIES, TIMEOUT = 1 << 20, 6, 120


def manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def url_of(kernel: dict, variant: str, rel: str) -> str:
    return f"https://huggingface.co/{kernel['repo']}/resolve/{kernel['commit']}/build/{variant}/{rel}"


def _hasher(want: dict):
    """a file is pinned by its sha256 where the Hub keeps it as a large file,
    and by git's own hash of its bytes otherwise"""
    return hashlib.sha256() if "sha256" in want else hashlib.sha1(b"blob %d\0" % want["size"])


def _pinned(want: dict) -> str:
    return want.get("sha256") or want["blob"]


def file_ok(path: Path, want: dict) -> bool:
    try:
        if path.stat().st_size != want["size"]:
            return False
        h = _hasher(want)
        with open(path, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
        return h.hexdigest() == _pinned(want)
    except OSError:
        return False


def fetch_file(url: str, out: Path, want: dict, opener=urllib.request.urlopen,
               tries: int = TRIES) -> None:
    """the pinned bytes at `out`, or nothing: a connection that drops is taken
    up where it left off (the larger kernel is 610 MB), and bytes that aren't
    the pinned ones are refused"""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    h, got, why = _hasher(want), 0, ""
    with open(part, "wb") as fh:
        for _ in range(tries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={got}-"} if got else {})
                with opener(req, timeout=TIMEOUT) as r:
                    if got and getattr(r, "status", 206) != 206:
                        # the whole file again, not the rest of it: start over
                        h, got = _hasher(want), 0
                        fh.seek(0)
                        fh.truncate()
                    while got <= want["size"] and (chunk := r.read(CHUNK)):
                        fh.write(chunk)
                        h.update(chunk)
                        got += len(chunk)
                break
            except (OSError, http.client.HTTPException) as e:
                why = f" ({e})"
    if got != want["size"] or h.hexdigest() != _pinned(want):
        part.unlink(missing_ok=True)
        raise SystemExit(f"{out.name} is not the pinned one: {got:,} bytes, {h.hexdigest()} "
                         f"(want {want['size']:,} bytes, {_pinned(want)}){why}")
    part.replace(out)


def fetch(dest: Path = DEST, man: dict | None = None, opener=urllib.request.urlopen) -> list[str]:
    """every pinned file under dest/<package>/, fetched unless it is there
    already; the lines of what was done"""
    man = man or manifest()
    said = []
    for pkg, k in man["kernels"].items():
        new = 0
        for rel, want in k["files"].items():
            out = dest / pkg / rel
            if not file_ok(out, want):
                fetch_file(url_of(k, man["variant"], rel), out, want, opener)
                new += 1
        size = sum(f["size"] for f in k["files"].values())
        said.append(f"{pkg}: {len(k['files'])} files, {size:,} bytes, {k['repo']} at "
                    f"{k['commit'][:12]} ({man['variant']}) · {new} fetched")
    return said


def verify(dest: Path = DEST, man: dict | None = None) -> list[str]:
    """every pinned file that isn't at `dest` as pinned"""
    man = man or manifest()
    return [f"{pkg}/{rel}" for pkg, k in man["kernels"].items()
            for rel, want in k["files"].items() if not file_ok(dest / pkg / rel, want)]


def pth_line(dest: Path = DEST) -> str:
    """the line in site-packages that puts `dest` on every Python's path —
    last, so it never stands in front of anything — unless FAST_KERNELS says off"""
    return (f"import os, sys; os.environ.get({SWITCH!r}, '1').strip().lower() in {OFF!r} "
            f"or sys.path.append({str(dest)!r})\n")


def install_pth(site_dir: Path, dest: Path = DEST) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    p = site_dir / PTH
    p.write_text(pth_line(dest), encoding="utf-8")
    return p


def check_imports() -> list[str]:
    """each package imports, and has the names transformers takes from it"""
    said = []
    for mod, names in NAMES.items():
        m = importlib.import_module(mod)
        gone = [n for n in names if not callable(getattr(m, n, None))]
        if gone:
            raise SystemExit(f"{mod} has no {', '.join(gone)}: transformers' fast path needs it")
    for top in MODULES:
        m = importlib.import_module(top)
        said.append(f"{top} {getattr(m, '__version__', '')}".strip() + f" at {Path(m.__file__).parent}")
    return said


_PROBE = ("#define PY_SSIZE_T_CLEAN\n#include <Python.h>\n"
          "static struct PyModuleDef m = {PyModuleDef_HEAD_INIT, \"evalboard_probe\", 0, -1, 0};\n"
          "PyMODINIT_FUNC PyInit_evalboard_probe(void) { return PyModule_Create(&m); }\n")


def check_compiler() -> str:
    """Triton builds a small C extension for each kernel the first time it
    runs (triton.runtime.build): a C compiler and Python's headers, or every
    fast path fails at its first call. Built here as Triton builds it"""
    from triton.runtime.build import _build
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "evalboard_probe.c"
        src.write_text(_PROBE, encoding="utf-8")
        so = _build("evalboard_probe", str(src), tmp, [], [], [], [])
        spec = importlib.util.spec_from_file_location("evalboard_probe", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return "Triton's C extension builds and loads"


def seen_by_transformers() -> dict[str, bool]:
    """what transformers itself makes of them: true only beside a GPU"""
    from transformers.utils import import_utils as iu
    return {"mamba_ssm": iu.is_mamba_ssm_available(), "causal_conv1d": iu.is_causal_conv1d_available(),
            "fla": iu.is_flash_linear_attention_available()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--site", type=Path, default=None,
                    help="where the path line goes (Python's own site-packages)")
    ap.add_argument("--check", action="store_true",
                    help="fetch nothing: the files as pinned, the imports, the compiler")
    a = ap.parse_args(argv)
    if not a.check:
        for line in fetch(a.dest):
            print(line)
        p = install_pth(a.site or Path(site.getsitepackages()[0]), a.dest)
        print(f"on every Python's path through {p} (off with {SWITCH}=0)")
        return 0
    if os.environ.get(SWITCH, "1").strip().lower() in OFF:
        print(f"{SWITCH}={os.environ[SWITCH]}: the fast kernels are off; every run is on the slow paths")
        return 0
    wrong = verify(a.dest)
    if wrong:
        raise SystemExit(f"not as pinned under {a.dest}: {', '.join(wrong[:8])}"
                         + (f" and {len(wrong) - 8} more" if len(wrong) > 8 else ""))
    print(f"every pinned file is at {a.dest}")
    for line in check_imports():
        print(line)
    print(check_compiler())
    seen = seen_by_transformers()
    print("transformers sees: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in seen.items())
          + ("" if all(seen.values()) else " (it counts them only beside a GPU)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
