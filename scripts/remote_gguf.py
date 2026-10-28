#!/usr/bin/env python3
"""17: the Frontier benchmarks asked of a GGUF on a rented GPU, with one
command — and back to the board as one file (scripts/import_remote.py takes
it in, onto the served model's row: --as).

It runs inside the board's runner image (ghcr.io/masein/evalboard-runner),
which holds no llama.cpp: the server comes as a tarball the person points it
at (--server: scripts/build_llama_tarball.sh makes one), fetched with the
person's HF_TOKEN from a repo of theirs, or a path on the box. The GGUF the
same way (--gguf). Neither is ever in the image or the bundle.

Then it does what a served model's run on the board does, through the same
code: it starts llama-server on this box (127.0.0.1 only) and waits until it
is healthy, registers it in a database of its own under --out as the served
model --as, and runs the board's runner (service/runner.py, run_submission →
service/frontier.py): the same prompts, the GGUF's own chat template
(--jinja), the model card's sampling, the thinking switch said out loud, one
seed a question and run. Its answers are kept as they land: run the same
command again after the process dies or the box is stopped, and it carries on
from the next unanswered question.

--shard i/n asks every n-th question from the i-th of each task, so n boxes
share a run, one bundle each; the server takes them in any order.

The bundle (frontier-<served id>-thinking-<on|off>[-shard-i-of-n].tar.gz)
records the GGUF's name, size and sha256 — the server refuses one whose file
isn't the file registered for --as — the llama-server build and its sha256,
the launch flags and environment, the GPU, and each task's settings. It holds
the answers of each task answered whole, and never the GGUF, the server, or a
key.

17b: the box runs the setup it says it runs. A port something already answers
on is refused before anything starts; once healthy, the child must still be
running and its /props must name this file. The build (the binary's sha256
and its commit), the flags and the environment are kept in --out's state and
a resume with others is refused (the slots may change; each session's are
recorded). --parity asks the parity check's 50 questions instead
(scripts/frontier_parity.py): the pilot's first step.

17c: the GGUF and the tarball are fetched into one folder every run on the
box shares (--files; a folder "files" beside --out), so the full run after
the parity check fetches and hashes nothing again. The parity file opens
with what answered: the file's name, size and sha256, and the launch.

20: --server-dir runs a llama.cpp build directory already on the box (the
team's lda branch, built there or copied in) in place of a tarball: its
bin/llama-server, with its own folders on LD_LIBRARY_PATH, refused in one
line when it isn't there or doesn't run on this box and card. The bundle
records its --version, the binary's sha256 and its libraries', and the
commit and branch of a .git near it. --runtime lda-2610 sets a named
runtime's environment, its server's flags and the sampling every request
carries (scripts/frontier.py RUNTIMES), each said on the step's line.

22: --runtime card-ornith, a third-party model on its card's settings: the
runtime names its llama.cpp (upstream b11555: the server's --version and the
tarball's VERSION are checked before the GGUF is fetched), its model (a
model a runtime names runs on it alone), the variables it never gives its
server (LOOKAHEAD_*, nor runs a build that reads them) and the chat template
it asks with (the GGUF's own, its sha256 checked against the server's /props
before a question). --pilot asks 20 GPQA Diamond and 5 MMLU-Pro questions
once each, thinking on: a line each (the thinking's length, the answer the
board read, the right one, the seconds), each benchmark's pace, and last
whether every answer that finished came back with its thinking apart
(reasoning_content) and its answer in the reply — no bundle, and no progress
file, so the fetch never takes it for a step.

23c: the card, measured (scripts/card_check.py). Once llama-server is up
and before the step asks anything, the card check: the card as nvidia-smi
says (its power limit and default, its PCIe link now and at most) and a
speed probe against this server — prompt and output tokens a second for one
user at 512 and 16,384 tokens, and for eight at once — in the progress file
and the bundle's setup. And a sampler beside the server, from its start to
its end: the card's power, use, memory and throttling every five seconds, to
the step's gpu.jsonl (in the bundle), its running means in the progress
file, and the step's watt-hours shared among its benchmarks by the seconds
their answers took. --price, the box's dollars an hour, kept in the setup:
the board works out dollars an answer from it, and never without it.

29: --own, one GPU of our own DGX Station (a label D3.0: station D3's GPU 0,
its container started by scripts/own_box.sh with --gpus device=0). The GPU
is colleagues' too: the step doesn't start while another process uses it
(scripts/gpu_free.py, the GPU named), and while it asks, someone else's
process appearing pauses it — the answers in flight finish, nothing else
starts, its server stops, its progress says "paused" and why, and it exits
75 (frontier_box.py waits for the GPU to be free, then carries on). No price
(an own box costs nothing to rent: the board says "own box"). --runtime
card-bolt: a model whose context is 64K — each slot holds 65,536 tokens, each
answer's limit is the benchmark's or what is left of 65,536 after its prompt
(kept in its line), and the server's flash attention with a q8_0 cache is
checked on the card before a question: refused, the runtime's next launch is
run, said and recorded. --pilot-each 5: the pilot asks 5 questions of each
benchmark given (--only; all when none), thinking on or (IFEval and IFBench)
off.

The token is typed, never on the command line (where the shell's history
would keep it):

    read -rs HF_TOKEN && export HF_TOKEN

    python scripts/remote_gguf.py --as served/qwen3-6-35b-a3b-q4 \\
        --gguf hf://you/private-ggufs/Qwen3.6-35B-A3B-Q4_K_M.gguf \\
        --server hf://you/private-llama/llama-server-cuda.tar.gz \\
        --based-on Qwen/Qwen3.6-35B-A3B --thinking on --out /workspace/run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier as fb  # noqa: E402
import card_check as cc  # noqa: E402
import remote_bundle as rb  # noqa: E402
from remote_run import Out, board_commit  # noqa: E402

SUITE = "frontier"
STATE = "remote_gguf.json"
# 17f: what the step is doing, for frontier_fetch.py to read without
# importing anything — its label, the model, the benchmark it asks, its line,
# when it started and its sessions; never the box's address
PROGRESS = "progress.json"
OWN_LOG = "remote_gguf.log"
GPU_FILE = "gpu.jsonl"              # 23c: the sampler's readings, in the bundle
SERVER_LOG = "llama-server.log"
# what this script sets itself: given in --flags, refused
OWN_FLAGS = {"-m", "--model", "--host", "--port", "-c", "--ctx-size", "-np", "--parallel",
             "-hf", "--hf-repo", "-mu", "--model-url", "--api-key", "--api-key-file"}
ENV = {"BENCH_ROOT": "BENCH_ROOT", "RESULTS_ROOT": "OUT_ROOT", "LOGS_DIR": "LOGS",
       "DB_PATH": "SERVICE_DB", "FRONTIER_SHARD": "FRONTIER_SHARD",
       "FRONTIER_SCORE_AFTER_RUN": "FRONTIER_SCORE_AFTER_RUN", "FRONTIER_WHERE": "FRONTIER_WHERE",
       "SERVED_CONCURRENCY": "SERVED_CONCURRENCY", "SERVED_TIMEOUT_S": "SERVED_TIMEOUT_S",
       "SERVED_RETRY_S": "SERVED_RETRY_S", "GPU_POLL_S": "GPU_POLL_S",
       "FRONTIER_ASK_WRITTEN_OFF": "FRONTIER_ASK_WRITTEN_OFF"}
WHERE = "a rented GPU"
OWN_WHERE = "our own GPU"           # 29: a DGX Station's, shared with colleagues
PAUSED = 75                         # 29: the exit of a step paused for someone else's process
# 29: a box label on our own station — D3.0, station D3's GPU 0
OWN_LABEL = re.compile(r"^(?P<station>[A-Z]\d{1,2})\.(?P<gpu>\d)$")


def own_box(label: str) -> dict | None:
    """29: {station, label, gpu} of an own box's label; None for another"""
    m = OWN_LABEL.match(label or "")
    return {"station": m["station"], "label": label, "gpu": int(m["gpu"])} if m else None


def settings(out: Path, slots: int, shard: tuple[int, int] | None,
             ask_again: bool = False) -> dict:
    bench = out / "bench"
    return {"BENCH_ROOT": bench, "RESULTS_ROOT": bench / "results", "LOGS_DIR": bench / "logs",
            "DB_PATH": bench / "service.sqlite3",
            "FRONTIER_SHARD": f"{shard[0]}/{shard[1]}" if shard else "",
            "FRONTIER_SCORE_AFTER_RUN": False,
            "FRONTIER_WHERE": OWN_WHERE if STEP.get("own") else WHERE,
            "SERVED_CONCURRENCY": slots,
            # a thinking answer of 81,920 tokens (17f's limits), at a slot's pace
            "SERVED_TIMEOUT_S": int(os.environ.get("SERVED_TIMEOUT_S") or 7200),
            "SERVED_RETRY_S": int(os.environ.get("SERVED_RETRY_S") or 300), "GPU_POLL_S": 10,
            "FRONTIER_ASK_WRITTEN_OFF": ask_again}


def configure(out: Path, slots: int, shard: tuple[int, int] | None,
              ask_again: bool = False) -> dict:
    """the board's config, pointed at --out: set in the environment before
    the service is imported, and on config itself when it already was"""
    vals = settings(out, slots, shard, ask_again)
    for k, v in vals.items():
        os.environ[ENV[k]] = "0" if v is False else str(v)
    from service import config, runner
    for k, v in vals.items():
        setattr(config, k, v)
    config.OUT_DIR = config.RESULTS_ROOT / "full"
    runner.LOCK = config.RESULTS_ROOT / ".run.lock"
    for d in (config.OUT_DIR, config.LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return vals


# ---------------------------------------------------------------------------
# the files: hf://[datasets/]<org>/<repo>[@<revision>]/<path>, or a path here
# ---------------------------------------------------------------------------

_HF = re.compile(r"^hf://(?P<type>datasets/|models/)?(?P<repo>[^/@\s]+/[^/@\s]+)"
                 r"(?:@(?P<rev>[^/\s]+))?/(?P<path>\S+)$")


def fetch(src: str, into: Path, say) -> Path:
    """the file `src` names, on this box: fetched from Hugging Face with
    HF_TOKEN (never printed), or the path given"""
    m = _HF.match(src.strip())
    if not m:
        p = Path(src).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"{src}: no such file on this box, and not hf://<org>/<repo>/<path>")
        return p
    import huggingface_hub as hh
    kind = "dataset" if m["type"] == "datasets/" else "model"
    # 18e point 19: the commit the file comes from, said — the revision given
    # (or main) read once, and the file fetched at that commit
    commit = repo_commit(hh, m["repo"], kind, m["rev"])
    say(f"fetching {m['path']} from {m['repo']}" + (f" at {m['rev']}" if m["rev"] else "")
        + (f": commit {commit}" if commit else ": its commit not said")
        + ("" if m["rev"] else f" — give @{commit} to fetch this one again" if commit else ""))
    try:
        got = hh.hf_hub_download(m["repo"], m["path"], repo_type=kind,
                                 revision=commit or m["rev"],
                                 token=os.environ.get("HF_TOKEN") or None, local_dir=str(into))
    except Exception as e:                          # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"{src} could not be fetched: {e}. A private repo needs "
                                  "HF_TOKEN set on this box (typed here, never stored)")) from None
    return Path(got).resolve()


def repo_commit(hh, repo: str, kind: str, rev: str | None) -> str:
    """18e point 19: the commit `rev` (main when none) is at in `repo` — ''
    when Hugging Face's client can't say"""
    try:
        info = hh.HfApi(token=os.environ.get("HF_TOKEN") or None).repo_info(
            repo, repo_type=kind, revision=rev or None)
        sha = str(getattr(info, "sha", "") or "")
    except Exception:                                   # noqa: BLE001 — said as not said
        return ""
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else ""


_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def split_parts(src: str) -> list[str]:
    """a split GGUF's every part, named as its first is (…-00001-of-00003.gguf);
    one name for a file in one piece"""
    m = _SPLIT.search(src)
    if not m:
        return [src]
    n = int(m.group(2))
    if int(m.group(1)) != 1:
        raise SystemExit(f"{src}: give the first part, …-00001-of-{n:05d}.gguf")
    return [src[:m.start()] + f"-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


split_sha = rb.split_sha                # 17i: one definition, the import's too


def sha256_cached(path: Path, state: dict, say, shared: Path | None = None) -> str:
    """a file's sha256, kept with its size and time so a second session
    doesn't read 20 GB again — 17c: and beside the shared files, so another
    --out on this box doesn't either"""
    st = path.stat()
    key = f"{path}:{st.st_size}:{int(st.st_mtime)}"
    known = state.setdefault("hashes", {})
    if key not in known and shared is not None:
        try:
            had = json.loads(shared.read_text(encoding="utf-8"))
            if isinstance(had.get(key), str):
                known[key] = had[key]
        except (OSError, ValueError, AttributeError):
            pass
    if key not in known:
        say(f"sha256 of {path.name} ({st.st_size / 1e9:.1f} GB)…")
        known[key] = rb.sha256_file(path)
        if shared is not None:
            try:
                had = json.loads(shared.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                had = {}
            had[key] = known[key]
            shared.write_text(json.dumps(had, indent=1), encoding="utf-8")
    return known[key]


def unpack_server(tarball: Path, into: Path) -> dict:
    """the llama-server tarball, unpacked under `into` (nothing outside it,
    no link out of it): {bin, libs, root}"""
    sha = rb.sha256_file(tarball)
    root = into / sha[:16]
    if not (root / ".unpacked").exists():
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:*") as tar:
            for m in tar.getmembers():
                parts = Path(m.name).parts
                if m.name.startswith("/") or ".." in parts or m.isdev():
                    raise SystemExit(f"{tarball.name} holds {m.name!r}: not unpacked")
                if (m.issym() or m.islnk()) and (m.linkname.startswith("/")
                                                 or ".." in Path(m.linkname).parts):
                    raise SystemExit(f"{tarball.name} links {m.name!r} outside itself: "
                                     "not unpacked")
            tar.extractall(root, **({"filter": "data"} if hasattr(tarfile, "data_filter")
                                    else {}))
        (root / ".unpacked").write_text(sha, encoding="utf-8")
    bins = sorted(p for p in root.rglob("llama-server") if p.is_file())
    if not bins:
        raise SystemExit(f"{tarball.name} holds no llama-server")
    exe = bins[0]
    exe.chmod(exe.stat().st_mode | 0o111)
    libs = sorted({str(p.parent) for p in root.rglob("*.so*") if p.is_file() or p.is_symlink()})
    return {"bin": exe, "libs": libs, "root": root, "tarball_sha256": sha}


def server_dir(path: str) -> dict:
    """20: a llama.cpp build directory on this box (--server-dir), as the team
    builds or copies it: its bin/llama-server, run with the folders holding
    its shared libraries on LD_LIBRARY_PATH — {bin, libs, root, lib_files,
    libs_sha256}. Refused in one line when llama-server isn't there"""
    root = Path(path).expanduser().resolve()
    exe = next((c for c in (root / "bin" / "llama-server", root / "llama-server")
                if c.is_file()), None)
    if exe is None:
        raise SystemExit(f"--server-dir {path}: no bin/llama-server there — give the llama.cpp "
                         "build directory, the one holding bin/llama-server and its libraries")
    if not os.access(exe, os.X_OK):
        raise SystemExit(f"--server-dir {path}: {exe.relative_to(root)} isn't executable "
                         f"(chmod +x {exe})")
    dirs = [exe.parent, *(d for d in (root / "lib", root / "lib64") if d.is_dir())]
    files: dict[str, str] = {}
    for d in dirs:
        for p in sorted(d.glob("*.so*")):
            real = p.resolve()
            if real.is_file() and p.name not in files:
                files[p.name] = rb.sha256_file(real)
    if not files:
        raise SystemExit(f"--server-dir {path}: no shared library (*.so) beside "
                         f"{exe.relative_to(root)} or in lib/ — copy the whole build directory")
    # one identity for the binary and its libraries: the server's code is in
    # them as much as in the binary
    ident = hashlib.sha256("".join(f"{n} {h}\n" for n, h in
                                   [("llama-server", rb.sha256_file(exe)), *sorted(files.items())])
                           .encode()).hexdigest()
    # CUDA's runtime and cuBLAS: the build's own when it carries them, else the
    # runner image's (torch's CUDA 12.8 packages), after the build's folders
    cuda = [] if any(n.startswith(("libcudart.so", "libcublas.so")) for n in files) \
        else cuda_lib_dirs()
    return {"bin": exe, "libs": [str(d) for d in dirs] + cuda, "root": root, "lib_files": files,
            "libs_sha256": ident, "cuda_libs": cuda}


def cuda_lib_dirs() -> list[str]:
    """20: the folders of CUDA's runtime and cuBLAS that come with torch in the
    runner image (its nvidia packages), [] where there are none"""
    import importlib.util
    out = []
    for mod in ("nvidia.cuda_runtime", "nvidia.cublas"):
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            spec = None
        for loc in (spec.submodule_search_locations or []) if spec else []:
            if (Path(loc) / "lib").is_dir():
                out.append(str(Path(loc) / "lib"))
    return out


def git_files(d: Path) -> dict | None:
    """the commit and branch a checkout's .git says, read from its files —
    the runner image has no git"""
    g = d / ".git"
    try:
        if g.is_file():                     # a worktree: "gitdir: <path>"
            g = (d / g.read_text(encoding="utf-8").split(":", 1)[1].strip()).resolve()
        head = (g / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, IndexError):
        return None
    if not head.startswith("ref: "):
        return {"commit": head or None, "branch": "HEAD"}
    ref = head[5:].strip()
    commit = None
    try:
        commit = (g / ref).read_text(encoding="utf-8").strip()
    except OSError:
        try:
            for ln in (g / "packed-refs").read_text(encoding="utf-8").splitlines():
                if ln.endswith(" " + ref):
                    commit = ln.split()[0]
        except OSError:
            pass
    return {"commit": commit, "branch": ref.removeprefix("refs/heads/")}


def git_of(root: Path) -> dict | None:
    """20: the build's commit and branch, when a .git is near its directory
    (it, or up to three folders above): {commit, branch, dirty, where} —
    dirty None when git isn't installed to tell"""
    for d in (root, *list(root.parents)[:3]):
        if not (d / ".git").exists():
            continue

        def git(*args: str) -> str | None:
            try:
                r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(d), *args],
                                   capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                return None
            return r.stdout.strip() if r.returncode == 0 else None
        commit = git("rev-parse", "HEAD")
        if commit is None:
            got = git_files(d) or {"commit": None, "branch": None}
            return {**got, "dirty": None, "where": str(d)}
        return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "where": str(d)}
    return None


def compute_cap() -> int | None:
    """the first card's CUDA compute capability as a number: 12.0 is 120"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.split()
        major, _, minor = out[0].partition(".")
        return int(major) * 10 + int(minor or 0)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def version_file(root: Path) -> dict:
    """a build packed as build_llama_tarball.sh packs one: its VERSION file's
    lines (commit, uncommitted_changes, build, cuda_archs, …), {} when none"""
    try:
        text = (root / "VERSION").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out = {}
    for ln in text.splitlines():
        k, _, v = ln.strip().partition(" ")
        if re.fullmatch(r"[a-z_]{1,40}", k) and v:
            out[k] = v.strip()[:200]
    return out


def cuda_archs(root: Path) -> list[str] | None:
    """the CUDA architectures the build was configured for (its
    CMakeCache.txt, or a packed build's VERSION), None when neither says"""
    try:
        text = (root / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"(?m)^CMAKE_CUDA_ARCHITECTURES:[A-Z]+=(.*)$", text)
        got = m.group(1) if m else ""
    except OSError:
        got = version_file(root).get("cuda_archs") or ""
    return [x for x in re.split(r"[;,\s]+", got) if x] or None


def archs_run_on(archs: list[str], cap: int) -> bool | None:
    """True when code built for `archs` runs on a card of compute capability
    `cap` — its own code, or PTX from an older one the driver compiles; None
    when the list can't tell (native, all)"""
    ok = False
    for a in archs:
        m = re.fullmatch(r"(\d+)([af]?)(?:-(real|virtual))?", a.strip())
        if not m:
            return None
        n, special, kind = int(m[1]), m[2], m[3]
        if n == cap and kind != "virtual":
            ok = True
        if not special and kind != "real" and n <= cap:
            ok = True
    return ok


def check_runs_here(srv: dict, env: dict, say, what: str = "") -> dict:
    """20: --server-dir's llama-server runs on this box and this card — its
    --version answers, it lists a CUDA device, and the architectures it was
    built for (its CMakeCache.txt) hold this card's. Refused in one line;
    {version, devices, cuda_archs, compute_cap} when it does. 18e point 18: a
    tarball's too, named by `what` (its file)"""
    exe, where = srv["bin"], srv["root"]
    what = what or f"--server-dir {where}"

    def first_line(text: str) -> str:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        bad = next((x for x in lines if re.search(r"(?i)error|cannot|failed|not found|"
                                                  r"insufficient|no cuda", x)), None)
        return rb.scrub(bad or (lines[0] if lines else "it printed nothing"))[:300]
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60,
                           env=env)
    except (OSError, subprocess.SubprocessError) as e:
        raise SystemExit(f"{what}: llama-server doesn't run here: {e}") from None
    text = (r.stdout + r.stderr).strip()
    if r.returncode != 0 or not re.search(r"version:\s*\d+", text):
        raise SystemExit(f"{what}: llama-server doesn't run here (exit "
                         f"{r.returncode}): {first_line(text)}")
    devices = None
    try:
        r = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True,
                           timeout=120, env=env)
        listed = (r.stdout + r.stderr)
        devices = re.findall(r"(?m)^\s*(CUDA\d+):\s*(.+?)\s*$", listed)
        if not devices and re.search(r"(?i)available devices", listed):
            raise SystemExit(f"{what}: llama-server sees no CUDA device on this "
                             f"box — {first_line(listed)}: a build without CUDA, or a driver "
                             "older than the CUDA it was built with")
        if not devices and r.returncode != 0 and not re.search(r"(?i)invalid argument|unknown "
                                                               r"argument|usage", listed):
            raise SystemExit(f"{what}: llama-server can't list this box's devices "
                             f"(exit {r.returncode}): {first_line(listed)}")
    except (OSError, subprocess.SubprocessError):
        devices = None
    cap, archs = compute_cap(), cuda_archs(where)
    if cap and archs:
        fits = archs_run_on(archs, cap)
        if fits is False:
            raise SystemExit(f"{what}: built for CUDA architectures "
                             f"{';'.join(archs)} (its CMakeCache.txt), and this card is "
                             f"{cap // 10}.{cap % 10} (sm_{cap}): it won't run here — build "
                             f"with -DCMAKE_CUDA_ARCHITECTURES={cap} (docs/STEP4000-RUN.md)")
    say(f"{what}: llama-server runs here"
        + (f" · sees {', '.join(f'{d} {n}' for d, n in devices)}" if devices else "")
        + (f" · built for {';'.join(archs)}" if archs else
           " · its CUDA architectures not said (no CMakeCache.txt or VERSION): checked when it "
           "loads")
        + (f", this card sm_{cap}" if cap else ""))
    return {"devices": [f"{d}: {n}" for d, n in devices or []], "cuda_archs": archs,
            "compute_cap": cap}


def parse_pins(given: list[str]) -> dict[str, str]:
    """20: --pin NAME=SHA256 — a file this box fetches or is given, by its
    name, and the sha256 it must have"""
    out = {}
    for x in given or []:
        name, _, sha = x.partition("=")
        if not name or "/" in name or not re.fullmatch(r"[0-9a-f]{64}", sha.strip().lower()):
            raise SystemExit(f"--pin {x!r}: NAME=SHA256, a file's name and its 64-hex sha256")
        out[name.strip()] = sha.strip().lower()
    return out


def pin_refused(name: str, sha: str, pins: dict) -> str:
    """'' when `name` has no pin or its sha256 is the pinned one; else why"""
    want = pins.get(name)
    if not want or want == sha:
        return ""
    return (f"{name} here has sha256 {sha[:16]}…, not the {want[:16]}… pinned for it (--pin): "
            "another file under that name — nothing was asked")


def build_reads(srv: dict, names: list[str]) -> list[str]:
    """20: the variables of `names` that llama-server and its libraries never
    mention — a runtime's variable a build doesn't read does nothing, and the
    run would say it ran with it. Its own libraries first (the lda branch's
    LOOKAHEAD_MOE_ROUTER_NORM is in libllama.so)"""
    files = [Path(srv["bin"])]
    own = [Path(d) for d in srv["libs"] if Path(d).is_relative_to(Path(srv["root"]))]
    libs = sorted({p.resolve() for d in own for p in d.glob("*.so*") if p.is_file()},
                  key=lambda p: (not p.name.startswith("libllama"), p.stat().st_size))
    left = list(names)
    for f in [*files, *libs]:
        if not left:
            break
        try:
            with open(f, "rb") as fh:
                tail = b""
                while left:
                    chunk = fh.read(8 * 1024 ** 2)
                    if not chunk:
                        break
                    blob = tail + chunk
                    left = [n for n in left if n.encode() not in blob]
                    tail = blob[-256:]
        except OSError:
            continue
    return left


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------

def parse_env(text: str) -> dict:
    out = {}
    for tok in shlex.split(text or ""):
        if "=" not in tok or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            raise SystemExit(f"--env: {tok!r} isn't NAME=value")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def own_flags(flags: list[str]) -> list[str]:
    return [f for f in flags if f.split("=", 1)[0] in OWN_FLAGS]


# 17c: llama-server's own variables reach it from --env only, so every one it
# gets is recorded and pinned — one exported in the box's shell reached it
# unrecorded, and a resume mixed two setups
# 20: and the lda runtime's LOOKAHEAD_*, which changes its routing
SERVER_VARS = ("LLAMA_", "GGML_", "LOOKAHEAD_")


# 17c: what G1's table fits beside the 23 GB file on a 32 GB card, the KV
# cache in q8_0, when the header can't be read. A run asking more is refused
# before anything is fetched (--max-context for a larger card). 17f: the
# pilot's card, 12.76 KiB a token measured, holds 8 slots of HLE's 86,016
# with 1.5 GB spare
MAX_CONTEXT = 8 * 86_016


# 17d: the KV cache's size, worked out from the GGUF's own header and the
# cache type, beside the file on this card — before anything is fetched
CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32,
               "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32}
# 17e: set from the pilot (an RTX 5090, 32,607 MiB; the phone build's file,
# 22,854,339,808 bytes; q8_0, flash attention): 8 slots of 36,864 tokens
# used 26,136 MiB after load, 8 of 67,584 used 29,198 — 13,066 bytes a token
# of context, of which the header's KV cache is 10,880, and 665 MiB above
# the file. CTX_BUFFERS: llama-server's buffers that grow with the context,
# beside its KV cache. ROOM: what the pilot didn't show — the longest
# prompts' compute, fragmentation — stated, never measured. 17g: the fixed
# part is the measured slope's, not an estimate on top of it — the pilot's
# 666 MiB above the file at 8 slots held their 8 recurrent states, so what
# is fixed is the rest (fixed_bytes): 17f counted 768 MiB and the recurrent
# state again, 622 MiB above both readings, and ran HLE at 7 slots where 8
# leave 1,570 MiB
CTX_BUFFERS = 2186
ABOVE_FILE, ABOVE_FILE_SLOTS = 666 * 1024 ** 2, 8
BASIS = ("the pilot's measured slope (12.76 KiB a token of context, 666 MiB above the file at "
         "8 slots)")
ROOM = 1024 ** 3
HEADER_BYTES = 24 * 1024 ** 2          # the metadata: a vocabulary's arrays are a few MB


def cache_types(flags: list[str]) -> tuple[str, str]:
    """-ctk and -ctv as given; llama-server's f16 when not"""
    k = v = "f16"
    for i, f in enumerate(flags):
        name, _, val = f.partition("=")
        val = val or (flags[i + 1] if i + 1 < len(flags) else "")
        if name in ("-ctk", "--cache-type-k"):
            k = val
        elif name in ("-ctv", "--cache-type-v"):
            v = val
    return k, v


def _full_layers(shape: dict) -> list[int]:
    """29: the layers whose cache holds the whole context — every layer with
    attention, but those sharing an earlier layer's cache and those with a
    sliding window (Gemma 4 E4B: 4 of its 42)"""
    swa = (shape.get("swa") or {}).get("layers") or []
    n = shape.get("kv_layers") or len(shape["kv_heads"])
    return [i for i in range(min(n, len(shape["kv_heads"])))
            if not (i < len(swa) and swa[i])]


def kv_per_token(shape: dict, ctk: str, ctv: str) -> float | None:
    """bytes of KV cache a token takes, every layer with attention — 29: whose
    cache holds the whole context (_full_layers)"""
    if ctk not in CACHE_BYTES or ctv not in CACHE_BYTES:
        return None
    return sum(shape["kv_heads"][i] * (shape["key_length"] * CACHE_BYTES[ctk]
                                       + shape["value_length"] * CACHE_BYTES[ctv])
               for i in _full_layers(shape))


SWA_UBATCH = 512                    # 29: llama-server's -ub, kept beside a window's tokens


def swa_per_slot(shape: dict, ctk: str, ctv: str) -> float:
    """29: a slot's cache for the sliding window's layers — the window and a
    batch's tokens, whatever the context (llama.cpp's iswa cache)"""
    swa = shape.get("swa")
    if not swa or ctk not in CACHE_BYTES or ctv not in CACHE_BYTES:
        return 0.0
    n = shape.get("kv_layers") or len(shape["kv_heads"])
    per = sum(shape["kv_heads"][i] * (swa["key_length"] * CACHE_BYTES[ctk]
                                      + swa["value_length"] * CACHE_BYTES[ctv])
              for i in range(min(n, len(shape["kv_heads"]), len(swa["layers"])))
              if swa["layers"][i])
    return per * (swa["window"] + SWA_UBATCH)


def recurrent_per_slot(shape: dict) -> float:
    """a hybrid's recurrent layers' state, a slot's (f32, as llama.cpp keeps it)"""
    m = shape.get("ssm") or {}
    if not m or not all(isinstance(m.get(k), int) for k in ("conv_kernel", "inner_size",
                                                            "state_size")):
        return 0.0
    n_group = m.get("group_count") if isinstance(m.get("group_count"), int) else 1
    conv = (m["conv_kernel"] - 1) * (m["inner_size"] + 2 * n_group * m["state_size"])
    state = m["state_size"] * m["inner_size"]
    return sum(1 for h in shape["kv_heads"] if h == 0) * (conv + state) * 4.0


def header_of(src: str) -> tuple[dict | None, int | None]:
    """(the GGUF's shape, its size in bytes, every part of a split one) — read
    before it is fetched: its first megabytes from Hugging Face, or the file
    here. (None, None) when it can't be"""
    import io

    import gguf_header
    parts = split_parts(src)
    try:
        if not _HF.match(src.strip()):
            with open(Path(parts[0]).expanduser(), "rb") as fh:
                got = gguf_header.shape(fh)
            return got, sum(Path(x).expanduser().stat().st_size for x in parts)
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or None)

        def path_of(x: str) -> str:
            m = _HF.match(x.strip())
            return (f"{'datasets/' if m['type'] == 'datasets/' else ''}{m['repo']}"
                    + (f"@{m['rev']}" if m["rev"] else "") + f"/{m['path']}")
        with fs.open(path_of(parts[0]), "rb", block_size=HEADER_BYTES) as fh:
            head = fh.read(HEADER_BYTES)
        return (gguf_header.shape(io.BytesIO(head)),
                sum(int(fs.info(path_of(x))["size"]) for x in parts))
    except Exception:                                   # noqa: BLE001 — checked after the fetch
        return None, None


def kv_fit(src: str, flags: list[str], slots: int, ctx: int,
           memory_mib: int | None) -> tuple[dict | None, str]:
    """({used, kv, per_token, per_slot, file, card, fit, types, arch}, '') —
    what these slots use on this card, in bytes, and the most that fit with
    ROOM to spare; (None, why) when the header, the cache type or the card
    can't be read"""
    shape, size = header_of(src)
    if not shape or not size:
        return None, "the GGUF's header couldn't be read before the fetch"
    if not memory_mib:
        return None, "the card's memory couldn't be read (nvidia-smi)"
    ctk, ctv = cache_types(flags)
    per_token = kv_per_token(shape, ctk, ctv)
    if per_token is None:
        return None, f"the cache type {ctk}/{ctv} isn't one this check knows"
    per_slot = ((per_token + CTX_BUFFERS) * ctx + recurrent_per_slot(shape)
                + swa_per_slot(shape, ctk, ctv))
    card = memory_mib * 1024 ** 2
    fixed = fixed_bytes(shape)
    room = card - size - fixed - ROOM
    return {"used": size + fixed + per_slot * slots, "kv": per_token * ctx * slots,
            "per_token": per_token, "per_slot": per_slot, "file": size, "card": card,
            "fit": max(0, int(room // per_slot)), "types": f"{ctk}/{ctv}",
            "arch": shape["arch"], "basis": BASIS}, ""


def fixed_bytes(shape: dict) -> float:
    """17g: what is fixed beside the file and the slots — the pilot's 666 MiB
    above the file at 8 slots, less those 8 slots' recurrent state (counted a
    slot at a time): 147 MiB for the phone build's 65 MiB a slot, 666 MiB for
    a model with none"""
    return max(0.0, ABOVE_FILE - ABOVE_FILE_SLOTS * recurrent_per_slot(shape))


def answered_here() -> bool:
    """17c: this --out holds an answer — its setup is pinned from then on"""
    from service import config
    root = Path(config.OUT_DIR)
    return any(p.stat().st_size for p in root.glob("*/*_0shot/frontier/answers.jsonl"))


def shell_server_vars() -> list[str]:
    """llama-server's variables set in this shell, which it isn't given"""
    return sorted(k for k in os.environ if k.startswith(SERVER_VARS))


def server_env(srv: dict, extra: dict) -> dict:
    """the child's environment, built here: the box's, without its secrets or
    any LLAMA_*/GGML_* it holds, the tarball's libraries, and --env"""
    env = {k: v for k, v in os.environ.items()
           if k not in rb.SECRETS and not k.startswith(SERVER_VARS)}
    libs = os.pathsep.join([*srv["libs"], os.environ.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    env["LD_LIBRARY_PATH"] = libs
    env.update(extra)
    return env


def version_of(exe: Path, env: dict) -> dict:
    """what `llama-server --version` says: {text, build, commit}"""
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60,
                           env=env)
        text = (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        text = f"--version failed: {e}"
    # 22: and upstream's words since its 0.x versions — "version: 0.6.0-dev
    # (build 11555, commit 1cf776776)" — where the lda build says "version:
    # 9954 (a4a76a75)"
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text) \
        or re.search(r"\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)", text)
    return {"text": "\n".join(text.splitlines()[:6]), "build": int(m[1]) if m else None,
            "commit": m[2] if m else None}


def _get(url: str, timeout: float = 5) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:                                   # noqa: BLE001 — not up yet
        return 0, b""


def of_words(task: str, items: list) -> str:
    """29.2: what a benchmark's questions are — ARC-AGI-2's its test grids,
    167 in its 120 tasks ("5 of its 167" read as a count that didn't agree)"""
    if fb.BENCH[task].get("aggregate") == "pass@2":
        tasks = len({fb.group_of(task, it) for it in items})
        return f"{len(items)} test grids (in its {tasks} tasks)"
    return f"{len(items)} questions"


def _started(s: "Server", timeout: float, say, gguf: Path) -> "Server":
    """29: a Server, started (or SystemExit, said in its words)"""
    s.start(timeout, say, gguf)
    return s


class Server:
    """llama-server on 127.0.0.1, started and stopped"""

    def __init__(self, srv: dict, gguf: Path, port: int, slots: int, ctx: int,
                 flags: list[str], env: dict, log: Path, host: str = "127.0.0.1"):
        # 29: --serve binds the container's every address — its port is
        # published on the station's loopback alone (scripts/own_box.sh serve)
        self.argv = [str(srv["bin"]), "-m", str(gguf), "--host", host, "--port", str(port),
                     "-c", str(ctx * slots), "-np", str(slots), "-ngl", "99", "--jinja",
                     "--metrics", *flags]
        self.env, self.log, self.port = server_env(srv, env), log, port
        self.base, self.slots = f"http://127.0.0.1:{port}", slots
        self.proc: subprocess.Popen | None = None

    def taken(self) -> bool:
        """something already answers on the port (an earlier llama-server?)"""
        with socket.socket() as sk:
            sk.settimeout(2)
            return sk.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self, timeout: float, say, gguf: Path | None = None) -> None:
        # 17b: a server already on the port would answer for this one — its
        # file's answers under this file's sha256. Nothing starts
        if self.taken():
            raise SystemExit(f"something already answers on 127.0.0.1:{self.port} — an earlier "
                             "llama-server? Stop it (pkill -f llama-server) or give another "
                             "--port. Nothing was started")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log, "ab")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting =====\n".encode())
        fh.flush()
        self.proc = subprocess.Popen(self.argv, stdout=fh, stderr=subprocess.STDOUT, env=self.env)
        t0, said = time.time(), 0.0
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server stopped as it started (exit {self.proc.returncode})"
                                 f":\n{self.tail(30)}")
            st, _ = _get(self.base + "/health")
            if st == 200:
                # 17b: the one that answers is ours, and serves this file
                if self.proc.poll() is not None:
                    raise SystemExit(f"something answers on 127.0.0.1:{self.port}, but the "
                                     f"llama-server started here has stopped (exit "
                                     f"{self.proc.returncode}):\n{self.tail(30)}")
                served = str(self.props().get("model_path") or "")
                if gguf is not None and Path(served).name != Path(gguf).name:
                    self.stop()
                    raise SystemExit(f"the server on 127.0.0.1:{self.port} serves "
                                     f"{Path(served).name or 'no file it names'}, not "
                                     f"{Path(gguf).name}: nothing was asked")
                say(f"llama-server is up ({time.time() - t0:.0f} s)")
                return
            if time.time() - t0 > timeout:
                self.stop()
                raise SystemExit(f"llama-server wasn't healthy after {timeout:.0f} s:\n"
                                 f"{self.tail(30)}")
            if time.time() - said > 30:
                say(f"waiting for llama-server to load the model… ({time.time() - t0:.0f} s)")
                said = time.time()
            time.sleep(1)

    def props(self) -> dict:
        st, raw = _get(self.base + "/props")
        try:
            return json.loads(raw) if st == 200 else {}
        except ValueError:
            return {}

    def tail(self, n: int) -> str:
        try:
            lines = self.log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return rb.scrub("\n".join(lines[-n:]))

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)


def startup_log(path: Path, limit: int = 400) -> str:
    """the server log's start, each session's, up to where it listens — what
    it loaded and how — never a request"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    out, on = [], False
    for ln in lines:
        if ln.startswith("===== ") and ln.rstrip().endswith("starting ====="):
            on = True
        if on:
            out.append(ln)
            if re.search(r"server is listening|all slots are idle", ln):
                on = False
    return rb.scrub("\n".join(out[:limit]))


# ---------------------------------------------------------------------------
# the run's record and its bundle
# ---------------------------------------------------------------------------

def register_here(served_as: str, name: str, based_on: str, base: str, thinking: bool,
                  how: str, flags: list[str] | None = None, env: dict | None = None,
                  runtime: str | None = None, slots: int | None = None) -> dict:
    """the served model, in this box's own database, as the board keeps one —
    17b: with the launch it was given, which each task's setup records"""
    from service import db, served
    p = served.probe(base + "/v1")
    rec = {"id": served_as, "name": name, "base_url": base + "/v1", "key": "",
           "based_on": based_on, "how": how, "thinking": "on" if thinking else "off",
           "phone": False, "gguf_path": "", "gguf_flags": "", "gguf_setups": [],
           "pin": served.pin_of(p), "answered": p["answered"], "by": "remote_gguf.py",
           "at": time.time(), "flags": shlex.join(flags or []),
           "env": " ".join(f"{k}={v}" for k, v in sorted((env or {}).items())),
           "speculative": p.get("speculative"),
           # 20: its runtime, whose sampling each request carries
           **({"runtime": runtime} if runtime else {}),
           # 23: its slots (-np), which each task's setup.json records
           **({"slots": slots} if slots else {})}
    db.served_put(rec)
    served.write_meta(rec)
    return rec


def task_state(row: Path | None, task: str) -> tuple[int, int]:
    """(answered, of) for this box's share of a task"""
    from service import config
    from service import frontier as sf
    items = fb.shard_of(fb.load(task, config.BENCH_ROOT), sf.shard())
    want = {(it["id"], e) for it in items for e in range(fb.BENCH[task]["epochs"])}
    got = set(sf.read_answers(sf.task_dir(row, task) / sf.ANSWERS)) if row else set()
    return len(want & got), len(want)


def make_bundle(out: Path, served_as: str, thinking: bool, tasks: list[str], state: dict,
                shard: tuple[int, int] | None, gguf: dict, server: dict, rec: dict
                ) -> tuple[Path | None, dict, dict]:
    """the bundle of every task this box answered whole: its path (None when
    none is), the tasks in it and those not finished"""
    from service import config
    from service import frontier as sf
    safe = served_as.replace("/", "__")
    row = config.OUT_DIR / (safe + "__thinking" if thinking else safe)
    done, incomplete = {}, {}
    for t in tasks:
        n, of = task_state(row, t)
        (done if n == of else incomplete)[t] = {"answers": n, "of": of}
    if not done:
        return None, done, incomplete
    files: dict[str, bytes] = {}
    for t in done:
        d = sf.task_dir(row, t)
        for name in (sf.ANSWERS, sf.SETUP):
            files[f"results/{row.name}/{d.relative_to(row).as_posix()}/{name}"] = \
                (d / name).read_bytes()
    for r in {row, config.OUT_DIR / safe}:
        if (r / "model_meta.json").exists():
            files[f"results/{r.name}/model_meta.json"] = (r / "model_meta.json").read_bytes()
    logs = sorted(config.LOGS_DIR.glob(f"service_*_{safe}.log"),
                  key=lambda p: int(p.name.split("_")[1]))
    log = "".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    own = out / OWN_LOG
    log += ("\n===== remote_gguf =====\n" + own.read_text(encoding="utf-8")) if own.exists() else ""
    log += "\n===== llama-server, as it started =====\n" + startup_log(out / SERVER_LOG)
    # 23c: the card's readings, its checks, and each benchmark's share of the
    # step's energy and time — by the seconds its answers took (a step asks
    # its benchmarks one after another); its dollars when a price was given
    samples = cc.read_samples(out / GPU_FILE)
    if samples:
        files[GPU_FILE] = (out / GPU_FILE).read_bytes()
    use = cc.summarise(samples) if samples else {}
    price = price_of(state)
    took = {}
    for t in tasks:
        got = sf.read_answers(sf.task_dir(row, t) / sf.ANSWERS).values() if row else []
        took[t] = (sum(float(r["wall_s"]) for r in got if isinstance(r.get("wall_s"), (int, float))),
                   sum(1 for _ in got))
    by = 0 if sum(w for w, _ in took.values()) > 0 else 1     # seconds, else answers
    whole = sum(x[by] for x in took.values())
    per = {}
    for t in tasks:
        spec = fb.BENCH[t]
        s = sf.settings(rec, t, thinking)
        share = took[t][by] / whole if whole else None
        spent = {}
        if share is not None and use.get("seconds"):
            spent["server_s"] = round(use["seconds"] * share, 1)
            if use.get("wh") is not None:
                spent["energy_wh"] = round(use["wh"] * share, 3)
            if price is not None:
                spent["cost_usd"] = round(price * spent["server_s"] / 3600, 4)
        per[t] = {"protocol_version": spec["protocol_version"],
                  "revision": spec["source"]["revision"], "epochs": spec["epochs"],
                  "budget": s["max_tokens"], "family": sf.family_of(rec) or "",
                  "sampling": {k: v for k, v in s.items()
                               if k not in ("max_tokens", "chat_template_kwargs")},
                  # 29: where each of the runtime's values comes from
                  **({"sampling_from": fb.RUNTIMES[rec["runtime"]]["sampling_from"]}
                     if fb.RUNTIMES.get(rec.get("runtime") or "", {}).get("sampling_from")
                     else {}),
                  "switch": s.get("chat_template_kwargs"), **spent,
                  **(done.get(t) or {}), **({"incomplete": True} if t in incomplete else {})}
    setup = {"suite": SUITE, "protocol": fb.VERSION, "served_as": served_as,
             **({"runtime": rec["runtime"]} if rec.get("runtime") else {}),
             "based_on": rec.get("based_on") or "", "thinking": "on" if thinking else "off",
             "gguf": gguf, "server": server, "tasks": per, "gpu": rb.gpu_info(),
             "where": OWN_WHERE if STEP.get("own") else WHERE,
             "runner": "service/runner.py → service/frontier.py, called directly",
             # 29: one GPU of our own station — its station, label and GPU,
             # never its address; it has no price
             **({"own_box": dict(STEP["own"])} if STEP.get("own") else {}),
             "board_commit": board_commit(), "sessions": state.get("sessions", 1),
             # 17f: G5's box, and the runner image's tag (the board's commit it was built at)
             "box": state.get("label") or None, "image": (board_commit() or "")[:7] or None,
             "shard": {"i": shard[0], "n": shard[1]} if shard else None,
             "started_at": state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             # 23c: each session's card check, the card's use over the step,
             # and the box's price when every session was given the same one
             **({"card_checks": state["card_checks"]} if state.get("card_checks") else {}),
             **({"gpu_use": use} if use else {}),
             **({"price": {"per_hour": price, "by": "the box line (--price)"}}
                if price is not None else {})}
    files["run.log"] = rb.scrub(log).encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    digest = hashlib.sha256(b"".join(files[n] for n in sorted(files)
                                     if n.endswith("/" + sf.ANSWERS))).hexdigest()
    bundle = {"format": rb.SUITES[SUITE]["format"], "suite": SUITE, "model": served_as,
              "thinking": thinking, "row": row.name, "gguf_sha256": gguf["sha256"],
              "tasks": {t: x["answers"] for t, x in done.items()}, "incomplete": incomplete,
              "answers_sha256": digest,
              "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    if shard:
        bundle["shard"] = {"i": shard[0], "n": shard[1]}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    # 17c: the benchmarks in the name when the box ran some of the suite
    parts = [fb.BENCH[t]["short"] for t in tasks] if len(tasks) < len(fb.TASKS) else None
    path = rb.write(out / rb.bundle_name(SUITE, served_as, thinking, shard, parts), files)
    return path, done, incomplete


def price_of(state: dict) -> float | None:
    """23c: the box's dollars an hour — when every session of the step was
    given one, and the same one; else none (a cost is never guessed)"""
    got = state.get("prices") or []
    vals = {x.get("per_hour") for x in got if isinstance(x, dict)}
    if not got or len(got) < int(state.get("sessions") or 1) or len(vals) != 1:
        return None
    v = next(iter(vals))
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


# ---------------------------------------------------------------------------
# 22: a runtime's pilot — one box asks fb.PILOT's questions before the other
# boxes are rented, through the board's own code (service/frontier.settings,
# served.answer_one, frontier.score), and says what came back
# ---------------------------------------------------------------------------

PILOT_FILE, PILOT_SUMMARY = "pilot.jsonl", "pilot.json"


def not_apart(r: dict) -> str:
    """22: '' when an answer that finished came back with its thinking apart
    from its reply (the server's reasoning_content) and a reply with no think
    tags in it; else what went wrong. One that ran out of room isn't judged:
    its reply never came"""
    if r.get("error"):
        return f"no answer ({r['error']})"
    if r.get("ran_out"):
        return ""
    if r.get("fallback"):
        return "asked without the server's chat parsing (its chat endpoint failed on it twice)"
    if r.get("thinking") == "off":
        # 29: thinking off — none at all, apart or in the reply
        if r.get("thinking_apart") or r.get("thinking_chars"):
            return "thinking, with thinking off"
        if r.get("reply_tags"):
            return "think tags in the reply"
        return "" if r.get("reply_chars") else "an empty reply"
    if not r.get("thinking_apart"):
        return "no thinking apart from the reply (reasoning_content empty)"
    if r.get("reply_tags"):
        return "think tags in the reply"
    if not r.get("reply_chars"):
        return "an empty reply"
    return ""


def serve_here(a: argparse.Namespace, srv: dict, gguf_path: Path, gguf: dict, flags: list[str],
               env: dict, ctx: int, out: Path, say, own: dict | None,
               sleep=time.sleep, every: float = 30.0) -> int:
    """29: the GGUF served on --serve's port for the board's own runner, until
    stopped (Ctrl-C, or own_box.sh stop) — the Mobile and Standard benchmarks
    run on the board as for any served model, over an SSH tunnel from the
    server. On our own GPU, someone else's process stops the server (the
    board's run stops, its answers kept) until the GPU is free again"""
    import gpu_free
    s = Server(srv, gguf_path, a.serve, a.slots, ctx, flags, env, out / SERVER_LOG,
               host="0.0.0.0")
    while True:
        s.start(a.load_timeout, say, gguf_path)
        gate = gpu_free.Gate(a.label, {s.proc.pid}, gpu_free.used_mib()) if own else None
        say(f"serving {gguf['name']} (sha256 {gguf['sha256'][:16]}) on port {a.serve}: "
            f"{a.slots} slots of {ctx:,} tokens · {' '.join(flags)}"
            + (f" · {a.label}'s GPU read every {every:.0f} s for other processes" if gate else ""))
        why = ""
        while s.proc is not None and s.proc.poll() is None:
            sleep(every)
            why = gate(True) if gate else ""
            if why:
                break
        tail = s.tail(20)
        s.stop()
        if not why:
            raise SystemExit(f"llama-server stopped serving:\n{tail}")
        say(f"{why} — the server is stopped until the GPU is free (nothing of theirs is touched)")
        while gpu_free.busy(a.label):
            sleep(60)


# ---------------------------------------------------------------------------
# 29: a runtime's attention on this card — card-bolt's flash attention with a
# q8_0 cache at Bolt's 512-wide heads on a V100 (sm_70) is checked before a
# question, its fallbacks after it when the card refuses it
# ---------------------------------------------------------------------------

ATTN_FILE = "attention.json"        # in the shared files folder: once a box
ATTN_LONG, ATTN_SHORT = 4096, 256   # a long prompt alone, then short ones side by side


def attn_probe(base: str, slots: int, alive) -> str:
    """'' when the server reads a long prompt alone and answers several short
    ones at once — each way its attention runs (a prompt's batch, one token,
    a batch of slots); else why not"""
    from concurrent.futures import ThreadPoolExecutor
    try:
        toks = cc.tokens_of(base)
        cc.complete(base, cc.prompt(toks, ATTN_LONG), 32)
        n = max(1, min(slots, 4))
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(lambda i: cc.complete(base, cc.prompt(toks, ATTN_SHORT, 37 * (i + 1)),
                                                64), range(n)))
    except Exception as e:                              # noqa: BLE001 — said, the next tried
        return f"{type(e).__name__}: {e}"[:300]
    return "" if alive() else "llama-server stopped"


def attn_words(flags: list[str]) -> str:
    """"flash attention on with a q8_0 cache" """
    k, v = cache_types(flags)
    fa = next((flags[i + 1] for i, f in enumerate(flags[:-1]) if f in ("--flash-attn", "-fa")),
              "auto")
    cache = k if k == v else f"{k} keys and {v} values"
    return f"flash attention {fa} with a {cache} cache"


def attn_flags(rt_name: str, flags: list[str], start, files: Path, key: str, say
               ) -> tuple[list[str], dict]:
    """(the flags to run, what was found) — the runtime's own launch, else the
    first fallback this card runs, each started with `start(flags)` (a Server,
    up, or SystemExit) and probed. Kept by `key` (the build, the file, the
    card) in the box's files folder, so the next step runs what this one found"""
    sets = fb.runtime_flag_sets(rt_name)
    extra = flags[len(sets[0]):] if flags[:len(sets[0])] == sets[0] else []
    memo = files / ATTN_FILE
    try:
        known = json.loads(memo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    got = known.get(key) if isinstance(known, dict) else None
    if isinstance(got, dict) and got.get("flags") in sets:
        say(f"attention on this card: {attn_words(got['flags'])}, as checked at "
            f"{got.get('at', '?')}" + (" — " + "; ".join(
                f"{attn_words(r['flags'])} refused" for r in got.get("refused") or [])
                if got.get("refused") else ""))
        return got["flags"] + extra, got
    refused = []
    for fs in sets:
        try:
            proc = start(fs + extra)
        except SystemExit as e:
            refused.append({"flags": fs, "why": str(e.code)[-600:]})
            say(f"{attn_words(fs)}: llama-server didn't start on this card — trying the next")
            continue
        try:
            why = attn_probe(proc.base, proc.slots,
                             lambda: proc.proc is not None and proc.proc.poll() is None)
            tail = proc.tail(12) if why else ""
        finally:
            proc.stop()
        if not why:
            rec = {"flags": fs, "refused": refused,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            say(f"attention on this card: {attn_words(fs)} answers a {ATTN_LONG:,}-token prompt "
                "and short ones side by side"
                + (" — " + "; ".join(f"{attn_words(r['flags'])} refused" for r in refused)
                   if refused else ""))
            if isinstance(known, dict):
                known[key] = rec
                try:
                    memo.write_text(json.dumps(known, indent=1), encoding="utf-8")
                except OSError:
                    pass
            return fs + extra, rec
        refused.append({"flags": fs, "why": f"{why}\n{tail}"[-600:]})
        say(f"{attn_words(fs)} failed on this card ({why}) — trying the next")
    raise SystemExit(f"runtime {rt_name}: none of its launches ran on this card — "
                     + "; ".join(f"{attn_words(r['flags'])}: {r['why'].splitlines()[0][:160]}"
                                 for r in refused) + ". Nothing was asked")


def pilot_line(r: dict, k: int, of: int) -> str:
    label = fb.BENCH[r["task"]]["label"]
    why = not_apart(r)
    toks = f" · {r['tokens']:,} tokens in all" if r.get("tokens") is not None else ""
    return (f"{label} {k} of {of} (question {r['id']}): thinking {r['thinking_chars']:,} "
            f"characters{toks} · read {r['read'] or '—'} · right {r['right']} "
            f"{'✓' if r['ok'] else '✗'} · {r['seconds']:.0f} s"
            + (" · ran out of room" if r.get("ran_out") else "")
            + (f" · NOT apart: {why}" if why else ""))


def pilot_verdict(rows: list[dict]) -> tuple[bool, str]:
    """22: the pilot's last line — whether the thinking and the answer came
    back apart in every answer that finished"""
    judged = [r for r in rows if not r.get("ran_out") or r.get("error")]
    out = len(rows) - len(judged)
    bad = [(r, not_apart(r)) for r in judged if not_apart(r)]
    if not judged:
        return False, ("Thinking and answer: not judged — every answer ran out of room before "
                       "it replied")
    if not bad:
        return True, (f"Thinking and answer separated correctly: all {len(judged)} answers that "
                      "finished came back with the thinking apart (reasoning_content) and the "
                      "answer in the reply, no think tags in it"
                      + (f"; {out} ran out of room before replying, not judged" if out else ""))
    r, why = bad[0]
    return False, (f"Thinking and answer NOT separated correctly in {len(bad)} of {len(judged)} "
                   f"answers — the first, {fb.BENCH[r['task']]['label']} question {r['id']}: "
                   f"{why}. Don't rent the other boxes yet")


def pilot_ask(rec: dict, tasks: list[str], out: Path, say, slots: int, each: int = 0,
              on: bool = True) -> dict:
    """fb.PILOT's questions of each task, `slots` at a time, a task after the
    other: each answer's line as it lands, each task's summary, and the
    verdict — {separated, verdict, paces}. 29: `each` questions of every task
    instead (an own box's pilot), thinking `on` or off, each answer's limit
    capped by the model's context as the run caps it, and whether it ended in
    a loop (27f's rule)"""
    from concurrent.futures import ThreadPoolExecutor

    from service import config, served
    from service import frontier as sf
    fb.set_root(config.BENCH_ROOT)
    dest = out / (PILOT_FILE if not each else f"pilot-thinking-{'on' if on else 'off'}.jsonl")
    dest.unlink(missing_ok=True)                # asked again, the pilot starts again
    lock = threading.Lock()
    rows: list[dict] = []
    paces: dict[str, dict] = {}
    cap = fb.context_cap(rec.get("runtime"))
    for task in tasks:
        items = fb.pilot_items(task, config.BENCH_ROOT, each or None)
        s = sf.settings(rec, task, on)
        done = {"n": 0}

        def one(it: dict, task=task, s=s, n=len(items)) -> None:
            text, need = fb.prompt(task, it)
            seed = fb.seed_of(task, it["id"], 0)
            si = {**s, "seed": seed,
                  **({"system": fb.system_of(task)} if fb.system_of(task) else {})}
            limit = {}
            if cap:
                try:
                    k = served.prompt_tokens(rec, text, si)
                except (served.ItemError, served._Retry) as e:
                    raise SystemExit(f"pilot: the server didn't count a prompt's tokens ({e})"
                                     ) from None
                si["max_tokens"] = fb.limit_of(s["max_tokens"], k, cap)
                limit = {"prompt_tokens": k, "limit": si["max_tokens"],
                         "capped": si["max_tokens"] < s["max_tokens"]}
            t0 = time.time()
            try:
                a = served.answer_one(rec, text, si)
            except served.ServerStopped as e:
                raise SystemExit(f"pilot: the server stopped answering "
                                 f"({getattr(e, 'refused', '') or e}) — paste the line again to "
                                 "ask the pilot again") from None
            secs = time.time() - t0
            answer = str(a)
            sc = fb.score(task, answer, a.finish, need) if not a.error else \
                {"read": None, "ok": False, "ran_out": False}
            content = a.content or ""
            r = {"task": task, "id": str(it["id"]), "seed": seed, "seconds": round(secs, 1),
                 "thinking": "on" if on else "off", **limit,
                 "loop": fb.ended_in_loop(answer, fb.ran_out(answer, a.finish)),
                 "tokens": a.tokens, "finish": a.finish,
                 "thinking_chars": len(fb.thinking_of(answer)), "reply_chars": len(content.strip()),
                 "thinking_apart": bool(a.reasoning_chars),
                 "reply_tags": "<think>" in content or "</think>" in content,
                 "read": sc["read"], "right": need.get("key"), "ok": bool(sc["ok"]),
                 "ran_out": bool(sc["ran_out"]), "fallback": bool(a.fallback),
                 "error": ((a.error or {}).get("chat") or "no answer")[:200] if a.error else "",
                 "answer": answer}
            with lock:
                rows.append(r)
                done["n"] += 1
                with open(dest, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                say(pilot_line(r, done["n"], n))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=slots) as pool:
            list(pool.map(one, items))
        wall = time.time() - t0
        mine = [r for r in rows if r["task"] == task]
        mean = sum(r["seconds"] for r in mine) / max(1, len(mine))
        at_once = min(slots, len(mine)) or 1
        out_toks = sum(r["tokens"] or 0 for r in mine)
        paces[task] = {"answers": len(mine), "right": sum(r["ok"] for r in mine),
                       "mean_s": round(mean, 1), "wall_s": round(wall, 1), "at_once": at_once,
                       # the box's seconds an answer, `at_once` asked side by side —
                       # what plan L's paces are (frontier_box.PACE, at 8 slots)
                       "pace_s": round(mean / at_once, 2),
                       "ran_out": sum(r["ran_out"] for r in mine),
                       # 29: the tokens all of them wrote, over the time they took
                       # together — the card's tokens a second at these slots
                       "tokens": out_toks, "tok_s": round(out_toks / wall, 1) if wall else None,
                       "loops": sum(r["loop"] for r in mine),
                       **({"capped": sum(bool(r.get("capped")) for r in mine),
                           "ran_out_capped": sum(bool(r.get("capped") and r["ran_out"])
                                                 for r in mine)} if cap else {})}
    say("")
    for task, p in paces.items():
        mine = [r for r in rows if r["task"] == task]
        say(f"{fb.BENCH[task]['label']}: {p['right']} of {p['answers']} right · thinking "
            f"{sum(r['thinking_chars'] for r in mine) // max(1, len(mine)):,} characters on "
            f"average · {p['mean_s']:.0f} s an answer, {p['at_once']} at a time ({p['wall_s']:.0f} "
            f"s for all {p['answers']}): about {p['pace_s']:.1f} s of the box's time an answer · "
            f"{p['ran_out']} ran out of room"
            # 29: the card's tokens a second, the loop rule, and the capped limit
            + (f" · {p['tok_s']:,.0f} tokens a second together" if p.get("tok_s") else "")
            + (f" · {p['loops']} ended in a loop" if p.get("loops") else "")
            + (f" · {p['capped']} with the limit capped by the 64K context"
               + (f", {p['ran_out_capped']} of them ran out at it" if p.get("ran_out_capped")
                  else "") if p.get("capped") else ""))
    ok, verdict = pilot_verdict(rows)
    say(verdict)
    return {"separated": ok, "verdict": verdict, "paces": paces}


_PROGRESS_LOCK = threading.Lock()


def progress(out: Path, **fields) -> None:
    """17f: the step's progress file, its fields merged in, the time it was
    last written with them — 17i: one writer at a time (the parity step's
    questions are asked side by side, and each writes its count)"""
    if STEP.get("pilot"):
        return                      # 22: a pilot is no step of a plan: the fetch never sees it
    p = out / PROGRESS
    out.mkdir(parents=True, exist_ok=True)
    with _PROGRESS_LOCK:
        try:
            was = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            was = {}
        if not isinstance(was, dict):
            was = {}
        tmp = p.with_suffix(".part")
        tmp.write_text(json.dumps({**was, **fields, "at": round(time.time())}),
                       encoding="utf-8")
        tmp.replace(p)


class Watch(threading.Thread):
    """the run's line, each time it changes — 17f: and in the progress file"""

    def __init__(self, sid: int, say, every: float = 2.0, out: Path | None = None):
        super().__init__(daemon=True)
        self.sid, self.say, self.every, self.prog, self.out = sid, say, every, "", out
        self.done_ev = threading.Event()

    def run(self) -> None:
        while not self.done_ev.wait(self.every):
            self.tick()

    def tick(self) -> None:
        try:
            from service import db
            prog = (db.get(self.sid) or {}).get("progress") or ""
        except Exception:                               # noqa: BLE001 — never the run's business
            return
        if prog and prog != self.prog:
            self.prog = prog
            self.say(f"· {prog}")
            if self.out is not None:
                progress(self.out, line=prog)

    def stop(self) -> None:
        self.done_ev.set()


# 17g: the step this process is, once its folder is known — a step that
# stops at start-up (llama-server dying at load, a gated question set, a
# refused memory check, a parity question the server fails) says so in its
# progress file, where it read "starting" for ever
STEP: dict = {}


def main(argv: list[str] | None = None) -> int:
    STEP.clear()
    try:
        return _main(argv)
    except SystemExit as e:
        if STEP and e.code not in (0, None):
            progress(STEP["out"], state="stopped",
                     why=str(e.code if isinstance(e.code, str) else f"exit {e.code}")[:400])
        raise
    except KeyboardInterrupt:
        if STEP:
            progress(STEP["out"], state="stopped", why="stopped by hand")
        raise
    except Exception as e:                              # noqa: BLE001 — said, then raised
        if STEP:
            progress(STEP["out"], state="stopped", why=f"{type(e).__name__}: {e}"[:400])
        raise


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as", dest="served_as", required=True,
                    help="the served model on the board whose row the answers go on "
                         "(served/<name>): its registered file must be this GGUF")
    ap.add_argument("--gguf", required=True, help="hf://<org>/<repo>[@<rev>]/<file>.gguf, or a "
                                                  "path on this box")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server",
                     help="the llama-server tarball (build_llama_tarball.sh): hf://… or a path")
    src.add_argument("--server-dir",
                     help="20: a llama.cpp build directory on this box (its bin/llama-server "
                          "and libraries), run as it is")
    ap.add_argument("--pin", action="append", default=[],
                    help="20: NAME=SHA256 — the sha256 a file of that name must have (the "
                         "GGUF, each of its parts, the tarball); another is refused")
    ap.add_argument("--runtime", choices=sorted(fb.RUNTIMES),
                    help="20: a named runtime — its environment, its server's flags and the "
                         "sampling each request carries (scripts/frontier.py RUNTIMES)")
    ap.add_argument("--based-on", default="", help="the model it is a GGUF of, as the board "
                                                   "registers it (Qwen/Qwen3.6-35B-A3B): it "
                                                   "picks the model card's sampling")
    ap.add_argument("--thinking", required=True, choices=("on", "off"))
    ap.add_argument("--only", action="append", choices=fb.TASKS,
                    help="one task (repeatable); every task when none is given")
    ap.add_argument("--out", default="remote-gguf", help="where the run and its bundle go")
    ap.add_argument("--files", default="",
                    help="17c: where the GGUF and the server tarball are fetched to, shared by "
                         "every run on this box (default: a folder 'files' beside --out)")
    ap.add_argument("--shard", default="", help="i/n: this box's share of each task, every n-th "
                                                "question from the i-th")
    ap.add_argument("--slots", type=int, default=8, help="questions at a time (llama-server -np)")
    ap.add_argument("--flags", default="", help="more llama-server flags, as typed")
    ap.add_argument("--env", default="", help="llama-server's environment, NAME=value …")
    ap.add_argument("--max-context", type=int, default=MAX_CONTEXT,
                    help="17c: the most context (slots × each slot's) the card holds beside "
                         f"the GGUF — G1's table for a 32 GB card: {MAX_CONTEXT:,}")
    ap.add_argument("--min-slots", type=int, default=0,
                    help="17f: when --slots don't fit this card, run as many as do, down to "
                         "this many, and say so (frontier_box.py gives one fewer than planned)")
    ap.add_argument("--ask-written-off", action="store_true",
                    help="17f: ask again the questions earlier runs wrote off as no answer (the "
                         "server failed on them)")
    ap.add_argument("--slots-fit", action="store_true",
                    help="17e: these --slots fit on this card — the memory check's estimate is "
                         "printed, and not held to")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="18e point 21: run more than --slots, up to this many, when the "
                         "memory check confirms they fit this card (an RTX PRO 6000 holds 25 of "
                         "BF16's at HLE's context) — never on --max-context alone")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--load-timeout", type=float, default=3600)
    ap.add_argument("--by", default="remote", help="who ran it, for the run's record")
    ap.add_argument("--label", default="", help="17f: the box's label in G5's plan (A3), kept "
                                                "in the bundle and the progress file")
    ap.add_argument("--n", type=int, default=fb.PARITY["n"],
                    help="17d: --parity's number of questions (the server's ask the same)")
    ap.add_argument("--parity", action="store_true",
                    help="17b: the parity check's --n questions, thinking off and greedy and "
                         "each asked twice, to --out/parity.jsonl — no bundle "
                         "(scripts/frontier_parity.py compare)")
    ap.add_argument("--price", type=float, default=None,
                    help="23c: the box's price, dollars an hour — kept in the setup, for the "
                         "board's dollars an answer (never guessed without it)")
    ap.add_argument("--pilot", action="store_true",
                    help="22: a runtime's pilot — 20 GPQA Diamond and 5 MMLU-Pro questions, "
                         "thinking on, each asked once: a line each (its thinking's length, the "
                         "answer the board read, the right one, its seconds), to "
                         "--out/pilot.jsonl, and last whether the thinking and the answer came "
                         "back apart. No bundle, no progress file: the fetch never sees it")
    ap.add_argument("--pilot-each", type=int, default=0,
                    help="29: the pilot asks this many questions of each benchmark given "
                         "(--only; every one when none), with --thinking on or off")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT",
                    help="29: serve the GGUF on this port for the board's own runner (Mobile "
                         "and Standard benchmarks over an SSH tunnel) until stopped — no "
                         "questions asked here; with --own, stopped while someone else's "
                         "process uses the GPU and started again once it is free")
    ap.add_argument("--serve-context", type=int, default=16_384,
                    help="29: --serve's context a slot (capped by the model's own)")
    ap.add_argument("--own", action="store_true",
                    help="29: one GPU of our own station (--label D3.0): never started while "
                         "another process uses the GPU, paused while one does (exit 75), no "
                         "price")
    a = ap.parse_args(argv)
    if not a.served_as.startswith("served/") or not re.fullmatch(r"served/[A-Za-z0-9._-]+",
                                                                 a.served_as):
        ap.error("--as: a served model's id on the board, served/<name>")
    try:
        shard = fb.parse_shard(a.shard)
    except ValueError as e:
        ap.error(f"--shard: {e}")
    if a.slots < 1:
        ap.error("--slots: at least 1")
    if a.price is not None and not (0 < a.price < 1000):
        ap.error("--price: the box's dollars an hour, e.g. 0.42")
    flags = shlex.split(a.flags)
    if own_flags(flags):
        ap.error(f"--flags: {', '.join(own_flags(flags))} are set by this script")
    extra_env = parse_env(a.env)
    pins = parse_pins(a.pin)
    # 22: a model a runtime names runs on it alone, and that runtime runs its
    # own models alone — Ornith-1.5 never on the lda runtime, nor without one
    why = fb.runtime_model_refused(a.runtime or "", a.served_as)
    if why:
        ap.error(f"--as: {why}")
    # 29: --pilot-each asks thinking off too, of the benchmarks the runtime
    # asks it of (its check below)
    if a.pilot and not (a.runtime and (a.thinking == "on" or a.pilot_each)):
        ap.error("--pilot: a runtime's pilot, thinking on (--runtime, --thinking on)")
    if a.pilot and a.parity:
        ap.error("--pilot or --parity, not both")
    if a.pilot_each and not a.pilot:
        ap.error("--pilot-each: with --pilot")
    if a.pilot_each < 0:
        ap.error("--pilot-each: a number of questions, 1 or more")
    # 29: our own GPU — its label names station and GPU, and it has no price
    own = own_box(a.label) if a.own else None
    if a.own and not own:
        ap.error("--own: give the box's label, --label D3.0 (station D3's GPU 0)")
    if a.own and a.price is not None:
        ap.error("--own: an own box has no price (the board says \"own box\")")
    if a.runtime:
        # 20: the runtime's environment and flags, nothing typed beside them
        # that would change what it is
        rt = fb.RUNTIMES[a.runtime]
        why = fb.runtime_thinking_refused(a.runtime, a.thinking, a.only or [])
        if why and not a.serve:         # 29: a served model is asked its thinking per request
            ap.error(why)
        if a.parity:
            ap.error(f"--runtime {a.runtime}: no parity questions (they ask thinking off)")
        names = {f.split("=", 1)[0] for f in flags}
        theirs = {x for x in fb.runtime_flags(a.runtime) if x.startswith("-")} | {
            "-fa", "--cache-type-k", "--cache-type-v", "--sampling-seq", "--samplers"}
        # 22: one asking with the GGUF's own template takes no other, and its
        # server splits the thinking off as it does by default
        if rt.get("template_sha256"):
            theirs |= {"--chat-template", "--chat-template-file", "--reasoning-format",
                       "--no-jinja", "--chat-template-kwargs"}
        refused = sorted(n for n in names if fb.RUNTIME_REFUSES.match(n) or n in theirs)
        if refused:
            ap.error(f"--flags: {', '.join(refused)} — runtime {a.runtime} sets the cache, "
                     "flash attention and the sampling, and runs no speculative decoding, no "
                     "reasoning budget and no --moe-stream")
        clash = sorted(k for k in extra_env if k in rt["env"] or re.match(
            r"^LLAMA_ARG_\w*(DRAFT|SPEC|REASONING|SAMPL|TEMP|TOP_|MIN_P|PENALTY|REPEAT)", k))
        if clash:
            ap.error(f"--env: {', '.join(clash)} — runtime {a.runtime} sets its own")
        # 22: and what it never gives its server (card-ornith: the lda branch's)
        never = fb.runtime_env_refused(a.runtime, extra_env)
        if never:
            ap.error(f"--env: {', '.join(never)} — runtime {a.runtime} never gives its server "
                     + " or ".join(f"{p}*" for p in rt["refuses_env"]))
        extra_env = {**rt["env"], **extra_env}
        flags = fb.runtime_flags(a.runtime) + flags
    out = Path(a.out).resolve()
    thinking = a.thinking == "on"
    STEP["out"] = out
    # 22: writes no progress file — 29: nor does --serve (no step of a plan)
    STEP["pilot"] = bool(a.pilot or a.serve)
    STEP["own"] = own
    progress(out, label=a.label or "", model=a.served_as, thinking=a.thinking,
             parity=bool(a.parity), tasks=[t for t in fb.TASKS if t in (a.only or [])],
             state="starting", why="", line="", incomplete={})
    if a.pilot and a.only and not a.pilot_each:
        ap.error("--pilot asks its own questions: no --only")
    if a.serve and (a.pilot or a.parity or a.only):
        ap.error("--serve asks nothing here: no --pilot, --parity or --only")
    if not a.only and not a.parity and not a.pilot and not a.serve:
        # 17d: a box names its benchmarks, thinking on or off — every one at
        # once is more context than a card holds, and asks ARC-AGI-2 with
        # thinking off
        ap.error("--only: name this box's benchmarks, one --only each (G5's table)")
    tasks = [t for t in fb.TASKS if t in (a.only or fb.TASKS)]
    if a.parity:
        # 17b: the parity check's questions, thinking off, whatever --thinking says
        tasks, thinking = [fb.PARITY["task"]], False
    if a.pilot and not a.pilot_each:
        tasks = [t for t in fb.TASKS if t in fb.PILOT["n"]]
    # 29: our own GPU, shared — nothing starts while another process uses it
    if own:
        import gpu_free
        why = gpu_free.busy(a.label)
        if why:
            raise SystemExit(f"{why} — this step doesn't start while it does (nothing of "
                             "theirs is touched). Nothing was fetched")
    # 17c: a context the card can't hold is refused here, before the download.
    # 17d: worked out from the GGUF's header, its cache type and this card;
    # the stated --max-context only when those can't be read. 29: a model's own
    # context caps it (Bolt's 64K)
    cap = fb.context_cap(a.runtime or "")
    ctx = fb.slot_context(tasks, thinking, cap)
    if a.serve:
        ctx = min(a.serve_context, cap) if cap else a.serve_context
    # 17h: what is free on the card, not its total — 300 MiB in use by
    # something else left 1,271 MiB where 1,570 was promised
    card = rb.gpu_info()
    in_use = int(card.get("memory_used_mib") or 0)
    mem = (card["memory_mib"] - in_use) if card.get("memory_mib") else None
    fit, unread = kv_fit(a.gguf, flags, a.slots, ctx, mem)
    big = max(tasks, key=lambda t: fb.slot_context([t], thinking))
    # 17f: a step planned at 8 slots runs what the card holds, down to
    # --min-slots, and says so (HLE's 86,016 with thinking on: 7 on a 5090)
    dropped = None
    if fit and a.min_slots and a.min_slots <= fit["fit"] < a.slots and not a.slots_fit:
        dropped = (a.slots, fit)
        a.slots = fit["fit"]
        fit, unread = kv_fit(a.gguf, flags, a.slots, ctx, mem)
    # 18e point 21: --slots only lowered; more run when the memory check, from
    # the GGUF's header and this card, confirms they fit
    raised = None
    if fit and a.max_slots > a.slots and fit["fit"] > a.slots and not a.parity:
        raised = (a.slots, fit)
        a.slots = min(fit["fit"], a.max_slots)
        fit, unread = kv_fit(a.gguf, flags, a.slots, ctx, mem)
    if fit and fit["fit"] < a.slots and not a.slots_fit:
        gb = 1024 ** 3
        raise SystemExit(
            f"{a.slots} slots of {ctx:,} tokens ({fb.BENCH[big]['label']}'s, thinking "
            f"{'on' if thinking else 'off'}) would use about {fit['used'] / gb:.1f} GB with the "
            f"{fit['file'] / gb:.1f} GB file ({fit['types']} cache, "
            f"{fit['per_token'] / 1024:.1f} KB a token from the GGUF's header; {fit['basis']}), "
            f"and this card has {fit['card'] / gb:.1f} GB, {ROOM / gb:.0f} GB kept spare: at "
            f"most {fit['fit']} slot{'s' if fit['fit'] != 1 else ''} fit — give --slots "
            f"{fit['fit']}" + (" (or a larger card)" if fit["fit"] else ": a larger card")
            + ", or --slots-fit if you know these do (frontier_box.py's line takes both). "
            "Nothing was fetched")
    if not fit and ctx * a.slots > a.max_context and not a.slots_fit:
        raise SystemExit(f"{a.slots} slots of {ctx:,} tokens ({fb.BENCH[big]['label']}'s, "
                         f"thinking {'on' if thinking else 'off'}) is {ctx * a.slots:,} tokens "
                         f"of context, and this card holds {a.max_context:,} (G1's table; "
                         "--max-context for a larger card) — " + unread + ": give the box fewer "
                         "benchmarks (--only) or fewer --slots, or --slots-fit if you know these "
                         "fit. Nothing was fetched")
    configure(out, a.slots, shard, a.ask_written_off)
    say = Out(out / OWN_LOG)
    from service import db, runner
    db.init()
    words = f"shard {shard[0]} of {shard[1]}" if shard else ""
    say(f"{a.served_as} · thinking {a.thinking} · {', '.join(tasks)}"
        + (f" · {words}" if shard else "") + f" · in {out}")

    # the run this folder holds: one served model, one mode, one file, one shard
    sp = out / STATE
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    if state and (state.get("served_as"), state.get("thinking")) != (a.served_as, thinking):
        raise SystemExit(f"{out} holds a run of {state.get('served_as')} (thinking "
                         f"{'on' if state.get('thinking') else 'off'}): give this one another "
                         "--out (17f: frontier_box.py puts each build in a folder of its own, "
                         "/workspace/<build>/<box>-<step>)")
    if state and (state.get("shard") or None) != (f"{shard[0]}/{shard[1]}" if shard else None):
        raise SystemExit(f"{out} holds {('shard ' + state['shard']) if state.get('shard') else 'a whole run'}"
                         f": give {words or 'a whole run'} another --out")

    # a split GGUF (…-00001-of-00002.gguf): every part, each hashed; llama-server
    # is given the first and finds the others beside it
    # 17c: one folder for every run on this box — the parity check's and the
    # full run's — so the 23 GB file is fetched and hashed once
    files = Path(a.files).resolve() if a.files else out.parent / "files"
    files.mkdir(parents=True, exist_ok=True)
    # 18e point 18: the server first — fetched, unpacked, pinned and run here
    # before the GGUF's tens of GB are fetched (a tarball for other cards was
    # found after the download)
    if a.server_dir:
        # 20: the team's build directory, run as it is — checked to run on this
        # box and card before anything else
        srv = server_dir(a.server_dir)
        env = server_env(srv, extra_env)
        ran = check_runs_here(srv, env, say)
    else:
        tb = fetch(a.server, files, say)
        srv = unpack_server(tb, out / "server")
        why = pin_refused(tb.name, srv["tarball_sha256"], pins)
        if why:
            raise SystemExit(why)
        env = server_env(srv, extra_env)
        # 20: a tarball built for some cards only (its VERSION's cuda_archs:
        # the lda tarball's is 120) is refused on another, before it loads
        archs, cap = cuda_archs(srv["bin"].parent.parent), compute_cap()
        if archs and cap and archs_run_on(archs, cap) is False:
            raise SystemExit(f"{tb.name} is built for CUDA architectures {';'.join(archs)} (its "
                             f"VERSION), and this card is sm_{cap}: it won't run here — rent "
                             "a card it was built for. Nothing was asked")
        # 18e point 18: and run here, as a build directory is — its version,
        # this box's card
        ran = check_runs_here(srv, env, say, what=tb.name)
    if a.runtime:
        # 20: the runtime's variables are read by this build, or it isn't the
        # build the runtime is for
        unset = build_reads(srv, list(fb.RUNTIMES[a.runtime]["env"]))
        if unset:
            raise SystemExit(f"runtime {a.runtime} sets {', '.join(unset)}, and this build "
                             "(llama-server and its libraries) never reads it: not the "
                             "build the runtime is for? Nothing was asked")
        rt = fb.RUNTIMES[a.runtime]
        # 22: a runtime that never gives its server the lda branch's variables
        # never runs a build that reads them — the other runtimes' of those
        # (LOOKAHEAD_MOE_ROUTER_NORM, in the lda build's libllama.so; upstream
        # names none)
        theirs = sorted({k for n, r in fb.RUNTIMES.items() if n != a.runtime for k in r["env"]
                         if fb.runtime_env_refused(a.runtime, {k: ""})})
        read = [k for k in theirs if k not in build_reads(srv, theirs)]
        if read:
            raise SystemExit(f"runtime {a.runtime} never runs a build that reads "
                             f"{', '.join(read)} (the lda branch's), and this one does. Nothing "
                             "was asked")
        # 22: and one that names its llama.cpp runs that one: its --version,
        # and the tarball's VERSION — before the GGUF's tens of GB are fetched
        if rt.get("build"):
            v0 = version_of(srv["bin"], env)
            why = fb.runtime_build_refused(a.runtime, v0["build"], v0["commit"], version_file(
                srv["root"] if a.server_dir else srv["bin"].parent.parent))
            if why:
                raise SystemExit(f"{why}: not the build the runtime is for. Nothing was asked")
    paths = [fetch(src, files, say) for src in split_parts(a.gguf)]
    gguf_path = paths[0]
    shas = [sha256_cached(x, state, say, files / "sha256.json") for x in paths]
    for x, h in zip(paths, shas):                      # 20: a pinned file is that file
        why = pin_refused(x.name, h, pins)
        if why:
            raise SystemExit(why)
    sha = shas[0] if len(paths) == 1 else split_sha(
        [(x.name, h) for x, h in zip(paths, shas)])
    # 17c: pinned once an answer is written — a first launch that failed (a
    # typo in --flags) pins nothing
    pinned = answered_here()
    if pinned and state.get("gguf_sha256") and state["gguf_sha256"] != sha:
        raise SystemExit(f"{out} holds answers of another file ({state['gguf_sha256'][:16]}, now "
                         f"{sha[:16]}): they can't be mixed — start another --out")
    # 24b: its quant, as the file's own header names it (general.file_type)
    import gguf_header
    ftype = (gguf_header.read(gguf_path) or {}).get("file_type")
    gguf = {"name": gguf_path.name, "sha256": sha, "size": sum(x.stat().st_size for x in paths),
            **({"file_type": str(ftype)} if ftype else {}),
            "source": a.gguf if a.gguf.startswith("hf://") else "a path on the box",
            **({"parts": [{"name": x.name, "sha256": h, "size": x.stat().st_size}
                          for x, h in zip(paths, shas)]} if len(paths) > 1 else {})}
    ver = version_of(srv["bin"], env)
    # 29: a runtime with fallbacks for its attention runs the first launch
    # this card takes (card-bolt: flash attention with a q8_0 cache, else f16,
    # else flash attention off) — checked before anything is asked, recorded
    attn = None
    if a.runtime and fb.RUNTIMES[a.runtime].get("attn_fallbacks"):
        gpu0 = rb.gpu_info()
        flags, attn = attn_flags(
            a.runtime, flags,
            lambda fs: _started(Server(srv, gguf_path, a.port, a.slots, ctx, fs, extra_env,
                                       out / SERVER_LOG), a.load_timeout, say, gguf_path),
            files, f"{rb.sha256_file(srv['bin'])[:16]}:{sha[:16]}:{gpu0.get('name')}:"
                   f"{a.slots}x{ctx}", say)
        if flags[:len(fb.RUNTIMES[a.runtime]["flags"])] != fb.RUNTIMES[a.runtime]["flags"]:
            # another cache type is another memory need: checked again
            fit2, _ = kv_fit(a.gguf, flags, a.slots, ctx, mem)
            if fit2 and fit2["fit"] < a.slots and not a.slots_fit:
                raise SystemExit(f"{attn_words(flags)} — the launch this card runs — fits "
                                 f"{fit2['fit']} slots of {ctx:,} tokens, not {a.slots}: give "
                                 f"--slots {fit2['fit']}. Nothing was asked")
    if a.serve:
        return serve_here(a, srv, gguf_path, gguf, flags, extra_env, ctx, out, say, own)
    server = {"version": ver["text"], "build": ver["build"], "commit": ver["commit"],
              "binary_sha256": rb.sha256_file(srv["bin"]),
              "tarball_sha256": srv.get("tarball_sha256"),
              **({"libs_sha256": srv["libs_sha256"],
                  "dir": {"path": str(srv["root"]), "libs": srv["lib_files"],
                          "git": git_of(srv["root"]),
                          "cuda_libs": ("the runner image's (torch's nvidia packages)"
                                        if srv["cuda_libs"] else "the build's own, or the box's"),
                          **({"version_file": version_file(srv["root"])}
                             if version_file(srv["root"]) else {}),
                          **(ran or {})}} if a.server_dir else {}),
              **({"runtime": a.runtime} if a.runtime else {}),
              # 29: the attention this card runs, and what it refused
              **({"attention": {"flags": attn["flags"], "words": attn_words(attn["flags"]),
                                "refused": [{"flags": r["flags"], "words": attn_words(r["flags"]),
                                             "why": r["why"]} for r in attn.get("refused") or []]}}
                 if attn else {}),
              **({"context_cap": cap} if cap else {}),
              "flags": flags, "env": extra_env, "slots": a.slots, "slot_context": ctx,
              "argv": ["llama-server", "-m", gguf["name"], "--host", "127.0.0.1", "--port",
                       str(a.port), "-c", str(ctx * a.slots), "-np", str(a.slots), "-ngl", "99",
                       "--jinja", "--metrics", *flags]}
    # 17b: one build, one launch in this --out: a resume with another binary,
    # other flags or another environment would mix two setups' answers
    setup = {"binary_sha256": server["binary_sha256"], "commit": ver["commit"],
             "build": ver["build"], "flags": flags, "env": extra_env,
             # 20: a build directory's libraries, and the runtime, pinned too
             **({"libs_sha256": server["libs_sha256"]} if a.server_dir else {}),
             **({"runtime": a.runtime} if a.runtime else {})}
    if pinned and state.get("setup") and state["setup"] != setup:
        was = state["setup"]
        diff = [k for k in setup if setup[k] != was.get(k)]
        raise SystemExit(f"{out} holds answers made with another setup — its "
                         + ", ".join(f"{k} {was.get(k)!r} then, {setup[k]!r} now" for k in diff)
                         + ": they can't be mixed — start another --out, or give the same")
    state["setup"] = setup
    state.setdefault("slots", []).append({"session": int(state.get("sessions", 0)) + 1,
                                          "slots": a.slots})
    # 23c: the price this session was given, or none
    state["prices"] = [*(state.get("prices") or []),
                       {"session": int(state.get("sessions", 0)) + 1, "per_hour": a.price}]
    server["sessions_slots"] = state["slots"]
    state.update({"served_as": a.served_as, "thinking": thinking, "gguf_sha256": sha,
                  "started_at": state.get("started_at")
                  or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "sessions": int(state.get("sessions", 0)) + 1,
                  "shard": f"{shard[0]}/{shard[1]}" if shard else None,
                  **({"label": a.label} if a.label else {})})
    sp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    progress(out, label=state.get("label") or "", model=a.served_as,
             thinking="on" if thinking else "off", tasks=tasks, shard=state["shard"],
             parity=bool(a.parity), started_at=state["started_at"], sessions=state["sessions"],
             state="starting", line="", incomplete={}, why="")
    gpu = rb.gpu_info()
    if shell_server_vars():
        say(f"not given to llama-server: {', '.join(shell_server_vars())}, set in this shell — "
            "its variables come from --env only, so each one it gets is recorded")
    if fit:
        # 17d: the estimate the slots were checked with, said and recorded.
        # 17e: either way, --slots-fit or not
        gb = 1024 ** 3
        server["kv_estimate"] = {k: fit[k] for k in ("used", "kv", "per_token", "file", "card",
                                                     "fit", "types")}
        mib = 1024 ** 2
        if raised:
            say(f"{raised[0]} slots asked, and up to {raised[1]['fit']} of {ctx:,} tokens fit "
                f"this card keeping {ROOM // mib:,} MiB spare: {a.slots} run (--max-slots "
                f"{a.max_slots})")
        if dropped:
            was, big_fit = dropped
            say(f"{was} slots of {ctx:,} tokens would leave "
                f"{int(big_fit['card'] - big_fit['used']) // mib:,} "
                f"MiB of this card's {big_fit['card'] // mib:,}, under the {ROOM // mib:,} kept "
                f"spare: {a.slots} run (--min-slots {a.min_slots})")
        say(f"memory, from {fit['basis']}: "
            f"about {fit['used'] / gb:.1f} GB for {a.slots} slots of {ctx:,} tokens "
            f"(the KV cache {fit['kv'] / gb:.1f} GB, {fit['types']}, "
            f"{fit['per_token'] / 1024:.1f} KB a token) with the {fit['file'] / gb:.1f} GB file "
            f"on a card with {fit['card'] / gb:.1f} GB free"
            + (f" ({in_use:,} MiB in use before this step)" if in_use else "") + ": " + (
                f"{int(fit['card'] - fit['used']) // mib:,} MiB spare" if fit["card"] >= fit["used"]
                else f"{int(fit['used'] - fit['card']) // mib:,} MiB short")
            + f"; up to {fit['fit']} slot{'s' if fit['fit'] != 1 else ''} fit keeping "
            f"{ROOM // mib:,}"
            + (f" — {a.slots} run, as --slots-fit says" if a.slots > fit["fit"] else ""))
    else:
        # 17e: said, where it carried on silently under the stated limit
        say(f"memory not worked out: {unread}. The slots were checked against "
            f"--max-context {a.max_context:,} only"
            + (", and --slots-fit says they fit" if a.slots_fit else ""))
    say(f"{gguf['name']} · {gguf['size'] / 1e9:.1f} GB · sha256 {sha[:16]} · llama.cpp "
        f"{ver['build'] or '?'} ({ver['commit'] or '?'}) · GPU {gpu.get('name') or 'unknown'} · "
        f"{a.slots} slots of {ctx:,} tokens")
    if a.server_dir:
        g = server["dir"]["git"]
        say(f"the build: {server['dir']['path']} · llama-server sha256 "
            f"{server['binary_sha256'][:16]} · with its {len(srv['lib_files'])} libraries "
            f"{srv['libs_sha256'][:16]} · "
            + (f"git {g['branch'] or '?'} at {(g['commit'] or '?')[:12]}"
               + (" (with changes not committed)" if g.get("dirty") else "")
               if g else f"its VERSION file: commit "
               f"{server['dir']['version_file'].get('commit', '?')[:12]}"
               if server["dir"].get("version_file") else "no .git near it, and no VERSION"))
    if a.runtime:
        # 20: the runtime's every value said, the repeat penalty among them
        say(fb.runtime_words(a.runtime, a.thinking))        # 29.2: this step's own setting

    # the questions, fetched before the GPU is spent: a gated set needs HF_TOKEN
    from service import config
    for t in [] if a.parity else tasks:
        try:
            items = fb.load(t, config.BENCH_ROOT)
        except Exception as e:                          # noqa: BLE001 — said in one line
            # 17c: the error's own words; the terms only when access was refused
            raise SystemExit(rb.scrub(fb.load_failed(t, e))) from None
        if a.pilot:
            # 29.1: --pilot-each's number here too, as pilot_ask asks it (it said
            # "the pilot's 20" of GPQA Diamond, then stopped at OTIS)
            mine = fb.pilot_items(t, config.BENCH_ROOT, a.pilot_each or None)
            say(f"{fb.BENCH[t]['label']}: the pilot's {len(mine)} of its {of_words(t, items)}, "
                f"each asked once, thinking {'on' if thinking else 'off'}")
            continue
        mine = fb.shard_of(items, shard)
        say(f"{fb.BENCH[t]['label']}: {len(mine)} of its {of_words(t, items)} here × "
            f"{fb.BENCH[t]['epochs']} run{'s' if fb.BENCH[t]['epochs'] > 1 else ''}")

    srv_proc = Server(srv, gguf_path, a.port, a.slots, ctx, flags, extra_env, out / SERVER_LOG)
    status = "failed"
    rec = None
    # 23c: the card's readings from the server's start to its end (a pilot,
    # no step of a plan, takes none)
    sampler = None if a.pilot else cc.Sampler(
        out / GPU_FILE, report=lambda use: progress(out, use=use))
    if sampler is not None:
        sampler.start()
    try:
        srv_proc.start(a.load_timeout, say, gguf_path)
        props = srv_proc.props()
        tmpl = props.get("chat_template") or ""
        server["chat_template_sha256"] = (hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
                                          if tmpl else None)
        server["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        server["total_slots"] = props.get("total_slots")
        # 22: a runtime that asks with the GGUF's own chat template — llama.cpp
        # serves it as the file holds it (its two patches are for other
        # families' templates) — is refused another, before a question
        want_tmpl = (fb.RUNTIMES.get(a.runtime or "") or {}).get("template_sha256")
        if want_tmpl and server["chat_template_sha256"] != want_tmpl:
            raise SystemExit(f"runtime {a.runtime} asks with the GGUF's own chat template "
                             f"(sha256 {want_tmpl[:16]}…), and this server's is "
                             f"{(server['chat_template_sha256'] or 'none')[:16]}…: another "
                             "file? Nothing was asked")
        if want_tmpl:
            say(f"the chat template: the GGUF's own (sha256 {want_tmpl[:16]}…), as runtime "
                f"{a.runtime} asks")
        # 29.2: and the file's own, read out of the GGUF — the one pinned, and
        # the one the server serves (as llama.cpp's Jinja lexer keeps it: the
        # file's ends in a newline it drops, so the two sha256 differ)
        rt_t = fb.RUNTIMES.get(a.runtime or "") or {}
        if rt_t.get("template_file_sha256"):
            import gguf_header
            ftmpl = gguf_header.chat_template(gguf_path)
            fsha = hashlib.sha256(ftmpl.encode("utf-8")).hexdigest() if ftmpl else None
            server["chat_template_file_sha256"] = fsha
            if fsha != rt_t["template_file_sha256"]:
                raise SystemExit(f"runtime {a.runtime} runs GGUFs whose own chat template is sha256 "
                                 f"{rt_t['template_file_sha256'][:16]}…, and {gguf_path.name}'s is "
                                 f"{(fsha or 'none')[:16]}…: another file? Nothing was asked")
            if tmpl != fb.as_lexed(ftmpl):
                raise SystemExit(f"this server doesn't serve {gguf_path.name}'s own chat template "
                                 "(its /props differs from the file's, as llama.cpp keeps it): "
                                 "--chat-template, or a build that patches it? Nothing was asked")
            say(f"the chat template, in the file: sha256 {fsha[:16]}…, the one pinned — and the one "
                "this server serves (its last newline dropped, as llama.cpp keeps a template)")
        # 29.2: the thinking switch, on this server: on writes the template's
        # marker into the prompt, off doesn't — before a question
        marker = rt_t.get("think_marker")
        if marker:
            from service import served as sv_t
            said = {}
            for on_ in (True, False):
                got = sv_t._post(srv_proc.base + "/apply-template", "", {
                    "messages": [{"role": "user", "content": "A question."}],
                    "chat_template_kwargs": {"enable_thinking": on_}}).get("prompt")
                said[on_] = isinstance(got, str) and marker in got
            server["thinking_switch"] = {"marker": marker, "on": said[True], "off": said[False]}
            if not said[True] or said[False]:
                raise SystemExit(f"the thinking switch doesn't work on this server: with "
                                 f"enable_thinking on the prompt {'holds' if said[True] else 'lacks'} "
                                 f"{marker}, with it off it {'holds' if said[False] else 'lacks'} it. "
                                 "Nothing was asked")
            say(f"the thinking switch: enable_thinking on writes {marker} into the prompt, off "
                "doesn't — checked on this server")
        # 17c: what its slots say of speculation — the import compares it with
        # the registered launch (a draft model under any spelling drafts)
        from service import served as sv
        try:
            seen = sv.probe(srv_proc.base)
        except ValueError:
            seen = {}
        if seen.get("speculative") is not None:
            server["speculative"] = bool(seen["speculative"])
        # 18e point 17: a runtime asks without a draft — a server whose slots
        # draft tokens (a draft head the GGUF carries, a draft model) is
        # refused here, before a question, not at the import
        if a.runtime and seen.get("speculative"):
            raise SystemExit(f"runtime {a.runtime} runs no speculative decoding, and this "
                             "server's slots say they draft tokens (its /slots): a GGUF with a "
                             "draft head, or a launch with a draft model? Nothing was asked")
        rec = register_here(a.served_as, a.served_as.split("/", 1)[1], a.based_on,
                            srv_proc.base, thinking,
                            f"llama.cpp {ver['build'] or '?'} on {gpu.get('name') or 'a GPU'}: "
                            + " ".join(server["argv"]), flags, extra_env, a.runtime,
                            a.slots)
        if sampler is not None:
            # 23c: the card check, before the first question — kept with this
            # session, said in the progress file at once. A failed probe is
            # said; the step goes on
            sampler.phase = "check"
            chk = {**cc.check(srv_proc.base, ctx, a.slots, say), "session": state["sessions"]}
            if own or a.runtime and fb.RUNTIMES[a.runtime].get("card"):
                # 29: the card named in the record, with its attention — a V100's
                # speed is never read as a 5090's
                chk["line"] = cc.record_line(chk, (server.get("attention") or {}).get("words")
                                             or attn_words(flags), cc.compute_cap())
                say(chk["line"])
            state["card_checks"] = [*(state.get("card_checks") or []), chk]
            sp.write_text(json.dumps(state, indent=1), encoding="utf-8")
            progress(out, card=chk)
            sampler.phase = "asking"
        elif a.pilot_each:
            # 29: an own box's pilot starts with the card check too (its
            # probe line, the V100's), kept in pilot.json
            server["card_check"] = cc.check(srv_proc.base, ctx, a.slots, say)
            server["card_check"]["line"] = cc.record_line(
                server["card_check"], (server.get("attention") or {}).get("words")
                or attn_words(flags), cc.compute_cap())
            say(server["card_check"]["line"])
        if own and not a.pilot:
            # 29: from here, someone else's process on the GPU pauses the run —
            # this server is the step's own, and the memory the card shows now
            # (the check's probe has run) is its own
            import gpu_free
            from service import frontier as sf_own
            sf_own.PAUSE = gpu_free.Gate(a.label, {srv_proc.proc.pid}, gpu_free.used_mib())
            say(f"{a.label}: the GPU is read for other processes before each benchmark and "
                "every minute while asking — someone else's pauses the run (nothing of theirs "
                "is touched)")
        if a.parity:
            from service import frontier as sf
            dest = out / "parity.jsonl"
            # 17c: what answered, for compare: the file, the launch, the slots
            ident = {"side": "box", "as": a.served_as,
                     # 17d: the file on disk, and llama-server's count of its
                     # weights (what the server's side has)
                     "file": {**{k: gguf[k] for k in ("name", "size", "sha256")},
                              "weights": seen.get("size")},
                     "server": {k: server[k] for k in ("build", "commit", "binary_sha256",
                                                       "flags", "env", "argv", "slots")},
                     "speculative": seen.get("speculative"), "gpu": gpu.get("name")}
            # 17d: each question twice — the box's agreement with itself, beside
            # its agreement with the server
            # 17i: its progress written as it goes — it read "starting" for its
            # whole run, then "stopped? paste its line again" past 45 minutes
            last = {"at": 0.0}

            def asked(k: int, of: int) -> None:
                say(f"parity {k} of {of}")
                if k >= of or time.time() - last["at"] >= 10:
                    progress(out, state="asking", line=f"parity {k:,} of {of:,}")
                    last["at"] = time.time()
            progress(out, state="asking", line=f"parity 0 of {a.n:,}",
                     session_at=round(time.time()))
            try:
                n = sf.parity_ask(rec, dest, asked, identity=ident, n=a.n, twice=True)
            except sv.ServerStopped as e:
                # 17g: in words, where it was a traceback; parity has no
                # resume — pasting the box's line asks it again
                raise SystemExit(f"parity: {getattr(e, 'refused', '') or e} — nothing kept; "
                                 "paste the box's line again to ask it again") from None
            say(f"parity: {n} answers · {dest} — frontier_fetch.py fetches it, and compares "
                "it with the server's when given that (--parity)")
            progress(out, state="whole", parity_file={"name": dest.name, "answers": n,
                                                      "sha256": rb.sha256_file(dest)})
            return 0
        if a.pilot:
            # 22: the pilot's questions, a line each, and its verdict last
            got = pilot_ask(rec, tasks, out, say, a.slots, each=a.pilot_each, on=thinking)
            # 29: an own box's pilot keeps one file a thinking setting
            summary = out / (PILOT_SUMMARY if not a.pilot_each else
                             f"pilot-thinking-{'on' if thinking else 'off'}.json")
            summary.write_text(json.dumps({
                **got, "model": a.served_as, "runtime": a.runtime, "slots": a.slots,
                "label": a.label or None, "gpu": gpu.get("name"),
                "thinking": "on" if thinking else "off",
                "gguf": {k: gguf[k] for k in ("name", "size", "sha256")},
                "server": {k: server.get(k) for k in ("build", "commit", "binary_sha256",
                                                      "tarball_sha256", "chat_template_sha256",
                                                      "chat_template_file_sha256",
                                                      "thinking_switch",
                                                      "attention", "context_cap", "card_check",
                                                      "slot_context", "kv_estimate")
                           if server.get(k) is not None},
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1),
                encoding="utf-8")
            return 0 if got["separated"] else 1
        sid = db.add(a.served_as, "instruct", SUITE, a.by,
                     f"run on {OWN_WHERE if own else WHERE} (remote_gguf.py)",
                     thinking=thinking, tasks=tasks if len(tasks) < len(fb.TASKS) else None)
        watch = Watch(sid, say, out=out)
        progress(out, state="asking", session_at=round(time.time()))
        watch.start()
        try:
            runner.run_submission(db.get(sid))
        finally:
            watch.stop()
            watch.tick()
            if sampler is not None:
                sampler.phase = "after"
        row = db.get(sid) or {}
        status = row.get("status") or "failed"
        say(f"the run: {status} · {row.get('progress') or ''}"
            + (f" · {row['error']}" if row.get("error") and row["error"] != row.get("progress")
               else ""))
    finally:
        srv_proc.stop()
        if sampler is not None:
            sampler.stop()
            if sampler.error:
                say(f"the card's power and use couldn't be read: {sampler.error}")
            use = cc.summarise(sampler.samples)
            if use.get("wh") is not None:
                say(f"the card while it ran: {use['wh']:.1f} Wh"
                    + (f" · {cc.use_words(use)} while asking" if cc.use_words(use) else "")
                    + (f" · peak memory {use['peak_mib']:,} MiB" if use.get("peak_mib") else ""))
            progress(out, use=cc.running(sampler.samples, time.time()))
    if rec is None:
        progress(out, state="stopped", why="llama-server didn't come up")
        return 1
    path, done, incomplete = make_bundle(out, a.served_as, thinking, tasks, state, shard, gguf,
                                         server, rec)
    for t, x in incomplete.items():
        say(f"{t}: {x['answers']:,} of {x['of']:,} answered — run the same command again to "
            "carry on")
    why = (row.get("error") or row.get("progress") or "")[:400]
    if status == "paused":
        # 29: someone else's process on our own GPU — the step waits for it
        # to go (frontier_box.py), its answers kept, and Runs says why
        say(f"{a.label}: paused — {why}")
        progress(out, state="paused", incomplete=incomplete, why=why,
                 **({"bundle": {"name": path.name, "sha256": rb.sha256_file(path)}}
                    if path else {}))
        return PAUSED
    if not path:
        say("no task is answered whole yet: no bundle")
        progress(out, state="stopped", incomplete=incomplete, why=why or "no task is whole yet")
        return 1
    sha = rb.sha256_file(path)
    say(f"bundle {path} · {path.stat().st_size / 1024 ** 2:.1f} MB · sha256 "
        f"{sha[:16]} · " + ", ".join(f"{t} ({x['answers']:,} answers)" for t, x in done.items()))
    # 17h: how many answers were written off — a whole step with none isn't
    # asked again by --ask-written-off
    from service import config
    from service import frontier as sf
    row = config.OUT_DIR / (a.served_as.replace("/", "__") + ("__thinking" if thinking else ""))
    off = sum(1 for t in done for r in sf.read_answers(sf.task_dir(row, t) / sf.ANSWERS).values()
              if r.get("unanswered"))
    progress(out, state="stopped" if incomplete else "whole", incomplete=incomplete,
             bundle={"name": path.name, "sha256": sha}, why=why if incomplete else "",
             written_off=off)
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
