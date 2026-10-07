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
import remote_bundle as rb  # noqa: E402
from remote_run import Out, board_commit  # noqa: E402

SUITE = "frontier"
STATE = "remote_gguf.json"
# 17f: what the step is doing, for frontier_fetch.py to read without
# importing anything — its label, the model, the benchmark it asks, its line,
# when it started and its sessions; never the box's address
PROGRESS = "progress.json"
OWN_LOG = "remote_gguf.log"
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


def settings(out: Path, slots: int, shard: tuple[int, int] | None,
             ask_again: bool = False) -> dict:
    bench = out / "bench"
    return {"BENCH_ROOT": bench, "RESULTS_ROOT": bench / "results", "LOGS_DIR": bench / "logs",
            "DB_PATH": bench / "service.sqlite3",
            "FRONTIER_SHARD": f"{shard[0]}/{shard[1]}" if shard else "",
            "FRONTIER_SCORE_AFTER_RUN": False, "FRONTIER_WHERE": WHERE,
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
    from huggingface_hub import hf_hub_download
    kind = "dataset" if m["type"] == "datasets/" else "model"
    say(f"fetching {m['path']} from {m['repo']}" + (f" at {m['rev']}" if m["rev"] else ""))
    try:
        got = hf_hub_download(m["repo"], m["path"], repo_type=kind, revision=m["rev"],
                              token=os.environ.get("HF_TOKEN") or None, local_dir=str(into))
    except Exception as e:                          # noqa: BLE001 — said in one line
        raise SystemExit(rb.scrub(f"{src} could not be fetched: {e}. A private repo needs "
                                  "HF_TOKEN set on this box (typed here, never stored)")) from None
    return Path(got).resolve()


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
SERVER_VARS = ("LLAMA_", "GGML_")


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


def kv_per_token(shape: dict, ctk: str, ctv: str) -> float | None:
    """bytes of KV cache a token takes, every layer with attention"""
    if ctk not in CACHE_BYTES or ctv not in CACHE_BYTES:
        return None
    return sum(h * (shape["key_length"] * CACHE_BYTES[ctk]
                    + shape["value_length"] * CACHE_BYTES[ctv]) for h in shape["kv_heads"])


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
    per_slot = (per_token + CTX_BUFFERS) * ctx + recurrent_per_slot(shape)
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
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
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


class Server:
    """llama-server on 127.0.0.1, started and stopped"""

    def __init__(self, srv: dict, gguf: Path, port: int, slots: int, ctx: int,
                 flags: list[str], env: dict, log: Path):
        self.argv = [str(srv["bin"]), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
                     "-c", str(ctx * slots), "-np", str(slots), "-ngl", "99", "--jinja",
                     "--metrics", *flags]
        self.env, self.log, self.port = server_env(srv, env), log, port
        self.base = f"http://127.0.0.1:{port}"
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
                  how: str, flags: list[str] | None = None, env: dict | None = None) -> dict:
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
           "speculative": p.get("speculative")}
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
    per = {}
    for t in tasks:
        spec = fb.BENCH[t]
        s = sf.settings(rec, t, thinking)
        per[t] = {"protocol_version": spec["protocol_version"],
                  "revision": spec["source"]["revision"], "epochs": spec["epochs"],
                  "budget": s["max_tokens"], "family": sf.family_of(rec) or "",
                  "sampling": {k: v for k, v in s.items()
                               if k not in ("max_tokens", "chat_template_kwargs")},
                  "switch": s.get("chat_template_kwargs"),
                  **(done.get(t) or {}), **({"incomplete": True} if t in incomplete else {})}
    setup = {"suite": SUITE, "protocol": fb.VERSION, "served_as": served_as,
             "based_on": rec.get("based_on") or "", "thinking": "on" if thinking else "off",
             "gguf": gguf, "server": server, "tasks": per, "gpu": rb.gpu_info(),
             "where": WHERE, "runner": "service/runner.py → service/frontier.py, called directly",
             "board_commit": board_commit(), "sessions": state.get("sessions", 1),
             # 17f: G5's box, and the runner image's tag (the board's commit it was built at)
             "box": state.get("label") or None, "image": (board_commit() or "")[:7] or None,
             "shard": {"i": shard[0], "n": shard[1]} if shard else None,
             "started_at": state.get("started_at"),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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


_PROGRESS_LOCK = threading.Lock()


def progress(out: Path, **fields) -> None:
    """17f: the step's progress file, its fields merged in, the time it was
    last written with them — 17i: one writer at a time (the parity step's
    questions are asked side by side, and each writes its count)"""
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
    ap.add_argument("--server", required=True,
                    help="the llama-server tarball (build_llama_tarball.sh): hf://… or a path")
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
    flags = shlex.split(a.flags)
    if own_flags(flags):
        ap.error(f"--flags: {', '.join(own_flags(flags))} are set by this script")
    extra_env = parse_env(a.env)
    out = Path(a.out).resolve()
    thinking = a.thinking == "on"
    STEP["out"] = out
    progress(out, label=a.label or "", model=a.served_as, thinking=a.thinking,
             parity=bool(a.parity), tasks=[t for t in fb.TASKS if t in (a.only or [])],
             state="starting", why="", line="", incomplete={})
    if not a.only and not a.parity:
        # 17d: a box names its benchmarks, thinking on or off — every one at
        # once is more context than a card holds, and asks ARC-AGI-2 with
        # thinking off
        ap.error("--only: name this box's benchmarks, one --only each (G5's table)")
    tasks = [t for t in fb.TASKS if t in (a.only or fb.TASKS)]
    if a.parity:
        # 17b: the parity check's questions, thinking off, whatever --thinking says
        tasks, thinking = [fb.PARITY["task"]], False
    # 17c: a context the card can't hold is refused here, before the download.
    # 17d: worked out from the GGUF's header, its cache type and this card;
    # the stated --max-context only when those can't be read
    ctx = fb.slot_context(tasks, thinking)
    # 17h: what is free on the card, not its total — 300 MiB in use by
    # something else left 1,271 MiB where 1,570 was promised
    card = rb.gpu_info()
    in_use = int(card.get("memory_used_mib") or 0)
    mem = (card["memory_mib"] - in_use) if card.get("memory_mib") else None
    fit, unread = kv_fit(a.gguf, shlex.split(a.flags), a.slots, ctx, mem)
    big = max(tasks, key=lambda t: fb.slot_context([t], thinking))
    # 17f: a step planned at 8 slots runs what the card holds, down to
    # --min-slots, and says so (HLE's 86,016 with thinking on: 7 on a 5090)
    dropped = None
    if fit and a.min_slots and a.min_slots <= fit["fit"] < a.slots and not a.slots_fit:
        dropped = (a.slots, fit)
        a.slots = fit["fit"]
        fit, unread = kv_fit(a.gguf, shlex.split(a.flags), a.slots, ctx, mem)
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
    paths = [fetch(src, files, say) for src in split_parts(a.gguf)]
    gguf_path = paths[0]
    shas = [sha256_cached(x, state, say, files / "sha256.json") for x in paths]
    sha = shas[0] if len(paths) == 1 else split_sha(
        [(x.name, h) for x, h in zip(paths, shas)])
    # 17c: pinned once an answer is written — a first launch that failed (a
    # typo in --flags) pins nothing
    pinned = answered_here()
    if pinned and state.get("gguf_sha256") and state["gguf_sha256"] != sha:
        raise SystemExit(f"{out} holds answers of another file ({state['gguf_sha256'][:16]}, now "
                         f"{sha[:16]}): they can't be mixed — start another --out")
    gguf = {"name": gguf_path.name, "sha256": sha, "size": sum(x.stat().st_size for x in paths),
            "source": a.gguf if a.gguf.startswith("hf://") else "a path on the box",
            **({"parts": [{"name": x.name, "sha256": h, "size": x.stat().st_size}
                          for x, h in zip(paths, shas)]} if len(paths) > 1 else {})}
    tb = fetch(a.server, files, say)
    srv = unpack_server(tb, out / "server")
    env = server_env(srv, extra_env)
    ver = version_of(srv["bin"], env)
    server = {"version": ver["text"], "build": ver["build"], "commit": ver["commit"],
              "binary_sha256": rb.sha256_file(srv["bin"]), "tarball_sha256": srv["tarball_sha256"],
              "flags": flags, "env": extra_env, "slots": a.slots, "slot_context": ctx,
              "argv": ["llama-server", "-m", gguf["name"], "--host", "127.0.0.1", "--port",
                       str(a.port), "-c", str(ctx * a.slots), "-np", str(a.slots), "-ngl", "99",
                       "--jinja", "--metrics", *flags]}
    # 17b: one build, one launch in this --out: a resume with another binary,
    # other flags or another environment would mix two setups' answers
    setup = {"binary_sha256": server["binary_sha256"], "commit": ver["commit"],
             "build": ver["build"], "flags": flags, "env": extra_env}
    if pinned and state.get("setup") and state["setup"] != setup:
        was = state["setup"]
        diff = [k for k in setup if setup[k] != was.get(k)]
        raise SystemExit(f"{out} holds answers made with another setup — its "
                         + ", ".join(f"{k} {was.get(k)!r} then, {setup[k]!r} now" for k in diff)
                         + ": they can't be mixed — start another --out, or give the same")
    state["setup"] = setup
    state.setdefault("slots", []).append({"session": int(state.get("sessions", 0)) + 1,
                                          "slots": a.slots})
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

    # the questions, fetched before the GPU is spent: a gated set needs HF_TOKEN
    from service import config
    for t in [] if a.parity else tasks:
        try:
            items = fb.load(t, config.BENCH_ROOT)
        except Exception as e:                          # noqa: BLE001 — said in one line
            # 17c: the error's own words; the terms only when access was refused
            raise SystemExit(rb.scrub(fb.load_failed(t, e))) from None
        mine = fb.shard_of(items, shard)
        say(f"{fb.BENCH[t]['label']}: {len(mine)} of its {len(items)} questions here × "
            f"{fb.BENCH[t]['epochs']} run{'s' if fb.BENCH[t]['epochs'] > 1 else ''}")

    srv_proc = Server(srv, gguf_path, a.port, a.slots, ctx, flags, extra_env, out / SERVER_LOG)
    status = "failed"
    rec = None
    try:
        srv_proc.start(a.load_timeout, say, gguf_path)
        props = srv_proc.props()
        tmpl = props.get("chat_template") or ""
        server["chat_template_sha256"] = (hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
                                          if tmpl else None)
        server["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        server["total_slots"] = props.get("total_slots")
        # 17c: what its slots say of speculation — the import compares it with
        # the registered launch (a draft model under any spelling drafts)
        from service import served as sv
        try:
            seen = sv.probe(srv_proc.base)
        except ValueError:
            seen = {}
        if seen.get("speculative") is not None:
            server["speculative"] = bool(seen["speculative"])
        rec = register_here(a.served_as, a.served_as.split("/", 1)[1], a.based_on,
                            srv_proc.base, thinking,
                            f"llama.cpp {ver['build'] or '?'} on {gpu.get('name') or 'a GPU'}: "
                            + " ".join(server["argv"]), flags, extra_env)
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
        sid = db.add(a.served_as, "instruct", SUITE, a.by, "run on a rented GPU (remote_gguf.py)",
                     thinking=thinking, tasks=tasks if len(tasks) < len(fb.TASKS) else None)
        watch = Watch(sid, say, out=out)
        progress(out, state="asking", session_at=round(time.time()))
        watch.start()
        try:
            runner.run_submission(db.get(sid))
        finally:
            watch.stop()
            watch.tick()
        row = db.get(sid) or {}
        status = row.get("status") or "failed"
        say(f"the run: {status} · {row.get('progress') or ''}"
            + (f" · {row['error']}" if row.get("error") and row["error"] != row.get("progress")
               else ""))
    finally:
        srv_proc.stop()
    if rec is None:
        progress(out, state="stopped", why="llama-server didn't come up")
        return 1
    path, done, incomplete = make_bundle(out, a.served_as, thinking, tasks, state, shard, gguf,
                                         server, rec)
    for t, x in incomplete.items():
        say(f"{t}: {x['answers']:,} of {x['of']:,} answered — run the same command again to "
            "carry on")
    why = (row.get("error") or row.get("progress") or "")[:400]
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
