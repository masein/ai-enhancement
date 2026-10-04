"""15.1, 15.2: a row run on another machine, as one file — what a rented GPU's
run (scripts/remote_run.py) hands the server (scripts/import_remote.py).

The suite is a parameter: DeviceMark's battery now. The Mobile tasks' "no
judge" part and Mobile-MMLU-Pro for Hugging Face models are the next
candidates; each needs only its entry in SUITES, its tasks' check and its
scoring on import.

A bundle, devicemark-<model>-thinking-<on|off>.tar.gz (15.5: a shard's,
devicemark-<model>-thinking-<on|off>-shard-<i>-of-<n>.tar.gz), holds:
  bundle.json    what it is: the suite, the model and its mode, the row's folder,
                 each task's answers, a digest of every answer, each file's
                 sha256, and the shard it holds when it is one
  setup.json     where it ran: the GPU and its driver, Python, torch and its
                 CUDA, transformers, lm_eval and the fast kernels, the model's
                 revision, the battery's hashes, the protocol's version, what
                 lm_eval was told, each task's pace
  run.log        the run's log, every session of it in order
  results/<row>/model_meta.json and results/<row>/<task>_0shot/…
                 the answers in the layout the board reads: lm_eval's samples and
                 results files, never its cache
Written the same, byte for byte, for the same files: sorted, no times, no
owners. No token is ever in it: the log and the setup are scrubbed of every
secret the environment holds, and the setup never reads the environment.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAT = 1
SUITES = {"devicemark": {"prefix": "devicemark",
                         "tasks": ("dm_ifeval", "dm_mmlu_pro", "dm_math")}}
# what must be the server's, exactly, for a bundle to be imported
PINNED = ("torch", "torch_cuda", "transformers", "lm_eval", "fla_core", "fast_kernels")
PINNED_WORDS = {"torch": "torch", "torch_cuda": "torch's CUDA", "transformers": "transformers",
                "lm_eval": "lm_eval", "fla_core": "fla-core (a fast kernel)",
                "fast_kernels": "the prebuilt fast kernels"}
# the environment's secrets: their values never reach a bundle
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENROUTER_API_KEY", "LLM_API_KEY",
           "JUDGE_API_KEY", "EXAM_API_KEY", "SUBMIT_TOKEN", "VAST_API_KEY")
MAX_BYTES = 2 * 1024 ** 3                  # a bundle is answers and a log: far less


def refused(suite: str) -> str:
    """14.4.4: '' when a suite's answers may travel in a bundle; else why not —
    a bundle leaves the server (a rented box, a download), and a
    non-commercial set never does"""
    sys.path.insert(0, str(HERE))
    import restrictions
    return restrictions.stays_here(SUITES[suite]["tasks"])


def bundle_name(suite: str, model: str, thinking: bool,
                shard: tuple[int, int] | None = None) -> str:
    """"devicemark-Qwen__Qwen3.5-4B-thinking-on.tar.gz", and 15.5's shards
    "devicemark-Qwen__Qwen3.5-4B-thinking-on-shard-1-of-2.tar.gz" """
    why = refused(suite)
    if why:
        raise ValueError(why)
    return (f"{SUITES[suite]['prefix']}-{model.replace('/', '__')}-thinking-"
            f"{'on' if thinking else 'off'}"
            + (f"-shard-{shard[0]}-of-{shard[1]}" if shard else "") + ".tar.gz")


def shard_of_bundle(bundle: dict) -> tuple[int, int] | None:
    """15.5: the shard a bundle holds, (i, n) — None for a whole run. A value
    that isn't one raises ValueError"""
    s = bundle.get("shard")
    if not s:
        return None
    try:
        i, n = int(s["i"]), int(s["n"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"its shard {s!r} isn't i of n") from None
    if not 1 <= i <= n or n < 2:
        raise ValueError(f"its shard {i} of {n} isn't one")
    return i, n


# ---------------------------------------------------------------------------
# where it ran
# ---------------------------------------------------------------------------

def _dist(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def library_versions() -> dict:
    """the versions a row depends on, as this Python has them: torch (its
    release and its CUDA), transformers, lm_eval, and the fast kernels — the
    pinned variant of the two prebuilt ones when both import, and fla-core.
    15.6: datasets, pyarrow, accelerate, huggingface_hub and tokenizers too"""
    out: dict = {"python": platform.python_version()}
    try:
        import torch
        out.update(torch=torch.__version__.split("+")[0], torch_build=torch.__version__,
                   torch_cuda=torch.version.cuda)
    except ImportError:
        out.update(torch=None, torch_build=None, torch_cuda=None)
    out["transformers"] = _dist("transformers")
    out["lm_eval"] = _dist("lm_eval") or _dist("lm-eval")
    out["fla_core"] = _dist("fla-core")
    # 15.6: what read the battery's datasets and loaded the model (recorded;
    # the items hash and the pins above are the checks)
    out["datasets"] = _dist("datasets")
    out["pyarrow"] = _dist("pyarrow")
    out["accelerate"] = _dist("accelerate")
    out["huggingface_hub"] = _dist("huggingface_hub") or _dist("huggingface-hub")
    out["tokenizers"] = _dist("tokenizers")
    on = os.environ.get("FAST_KERNELS", "1").strip().lower() not in ("0", "off", "no", "false")
    both = on and all(importlib.util.find_spec(m) for m in ("mamba_ssm", "causal_conv1d"))
    out["fast_kernels"] = (json.loads((HERE / "fast_kernels.json").read_text(encoding="utf-8"))
                           ["variant"] if both else None)
    return out


def pin_differences(theirs: dict, ours: dict) -> list[str]:
    """each pinned library whose version differs, in words"""
    return [f"{PINNED_WORDS[k]}: the bundle's {theirs.get(k) or 'none'}, this server's "
            f"{ours.get(k) or 'none'}" for k in PINNED if theirs.get(k) != ours.get(k)]


def gpu_info() -> dict:
    """the GPU's name, its driver and its memory, as nvidia-smi says"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=30).stdout.splitlines()[0]
        name, driver, mib = (x.strip() for x in out.split(","))
        return {"name": name, "driver": driver, "memory_mib": int(float(mib))}
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return {"name": None, "driver": None, "memory_mib": None}


def scrub(text: str, env: dict | None = None) -> str:
    """the text with every secret the environment holds taken out"""
    env = os.environ if env is None else env
    for var in SECRETS:
        v = env.get(var) or ""
        if len(v) >= 6:
            text = text.replace(v, f"[{var} withheld]")
    return text


# ---------------------------------------------------------------------------
# the answers
# ---------------------------------------------------------------------------

def samples_files(task_out: Path, task: str) -> list[Path]:
    return sorted(task_out.rglob(f"samples_{task}_*.jsonl"))


def task_answers(task_out: Path, task: str) -> dict[str, str]:
    """the answers scoring reads — the task's newest samples file — by key"""
    files = samples_files(task_out, task)
    out: dict[str, str] = {}
    for line in files[-1].read_text(encoding="utf-8").splitlines() if files else []:
        try:
            s = json.loads(line)
        except ValueError:
            continue
        key = (s.get("doc") or {}).get("key")
        if key is not None:
            out[str(key)] = ((s.get("resps") or [[""]])[0] or [""])[0] or ""
    return out


def answers_digest(answers: dict[str, dict[str, str]]) -> str:
    """one sha256 over every task's answers, key by key"""
    rows = [[t, k, answers[t][k]] for t in sorted(answers) for k in sorted(answers[t])]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def cache_count(task_out: Path) -> int:
    """how many answers lm_eval's cache holds for a task (runner.dm_cache)"""
    import sqlite3
    n = 0
    for db in task_out.glob("lm-cache/*.db"):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5) as c:
                n += c.execute("SELECT COUNT(*) FROM unnamed").fetchone()[0]
        except sqlite3.Error:
            continue
    return n


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, files: dict[str, bytes]) -> Path:
    """the bundle: every file, sorted, with no times and no owners"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                filename="") as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            data = files[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tar.addfile(ti, io.BytesIO(data))
    tmp.replace(path)
    return path


_SAFE = re.compile(r"^(bundle\.json|setup\.json|run\.log|results/[^/]+/[^\0]+)$")


def read(path: Path) -> dict:
    """{bundle, setup, log, files: {name: bytes}, sha256} — or ValueError
    saying what is wrong with the file. Only regular files under the names a
    bundle has; nothing outside them, nothing climbing out"""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name} is larger than a bundle can be")
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:gz") as tar:
            for m in tar.getmembers():
                name = m.name
                if (not m.isfile() or not _SAFE.match(name) or name.startswith("/")
                        or ".." in Path(name).parts or "\\" in name):
                    raise ValueError(f"{path.name} holds {name!r}, which no bundle does")
                files[name] = tar.extractfile(m).read()
    except (tarfile.TarError, OSError, EOFError) as e:
        raise ValueError(f"{path.name} isn't a bundle: {e}") from None
    for need in ("bundle.json", "setup.json"):
        if need not in files:
            raise ValueError(f"{path.name} has no {need}")
    bundle = json.loads(files["bundle.json"])
    setup = json.loads(files["setup.json"])
    for name, want in (bundle.get("files") or {}).items():
        got = files.get(name)
        if got is None or hashlib.sha256(got).hexdigest() != want:
            raise ValueError(f"{path.name}: {name} isn't as its bundle.json says")
    return {"bundle": bundle, "setup": setup,
            "log": files.get("run.log", b"").decode("utf-8", "replace"),
            "files": files, "sha256": sha256_file(path)}
