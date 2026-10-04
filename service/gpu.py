"""16.2: GPU memory — one status, said the same way everywhere.

The server has one card, shared by the judge (gemma-vllm), the board's runs,
the Playground's models and llama-server processes started by hand. This says
how much of it is in use and free, and what holds it, from what the board
knows for certain:

- the card's total, used and free memory, from nvidia-smi;
- each process's memory, when nvidia-smi can give it, named by what it runs:
  the judge (vLLM), a served model (llama-server, by its port), a GGUF run
  (llama-perplexity), a board run (a child of this service), the Playground
  (this service's own process). What it runs is its command line, or — when
  the container can't read a host process's /proc, as on the server —
  nvidia-smi's own name for it ("VLLM::EngineCore"): a llama-server's port
  isn't in that, so it is the one served model answering, else "llama-server"
  (16.8);
- what it can't name is "other". When nvidia-smi can't give memory for each
  process, the Playground's own (torch's count, in this process) is the one
  number besides the card's, and the rest is "other: N GB" — never a guess
  shown as a fact.

GB here is 1024³ bytes, as nvidia-smi's MiB are 1024² — so the 32 GB card
reads 31.8 GB, not 34.2. Every line people read says "GPU memory" in GB
(`gb_text`). Cached for a few seconds; nothing waits on it."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path


GB = 1024 ** 3
MIB = 1024 ** 2
TTL_S = 5.0
WHY_NONE = "GPU memory can't be read on this server"

_cache: dict = {}
_lock = threading.Lock()


def gb(n_bytes: float | None) -> float | None:
    return None if n_bytes is None else n_bytes / GB


def gb_text(n_bytes: float | None) -> str:
    """"12.6 GB" — one decimal, "about 0.1 GB" at the least"""
    if n_bytes is None:
        return "—"
    v = n_bytes / GB
    return f"{v:.0f} GB" if v >= 99.5 else f"{max(v, 0.1):.1f} GB"


def mib_to_bytes(mib: float | None) -> float | None:
    return None if mib is None else float(mib) * MIB


# ---------------------------------------------------------------------------
# what nvidia-smi says
# ---------------------------------------------------------------------------

def _smi(args: list[str], timeout: float = 10.0) -> str:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                         timeout=timeout, check=True)
    return out.stdout


def card() -> dict | None:
    """{total, used, free} in bytes for the first GPU, or None (no nvidia-smi,
    no GPU, or an answer that isn't numbers)"""
    try:
        line = _smi(["--query-gpu=memory.total,memory.used,memory.free",
                     "--format=csv,noheader,nounits"]).splitlines()[0]
        total, used, free = (float(x) for x in line.split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {"total": total * MIB, "used": used * MIB, "free": free * MIB}


def processes() -> list[tuple[int, float, str]] | None:
    """[(pid, bytes, name)] for each process on the card — name is
    nvidia-smi's own ("VLLM::EngineCore") — or None when nvidia-smi can't give
    memory for each (a container often sees "[N/A]"). A header line and "MiB"
    after a number are read past, as `--format=csv` writes them"""
    try:
        text = _smi(["--query-compute-apps=pid,used_memory,process_name",
                     "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError):
        return None
    out = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) < 2 or not parts[0].isdigit():
            continue                          # a header, or a blank line
        try:
            n = float(re.sub(r"\s*MiB$", "", parts[1])) * MIB
        except ValueError:
            return None                       # "[N/A]": no memory for each process
        out.append((int(parts[0]), n, parts[2] if len(parts) > 2 else ""))
    return out


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace")
    except OSError:
        return ""


def _parent(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _descends(pid: int, ancestor: int) -> bool:
    for _ in range(32):
        if pid in (0, 1):
            return False
        pid = _parent(pid)
        if pid == ancestor:
            return True
    return False


_PORT = re.compile(r"--port[ =](\d+)")


# ---------------------------------------------------------------------------
# who holds it
# ---------------------------------------------------------------------------

def _run_now() -> dict | None:
    """the board run holding the GPU: {sid, model, left} — its time left from
    its own progress line"""
    from . import db, runner
    run = runner.run_holding()
    if not run:
        return None
    sub = db.get(run["sid"]) if run.get("sid") else None
    m = re.search(r"about [^·]*? left", (sub or {}).get("progress") or "")
    return {"sid": run.get("sid") or 0, "model": run.get("hf_id") or "",
            "left": m.group(0) if m else ""}


def run_words(run: dict) -> str:
    """"run #181 (Qwen3.5-2B, about 40 min left)" — or "a command-line run" """
    if not run.get("sid"):
        return "a command-line run"
    bits = [b for b in (run.get("model", "").split("/")[-1], run.get("left")) if b]
    return f"run #{run['sid']}" + (f" ({', '.join(bits)})" if bits else "")


def _chat_loaded() -> list[str]:
    from . import chat
    return [ld.model.split("/")[-1] for ld in list(chat.ENGINE.loaded.values())
            if ld.device == "cuda"]


def _chat_bytes() -> float | None:
    """what the Playground's models hold, as torch counts it in this process"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_reserved())
    except Exception:                                   # noqa: BLE001 — no torch, no CUDA
        return None


def _served_ports() -> dict[int, str]:
    from . import db
    out = {}
    for rec in db.served_all():
        m = re.search(r":(\d+)(?:/|$)", rec.get("base_url") or "")
        if m:
            out.setdefault(int(m.group(1)), rec.get("name") or rec["id"])
    return out


def _served_answering() -> str:
    """16.8: a llama-server whose port the board can't see: the registered
    served model answering, when exactly one server is — by the Playground's
    own background check (chat.served_up), never asked here. Setups on one
    server are one server. '' when none answers, or more than one does"""
    from . import chat, db, served
    up: dict[str, list[str]] = {}
    for rec in db.served_all():
        if served.is_openrouter(rec) or chat.served_up(rec["id"]) is not True:
            continue
        up.setdefault(str(rec.get("base_url") or "").rstrip("/"), []).append(
            rec.get("name") or rec["id"])
    return " / ".join(sorted(next(iter(up.values())))) if len(up) == 1 else ""


def holders(c: dict, procs: list[tuple[int, float]] | None) -> list[dict]:
    """[{kind, name, bytes}] — bytes None where the board can't say. Kinds:
    judge, run, gguf, chat, served, other"""
    run = _run_now()
    loaded = _chat_loaded()
    me = os.getpid()
    out: list[dict] = []

    def add(kind, name, n):
        for h in out:
            if h["kind"] == kind and h["name"] == name:
                h["bytes"] = (h["bytes"] or 0) + n if n is not None else h["bytes"]
                return
        out.append({"kind": kind, "name": name, "bytes": n})

    if procs is not None:
        ports = _served_ports()
        for pid, n, *smi in procs:
            # 16.8: its command line, else nvidia-smi's name for it — the
            # container can't read a host process's /proc
            cmd = _cmdline(pid) or (smi[0] if smi else "")
            low = cmd.lower()
            if pid == me:
                add("chat", "the Playground" + (f": {', '.join(loaded)}" if loaded else ""), n)
            elif "vllm" in low:
                add("judge", "the judge", n)
            elif "llama-server" in low:
                port = _PORT.search(cmd)
                name = ports.get(int(port.group(1))) if port else _served_answering()
                add("served", name or ("llama-server" + (f" on port {port.group(1)}" if port
                                                          else "")), n)
            elif "llama-perplexity" in low:
                add("gguf", "a GGUF run (llama.cpp)", n)
            elif _descends(pid, me) or (run and "lm_eval" in cmd):
                add("run", run_words(run) if run else "a board run", n)
            else:
                add("other", "other", n)
        if run and not any(h["kind"] == "run" for h in out):
            add("run", run_words(run), None)            # its process isn't on the card yet
        return out
    # no memory for each process: what the board knows for certain, and the rest
    known = 0.0
    chat_n = _chat_bytes() if loaded else None
    if loaded:
        add("chat", "the Playground: " + ", ".join(loaded), chat_n)
        known += chat_n or 0
    if run:
        add("run", run_words(run), None)
    rest = max(0.0, c["used"] - known)
    if rest >= 0.05 * GB:
        add("other", "other", rest)
    return out


def _compute() -> dict:
    c = card()
    if c is None:
        return {"ok": False, "why": WHY_NONE + " (nvidia-smi didn't answer)", "at": time.time(),
                "holders": [], "line": WHY_NONE + "."}
    procs = processes()
    hs = holders(c, procs)
    return {"ok": True, "total": c["total"], "used": c["used"], "free": c["free"],
            "total_gb": round(c["total"] / GB, 1), "used_gb": round(c["used"] / GB, 1),
            "free_gb": round(c["free"] / GB, 1), "per_process": procs is not None,
            "holders": [{**h, "gb": None if h["bytes"] is None else round(h["bytes"] / GB, 1)}
                        for h in hs],
            "line": line(c), "at": time.time(), "why": ""}


def line(c: dict) -> str:
    """"GPU memory: 19.4 of 32 GB in use · 12.6 GB free" """
    total = c["total"] / GB
    return (f"GPU memory: {c['used'] / GB:.1f} of {total:.0f} GB in use · "
            f"{gb_text(c['free'])} free")


def status(max_age: float = TTL_S) -> dict:
    """the card's memory and what holds it, at most a few seconds old"""
    with _lock:
        got = _cache.get("v")
        if got and time.time() - got["at"] < max_age:
            return got
    v = _compute()
    with _lock:
        _cache["v"] = v
    return v


def holder_words(h: dict) -> str:
    """"the judge 13.2 GB", "other: 2.1 GB", "run #181 (…)" """
    if h.get("bytes") is None:
        return h["name"]
    return (f"other: {gb_text(h['bytes'])}" if h["kind"] == "other"
            else f"{h['name']} {gb_text(h['bytes'])}")


def waiting_line(need_bytes: float, free_bytes: float) -> str:
    """a run's words while it waits (16.2): "Waiting for GPU memory: needs
    9.5 GB, 6.2 GB is free." """
    return f"Waiting for GPU memory: needs {gb_text(need_bytes)}, {gb_text(free_bytes)} is free."

