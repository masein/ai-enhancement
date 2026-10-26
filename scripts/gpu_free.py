#!/usr/bin/env python3
"""29: our own boxes' GPUs are colleagues' too (dgx-03's V100s) — whether a
GPU is free of anyone else's work, from nvidia-smi alone. The board only
reads, says and waits: it never stops, signals or touches another process.

A box on a station is one GPU, its container started with
`--gpus device=N --pid=host` by scripts/own_box.sh as the board's account
(no root, no capabilities): nvidia-smi inside sees that one card and the
processes on it with their real PIDs, and /proc says whose they are, as the
account sees them on the station itself.

- Before a step starts: any process on the GPU, or memory in use beyond what
  an idle card shows, and it doesn't start — the GPU named.
- While it runs (service/frontier.PAUSE, before each benchmark and at most
  once a minute before a question): a process that isn't this step's own
  llama-server, or memory in use that no process it can see explains, and the
  step pauses — its answers in flight finish, nothing else starts, its server
  stops — until the GPU is free again (scripts/frontier_box.py waits).

    python scripts/gpu_free.py            # each GPU this box sees, free or not

EVALBOARD_NVIDIA_SMI names another nvidia-smi (the tests' fake one).
Python's standard library only."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time

IDLE_MIB = 256          # memory an idle card may show in use (the driver's own)
UNSEEN_MIB = 1024       # memory in use beyond the processes seen: someone's, unseen
EVERY_S = 60.0          # while asking, the card is read at most this often


def smi() -> list[str]:
    return shlex.split(os.environ.get("EVALBOARD_NVIDIA_SMI") or "nvidia-smi")


def RUN(args: list[str]) -> str:
    got = subprocess.run([*smi(), *args], capture_output=True, text=True, timeout=30)
    if got.returncode != 0:
        raise OSError(f"nvidia-smi exit {got.returncode}: "
                      f"{(got.stderr or got.stdout).strip()[:200]}")
    return got.stdout


def _rows(out: str) -> list[list[str]]:
    return [[c.strip() for c in ln.split(",")] for ln in out.splitlines() if ln.strip()]


def _int(v: str) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def read(run=None) -> list[dict]:
    """each GPU this box sees: {index, uuid, name, used_mib, total_mib, apps:
    [{pid, name, mib}]} — raises when nvidia-smi can't be read"""
    run = run or RUN
    cards = [{"index": _int(r[0]), "uuid": r[1], "name": r[2], "used_mib": _int(r[3]),
              "total_mib": _int(r[4]), "apps": []}
             for r in _rows(run(["--query-gpu=index,uuid,name,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"])) if len(r) >= 5]
    by = {c["uuid"]: c for c in cards}
    for r in _rows(run(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"])):
        if len(r) >= 4 and r[0] in by and _int(r[1]) is not None:
            by[r[0]]["apps"].append({"pid": _int(r[1]), "name": os.path.basename(r[2]),
                                     "mib": _int(r[3])})
    return cards


def owner(pid: int) -> str:
    """whose process it is, as /proc says (a uid with no name here: the uid)"""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            uid = next((ln.split()[1] for ln in fh if ln.startswith("Uid:")), "")
    except OSError:
        return ""
    try:
        import pwd
        return pwd.getpwuid(int(uid)).pw_name
    except (KeyError, ValueError, ImportError):
        return f"uid {uid}" if uid else ""


def _app_words(a: dict) -> str:
    who = owner(a["pid"])
    return (f"pid {a['pid']}" + (f", {who}" if who else "") + (f", {a['name']}" if a["name"] else "")
            + (f", {a['mib']:,} MiB" if a.get("mib") is not None else ""))


def others(card: dict, own: set[int] | frozenset = frozenset(),
           baseline_mib: int | None = None) -> str:
    """'' when nothing but `own` uses this card; else what does, in words.
    Before a step's server starts (`baseline_mib` None), memory over IDLE_MIB
    counts too; once it runs, memory beyond what every process seen uses, by
    UNSEEN_MIB — someone's process this box can't see"""
    theirs = [a for a in card["apps"] if a["pid"] not in own]
    if theirs:
        return "; ".join(_app_words(a) for a in theirs)
    used = card.get("used_mib")
    if used is None:
        return ""
    if baseline_mib is None:
        return (f"{used:,} MiB in use, by no process this box can see" if used > IDLE_MIB
                and not card["apps"] else "")
    seen = sum(a.get("mib") or 0 for a in card["apps"])
    unexplained = used - max(seen, baseline_mib)
    return (f"{unexplained:,} MiB more in use than this step's own, by a process this box can't "
            "see" if unexplained > UNSEEN_MIB else "")


def busy(label: str = "", own: set[int] | frozenset = frozenset(),
         baseline_mib: int | None = None, run=None) -> str:
    """'' when every GPU this box sees is free of anyone else's work; else
    which GPU, and what is on it — the line a refusal or a pause says"""
    try:
        cards = read(run)
    except (OSError, subprocess.SubprocessError) as e:
        return f"its GPU couldn't be read ({e})"
    if not cards:
        return "nvidia-smi lists no GPU here"
    bad = [(c, w) for c in cards if (w := others(c, own, baseline_mib))]
    if not bad:
        return ""
    return "; ".join(f"{gpu_name(label, c)} has another process: {w}" for c, w in bad)


def gpu_name(label: str, card: dict) -> str:
    """"GPU 0 of D3 (Tesla V100-DGXS-32GB)" — a box's label D3.0 is station
    D3's GPU 0, whatever index its container sees it under"""
    st, _, n = label.partition(".")
    idx = n if n.isdigit() else card.get("index")
    return f"GPU {idx}" + (f" of {st}" if st and n.isdigit() else "") + (
        f" ({card['name']})" if card.get("name") else "")


class Gate:
    """service/frontier.PAUSE for a step on our own box: its own server's
    PIDs and the memory the card showed with it up, then '' or why to pause —
    read fresh between benchmarks, at most once a minute otherwise"""

    def __init__(self, label: str, own: set[int], baseline_mib: int | None, run=None,
                 every: float | None = None):
        self.label, self.own, self.baseline, self.run = label, set(own), baseline_mib, run
        self.every = EVERY_S if every is None else every
        self.at, self.last = 0.0, ""

    def __call__(self, fresh: bool = False) -> str:
        now = time.time()
        if fresh or now - self.at >= self.every:
            self.last = busy(self.label, self.own, self.baseline, self.run)
            self.at = now
        return self.last


def used_mib(run=None) -> int | None:
    """the memory in use on the first GPU this box sees"""
    try:
        cards = read(run)
    except (OSError, subprocess.SubprocessError):
        return None
    return cards[0].get("used_mib") if cards else None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    label = args[0] if args else os.environ.get("EVALBOARD_OWN", "")
    try:
        cards = read()
    except (OSError, subprocess.SubprocessError) as e:
        print(f"its GPU couldn't be read ({e})")
        return 2
    code = 0
    for c in cards:
        w = others(c)
        print(f"{gpu_name(label, c)}: " + (f"busy — {w}" if w else "free"))
        code = code or (1 if w else 0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
