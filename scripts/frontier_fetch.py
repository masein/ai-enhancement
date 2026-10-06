#!/usr/bin/env python3
"""17e: every box's bundles fetched and imported with one command, on the
server (as you, in ~/benchmarks/aienh) — G3 and G4's scp and import, a box at
a time, without typing either.

    python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 \\
        --sha served/<phone-build>=<its sha256> \\
        --sha served/<original-build>=<its sha256> \\
        --parity served/<phone-build>=/home/masein/benchmarks/parity/phone-server.jsonl \\
        <host>:<port> <host>:<port> …

Each box (root@ unless another user is given) is asked over SSH, once, for
what it holds under /workspace — every bundle and parity file with its
sha256, and each step's progress (17f) — and nothing else.

17f:
- a bundle is copied to a temporary name and counts as copied only when its
  sha256 is the box's; a failed copy says NOT copied and leaves an older one
  here as it was;
- each box's bundles are imported as soon as they are here, before the next
  box is asked; SSH gives up on a box that doesn't answer (a connect timeout,
  batch mode, keep-alives, a timeout on each copy);
- the bundles of either build, in its own folder (/workspace/<build>/…);
- a box's parity file, with --parity for its build, is fetched and compared
  with the server's (scripts/frontier_parity.py compare), the verdict
  printed — an import never waits for it;
- each box ends with one line: done and safe to destroy (every step whole,
  every bundle it holds here with the same sha256 and imported), or not, and
  why. The exit code is not 0 when a box couldn't be reached, a copy failed
  or an import was refused;
- --every 15m does it again every 15 minutes until every box is done.

Imported already is told by the answers (scripts/import_frontier.py): a
bundle made again changes nothing. A bundle of a model with no --sha (G6's
calibration, its first import registering it) is fetched and left for you
to import by hand. A box's address is printed here only, never kept.

Python's standard library only: it runs on the server, outside the image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMPORT = ["sudo", "docker", "compose", "exec", "-T", "bench", "python",
          "scripts/import_remote.py"]
COMPARE = ["sudo", "docker", "compose", "exec", "-T", "bench", "python",
           "scripts/frontier_parity.py", "compare"]
_BOX = re.compile(r"(?:(?P<user>[A-Za-z0-9._-]+)@)?(?P<host>[A-Za-z0-9.-]+):(?P<port>\d+)")
_SHA = re.compile(r"(?P<model>served/[A-Za-z0-9._-]+)=(?P<sha>[0-9a-fA-F]{64})")
_PARITY = re.compile(r"(?P<model>served/[A-Za-z0-9._-]+)=(?P<path>\S+)")
LIST_S, COPY_S, IMPORT_S = 120, 1800, 900

# what a box is asked, run there by its own Python: every bundle and parity
# file under /workspace with its sha256, and each step's progress
LISTING = r'''
import glob, hashlib, json, os
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out = {"bundles": [], "parity": [], "progress": []}
for p in sorted(glob.glob("/workspace/**/frontier-*.tar.gz", recursive=True)):
    out["bundles"].append({"path": p, "sha256": sha(p)})
for p in sorted(glob.glob("/workspace/**/parity.jsonl", recursive=True)):
    out["parity"].append({"path": p, "sha256": sha(p)})
for p in sorted(glob.glob("/workspace/**/progress.json", recursive=True)):
    try:
        with open(p) as f:
            out["progress"].append({"dir": os.path.dirname(p), **json.load(f)})
    except (OSError, ValueError):
        pass
print(json.dumps(out))
'''


def run(cmd: list[str], cwd: Path | None = None, timeout: float | None = None,
        stdin: str | None = None) -> tuple[int, str]:
    """(exit code, what it printed) — its output kept, not shown; 124 when
    it ran out of time"""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           input=stdin)
    except subprocess.TimeoutExpired:
        return 124, f"no answer within {int(timeout or 0)} s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ssh_opts(key: str, port: str, scp: bool = False) -> list[str]:
    # a rented box is a new host each time: its key is taken on first contact,
    # as answering yes would. 17f: never a prompt, and a box that stops
    # answering is given up on
    return ["-i", key, "-P" if scp else "-p", port, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"]


def last(said: str, fallback: str) -> str:
    lines = [x.strip() for x in said.splitlines() if x.strip()]
    return lines[-1] if lines else fallback


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listing(box: re.Match, key: str) -> tuple[dict | None, str]:
    """(what the box holds, '') — or (None, why it couldn't be asked)"""
    who = f"{box['user'] or 'root'}@{box['host']}"
    code, said = run(["ssh", *ssh_opts(key, box["port"]), who,
                      "command -v python3 >/dev/null && exec python3 - || exec python -"],
                     timeout=LIST_S, stdin=LISTING)
    if code != 0:
        return None, last(said, f"ssh exit {code}")
    try:
        return json.loads(said.strip().splitlines()[-1]), ""
    except (ValueError, IndexError):
        return None, f"its listing couldn't be read: {last(said, 'empty')[:120]}"


def copy(box: re.Match, key: str, remote: str, want: str, dest: Path,
         name: str = "") -> tuple[bool, str]:
    """(copied, words): to a temporary name, kept only when its sha256 is the
    box's — a copy that failed leaves what was here as it was"""
    who = f"{box['user'] or 'root'}@{box['host']}"
    here = dest / (name or Path(remote).name)
    if here.exists() and sha256(here) == want:
        return True, "here already"
    part = here.with_name(here.name + ".part")
    code, said = run(["scp", *ssh_opts(key, box["port"], scp=True), f"{who}:{remote}",
                      str(part)], timeout=COPY_S)
    if code != 0 or not part.exists() or sha256(part) != want:
        part.unlink(missing_ok=True)
        return False, ("NOT copied — " + (last(said, f"scp exit {code}") if code != 0 else
                                           "its sha256 isn't the box's"))
    os.replace(part, here)
    return True, "copied"


def model_of(path: Path) -> str:
    """the served model a bundle's bundle.json names — '' when it can't be read"""
    try:
        with tarfile.open(path, "r:gz") as tar:
            fh = tar.extractfile("bundle.json")
            return str(json.load(fh).get("model") or "") if fh else ""
    except (OSError, tarfile.TarError, ValueError, KeyError):
        return ""


def summary(code: int, said: str) -> str:
    """one line of import_remote.py's: its row's line, or why it refused"""
    lines = [x.strip() for x in said.splitlines() if x.strip()]
    if code == 0:
        return next((x for x in lines if x.startswith(("the row ", "imported already"))),
                    lines[-1] if lines else "imported")
    refused = [x for x in lines if x.startswith("refused")]
    return "; ".join(refused) if refused else (lines[-1] if lines else f"exit {code}")


def progress_words(p: dict) -> str:
    """a step's progress in a few words: its label, what it asks, n of N,
    when it started, its restarts, when the box last wrote it"""
    restarts = max(0, int(p.get("sessions") or 1) - 1)
    ago = max(0, round((time.time() - float(p.get("at") or 0)) / 60))
    what = ("the parity questions" if p.get("parity") else
            ", ".join(p.get("tasks") or []) + f" · thinking {p.get('thinking')}")
    return (f"{p.get('label') or '?'} {Path(str(p.get('dir'))).name} · {what} · "
            f"{p.get('state') or '?'}" + (f" · {p['line']}" if p.get("line") and
                                         p.get("state") == "asking" else "")
            + f" · started {p.get('started_at') or '?'}"
            + (f" · {restarts} restart{'s' if restarts != 1 else ''}" if restarts else "")
            + f" · written {ago} min ago")


def one_box(box: re.Match, a: argparse.Namespace, key: str, dest: Path,
            shas: dict, parity: dict, seen: dict) -> tuple[bool, bool, list[str], list[dict]]:
    """(done and safe to destroy, failed, its lines, its steps for the board):
    one box listed, copied, imported and compared, its last line saying which"""
    name = f"{box['host']}:{box['port']}"
    got, why = listing(box, key)
    if got is None:
        return False, True, [f"{name}: couldn't be asked — {why}. NOT safe to destroy"], []
    lines, problems, failed = [], [], False
    labels = sorted({str(p.get("label")) for p in got["progress"] if p.get("label")})
    who = f"{name} ({', '.join(labels) or 'no label'})"
    for p in got["progress"]:
        lines.append(f"  {progress_words(p)}")
        if p.get("state") != "whole":
            problems.append(f"{Path(str(p.get('dir'))).name} is {p.get('state') or 'not whole'}")
    if not got["bundles"] and not got["parity"] and not got["progress"]:
        return False, False, [f"{name}: no bundle there yet — NOT safe to destroy"], []
    for b in got["bundles"]:
        ok, words = copy(box, key, b["path"], b["sha256"], dest)
        here = dest / Path(b["path"]).name
        if not ok:
            lines.append(f"  {here.name}: {words}")
            problems.append(f"{here.name} NOT copied")
            failed = True
            continue
        model = model_of(here)
        if model not in shas:
            lines.append(f"  {here.name}: {words}; no --sha for {model or 'its model'} — import "
                         "it by hand (G4, G6)")
            problems.append(f"{here.name} not imported")
            continue
        if seen.get(here.name) == b["sha256"]:
            continue                                # imported on an earlier round of --every
        code, said = run([*IMPORT, str(here), "--by", a.by, "--file-sha256", shas[model]],
                         cwd=REPO, timeout=IMPORT_S)
        lines.append(f"  {here.name}: {words} · {summary(code, said)}")
        if code != 0:
            problems.append(f"{here.name}'s import refused")
            failed = True
        else:
            seen[here.name] = b["sha256"]
    for f in got["parity"]:
        # /workspace/<build>/<box>-parity/parity.jsonl: the build's own folder
        parts = Path(f["path"]).parts
        build = parts[2] if len(parts) >= 5 else "unknown"
        model = f"served/{build}"
        local = dest.parent / "parity" / f"{build}-box.jsonl"
        ok, words = copy(box, key, f["path"], f["sha256"], local.parent, name=local.name)
        if not ok:
            lines.append(f"  {build}'s parity file: {words}")
            problems.append("its parity file NOT copied")
            failed = True
        elif model in parity:
            code, said = run([*COMPARE, parity[model], str(local),
                              *(["--file-sha256", shas[model]] if model in shas else [])],
                             cwd=REPO, timeout=IMPORT_S)
            first = next((x for x in said.splitlines() if x.strip()), f"exit {code}")
            lines.append(f"  {build}'s parity: {first}")
        else:
            lines.append(f"  {build}'s parity file is at {local} — give --parity "
                         f"{model}=<the server's file> to compare it")
    safe = not problems
    lines.insert(0, f"{who}: " + ("done, safe to destroy — every step whole, every bundle here "
                                  "with the box's sha256 and imported" if safe else
                                  "NOT safe to destroy — " + "; ".join(problems)))
    # 17f: each step as the board shows it on Runs — its label and progress,
    # never the box's address
    now = time.time()
    steps = [{**{k: p.get(k) for k in ("label", "model", "thinking", "tasks", "shard", "parity",
                                        "state", "line", "started_at", "sessions", "at")
                 if p.get(k) is not None},
              "step": Path(str(p.get("dir"))).name, "seen_at": now, "reachable": True,
              "safe": safe} for p in got["progress"]]
    return safe, failed, lines, steps


def post_boxes(steps: list[dict], dest: Path) -> str:
    """17f: the boxes' steps to the board, for Runs' "On rented boxes" list"""
    path = dest.parent / "boxes.json"
    path.write_text(json.dumps(steps), encoding="utf-8")
    code, said = run([*IMPORT, "--boxes", str(path)], cwd=REPO, timeout=IMPORT_S)
    return ("the board's list of rented boxes: " + last(said, "updated") if code == 0 else
            "the board's list of rented boxes NOT updated — " + last(said, f"exit {code}"))


def every_s(text: str) -> float:
    m = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", text or "")
    if not m:
        raise argparse.ArgumentTypeError(f"{text!r}: give 15m, 900s or 1h")
    return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", nargs="+", help="each box as <host>:<port> (root@ unless "
                                              "<user>@ is given), as vast.ai's SSH line says")
    ap.add_argument("--key", required=True, help="your SSH key, e.g. ~/.ssh/id_ed25519")
    ap.add_argument("--sha", action="append", default=[],
                    help="served/<name>=<sha256>: a build's file, as G0 gave it (one each)")
    ap.add_argument("--parity", action="append", default=[],
                    help="served/<name>=<the server's parity file>: compare the box's with it")
    ap.add_argument("--dest", default="~/benchmarks/bundles", help="where the bundles go")
    ap.add_argument("--by", default="masein", help="your name, for the Runs list")
    ap.add_argument("--no-board", dest="board", action="store_false",
                    help="17f: don't send the boxes' progress to the board (Runs' list)")
    ap.add_argument("--every", type=every_s, default=0,
                    help="17f: do it again every 15m (or 900s, 1h) until every box is done")
    a = ap.parse_args(argv)
    boxes = []
    for b in a.boxes:
        m = _BOX.fullmatch(b.strip())
        if not m:
            ap.error(f"{b!r}: give a box as <host>:<port>, e.g. 203.0.113.7:41022")
        boxes.append(m)
    shas = {}
    for s in a.sha:
        m = _SHA.fullmatch(s.strip())
        if not m:
            ap.error(f"--sha {s!r}: served/<name>=<the file's 64-character sha256>")
        shas[m["model"]] = m["sha"].lower()
    parity = {}
    for s in a.parity:
        m = _PARITY.fullmatch(s.strip())
        if not m:
            ap.error(f"--parity {s!r}: served/<name>=<the server's parity file>")
        parity[m["model"]] = m["path"]
    key = str(Path(a.key).expanduser())
    if not Path(key).exists():
        ap.error(f"--key {a.key}: no such file")
    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest.parent / "parity").mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    known: dict[str, list[dict]] = {}
    while True:
        failed_any, done = False, []
        for box in boxes:
            safe, failed, lines, steps = one_box(box, a, key, dest, shas, parity, seen)
            print("\n".join(lines), flush=True)
            failed_any |= failed
            done.append(safe)
            name = f"{box['host']}:{box['port']}"
            if steps:
                known[name] = steps
            elif failed and name in known:
                known[name] = [{**x, "reachable": False} for x in known[name]]
        if a.board and known:
            print(post_boxes([x for v in known.values() for x in v], dest), flush=True)
        if not a.every or all(done):
            return 1 if failed_any else 0
        print(f"— again in {round(a.every / 60)} min ({sum(done)} of {len(done)} boxes done)",
              flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
