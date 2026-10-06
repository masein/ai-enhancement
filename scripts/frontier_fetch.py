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
out = {"bundles": [], "parity": [], "progress": [], "steps": []}
for p in sorted(glob.glob("/workspace/**/frontier-*.tar.gz", recursive=True)):
    out["bundles"].append({"path": p, "sha256": sha(p)})
for p in sorted(glob.glob("/workspace/**/parity.jsonl", recursive=True)):
    out["parity"].append({"path": p, "sha256": sha(p)})
for d in sorted(glob.glob("/workspace/*/*/")):
    parts = d.rstrip("/").split("/")
    if len(parts) == 4 and parts[2] != "files" and "-" in parts[3]:
        out["steps"].append({"build": parts[2], "step": parts[3]})
for p in sorted(glob.glob("/workspace/**/progress.json", recursive=True)):
    try:
        with open(p) as f:
            got = json.load(f)
    except (OSError, ValueError):
        continue
    if isinstance(got, dict):
        out["progress"].append({**got, "dir": os.path.dirname(p)})
print(json.dumps(out, default=str))
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


def _int(v, default: int = 0) -> int:
    """17g: a number from a box's file, whatever it holds"""
    try:
        x = float(v)
        return int(x) if x == x and abs(x) != float("inf") else default
    except (TypeError, ValueError):
        return default


def progress_words(p: dict) -> str:
    """a step's progress in a few words: its label, what it asks, n of N,
    when it started, its restarts, when the box last wrote it — 17g: and why
    it stopped. A field of the wrong kind is said as unknown, never a crash"""
    restarts = max(0, _int(p.get("sessions"), 1) - 1)
    at = _int(p.get("at"))
    ago = max(0, round((time.time() - at) / 60)) if at else None
    tasks = [str(t) for t in p.get("tasks") or [] if isinstance(t, str)] \
        if isinstance(p.get("tasks"), list) else []
    what = ("the parity questions" if p.get("parity") is True else
            ", ".join(tasks) + f" · thinking {p.get('thinking')}")
    state = str(p.get("state") or "?")
    return (f"{p.get('label') or '?'} {Path(str(p.get('dir'))).name} · {what} · {state}"
            + (f" · {p['line']}" if p.get("line") and state == "asking" else "")
            + (f" · {p['why']}" if p.get("why") and state == "stopped" else "")
            + f" · started {p.get('started_at') or '?'}"
            + (f" · {restarts} restart{'s' if restarts != 1 else ''}" if restarts else "")
            + (f" · written {ago} min ago" if ago is not None else ""))


def by_hand(here: Path, by: str) -> str:
    """17g: a bundle of a model with no --sha (G6's calibration): its import, to
    type — its first import registers the model"""
    return (f"sudo docker compose exec -T bench python scripts/import_remote.py {here} --by {by} "
            '--register "<its name>" --file-sha256 <its sha256>')


def one_box(box: re.Match, a: argparse.Namespace, key: str, dest: Path,
            shas: dict, parity: dict, seen: dict, verdicts: dict
            ) -> tuple[bool, bool, list[str], list[dict]]:
    """(done and safe to destroy, failed, its lines, its steps for the board):
    one box listed, copied, imported and compared, its last line saying which.
    17g: done only when every step its plan gives each build started there
    is whole, and its bundle home with the box's sha256 and imported (or, for
    a model with no --sha, home: imported by hand)"""
    import frontier_box as fbx
    name = f"{box['host']}:{box['port']}"
    got, why = listing(box, key)
    if got is None:
        return False, True, [f"{name}: couldn't be asked — {why}. NOT safe to destroy"], []
    progress = [p for p in got.get("progress") or [] if isinstance(p, dict)]
    lines, problems, failed = [], [], False
    # the steps this box started, by build: their folders, and what each wrote
    started: dict[str, dict[str, dict]] = {}
    for d in got.get("steps") or []:
        if isinstance(d, dict) and d.get("build") and d.get("step"):
            started.setdefault(str(d["build"]), {}).setdefault(str(d["step"]), {})
    for p in progress:
        parts = Path(str(p.get("dir"))).parts
        if len(parts) >= 4:
            started.setdefault(parts[-2], {})[parts[-1]] = p
    labels = sorted({k.rsplit("-", 1)[0] for v in started.values() for k in v}
                    | {str(p["label"]) for p in progress if p.get("label")})
    who = f"{name} ({', '.join(labels) or 'no label'})"
    for p in progress:
        lines.append(f"  {progress_words(p)}")
    if not got["bundles"] and not got["parity"] and not started:
        return False, False, [f"{name}: no step has started yet — NOT safe to destroy"], []
    # each bundle: copied, then imported (or, with no --sha, home to import by hand)
    home: dict[str, str] = {}                       # a step's folder -> what became of its bundle
    for b in got["bundles"]:
        step = "/".join(Path(b["path"]).parts[-3:-1])
        ok, words = copy(box, key, b["path"], b["sha256"], dest)
        here = dest / Path(b["path"]).name
        if not ok:
            lines.append(f"  {here.name}: {words}")
            problems.append(f"{here.name} NOT copied")
            home[step] = "not copied"
            failed = True
            continue
        model = model_of(here)
        if model not in shas:
            lines.append(f"  {here.name}: {words} — home; no --sha for {model or 'its model'}: "
                         f"import it by hand: {by_hand(here, a.by)}")
            home[step] = "by hand"
            continue
        if seen.get(here.name) == b["sha256"]:
            home[step] = "imported"
            continue                                # imported on an earlier round of --every
        code, said = run([*IMPORT, str(here), "--by", a.by, "--file-sha256", shas[model]],
                         cwd=REPO, timeout=IMPORT_S)
        lines.append(f"  {here.name}: {words} · {summary(code, said)}")
        if code != 0:
            problems.append(f"{here.name}'s import refused")
            home[step] = "refused"
            failed = True
        else:
            seen[here.name] = b["sha256"]
            home[step] = "imported"
    for f in got["parity"]:
        # /workspace/<build>/<box>-parity/parity.jsonl: the build's own folder
        parts = Path(f["path"]).parts
        build = parts[2] if len(parts) >= 5 else "unknown"
        model = f"served/{build}"
        local = dest.parent / "parity" / f"{build}-box.jsonl"
        ok, words = copy(box, key, f["path"], f["sha256"], local.parent, name=local.name)
        step = "/".join(parts[-3:-1])
        if not ok:
            lines.append(f"  {build}'s parity file: {words}")
            problems.append("its parity file NOT copied")
            home[step] = "not copied"
            failed = True
            continue
        home[step] = "imported"
        if model in parity:
            code, said = run([*COMPARE, parity[model], str(local),
                              *(["--file-sha256", shas[model]] if model in shas else [])],
                             cwd=REPO, timeout=IMPORT_S)
            first = next((x for x in said.splitlines() if x.strip()), f"exit {code}")
            # 17g: kept, said on a line of its own every round, and the exit
            # code says when a build isn't the same or couldn't be compared
            verdicts[build] = (code == 0 and first.startswith("The same"), first)
        else:
            lines.append(f"  {build}'s parity file is at {local} — give --parity "
                         f"{model}=<the server's file> to compare it")
    # every planned step of every build started here: whole, and home
    for build, steps in sorted(started.items()):
        for label in sorted({k.rsplit("-", 1)[0] for k in steps}):
            plan = fbx.planned(label) or sorted(k for k in steps if k.rsplit("-", 1)[0] == label)
            for step in plan:
                p = steps.get(step)
                state = str((p or {}).get("state") or "")
                if p is None:
                    problems.append(f"{build} {step} hasn't started")
                elif state != "whole":
                    problems.append(f"{build} {step} is {state or 'starting'}"
                                    + (f": {p['why']}" if p.get("why") and state == "stopped"
                                       else ""))
                elif home.get(f"{build}/{step}") not in ("imported", "by hand"):
                    problems.append(f"{build} {step}'s file isn't home yet")
    safe = not problems
    lines.insert(0, f"{who}: " + ("done, safe to destroy — every step whole, every bundle here "
                                  "with the box's sha256 and imported" if safe else
                                  "NOT safe to destroy — " + "; ".join(problems)))
    # 17f: each step as the board shows it on Runs — its label and progress,
    # never the box's address
    now = time.time()
    steps = [{**{k: p.get(k) for k in ("label", "model", "thinking", "tasks", "shard", "parity",
                                        "state", "line", "started_at", "sessions", "at", "why")
                 if p.get(k) is not None},
              "step": Path(str(p.get("dir"))).name, "seen_at": now, "reachable": True,
              "safe": safe} for p in progress]
    return safe, failed, lines, steps


def post_boxes(steps: list[dict], dest: Path) -> str:
    """17f: the boxes' steps to the board, for Runs' "On rented boxes" list"""
    path = dest.parent / "boxes.json"
    path.write_text(json.dumps(steps), encoding="utf-8")
    code, said = run([*IMPORT, "--boxes", str(path)], cwd=REPO, timeout=IMPORT_S)
    return ("the board's list of rented boxes: " + last(said, "updated") if code == 0 else
            "the board's list of rented boxes NOT updated — " + last(said, f"exit {code}"))


def keep_sudo(every: float = 120.0) -> None:
    """17g: sudo's password asked once, at the start, and its timestamp kept
    alive — unattended, --every 15m met sudo's 15 minutes and each import
    waited for a password, then read "refused". When it can't be kept, one
    line says the imports wait for the password"""
    import shutil
    import threading
    if not shutil.which("sudo"):
        return
    # with no terminal to type it in (nohup), never a prompt: -n
    asks = ["sudo", "-v"] if sys.stdin and sys.stdin.isatty() else ["sudo", "-n", "-v"]
    if subprocess.run(asks, capture_output=asks[1] == "-n").returncode != 0:
        print("sudo's password wasn't given: each import will ask for it", flush=True)
        return

    def refresh() -> None:
        said = False
        while True:
            time.sleep(every)
            ok = subprocess.run(["sudo", "-n", "-v"], capture_output=True).returncode == 0
            if not ok and not said:
                print("sudo's password has lapsed: the next import waits for it here",
                      flush=True)
            said = not ok
    threading.Thread(target=refresh, daemon=True).start()


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
                    help="served/<name>=<the server's parity file> (G1b's, e.g. /home/masein/"
                         "benchmarks/parity/phone-server-500.jsonl): compare the box's with it")
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
        # 17g: checked now, as --key is — a wrong name was a traceback at the
        # first compare
        if not Path(m["path"]).expanduser().is_file():
            ap.error(f"--parity {s}: no such file ({m['path']})")
    key = str(Path(a.key).expanduser())
    if not Path(key).exists():
        ap.error(f"--key {a.key}: no such file")
    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest.parent / "parity").mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    known: dict[str, list[dict]] = {}
    verdicts: dict[str, tuple[bool, str]] = {}
    keep_sudo()
    while True:
        failed_any, done = False, []
        for box in boxes:
            safe, failed, lines, steps = one_box(box, a, key, dest, shas, parity, seen,
                                                 verdicts)
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
        # 17g: each build's parity verdict, a line of its own every round; one
        # that isn't the same, or couldn't be compared, is a failure
        for build, (same, first) in sorted(verdicts.items()):
            print(f"{build}'s parity: {first}", flush=True)
            failed_any |= not same
        if not a.every or all(done):
            return 1 if failed_any else 0
        print(f"— again in {round(a.every / 60)} min ({sum(done)} of {len(done)} boxes done)",
              flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
