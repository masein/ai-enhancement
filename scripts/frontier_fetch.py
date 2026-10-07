#!/usr/bin/env python3
"""17e: every box's bundles fetched and imported with one command, on the
server (as you, in ~/benchmarks/aienh) — G3 and G4's scp and import, a box at
a time, without typing either.

    python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 \\
        --sha served/<phone-build>=<its sha256> \\
        --sha served/<original-build>=<its sha256> \\
        --parity served/<phone-build>=/home/masein/benchmarks/parity/phone-server-500.jsonl \\
        --parity served/<original-build>=/home/masein/benchmarks/parity/orig-server-500.jsonl \\
        --every 3m <host>:<port> <host>:<port> …

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
- --every 3m does it again every 3 minutes until every box is done.

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
# 17j: an import refused for an off run's thinking share — the board's words,
# before 17j (more than 1%) and since (more than a quarter)
THINK_REFUSED = re.compile(r"thinking was off, and [\d,]+ of its [\d,]+ answers hold thinking")
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
    it ran out of time. 17h: what it printed is its stdout, and its stderr
    only when it failed — a box's banner ("Welcome to vast.ai … Have fun!")
    and ssh's "Permanently added" come on stderr, and were read as the
    listing"""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           input=stdin)
    except subprocess.TimeoutExpired:
        return 124, f"no answer within {int(timeout or 0)} s"
    return p.returncode, (p.stdout or "") + ((p.stderr or "") if p.returncode else "")


def ssh_opts(key: str, port: str, scp: bool = False) -> list[str]:
    # a rented box is a new host each time: its key is taken on first contact,
    # as answering yes would. 17f: never a prompt, and a box that stops
    # answering is given up on
    # 17h: and nothing but errors on stderr — no banner, no known-hosts line
    return ["-i", key, "-P" if scp else "-p", port, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4", "-o", "LogLevel=ERROR"]


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
        got = json.loads(said.strip().splitlines()[-1])
        if not isinstance(got, dict):
            raise ValueError("not a listing")
        return got, ""
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


def gguf_of(path: Path) -> dict:
    """17i: the GGUF a bundle's setup.json names — {name, sha256, parts: [{name,
    sha256}]}: for a GGUF in parts, its sha256 is the split identity (the
    sha256 of its parts' names and sha256s), what the board registers. {}
    when it can't be read"""
    try:
        with tarfile.open(path, "r:gz") as tar:
            fh = tar.extractfile("setup.json")
            g = (json.load(fh).get("gguf") if fh else None) or {}
    except (OSError, tarfile.TarError, ValueError, KeyError, AttributeError):
        return {}
    if not isinstance(g, dict):
        return {}
    ok = re.compile(r"[0-9a-f]{64}")
    parts = [{"name": str(x.get("name") or "")[:200], "sha256": str(x.get("sha256") or "")}
             for x in g.get("parts") or [] if isinstance(x, dict)
             and ok.fullmatch(str(x.get("sha256") or ""))] \
        if isinstance(g.get("parts"), list) else []
    sha = str(g.get("sha256") or "")
    return {"name": str(g.get("name") or "")[:200], "sha256": sha if ok.fullmatch(sha) else "",
            **({"parts": parts} if parts else {})}


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


def by_hand(here: Path, by: str, registered: bool | None, model: str = "") -> str:
    """17g: a bundle of a model with no --sha: its import, to type. 17h: with
    --register only for a model the board doesn't serve yet. 17i: with the
    file's sha256 read from the bundle — for a GGUF in parts its split
    identity, each part's name and sha256 under it to check against Hugging
    Face — and the --sha that lets the fetch import it by itself"""
    g = gguf_of(here)
    sha = g.get("sha256") or "<its sha256>"
    reg = ' --register "<its name>"' if registered is False else ""
    out = (f"sudo docker compose exec -T bench python scripts/import_remote.py {here} --by {by}"
           f" --file-sha256 {sha}" + reg)
    if registered is None:
        # 17j: a line that pastes as it is — the bracket after it didn't
        out += ('\n      the board couldn\'t be asked whether it serves '
                f'{model or "the model"}: add --register "<its name>" if it doesn\'t yet')
    if g.get("parts"):
        out += (f"\n      {g.get('sha256') or '?'} is the split identity of its "
                f"{len(g['parts'])} parts, as the box hashed them — check each against Hugging "
                "Face before you import:"
                + "".join(f"\n        {x['name']}  {x['sha256']}" for x in g["parts"]))
    elif g.get("sha256"):
        out += f"\n      {g['name'] or 'its file'}'s sha256, as the box hashed it"
    if g.get("sha256") and model:
        out += (f"\n      once checked, give the fetch --sha {model}={g['sha256']} and it "
                "imports this build by itself")
    return out


QUIET_S = 45 * 60                   # a step asking that hasn't written for this long says so


def box_id(name: str, dest: Path) -> str:
    """17h: a box's name on the board — never its address: a keyed hash of it,
    the key kept beside the bundles. The board keeps the steps of the boxes a
    fetch didn't ask as they were (fetching one box marked every other "not
    reached")"""
    import hmac
    import secrets
    f = dest.parent / ".box-key"
    if not f.exists():
        f.write_text(secrets.token_hex(16), encoding="utf-8")
        f.chmod(0o600)
    return hmac.new(f.read_text(encoding="utf-8").strip().encode(), name.encode(),
                    hashlib.sha256).hexdigest()[:16]


def one_box(box: re.Match, a: argparse.Namespace, key: str, dest: Path,
            shas: dict, parity: dict, seen: dict, verdicts: dict
            ) -> tuple[bool, bool, list[str], list[dict], bool]:
    """(done and safe to destroy, failed, its lines, its steps for the board):
    one box listed, copied, imported and compared, its last line saying which.
    17g: done only when every step its plan gives each build started there
    is whole, and its bundle home with the box's sha256 and imported (or, for
    a model with no --sha, home: imported by hand). 17h: and every other step
    on the box too, planned or not, wherever its folder (--out /workspace/run,
    G6's /workspace/gemma-cal, a step started by hand). 17i: the last, whether
    the box was reached; a step given as --abandoned is left out, and said"""
    import frontier_box as fbx
    name = f"{box['host']}:{box['port']}"
    got, why = listing(box, key)
    if got is None:
        return False, True, [f"{name}: couldn't be asked — {why}. NOT safe to destroy"], [], False
    gone = set(getattr(a, "abandoned", None) or [])
    progress = [p for p in got.get("progress") or [] if isinstance(p, dict)]
    lines, problems, failed = [], [], False
    # every step this box started — a folder holding a progress file, a
    # bundle or a parity file, or one the plan's layout made — by its folder
    started: dict[str, dict | None] = {}
    for d in got.get("steps") or []:
        if isinstance(d, dict) and d.get("build") and d.get("step"):
            started.setdefault(f"/workspace/{d['build']}/{d['step']}", None)
    for f in [*(got.get("bundles") or []), *(got.get("parity") or [])]:
        started.setdefault(str(Path(f["path"]).parent), None)
    for p in progress:
        started[str(p.get("dir"))] = p

    def build_of(d: str) -> str:
        parts = Path(d).parts
        return parts[-2] if len(parts) >= 4 else ""

    def of(d: str) -> str:
        return f"{build_of(d)}/{Path(d).name}"
    # 17i: a step given up on (--abandoned <build>/<step>) — the BF16 box's
    # A3-2, stopped by hand: never safe while it stood, and deleted, A3's plan
    # said it hadn't started. Left out of the steps and of the plan, and said
    dropped = sorted(of(d) for d in started if of(d) in gone)
    # 17j: which values matched a step (one that matched nothing is named
    # after the round), and an abandoned step still writing keeps its box
    # not safe: given for a step still asking, it hid the step, read the box
    # safe, and its bundle was never fetched
    matched = getattr(a, "abandoned_matched", None)
    if matched is not None:
        matched.update(dropped)
    now0 = time.time()
    writing = sorted(
        (of(d), str(p.get("state")), max(0, round((now0 - _int(p.get("at"))) / 60)))
        for d, p in started.items() if of(d) in gone and isinstance(p, dict)
        and p.get("state") in ("asking", "starting") and _int(p.get("at"))
        and now0 - _int(p.get("at")) <= QUIET_S)
    started = {d: p for d, p in started.items() if of(d) not in gone}
    labels = sorted({Path(d).name.rsplit("-", 1)[0] for d in started
                     if fbx.planned(Path(d).name.rsplit("-", 1)[0])}
                    | {str(p["label"]) for p in progress if p.get("label")})
    who = f"{name} ({', '.join(labels) or 'no label'})"
    for p in progress:
        if of(str(p.get("dir"))) not in gone or of(str(p.get("dir"))) in {w[0] for w in writing}:
            lines.append(f"  {progress_words(p)}")
    for x in dropped:
        if x not in {w[0] for w in writing}:
            lines.append(f"  {x}: abandoned (--abandoned) — left out of this box's steps")
    for x, state, ago in writing:
        problems.append(f"{x} is given as --abandoned but is still {state} (written {ago} min "
                        "ago): stop it on the box, or leave it out of --abandoned")
    if not started and not writing:
        # 17j: its abandoned steps said to the board all the same
        return False, False, [f"{name}: no step has started yet — NOT safe to destroy"], [
            {**{k: p.get(k) for k in ("model", "thinking", "tasks", "shard", "parity",
                                      "started_at") if p.get(k) is not None},
             "state": "abandoned", "line": "", "label": str(p.get("label") or Path(str(
                 p.get("dir"))).name), "step": Path(str(p.get("dir"))).name,
             "seen_at": time.time(), "reachable": True, "safe": False,
             "box_id": box_id(name, dest)}
            for p in progress if of(str(p.get("dir"))) in gone], True
    # each bundle: copied, then imported (or, with no --sha, home to import by hand)
    home: dict[str, str] = {}                       # a step's folder -> what became of its bundle
    for b in got["bundles"]:
        step = str(Path(b["path"]).parent)
        if of(step) in gone:
            continue
        ok, words = copy(box, key, b["path"], b["sha256"], dest)
        here = dest / Path(b["path"]).name
        if not ok:
            lines.append(f"  {here.name}: {words}")
            problems.append(f"{here.name} NOT copied")
            home[step] = "not copied"
            failed = True
            continue
        model = model_of(here)
        if model in shas and serves(model) is False:
            # 17j: --sha for a model the board doesn't serve yet: every round's
            # import was refused ("add it under Add a model"), and the line to
            # type stopped being printed — it is, with --register
            lines.append(f"  {here.name}: {words} — home; the board doesn't serve {model} "
                         "yet: import it once by hand, with --register, and --sha imports the "
                         "rest by itself: " + by_hand(here, a.by, False, model))
            home[step] = "by hand"
            continue
        if model not in shas:
            lines.append(f"  {here.name}: {words} — home; no --sha for {model or 'its model'}: "
                         "import it by hand: " + by_hand(here, a.by, serves(model), model))
            home[step] = "by hand"
            continue
        if seen.get(here.name) == b["sha256"]:
            home[step] = "imported"
            continue                                # imported on an earlier round of --every
        code, said = run([*IMPORT, str(here), "--by", a.by, "--file-sha256", shas[model]],
                         cwd=REPO, timeout=IMPORT_S)
        lines.append(f"  {here.name}: {words} · {summary(code, said)}")
        if code != 0 and THINK_REFUSED.search(said or ""):
            # 17j: refused for its thinking share — the file is home and the
            # box has no other to give: the box can go, said. The board tries
            # it again each round, and it imports by hand once the board
            # takes it
            lines.append("    kept here, refused for its thinking share: the box isn't needed "
                         "for it — it is tried again each round, or import it by hand: "
                         + by_hand(here, a.by, True, model))
            home[step] = "kept (thinking)"
        elif code != 0:
            problems.append(f"{here.name}'s import refused")
            home[step] = "refused"
            failed = True
        else:
            seen[here.name] = b["sha256"]
            home[step] = "imported"
    bid = box_id(name, dest)
    for f in got["parity"]:
        # /workspace/<build>/<box>-parity/parity.jsonl: the build's own folder
        parts = Path(f["path"]).parts
        build = parts[2] if len(parts) >= 5 else "unknown"
        model = f"served/{build}"
        local = dest.parent / "parity" / f"{build}-box.jsonl"
        step = str(Path(f["path"]).parent)
        if of(step) in gone:
            continue
        ok, words = copy(box, key, f["path"], f["sha256"], local.parent, name=local.name)
        if not ok:
            lines.append(f"  {build}'s parity file: {words}")
            problems.append("its parity file NOT copied")
            home[step] = "not copied"
            failed = True
            continue
        home[step] = "parity"
        was = verdicts.get(build) or {}
        if was.get("gone"):
            was = {}
        if was and was.get("file_sha256") != f["sha256"]:
            # 17i: a new parity file replaces the verdict of the old one
            verdicts[build] = {"gone": True, "at": time.time()}
            was = {}
        if model in parity:
            if was.get("box") == bid and was.get("server") == parity[model]:
                continue                            # compared on an earlier round
            code, said = run([*COMPARE, parity[model], str(local),
                              *(["--file-sha256", shas[model]] if model in shas else [])],
                             cwd=REPO, timeout=IMPORT_S)
            first = next((x for x in said.splitlines() if x.strip()), f"exit {code}")
            # 17g: kept, said on a line of its own every round, and the exit
            # code says when a build isn't the same or couldn't be compared.
            # 17i: with the box, the file's sha256 and when
            verdicts[build] = {"same": code == 0 and first.startswith("The same"),
                               "first": first, "box": bid, "file_sha256": f["sha256"],
                               "server": parity[model], "at": time.time()}
        else:
            lines.append(f"  {build}'s parity file is at {local} — give --parity "
                         f"{model}=<the server's file> to compare it")
    # every planned step of every build started here has its folder —
    # 17i: but one abandoned
    for build in sorted({build_of(d) for d in started if build_of(d)}):
        for label in sorted({Path(d).name.rsplit("-", 1)[0] for d in started
                             if build_of(d) == build}):
            for step in fbx.planned(label):
                if f"/workspace/{build}/{step}" not in started and f"{build}/{step}" not in gone:
                    problems.append(f"{build} {step} hasn't started")
    # and every step on the box, planned or not: whole, and its file home
    now = time.time()
    for d, p in sorted(started.items()):
        what = f"{build_of(d)} {Path(d).name}".strip()
        state = str((p or {}).get("state") or "")
        if p is None and home.get(d) in ("imported", "by hand", "parity", "kept (thinking)"):
            continue                                # a bundle with no progress file, home
        if state != "whole":
            at = _int((p or {}).get("at"))
            quiet = round((now - at) / 60) if at and state in ("asking", "starting") \
                and now - at > QUIET_S else 0
            problems.append(f"{what} is {state or 'starting'}"
                            + (f": {p['why']}" if p and p.get("why") and state == "stopped"
                               else "")
                            + (f", but hasn't written for {quiet:,} min — stopped? paste its "
                               "line again" if quiet else ""))
        elif home.get(d) not in ("imported", "by hand", "parity", "kept (thinking)"):
            problems.append(f"{what}'s file isn't home yet")
    safe = not problems
    by_h = sum(1 for v in home.values() if v == "by hand")
    # 17i: bundles alone — "1 imported" counted the parity file
    n_in = sum(1 for v in home.values() if v == "imported")
    # 17j: and those refused for their thinking share, kept here
    n_th = sum(1 for v in home.values() if v == "kept (thinking)")
    rest = [f"{n_in} imported", *([f"{by_h} to import by hand"] if by_h else []),
            *([f"{n_th} refused for its thinking share and kept here"] if n_th else [])]
    lines.insert(0, f"{who}: " + (
        ("done, safe to destroy — every step whole, every bundle here with the box's sha256 "
         + ("and imported" if not by_h and not n_th else
            f"({', '.join(rest)}: the lines below)"))
        if safe else "NOT safe to destroy — " + "; ".join(problems))
        + (f" · abandoned: {', '.join(dropped)}" if dropped else ""))
    # 17f: each step as the board shows it on Runs — its label and progress,
    # never the box's address. 17h: a step with no label (G6's) by its folder
    # 17j: an abandoned step too, as abandoned — the board says so on its
    # box's line and lists it no more (it read "No contact" for a day)
    steps = [{**{k: p.get(k) for k in ("model", "thinking", "tasks", "shard", "parity",
                                        "state", "line", "started_at", "sessions", "at", "why")
                 if p.get(k) is not None},
              **({"state": "abandoned", "line": ""} if of(str(p.get("dir"))) in gone else {}),
              "label": str(p.get("label") or Path(str(p.get("dir"))).name),
              "step": Path(str(p.get("dir"))).name, "seen_at": now, "reachable": True,
              "safe": safe, "box_id": bid} for p in progress]
    return safe, failed, lines, steps, True


def post_boxes(steps: list[dict], dest: Path, asked: list[str] | None = None,
               reached: list[str] | None = None, every: float = 0) -> str:
    """17f: the boxes' steps to the board, for Runs' "On rented boxes" list —
    17h: with the boxes this fetch asked, by their board names. 17i: and those
    it reached (a build's steps gone from a box reached are gone), as a list
    whose first element says so: a board before 17i reads the steps and
    passes it by (17h's {"steps", "asked"} marked every row "not reached" on
    the board then deployed)"""
    path = dest.parent / "boxes.json"
    # 17i: and how often it reads them, so the board says when the next is due
    head = [{"asked": sorted(asked or []), "reached": sorted(reached or []),
             **({"every": every} if every else {})}] if asked is not None else []
    path.write_text(json.dumps([*head, *steps]), encoding="utf-8")
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


_SERVED: dict = {}


def serves(model: str) -> bool | None:
    """17h: whether the board serves `model` — asked once, when a by-hand line
    needs it (it says --register only for one the board doesn't). None when
    the board can't be asked"""
    if "ids" not in _SERVED:
        code, said = run([*IMPORT, "--served"], cwd=REPO, timeout=IMPORT_S)
        try:
            got = json.loads(said.strip().splitlines()[-1]) if code == 0 else None
        except (ValueError, IndexError):
            got = None
        _SERVED["ids"] = set(got) if isinstance(got, list) else None
    return None if _SERVED["ids"] is None else model in _SERVED["ids"]


def read_verdicts(f: Path) -> dict:
    """17i: the parity verdicts kept — {build: {same, first, box, file_sha256,
    server, at}}; 17h's [same, first] read as one with no file (replaced by
    the next file fetched)"""
    try:
        got = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in (got.items() if isinstance(got, dict) else []):
        if isinstance(v, list) and len(v) == 2:
            v = {"same": v[0], "first": v[1]}
        if isinstance(v, dict) and isinstance(k, str) and v.get("gone"):
            out[k] = {"gone": True, "at": float(v.get("at") or 0)}   # 17j: replaced, merged
            continue
        if isinstance(v, dict) and isinstance(k, str):
            out[k] = {"same": bool(v.get("same")), "first": str(v.get("first") or "")[:300],
                      **{x: v[x] for x in ("box", "file_sha256", "server") if isinstance(
                          v.get(x), str)},
                      **({"at": v["at"]} if isinstance(v.get("at"), (int, float)) else {})}
    return out


SAFE_KEEP_S = 2 * 86400              # 17j: a box's "safe" forgotten after this long
UNREACHED_ROUNDS = 2                 # 17j: rounds a safe box goes unreached before it's destroyed


def read_safe(f: Path) -> dict:
    """17j: the boxes that read safe — {board name: {at, missed}} (17i's list
    read as safe when the file was written), and {gone, at} for one forgotten"""
    try:
        got = json.loads(f.read_text(encoding="utf-8"))
        mtime = f.stat().st_mtime
    except (OSError, ValueError):
        return {}
    if isinstance(got, list):
        return {x: {"at": mtime, "missed": 0} for x in got if isinstance(x, str)}
    out = {}
    for k, v in (got.items() if isinstance(got, dict) else []):
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = {"at": float(v.get("at") or 0), "missed": int(v.get("missed") or 0),
                      **({"gone": True} if v.get("gone") else {})}
    return out


def merge_write(f: Path, mine: dict, read) -> dict:
    """17j: read, merge, write, under a lock — two fetches at once each
    rewrote the file from memory, and erased each other's verdicts and safe
    boxes. The newer entry of each key wins ("at"; a forgotten one is kept as
    {gone, at} so it isn't brought back)"""
    import fcntl
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = read(f)
        for k, v in mine.items():
            if k not in out or float(v.get("at") or 0) >= float(out[k].get("at") or 0):
                out[k] = v
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        tmp.replace(f)
    return out


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
                    help="17f: do it again every 3m (or 180s, 1h) until every box is done")
    ap.add_argument("--abandoned", action="append", default=[], metavar="BUILD/STEP",
                    help="17i: a step given up on, e.g. Qwen3.6-35B-A3B-BF16/A3-2 — left out of "
                         "its box's steps and plan, and said on its line (one each)")
    a = ap.parse_args(argv)
    for x in a.abandoned:
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", x.strip()):
            ap.error(f"--abandoned {x!r}: <build>/<step>, as its folder under /workspace, "
                     "e.g. Qwen3.6-35B-A3B-BF16/A3-2")
    a.abandoned = {x.strip() for x in a.abandoned}
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
        # 17g: checked now, as --key is — a wrong name was a traceback at the
        # first compare. 17h: ~ read as the shell would, for the compare too
        parity[m["model"]] = str(Path(m["path"]).expanduser())
        if not Path(parity[m["model"]]).is_file():
            ap.error(f"--parity {s}: no such file ({m['path']})")
    key = str(Path(a.key).expanduser())
    if not Path(key).exists():
        ap.error(f"--key {a.key}: no such file")
    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest.parent / "parity").mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    known: dict[str, list[dict]] = {}
    # 17h: each build's parity verdict kept on disk — a fetch started again
    # after A3 was destroyed said nothing, and exited 0, after "Not the same".
    # 17i: with its box, its file's sha256 and when; one written before 17i
    # (no file) is replaced by the next file fetched
    vfile = dest.parent / "parity" / "verdicts.json"
    verdicts = read_verdicts(vfile)
    began = time.time()
    mine = {m.split("/", 1)[1] for m in [*shas, *parity]}
    # 17i: a box that read safe, kept on disk by its board name (never its
    # address) — after a restart a destroyed box looped for ever — and
    # forgotten whenever it is reached and isn't safe. 17j: with when it read
    # safe and how many rounds it has gone unreached since: destroyed only
    # after two (one failed ssh read a box given new work "destroyed: done"),
    # and forgotten after two days (a new box at the same address took it)
    sfile = dest.parent / "safe-boxes.json"
    was_safe = read_safe(sfile)
    a.abandoned_matched = set()
    keep_sudo()
    rounds = 0
    while True:
        failed_any, done, reached = False, [], []
        rounds += 1
        now = time.time()

        def safe_entry(bid: str) -> dict | None:
            e = was_safe.get(bid)
            return e if e and not e.get("gone") and now - e["at"] < SAFE_KEEP_S else None
        # 17i: whether the board serves each model, asked each round — the
        # first bundle's answer was printed with --register on every later
        # one, where the import refused it
        _SERVED.clear()
        for box in boxes:
            name = f"{box['host']}:{box['port']}"
            bid = box_id(name, dest)
            safe, failed, lines, steps, got = one_box(box, a, key, dest, shas, parity, seen,
                                                      verdicts)
            e = safe_entry(bid)
            if not got and e:
                # 17h: safe on an earlier round, not reached now: destroyed —
                # done (--every ran for ever). 17j: after two rounds unreached
                missed = e["missed"] + 1
                was_safe[bid] = {**e, "missed": missed, "at": e["at"]}
                if missed >= UNREACHED_ROUNDS:
                    safe, failed = True, False
                    lines = [f"{name}: read safe to destroy earlier, not reached for {missed} "
                             "rounds — destroyed: done"]
                    known.pop(name, None)
                else:
                    lines = [f"{name}: read safe to destroy earlier, not reached now — "
                             "destroyed if it isn't reached on the next round either. "
                             "NOT safe to destroy yet"]
            elif got:
                reached.append(bid)
                if safe:
                    was_safe[bid] = {"at": now, "missed": 0}
                elif bid in was_safe:
                    was_safe[bid] = {"gone": True, "at": now}
            print("\n".join(lines), flush=True)
            failed_any |= failed
            done.append(safe)
            if got:
                known[name] = steps
            elif failed and name in known:
                known[name] = [{**x, "reachable": False} for x in known[name]]
        was_safe = merge_write(sfile, was_safe, read_safe)
        # 17j: an --abandoned value that matched no step on the boxes reached
        for x in sorted(set(a.abandoned) - a.abandoned_matched):
            print(f"--abandoned {x} matched no step on the boxes reached — check it against "
                  "the box's line (<build>/<step>, as its folder under /workspace)", flush=True)
        # 17j: --parity for a build with no verdict, compared with its copy
        # here — a verdict printed by an older fetch was lost on the upgrade,
        # and --parity given after the box was destroyed said nothing
        for model, server in sorted(parity.items()):
            build = model.split("/", 1)[1]
            here = dest.parent / "parity" / f"{build}-box.jsonl"
            v = verdicts.get(build) or {}
            if (not v or v.get("gone")) and here.exists():
                code, said = run([*COMPARE, server, str(here),
                                  *(["--file-sha256", shas[model]] if model in shas else [])],
                                 cwd=REPO, timeout=IMPORT_S)
                first = next((x for x in said.splitlines() if x.strip()), f"exit {code}")
                verdicts[build] = {"same": code == 0 and first.startswith("The same"),
                                   "first": first, "box": "the copy here",
                                   "file_sha256": sha256(here), "server": server, "at": now}
        if a.board:
            print(post_boxes([x for v in known.values() for x in v], dest,
                             [box_id(f"{b['host']}:{b['port']}", dest) for b in boxes],
                             reached, a.every), flush=True)
        # 17g: each build's parity verdict, a line of its own every round; one
        # that isn't the same, or couldn't be compared, is a failure. 17i: only
        # a build on this command line, with its box and when
        verdicts = merge_write(vfile, verdicts, read_verdicts)
        for build, v in sorted(verdicts.items()):
            if build not in mine or v.get("gone"):
                continue
            # a verdict from an earlier fetch says when, and of which file
            when = time.strftime("%d %b %H:%M", time.localtime(v["at"])) \
                if v.get("at") and v["at"] < began else ""
            print(f"{build}'s parity: {v['first']}"
                  + (f" (compared {when}, its file {v['file_sha256'][:12]})"
                     if when and v.get("file_sha256") else ""), flush=True)
            failed_any |= not v["same"]
        if not a.every or all(done):
            return 1 if failed_any else 0
        print(f"— again in {round(a.every / 60)} min ({sum(done)} of {len(done)} boxes done)",
              flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
