#!/usr/bin/env python3
"""17e: every box's bundles fetched and imported with one command, on the
server (as you, in ~/benchmarks/aienh) — G3 and G4's scp and import, a box at
a time, without typing either.

    python3 scripts/frontier_fetch.py --key ~/.ssh/id_ed25519 \\
        --sha served/<phone-build>=<its sha256> \\
        --sha served/<original-build>=<its sha256> \\
        <host>:<port> <host>:<port> …

Each box (root@ unless another user is given) is asked over SSH for every
bundle under /workspace (/workspace/*/frontier-*.tar.gz), and they are copied
into ~/benchmarks/bundles. Each is then imported into the board's container
(sudo docker compose exec -T bench python scripts/import_remote.py), with the
--file-sha256 given for its served model (the bundle's bundle.json names it).
One line a box, then one line a bundle: imported, imported already, or
refused and why. Run it again whenever: a bundle imported already is said so
and nothing changes. A bundle of a model with no --sha (G6's calibration, its
first import registering it) is fetched and left for you to import by hand.

Python's standard library only: it runs on the server, outside the image.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REMOTE = "/workspace/*/frontier-*.tar.gz"
IMPORT = ["sudo", "docker", "compose", "exec", "-T", "bench", "python",
          "scripts/import_remote.py"]
_BOX = re.compile(r"(?:(?P<user>[A-Za-z0-9._-]+)@)?(?P<host>[A-Za-z0-9.-]+):(?P<port>\d+)")
_SHA = re.compile(r"(?P<model>served/[A-Za-z0-9._-]+)=(?P<sha>[0-9a-fA-F]{64})")


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """(exit code, what it printed) — its output kept, not shown"""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ssh_opts(key: str, port: str, scp: bool = False) -> list[str]:
    # a rented box is a new host each time: its key is taken on first
    # contact, as answering yes would
    return ["-i", key, "-P" if scp else "-p", port, "-o", "StrictHostKeyChecking=accept-new"]


def fetch(box: re.Match, key: str, dest: Path) -> tuple[list[Path], str]:
    """(the bundles copied from `box`, the line that says so)"""
    who = f"{box['user'] or 'root'}@{box['host']}"
    name = f"{box['host']}:{box['port']}"
    code, said = run(["ssh", *ssh_opts(key, box["port"]), who, f"ls -1 {REMOTE}"])
    paths = [x.strip() for x in said.splitlines() if x.strip().endswith(".tar.gz")]
    if code != 0 and not paths:
        last = [x for x in said.splitlines() if x.strip()]
        why = last[-1] if last else f"ssh exit {code}"
        return [], (f"{name}: no bundle there yet" if "No such file" in said else
                    f"{name}: couldn't list its bundles — {why}")
    code, said = run(["scp", *ssh_opts(key, box["port"], scp=True),
                      *[f"{who}:{p}" for p in paths], str(dest) + "/"])
    got = [dest / Path(p).name for p in paths if (dest / Path(p).name).exists()]
    if code != 0:
        last = [x for x in said.splitlines() if x.strip()]
        return got, (f"{name}: {len(got)} of {len(paths)} bundles copied — "
                     + (last[-1] if last else f"scp exit {code}"))
    return got, f"{name}: {len(got)} bundle{'s' if len(got) != 1 else ''} · " + ", ".join(
        p.name for p in got)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", nargs="+", help="each box as <host>:<port> (root@ unless "
                                              "<user>@ is given), as vast.ai's SSH line says")
    ap.add_argument("--key", required=True, help="your SSH key, e.g. ~/.ssh/id_ed25519")
    ap.add_argument("--sha", action="append", default=[],
                    help="served/<name>=<sha256>: a build's file, as G0 gave it (one each)")
    ap.add_argument("--dest", default="~/benchmarks/bundles", help="where the bundles go")
    ap.add_argument("--by", default="masein", help="your name, for the Runs list")
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
    key = str(Path(a.key).expanduser())
    if not Path(key).exists():
        ap.error(f"--key {a.key}: no such file")
    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    fetched: list[Path] = []
    for box in boxes:
        got, line = fetch(box, key, dest)
        print(line, flush=True)
        fetched += [p for p in got if p not in fetched]
    bad = 0
    for path in sorted(fetched):
        model = model_of(path)
        if not model:
            print(f"{path.name}: its bundle.json couldn't be read — not imported", flush=True)
            bad += 1
            continue
        if model not in shas:
            print(f"{path.name}: no --sha for {model} — import it by hand (G4, G6)", flush=True)
            continue
        code, said = run([*IMPORT, str(path), "--by", a.by, "--file-sha256", shas[model]],
                         cwd=REPO)
        print(f"{path.name}: {summary(code, said)}", flush=True)
        bad += code != 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
