#!/usr/bin/env python3
"""15.5: what the runner image (the Dockerfile's `runner` stage) must never
hold, checked on the image itself — every file in it — before it is pushed to
ghcr.io/masein/evalboard-runner, which is public:
- the Knowledge exam's banks and the judge's rubrics (eval_tasks/fr), the
  Everyday bank (eval_tasks/everyday), the trust sets (eval_tasks/trust_safety),
  or any benchmark's data but DeviceMark's battery ids;
- in the board's folder (/app), anything but scripts/, service/ and
  eval_tasks/devicemark/, and any data file there named like a bank, a rubric
  or a hidden half;
- an .env file, anywhere in the image;
- a secret's name in the image's environment, or the board's start command or
  its port.

    python3 scripts/check_runner_image.py ghcr.io/masein/evalboard-runner:<tag>
    python3 scripts/check_runner_image.py --files listing.txt   (one path a line)

Each file found is named, with why, and it exits 1. The standard library only:
it runs on the Actions runner, outside the image (.github/workflows/
runner-image.yml before the push, ci.yml's docker-image job).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from remote_bundle import SECRETS  # noqa: E402 — the environment's secrets, by name

APP = "app/"
RUNNER_DIRS = ("scripts/", "service/", "eval_tasks/devicemark/")
STAYS = {"fr": "the Knowledge exam's banks and the judge's rubrics",
         "everyday": "the Everyday bank", "trust_safety": "the trust sets"}
_EVAL = re.compile(r"(?:^|/)eval_tasks/([^/]+)/")
_DATA = re.compile(r"\.(jsonl?|ya?ml|csv|tsv|md|txt|parquet|arrow)$", re.I)
_NAMED = re.compile(r"bank|rubric|criteria|hidden", re.I)
BOARD_PORT = "8899"


def _norm(path: str) -> str:
    p = path.strip()
    while p.startswith(("./", "/")):
        p = p[2:] if p.startswith("./") else p[1:]
    return p


def problems(paths: list[str], env: list[str] | None = None, cmd: list | None = None,
             ports: list[str] | None = None) -> list[str]:
    """each thing the image must not hold, in words; [] when it holds none"""
    out: list[str] = []
    files = [p for p in (_norm(x) for x in paths) if p]
    if APP + "scripts/remote_run.py" not in files:
        out.append(f"no {APP}scripts/remote_run.py: this isn't the runner image, or the "
                   f"listing is empty")
    for p in files:
        base = p.rsplit("/", 1)[-1]
        if base == ".env" or base.startswith(".env."):
            out.append(f"{p}: an .env file")
            continue
        m = _EVAL.search(p)
        if m and m[1] != "devicemark":
            out.append(f"{p}: {STAYS.get(m[1], 'a benchmark’s data other than DeviceMark’s')} "
                       f"(eval_tasks/{m[1]})")
            continue
        if p.startswith(APP):
            rel = p[len(APP):]
            if not rel.startswith(RUNNER_DIRS):
                out.append(f"{p}: not the runner's — {APP} holds {', '.join(RUNNER_DIRS)} only")
            elif _DATA.search(base) and _NAMED.search(base):
                out.append(f"{p}: a data file named like a bank, a rubric or a hidden half")
    for e in env or []:
        name = e.split("=", 1)[0]
        if name in SECRETS:
            out.append(f"the image's environment sets {name}")
    if any("uvicorn" in str(c) for c in cmd or []):
        out.append(f"its start command starts the board: {' '.join(map(str, cmd))}")
    if any(str(x).split("/")[0] == BOARD_PORT for x in ports or []):
        out.append(f"it exposes {BOARD_PORT}, the board's port")
    return out


def image_files(image: str) -> list[str]:
    """every file and link in the image, from inside it (find, on its own
    filesystem: not /proc or /sys)"""
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "find",
                        image, "/", "-xdev", "(", "-type", "f", "-o", "-type", "l", ")",
                        "-print"], capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def image_config(image: str) -> dict:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config}}", image],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="the runner image, built")
    ap.add_argument("--files", type=Path, help="a listing of its files instead, one a line")
    a = ap.parse_args(argv)
    if a.files:
        what, paths, cfg = str(a.files), a.files.read_text(encoding="utf-8").splitlines(), {}
    elif a.image:
        what, paths, cfg = a.image, image_files(a.image), image_config(a.image)
    else:
        ap.error("an image, or --files")
    found = problems(paths, cfg.get("Env"), (cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []),
                     list((cfg.get("ExposedPorts") or {}).keys()))
    app = sum(1 for p in paths if _norm(p).startswith(APP))
    if found:
        print(f"{what}: {len(found)} thing{'s' if len(found) > 1 else ''} that must stay on the "
              f"server:")
        for x in found:
            print(f"  {x}")
        return 1
    print(f"{what}: {len(paths):,} files, {app:,} of them the runner's ({', '.join(RUNNER_DIRS)}): "
          f"no bank, rubric, trust set or .env file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
