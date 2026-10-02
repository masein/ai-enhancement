#!/usr/bin/env python3
"""15.6: the Python packages an image holds, and constraints.txt.

The board's image is built on the server at deploy and the runner image by the
mirror's Actions at merge. requirements.txt pins what we ask for; constraints.txt
pins everything else that comes with it (tokenizers, starlette, pydantic, …) and
what the base image brings, so the two builds install the same version of every
package — as they didn't for datasets. It is the board image's own list, made
by `constraints`; both Dockerfile stages install with it.

    python3 scripts/image_packages.py list <image>
    python3 scripts/image_packages.py constraints <board image> > constraints.txt
    python3 scripts/image_packages.py compare <board image> <runner image>
    python3 scripts/image_packages.py check <runner image>

compare (ci.yml's docker-image job) fails when the two images' lists differ —
but for what only the board stage installs (BOARD_ONLY) — or when either isn't
constraints.txt. check (runner-image.yml, before the push) holds one runner
image to constraints.txt. Each names every package that differs, and exits 1.

To change a requirement: edit requirements.txt, build the board image with
`--build-arg USE_CONSTRAINTS=0`, and make constraints.txt again from it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTRAINTS = HERE.parent / "constraints.txt"
# what only the board stage installs: the unit suite's runner, run in the
# container at deploy (pytest, and what it brings that the deps stage hasn't)
BOARD_ONLY = {"pytest", "iniconfig", "pluggy"}
HEADER = ("# 15.6: every Python package in the board's image, as `pip list` gives it —\n"
          "# both Dockerfile stages install with it (-c), so the board image built at\n"
          "# deploy and the runner image built by CI hold the same versions.\n"
          "# Made by: python3 scripts/image_packages.py constraints <board image>\n"
          "# (scripts/image_packages.py says when and how to make it again)\n")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(lines: list[str]) -> dict[str, str]:
    """{name: version} from `name==version` lines; comments and blanks skipped"""
    out = {}
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            out[norm(name.split("[")[0])] = version.strip()
    return out


def image_list(image: str) -> dict[str, str]:
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
                        image, "-m", "pip", "list", "--format=freeze",
                        "--disable-pip-version-check"],
                       capture_output=True, text=True, check=True)
    return parse(r.stdout.splitlines())


def against(name: str, got: dict[str, str], want: dict[str, str],
            missing_ok: set[str] = frozenset()) -> list[str]:
    """every package where an image and constraints.txt differ, in words"""
    out = [f"{name}: {p} {got[p]}, constraints.txt {want[p]}" for p in sorted(got)
           if p in want and got[p] != want[p]]
    out += [f"{name}: {p} {got[p]}, not in constraints.txt" for p in sorted(set(got) - set(want))]
    out += [f"{name}: {p} {want[p]} in constraints.txt, not in the image"
            for p in sorted(set(want) - set(got) - missing_ok)]
    return out


def compare(board: dict[str, str], runner: dict[str, str], want: dict[str, str]) -> list[str]:
    out = [f"{p}: the board's {board[p]}, the runner's {runner[p]}" for p in sorted(board)
           if p in runner and board[p] != runner[p]]
    out += [f"{p} {runner[p]}: in the runner image only" for p in sorted(set(runner) - set(board))]
    out += [f"{p} {board[p]}: in the board image only, and not from its own stage"
            for p in sorted(set(board) - set(runner) - BOARD_ONLY)]
    out += against("the board image", board, want)
    out += against("the runner image", runner, want, missing_ok=BOARD_ONLY)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("list", "constraints", "compare", "check"))
    ap.add_argument("images", nargs="+")
    ap.add_argument("--constraints", type=Path, default=CONSTRAINTS)
    a = ap.parse_args(argv)
    lists = [image_list(i) for i in a.images]
    if a.what in ("list", "constraints"):
        print((HEADER if a.what == "constraints" else "")
              + "".join(f"{p}=={v}\n" for p, v in sorted(lists[0].items())), end="")
        return 0
    want = parse(a.constraints.read_text(encoding="utf-8").splitlines())
    if a.what == "compare":
        if len(lists) != 2:
            ap.error("compare: the board image, then the runner image")
        found = compare(lists[0], lists[1], want)
        ok = (f"the board and runner images: {len(lists[1])} packages the same, the board's "
              f"{len(lists[0]) - len(lists[1])} more its own ({', '.join(sorted(BOARD_ONLY))}), "
              f"both as constraints.txt")
    else:
        found = against(a.images[0], lists[0], want, missing_ok=BOARD_ONLY)
        ok = f"{a.images[0]}: {len(lists[0])} packages, each as constraints.txt"
    if found:
        print(f"{len(found)} package{'s' if len(found) > 1 else ''} differ:")
        for x in found:
            print(f"  {x}")
        print("A requirement changed? Make constraints.txt again: see scripts/image_packages.py.")
        return 1
    print(ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
