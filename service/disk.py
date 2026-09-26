"""12f.0: the server's free disk, on the status dot and before a run.

On 2026-09-26 the system disk filled (Docker's images) and nothing on the
board said so. The check looks at the filesystem that holds the results
folder and at `/` as the container sees it, and takes the lower: amber under
DISK_AMBER_GB, red under DISK_RED_GB — and below red a run does not start."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import config


def _free(path: Path) -> int | None:
    p = Path(path)
    while not p.exists() and p != p.parent:        # the results folder may not exist yet
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def check() -> dict:
    """{free_gb, level: ok | amber | red, line}"""
    frees = [f for f in (_free(config.RESULTS_ROOT), _free(Path("/"))) if f is not None]
    if not frees:
        return {"free_gb": None, "level": "ok", "line": ""}
    gb = min(frees) / 1e9
    level = "red" if gb < config.DISK_RED_GB else "amber" if gb < config.DISK_AMBER_GB else "ok"
    line = f"The server's disk has {gb:.1f} GB free. Runs may fail to save." if level != "ok" else ""
    return {"free_gb": round(gb, 1), "level": level, "line": line}


def status_check() -> dict | None:
    """the check the status dot shows, or None when there's room"""
    c = check()
    if c["level"] == "ok":
        return None
    return {"key": "disk", "severity": "error" if c["level"] == "red" else "warning",
            "short": c["line"], "show": None, "judged": False, "limit": False,
            "text": c["line"] + (" New runs won't start until there's more room: free some space "
                                 "(old Docker images often hold the most)."
                                 if c["level"] == "red" else
                                 f" Under {config.DISK_RED_GB:g} GB, new runs won't start.")}


def blocks_run() -> str:
    """'' when a run may start, else why not — the dot's own words"""
    c = check()
    return c["line"] if c["level"] == "red" else ""
