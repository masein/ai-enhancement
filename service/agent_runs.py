"""18: agent runs on the board — the benchmarks' cards, each run with its
score and Runs row, a run's tasks, a task's conversation, and the published
numbers shown beside ours as reference (docs/AGENT-RUNS-design.md § 8).

A run's folder is the runner's (scripts/agent_run.py, under
$BENCH_ROOT/agent-runs/); the board reads it where it is. Conversations,
patches and verifier output are read from there, shown on the board, and in
no export."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from . import config, db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import agent_bench as ab  # noqa: E402

PUBLISHED = Path(__file__).resolve().parent / "agent_published.json"
_KEY = re.compile(r"[A-Za-z0-9._-]{1,200}")


def store() -> Path:
    return Path(config.BENCH_ROOT) / "agent"


def runs_root() -> Path:
    return Path(config.BENCH_ROOT) / "agent-runs"


def published() -> list[dict]:
    try:
        got = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [x for x in got.get("numbers") or [] if isinstance(x, dict)]


def _folder(key: str) -> Path | None:
    """a run's folder by its key — only under agent-runs/, never elsewhere"""
    if not _KEY.fullmatch(key or ""):
        return None
    d = runs_root() / key
    return d if d.is_dir() and (d / "run.json").exists() else None


def summary(key: str) -> dict | None:
    """a run as the board shows it, read from its folder now"""
    d = _folder(key)
    if d is None:
        return None
    import import_agent
    s = import_agent.summary(d)
    reg = import_agent.board_of(d)
    row = db.get(reg["sid"]) if reg.get("sid") else None
    run = s["run"]
    return {"key": key, "sid": reg.get("sid"), "status": (row or {}).get("status") or "",
            "benchmark": run.get("benchmark"), "label": run.get("label"),
            "model": run.get("model") or "oracle", "oracle": not run.get("model"),
            "score": s["score"], "words": s["words"], "again": s.get("again") or "",
            "left_out": s.get("left_out") or [],
            "line": (s.get("progress") or {}).get("line") or "",
            "settings": {k: run.get(k) for k in (
                "dataset", "pin", "tasks_sha256", "attempts", "at_once", "agent", "config",
                "versions", "sampling", "window", "file_sha256", "build", "flags", "where",
                "started_at")},
            "results": s["results"]}


def runs() -> list[dict]:
    """every agent run the board stands for (a Runs row), newest first"""
    out = []
    for p in sorted(store().glob("*.json")) if store().is_dir() else []:
        if p.name.endswith(".summary.json"):
            continue
        s = summary(p.stem)
        if s:
            out.append({k: v for k, v in s.items() if k != "results"})
    return sorted(out, key=lambda r: -float((r.get("settings") or {}).get("started_at") or 0))


def run(key: str) -> dict | None:
    return summary(key)


def task(key: str, name: str, attempt: int = 1) -> dict | None:
    """a task's conversation, its patch and its verifier's output"""
    d = _folder(key)
    if d is None or not _KEY.fullmatch(name or ""):
        return None
    s = summary(key) or {}
    hit = next((r for r in s.get("results") or []
                if r.get("task") == name and int(r.get("attempt") or 1) == attempt), None)
    if not hit:
        return None
    if not (_KEY.fullmatch(hit.get("job") or "") and _KEY.fullmatch(hit.get("trial") or "")):
        return None
    # every file read from the run's folder down, never through a link
    return {"task": name, "attempt": attempt, "result": hit,
            **ab.conversation(d, hit["job"], hit["trial"])}


def left_out(benchmark: str) -> list[str]:
    """18c point 9: the tasks whose reference solution doesn't pass here, as
    the benchmark's oracle run read them"""
    import agent_run as ar
    try:
        return ar.oracle_states(Path(config.BENCH_ROOT), benchmark, [])["left_out"]
    except OSError:
        return []


def catalogue() -> dict:
    """the benchmarks' cards: what each is, our runs' scores, the tasks left
    out, and the published numbers beside them"""
    pub = published()
    rs = runs()
    return {"benchmarks": [
        {"key": k, **{f: ab.BENCHES[k][f] for f in (
            "label", "line", "marked", "by", "url", "licence", "tasks", "languages", "agent",
            "limits", "not_comparable")},
         "left_out": left_out(k),
         "runs": [r for r in rs if r["benchmark"] == k and not r["oracle"]],
         "published": [x for x in pub if x.get("benchmark") == k]}
        for k in ab.ORDER], "oracle": [r for r in rs if r["oracle"]],
        "read": json.loads(PUBLISHED.read_text()).get("read")
        if PUBLISHED.exists() else None}


# a running agent run whose last line is older than this has stopped posting:
# its runner died, and the model isn't held
BUSY_STALE_S = 15 * 60


def busy() -> dict:
    """the served models an agent run holds now (18b point 15): {model:
    {key, until, line}} — the Playground and the model's page say so, and
    (18c point 11) the board's queue holds its runs on that model. Read
    light, from each run's registration, Runs row and last line: the worker
    asks every few seconds"""
    out: dict = {}
    for p in sorted(store().glob("*.json")) if store().is_dir() else []:
        if p.name.endswith(".summary.json"):
            continue
        d = _folder(p.stem)
        if d is None:
            continue
        try:
            reg = json.loads(p.read_text())
            run = json.loads((d / "run.json").read_text())
            prog = json.loads((d / "progress.json").read_text())
        except (OSError, ValueError):
            continue
        row = db.get(reg.get("sid")) if reg.get("sid") else None
        if not run.get("model") or (row or {}).get("status") != "running":
            continue
        if time.time() - float(prog.get("at") or 0) > BUSY_STALE_S:
            continue
        out[run["model"]] = {"key": p.stem, "until": prog.get("until"),
                             "line": str(prog.get("line") or "")}
    return out
