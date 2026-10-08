#!/usr/bin/env python3
"""18: the board's side of an agent run — run inside the board's container by
scripts/agent_run.py on the host (docs/AGENT-RUNS.md):

    python scripts/import_agent.py --served served/<model>     the model, for the run's checks
    python scripts/import_agent.py --progress <run folder>     the run's Runs row, kept up to date
    python scripts/import_agent.py <run folder>                the finished run, imported

--served prints one JSON line: the served model's address and key (to the
runner's relay, never written anywhere), whether its server serves the
registered file, the context window it runs with now, its file's sha256,
its build and flags.

A run's folder stays where the runner wrote it, under $BENCH_ROOT/agent-runs/
(the board's container sees it at the same path): its conversations,
patches and verifier output are read from there by the board and are in no
export. The board keeps, under $BENCH_ROOT/agent/, which Runs row stands for
which run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import agent_bench as ab  # noqa: E402

SUITE = "agent"


def served(model: str) -> dict:
    from service import served as sv
    rec = sv.get(model)
    if not rec:
        return {"why": f"{model} isn't registered on the board (Test a model ▸ a model served "
                       "elsewhere)"}
    if sv.is_openrouter(rec):
        return {"why": f"{model} is a model from OpenRouter: agent runs ask this server's own"}
    why = sv.check_pin(rec)
    if why and not sv.changed(why):
        return {"why": why}                             # its server doesn't answer
    if why and why.startswith(sv.WINDOW_LINE.split("{", 1)[0]):
        why = ""                                        # the window alone: the runner checks it
    pin = rec.get("pin") or {}
    fs, gp = rec.get("file_sha256") or {}, rec.get("gguf_pin") or {}
    return {"why": why, "base_url": rec.get("base_url") or "", "key": rec.get("key") or "",
            "model": pin.get("model") or rec.get("name") or "", "window": sv.live_window(rec),
            "file": pin.get("file"), "file_sha256": gp.get("sha256") or fs.get("sha256") or None,
            "build": pin.get("build") or "", "flags": " ".join(
                x for x in (rec.get("flags") or "", rec.get("env") or "") if x)}


def store() -> Path:
    from service import config
    return Path(config.BENCH_ROOT) / "agent"


def board_of(rdir: Path) -> dict:
    try:
        return json.loads((store() / f"{rdir.name}.json").read_text())
    except (OSError, ValueError):
        return {}


def summary(rdir: Path) -> dict:
    """the run as the board shows it: its settings, each task's last result,
    its score — the conversations stay in its folder"""
    import agent_run as ar
    run = ar.read_json(rdir / "run.json")
    tasks = run.get("tasks") or []
    rs = ar.results(rdir, tasks, int(run.get("attempts") or 1))
    s = ab.score(rs, int(run.get("of") or len(tasks)))
    return {"key": rdir.name, "folder": str(rdir), "run": run, "results": rs, "score": s,
            "words": ab.score_words(s, int(run.get("of") or len(tasks))),
            "progress": ar.read_json(rdir / "progress.json")}


def runs_row(rdir: Path, status: str, line: str) -> int:
    """the Runs row that stands for this run — made with its first progress"""
    from service import db
    run = json.loads((rdir / "run.json").read_text())
    reg = board_of(rdir)
    sid = reg.get("sid")
    if not sid or not db.get(sid):
        b = ab.bench(run["benchmark"])
        sid = db.add(run.get("model") or "oracle", "instruct", SUITE, run.get("by") or "masein",
                     f"{b['label']} · " + ("oracle (no model)" if not run.get("model") else
                                           "agent on this server"),
                     tasks=[run["benchmark"]], status="running")
        store().mkdir(parents=True, exist_ok=True)
        (store() / f"{rdir.name}.json").write_text(json.dumps({"sid": sid, "folder": str(rdir)}))
    db.update(sid, status=status, progress=line[:400],
              **({"started_at": float(run.get("started_at"))} if run.get("started_at") else {}),
              **({"finished_at": time.time()} if status in ("done", "failed") else {}))
    return sid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("--served")
    ap.add_argument("--progress", type=Path)
    a = ap.parse_args(argv)
    from service import db
    db.init()
    if a.served:
        print(json.dumps(served(a.served)))
        return 0
    rdir = (a.progress or a.folder)
    if rdir is None or not (rdir / "run.json").exists():
        print(f"no agent run at {rdir}")
        return 2
    if a.progress:
        p = json.loads((rdir / "progress.json").read_text())
        sid = runs_row(rdir, "running", p.get("line") or "starting")
        print(f"Runs #{sid}: {p.get('line')}")
        return 0
    s = summary(rdir)
    want = len(s["run"].get("tasks") or []) * int(s["run"].get("attempts") or 1)
    done = s["score"]["n"]
    status = "done" if done >= want else "failed"
    line = s["words"] + ("" if status == "done" else f" · {want - done} without a result")
    sid = runs_row(rdir, status, line)
    store().mkdir(parents=True, exist_ok=True)
    (store() / f"{rdir.name}.summary.json").write_text(json.dumps(s))
    print(f"Runs #{sid}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
