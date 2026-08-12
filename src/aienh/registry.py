"""
The model registry: names, configs, points, lineage.

"Each model got a name and points and configurations" — this file is that,
concretely. One append-only JSONL file is the whole database, and that is a
deliberate choice: it is diffable, greppable, mergeable in git, and impossible to
corrupt with a bad migration. Move to Postgres when you have a reason, not before.

The four fields that make a registry useful instead of decorative:

    name          stable, human-sayable, unique. Never reuse one.
    config_hash   hash of everything that could change the numbers (model config,
                  train config, data label). Two rows with the same name and
                  different config hashes means someone overwrote a result, which
                  is the most expensive class of mistake on an eval team.
    suite_hash    which exam this score came from. Rows with different suite
                  hashes are NOT comparable, and the leaderboard refuses to
                  pretend otherwise.
    parent        the checkpoint this run started from. This is lineage:
                  pre-lucid-ridge -> sft-keen-onyx -> grpo-warm-beacon.
                  Without it, "why did the RL run regress?" is unanswerable,
                  because you cannot find the model it regressed from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_PATH = Path("runs/registry.jsonl")


@dataclass
class RunRecord:
    name: str
    kind: str                       # pretrain | sft | distill | grpo | eval
    created_at: str
    config_hash: str
    points: float | None = None
    suite_hash: str | None = None
    parent: str | None = None       # the run whose checkpoint this started from
    checkpoint: str | None = None
    data: str | None = None
    params_total: int | None = None
    params_active: int | None = None
    train_tokens: int | None = None
    steps: int | None = None
    wall_clock_s: float | None = None
    git_sha: str | None = None
    metrics: dict = field(default_factory=dict)      # flat, chartable scalars
    breakdown: dict = field(default_factory=dict)    # per-task points
    config: dict = field(default_factory=dict)       # full config, for reproduction
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def record(rec: RunRecord, path: str | Path = DEFAULT_PATH) -> RunRecord:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
    return rec


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_all(path: str | Path = DEFAULT_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Last write wins per name, so re-running an eval updates the row instead of
    # duplicating it — while the full history stays in the file.
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["name"]] = r
    return list(latest.values())


def leaderboard(path: str | Path = DEFAULT_PATH, suite_hash: str | None = None) -> list[dict]:
    rows = [r for r in load_all(path) if r.get("points") is not None]
    if suite_hash:
        rows = [r for r in rows if r.get("suite_hash") == suite_hash]
    return sorted(rows, key=lambda r: r["points"], reverse=True)


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "(registry empty — run a training job first)"
    hashes = {r.get("suite_hash") for r in rows}
    header = f"{'#':>2}  {'name':<28} {'kind':<9} {'points':>7}  {'params':>10} {'active':>10}  {'parent':<24} suite"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:>2}  {r['name']:<28} {r['kind']:<9} {r['points']:>7.2f}  "
            f"{(r.get('params_total') or 0):>10,} {(r.get('params_active') or 0):>10,}  "
            f"{(r.get('parent') or '-'):<24} {r.get('suite_hash') or '-'}"
        )
    if len(hashes) > 1:
        lines.append("")
        lines.append(f"!! WARNING: {len(hashes)} different suite hashes in this table "
                     f"({sorted(h for h in hashes if h)}). Rows from different suites are "
                     f"NOT comparable — filter by one suite_hash before drawing conclusions.")
    return "\n".join(lines)


def from_train_result(result: dict, kind: str, eval_out: dict | None = None,
                      parent: str | None = None, notes: str = "") -> RunRecord:
    """Adapter: turn a train()/eval() return value into a registry row."""
    from .utils import git_sha
    s = result.get("summary", {})
    cfg = result.get("config", {})
    metrics = {k: v for k, v in s.items() if isinstance(v, (int, float))}
    if eval_out:
        for task, r in eval_out["results"].items():
            metrics[f"{task}/{r['metric']}"] = r["value"]
            if r.get("stderr"):
                metrics[f"{task}/stderr"] = r["stderr"]
    return RunRecord(
        name=result["name"], kind=kind, created_at=now(),
        config_hash=result.get("config_hash", ""),
        points=(eval_out or {}).get("points"),
        suite_hash=(eval_out or {}).get("suite_hash"),
        parent=parent, checkpoint=result.get("checkpoint"),
        data=(result.get("dataset") or {}).get("label"),
        params_total=s.get("params_total"), params_active=s.get("params_active"),
        train_tokens=s.get("total_train_tokens"),
        # step count lives in a different key per stage: `steps` for pretraining,
        # `iterations` for RL, and only in the summary for SFT (where it is derived
        # from epochs). Try all of them rather than silently recording null.
        steps=cfg.get("steps") or cfg.get("iterations") or s.get("steps"),
        wall_clock_s=s.get("perf/wall_clock_s"), git_sha=git_sha(),
        metrics=metrics, breakdown=(eval_out or {}).get("breakdown", {}),
        config=cfg, notes=notes,
    )


if __name__ == "__main__":  # python -m aienh.registry
    print(render_table(leaderboard()))
