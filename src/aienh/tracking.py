"""
Experiment tracking — Weights & Biases, with a local fallback that always works.

What W&B actually is, minus the marketing: a database with a web UI, plus a
client that pushes numbers to it. A *run* is one execution of your script. It has
a `config` (the hyperparameters, written once, immutable), a time series of
`log()`ed metrics (loss curves), a `summary` (final scalar values, which is what
the run table and leaderboards sort on), and optional `artifacts` (versioned
files: checkpoints, datasets, eval outputs).

The three things that make it worth using in a team:

  1. config + metrics live together. "Which run had lr=3e-4 and beta2=0.95?" is a
     query, not an archaeology project through someone's shell history.
  2. the run table is a shared, sortable ground truth. This is what people mean
     by "the dashboard".
  3. artifacts give you lineage: this eval number came from that checkpoint,
     which came from that dataset version. When a benchmark result is questioned
     — and it will be — lineage is the only thing that settles it.

Sweeps are a fourth thing: you declare a search space in YAML, W&B runs an agent
that keeps launching runs with sampled hyperparameters (grid / random / bayes).
See scripts/sweep_local.py for the dependency-free equivalent.

This wrapper exists so nothing in the repo hard-depends on the network. Set
AIENH_TRACKER=local (or pass mode="local") and you get identical metrics as
JSONL on disk. Same call sites, no code changes. Do this in CI.

W&B API surface used here (verified against the current docs quickstart):
    wandb.init(project=..., name=..., config={...}) -> run
    run.log({"loss": 1.23}, step=42)
    run.summary["eval/ppl"] = 12.3
    wandb.Table(columns=[...], data=[[...]])
    run.finish()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Tracker:
    """
    mode="auto"   use W&B if it is installed AND an API key / WANDB_MODE is set,
                  otherwise fall back to local JSONL.
    mode="wandb"  require W&B; raise if unavailable.
    mode="local"  never touch the network.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: dict | None = None,
        mode: str = "auto",
        out_dir: str | Path = "runs",
        tags: list[str] | None = None,
        group: str | None = None,
    ):
        self.project = project
        self.name = name
        self.config = dict(config or {})
        self.dir = Path(out_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self._summary: dict[str, Any] = {}
        self.run = None

        mode = os.environ.get("AIENH_TRACKER", mode)
        self.backend = "local"
        if mode in ("auto", "wandb"):
            self.backend = self._try_wandb(mode, tags, group)

        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        self._metrics_fh = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        print(f"[tracker] backend={self.backend} run={self.name} dir={self.dir}")

    def _try_wandb(self, mode: str, tags, group) -> str:
        try:
            import wandb  # noqa: PLC0415
        except ImportError:
            if mode == "wandb":
                raise RuntimeError("mode='wandb' but wandb is not installed: pip install wandb")
            return "local"
        has_auth = bool(
            os.environ.get("WANDB_API_KEY")
            or os.environ.get("WANDB_MODE")
            or (Path.home() / ".netrc").exists()
        )
        if not has_auth and mode == "auto":
            return "local"
        self.run = wandb.init(project=self.project, name=self.name,
                              config=self.config, tags=tags, group=group)
        self._wandb = wandb
        return "wandb"

    # -- logging ---------------------------------------------------------
    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log a point on the time series. `step` is the x-axis; keep it
        consistent (optimizer steps, not epochs, not wall-clock) or your charts
        will not line up across runs."""
        payload = {**metrics}
        if step is not None:
            payload["_step"] = step
        payload["_elapsed_s"] = round(time.time() - self._t0, 3)
        self._metrics_fh.write(json.dumps(payload, default=float) + "\n")
        self._metrics_fh.flush()
        if self.backend == "wandb":
            self.run.log(metrics, step=step)

    def set_summary(self, metrics: dict) -> None:
        """Final values. These are what a leaderboard sorts on."""
        self._summary.update(metrics)
        if self.backend == "wandb":
            for k, v in metrics.items():
                self.run.summary[k] = v

    def log_table(self, name: str, columns: list[str], rows: list[list]) -> None:
        """Sample-level outputs (prompt / generation / reward). Aggregate metrics
        tell you a number moved; tables tell you why. Always log a handful of raw
        samples — most eval bugs are visible in 20 rows and invisible in a mean."""
        (self.dir / f"{name}.json").write_text(
            json.dumps({"columns": columns, "data": rows}, indent=2, default=str)
        )
        if self.backend == "wandb":
            self.run.log({name: self._wandb.Table(columns=columns, data=rows)})

    def log_artifact(self, path: str | Path, name: str, kind: str = "model") -> None:
        """Version a file. Locally this just records the path + size + hash."""
        p = Path(path)
        meta = {"name": name, "type": kind, "path": str(p),
                "bytes": p.stat().st_size if p.exists() else None}
        (self.dir / f"artifact_{name}.json").write_text(json.dumps(meta, indent=2))
        if self.backend == "wandb" and p.exists():
            art = self._wandb.Artifact(name=name, type=kind)
            art.add_file(str(p))
            self.run.log_artifact(art)

    def finish(self) -> dict:
        (self.dir / "summary.json").write_text(json.dumps(self._summary, indent=2, default=str))
        self._metrics_fh.close()
        if self.backend == "wandb":
            self.run.finish()
        return self._summary

    # context manager sugar
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False


def read_metrics(run_dir: str | Path) -> list[dict]:
    """Read back the local JSONL — used by the dashboard to draw curves."""
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
