#!/usr/bin/env python3
"""15.1: one DeviceMark row on a rented GPU, with one command — and back to the
board as one file (scripts/import_remote.py takes it in).

It runs inside the board's own runner image (the Dockerfile's `runner` stage:
the same torch, lm_eval, transformers and fast kernels as the board, the
board's code and DeviceMark's battery and nothing else), which vast.ai starts
from ghcr.io/masein/evalboard-runner (15.5: built and pushed by the mirror's
Actions); docs/REMOTE-RUNS.md has the commands.

It runs the board's own runner (service/runner.py, run_submission), not a copy
of it: the same task files (devicemark.build_tasks), prompts, generation
settings, max length, chat template, thinking switch and answer layout. What a
rented box changes, and nothing else:
- its folders are under --out, never a server's;
- no time limit on a task (TASK_TIMEOUT_S=0): nothing else is queued;
- one answer written at a time (DM_HF_MAX_BATCH=1), as the board writes them;
- no parameter cap (MAX_PARAMS_B): the box is rented for this model;
- no scoring after the run (DM_SCORE_AFTER_RUN=0): the server scores the
  answers with its own scorers when the bundle is imported.

The battery comes from its pinned sources — battery-v1.json's ids and its
datasets' revisions (devicemark.load_items) — and its hashes go in the bundle:
the server refuses one whose battery isn't its own. --battery takes the
server's items hash (import_remote.py --battery prints it), and a run whose
battery differs refuses to start rather than hours later.

Resumable per answer: lm_eval writes each answer to its cache as it lands
(runner.dm_cache), and a task already answered is skipped. Run the same
command again after the process dies or the box is stopped: it carries on from
the last saved answer. The model's revision is recorded at the start, and a
later session on another revision refuses to mix the two.

A line an answer — the task, how many of how many, the pace, the time left on
the task and on the run — to watch under tmux.

15.5: --shard i/n runs this instance's share of each task — every n-th item
from the i-th, in the battery's order (devicemark.shard_of), the same split on
every machine — so n instances answer a battery together, one bundle each
(…-shard-<i>-of-<n>.tar.gz). The server takes the n bundles in any order and
scores a task once every shard of it is in. Per-answer resume is the same.

Gated models: HF_TOKEN from the environment, never printed, never in the
bundle.

    python scripts/remote_run.py --model Qwen/Qwen3.5-4B --thinking on \\
        --only dm_ifeval --out /workspace/run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import remote_bundle as rb  # noqa: E402

SUITE = "devicemark"
TASKS = rb.SUITES[SUITE]["tasks"]
STATE = "remote_run.json"
OWN_LOG = "remote_run.log"
# the config each run here sets, and the environment variable config reads it from
ENV = {"BENCH_ROOT": "BENCH_ROOT", "RESULTS_ROOT": "OUT_ROOT", "LOGS_DIR": "LOGS",
       "DB_PATH": "SERVICE_DB", "DM_ITEMS": "DM_ITEMS", "DM_TASKS_DIR": "DM_TASKS_DIR",
       "TASK_TIMEOUT_S": "TASK_TIMEOUT_S", "DM_HF_MAX_BATCH": "DM_HF_MAX_BATCH",
       "DM_SCORE_AFTER_RUN": "DM_SCORE_AFTER_RUN", "MAX_PARAMS_B": "MAX_PARAMS_B",
       "GPU_POLL_S": "GPU_POLL_S", "DM_SHARD": "DM_SHARD"}


def settings(out: Path, shard: tuple[int, int] | None = None) -> dict:
    bench = out / "bench"
    return {"BENCH_ROOT": bench, "RESULTS_ROOT": bench / "results",
            "LOGS_DIR": bench / "logs", "DB_PATH": bench / "service.sqlite3",
            "DM_ITEMS": bench / "devicemark" / "items-v1.jsonl",
            "DM_TASKS_DIR": bench / "devicemark" / "tasks", "TASK_TIMEOUT_S": 0,
            "DM_HF_MAX_BATCH": 1, "DM_SCORE_AFTER_RUN": False,
            "MAX_PARAMS_B": float(os.environ.get("MAX_PARAMS_B") or 1000), "GPU_POLL_S": 10,
            "DM_SHARD": f"{shard[0]}/{shard[1]}" if shard else ""}


def configure(out: Path, shard: tuple[int, int] | None = None) -> dict:
    """the board's config, pointed at --out: set in the environment before
    the service is imported, and on config itself when it already was"""
    vals = settings(out, shard)
    for k, v in vals.items():
        os.environ[ENV[k]] = "0" if v is False else str(v)
    # lm_eval, the runner's child, beside this Python
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from service import config, runner
    for k, v in vals.items():
        setattr(config, k, v)
    config.OUT_DIR = config.RESULTS_ROOT / "full"
    runner.LOCK = config.RESULTS_ROOT / ".run.lock"
    for d in (config.OUT_DIR, config.LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return vals


class Out:
    """a line to the terminal and to the run's own log"""

    def __init__(self, path: Path, stream=None):
        self.path, self.stream = path, stream or sys.stdout
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, line: str) -> None:
        line = rb.scrub(line)
        print(line, file=self.stream, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")


def hours(s: float | None) -> str:
    if s is None:
        return "—"
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{max(1, round(s / 60))} min"


class Watch(threading.Thread):
    """a line an answer, read from lm_eval's cache as each lands, and the
    runner's own lines as they change"""

    def __init__(self, sid: int, tasks: list[str], say, every: float = 2.0):
        super().__init__(daemon=True)
        self.sid, self.tasks, self.say, self.every = sid, tasks, say, every
        self.done_ev = threading.Event()
        self.last: dict[str, int] = {}
        self.first: dict[str, tuple[int, float]] = {}
        self.totals: dict[str, int] = {}
        self.prog = ""
        self.pace: float | None = None
        for t in tasks:                     # what this session starts from: a line a new answer
            n = self._answered(t)
            self.first[t], self.last[t] = (n, time.time()), n

    def run(self) -> None:
        while not self.done_ev.wait(self.every):
            try:
                self.tick()
            except Exception:                       # noqa: BLE001 — never the run's business
                pass

    def stop(self) -> None:
        self.done_ev.set()

    def _total(self, task: str) -> int:
        if not self.totals.get(task):
            import devicemark as dm
            from service import config
            self.totals[task] = len(dm.task_keys(config.DM_TASKS_DIR, task))
        return self.totals[task]

    def _answered(self, task: str) -> int:
        from service import config
        outs = list(config.OUT_DIR.glob(f"*/{task}_0shot"))
        n = max((rb.cache_count(o) for o in outs), default=0)
        whole = max((len(rb.task_answers(o, task)) for o in outs), default=0)
        return max(n, whole)

    def tick(self) -> None:
        from service import db
        row = db.get(self.sid) or {}
        prog = row.get("progress") or ""
        if prog and prog != self.prog:
            self.prog = prog
            self.say(f"· {prog}")
        for t in self.tasks:
            total, n = self._total(t), self._answered(t)
            while self.last[t] < n:
                self.last[t] += 1
                n0, t0 = self.first[t]
                if self.last[t] > n0:
                    self.pace = (time.time() - t0) / (self.last[t] - n0)
                left_task = (total - self.last[t]) * self.pace if self.pace else None
                left_run = (sum(max(0, self._total(x) - self.last.get(x, 0)) for x in self.tasks)
                            * self.pace if self.pace else None)
                self.say(f"{t}  {self.last[t]}/{total or '?'}"
                         + (f" · {self.pace:.1f} s an answer · {hours(left_task)} left on this "
                            f"task · {hours(left_run)} on the run" if self.pace else ""))


def model_revision(model: str) -> str | None:
    """the model's commit on the Hub now, asked with HF_TOKEN when there is one"""
    try:
        from huggingface_hub import model_info
        return model_info(model, token=os.environ.get("HF_TOKEN") or None).sha
    except Exception:                               # noqa: BLE001 — recorded as unknown
        return None


def board_commit() -> str | None:
    """the board's commit this ran: the image's (EVALBOARD_BUILD), else the clone's"""
    if os.environ.get("EVALBOARD_BUILD"):
        return os.environ["EVALBOARD_BUILD"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def row_dir(model: str):
    """the row's folder: where the runner put the tasks' answers"""
    from service import config
    safe = model.replace("/", "__")
    rows = [d for d in sorted(config.OUT_DIR.glob(safe + "*"))
            if d.is_dir() and d.name in (safe, safe + "__thinking")
            and any((d / f"{t}_0shot").is_dir() for t in TASKS)]
    return rows[-1] if rows else None


_MODE = re.compile(r"\[generative\] thinking (on|off) \((\w+)\)")
_LEN = re.compile(r"lm_eval is told max_length=(\d+)")
_BATCH = re.compile(r"\[devicemark\] (\d+) answers? written at a time")


def setup_record(model: str, thinking: bool, tasks: list[str], hashes: dict, state: dict,
                 log: str, answers: dict, incomplete: dict,
                 shard: tuple[int, int] | None = None) -> dict:
    import devicemark as dm
    mode = _MODE.findall(log)
    ml, batch = _LEN.findall(log), _BATCH.findall(log)
    return {"suite": SUITE, "protocol": dm.VERSION, "cap": dm.CAP, "seed": dm.SEED,
            "battery": hashes, "model": model, "revision": state.get("revision"),
            "thinking": "on" if thinking else "off",
            "thinking_mode": mode[-1][1] if mode else None,
            "plan": {"max_length": int(ml[-1]) if ml else None,
                     "batch": int(batch[-1]) if batch else None},
            "tasks": {t: {"answers": len(answers.get(t, {})),
                          **({"incomplete": incomplete[t]} if t in incomplete else {})}
                      for t in tasks},
            "gpu": rb.gpu_info(), "libraries": rb.library_versions(),
            "where": "a rented GPU", "runner": "service/runner.py, called directly",
            "board_commit": board_commit(), "sessions": state.get("sessions", 1),
            "shard": {"i": shard[0], "n": shard[1]} if shard else None,
            "started_at": state.get("started_at"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def make_bundle(out: Path, model: str, thinking: bool, tasks: list[str], hashes: dict,
                state: dict, shard: tuple[int, int] | None = None
                ) -> tuple[Path | None, dict, dict]:
    """the bundle of every task answered whole — 15.5: its shard's share of
    it, as the run built it — its path (None when none is), the tasks in it
    and those not finished"""
    import hashlib

    import devicemark as dm
    from service import config
    row = row_dir(model)
    answers: dict[str, dict[str, str]] = {}
    incomplete: dict[str, dict] = {}
    for t in tasks:
        task_out = row / f"{t}_0shot" if row else None
        want = dm.task_keys(config.DM_TASKS_DIR, t)
        got = rb.task_answers(task_out, t) if task_out else {}
        if want and want <= set(got):
            answers[t] = {k: got[k] for k in sorted(want)}
        else:
            incomplete[t] = {"answers": max(len(got), rb.cache_count(task_out) if task_out
                                            else 0), "of": len(want)}
    if not answers:
        return None, answers, incomplete
    files: dict[str, bytes] = {}
    for t in answers:
        for f in sorted((row / f"{t}_0shot").rglob("*")):
            rel = f.relative_to(row)
            if f.is_file() and "lm-cache" not in rel.parts:
                files[f"results/{row.name}/{rel.as_posix()}"] = f.read_bytes()
    if (row / "model_meta.json").exists():
        files[f"results/{row.name}/model_meta.json"] = (row / "model_meta.json").read_bytes()
    # a thinking row's model, as the runner writes it beside the row
    base = config.OUT_DIR / model.replace("/", "__")
    if base != row and (base / "model_meta.json").exists():
        files[f"results/{base.name}/model_meta.json"] = (base / "model_meta.json").read_bytes()
    logs = sorted(config.LOGS_DIR.glob(f"service_*_{model.replace('/', '__')}.log"),
                  key=lambda p: int(p.name.split("_")[1]))
    log = "".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    own = out / OWN_LOG
    log = rb.scrub(log + ("\n===== remote_run =====\n" + own.read_text(encoding="utf-8")
                          if own.exists() else ""))
    setup = setup_record(model, thinking, tasks, hashes, state, log, answers, incomplete, shard)
    files["run.log"] = log.encode("utf-8")
    files["setup.json"] = rb.scrub(json.dumps(setup, indent=1, sort_keys=True)).encode("utf-8")
    bundle = {"format": rb.FORMAT, "suite": SUITE, "model": model, "thinking": thinking,
              "row": row.name, "tasks": {t: len(a) for t, a in answers.items()},
              "incomplete": incomplete, "answers_sha256": rb.answers_digest(answers),
              "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    if shard:
        bundle["shard"] = {"i": shard[0], "n": shard[1]}
    files["bundle.json"] = json.dumps(bundle, indent=1, sort_keys=True).encode("utf-8")
    path = rb.write(out / rb.bundle_name(SUITE, model, thinking, shard), files)
    return path, answers, incomplete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a Hugging Face model id")
    ap.add_argument("--thinking", required=True, choices=("on", "off"))
    ap.add_argument("--only", action="append", choices=TASKS,
                    help="one task (repeatable); every task when none is given")
    ap.add_argument("--out", default="remote-run", help="where the run and its bundle go")
    ap.add_argument("--shard", default="",
                    help="i/n: this instance's share of each task, every n-th item from the "
                         "i-th — n instances, one bundle each; the server scores a task once "
                         "every shard of it is in")
    ap.add_argument("--battery", default="",
                    help="the server's items sha256 (import_remote.py --battery): refuse "
                         "another battery before starting")
    ap.add_argument("--suite", default=SUITE, choices=tuple(rb.SUITES))
    ap.add_argument("--by", default="remote", help="who ran it, for the run's record")
    a = ap.parse_args(argv)
    import devicemark as dm
    try:
        shard = dm.parse_shard(a.shard)
    except ValueError as e:
        ap.error(f"--shard: {e}")
    out = Path(a.out).resolve()
    thinking = a.thinking == "on"
    tasks = [t for t in TASKS if t in (a.only or TASKS)]
    configure(out, shard)
    say = Out(out / OWN_LOG)
    from service import config, db, runner
    db.init()
    words = f"shard {shard[0]} of {shard[1]}" if shard else ""
    say(f"{a.model} · thinking {a.thinking} · {', '.join(tasks)}"
        + (f" · {words}" if shard else "") + f" · in {out}")

    # the run this folder holds: one model, one mode, one revision, one shard
    sp = out / STATE
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    if state and (state.get("model"), state.get("thinking")) != (a.model, thinking):
        raise SystemExit(f"{out} holds a run of {state.get('model')} (thinking "
                         f"{'on' if state.get('thinking') else 'off'}): give this one another --out")
    if state and (state.get("shard") or "") != (a.shard.strip() if shard else ""):
        raise SystemExit(f"{out} holds {('shard ' + state['shard']) if state.get('shard') else 'a whole run'}"
                         f" of {a.model}: give {words or 'a whole run'} another --out")
    rev = model_revision(a.model)
    if state.get("revision") and rev and rev != state["revision"]:
        raise SystemExit(f"{a.model} changed on the Hub since this run began ({state['revision'][:12]} "
                         f"then, {rev[:12]} now): its answers can't be mixed — start another --out")
    state = {"model": a.model, "thinking": thinking, "revision": state.get("revision") or rev,
             "started_at": state.get("started_at")
             or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "sessions": int(state.get("sessions", 0)) + 1,
             "shard": f"{shard[0]}/{shard[1]}" if shard else None}
    sp.write_text(json.dumps(state, indent=1), encoding="utf-8")

    # the battery, from its pinned sources, as the board builds it
    items = dm.load_items(config.DM_ITEMS)
    hashes = dm.battery_hashes(items)
    say(f"battery {dm.VERSION}: {len(items)} items · items sha256 {hashes['items'][:16]}")
    if a.battery and a.battery != hashes["items"]:
        raise SystemExit(f"this battery's items hash is {hashes['items'][:16]}, the server's "
                         f"{a.battery[:16]}: the server would refuse the bundle. Nothing was run.")
    if shard:
        share = []
        for t in tasks:
            every = [k for b, k in dm.keys_for("full") if dm.TASK[b] == t]
            share.append(f"{t} {len(dm.shard_of(every, shard))} of {len(every)}")
        i, n = shard
        say(f"{words}: items {i}, {i + n}, {i + 2 * n}, … of each task · " + ", ".join(share))
    gpu, libs = rb.gpu_info(), rb.library_versions()
    say(f"GPU {gpu.get('name') or 'unknown'} (driver {gpu.get('driver') or '?'}) · torch "
        f"{libs.get('torch_build')} · transformers {libs.get('transformers')} · lm_eval "
        f"{libs.get('lm_eval')} · fast kernels {libs.get('fast_kernels') or 'none'} · "
        f"model revision {(state.get('revision') or 'unknown')[:12]}")

    sid = db.add(a.model, "instruct", "devicemark", a.by, "run on a rented GPU (remote_run.py)",
                 thinking=thinking, part="full",
                 tasks=tasks if len(tasks) < len(TASKS) else None)
    watch = Watch(sid, tasks, say)
    watch.start()
    try:
        runner.run_submission(db.get(sid))
    finally:
        watch.stop()
        watch.tick()
    row = db.get(sid) or {}
    say(f"the run: {row.get('status')} · {row.get('progress') or ''}"
        + (f" · {row['error']}" if row.get("error") else ""))
    path, answers, incomplete = make_bundle(out, a.model, thinking, tasks, hashes, state, shard)
    for t, x in incomplete.items():
        say(f"{t}: {x['answers']} of {x['of']} answered — run the same command again to carry on")
    if not path:
        say("no task is answered whole yet: no bundle")
        return 1
    size = path.stat().st_size
    say(f"bundle {path} · {size / 1024 ** 2:.1f} MB · sha256 {rb.sha256_file(path)[:16]} · "
        + ", ".join(f"{t} ({len(v)} answers)" for t, v in answers.items()))
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
