#!/usr/bin/env python3
"""18: an agent benchmark on this server (docs/AGENT-RUNS.md) — one command,
under tmux, as the fetch is:

    python3 scripts/agent_run.py <benchmark> --as served/<model> [--tasks N] [--attempts K]
    python3 scripts/agent_run.py <benchmark> --oracle [--tasks N]     (step A: no model)

Run it with the agent venv's python (Harbor 0.24.0 and mini-swe-agent 2.4.6;
docs/AGENT-RUNS.md § A). It never starts or stops a model server.

Before the first task, each a line with what to do when it fails:
- the pinned versions; Docker, without sudo; at least 50 GB free where Docker
  keeps its images;
- the model (not with --oracle): the board serves it, its server answers
  with the registered file, its window is at least 131,072 tokens (named in
  the refusal), and a chat request with the bash tool comes back as a tool
  call, through the relay.

Then the tasks, one Harbor job each, at most --at-once at a time:
- a pilot's tasks are fixed: --tasks 10 is the same ten every time, spread
  over the benchmark's languages;
- it carries on: a task with a result is never asked again; an error of
  ours (Docker, the relay, the server down) is asked again, three times at
  most; a run killed at any point continues with the same command;
- each task's image is removed once no waiting task needs it, and nothing
  more is pulled when Docker's disk would fall under 50 GB free: the tasks
  running finish, and the run stops in one line;
- progress goes to the board every --every (3 minutes), as a Runs row: "37
  of 300 · 21 resolved · 28 min a task · about 5 days left";
- at the end it is imported into the board, through the container.

The run's folder, under $BENCH_ROOT/agent-runs/, keeps every pinned version,
the sampling, the window, the agent's prompt hash, the model file's sha256,
the llama.cpp build and flags (run.json), and each task's Harbor job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import agent_bench as ab  # noqa: E402

BOARD = ["docker", "compose", "exec", "-T", "bench", "python", "scripts/import_agent.py"]
TRIES = 3                               # an error of ours is asked again this many times
EVERY_S = 180


def run(cmd: list[str], cwd: Path | None = None, timeout: float | None = None,
        env: dict | None = None) -> tuple[int, str]:
    """a command, its exit code and output — replaced in tests"""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd[:3])}: no answer in {timeout} s"


def say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# where things are
# ---------------------------------------------------------------------------

def bench_root() -> Path:
    """$BENCH_ROOT, or the board's .env beside this checkout — the board's
    container sees the run's folder at the same path"""
    v = os.environ.get("BENCH_ROOT")
    if not v:
        try:
            for line in (REPO / ".env").read_text().splitlines():
                if line.startswith("BENCH_ROOT="):
                    v = line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            pass
    if not v:
        raise SystemExit("BENCH_ROOT isn't set, and the board's .env doesn't give it: "
                         "export BENCH_ROOT=<the board's BENCH_ROOT>")
    return Path(v).expanduser()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80]


def run_dir(root: Path, name: str, model: str, attempts: int, oracle: bool) -> Path:
    who = "oracle" if oracle else slug(model.removeprefix("served/"))
    return root / "agent-runs" / f"{name}__{who}__k{attempts}"


# ---------------------------------------------------------------------------
# before the first task
# ---------------------------------------------------------------------------

def versions() -> dict:
    """the pinned versions, as installed — {} for one that isn't"""
    from importlib import metadata
    out = {}
    for pkg in ("harbor", "mini-swe-agent", "litellm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = ""
    return out


def check_versions(v: dict, oracle: bool) -> str:
    if v.get("harbor") != ab.HARBOR_VERSION:
        return (f"Harbor {v.get('harbor') or 'isn’t installed'} — {ab.HARBOR_VERSION} is pinned: run "
                "this with the agent venv's python (docs/AGENT-RUNS.md § A)")
    if not oracle and v.get("mini-swe-agent") != ab.MINI_VERSION:
        return (f"mini-swe-agent {v.get('mini-swe-agent') or 'isn’t installed'} — "
                f"{ab.MINI_VERSION} is pinned (docs/AGENT-RUNS.md § A)")
    return ""


def docker_root() -> tuple[str, str]:
    """(where Docker keeps its images, why not) — Docker as this user"""
    code, out = run(["docker", "info", "-f", "{{.DockerRootDir}}"], timeout=30)
    if code != 0:
        return "", ("Docker doesn't answer this user without sudo — Harbor runs `docker` as "
                    "you: add yourself to the docker group (docs/AGENT-RUNS.md § A) — "
                    + out.strip()[-200:])
    return out.strip().splitlines()[-1], ""


def free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def disk_line(path: str, need_gb: float = 0.0) -> str:
    """'' while Docker's disk keeps 50 GB free after `need_gb` more; else why"""
    free = free_gb(path)
    if free - need_gb < ab.DISK_FLOOR_GB:
        return (f"Docker's disk ({path}) has {free:.0f} GB free"
                + (f" and the next image needs about {need_gb:.0f}" if need_gb else "")
                + f" — {ab.DISK_FLOOR_GB:.0f} GB must stay free: remove images "
                "(`docker image prune -a`) or free space there")
    return ""


def board(*args: str) -> tuple[int, str]:
    return run([*BOARD, *args], cwd=REPO, timeout=600)


def served_info(model: str) -> tuple[dict, str]:
    """the board's record of the served model, its server's pin check and
    its window now — asked inside the board's container"""
    code, out = board("--served", model)
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}, f"the board didn't answer for {model} — {out.strip()[-300:]}"
    if code != 0 or info.get("why"):
        return info, str(info.get("why") or out.strip()[-300:])
    return info, ""


def window_line(info: dict) -> str:
    w = info.get("window")
    if not isinstance(w, int):
        return "its server didn't say its context window (llama-server's /props)"
    if w < ab.MIN_WINDOW:
        return (f"its server runs a window of {w:,} tokens; an agent run needs at least "
                f"{ab.MIN_WINDOW:,} — start it with CTX={ab.WANT_WINDOW} (or at least "
                f"{ab.MIN_WINDOW}) and use the new window on the model's page")
    return ""


def host_address(base_url: str) -> str:
    """the server's address as this host reaches it: the board's container
    says host.docker.internal for the Docker bridge's gateway"""
    if "host.docker.internal" not in base_url:
        return base_url
    code, out = run(["docker", "network", "inspect", "bridge", "-f",
                     "{{(index .IPAM.Config 0).Gateway}}"], timeout=30)
    gw = out.strip().splitlines()[-1] if code == 0 and out.strip() else "172.17.0.1"
    return base_url.replace("host.docker.internal", gw)


def tool_call_line(relay_url: str, model: str) -> str:
    """'' when a chat request with the bash tool comes back as a tool call"""
    import urllib.request
    body = {"model": model, "max_tokens": 4096, "messages": [
        {"role": "user", "content": "Use the bash tool to run: echo ready"}],
        "tools": [{"type": "function", "function": {
            "name": "bash", "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]}
    req = urllib.request.Request(relay_url + "/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
    except Exception as e:                              # noqa: BLE001 — said in one line
        return f"a chat request through the relay failed: {e}"
    if not msg.get("tool_calls"):
        return ("its server answered with no tool call — mini-swe-agent sends its commands as "
                "tool calls: start llama-server with --jinja")
    return ""


def reach_targets(board_port: int = 8899, llama: str = "") -> list[str]:
    """what a task's container must never reach: the model's server, the
    board, the host's addresses (the tailnet's among them), the LAN's
    gateway, the internet"""
    out = []
    if llama:
        m = re.match(r"https?://([^/:]+):?(\d+)?", llama)
        if m:
            out.append(f"{m.group(1)}:{m.group(2) or 80}")
    code, addrs = run(["hostname", "-I"], timeout=10)
    for a in addrs.split() if code == 0 else []:
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", a):
            out.append(f"{a}:{board_port}")
    code, route = run(["ip", "route", "show", "default"], timeout=10)
    m = re.search(r"via (\d+\.\d+\.\d+\.\d+)", route if code == 0 else "")
    if m:
        out.append(f"{m.group(1)}:80")
    out += ["1.1.1.1:443", "8.8.8.8:53"]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# the tasks
# ---------------------------------------------------------------------------

def tasks_dir(root: Path, b: dict) -> Path:
    d = root / "agent-tasks" / slug(b["dataset"])
    return d / b["subdir"] if b.get("subdir") else d


def fetch_tasks(root: Path, b: dict) -> tuple[Path, str]:
    """the benchmark's tasks, once, at their pinned revision — Harbor's
    registry's, or a commit of their repository — and a hash of them"""
    d = tasks_dir(root, b)
    if not any(d.glob("*/task.toml")):
        top = root / "agent-tasks" / slug(b["dataset"])
        top.mkdir(parents=True, exist_ok=True)
        if b.get("git"):
            steps = [["git", "init", "-q", str(top)],
                     ["git", "-C", str(top), "fetch", "-q", "--depth", "1", b["git"], b["commit"]],
                     ["git", "-C", str(top), "checkout", "-q", "FETCH_HEAD"]]
        else:
            steps = [["harbor", "download", b["dataset"], "-o", str(top), "--export"]]
        for cmd in steps:
            code, out = run(cmd, timeout=3600)
            if code != 0:
                raise SystemExit(f"{b['label']}'s tasks couldn't be fetched — "
                                 f"{out.strip()[-300:]}")
        if not any(d.glob("*/task.toml")):
            raise SystemExit(f"{b['label']}: no task found in {d}")
    h = hashlib.sha256()
    for p in sorted(d.glob("*/task.toml")):
        h.update(p.parent.name.encode() + b"\0" + p.read_bytes())
    return d, h.hexdigest()


def languages(d: Path, b: dict) -> dict[str, str]:
    return {p.parent.name: ab.language_of(p.read_text(errors="replace"), b["language_from"])
            for p in sorted(d.glob("*/task.toml"))}


def images_of(task: Path) -> list[str]:
    """the images a task pulls: its task.toml's docker_image, its
    Dockerfile's FROM"""
    out = []
    m = re.search(r'(?m)^\s*docker_image\s*=\s*"([^"]+)"',
                  (task / "task.toml").read_text(errors="replace"))
    if m:
        out.append(m.group(1))
    try:
        out += re.findall(r"(?mi)^\s*FROM\s+(\S+)", (task / "environment" / "Dockerfile").read_text())
    except OSError:
        pass
    return out


def attempts_of(rdir: Path, task: str, k: int) -> list[dict]:
    """each finished trial of a task's attempt `k`, oldest first"""
    out = []
    for job in sorted((rdir / "jobs").glob(f"{task}__a{k}__*")):
        for trial in job.iterdir() if job.is_dir() else []:
            r = ab.read_trial(trial) if trial.is_dir() else None
            if r:
                out.append({**r, "job": job.name})
    return out


def state_of(rdir: Path, task: str, k: int) -> tuple[str, int]:
    """('done' | 'todo' | 'given up', tries so far) — done once the model's
    result is in; an error of ours is asked again, TRIES times at most"""
    tries = attempts_of(rdir, task, k)
    if any(t["result"] != "error" for t in tries):
        return "done", len(tries)
    return ("given up" if len(tries) >= TRIES else "todo"), len(tries)


def results(rdir: Path, tasks: list[str], attempts: int) -> list[dict]:
    """the last trial of each task's attempts"""
    out = []
    for t in tasks:
        for k in range(1, attempts + 1):
            tries = attempts_of(rdir, t, k)
            done = [x for x in tries if x["result"] != "error"]
            if done or tries:
                out.append({**(done or tries)[-1], "attempt": k, "tries": len(tries)})
    return out


def progress(rdir: Path, b: dict, tasks: list[str], attempts: int, started: float) -> dict:
    """the run's line for Runs: "37 of 300 · 21 resolved · 28 min a task ·
    about 5 days left" """
    rs = results(rdir, tasks, attempts)
    done = [r for r in rs if r["result"] != "error"]
    want = len(tasks) * attempts
    mins = [r["minutes"] for r in done if isinstance(r.get("minutes"), (int, float))]
    each = sum(mins) / len(mins) if mins else None
    left = each * (want - len(done)) / max(1, int(read_json(rdir / "run.json").get("at_once")
                                                      or 1)) if each else None
    words = [f"{len(done)} of {want}",
             f"{sum(1 for r in done if r['result'] == 'resolved')} resolved"]
    if each:
        words.append(f"{each:.0f} min a task")
    if left:
        words.append("about " + (f"{left / 1440:.0f} days" if left >= 2880 else
                                 f"{left / 60:.0f} h" if left >= 90 else f"{left:.0f} min")
                     + " left")
    errs = sum(1 for r in rs if r["result"] == "error")
    if errs:
        words.append(f"{errs} error{'s' if errs > 1 else ''} of ours")
    return {"line": " · ".join(words), "done": len(done), "of": want, "of_bench": b["tasks"],
            "started_at": started, "at": time.time(), "minutes_each": each}


def read_json(p: Path) -> dict:
    try:
        got = json.loads(p.read_text())
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def override(rdir: Path) -> Path:
    """the task's container: no network at all, a process limit — merged
    over every task's own compose by Harbor"""
    p = rdir / "no-network.yaml"
    p.write_text("services:\n  main:\n    network_mode: none\n    pids_limit: 4096\n")
    return p


def harbor_cmd(task: Path, job: str, rdir: Path, b: dict, a: argparse.Namespace,
               model: str, relay: str, reach: list[str]) -> list[str]:
    cmd = ["harbor", "run", "-p", str(task), "-k", "1", "-n", "1", "-o", str(rdir / "jobs"),
           "--job-name", job, "--extra-docker-compose", str(override(rdir))]
    if a.oracle:
        return [*cmd, "-a", "oracle"]
    return [*cmd, "-a", "agent_host_mini:HostMini", "-m", f"openai/{model}",
            "--ak", f"config={b['config']}", "--ak", f"cost_limit={b['cost_limit']}",
            *(["--ak", f"cwd={b['cwd']}"] if b.get("cwd") else []),
            "--ak", f"relay={relay}", "--ak", f"reach={','.join(reach)}"]


def image_gb(img: str) -> float:
    code, out = run(["docker", "image", "inspect", "-f", "{{.Size}}", img], timeout=60)
    try:
        return int(out.strip().splitlines()[-1]) / 1e9 if code == 0 else 0.0
    except (ValueError, IndexError):
        return 0.0


PROBE_IMAGE = "bash:5.2"


def check_reach(rdir: Path, targets: list[str]) -> str:
    """a container started as Harbor starts a task's, with the override:
    '' when nothing answers from inside it and no host folder is there;
    else what did"""
    import agent_host_mini as hm
    marker = rdir / "reach-marker"
    marker.write_text("x")
    base = rdir / "reach-base.yaml"
    base.write_text(f"services:\n  main:\n    image: {PROBE_IMAGE}\n"
                    "    command: [\"sh\", \"-c\", \"sleep infinity\"]\n")
    files = ["-f", str(base), "-f", str(override(rdir))]
    proj = ["docker", "compose", "-p", "agent-reach-check", *files]
    code, out = run([*proj, "up", "-d"], timeout=600)
    if code != 0:
        return f"the check's container didn't start — {out.strip()[-300:]}"
    try:
        script = hm.reach_script(targets) + f"\n[ -e {marker} ] && echo 'REACHED host folder'"
        code, out = run([*proj, "exec", "-T", "main", "bash", "-c", script], timeout=600)
        return hm.reach_verdict(out)
    finally:
        run([*proj, "down", "-v", "--remove-orphans"], timeout=600)


def remove_images(images: list[str], keep: set[str], rdir: Path | None = None) -> float:
    """the images no waiting task needs, removed — each one's digest kept
    first; the largest one's size"""
    big = 0.0
    for img in images:
        big = max(big, image_gb(img))
        if rdir is not None:
            code, out = run(["docker", "image", "inspect", "-f", "{{json .RepoDigests}}", img],
                            timeout=60)
            if code == 0:
                with open(rdir / "images.jsonl", "a") as fh:
                    fh.write(json.dumps({"image": img, "digests": out.strip()}) + "\n")
        if img not in keep:
            run(["docker", "image", "rm", img], timeout=300)
    return big


def project_of(name: str) -> str:
    """Harbor's Docker project name for a trial (its _sanitize…)"""
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def leftovers(task: str) -> None:
    """a killed run's containers of this task, removed before it is asked
    again: Harbor names a trial's project <task[:32]>__<id>__env"""
    code, out = run(["docker", "ps", "-a", "--format",
                     '{{.ID}} {{.Label "com.docker.compose.project"}}'], timeout=60)
    head = project_of(task[:32]) + "__"
    ids = [x.split()[0] for x in out.splitlines() if code == 0 and len(x.split()) == 2
           and x.split()[1].startswith(head)]
    if ids:
        run(["docker", "rm", "-f", *ids], timeout=300)


NEXT_IMAGE_GB = 5.0                     # until a task's image has been measured


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=ab.ORDER)
    ap.add_argument("--as", dest="model", default="",
                    help="the served model, as the board registers it: served/<name>")
    ap.add_argument("--oracle", action="store_true",
                    help="step A: Harbor's oracle agent applies each task's reference solution — "
                         "no model")
    ap.add_argument("--tasks", type=int, default=0, help="a pilot: this many, the same every time")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--at-once", type=int, default=1, help="tasks at a time (default 1)")
    ap.add_argument("--every", type=float, default=EVERY_S, help="seconds between progress posts")
    ap.add_argument("--no-board", action="store_true", help="nothing goes to the board")
    ap.add_argument("--by", default=os.environ.get("USER", ""), help="your name, on its Runs row")
    ap.add_argument("--check", action="store_true", help="the checks before the first task, only")
    ap.add_argument("--check-reach", action="store_true",
                    help="step A: a container as a task's tries the model's server, the board, "
                         "the host's addresses, the LAN and the internet — nothing may answer")
    a = ap.parse_args(argv)
    if not a.oracle and not a.model.startswith("served/"):
        ap.error("--as served/<model> (or --oracle for step A)")
    b = ab.bench(a.benchmark)
    root = bench_root()
    if a.check_reach:
        rdir = root / "agent-runs" / "reach-check"
        rdir.mkdir(parents=True, exist_ok=True)
        targets = reach_targets(llama=host_address("http://host.docker.internal:8090/v1"))
        why = check_reach(rdir, targets)
        say(f"refused — {why}" if why else "a task's container reaches nothing of: "
            + ", ".join(targets) + "; no host folder; at most 4,096 processes")
        return 2 if why else 0
    # before the first task: each refusal one line, with what to do
    v = versions()
    why = check_versions(v, a.oracle)
    droot, why2 = docker_root() if not why else ("", "")
    why = why or why2 or disk_line(droot)
    info: dict = {}
    if not why and not a.oracle:
        info, why = served_info(a.model)
        why = why or window_line(info)
    if why:
        say(f"refused — {why}")
        return 2
    rdir = run_dir(root, a.benchmark, a.model, a.attempts, a.oracle)
    rdir.mkdir(parents=True, exist_ok=True)
    tdir, tasks_sha = fetch_tasks(root, b)
    langs = languages(tdir, b)
    picked = ab.pilot(langs, a.tasks) if a.tasks else sorted(langs)
    relay = None
    if not a.oracle:
        import agent_relay
        relay = agent_relay.Relay(host_address(info["base_url"]), info.get("key") or "",
                                  {**ab.SAMPLING, "chat_template_kwargs": {"enable_thinking": True}},
                                  usage=rdir / "relay-usage.jsonl").start()
        why = tool_call_line(relay.url, info.get("model") or "")
        if why:
            relay.stop()
            say(f"refused — {why}")
            return 2
    if a.check:
        say(f"ready — {len(picked)} task(s) of {b['label']}; Docker keeps images at {droot} "
            f"({free_gb(droot):.0f} GB free)")
        if relay:
            relay.stop()
        return 0
    started = read_json(rdir / "run.json").get("started_at") or time.time()
    write_json(rdir / "run.json", {
        "benchmark": a.benchmark, "label": b["label"], "dataset": b["dataset"], "pin": b["pin"],
        "tasks_sha256": tasks_sha, "tasks": picked, "of": b["tasks"], "attempts": a.attempts,
        "at_once": a.at_once, "agent": "oracle" if a.oracle else b["agent"],
        "config": None if a.oracle else b["config"], "model": a.model or None,
        "versions": v, "sampling": None if a.oracle else {**ab.SAMPLING, "thinking": "on"},
        "window": info.get("window"), "file_sha256": info.get("file_sha256"),
        "build": info.get("build"), "flags": info.get("flags"), "where": "this server",
        "docker_root": droot, "started_at": started, "by": a.by})
    reach = reach_targets(llama=host_address(info.get("base_url") or "")) if not a.oracle else []
    todo = [(t, k) for t in picked for k in range(1, a.attempts + 1)
            if state_of(rdir, t, k)[0] == "todo"]
    say(f"{b['label']}: {len(picked) * a.attempts - len(todo)} of {len(picked) * a.attempts} "
        f"done before this run; {len(todo)} to ask")
    running: dict = {}
    last_post = 0.0
    stop = ""
    biggest = NEXT_IMAGE_GB
    (rdir / "jobs").mkdir(parents=True, exist_ok=True)
    while todo or running:
        # a task finished: its images go, once no waiting task needs them
        for key, (proc, imgs, log) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[key]
            log.close()
            keep = {i for t, _ in todo for i in images_of(tdir / t)} | {
                i for _, (_, ii, _) in running.items() for i in ii}
            biggest = max(biggest, remove_images(imgs, keep, rdir))
            state, tries = state_of(rdir, *key)
            last = (results(rdir, [key[0]], key[1])[-1:] or [{}])[0]
            say(f"{key[0]} (attempt {key[1]}): " + (
                ab.RESULT_WORDS.get(last.get("result", ""), "no result") if state == "done"
                else f"an error of ours ({last.get('why') or 'no result'}) — "
                + ("asked again" if state == "todo" else f"given up after {tries}")))
            if state == "todo":
                todo.append(key)
        # nothing more is pulled when the disk would fall under the floor
        if not stop and todo:
            stop = disk_line(droot, biggest)
            if stop:
                say(f"stopping — {stop}; the tasks running finish")
                todo = []
        while todo and len(running) < max(1, a.at_once) and not stop:
            t, k = todo.pop(0)
            leftovers(t)
            job = f"{t}__a{k}__{int(time.time())}"
            cmd = harbor_cmd(tdir / t, job, rdir, b, a, info.get("model") or "",
                             relay.url if relay else "", reach)
            env = {**os.environ, "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH', '')}"}
            log = open(rdir / "jobs" / f"{job}.log", "w")
            proc = subprocess.Popen(cmd, cwd=rdir, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[(t, k)] = (proc, images_of(tdir / t), log)
        if not a.no_board and time.time() - last_post >= a.every:
            post(rdir, b, picked, a.attempts, started)
            last_post = time.time()
        time.sleep(2)
    if relay:
        relay.stop()
    p = progress(rdir, b, picked, a.attempts, started)
    say(f"{b['label']}: {p['line']}")
    if not a.no_board:
        post(rdir, b, picked, a.attempts, started)
        code, out = board(str(rdir))
        say(out.strip().splitlines()[-1] if out.strip() else f"import: exit {code}")
    return 1 if stop else 0


def post(rdir: Path, b: dict, tasks: list[str], attempts: int, started: float) -> None:
    """the run's line to the board, as a Runs row"""
    write_json(rdir / "progress.json", progress(rdir, b, tasks, attempts, started))
    code, out = board("--progress", str(rdir))
    if code != 0:
        say(f"the board's Runs row NOT updated — {out.strip()[-200:]}")


if __name__ == "__main__":
    raise SystemExit(main())
