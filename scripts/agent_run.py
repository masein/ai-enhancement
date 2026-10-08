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
- it carries on: a task with the model's result is never asked again, a
  failure included; an error of ours (Docker failing to pull, build or start
  the task's container, the container reaching something, the model's server
  failing its health check) is asked again, three times at most; a run
  killed at any point continues with the same command;
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
import shlex
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
MAX_REPLY = 32_768                      # the relay's cap on a reply (agent_relay.MAX_REPLY_TOKENS)
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


def run_dir(root: Path, name: str, model: str, attempts: int, oracle: bool,
            number: int = 1) -> Path:
    """a run's folder; run 2 and on of the same model sit beside the first"""
    who = "oracle" if oracle else slug(model.removeprefix("served/"))
    return root / "agent-runs" / (f"{name}__{who}__k{attempts}"
                                  + (f"-{number}" if number > 1 else ""))


# what a run is, which never changes once it started (18b point 12)
SETTINGS = ("benchmark", "dataset", "pin", "tasks_sha256", "attempts", "agent", "config", "model",
            "sampling", "window", "file_sha256", "build", "flags", "prompt_sha256", "lock_sha256")


def changed_settings(before: dict, now: dict) -> list[str]:
    """each setting a run's folder was started with that differs now"""
    def say_(v) -> str:
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else (
            v if isinstance(v, str) else json.dumps(v, sort_keys=True))[:60]
    out = [f"{k} {say_(before.get(k))} → {say_(now.get(k))}" for k in SETTINGS
           if k in before and before.get(k) != now.get(k)]
    vb, vn = before.get("versions") or {}, now.get("versions") or {}
    out += [f"{k} {vb.get(k)} → {vn.get(k)}" for k in ("harbor", "mini-swe-agent", "litellm")
            if k in vb and vb.get(k) != vn.get(k)]
    return out


# ---------------------------------------------------------------------------
# before the first task
# ---------------------------------------------------------------------------

LOCK = REPO / "docs" / "agent-requirements.txt"


def lock() -> dict[str, str]:
    """the agent venv's lock: every package and its version (18b point 5)"""
    out = {}
    try:
        for line in LOCK.read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
            if m:
                out[m.group(1).lower().replace("_", "-")] = m.group(2)
    except OSError:
        pass
    return out


def versions() -> dict:
    """the pinned versions, as installed — {} for one that isn't — and every
    locked package that differs from the lock"""
    from importlib import metadata
    out: dict = {}
    for pkg in ("harbor", "mini-swe-agent", "litellm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = ""
    off = []
    for pkg, want in lock().items():
        try:
            got = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            got = ""
        if got != want:
            off.append(f"{pkg} {got or 'missing'} (the lock: {want})")
    out["off_lock"] = off
    return out


def check_versions(v: dict, oracle: bool) -> str:
    if v.get("harbor") != ab.HARBOR_VERSION:
        return (f"Harbor {v.get('harbor') or 'isn’t installed'} — {ab.HARBOR_VERSION} is pinned: run "
                "this with the agent venv's python (docs/AGENT-RUNS.md § A)")
    if not oracle and v.get("mini-swe-agent") != ab.MINI_VERSION:
        return (f"mini-swe-agent {v.get('mini-swe-agent') or 'isn’t installed'} — "
                f"{ab.MINI_VERSION} is pinned (docs/AGENT-RUNS.md § A)")
    off = v.get("off_lock") or []
    if off:
        return (f"{len(off)} package{'s' if len(off) != 1 else ''} in the agent venv differ from "
                "its lock: " + "; ".join(off[:5]) + (" …" if len(off) > 5 else "")
                + " — make the venv again from the lock (docs/AGENT-RUNS.md § A, steps 2–4)")
    return ""


def docker_root() -> tuple[str, str]:
    """(where Docker keeps its images, why not) — Docker as this user"""
    code, out = run(["docker", "info", "-f", "{{.DockerRootDir}}"], timeout=30)
    if code != 0:
        return "", ("Docker doesn't answer this user — Harbor runs `docker` as you: run "
                    "this with sudo, or add yourself to the docker group (docs/AGENT-RUNS.md "
                    "§ A, step 5: each makes the run root) — " + out.strip()[-200:])
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
                + (f" and the next task needs about {need_gb:.0f}" if need_gb else "")
                + f" — {ab.DISK_FLOOR_GB:.0f} GB must stay free: remove images "
                "(`docker image prune -a`) or free space there")
    return ""


def board(*args: str) -> tuple[int, str]:
    return run([*BOARD, *args], cwd=REPO, timeout=600)


def served_info(model: str) -> tuple[dict, str]:
    """the board's record of the served model, its server's pin check and
    its window now — asked inside the board's container"""
    code, out = board("--served", model)
    info: dict = {}
    for line in reversed(out.strip().splitlines()):     # its JSON line, wherever it is
        try:
            got = json.loads(line)
        except ValueError:
            continue
        if isinstance(got, dict):
            info = got
            break
    if not info:
        return {}, f"the board didn't answer for {model} (exit {code}) — {quiet(out)}"
    if code != 0 or info.get("why"):
        return info, str(info.get("why") or f"the board answered exit {code} — {quiet(out)}")
    return info, ""


def quiet(out: str) -> str:
    """the board's words for a refusal line — never a line that could hold
    the server's key (its JSON, any line that names a key) (18b point 18)"""
    keep = [x for x in out.strip().splitlines()
            if not x.lstrip().startswith(("{", "[")) and "key" not in x.lower()]
    return " / ".join(keep)[-300:] or "no words"


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


# ---------------------------------------------------------------------------
# 18b points 14 and 16: --check asks the server's template and its cache
# ---------------------------------------------------------------------------

# mini-swe-agent's own bash tool, as it sends it
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {
        "type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}}


def three_steps() -> list[dict]:
    """an invented conversation of three steps, as mini-swe-agent keeps it:
    each reply its thinking and a tool call, each result a tool message"""
    msgs = [{"role": "system", "content": "You are a helpful assistant that can interact with a "
                                          "computer shell to solve programming tasks."},
            {"role": "user", "content": "Invented task: make the invented test pass. Start by "
                                        "listing the files."}]
    for i, (cmd, out) in enumerate([("ls", "a.c\nb.c\nMakefile"), ("cat a.c", "int main(){}"),
                                    ("make test", "1 passed")]):
        msgs += [{"role": "assistant", "content": "",
                  "reasoning_content": f"Thinking at step {i + 1}: run {cmd} next.",
                  "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {
                      "name": "bash", "arguments": json.dumps({"command": cmd})}}]},
                 {"role": "tool", "tool_call_id": f"call_{i}",
                  "content": f"<returncode>0</returncode>\n<output>\n{out}\n</output>"}]
    return msgs


def apply_template(base_url: str, key: str, body: dict) -> str:
    """the prompt llama-server's own template makes of a request"""
    import urllib.request
    root = base_url.rstrip("/").removesuffix("/v1")
    req = urllib.request.Request(root + "/apply-template", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return str(json.loads(r.read()).get("prompt") or "")


def template_line(base_url: str, key: str, apply=None) -> str:
    """'' when the server's template, thinking on, renders each step of a
    three-step conversation as it did before the next step came — each
    prompt the start of the next, so the server can reuse its cache — keeps
    every step's thinking, and leaves the thinking open; else why"""
    apply = apply or apply_template
    msgs = three_steps()
    try:
        renders = [apply(base_url, key, {"messages": msgs[:2 + 2 * k], "tools": [BASH_TOOL],
                                         "chat_template_kwargs": {"enable_thinking": True}})
                   for k in (1, 2, 3)]
    except Exception as e:                              # noqa: BLE001 — said in one line
        return f"its server didn't render the agent's conversation (/apply-template): {e}"
    if any(re.search(r"<think>\s*</think>\s*$", r) for r in renders):
        return "its template closes the thinking before the reply: thinking is off"
    for k in (1, 2):
        if not renders[k].startswith(renders[k - 1]):
            i = next((j for j, (x, y) in enumerate(zip(renders[k - 1], renders[k])) if x != y),
                     min(len(renders[k - 1]), len(renders[k])))
            return (f"its template renders the conversation differently once step {k + 1} "
                    f"follows (they part at {renders[k - 1][max(0, i - 30):i + 30]!r}): the "
                    "server can't reuse its cache, and every step would read the whole "
                    "conversation again")
    lost = [str(i + 1) for i in range(3) if f"Thinking at step {i + 1}:" not in renders[2]]
    if lost:
        return (f"its template leaves out the thinking of step {', '.join(lost)}: Qwen's own "
                "runs keep every step's thinking in a tool loop")
    return ""


INVENTED_TASK = "\n".join(
    f"{i}. The invented parser's option --part-{i} must keep the order of the parts it is given "
    f"and say which part {i} it read last; today it drops the last part when there are more "
    "than two, and its test for this fails." for i in range(1, 121)) + (
    "\n\nStart by listing the files of the repository.")


def first_messages(config: str) -> list[dict]:
    """the agent's first request: its config's system and task templates,
    with an invented task of a few thousand tokens"""
    import platform

    import agent_host_mini as hm
    from jinja2 import Template
    a = hm.load_config(config).get("agent") or {}
    vars_ = {**platform.uname()._asdict(), "task": INVENTED_TASK, "cwd": "/testbed",
             "timeout": 60}
    return [{"role": "system", "content": Template(str(a.get("system_template"))).render(**vars_)},
            {"role": "user", "content": Template(str(a.get("instance_template"))).render(**vars_)}]


def mini_ask(relay_url: str, model: str, messages: list[dict], trial: str) -> dict:
    """one step as the agent asks it: mini-swe-agent's own model class,
    through the relay — its reply as the agent keeps it"""
    from minisweagent.models.litellm_model import LitellmModel
    m = LitellmModel(model_name=f"openai/{model}", cost_tracking="ignore_errors", model_kwargs={
        "drop_params": True, "parallel_tool_calls": True, "api_base": relay_url,
        "api_key": "relay", "timeout": 1800, "max_tokens": 8192,
        "extra_headers": {"X-Agent-Trial": trial}})
    return m.query(messages)


CACHE_TRIAL = "check-cache"


def cache_line(relay, model: str, config: str, ask=None, first=None) -> tuple[str, dict]:
    """two requests that extend one conversation, sent as the agent sends
    them, through the relay: '' when the second took most of its prompt
    from the server's cache; and the speeds the server measured"""
    ask = ask or mini_ask
    msgs = first if first is not None else first_messages(config)
    try:
        r1 = ask(relay.url, model, msgs, CACHE_TRIAL)
        call = ((r1.get("tool_calls") or [{}])[0] or {}).get("id") or "call_0"
        ask(relay.url, model, [*msgs, r1, {
            "role": "tool", "tool_call_id": call,
            "content": "<returncode>0</returncode>\n<output>\nREADME.md\nsrc\ntests\n</output>"}],
            CACHE_TRIAL)
    except Exception as e:                              # noqa: BLE001 — said in one line
        return f"the cache check's requests failed: {str(e)[:240]}", {}
    recs = [x for x in relay.records if x.get("trial") == CACHE_TRIAL and x.get("status") == 200]
    if len(recs) < 2:
        return "the cache check's two requests weren't both answered", {}
    a, b = recs[-2:]
    if not all(k in b for k in ("cache_n", "prompt_n")):
        return ("its server didn't say what it took from its cache (llama-server's `timings`): "
                "the cache can't be checked"), {}
    speeds = {}
    if a.get("prompt_n") and a.get("prompt_ms"):
        speeds["prefill"] = 1000 * float(a["prompt_n"]) / float(a["prompt_ms"])
    out_n = sum(float(x.get("predicted_n") or 0) for x in (a, b))
    out_ms = sum(float(x.get("predicted_ms") or 0) for x in (a, b))
    if out_n and out_ms:
        speeds["decode"] = 1000 * out_n / out_ms
    whole = int(b["cache_n"]) + int(b["prompt_n"])
    speeds.update(reread=int(b["prompt_n"]), whole=whole)
    if whole and int(b["prompt_n"]) > whole / 2:
        return (f"the second step read {int(b['prompt_n']):,} of its {whole:,} prompt tokens "
                "again: the server didn't reuse its cache — each step would read the whole "
                "conversation again (50–150k tokens, with the experts on the CPU), and most "
                "tasks would hit the 50-minute limit"), speeds
    return "", speeds


def estimate_words(speeds: dict, tasks: int = 300) -> str:
    """the measured speeds, and what they make of a whole run"""
    pre, dec = speeds.get("prefill"), speeds.get("decode")
    if not pre or not dec:
        return ""

    def task_s(steps: int, new: int, out: int, cmd: float) -> float:
        return min(50 * 60, steps * (new / pre + out / dec + cmd))
    lo = tasks * task_s(30, 1500, 400, 5) / 86400
    hi = tasks * task_s(90, 3000, 1000, 15) / 86400
    return (f"it read {pre:,.0f} prompt tokens a second and wrote {dec:,.0f}: about "
            f"{lo:.1f}–{hi:.1f} days for {tasks} tasks, one at a time (30–90 steps a task; a step "
            "reads 1,500–3,000 new tokens, writes 400–1,000 and runs 5–15 s of commands; 50 "
            "minutes at most a task)")


def tailnet_dns() -> str:
    """the tailnet's own DNS address, the same on every tailnet — built here
    so no tracked file holds a tailnet address (the mirror is public)"""
    return ".".join(["100"] * 4)


def reach_targets(board_port: int = 8899, llama: str = "", relay_port: int = 0) -> list[str]:
    """what a task's container must never reach (18b point 8): the model's
    server, the board, the relay and ssh on every address of this host —
    IPv4 and IPv6, the tailnet's among them — and on Docker's bridge
    gateway; the LAN's gateway; Docker's own DNS; the tailnet's DNS; the
    internet over IPv4 and IPv6"""
    llama_host, llama_port = "", 8090
    m = re.match(r"https?://\[?([^/\]]+?)\]?(?::(\d+))?(?:/|$)", llama or "")
    if m:
        llama_host, llama_port = m.group(1), int(m.group(2) or 80)
    ports = [board_port, llama_port, 22] + ([relay_port] if relay_port else [])
    out = [f"{llama_host}:{llama_port}"] if llama_host else []
    code, addrs = run(["hostname", "-I"], timeout=10)
    hosts = [a for a in (addrs.split() if code == 0 else [])
             if re.fullmatch(r"\d+\.\d+\.\d+\.\d+|[0-9a-fA-F:]+", a)]
    code, gw = run(["docker", "network", "inspect", "bridge", "-f",
                    "{{(index .IPAM.Config 0).Gateway}}"], timeout=30)
    gw = gw.strip().splitlines()[-1] if code == 0 and gw.strip() else "172.17.0.1"
    for h in [*hosts, gw, "host.docker.internal"]:
        out += [f"{h}:{p}" for p in ports]
    code, route = run(["ip", "route", "show", "default"], timeout=10)
    m = re.search(r"via (\d+\.\d+\.\d+\.\d+)", route if code == 0 else "")
    if m:
        out.append(f"{m.group(1)}:80")
    out += ["127.0.0.11:53", f"{tailnet_dns()}:53", "1.1.1.1:443", "8.8.8.8:53",
            "2606:4700:4700::1111:443", "2001:4860:4860::8888:53"]
    if relay_port:
        out.append(f"127.0.0.1:{relay_port}")
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
    result is in, whatever it is (18b point 2: a failure of the model's is
    never asked again); an error of ours is asked again, TRIES times at most"""
    tries = attempts_of(rdir, task, k)
    if any(t["result"] != "error" for t in tries):
        return "done", len(tries)
    return ("given up" if len(tries) >= TRIES else "todo"), len(tries)


def results(rdir: Path, tasks: list[str], attempts: int) -> list[dict]:
    """each task's attempt: the model's first result — or, while it has
    only errors of ours, the last of them — with why it was asked again"""
    out = []
    for t in tasks:
        for k in range(1, attempts + 1):
            tries = attempts_of(rdir, t, k)
            if not tries:
                continue
            first = next((i for i, x in enumerate(tries) if x["result"] != "error"), None)
            final = tries[first] if first is not None else tries[-1]
            before = tries[:first] if first is not None else tries[:-1]
            out.append({**final, "attempt": k, "tries": len(tries),
                        "again": [x.get("why") or "an error of ours" for x in before]})
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
            "started_at": started, "at": time.time(), "minutes_each": each,
            "until": time.time() + 60 * left if left else None}


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


# the task's container — and DeepSWE's separate verifier's — no network at
# all, a process limit, no new privileges, and every capability dropped but
# the three Harbor's own steps need on the folders it mounts (it chmods and
# chowns them as root: CHOWN, DAC_OVERRIDE, FOWNER). Docker's others — MKNOD
# (device nodes in a mounted folder), NET_RAW, SETUID, SETGID, KILL,
# NET_BIND_SERVICE, SYS_CHROOT, SETFCAP, SETPCAP, FSETID, AUDIT_WRITE — come
# back only for a task that proves it needs one (18b point 16)
OVERRIDE_YAML = ("services:\n  main:\n    network_mode: none\n    pids_limit: 4096\n"
                 "    security_opt:\n      - no-new-privileges:true\n"
                 "    cap_drop:\n      - ALL\n"
                 "    cap_add:\n      - CHOWN\n      - DAC_OVERRIDE\n      - FOWNER\n")


def override(rdir: Path) -> Path:
    """the task's container's limits, merged over every task's own compose
    by Harbor"""
    p = rdir / "no-network.yaml"
    p.write_text(OVERRIDE_YAML)
    return p


# ---------------------------------------------------------------------------
# 18b point 7: a copy of each task whose verification needs no network
# ---------------------------------------------------------------------------

OFFLINE_MARK = ("# --- added by evalboard's agent runner (18b): verification runs with no "
                "network ---")


def eval_part(test_sh: str) -> list[str]:
    """the test commands of a SWE-bench task's test.sh: up to its parser"""
    out = []
    for line in test_sh.splitlines():
        if line.strip() == "cd ..":
            break
        out.append(line)
    return out


def fetch_lines(test_sh: str) -> list[str]:
    """its test commands that fetch packages"""
    return [x.strip() for x in eval_part(test_sh) if ab.FETCHES.match(x)]


def parser_head(test_sh: str) -> list[str]:
    """the inline-script header of the verifier's parser (its Python and
    packages), as test.sh writes it"""
    m = re.search(r"(?ms)^# /// script\n.*?^# ///$", test_sh)
    return m.group(0).splitlines() if m else []


def dockerfile_additions(test_sh: str) -> list[str]:
    """what a SWE-bench task's image gets so its verification needs no
    network — '' parts left out"""
    out = [OFFLINE_MARK]
    head = parser_head(test_sh)
    if head:
        script = " ".join(shlex.quote(x) for x in [*head, "pass"])
        out += ["# the verifier's parser's Python and packages, fetched now; uv offline after",
                f"RUN cd / && printf '%s\\n' {script} > parser.py && uv run parser.py "
                "&& rm -f parser.py",
                "ENV UV_OFFLINE=1"]
    lines = fetch_lines(test_sh)
    if lines:
        repo = next((m.group(1) for x in eval_part(test_sh)
                     if (m := re.match(r"\s*cd\s+(/\S+)\s*$", x))), "/testbed")
        tools = sorted({x.split()[0] for x in lines})
        steps = " ; ".join(f"( {x} )" for x in lines) + (" ; ( cargo fetch )"
                                                         if "cargo" in tools else "")
        env = sorted({ab.OFFLINE_ENV[t] for t in tools if t in ab.OFFLINE_ENV})
        out += ["# its tests' own package fetches, once, in a throwaway copy: the caches stay",
                f"RUN cp -a {repo} /tmp/evalboard-warm && cd /tmp/evalboard-warm && {steps} ; "
                "cd / && rm -rf /tmp/evalboard-warm",
                *([f"ENV {' '.join(env)}"] if env else [])]
    return out


def task_test(task: Path) -> str:
    try:
        return (task / "tests" / "test.sh").read_text(errors="replace")
    except OSError:
        return ""


def offline_tasks(root: Path, b: dict, tdir: Path, names: list[str]) -> tuple[Path, dict]:
    """a copy of each task asked, made again each run: a SWE-bench task's
    Dockerfile installs what its verification would fetch; a task verified
    in a separate container (DeepSWE) gives that container the agent's
    limits. The tasks as fetched stay as they are (their hash is the run's)"""
    out = root / "agent-tasks" / (slug(b["dataset"]) + "+offline")
    h = hashlib.sha256()
    fetching = []
    for name in names:
        src, dst = tdir / name, out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        test_sh = task_test(src)
        toml = (src / "task.toml").read_text(errors="replace")
        if re.search(r'(?m)^\s*environment_mode\s*=\s*"separate"', toml):
            (dst / "tests" / "docker-compose.yaml").write_text(OVERRIDE_YAML)
            added = OVERRIDE_YAML
        else:
            add = dockerfile_additions(test_sh)
            with open(dst / "environment" / "Dockerfile", "a", encoding="utf-8") as fh:
                fh.write("\n" + "\n".join(add) + "\n")
            added = "\n".join(add)
        if fetch_lines(test_sh):
            fetching.append(name)
        h.update(name.encode() + b"\0" + added.encode() + b"\0")
    return out, {"sha256": h.hexdigest(), "fetching": fetching}


def prompt_hash(config: str) -> str:
    """the agent's prompt — its config's system and instance templates — as
    installed in the venv"""
    try:
        import agent_host_mini as hm
        return hm.prompt_sha(hm.load_config(config))
    except Exception as e:                              # noqa: BLE001 — said, never fatal
        return f"unknown ({type(e).__name__})"


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


def remove_built(task: str) -> None:
    """what Harbor built for a task's trials (18b point 13): the images
    Docker Compose built for its projects (the task's container, and a
    separate verifier's), then the build cache no image uses any more —
    Docker keeps the cache of a removed image until it is pruned"""
    code, out = run(["docker", "image", "ls", "--filter", "label=com.docker.compose.project",
                     "--format", '{{.ID}} {{.Label "com.docker.compose.project"}}'], timeout=60)
    head = project_of(task[:32]) + "__"
    ids = sorted({x.split()[0] for x in out.splitlines() if code == 0 and len(x.split()) == 2
                  and x.split()[1].startswith(head)})
    if ids:
        run(["docker", "image", "rm", "-f", *ids], timeout=300)
    run(["docker", "builder", "prune", "-f"], timeout=900)


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


NEXT_IMAGE_GB = 8.0                     # a task's pull and build, until one has been measured


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
    ap.add_argument("--only", default="", help="these tasks, by name, comma-separated; "
                    "`fetching`: those whose own tests fetch packages")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--run", type=int, default=1, help="run 2, 3…: another run of the same model "
                    "beside the first, with its own settings")
    ap.add_argument("--at-once", type=int, default=1, help="tasks at a time (default 1)")
    ap.add_argument("--every", type=float, default=EVERY_S, help="seconds between progress posts")
    ap.add_argument("--no-board", action="store_true", help="nothing goes to the board")
    ap.add_argument("--by", default=os.environ.get("SUDO_USER") or os.environ.get("USER", ""),
                    help="your name, on its Runs row (under sudo: the user who ran sudo)")
    ap.add_argument("--check", action="store_true", help="the checks before the first task, only")
    ap.add_argument("--check-reach", action="store_true",
                    help="step A: a container as a task's tries the model's server, the board, "
                         "the host's addresses, the LAN and the internet — nothing may answer")
    a = ap.parse_args(argv)
    b = ab.bench(a.benchmark)
    if a.check_reach:                   # no model: step A (18b point 4)
        root = bench_root()
        rdir = root / "agent-runs" / "reach-check"
        rdir.mkdir(parents=True, exist_ok=True)
        targets = reach_targets(llama=host_address("http://host.docker.internal:8090/v1"))
        why = check_reach(rdir, targets)
        say(f"refused — {why}" if why else "a task's container reaches nothing of: "
            + ", ".join(targets) + "; no name resolves; its only interface is the loopback; "
            "no host folder; at most 4,096 processes")
        return 2 if why else 0
    if not a.oracle and not a.model.startswith("served/"):
        ap.error("--as served/<model> (or --oracle, or --check-reach, for step A)")
    root = bench_root()
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
    rdir = run_dir(root, a.benchmark, a.model, a.attempts, a.oracle, a.run)
    rdir.mkdir(parents=True, exist_ok=True)
    tdir, tasks_sha = fetch_tasks(root, b)
    langs = languages(tdir, b)
    picked = ab.pilot(langs, a.tasks) if a.tasks else sorted(langs)
    if a.only == "fetching":            # the tasks whose own tests fetch packages
        picked = [t for t in sorted(langs) if fetch_lines(task_test(tdir / t))]
    elif a.only:
        unknown = [t for t in a.only.split(",") if t and t not in langs]
        if unknown:
            say(f"refused — no such task in {b['label']}: {', '.join(unknown[:5])}")
            return 2
        picked = [t for t in a.only.split(",") if t]
    # each asked task's copy whose verification needs no network (18b point 7)
    tdir, offline = offline_tasks(root, b, tdir, picked)
    if offline["fetching"]:
        say(f"{len(offline['fetching'])} of these tasks fetch packages in their tests: those "
            "fetches run once while the image is built, and the tools are told they are "
            "offline — " + ", ".join(offline["fetching"]))
    # what this run is: its settings never change once it started (18b
    # point 12) — the window the model's server reports, its file and build
    now = {
        "benchmark": a.benchmark, "label": b["label"], "dataset": b["dataset"], "pin": b["pin"],
        "tasks_sha256": tasks_sha, "of": b["tasks"], "attempts": a.attempts,
        "at_once": a.at_once, "agent": "oracle" if a.oracle else b["agent"],
        "config": None if a.oracle else b["config"], "model": a.model or None,
        "versions": {k: x for k, x in v.items() if k != "off_lock"},
        "sampling": None if a.oracle else {**ab.SAMPLING, "thinking": "on"},
        "max_reply_tokens": None if a.oracle else MAX_REPLY,
        "window": info.get("window"), "file_sha256": info.get("file_sha256"),
        "build": info.get("build"), "flags": info.get("flags"), "where": "this server",
        "docker_root": droot, "by": a.by, "offline": offline,
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.exists() else None,
        "prompt_sha256": None if a.oracle else prompt_hash(b["config"])}
    before = read_json(rdir / "run.json")
    if before.get("benchmark"):
        diff = changed_settings(before, now)
        if diff:
            say(f"refused — this run was started with other settings ({'; '.join(diff)}): a "
                "run's settings never change. Finish it with its own, or start another run "
                f"beside it: add --run {a.run + 1}")
            return 2
    relay = None
    if not a.oracle:
        import agent_relay
        relay = agent_relay.Relay(host_address(info["base_url"]), info.get("key") or "",
                                  {**ab.SAMPLING, "chat_template_kwargs": {"enable_thinking": True}},
                                  usage=rdir / "relay-usage.jsonl").start()
        why = tool_call_line(relay.url, info.get("model") or "")
        if not why and a.check:
            # the server's template, and whether it reuses its cache (18b 14, 16)
            why = template_line(host_address(info["base_url"]), info.get("key") or "")
            if not why:
                why, speeds = cache_line(relay, info.get("model") or "", b["config"])
                words = estimate_words(speeds, b["tasks"])
                if words:
                    say(words)
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
    # the run's tasks: those asked before and these (a pilot grown, step A's
    # oracle on more tasks), each kept with its result; its first settings kept
    every = [*(before.get("tasks") or []),
             *[t for t in picked if t not in (before.get("tasks") or [])]]
    started = before.get("started_at") or time.time()
    write_json(rdir / "run.json", {**now, **{k: x for k, x in before.items() if k in SETTINGS},
                                   "tasks": every, "started_at": started})
    reach = (reach_targets(llama=host_address(info.get("base_url") or ""),
                           relay_port=relay.server.server_address[1] if relay else 0)
             if not a.oracle else [])
    todo = [(t, k) for t in picked for k in range(1, a.attempts + 1)
            if state_of(rdir, t, k)[0] == "todo"]
    say(f"{b['label']}: {len(picked) * a.attempts - len(todo)} of {len(picked) * a.attempts} "
        f"done before this run; {len(todo)} to ask")
    running: dict = {}
    last_post = 0.0
    stop = ""
    # 18b point 13: what a task takes of Docker's disk at its peak — its pull,
    # its build and its build cache — measured while it runs, and asked for
    # before the next one is started (its pull and build come first)
    need = NEXT_IMAGE_GB
    peaks: list[float] = []
    lowest: dict = {}
    (rdir / "jobs").mkdir(parents=True, exist_ok=True)
    while todo or running:
        now_free = free_gb(droot) if running else 0.0
        for key in running:
            lowest[key] = (lowest[key][0], min(lowest[key][1], now_free))
        # a task finished: what Harbor built goes, its images once no waiting
        # task needs them, and the build cache
        for key, (proc, imgs, log) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[key]
            log.close()
            keep = {i for t, _ in todo for i in images_of(tdir / t)} | {
                i for _, (_, ii, _) in running.items() for i in ii}
            remove_built(key[0])
            remove_images(imgs, keep, rdir)
            start, low = lowest.pop(key, (0.0, 0.0))
            if start - low > 0.5:
                peaks.append(start - low)
                need = max(peaks)
            state, tries = state_of(rdir, *key)
            last = (results(rdir, [key[0]], key[1])[-1:] or [{}])[0]
            say(f"{key[0]} (attempt {key[1]}): " + (
                ab.RESULT_WORDS.get(last.get("result", ""), "no result") if state == "done"
                else f"an error of ours ({last.get('why') or 'no result'}) — "
                + ("asked again" if state == "todo" else f"given up after {tries}")))
            if state == "todo":
                todo.append(key)
        while todo and len(running) < max(1, a.at_once) and not stop:
            # nothing more is pulled or built when the disk would fall under
            # the floor — checked before each task starts
            stop = disk_line(droot, need * (len(running) + 1))
            if stop:
                say(f"stopping — {stop}; the tasks running finish")
                todo = []
                break
            t, k = todo.pop(0)
            leftovers(t)
            # a new job each time: Harbor resumes a job whose name it has
            # seen, so a try asked again within the second would not run
            job = f"{t}__a{k}__{int(time.time() * 1000)}"
            while (rdir / "jobs" / job).exists():
                job = f"{t}__a{k}__{int(job.rsplit('__', 1)[1]) + 1}"
            cmd = harbor_cmd(tdir / t, job, rdir, b, a, info.get("model") or "",
                             relay.url if relay else "", reach)
            env = {**os.environ, "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH', '')}"}
            log = open(rdir / "jobs" / f"{job}.log", "w")
            proc = subprocess.Popen(cmd, cwd=rdir, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[(t, k)] = (proc, images_of(tdir / t), log)
            f = free_gb(droot)
            lowest[(t, k)] = (f, f)
        if not a.no_board and time.time() - last_post >= a.every:
            post(rdir, b, every, a.attempts, started)
            last_post = time.time()
        time.sleep(2)
    if relay:
        relay.stop()
    p = progress(rdir, b, every, a.attempts, started)
    say(f"{b['label']}: {p['line']}")
    if not a.no_board:
        post(rdir, b, every, a.attempts, started)
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
