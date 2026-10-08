"""18: the agent benchmarks — each a few lines of data, not new code
(docs/AGENT-RUNS-design.md). What a benchmark is, where Harbor gets its tasks,
the agent's config, a pilot's fixed tasks, and how a task's result is read.

Used on the server by scripts/agent_run.py (in the agent venv) and on the
board by scripts/import_agent.py and the page — standard library only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from pathlib import Path

HARBOR_VERSION = "0.24.0"
MINI_VERSION = "2.4.6"
# decision 4 (8 Oct): 262,144 if it fits beside the judge with one conversation;
# the runner refuses anything under this
MIN_WINDOW = 131_072
WANT_WINDOW = 262_144
# decision 5: Qwen's own settings for its SWE-bench numbers; no presence penalty
# (it punishes code for repeating its own names); min_p 0, Qwen's own too
# (llama-server's default is 0.05: 18b point 16)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0}
# 50 GB free at Docker's root before anything more is pulled
DISK_FLOOR_GB = 50.0

BENCHES: dict[str, dict] = {
    "swebench-multilingual": {
        "label": "SWE-bench Multilingual",
        "dataset": "swebench_multilingual@1.0",
        "pin": "laude-institute/harbor-datasets@61366953049fd63a7dff54f67193bf39f5d1cde6",
        "tasks": 300,
        "languages": "C, C++, Go, Java, JavaScript/TypeScript, PHP, Ruby and Rust",
        "line": "Fix a real issue in an open-source repository in one of nine languages; the "
                "repository's own tests decide.",
        "marked": "resolved when every test the fix should make pass passes and none that "
                  "passed breaks — the SWE-bench rule; one attempt",
        "by": "SWE-bench (Princeton, Stanford and others)",
        "url": "https://www.swebench.com/multilingual.html",
        "licence": "MIT",
        # the board's agent: mini-swe-agent's own SWE-bench config
        "config": "benchmarks/swebench.yaml",
        "cwd": "/testbed",
        "cost_limit": 3.0,
        "agent": "mini-swe-agent 2.4.6, one bash tool, its SWE-bench config (250 steps, 60 s a "
                 "command) — the board's",
        "limits": "50 minutes a task for the agent, 50 for the tests",
        "language_from": "tags",
    },
    "deepswe": {
        "label": "DeepSWE 1.1",
        # Harbor's Hub shows no revision to pin: its tasks at a commit of
        # their repository (26 Aug 2026: the 10,800 s agent limit)
        "dataset": "deep-swe@0b9fabbb63b9",
        "git": "https://github.com/datacurve-ai/deep-swe.git",
        "commit": "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea",
        "subdir": "tasks",
        "pin": "datacurve-ai/deep-swe@0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea",
        "tasks": 113,
        "languages": "TypeScript, Go, Python, JavaScript and Rust",
        "line": "Long tasks written from scratch in active open-source repositories; a "
                "program checks what the change does, in a clean copy of the repository.",
        "marked": "its verifier passes or fails each task; its board runs the suite four times "
                  "— we run it once first",
        "by": "Datacurve",
        "url": "https://deepswe.datacurve.ai",
        "licence": "Apache-2.0",
        # its board's agent: mini-swe-agent's default config, no cost limit, the
        # task's instruction as the task (Pier's mini-swe-agent)
        "config": "mini.yaml",
        "cwd": "",
        "cost_limit": 0,
        "agent": "mini-swe-agent 2.4.6, one bash tool, its default config with no step or cost "
                 "limit — as DeepSWE's board runs it (it pins 2026-05-21's commit, the same "
                 "prompts)",
        "limits": "3 hours a task for the agent, 30 minutes for the verifier",
        "language_from": "metadata",
    },
}
ORDER = ["swebench-multilingual", "deepswe"]

# 18b point 7: verification runs with no network, as the agent does — it runs
# the model's code. What a benchmark's verification would fetch is installed
# while the task's image is built, in a copy of each task the runner makes
# (agent_run.offline_tasks):
# - SWE-bench Multilingual verifies in the agent's container. Its test.sh
#   runs `uv run parser.py`, whose inline dependencies (swebench, datasets
#   and a Python ≥ 3.11) uv fetches from PyPI. 18c point 2: the runner
#   fetches them once (docs/agent-build.json: a lock 14 days old, each file
#   checked against its hash), and the image installs them with Docker's
#   build network off; uv is then told it is offline. A few tasks' own test
#   commands run a package manager (npm, composer, cargo): those lines run
#   once at build time in a throwaway copy of /testbed, with Docker's own
#   network and no package's own scripts, filling the tools' caches, and
#   the tools are told they are offline.
# - DeepSWE verifies in a separate container built from the task's tests/
#   (its "clean copy"), offline by its own design (every task says
#   network_mode "no-network"); Harbor gives that container none of the
#   runner's compose files, so the copy adds its own: no network, the same
#   limits as the agent's.
# A line run at build time is the task's own: nothing of a task is in this
# repository.
FETCHES = re.compile(r"^\s*((npm|pnpm) (i|install|ci)\b|yarn( install)?\s*$|composer (install|update)"
                     r"\b|bundle install\b|cargo (update|fetch)\b|pip3? install\b|go (mod download|get)"
                     r"\b|gem install\b)")
OFFLINE_ENV = {"npm": "npm_config_offline=true", "pnpm": "npm_config_offline=true",
               "yarn": "YARN_ENABLE_OFFLINE_MODE=1", "composer": "COMPOSER_DISABLE_NETWORK=1",
               "cargo": "CARGO_NET_OFFLINE=true", "pip": "PIP_NO_INDEX=1",
               "pip3": "PIP_NO_INDEX=1", "go": "GOFLAGS=-mod=mod GOPROXY=off"}

# the words a task's result is said in, everywhere
RESULT_WORDS = {"resolved": "Resolved", "unresolved": "Not resolved", "timeout": "Timed out",
                "error": "Error (ours)"}


def bench(name: str) -> dict:
    if name not in BENCHES:
        raise SystemExit(f"{name}: no such benchmark — "
                         + ", ".join(ORDER) + " (SWE-bench Pro waits until both have run)")
    return BENCHES[name]


# ---------------------------------------------------------------------------
# a task's language, and a pilot's fixed tasks
# ---------------------------------------------------------------------------

_LANGS = ("c", "c++", "cpp", "go", "java", "javascript", "typescript", "js", "ts", "php",
          "ruby", "rust", "python")


def language_of(task_toml: str, how: str) -> str:
    """a task's language from its task.toml: Multilingual's tags list it,
    DeepSWE's metadata names it. '' when it doesn't say"""
    if how == "metadata":
        m = re.search(r'(?m)^\s*language\s*=\s*"([^"]+)"', task_toml)
        if m:
            return m.group(1).strip().lower()
    m = re.search(r"(?m)^\s*tags\s*=\s*\[([^\]]*)\]", task_toml)
    if m:
        tags = [t.strip().strip("'\"").lower() for t in m.group(1).split(",")]
        for t in reversed(tags):
            if t in _LANGS:
                return t
    return ""


def pilot(tasks: dict[str, str], n: int) -> list[str]:
    """`n` tasks spread over the languages, the same every time: each
    language's tasks in a fixed order (by a hash of the task's name), then
    one of each language in turn. `tasks`: {name: language}"""
    by: dict[str, list[str]] = {}
    for name, lang in tasks.items():
        by.setdefault(lang or "?", []).append(name)
    for lang in by:
        by[lang].sort(key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    langs = sorted(by)
    out: list[str] = []
    k = 0
    while len(out) < min(n, len(tasks)):
        for lang in langs:
            if k < len(by[lang]) and len(out) < n:
                out.append(by[lang][k])
        k += 1
    return out


# ---------------------------------------------------------------------------
# the trial's files: ours written where no container reaches, theirs read
# without following anything they planted (18b point 1)
# ---------------------------------------------------------------------------

# Harbor bind-mounts a trial's agent/, verifier/ and artifacts/ folders into
# the task's container, where the model's commands run as root: anything in
# them may be a link, a FIFO or a device it made. The agent's own files go in
# this folder of the trial's instead, which no container mounts.
HOST_DIR = "agent-host"
TRAJECTORY = "mini-swe-agent.trajectory.json"
META = "meta.json"
PATCH = "patch.diff"
READ_CAP = 2_000_000                    # bytes of one file the board shows


def safe_write(folder: Path, name: str, data: str | bytes) -> None:
    """one of our files: its folder made by us and never a link, the file
    written under a fresh name opened with O_NOFOLLOW, then renamed into
    place (a rename replaces a link; it never writes through one)"""
    if not name or name in (".", "..") or "/" in name:
        raise ValueError(f"not a file name: {name!r}")
    try:
        os.mkdir(folder, 0o755)
    except FileExistsError:
        pass
    st = os.lstat(folder)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise OSError(f"{folder} isn't a folder of ours: not written")
    dfd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        tmp = f".{name}.{os.getpid()}.{threading.get_ident()}.part"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644,
                     dir_fd=dfd)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data.encode("utf-8") if isinstance(data, str) else data)
        os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
    finally:
        os.close(dfd)


def safe_read(base: Path, rel: str, cap: int = READ_CAP) -> str:
    """a file for the board, '' unless it is a regular file with one link,
    reached from `base` through real folders only (no link followed, no
    '..'), so its real path is inside `base`; at most `cap` bytes. A FIFO, a
    device, a link or a hard link is never opened as a file"""
    parts = Path(rel).parts
    if not parts or any(p in ("", ".", "..", "/") for p in parts):
        return ""
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        fd = os.open(base, flags | os.O_DIRECTORY)
    except OSError:
        return ""
    try:
        for p in parts[:-1]:
            nxt = os.open(p, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = nxt
        st = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            return ""
        f = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=fd)
        try:
            st = os.fstat(f)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                return ""
            out, left = [], cap
            while left > 0:
                chunk = os.read(f, min(left, 1 << 20))
                if not chunk:
                    break
                out.append(chunk)
                left -= len(chunk)
            return b"".join(out).decode("utf-8", errors="replace")
        finally:
            os.close(f)
    except OSError:
        return ""
    finally:
        os.close(fd)


def safe_json(base: Path, rel: str, cap: int = READ_CAP) -> dict:
    try:
        got = json.loads(safe_read(base, rel, cap) or "{}")
        return got if isinstance(got, dict) else {}
    except ValueError:
        return {}


# ---------------------------------------------------------------------------
# a task's result, from Harbor's trial folder
# ---------------------------------------------------------------------------

# An error of ours is asked again; everything else is the model's result
# (18b point 2). Ours is an allow-list: Docker failing to pull, build or start
# the task's container (before the agent ran, or Harbor's own Docker steps),
# the task's container reaching something (ReachRefused), the model's server
# failing its health check (ServerDown, said by the relay). The disk guard
# stops the run before a task starts, so it leaves no trial.
OURS_TYPES = ("ReachRefused", "ServerDown", "EnvironmentStartTimeoutError", "AddTestsDirError",
              "DownloadVerifierDirError")
DOCKER_FAILED = "Docker compose command failed"
# the model's: not resolved, with why, as each benchmark counts them
MODELS_WHY = {"ContextWindowExceededError": "the window was outgrown",
              "LimitsExceeded": "the step limit",
              "RepeatedFormatError": "three malformed replies in a row",
              "VerifierTimeoutError": "the tests ran past their time limit",
              "InternalServerError": "the server refused a reply (HTTP 500)",
              "RewardFileNotFoundError": "the tests left no result",
              "RewardFileEmptyError": "the tests left an empty result",
              "VerifierOutputParseError": "the tests' result couldn't be read"}


def ours_why(r: dict, meta: dict) -> str:
    """why a trial's failure is ours, '' when it is the model's"""
    if meta.get("ours"):
        return str(meta["ours"])[:300]
    exc = r.get("exception_info") or {}
    etype = str(exc.get("exception_type") or "")
    msg = str(exc.get("exception_message") or "")
    if not etype:
        return ""
    started = (r.get("agent_execution") or {}).get("started_at")
    if etype in OURS_TYPES or msg.startswith(DOCKER_FAILED) or not started:
        return f"{etype}: {msg}"[:300]
    return ""


def read_trial(trial: Path) -> dict | None:
    """one task's attempt, as the board keeps it — None while it runs (no
    result.json yet). result: resolved, unresolved or timeout (the model's),
    or error (ours, with why). Resolved only when the agent submitted and the
    verifier passed it (18b point 10): a working tree it never submitted is
    not resolved, as mini-swe-agent's own numbers count it"""
    r = safe_json(trial, "result.json", 20_000_000)
    if not r:
        return None
    meta = safe_json(trial, f"{HOST_DIR}/{META}")
    exc = r.get("exception_info") or {}
    etype = str(exc.get("exception_type") or "")
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward", next(iter(rewards.values()), None)) if rewards else None
    ctx = r.get("agent_result") or {}
    exit_status = str(meta.get("exit_status") or (ctx.get("metadata") or {}).get("exit_status")
                      or "")
    oracle = str((r.get("agent_info") or {}).get("name") or "") == "oracle"

    def minutes(a, b):
        from datetime import datetime
        try:
            return round((datetime.fromisoformat(b) - datetime.fromisoformat(a))
                         .total_seconds() / 60, 1)
        except (TypeError, ValueError):
            return None
    out = {"task": r.get("task_name") or trial.name.split("__")[0],
           "trial": trial.name,
           "steps": meta.get("steps", (ctx.get("metadata") or {}).get("steps")),
           "tokens_in": meta.get("tokens_in", ctx.get("n_input_tokens")),
           "tokens_out": meta.get("tokens_out", ctx.get("n_output_tokens")),
           "last_prompt": meta.get("last_prompt"),
           "minutes": minutes(r.get("started_at"), r.get("finished_at")),
           "exit": exit_status, "why": ""}
    ours = ours_why(r, meta)
    if etype == "CleanupFailed":
        # 18c point 3: never verified; the model had run, so it isn't asked
        # again — an error of ours, counted not resolved
        out.update(result="error", final=True,
                   why=str(exc.get("exception_message") or "the clean-up after the agent didn't "
                           "run to its end")[:240] + " — not asked again: the model had run")
    elif ours:
        out.update(result="error", why=ours)
    elif etype == "AgentTimeoutError":
        out.update(result="timeout", why=str(exc.get("exception_message") or "")[:300])
    elif etype:
        out.update(result="unresolved", why=MODELS_WHY.get(etype) or MODELS_WHY.get(exit_status)
                   or f"{etype}: {str(exc.get('exception_message') or '')[:240]}")
    elif reward is None:
        out.update(result="unresolved", why="the verifier gave no result")
    elif float(reward) < 1:
        out.update(result="unresolved", why=MODELS_WHY.get(exit_status, ""))
    elif oracle or exit_status == "Submitted":
        out["result"] = "resolved"
    else:
        out.update(result="unresolved", why="it never submitted"
                   + (f" ({MODELS_WHY.get(exit_status, exit_status)})" if exit_status else "")
                   + ": the tests passing on its working tree don't count")
    return out


def score(results: list[dict], total: int) -> dict:
    """% resolved ± one standard error over every task attempted (18b point
    2: a task never leaves the denominator — an error of ours not yet asked
    again counts as not resolved, and is said), and whether it is a part-run"""
    n = len(results)
    k = sum(1 for r in results if r["result"] == "resolved")
    p = k / n if n else 0.0
    se = (p * (1 - p) / n) ** 0.5 if n else 0.0
    return {"resolved": k, "n": n, "of": total, "score": round(100 * p, 1),
            "se": round(100 * se, 1), "part": n < total,
            "errors": sum(1 for r in results if r["result"] == "error"),
            "timeouts": sum(1 for r in results if r["result"] == "timeout"),
            "again": sum(1 for r in results if r.get("again"))}


def score_words(s: dict, total: int) -> str:
    """"% resolved ± one standard error · N tasks"; a part-run says so, and
    errors of ours counted as not resolved are said"""
    head = (f"{s['score']:.1f}% resolved ± {s['se']:.1f} (one standard error) · {s['n']} "
            f"task{'s' if s['n'] != 1 else ''}")
    tail = f" · pilot: {s['n']} of {total}" if s["part"] else ""
    if s.get("errors"):
        tail += (f" · {s['errors']} error{'s' if s['errors'] != 1 else ''} of ours, counted "
                 "not resolved")
    return head + tail


def again_kind(why: str) -> str:
    """an error of ours, in a few words"""
    t = (why or "").split(":", 1)[0]
    return {"ReachRefused": "its container reached something",
            "ServerDown": "the model's server was down"}.get(
        t, "Docker couldn't pull, build or start its container")


def again_words(results: list[dict]) -> str:
    """how many tasks were asked again after an error of ours, and why:
    "2 tasks run again: Docker couldn't … ×2 · the model's server was down ×1" """
    count: dict[str, int] = {}
    for r in results:
        for w in r.get("again") or []:
            count[again_kind(w)] = count.get(again_kind(w), 0) + 1
    n = sum(1 for r in results if r.get("again"))
    if not n:
        return ""
    return (f"{n} task{'s' if n != 1 else ''} run again after an error of ours: "
            + " · ".join(f"{k} ×{v}" for k, v in sorted(count.items(), key=lambda x: -x[1])))


def conversation(rdir: Path, job: str, trial: str) -> dict:
    """a finished task's conversation, step by step, for the board: what it
    thought, what it said, what it ran and what came back — from
    mini-swe-agent's trajectory; then the patch and the verifier's output.
    Every file is read through safe_read, from the run's folder down"""
    t = f"jobs/{job}/{trial}"
    try:
        data = json.loads(safe_read(rdir, f"{t}/{HOST_DIR}/{TRAJECTORY}", 50_000_000) or "{}")
    except ValueError:
        data = {}
    steps: list[dict] = []
    for m in (data.get("messages") or []) if isinstance(data, dict) else []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            cmds = []
            for c in m.get("tool_calls") or []:
                f = (c or {}).get("function") or {}
                try:
                    cmds.append(str(json.loads(f.get("arguments") or "{}").get("command", "")))
                except (ValueError, AttributeError):
                    cmds.append(str(f.get("arguments") or ""))
            steps.append({"thought": str(m.get("reasoning_content") or ""),
                          "said": str(m.get("content") or ""), "ran": cmds, "came_back": []})
        elif role in ("tool", "user") and steps and "ran" in steps[-1]:
            steps[-1]["came_back"].append(str(m.get("content") or ""))
        elif role == "exit":
            steps.append({"exit": str((m.get("extra") or {}).get("exit_status") or "")})
    return {"steps": steps, "patch": safe_read(rdir, f"{t}/{HOST_DIR}/{PATCH}"),
            "verifier": safe_read(rdir, f"{t}/verifier/test-stdout.txt", 200_000)
            + safe_read(rdir, f"{t}/verifier/test-stderr.txt", 50_000)}
