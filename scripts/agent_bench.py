"""18: the agent benchmarks — each a few lines of data, not new code
(docs/AGENT-RUNS-design.md). What a benchmark is, where Harbor gets its tasks,
the agent's config, a pilot's fixed tasks, and how a task's result is read.

Used on the server by scripts/agent_run.py (in the agent venv) and on the
board by scripts/import_agent.py and the page — standard library only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HARBOR_VERSION = "0.24.0"
MINI_VERSION = "2.4.6"
# decision 4 (8 Oct): 262,144 if it fits beside the judge with one conversation;
# the runner refuses anything under this
MIN_WINDOW = 131_072
WANT_WINDOW = 262_144
# decision 5: Qwen's own settings for its SWE-bench numbers; no presence penalty
# (it punishes code for repeating its own names)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
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
# a task's result, from Harbor's trial folder
# ---------------------------------------------------------------------------

# what isn't the model's: run again, never counted
OURS = ("DockerError", "DockerException", "EnvironmentStartTimeoutError",
        "VerifierTimeoutError", "RelayDown", "ReachRefused", "DiskGuard", "RewardTampered",
        "APIConnectionError", "ServiceUnavailableError", "InternalServerError",
        "Timeout", "ConnectionError", "RuntimeError")
# the model's: counted, as each benchmark counts them
COUNTED_EXITS = {"ContextWindowExceededError": "the window was outgrown",
                 "LimitsExceeded": "the step limit",
                 "RepeatedFormatError": "three malformed replies in a row"}


def read_trial(trial: Path) -> dict | None:
    """one task's attempt, as the board keeps it — None while it runs (no
    result.json yet). result: resolved, unresolved, timeout or error (ours,
    with why)"""
    try:
        r = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    exc = r.get("exception_info") or {}
    etype = str(exc.get("exception_type") or "")
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward", next(iter(rewards.values()), None)) if rewards else None
    ctx = r.get("agent_result") or {}
    meta = ctx.get("metadata") or {}
    exit_status = str(meta.get("exit_status") or "")

    def minutes(a, b):
        from datetime import datetime
        try:
            return round((datetime.fromisoformat(b) - datetime.fromisoformat(a))
                         .total_seconds() / 60, 1)
        except (TypeError, ValueError):
            return None
    out = {"task": r.get("task_name") or trial.name.split("__")[0],
           "trial": trial.name, "steps": meta.get("steps"),
           "tokens_in": ctx.get("n_input_tokens"), "tokens_out": ctx.get("n_output_tokens"),
           "minutes": minutes(r.get("started_at"), r.get("finished_at")),
           "exit": exit_status, "why": ""}
    if etype == "AgentTimeoutError":
        out.update(result="timeout", why=str(exc.get("exception_message") or "")[:300])
    elif etype or meta.get("ours"):
        out.update(result="error", why=(f"{etype}: " if etype else "")
                   + str(exc.get("exception_message") or meta.get("ours") or "")[:300])
    elif reward is None:
        out.update(result="error", why="the verifier gave no result")
    else:
        out["result"] = "resolved" if float(reward) >= 1 else "unresolved"
        if out["result"] == "unresolved" and exit_status in COUNTED_EXITS:
            out["why"] = COUNTED_EXITS[exit_status]
    return out


def score(results: list[dict], total: int) -> dict:
    """% resolved ± its standard error over the tasks with a result of the
    model's (an error of ours is no result), and whether it is a part-run"""
    counted = [r for r in results if r["result"] in ("resolved", "unresolved", "timeout")]
    n = len(counted)
    k = sum(1 for r in counted if r["result"] == "resolved")
    p = k / n if n else 0.0
    se = (p * (1 - p) / n) ** 0.5 if n else 0.0
    return {"resolved": k, "n": n, "of": total, "score": round(100 * p, 1),
            "se": round(100 * se, 1), "part": n < total,
            "errors": sum(1 for r in results if r["result"] == "error"),
            "timeouts": sum(1 for r in results if r["result"] == "timeout")}


def score_words(s: dict, total: int) -> str:
    """"% resolved ± its error · N tasks"; a part-run says so"""
    head = f"{s['score']:.1f}% resolved ± {s['se']:.1f} · {s['n']} task{'s' if s['n'] != 1 else ''}"
    return head + (f" · pilot: {s['n']} of {total}" if s["part"] else "")


def conversation(trial: Path) -> dict:
    """a finished task's conversation, step by step, for the board: what it
    thought, what it said, what it ran and what came back — from
    mini-swe-agent's trajectory; then the patch and the verifier's output"""
    def text(p: Path, cap: int = 2_000_000) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")[:cap]
        except OSError:
            return ""
    try:
        data = json.loads((trial / "agent" / "mini-swe-agent.trajectory.json").read_text())
    except (OSError, ValueError):
        data = {}
    steps: list[dict] = []
    for m in data.get("messages") or []:
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
    return {"steps": steps, "patch": text(trial / "agent" / "patch.diff"),
            "verifier": text(trial / "verifier" / "test-stdout.txt", 200_000)
            + text(trial / "verifier" / "test-stderr.txt", 50_000)}
