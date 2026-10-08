"""18's tests' world: invented tasks and Harbor trial folders, as Harbor 0.24.0
and mini-swe-agent 2.4.6 write them — no benchmark's task, solution,
verifier or trajectory is here, and nothing is fetched."""

from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

import agent_bench as ab

LANGS = ["c", "go", "java", "javascript", "php", "ruby", "rust", "typescript"]
MODEL = "served/invented-agent"


def tasks_folder(d: Path, n: int = 40, how: str = "tags") -> list[str]:
    """`n` invented tasks, their languages in their task.toml as each
    benchmark says them"""
    names = []
    for i in range(n):
        name = f"inv__repo{i % 8}-{1000 + i}"
        t = d / name
        (t / "environment").mkdir(parents=True, exist_ok=True)
        lang = LANGS[i % len(LANGS)]
        meta = (f'[metadata]\ntags = ["debugging", "swe-bench", "{lang}"]\n' if how == "tags"
                else f'[metadata]\nlanguage = "{lang}"\n')
        (t / "task.toml").write_text(meta + "[agent]\ntimeout_sec = 3000\n")
        (t / "environment" / "Dockerfile").write_text(f"FROM invented/image-{i}:latest\n")
        names.append(name)
    return names


def conversation(html: bool = False) -> dict:
    """a trajectory as mini-swe-agent writes it — invented words"""
    said = "<script>alert('x')</script> <b>bold</b>" if html else "I will look around."
    return {"info": {"exit_status": "Submitted", "mini_version": "2.4.6"}, "messages": [
        {"role": "system", "content": "invented system"},
        {"role": "user", "content": "invented task: make the invented test pass"},
        {"role": "assistant", "content": said, "reasoning_content": "thinking about <i>it</i>",
         "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps(
             {"command": "ls -la && echo '<img src=x onerror=alert(1)>'"})}}],
         "extra": {"response": {"usage": {"prompt_tokens": 1200, "completion_tokens": 300}}}},
        {"role": "tool", "content": "<returncode>0</returncode>\n<output>\n" +
         "\n".join(f"line {i} <tag>" for i in range(60)) + "\n</output>"},
        {"role": "assistant", "content": "", "reasoning_content": "done",
         "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps(
             {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})}}],
         "extra": {"response": {"usage": {"prompt_tokens": 1600, "completion_tokens": 40}}}},
        {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}}]}


def trial(rdir: Path, task: str, k: int = 1, stamp: int | None = None, result: str = "resolved",
          exc: str = "", html: bool = False, minutes: float = 20.0, exit_status: str = "Submitted",
          started: bool = True, agent: str = "host-mini-swe-agent", ours: str = "",
          message: str = "") -> Path:
    """one finished trial of a task's attempt, as Harbor and the host agent
    leave it: Harbor's result.json, the agent's own files in agent-host/
    (never mounted), the verifier's output in verifier/ (mounted)"""
    stamp = stamp or int(time.time() * 1000)
    job = rdir / "jobs" / f"{task}__a{k}__{stamp}"
    t = job / f"{task[:32]}__abc1234"
    for d in ("agent", "verifier", ab.HOST_DIR):
        (t / d).mkdir(parents=True, exist_ok=True)
    start = "2026-10-08T10:00:00+00:00"
    # 18b: UTC both ways (time.mktime read it as local time: wrong in Asia/Dubai)
    end = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(
        calendar.timegm(time.strptime("2026-10-08T10:00:00", "%Y-%m-%dT%H:%M:%S")) + minutes * 60))
    r = {"task_name": task, "trial_name": t.name, "started_at": start, "finished_at": end,
         "agent_info": {"name": agent},
         "agent_execution": {"started_at": start} if started else None,
         "agent_result": {"n_input_tokens": 2800, "n_output_tokens": 340,
                          "metadata": {"steps": 2, "exit_status": exit_status}}}
    if result in ("resolved", "unresolved"):
        r["verifier_result"] = {"rewards": {"reward": 1.0 if result == "resolved" else 0.0}}
    if exc:
        r["exception_info"] = {"exception_type": exc, "exception_message":
                               message or f"{exc} happened", "exception_traceback": "",
                               "occurred_at": start}
    (t / "result.json").write_text(json.dumps(r))
    host = t / ab.HOST_DIR
    (host / ab.TRAJECTORY).write_text(json.dumps(conversation(html)))
    (host / ab.PATCH).write_text("--- a/x.c\n+++ b/x.c\n@@ -1 +1 @@\n-<old>\n+<new>\n")
    (host / ab.META).write_text(json.dumps({"exit_status": exit_status, "steps": 2,
                                            "tokens_in": 2800, "tokens_out": 340,
                                            "last_prompt": 1600, "ours": ours}))
    (t / "verifier" / "test-stdout.txt").write_text("invented test: PASSED <ok>\n")
    return t


def run_json(rdir: Path, tasks: list[str], bench: str = "swebench-multilingual",
             model: str = MODEL, of: int = 300) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "run.json").write_text(json.dumps({
        "benchmark": bench, "label": "SWE-bench Multilingual", "dataset": "x", "tasks": tasks,
        "of": of, "attempts": 1, "at_once": 1, "model": model, "agent": "invented agent",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": "on"},
        "window": 262144, "versions": {"harbor": "0.24.0"}, "where": "this server",
        "started_at": time.time() - 3600, "by": "masein"}))


INVENTED_PARSER = {"requires-python": ">=3.11",
                   "dependencies": ["invented-parser==1.0", "invented-data==2.0"]}


def build_world(root: Path, monkeypatch) -> tuple[Path, dict]:
    """18c: the build's files, as the runner keeps them once fetched — an
    invented Python, uv and two invented wheels; nothing is fetched"""
    import agent_run as ar
    d = root / "agent-build" / "0123456789abcdef"
    (d / "wheels").mkdir(parents=True, exist_ok=True)
    for name in ("python.tar.gz", "uv.tar.gz", "wheels/invented_parser-1.0-py3-none-any.whl",
                 "wheels/invented_data-2.0-py3-none-any.whl"):
        (d / name).write_bytes(b"invented " + name.encode())
    info = {"sha256": "ab" * 32, "exclude_newer": "2026-09-24", "python": "3.11.16+20260901",
            "uv": "0.7.13", "packages": 2}
    monkeypatch.setattr(ar, "build_spec", lambda: {**json.loads(ar.BUILD.read_text()),
                                                   "parser": INVENTED_PARSER})
    monkeypatch.setattr(ar, "build_files", lambda r, today=None: (d, info, ""))
    return d, info
