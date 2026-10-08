"""18b part 1: the agent runs' blockers — containment and counting.

1. A task's container can plant links, FIFOs and hard links in the folders
   Harbor mounts (agent/, verifier/, artifacts/). The host agent writes its
   own files where no container reaches (agent-host/), never through a link,
   and empties the mounted folders however it stops; the board reads only
   regular files with one link, reached through real folders.
2. Only an allow-list of errors is ours and asked again; a failure of the
   model's is its result, never replaced by a later success, and a task
   never leaves the denominator.
3. The runner's board calls never put the board's running runs back in the
   queue.

Invented tasks, trials and conversations; no model, no Docker, nothing
fetched."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_relay
import agent_run as ar
from agent18 import MODEL, conversation, run_json, trial
from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_18_agent import post, server_world  # noqa: F401 — server_world is the fixture

MOUNTED = ("agent", "verifier", "artifacts")
OUR_NAMES = (ab.TRAJECTORY, ab.PATCH, ab.META, "trajectory.json", "test-stdout.txt",
             "test-stderr.txt", "reward.txt")


def plant(folder: Path, victim: Path, names=OUR_NAMES) -> dict[str, str]:
    """what a model's command can leave in a mounted folder: for each name a
    link to the host's file, a FIFO, or a hard link to a file of its own"""
    folder.mkdir(parents=True, exist_ok=True)
    kinds = {}
    for i, name in enumerate(names):
        p = folder / name
        kind = ("link", "fifo", "hard")[i % 3]
        if kind == "link":
            p.symlink_to(victim)
        elif kind == "fifo":
            os.mkfifo(p)
        else:
            mine = folder / f".mine-{name}"
            mine.write_text("SECRET-OF-THE-CONTAINER")
            os.link(mine, p)
        kinds[name] = kind
    return kinds


# ---------------------------------------------------------------------------
# 1. the write side: the host agent, run as Harbor runs it, with stand-ins
# ---------------------------------------------------------------------------

class Env:
    """Harbor's environment, stood in for: what was run inside, and answers"""

    def __init__(self):
        self.calls: list[dict] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append({"command": command, "user": user})
        out = ("PROBE tcp-ok\nNET lo \nPIDS 4096\nCAPS 000000000000000b\nNNP 1\nDONE\n"
               if "pids.max" in command else
               "CLEAN files=0 processes=0\n" if command == hm.CLEAN else "")
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


def fake_mini(behaviour):
    """mini-swe-agent's loop and model, stood in for: `behaviour(agent)` is
    its run — the adapter around it is the real one"""
    class Model:
        abort_exceptions: list = []

        def __init__(self, **kw):
            self.kw = kw

        def _query(self, messages, **kw):
            return {"role": "assistant"}

    class DefaultAgent:
        def __init__(self, model, env, **kw):
            self.model, self.env, self.kw, self.messages = model, env, kw, []

        def serialize(self, *extra):
            return {"info": {"exit_status": "x"}, "messages": self.messages}

        def run(self, task):
            try:
                return behaviour(self)
            finally:
                self.save(self.kw.get("output_path"))
    return DefaultAgent, Model


def host_agent(tmp_path, monkeypatch, behaviour):
    monkeypatch.setattr(hm, "mini_classes", lambda: fake_mini(behaviour))
    monkeypatch.setattr(hm, "load_config", lambda name: {
        "agent": {"system_template": "s", "instance_template": "i", "step_limit": 250},
        "environment": {"environment_class": "docker", "cwd": "/testbed", "timeout": 60},
        "model": {"model_kwargs": {"drop_params": True}}})
    t = tmp_path / "jobs" / "inv__a-1__a1__1" / "inv__a-1__abc1234"
    for d in MOUNTED:                                    # as Harbor makes the trial's folder
        (t / d).mkdir(parents=True, exist_ok=True)
    agent = hm.HostMini(config="benchmarks/swebench.yaml", relay="http://127.0.0.1:9/v1",
                        reach="172.17.0.1:8090")
    agent.logs_dir, agent.model_name, agent.session_id = t / "agent", "openai/m", "x__agent"
    return agent, t


def submitted(agent):
    agent.messages = [*conversation()["messages"][:-1], {
        "role": "exit", "content": "", "extra": {"exit_status": "Submitted",
                                                 "submission": "diff --git a/x b/x\n"}}]
    return agent.messages[-1]["extra"]


def test_the_host_agent_writes_through_no_link_the_container_planted(tmp_path, monkeypatch):
    victim = tmp_path / "home" / ".bashrc"
    victim.parent.mkdir()
    victim.write_text("masein's own file\n")
    agent, t = host_agent(tmp_path, monkeypatch, submitted)
    for d in MOUNTED:
        plant(t / d, victim)
    env = Env()
    ctx = SimpleNamespace()
    asyncio.run(agent.run("invented task", env, ctx))
    assert victim.read_text() == "masein's own file\n"          # 70df001: overwritten
    host = t / ab.HOST_DIR
    assert (host / ab.PATCH).read_text() == "diff --git a/x b/x\n"
    assert json.loads((host / ab.TRAJECTORY).read_text())["messages"][-1]["role"] == "exit"
    meta = json.loads((host / ab.META).read_text())
    assert (meta["exit_status"], meta["steps"], meta["last_prompt"]) == ("Submitted", 2, 1600)
    for name in (ab.PATCH, ab.TRAJECTORY, ab.META):
        assert not (host / name).is_symlink() and (host / name).stat().st_nlink == 1
    # Harbor is told the counts, and never reads a usage file from agent/
    assert ctx.model_usage == {} and ctx.n_output_tokens == 340
    # once it stopped: every process killed, the mounted folders emptied, as root
    last = env.calls[-1]
    assert last["user"] == "root" and last["command"] == hm.CLEAN
    for d in ("/logs/agent", "/logs/artifacts", "/logs/verifier"):
        assert d in hm.CLEAN
    assert hm.CLEAN.startswith("kill -9 -1")


def test_our_folder_and_files_are_never_a_link(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    # a file of ours that is a link is replaced, its target untouched
    host = tmp_path / "t" / ab.HOST_DIR
    host.mkdir(parents=True)
    (host / ab.PATCH).symlink_to(victim)
    ab.safe_write(host, ab.PATCH, "ours")
    assert victim.read_text() == "keep" and (host / ab.PATCH).read_text() == "ours"
    assert not (host / ab.PATCH).is_symlink()
    # a FIFO with our name doesn't block the write
    os.mkfifo(host / ab.META)
    ab.safe_write(host, ab.META, "{}")
    assert (host / ab.META).read_text() == "{}"
    # a folder that is a link is refused, nothing written through it
    out = tmp_path / "elsewhere"
    out.mkdir()
    (tmp_path / "t2").mkdir()
    (tmp_path / "t2" / ab.HOST_DIR).symlink_to(out)
    with pytest.raises(OSError):
        ab.safe_write(tmp_path / "t2" / ab.HOST_DIR, ab.PATCH, "x")
    assert not list(out.iterdir())
    with pytest.raises(ValueError):
        ab.safe_write(host, "../x", "x")


@pytest.mark.parametrize("how", ["raises", "time is up"])
def test_the_mounted_folders_are_emptied_however_the_agent_stops(tmp_path, monkeypatch, how):
    started = threading.Event()

    def behaviour(agent):
        if how == "raises":
            raise RuntimeError("the model's own failure")
        started.set()
        while not agent.env.stop.is_set():            # a command running when time is up
            time.sleep(0.01)
        raise hm.Stopped("the agent's time is up")
    agent, t = host_agent(tmp_path, monkeypatch, behaviour)
    env = Env()

    async def go():
        if how == "raises":
            await agent.run("x", env, SimpleNamespace())
        else:                                          # Harbor's limit: wait_for cancels it
            await asyncio.wait_for(agent.run("x", env, SimpleNamespace()), timeout=0.3)
    with pytest.raises(RuntimeError if how == "raises" else asyncio.TimeoutError):
        asyncio.run(go())
    assert env.calls[-1]["command"] == hm.CLEAN                  # 70df001: skipped
    assert json.loads((t / ab.HOST_DIR / ab.META).read_text())["exit_status"] == ""


def test_a_reach_refused_is_ours_and_said(tmp_path, monkeypatch):
    agent, t = host_agent(tmp_path, monkeypatch, submitted)

    class Reaching(Env):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            self.calls.append({"command": command, "user": user})
            return SimpleNamespace(stdout="PROBE tcp-ok\nREACHED 1.1.1.1:443\nNET lo\nPIDS 4096"
                                   "\nCAPS 000000000000000b\nNNP 1\nDONE\n", stderr="",
                                   return_code=0)
    env = Reaching()
    with pytest.raises(hm.ReachRefused):
        asyncio.run(agent.run("x", env, SimpleNamespace()))
    meta = json.loads((t / ab.HOST_DIR / ab.META).read_text())
    assert meta["ours"] == "ReachRefused: the task's container reached 1.1.1.1:443"
    assert env.calls[-1]["command"] == hm.CLEAN


# ---------------------------------------------------------------------------
# 1. the read side: the board
# ---------------------------------------------------------------------------

def test_the_board_reads_no_link_fifo_or_hard_link_the_container_planted(tmp_path):
    victim = tmp_path / "environ"
    victim.write_text("OPENROUTER_API_KEY=sk-or-invented SUBMIT_TOKEN=invented")
    rd = tmp_path / "run"
    t = trial(rd, "inv__a-1")
    for d in MOUNTED:
        for p in (t / d).iterdir() if (t / d).exists() else []:
            p.unlink()
        plant(t / d, victim)
    # agent-host/ is never mounted, but read the same way
    for p in (t / ab.HOST_DIR).iterdir():
        p.unlink()
    plant(t / ab.HOST_DIR, victim)
    t0 = time.time()
    c = ab.conversation(rd, t.parent.name, t.name)                 # a FIFO: never waited on
    assert time.time() - t0 < 5
    text = json.dumps(c)
    assert "sk-or-invented" not in text and "SECRET-OF-THE-CONTAINER" not in text
    assert c["patch"] == "" and c["verifier"] == "" and c["steps"] == []
    # each kind, one at a time, through the one reader
    for name, kind in plant(tmp_path / "each", victim).items():
        assert ab.safe_read(tmp_path / "each", name) == "", (name, kind)
    # a folder on the way that is a link, and '..'
    (tmp_path / "via").symlink_to(tmp_path)
    assert ab.safe_read(tmp_path, "via/environ") == ""
    assert ab.safe_read(rd, "../environ") == ""
    # a regular file of one link is read, up to its cap
    (tmp_path / "ok.txt").write_text("x" * 100)
    assert ab.safe_read(tmp_path, "ok.txt", cap=10) == "x" * 10


def test_the_task_page_shows_nothing_of_the_boards_environment(svc, monkeypatch):  # noqa: F811
    from service import agent_runs, config
    import import_agent
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / "swebench-multilingual__invented-agent__k1"
    names = ["inv__repo1-1001"]
    run_json(rdir, names)
    t = trial(rdir, names[0])
    (t / "verifier" / "test-stdout.txt").unlink()
    (t / "verifier" / "test-stdout.txt").symlink_to("/proc/self/environ" if Path(
        "/proc/self/environ").exists() else str(Path(sys.executable)))
    os.mkfifo(t / "verifier" / "test-stderr.txt")
    (rdir / "progress.json").write_text(json.dumps({"line": "1 of 1"}))
    assert import_agent.main([str(rdir)]) == 0
    got = agent_runs.task(rdir.name, names[0])
    assert got and got["verifier"] == ""                         # 70df001: the environment
    assert got["steps"] and got["patch"].startswith("--- a/x.c")


# ---------------------------------------------------------------------------
# 2. ours is an allow-list; the model's result is its result
# ---------------------------------------------------------------------------

def test_a_failure_of_the_models_is_never_asked_again_nor_replaced(server_world, capsys):  # noqa: F811
    w = server_world
    seen: list[str] = []
    fails = ["VerifierTimeoutError", "ContextWindowExceededError", "InternalServerError",
             "RuntimeError"]

    def outcome(task, n):
        seen.append(task) if task not in seen else None
        return fails[seen.index(task)] if n == 1 else "resolved"   # a success if asked again
    w["outcome"] = outcome
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "4", "--no-board"]) == 0
    assert sorted(h["task"] for h in w["harbor"]) == sorted(seen)    # each asked once
    rdir = ar.run_dir(Path(os.environ["BENCH_ROOT"]), "swebench-multilingual", MODEL, 1, False)
    rs = ar.results(rdir, seen, 1)
    assert [r["result"] for r in rs] == ["unresolved"] * 4
    whys = {r["why"] for r in rs}
    assert {"the tests ran past their time limit", "the window was outgrown",
            "the server refused a reply (HTTP 500)"} <= whys
    s = ab.score(rs, 300)
    assert (s["resolved"], s["n"], s["errors"]) == (0, 4, 0)       # 70df001: 100%, 0 errors


def test_an_error_of_ours_is_asked_again_and_said_and_a_given_up_task_stays_counted(
        server_world, capsys):  # noqa: F811
    w = server_world
    order: list[str] = []

    def outcome(task, n):
        order.append(task) if task not in order else None
        if task == order[0]:
            return "docker"                                         # every time: given up
        if len(order) > 1 and task == order[1] and n == 1:
            return "docker"                                         # once, then the model's
        return "resolved"
    w["outcome"] = outcome
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "3", "--no-board"]) == 0
    rdir = ar.run_dir(Path(os.environ["BENCH_ROOT"]), "swebench-multilingual", MODEL, 1, False)
    rs = {r["task"]: r for r in ar.results(rdir, order, 1)}
    assert rs[order[0]]["result"] == "error" and rs[order[0]]["tries"] == 3
    assert rs[order[1]]["result"] == "resolved" and rs[order[1]]["tries"] == 2
    assert rs[order[1]]["again"][0].startswith("RuntimeError: Docker compose command failed")
    s = ab.score(list(rs.values()), 300)
    assert (s["n"], s["resolved"], s["errors"], s["again"]) == (3, 2, 1, 2)
    assert ab.score_words(s, 300).endswith("· 1 error of ours, counted not resolved")
    assert ab.again_words(list(rs.values())) == (
        "2 tasks run again after an error of ours: Docker couldn't pull, build or start its "
        "container ×3")


def test_only_the_allow_list_is_ours(tmp_path):
    rd = tmp_path / "r"
    # 18c point 5: Docker's failures come before the agent started (Harbor
    # never began its phase); after it, they are ours but never asked again
    cases = [
        (dict(exc="ServerDown", ours="ServerDown: the model's server didn't answer"), "error"),
        (dict(exc="EnvironmentStartTimeoutError", started=False), "error"),
        (dict(exc="RuntimeError", message="Docker compose command failed: pull", started=False),
         "error"),
        (dict(exc="ImagePullError", started=False), "error"),          # before the agent ran
        (dict(exc="APIConnectionError"), "unresolved"),               # one bad reply
        (dict(exc="ServiceUnavailableError"), "unresolved"),
        (dict(exc="RuntimeError", message="the model's command broke the agent"), "unresolved"),
        (dict(exc="RewardFileNotFoundError"), "unresolved")]
    for i, (kw, want) in enumerate(cases):
        t = trial(rd, f"inv__o-{i}", result="", **kw)
        assert ab.read_trial(t)["result"] == want, kw


def test_the_relay_says_the_server_is_down_only_when_its_health_fails(tmp_path):
    fake = FakeServer()
    usage = tmp_path / "u.jsonl"
    relay = agent_relay.Relay(fake.base, "k", ab.SAMPLING, usage=usage).start()
    try:
        # one bad reply from a server that is up: passed back as it came
        fake.chat_error = lambda body: (503, "busy")
        st, body = post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        assert st == 503 and hm.DOWN not in body
        # the server gone: the relay's own 503, its type the agent reads as ours
        fake.close()
        st, body = post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                        headers={"X-Agent-Trial": "t__agent"})
        assert st == 503 and json.loads(body)["error"]["type"] == hm.DOWN
        assert json.loads(usage.read_text().splitlines()[-1])["down"] is True
    finally:
        relay.stop()


# ---------------------------------------------------------------------------
# 3. the runner's board calls leave the board's running runs alone
# ---------------------------------------------------------------------------

def test_a_running_run_stays_running_through_progress_and_the_import(svc):  # noqa: F811
    import import_agent
    from service import config, db
    lm = db.add("google/gemma-3-1b-it", "instruct", "standard", "colleague", "", status="queued")
    db.update(lm, status="running", progress="hellaswag 40%")
    stop = db.add("google/gemma-3-1b-it", "instruct", "standard", "colleague", "",
                  status="queued")
    db.update(stop, status="canceling")
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / "swebench-multilingual__invented-agent__k1"
    run_json(rdir, ["inv__repo1-1001"])
    trial(rdir, "inv__repo1-1001")
    (rdir / "progress.json").write_text(json.dumps({"line": "1 of 1"}))
    for _ in range(3):                                   # every 3 minutes, for days
        assert import_agent.main(["--progress", str(rdir)]) == 0
    assert import_agent.main([str(rdir)]) == 0
    assert (db.get(lm)["status"], db.get(lm)["progress"]) == ("running", "hellaswag 40%")
    assert db.get(stop)["status"] == "canceling"           # 70df001: canceled
    # the service's own start-up still does it
    db.init(startup=True)
    assert db.get(lm)["status"] == "queued" and db.get(stop)["status"] == "canceled"


def test_no_script_but_the_service_starts_the_board():
    import inspect

    import import_agent
    from service import app as appmod
    assert "db.init(" not in inspect.getsource(import_agent.main).replace(
        "never db.init()", "")
    assert "db.init(startup=True)" in inspect.getsource(appmod.lifespan)
    root = Path(__file__).resolve().parent.parent
    for p in [*(root / "scripts").glob("*.py"), *(root / "service").glob("*.py")]:
        if p.name == "app.py":
            continue
        assert "init(startup=True)" not in p.read_text(), p.name
