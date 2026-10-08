"""18c part 2: before the full run — an error of ours asked again only with
evidence it came before the agent; a server that goes down stops the run
and its try isn't counted; a job with no result counts as a try and one
past its limits is killed; a resume refuses a changed cap, override or
runner; the oracle on every task, and the tasks whose reference solution
doesn't pass here left out and said; what isn't comparable said; and the
small ones (the cache check, the reach check's ports, a verifier's own
compose file, the venv held to its lock, the build cache pruned by its own
records, the board's queue held, the exec path closed before the kill).

Stand-ins for Harbor, Docker and the board; invented tasks and trials; no
model, nothing fetched."""

from __future__ import annotations

import asyncio
import base64
import hashlib
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
import agent_run as ar
from agent18 import MODEL, run_json, tasks_folder, trial
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_18_agent import server_world  # noqa: F401 — server_world is the fixture
from test_18b1_containment import Env, host_agent

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()


def rdir_of(model: str = MODEL, oracle: bool = False, number: int = 1) -> Path:
    return ar.run_dir(Path(os.environ["BENCH_ROOT"]), "swebench-multilingual", model, 1, oracle,
                      number)


# ---------------------------------------------------------------------------
# 5. asked again only with evidence that it came before the agent started
# ---------------------------------------------------------------------------

AFTER = [dict(exc="AddTestsDirError", message="copying the tests in failed"),
         dict(exc="RuntimeError", message="Docker compose command failed for environment x: cp"),
         dict(exc="RuntimeError", message="Docker compose command failed for environment x: down"),
         dict(exc="EnvironmentStartTimeoutError", message="the verifier's container didn't start")]


def test_four_failures_after_the_agent_ran_then_successes_score_nothing(  # noqa: F811
        server_world, capsys):  # noqa: F811
    """the reviewers' case: each failure came after the model had run; asked
    again, a success replaced it — 4 of 4. Now: never asked again"""
    w = server_world
    order: list[str] = []

    def outcome(task, n):
        order.append(task) if task not in order else None
        return "after" if n == 1 else "resolved"
    w["outcome"] = outcome

    import test_18_agent as t18
    real = t18.trial

    def trial_after(rd, task, k, stamp=None, **kw):               # Harbor's failure, after
        if kw.get("exc") == "after":
            kw = {**AFTER[order.index(task) % 4], "result": ""}
        return real(rd, task, k, stamp=stamp, **kw)
    t18.trial = trial_after
    try:
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "4",
                        "--no-board"]) == 0
    finally:
        t18.trial = real
    assert len(w["harbor"]) == 4                                  # d94f7f6: 8, each asked again
    rs = ar.results(rdir_of(), order, 1)
    assert [r["result"] for r in rs] == ["error"] * 4 and all(r["final"] for r in rs)
    s = ab.score(rs, 300)
    assert (s["resolved"], s["n"], s["errors"]) == (0, 4, 4)     # d94f7f6: 4 of 4 resolved
    assert all(r["why"].endswith("after the agent ran: counted not resolved, never asked again")
               for r in rs)


def test_only_evidence_of_before_asks_again(tmp_path):
    rd = tmp_path / "r"
    cases = {
        # Harbor never began the agent's phase, and the agent wrote nothing
        "inv__e-1": (dict(exc="RuntimeError", started=False,
                          message="Docker compose command failed: build"), "before"),
        # the agent's own word: the reach, before the model was asked
        "inv__e-2": (dict(exc="ReachRefused", ours="ReachRefused: reached 1.1.1.1:443"), "before"),
        # no agent_execution, but the agent's own files are there: it ran
        "inv__e-3": (dict(exc="RuntimeError", started=False, message="Docker compose command "
                          "failed: cp"), "after"),
        "inv__e-4": (dict(exc="ServerDown"), "down"),
        "inv__e-5": (dict(exc="KeyboardInterrupt"), "before"),     # the run was stopped
        "inv__e-6": (dict(exc="CleanupFailed", message="the clean-up …: not verified"), "after")}
    for task, (kw, want) in cases.items():
        t = trial(rd, task, result="", **kw)
        if task == "inv__e-3":                     # the agent's files, Harbor's record lost
            (t / ab.HOST_DIR).mkdir(exist_ok=True)
            (t / ab.HOST_DIR / ab.META).write_text("{}")
        r = ab.read_trial(t)
        assert (r["result"], r.get("ours")) == ("error", want), (task, r)
        assert r["final"] is (want == "after")
    # a teardown that failed after the tests' result was in: the result stands
    t = trial(rd, "inv__e-7", result="resolved", exc="RuntimeError",
              message="Docker compose command failed for environment x: down")
    assert ab.read_trial(t)["result"] == "resolved"
    # one bad reply from a server that is up is the model's
    assert ab.read_trial(trial(rd, "inv__e-8", result="", exc="APIConnectionError"))[
        "result"] == "unresolved"


# ---------------------------------------------------------------------------
# 6. a server that goes down stops the run; its try isn't counted
# ---------------------------------------------------------------------------

def test_a_server_that_goes_down_stops_the_run_and_its_try_isnt_counted(  # noqa: F811
        server_world, capsys):  # noqa: F811
    w = server_world
    order: list[str] = []

    def outcome(task, n):
        order.append(task) if task not in order else None
        return "ServerDown" if len(order) > 1 and task == order[1] and n == 1 else "resolved"
    w["outcome"] = outcome
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "5", "--no-board"]) == 1
    out = capsys.readouterr().out
    assert "stopping — the model's server went down (" in out and \
        "the task it was on is asked again from its start" in out
    assert len(w["harbor"]) == 2                              # d94f7f6: every task, 3 times
    assert ar.state_of(rdir_of(), order[1], 1) == ("todo", 0)  # not counted
    # the server is back: the same command asks it again, from its start
    w["harbor"].clear()
    w["outcome"] = lambda task, n: "resolved"
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "5", "--no-board"]) == 0
    assert w["harbor"][0]["task"] == order[1] and len(w["harbor"]) == 4
    r = next(x for x in ar.results(rdir_of(), [order[1]], 1))
    assert r["result"] == "resolved" and r["again"][0].startswith(ab.DOWN_WORDS)
    assert "the model's server went down (1 task, 1 try)" in ab.again_words([r])


# ---------------------------------------------------------------------------
# 7. a job with no result counts; a job past its limits is killed
# ---------------------------------------------------------------------------

def test_a_job_that_ends_with_no_result_counts_as_a_try(server_world, capsys):  # noqa: F811
    w = server_world
    import test_18_agent as t18
    real = t18.trial

    def nothing(rd, task, k, stamp=None, **kw):                  # Harbor wrote no result.json
        if kw.get("exc") == "noresult":
            return None
        return real(rd, task, k, stamp=stamp, **kw)
    t18.trial = nothing
    w["outcome"] = lambda task, n: "noresult"
    try:
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1",
                        "--no-board"]) == 0
    finally:
        t18.trial = real
    assert len(w["harbor"]) == ar.TRIES                         # d94f7f6: restarted forever
    task = w["harbor"][0]["task"]
    assert ar.state_of(rdir_of(), task, 1) == ("given up", 3)
    r = ar.results(rdir_of(), [task], 1)[0]
    assert r["result"] == "error" and r["why"].startswith("its job ended with no result")
    assert not list((rdir_of() / "jobs").glob("*.running"))


class Hung:
    """Harbor's process, hung: it never ends until killed"""
    made: list = []

    def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, **kw):
        self.cmd, self.killed = cmd, False
        job = Path(cmd[cmd.index("-o") + 1]) / cmd[cmd.index("--job-name") + 1]
        t = job / "inv__trial"
        (t / ab.HOST_DIR).mkdir(parents=True)                    # the agent had run
        (t / ab.HOST_DIR / ab.TRAJECTORY).write_text("{}")
        Hung.made.append(self)

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


def test_a_job_past_its_limits_is_killed_said_and_never_asked_again(  # noqa: F811
        server_world, monkeypatch, capsys):  # noqa: F811
    calls = []
    real_run = ar.run

    def run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        return real_run(cmd, cwd=cwd, timeout=timeout, env=env)
    monkeypatch.setattr(ar, "run", run)
    monkeypatch.setattr(ar.subprocess, "Popen", Hung)
    monkeypatch.setattr(ar, "job_limit", lambda task: -1.0)        # already past it
    Hung.made = []
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1", "--no-board"]) == 0
    assert len(Hung.made) == 1 and Hung.made[0].killed           # d94f7f6: waited forever
    task = Path(Hung.made[0].cmd[Hung.made[0].cmd.index("-p") + 1]).name
    killed = list((rdir_of() / "jobs").glob("*.killed"))
    assert len(killed) == 1 and "ran past its limits plus 30 minutes" in killed[0].read_text()
    assert ["docker", "ps", "-a", "--format",
            '{{.ID}} {{.Label "com.docker.compose.project"}}'] in calls      # its containers
    r = ar.results(rdir_of(), [task], 1)[0]
    assert r["result"] == "error" and r["final"] and r["why"].startswith("killed: it ran past")


def test_a_jobs_limit_is_its_tasks_own_plus_30_minutes(tmp_path):
    t = tmp_path / "t"
    t.mkdir()
    (t / "task.toml").write_text("[agent]\ntimeout_sec = 3000\n[verifier]\ntimeout_sec = 3000\n"
                                 "[environment]\nbuild_timeout_sec = 1800.0\n")
    assert ar.job_limit(t) == 3000 + 3000 + 1800 + 1800
    (t / "task.toml").write_text(
        '[agent]\ntimeout_sec = 10800.0\n[verifier]\ntimeout_sec = 1800.0\n'
        'environment_mode = "separate"\n[[verifier.collect]]\ncommand = "x"\n'
        'timeout_sec = 300.0\n[environment]\nbuild_timeout_sec = 1800.0\n')
    assert ar.job_limit(t) == 10800 + 1800 + 2 * 1800 + 300 + 1800


# ---------------------------------------------------------------------------
# 8. a resume refuses a changed cap, override, runner
# ---------------------------------------------------------------------------

def test_a_resume_refuses_another_runner_cap_or_override(server_world, monkeypatch,  # noqa: F811
                                                        capsys):
    for k in ("max_reply_tokens", "override_sha256", "runner_commit", "runner_sha256",
              "build_lock"):
        assert k in ar.SETTINGS
    monkeypatch.setattr(ar, "runner_commit", lambda: "a" * 40)
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0
    run = json.loads((rdir_of() / "run.json").read_text())
    assert run["runner_commit"] == "a" * 40 and run["max_reply_tokens"] == 32_768
    assert len(run["override_sha256"]) == 64 and len(run["runner_sha256"]) == 64
    monkeypatch.setattr(ar, "runner_commit", lambda: "b" * 40)        # a git pull
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--no-board"]) == 2
    out = capsys.readouterr().out
    assert "refused — this run was started with other settings (runner_commit aaaa" in out
    assert "add --run 2" in out                               # d94f7f6: rolled into it
    before = {"max_reply_tokens": 32_768, "override_sha256": "x"}
    assert len(ar.changed_settings(before, {"max_reply_tokens": 16_384, "override_sha256": "y"})) \
        == 2


def test_the_runners_commit_is_read_as_root_too(monkeypatch):
    seen = []
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (seen.append(cmd), (0, "c" * 40 + "\n"))[1])
    assert ar.runner_commit() == "c" * 40
    assert seen[0][:3] == ["git", "-c", f"safe.directory={ar.REPO}"]   # "dubious" under sudo
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (128, "fatal: not a git repository"))
    assert ar.runner_commit() == "unknown"


# ---------------------------------------------------------------------------
# 9 and 10. the oracle on every task; tasks left out, said
# ---------------------------------------------------------------------------

def oracle_world(root: Path, names: list[str]) -> Path:
    od = ar.run_dir(root, "swebench-multilingual", "", 1, True)
    run_json(od, names[:4], model="")
    trial(od, names[0], agent="oracle", exit_status="")
    trial(od, names[1], result="unresolved", agent="oracle", exit_status="")   # doesn't pass
    trial(od, names[2], result="", exc="RuntimeError", started=False,
          message="Docker compose command failed: pull")                      # not known yet
    trial(od, names[3], agent="oracle", exit_status="")
    return od


def test_which_tasks_is_always_said(server_world):  # noqa: F811
    with pytest.raises(SystemExit) as e:
        ar.main(["swebench-multilingual", "--oracle", "--no-board"])
    assert e.value.code == 2                                  # d94f7f6: all 300, unasked


def test_the_tasks_whose_reference_doesnt_pass_are_left_out_and_said(  # noqa: F811
        server_world, capsys):  # noqa: F811
    w = server_world
    root = Path(os.environ["BENCH_ROOT"])
    names = w["names"]
    oracle_world(root, names)
    got = ar.oracle_states(root, "swebench-multilingual", names[:6])
    assert got["left_out"] == [names[1]]
    assert got["unknown"] == sorted([names[2], *names[4:6]])
    # the full run waits for the oracle on every task
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--all", "--no-board"]) == 2
    out = capsys.readouterr().out
    assert "refused — the oracle hasn't run on 37 of the 40 tasks: run it on every task first " \
           "(--oracle --all)" in out
    # a model run leaves the task out, said
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--only", ",".join(names[:4]),
                    "--no-board"]) == 0
    out = capsys.readouterr().out
    assert f"1 of these tasks is left out: their reference solution doesn't pass here (the " \
           f"oracle run) — {names[1]}" in out
    assert names[1] not in [h["task"] for h in w["harbor"]]
    # the board's score leaves it out of the denominator too, and says so
    import import_agent
    rdir = rdir_of()
    trial(rdir, names[1], result="unresolved")                  # asked before the oracle ran
    run = json.loads((rdir / "run.json").read_text())
    run["tasks"] = [*run["tasks"], names[1]]
    (rdir / "run.json").write_text(json.dumps(run))
    s = import_agent.summary(rdir)
    assert s["left_out"] == [names[1]] and s["score"]["n"] == 3 and s["score"]["of"] == 299
    assert s["words"].endswith("· pilot: 3 of 299 · 1 task left out: their reference solution "
                               "doesn't pass here")


def test_the_card_says_whats_left_out_and_what_isnt_comparable(svc, monkeypatch):  # noqa: F811
    from service import agent_runs, config
    root = Path(config.BENCH_ROOT)
    names = tasks_folder(root / "x", 4)
    oracle_world(root, names)
    cat = agent_runs.catalogue()
    ml = next(b for b in cat["benchmarks"] if b["key"] == "swebench-multilingual")
    assert ml["left_out"] == [names[1]]
    assert ml["not_comparable"] == ("A task that runs past its 50 minutes counts as not resolved "
                                    "here; the published runs had no such limit.")
    deep = next(b for b in cat["benchmarks"] if b["key"] == "deepswe")
    assert deep["left_out"] == [] and deep["not_comparable"] == ""


# ---------------------------------------------------------------------------
# 11. the small ones
# ---------------------------------------------------------------------------

class Relay:
    url = "http://127.0.0.1:9/v1"

    def __init__(self, second: dict):
        self.records = [{"trial": ar.CACHE_TRIAL, "status": 200, "prompt_n": 5000,
                         "cache_n": 0, "prompt_ms": 5000, "predicted_n": 300,
                         "predicted_ms": 10000},
                        {"trial": ar.CACHE_TRIAL, "status": 200, **second}]


def test_the_cache_check_refuses_a_fifth_of_a_long_prompt_read_again():
    ask = lambda *a, **k: {"role": "assistant", "tool_calls": [{"id": "c1"}]}  # noqa: E731
    why, _ = ar.cache_line(Relay({"prompt_n": 1300, "cache_n": 4700}), "m", "x", ask=ask, first=[])
    assert why.startswith("the second step read 1,300 of its 6,000 prompt tokens again")
    # d94f7f6: only past half
    why, _ = ar.cache_line(Relay({"prompt_n": 800, "cache_n": 5200}), "m", "x", ask=ask, first=[])
    assert why == ""
    why, _ = ar.cache_line(Relay({"prompt_n": 400, "cache_n": 600}), "m", "x", ask=ask, first=[])
    assert why == ""                                           # a short prompt: past half only
    why, _ = ar.cache_line(Relay({"prompt_n": 40}), "m", "x", ask=ask, first=[])
    assert "no `timings.cache_n`" in why and "build_info" in why and "llama-server" in why


def test_the_checks_have_the_agents_own_budget(monkeypatch):
    import urllib.request
    sent = {}

    class R:
        def __init__(self, req, timeout=None):
            sent.update(json.loads(req.data), timeout=timeout)

        def __enter__(self):
            return SimpleNamespace(read=lambda: json.dumps({"choices": [{"message": {
                "tool_calls": [{"id": "x"}]}}]}).encode())

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(urllib.request, "urlopen", R)
    assert ar.tool_call_line("http://relay/v1", "m") == ""
    assert sent["max_tokens"] == ar.MAX_REPLY == 32_768        # d94f7f6: 4,096
    # mini-swe-agent's own model class, as the cache check asks it
    made = {}

    class Litellm:
        def __init__(self, **kw):
            made.update(kw)

        def query(self, messages):
            return {}
    mod = SimpleNamespace(LitellmModel=Litellm)
    monkeypatch.setitem(sys.modules, "minisweagent", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "minisweagent.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "minisweagent.models.litellm_model", mod)
    ar.mini_ask("http://relay/v1", "m", [], "t")
    assert made["model_kwargs"]["max_tokens"] == 32_768         # d94f7f6: 8,192


def test_the_reach_check_takes_each_servers_port_from_the_board(  # noqa: F811
        server_world, monkeypatch, capsys):  # noqa: F811
    real_run = ar.run

    def run(cmd, cwd=None, timeout=None, env=None):
        if cmd[:len(ar.BOARD)] == ar.BOARD and cmd[len(ar.BOARD)] == "--served-all":
            return 0, json.dumps([{"id": "served/k8", "base_url": "http://host.docker.internal:8091/v1"},
                                  {"id": "served/x", "base_url": "http://host.docker.internal:8094/v1"}])
        return real_run(cmd, cwd=cwd, timeout=timeout, env=env)
    monkeypatch.setattr(ar, "run", run)
    seen = {}
    monkeypatch.setattr(ar, "check_reach", lambda rdir, targets: seen.setdefault("t", targets)
                        and "")
    monkeypatch.setattr(ar, "check_build_reach", lambda rdir, targets: ("", []))
    assert ar.main(["swebench-multilingual", "--check-reach"]) == 0
    t = seen["t"]
    assert "172.17.0.1:8091" in t and "172.17.0.1:8094" in t
    assert not any(x.endswith(":8090") for x in t)            # d94f7f6: 8090, whatever was served


def test_a_verifiers_own_compose_file_is_never_replaced(tmp_path):
    src = tmp_path / "src"
    names = tasks_folder(src, 2, how="metadata")
    for n in names:
        (src / n / "task.toml").write_text('[verifier]\nenvironment_mode = "separate"\n')
        (src / n / "tests").mkdir()
    (src / names[1] / "tests" / "docker-compose.yaml").write_text("services: {side: {}}\n")
    out, got = ar.offline_tasks(tmp_path, ab.BENCHES["deepswe"], src, names)
    assert got["refused"] == {names[1]: "it has its own tests/docker-compose.yaml"}
    assert (out / names[0] / "tests" / "docker-compose.yaml").read_text() == ar.OVERRIDE_YAML


def test_the_venv_is_held_to_its_lock_extras_and_files_too(tmp_path, monkeypatch):
    from importlib import metadata
    lock = ar.lock()
    assert lock["pip"] == "26.2.1"                             # pip pinned, with its hashes
    a = DOC[DOC.index("## A."):DOC.index("## B.")]
    assert "--seed" not in a and "uv venv --python 3.12 ~/agent-venv" in a
    # an installed package as uv or pip leaves it: its files and their record
    site = tmp_path / "site"
    (site / "pkg").mkdir(parents=True)
    (site / "pkg" / "__init__.py").write_text("x = 1\n")
    di = site / "pkg-1.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(b"x = 1\n").digest()).rstrip(b"=").decode()
    (di / "RECORD").write_text(f"pkg/__init__.py,sha256={digest},6\npkg-1.0.dist-info/RECORD,,\n")
    d = metadata.PathDistribution(di)
    assert ar.changed_files({"pkg": d}) == []
    (site / "pkg" / "__init__.py").write_text("x = 2  # changed after it was installed\n")
    assert ar.changed_files({"pkg": d}) == ["pkg's pkg/__init__.py differs from its record"]
    # a package the lock doesn't hold
    have = {k: SimpleNamespace(version=v, files=[]) for k, v in lock.items()}
    have["evil"] = SimpleNamespace(version="0.1", files=[])
    monkeypatch.setattr(ar, "installed", lambda: have)
    why = ar.check_versions(ar.versions(), oracle=False)
    assert "evil 0.1 (not in the lock)" in why                 # d94f7f6: ignored


def test_the_build_cache_is_pruned_by_its_own_records_only(monkeypatch):
    held = [{"ID": "board1"}, {"ID": "board2"}]
    calls = []

    def run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        if cmd[:3] == ["docker", "system", "df"]:
            return 0, json.dumps(held) + "\n"
        if cmd[:3] == ["docker", "builder", "prune"]:
            i = cmd[-1].split("=", 1)[1]
            held[:] = [x for x in held if x["ID"] != i]
            return 0, f"ID\n{i}\nTotal:\t1MB\n"
        return 0, ""
    monkeypatch.setattr(ar, "run", run)
    before = ar.cache_ids()
    held += [{"ID": "task1"}, {"ID": "task2"}]                      # the task's build
    ar.remove_built("inv__repo1-1001", before)
    pruned = [c[-1] for c in calls if c[:3] == ["docker", "builder", "prune"]]
    assert sorted(pruned) == ["id=task1", "id=task2"]               # d94f7f6: everything
    assert [x["ID"] for x in held] == ["board1", "board2"]
    assert ["docker", "builder", "prune", "-f"] not in calls


def test_the_queue_holds_runs_on_a_model_an_agent_run_uses(svc, monkeypatch):  # noqa: F811
    from service import db, worker
    a = db.add("served/held-18c", "instruct", "frontier", "masein", "", tasks=["x"])
    b = db.add("served/free-18c", "instruct", "frontier", "masein", "", tasks=["x"])
    held = {"served/held-18c": {"key": "k", "until": None, "line": ""}}
    got = db.claim_next(held=held)
    assert got["id"] == b                                       # d94f7f6: the held one first
    assert db.get(a)["status"] == "queued" and db.get(a)["progress"] == db.HELD_LINE
    assert db.claim_next(held=held) is None                     # it waits
    assert db.claim_next(held={})["id"] == a                    # the run stopped: it goes
    # the worker asks which models are held
    monkeypatch.setattr("service.agent_runs.busy", lambda: held)
    assert worker._held() == held


def test_the_exec_path_is_closed_before_the_clean_up_kills(tmp_path, monkeypatch):
    """a command sent just before the time is up is waited for: none starts
    after the clean-up's kill"""
    order: list[str] = []

    class Slow(Env):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if command.startswith("timeout -s KILL"):            # the model's command
                order.append("command sent")
                await asyncio.sleep(0.5)
                order.append("command back")
                return SimpleNamespace(stdout="", stderr="", return_code=0)
            if command == hm.CLEAN:
                order.append("clean-up")
            return await super().exec(command, cwd, env, timeout_sec, user)

    def behaviour(agent):
        agent.env.execute({"command": "make test"})                 # sent as the time runs out
        while not agent.env.stop.is_set():
            time.sleep(0.01)
        with pytest.raises(hm.Stopped):                             # the path is closed
            agent.env.execute({"command": "echo after"})
        raise hm.Stopped("the agent's time is up")
    agent, t = host_agent(tmp_path, monkeypatch, behaviour)

    async def go():
        await asyncio.wait_for(agent.run("x", Slow(), SimpleNamespace()), timeout=0.2)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert order == ["command sent", "command back", "clean-up"]   # d94f7f6: kill, then back
    assert threading.active_count() >= 1


def test_the_runs_last_lines_count_timeouts_and_steps_read_again(tmp_path):
    rd = tmp_path / "r"
    rd.mkdir()
    lines = [{"trial": ar.CACHE_TRIAL, "status": 200, "prompt_n": 9, "cache_n": 0},
             {"trial": "t1", "status": 200, "prompt_n": 5000, "cache_n": 0},     # its first
             {"trial": "t1", "status": 200, "prompt_n": 90, "cache_n": 5000},
             {"trial": "t1", "status": 200, "prompt_n": 6000, "cache_n": 100},   # read again
             {"trial": "t1", "status": 503, "down": True}]
    (rd / "relay-usage.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    words = ar.usage_words(rd, [{"result": "timeout"}, {"result": "resolved"}])
    assert words.startswith("1 task timed out · 2 of 3 steps read more of their prompt again "
                            "than they took from the server's cache")


def test_step_b_finds_the_server_by_its_port_and_checks_the_others():
    b = DOC[DOC.index("## B."):DOC.index("## C.")]
    steps = {int(x.split(".")[0]) for x in b.splitlines() if x[:1].isdigit() and ". " in x[:4]}
    assert steps == set(range(0, 9))
    assert "sudo ss -tnp 'dport = :8091'" in b and "loop" in b and "calibration" in b   # B0
    assert "sudo ss -ltnp 'sport = :8091'" in b and "pgrep" not in b                    # B1
    assert ("while sudo ss -ltn 'sport = :8091' | grep -q LISTEN; do sleep 2; done; echo "
            "\"8091 closed\"") in b                                                     # B2
    assert "sudo ss -ltn '( sport = :8090 or sport = :8092 or sport = :8094 or sport = :8096 )'" \
        in b
    assert "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv" in b
    assert "nvidia-smi --query-gpu=memory.used,memory.total --format=csv" in b and "1.5 GB" in b
    assert "timed out" in b and "relay-usage.jsonl" in b                                # B7
    c = DOC[DOC.index("## C."):]
    assert c.count("--oracle --all") >= 2
    for line in (x for x in b.splitlines() + c.splitlines() if "agent_run.py" in x):
        assert 'sudo env PATH="$HOME/agent-venv/bin:$PATH" ~/agent-venv/bin/python' in line
