"""18: the agent runs — the runner on invented tasks with a stand-in for
Harbor (carrying on after a kill, the disk guard, each refusal before the
first task, the import, a part-run's label), the relay (the key never
reaches the tasks' side; anything but the chat request is refused), and the
host agent's adapter. No model, no benchmark's task, no paid API."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_relay
import agent_run as ar
from agent18 import LANGS, MODEL, conversation, run_json, tasks_folder, trial
from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture

# ---------------------------------------------------------------------------
# the benchmarks' data: a pilot's tasks, a task's result, a score
# ---------------------------------------------------------------------------


def test_a_pilot_is_the_same_tasks_every_time_spread_over_the_languages(tmp_path):
    names = tasks_folder(tmp_path)
    langs = ar.languages(tmp_path, ab.BENCHES["swebench-multilingual"])
    assert set(langs.values()) == set(LANGS)
    ten = ab.pilot(langs, 10)
    assert ten == ab.pilot(dict(reversed(list(langs.items()))), 10)     # the order given: none
    assert len(ten) == 10 and len({langs[t] for t in ten}) == 8          # every language
    assert set(ab.pilot(langs, 3)) <= set(ten) and len(ab.pilot(langs, 99)) == len(names)
    # DeepSWE names the language in its metadata
    d = tmp_path / "deep"
    tasks_folder(d, 10, how="metadata")
    assert set(ar.languages(d, ab.BENCHES["deepswe"]).values()) == set(LANGS)


def test_a_tasks_result_in_the_boards_words(tmp_path):
    rd = tmp_path / "r"
    docker = "Docker compose command failed for environment x. Command: docker compose up"
    cases = {"inv__a-1": dict(result="resolved"), "inv__a-2": dict(result="unresolved"),
             "inv__a-3": dict(result="", exc="AgentTimeoutError"),
             "inv__a-4": dict(result="", exc="RuntimeError", message=docker, started=False),
             "inv__a-5": dict(result="", exc="ReachRefused", ours="ReachRefused: reached x")}
    for t, kw in cases.items():
        trial(rd, t, **kw)
    got = {t: ab.read_trial(next((rd / "jobs").glob(f"{t}__*/*"))) for t in cases}
    assert [got[t]["result"] for t in cases] == ["resolved", "unresolved", "timeout", "error",
                                                  "error"]
    assert got["inv__a-4"]["why"].startswith("RuntimeError: Docker compose command failed")
    assert got["inv__a-1"]["tokens_out"] == 340 and got["inv__a-1"]["minutes"] == 20.0
    s = ab.score(list(got.values()), 300)
    # 18b: every task attempted is in the denominator, an error of ours as not
    # resolved, and said
    assert (s["resolved"], s["n"], s["errors"], s["part"]) == (1, 5, 2, True)
    assert ab.score_words(s, 300) == ("20.0% resolved ± 17.9 (one standard error) · 5 tasks · "
                                      "pilot: 5 of 300 · 2 errors of ours, counted not resolved")
    assert ab.RESULT_WORDS == {"resolved": "Resolved", "unresolved": "Not resolved",
                               "timeout": "Timed out", "error": "Error (ours)"}


# ---------------------------------------------------------------------------
# the relay: the key never reaches the tasks' side; only the chat request goes
# ---------------------------------------------------------------------------

def post(url: str, body, path="/chat/completions", method="POST", headers=None):
    req = urllib.request.Request(url + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer what-the-agent-sends",
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_the_relay_adds_the_key_takes_only_the_chat_request_and_sets_the_sampling(tmp_path):
    fake = FakeServer()
    fake.key = "the-servers-key-18"
    usage = tmp_path / "relay-usage.jsonl"
    relay = agent_relay.Relay(fake.base, fake.key, {**ab.SAMPLING,
                              "chat_template_kwargs": {"enable_thinking": True}},
                              usage=usage).start()
    try:
        assert relay.url.startswith("http://127.0.0.1:")
        st, body = post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                    "temperature": 0.0, "stream": True},
                        headers={"X-Agent-Trial": "inv__a-1__abc__agent"})
        assert st == 200, body
        assert fake.auth[-1] == "Bearer the-servers-key-18"                # added by the relay
        sent = fake.requests[-1]
        assert (sent["temperature"], sent["top_p"], sent["top_k"], sent["presence_penalty"]) \
            == (1.0, 0.95, 20, 0.0) and sent["stream"] is False             # the run's
        assert sent["chat_template_kwargs"] == {"enable_thinking": True}
        # anything but the chat request is refused, the key nowhere
        for path, method, b in (("/models", "GET", None), ("/props", "GET", None),
                                ("/../props", "GET", None), ("/completions", "POST", {"x": 1}),
                                ("/chat/completions", "GET", None),
                                ("/embeddings", "POST", {"input": "x"})):
            st, body = post(relay.url, b, path=path, method=method)
            assert st in (403, 404, 405), (path, st)
            assert "the-servers-key-18" not in body
        st, body = post(relay.url, {"no": "messages"})
        assert st == 400
        lines = [json.loads(x) for x in usage.read_text().splitlines()]
        assert lines[0]["trial"] == "inv__a-1__abc__agent" and lines[0]["status"] == 200
        assert "the-servers-key-18" not in usage.read_text()
    finally:
        relay.stop()
        fake.close()


def test_a_window_outgrown_comes_back_as_the_servers_own_refusal(tmp_path):
    fake = FakeServer()
    fake.chat_error = lambda body: (400, "request (300000 tokens) exceeds the available context "
                                         "size (262144 tokens), try increasing it")
    relay = agent_relay.Relay(fake.base, "", ab.SAMPLING).start()
    try:
        st, body = post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        assert st == 400 and "exceeds the available context size" in body
    finally:
        relay.stop()
        fake.close()


# ---------------------------------------------------------------------------
# the host agent's adapter: mini-swe-agent's command, as Harbor runs it
# ---------------------------------------------------------------------------

class Exec:
    """Harbor's exec, stood in for: what it was asked, and what it answers"""

    def __init__(self, answer):
        self.calls, self.answer = [], answer

    async def __call__(self, command, cwd=None, env=None, timeout_sec=None):
        self.calls.append({"command": command, "cwd": cwd, "env": env, "timeout": timeout_sec})
        return self.answer(command)


def adapter(answer, **kw):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    ex = Exec(answer)
    return hm.HarborEnv(ex, loop, **kw), ex, loop


def test_a_command_runs_as_mini_swe_agent_runs_it():
    env, ex, loop = adapter(lambda c: SimpleNamespace(stdout="out\n", stderr="err\n",
                                                      return_code=3),
                            cwd="/testbed", timeout=60, interpreter=("bash", "-c"),
                            env={"PAGER": "cat"}, kind="docker")
    try:
        got = env.execute({"command": "ls 'a b'"})
        call = ex.calls[-1]
        assert call["command"] == "timeout -s KILL 60 bash -c 'ls '\"'\"'a b'\"'\"'' 2>&1"
        assert call["cwd"] == "/testbed" and call["env"] == {"PAGER": "cat"}
        assert got == {"output": "out\nerr\n", "returncode": 3, "exception_info": ""}
        # the default config's local environment: sh -c, its own working folder
        env2, ex2, loop2 = adapter(lambda c: SimpleNamespace(stdout="", stderr="", return_code=0))
        env2.execute({"command": "pwd"})
        assert ex2.calls[-1]["command"].startswith("timeout -s KILL 30 sh -c ")
        assert ex2.calls[-1]["cwd"] is None
        loop2.call_soon_threadsafe(loop2.stop)
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_a_command_past_its_limit_and_the_submission_line():
    def slow(c):
        time.sleep(1.05)
        return SimpleNamespace(stdout="partial", stderr="", return_code=137)
    env, ex, loop = adapter(slow, timeout=1)
    try:
        got = env.execute({"command": "sleep 99"})
        assert got["returncode"] == -1 and "timed out after 1 seconds" in got["exception_info"]
        env2, _, loop2 = adapter(lambda c: SimpleNamespace(
            stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/x b/x\n", stderr="",
            return_code=0))
        with pytest.raises(hm.Submitted) as e:
            env2.execute({"command": "echo done"})
        assert e.value.messages[0]["extra"]["submission"] == "diff --git a/x b/x\n"
        # Harbor's time limit: the next command stops the loop
        env2.stop.set()
        with pytest.raises(hm.Stopped):
            env2.execute({"command": "ls"})
        loop2.call_soon_threadsafe(loop2.stop)
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_the_containers_check_and_the_tokens_from_the_trajectory():
    script = hm.reach_script(["172.17.0.1:8090", "1.1.1.1:443"])
    assert '/dev/tcp/$1/$2' in script and "for t in 172.17.0.1:8090 1.1.1.1:443;" in script
    ok = "PROBE tcp-ok\nNET lo\nCAPS 000000000000000b\nNNP 1\n{}\nDONE"
    assert hm.reach_verdict(ok.format("PIDS 4096")) == ""
    assert hm.reach_verdict(ok.format("REACHED 172.17.0.1:8090\nPIDS 4096")) == \
        "the task's container reached 172.17.0.1:8090"
    assert hm.reach_verdict(ok.format("REACHED docker.sock\nPIDS 4096")).endswith("docker.sock")
    assert hm.reach_verdict(ok.format("PIDS max")) == "the task's container has no process limit"
    assert hm.usage_of(conversation()["messages"]) == (2800, 340, 2)


def test_the_conversation_as_the_board_reads_it(tmp_path):
    t = trial(tmp_path / "r", "inv__a-1", html=True)
    c = ab.conversation(tmp_path / "r", t.parent.name, t.name)
    first = c["steps"][0]
    assert first["thought"] == "thinking about <i>it</i>" and first["ran"][0].startswith("ls -la")
    assert first["came_back"][0].count("\n") > 40                   # long: folded on the page
    assert c["steps"][-1] == {"exit": "Submitted"}
    assert c["patch"].startswith("--- a/x.c") and "PASSED" in c["verifier"]


# ---------------------------------------------------------------------------
# the runner: refusals before the first task, carrying on, the disk guard
# ---------------------------------------------------------------------------

@pytest.fixture
def server_world(tmp_path, monkeypatch):
    """the server, stood in for: its board, Docker and Harbor — answers set
    per test; Harbor's run is a stand-in that writes the trial folder"""
    monkeypatch.setenv("BENCH_ROOT", str(tmp_path))
    names = tasks_folder(tmp_path / "agent-tasks" / ar.slug("swebench_multilingual@1.0"))
    w = {"versions": {"harbor": "0.24.0", "mini-swe-agent": "2.4.6", "litellm": "1.80"},
         "docker": (0, "/var/lib/docker\n"), "free": 400.0, "served": {
             "why": "", "base_url": "http://host.docker.internal:8090/v1", "key": "k18",
             "model": "invented.gguf", "window": 262144, "file_sha256": "ab" * 32,
             "build": "b6500", "flags": "--jinja"}, "tool": "", "board": [], "harbor": [],
         "outcome": lambda task, n: "resolved", "kill_at": None, "names": names}
    monkeypatch.setattr(ar, "versions", lambda: dict(w["versions"]))
    monkeypatch.setattr(ar, "free_gb", lambda path: w["free"])
    monkeypatch.setattr(ar, "tool_call_line", lambda url, model: w["tool"])
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)

    def run(cmd, cwd=None, timeout=None, env=None):
        if cmd[:2] == ["docker", "info"]:
            return w["docker"]
        if cmd[:len(ar.BOARD)] == ar.BOARD:
            w["board"].append(cmd[len(ar.BOARD):])
            if cmd[len(ar.BOARD)] == "--served":
                return 0, json.dumps(w["served"])
            return 0, "Runs #1: ok"
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 0, "172.17.0.1\n"
        return 0, ""
    monkeypatch.setattr(ar, "run", run)

    class Popen:
        def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None):
            task = Path(cmd[cmd.index("-p") + 1]).name
            job = cmd[cmd.index("--job-name") + 1]
            w["harbor"].append({"task": task, "cmd": cmd, "env": env})
            n = sum(1 for h in w["harbor"] if h["task"] == task)
            if w["kill_at"] is not None and len(w["harbor"]) == w["kill_at"]:
                raise KeyboardInterrupt                 # the run is killed here
            out = w["outcome"](task, n)
            rd = Path(cmd[cmd.index("-o") + 1]).parent
            k = int(job.split("__a")[1].split("__")[0])
            # "docker": Docker failed before the agent ran (ours); an
            # exception's name: raised while the model worked (the model's)
            kw = (dict(result=out) if out in ("resolved", "unresolved") else
                  dict(result="", exc="RuntimeError", started=False,
                       message="Docker compose command failed for environment x")
                  if out == "docker" else dict(result="", exc=out))
            trial(rd, task, k, stamp=int(job.rsplit("__", 1)[1]) * 1000 + n, **kw)

        def poll(self):
            return 0
    monkeypatch.setattr(ar.subprocess, "Popen", Popen)
    return w


def test_each_refusal_before_the_first_task_in_one_line(server_world, capsys):
    w = server_world
    cases = [
        ("versions", {"harbor": "0.23.1", "mini-swe-agent": "2.4.6"}, "Harbor 0.23.1 — 0.24.0 is pinned"),
        ("docker", (1, "permission denied"), "Docker doesn't answer this user"),
        ("free", 42.0, "has 42 GB free — 50 GB must stay free"),
        ("served", {"why": "The server now serves a different file than the one registered."},
         "a different file"),
        ("served", {**w["served"], "window": 16384}, "a window of 16,384 tokens; an agent run "
                                                     "needs at least 131,072"),
        ("tool", "its server answered with no tool call", "no tool call")]
    for field, value, words in cases:
        keep = w[field]
        w[field] = value
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10"]) == 2, field
        out = capsys.readouterr().out
        assert out.startswith("refused — ") and words in out, (field, out)
        assert not w["harbor"]                                      # nothing was asked
        w[field] = keep


def test_it_carries_on_after_a_kill_and_never_asks_a_finished_task_again(server_world, capsys,
                                                                       tmp_path):
    w = server_world
    w["kill_at"] = 4
    with pytest.raises(KeyboardInterrupt):
        ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--no-board"])
    asked = [h["task"] for h in w["harbor"]]
    w["kill_at"] = None
    w["harbor"].clear()
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--no-board"]) == 0
    again = [h["task"] for h in w["harbor"]]
    assert set(again) == set(ab.pilot(ar.languages(ar.tasks_dir(tmp_path, ab.BENCHES[
        "swebench-multilingual"]), ab.BENCHES["swebench-multilingual"]), 10)) - set(asked[:3])
    assert asked[3] in again                                    # killed: asked again
    assert not set(asked[:3]) & set(again)                      # finished: never again
    # the agent and the relay's address went to Harbor; the key never did
    cmd = w["harbor"][-1]["cmd"]
    assert "agent_host_mini:HostMini" in cmd and "openai/invented.gguf" in cmd
    assert not any("k18" in x for x in cmd) and "k18" not in json.dumps(
        {k: v for k, v in (w["harbor"][-1]["env"] or {}).items() if k != "PATH"})
    assert any(x.startswith("relay=http://127.0.0.1:") for x in cmd)
    run = ar.read_json(ar.run_dir(tmp_path, "swebench-multilingual", MODEL, 1, False) / "run.json")
    for k in ("versions", "sampling", "window", "prompt_sha256", "file_sha256", "build", "flags",
              "tasks_sha256"):
        assert k in run, k                                          # what the run leaves
    assert run["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                               "presence_penalty": 0.0, "thinking": "on"}


def test_an_error_of_ours_is_asked_again_three_times_at_most(server_world, capsys):
    w = server_world
    first = []

    def outcome(task, n):
        first.append(task) if not first else None
        return "docker" if task == first[0] else "resolved"
    w["outcome"] = outcome
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "3", "--no-board"]) == 0
    tries = [h["task"] for h in w["harbor"] if h["task"] == first[0]]
    assert len(tries) == 3
    assert "given up after 3" in capsys.readouterr().out


def test_the_disk_guard_stops_before_the_next_pull_and_says_so(server_world, capsys):
    w = server_world
    pulled = []
    real = w["outcome"]

    def outcome(task, n):
        pulled.append(task)
        if len(pulled) == 2:
            w["free"] = 52.0                    # under 50 GB once the next image comes
        return real(task, n)
    w["outcome"] = outcome
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--no-board"]) == 1
    out = capsys.readouterr().out
    assert "stopping — Docker's disk (/var/lib/docker) has 52 GB free and the next task needs " \
        "about 8 — 50 GB must stay free" in out, out
    assert len(w["harbor"]) == 2


def test_the_oracle_needs_no_model(server_world, capsys):
    w = server_world
    w["served"] = {"why": "never asked"}
    assert ar.main(["swebench-multilingual", "--oracle", "--tasks", "3", "--no-board"]) == 0
    assert all(h["cmd"][h["cmd"].index("-a") + 1] == "oracle" for h in w["harbor"])
    assert not [b for b in w["board"] if b and b[0] == "--served"]


# ---------------------------------------------------------------------------
# the board: the Runs row, the import, a part-run's label, agent rows left alone
# ---------------------------------------------------------------------------

def test_the_import_makes_one_runs_row_and_says_a_part_run(svc, monkeypatch):  # noqa: F811
    import import_agent
    from service import config, db
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / "swebench-multilingual__invented-agent__k1"
    names = [f"inv__repo{i}-{1000 + i}" for i in range(10)]
    run_json(rdir, names)
    for i, t in enumerate(names):
        trial(rdir, t, result="resolved" if i < 6 else "unresolved")
    (rdir / "progress.json").write_text(json.dumps({"line": "10 of 10 · 6 resolved"}))
    assert import_agent.main(["--progress", str(rdir)]) == 0
    assert import_agent.main([str(rdir)]) == 0
    rows = [r for r in db.recent(50) if r.get("suite") == "agent"]
    assert len(rows) == 1                                       # one row for the run
    r = rows[0]
    assert r["status"] == "done" and r["hf_id"] == MODEL and not r.get("where_ran")
    assert r["progress"] == ("60.0% resolved ± 15.5 (one standard error) · 10 tasks · pilot: 10 "
                             "of 300")
    # an agent row is the host's runner's: never re-queued, never the worker's,
    # never holding a GGUF job back
    db.update(r["id"], status="running")
    db.init(startup=True)
    assert db.get(r["id"])["status"] == "running"                  # 06ee7e8: re-queued
    db.update(r["id"], status="queued")
    nxt = db.claim_next()
    assert not nxt or nxt.get("suite") != "agent"
    db.update(r["id"], status="running")
    g = db.add("served/x", "instruct", "gguf", "masein", "", status="queued")
    assert db.board_ahead(g) is None                                 # 06ee7e8: held back


def test_deepswe_s_tasks_are_fetched_at_their_commit(tmp_path, monkeypatch):
    b = ab.BENCHES["deepswe"]
    calls = []

    def run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        if cmd[-2:] == ["checkout", "-q"] or cmd[-1] == "FETCH_HEAD":
            tasks_folder(ar.tasks_dir(tmp_path, b), 4, how="metadata")
        return 0, ""
    monkeypatch.setattr(ar, "run", run)
    d, sha = ar.fetch_tasks(tmp_path, b)
    assert d.name == "tasks" and len(list(d.glob("*/task.toml"))) == 4
    fetch = next(c for c in calls if "fetch" in c)
    assert fetch[-2:] == [b["git"], "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"]
    assert not any(c[0] == "harbor" for c in calls)
    assert ar.fetch_tasks(tmp_path, b) == (d, sha) and len(calls) == 3      # once
