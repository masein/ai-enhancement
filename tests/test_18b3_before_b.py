"""18b part 3: what step B needs — resolved only when submitted, the agent's
time limit cutting its request in flight and a reply's length capped, a
resume that refuses changed settings, a disk guard that frees what it
counts, --check reading the server's template and cache, the container
hardened and min_p pinned, step B's server lines, and the small ones.

A stand-in llama-server (FakeServer) where a server is needed; invented
tasks and trials; no model, nothing fetched."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_relay
import agent_run as ar
from agent18 import MODEL, run_json, trial
from fake_openai import FakeServer
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_18_agent import post, server_world  # noqa: F401 — server_world is the fixture

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()


# ---------------------------------------------------------------------------
# 10. resolved only when it submitted
# ---------------------------------------------------------------------------

def test_a_working_tree_it_never_submitted_is_not_resolved(tmp_path):
    rd = tmp_path / "r"
    cases = {"inv__s-1": dict(exit_status="Submitted"),               # resolved
             "inv__s-2": dict(exit_status="LimitsExceeded"),          # the step limit: no
             "inv__s-3": dict(exit_status=""),                        # nothing said: no
             "inv__s-4": dict(exit_status="", agent="oracle")}        # the oracle has none
    got = {t: ab.read_trial(trial(rd, t, **kw)) for t, kw in cases.items()}
    assert [got[t]["result"] for t in cases] == ["resolved", "unresolved", "unresolved",
                                                  "resolved"]           # 70df001: all resolved
    assert got["inv__s-2"]["why"] == ("it never submitted (the step limit): the tests passing on "
                                      "its working tree don't count")


# ---------------------------------------------------------------------------
# 11. the time limit cuts the request in flight; a reply's length is capped
# ---------------------------------------------------------------------------

def test_the_relay_caps_a_reply_and_cuts_a_trials_request_in_flight(tmp_path):
    fake = FakeServer()
    relay = agent_relay.Relay(fake.base, "k", ab.SAMPLING).start()
    try:
        post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        sent = fake.requests[-1]
        assert sent["max_tokens"] == agent_relay.MAX_REPLY_TOKENS == 32_768  # 70df001: none
        assert agent_relay.MAX_REPLY_TOKENS < ab.MIN_WINDOW               # fits any window
        post(relay.url, {"model": "m", "max_tokens": 100,
                         "messages": [{"role": "user", "content": "x"}]})
        assert fake.requests[-1]["max_tokens"] == 100
        assert sent["min_p"] == 0.0                                        # 16: Qwen's own
        # a reply still being written when the time is up: cut
        fake.delay_s = 5
        got: dict = {}

        def ask():
            t0 = time.time()
            got["st"], got["body"] = post(relay.url, {"model": "m", "messages": [
                {"role": "user", "content": "x"}]}, headers={"X-Agent-Trial": "t1__agent"})
            got["s"] = time.time() - t0
        th = threading.Thread(target=ask)
        th.start()
        for _ in range(100):
            if fake.in_flight:
                break
            time.sleep(0.05)
        hm.abort_request(relay.url, "t2__agent")                    # another trial's: nothing
        time.sleep(0.3)
        assert th.is_alive()
        hm.abort_request(relay.url, "t1__agent")
        th.join(10)
        assert not th.is_alive() and got["s"] < 3                   # 70df001: up to 3,600 s
        assert got["st"] == 499 and agent_relay.ABORTED in got["body"]
        # only the chat request and the abort go through
        assert post(relay.url, None, path="/agent/stop")[0] in (403, 404)
    finally:
        fake.delay_s = 0
        relay.stop()
        fake.close()


def test_the_agent_cuts_its_request_when_its_time_is_up(tmp_path, monkeypatch):
    from test_18b1_containment import Env, fake_mini, host_agent  # the stand-ins of part 1
    cut: list = []
    monkeypatch.setattr(hm, "abort_request", lambda relay, trial: cut.append((relay, trial)))

    def behaviour(agent):
        while not agent.env.stop.is_set():
            time.sleep(0.01)
        raise hm.Stopped("the agent's time is up")
    agent, t = host_agent(tmp_path, monkeypatch, behaviour)
    env = Env()

    async def go():
        import asyncio
        await asyncio.wait_for(agent.run("x", env, type("C", (), {})()), timeout=0.3)
    import asyncio
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert cut == [("http://127.0.0.1:9/v1", "x__agent")]
    assert env.calls[-1]["command"] == hm.CLEAN
    assert fake_mini                                                   # the same stand-ins


# ---------------------------------------------------------------------------
# 12. a resume carries on only what didn't change
# ---------------------------------------------------------------------------

def test_a_resume_with_other_settings_is_refused_and_the_first_kept(server_world, capsys,  # noqa: F811
                                                                     tmp_path):
    w = server_world
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0
    capsys.readouterr()
    rdir = ar.run_dir(tmp_path, "swebench-multilingual", MODEL, 1, False)
    first = json.loads((rdir / "run.json").read_text())
    for field, value, words in (("window", 131072, "window 262,144 → 131,072"),
                                ("build", "b6600", "build b6500 → b6600"),
                                ("file_sha256", "cd" * 32, "file_sha256 "),
                                ("flags", "--jinja -np 2", "flags --jinja → --jinja -np 2")):
        keep = w["served"][field]
        w["served"][field] = value
        w["harbor"].clear()
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "3",
                        "--no-board"]) == 2, field
        out = capsys.readouterr().out
        assert out.startswith("refused — this run was started with other settings (") and \
            words in out and "add --run 2" in out, out                   # 70df001: carried on
        assert not w["harbor"]
        assert json.loads((rdir / "run.json").read_text()) == first     # 70df001: overwritten
        w["served"][field] = keep
    # the same settings: it carries on, its tasks grown, its first settings kept
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "3", "--no-board"]) == 0
    now = json.loads((rdir / "run.json").read_text())
    assert len(now["tasks"]) == 3 and now["started_at"] == first["started_at"]
    # another run beside it, with its own settings
    w["served"]["window"] = 131072
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--run", "2",
                    "--no-board"]) == 0
    two = json.loads((rdir.parent / (rdir.name + "-2") / "run.json").read_text())
    assert two["window"] == 131072 and first["window"] == 262144


# ---------------------------------------------------------------------------
# 13. the disk guard frees what it counts, and asks before each start
# ---------------------------------------------------------------------------

def test_the_disk_guard_frees_the_build_and_its_cache(server_world, monkeypatch,  # noqa: F811
                                                      capsys):
    w = server_world
    calls: list = []
    real_run = ar.run
    started: list = []

    def run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "ls"]:
            proj = ar.project_of(started[-1][:32])
            return 0, (f"img1 {proj}__abc1234__env\nimg2 {proj}__abc1234__verifier__trial\n"
                       "img3 someone-elses-project\n")
        return real_run(cmd, cwd=cwd, timeout=timeout, env=env)
    real_outcome = w["outcome"]

    def outcome(task, n):
        started.append(task)
        return real_outcome(task, n)
    w["outcome"] = outcome
    monkeypatch.setattr(ar, "run", run)
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0
    rm = [c for c in calls if c[:4] == ["docker", "image", "rm", "-f"]]
    assert len(rm) == 2 and set(rm[0][4:]) == {"img1", "img2"}       # never someone else's
    # 18c: the cache its build made, by id (no host-wide prune) — 18c's tests
    assert ["docker", "builder", "prune", "-f"] not in calls


def test_the_guard_asks_before_each_start_with_what_a_task_took(server_world,  # noqa: F811
                                                                monkeypatch, capsys):
    w = server_world
    asked: list = []
    real_outcome = w["outcome"]

    def outcome(task, n):
        asked.append(task)
        return real_outcome(task, n)
    w["outcome"] = outcome
    # Docker's disk: 61 GB free at the checks, before the first task and when
    # it starts; 49 while it pulls and builds (12 GB at its peak); 61 again
    # once it is gone
    frees = iter([61.0, 61.0, 61.0, 49.0] + [61.0] * 50)
    monkeypatch.setattr(ar, "free_gb", lambda path: next(frees))
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "5", "--no-board"]) == 1
    out = capsys.readouterr().out
    # 61 - 8 (until one is measured) leaves 53: the first starts; then 61 - 12
    # would leave 49: the second is never started (70df001: an image's size)
    assert len(asked) == 1, out
    assert "stopping — Docker's disk (/var/lib/docker) has 61 GB free and the next task needs " \
           "about 12 — 50 GB must stay free" in out, out


# ---------------------------------------------------------------------------
# 14 and 16. --check reads the server's template and its cache
# ---------------------------------------------------------------------------

def agentish(relay_url, model, messages, trial):
    """the agent's request through the relay, stood in for: its reply as the
    agent keeps it (a tool call)"""
    st, body = post(relay_url, {"model": model, "messages": [
        {k: v for k, v in m.items() if k != "extra"} for m in messages],
        "tools": [ar.BASH_TOOL]}, headers={"X-Agent-Trial": trial})
    assert st == 200, body
    return {"role": "assistant", "content": "", "reasoning_content": "look first",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": "ls"})}}]}


@pytest.mark.parametrize("reused", [True, False])
def test_check_reads_what_the_server_took_from_its_cache(reused):
    fake = FakeServer()
    first = [{"role": "system", "content": "s"}, {"role": "user", "content": "task " * 50}]

    def timings(body):
        n = len(body["messages"])
        whole = 6000 if n == 2 else 6800
        cached = 0 if n == 2 else (6500 if reused else 0)
        return {"cache_n": cached, "prompt_n": whole - cached, "prompt_ms": 6.0 * (whole - cached),
                "predicted_n": 400, "predicted_ms": 5000.0}
    fake.timings = timings
    relay = agent_relay.Relay(fake.base, "k", ab.SAMPLING).start()
    try:
        why, speeds = ar.cache_line(relay, "m", "mini.yaml", ask=agentish, first=first)
    finally:
        relay.stop()
        fake.close()
    if reused:
        assert why == "" and speeds["reread"] == 300 and speeds["whole"] == 6800
        assert round(speeds["prefill"]) == 167 and round(speeds["decode"]) == 80
        words = ar.estimate_words(speeds, 300)
        assert words.startswith("it read 167 prompt tokens a second and wrote 80: about ")
        assert "days for 300 tasks, one at a time" in words
    else:
        assert why.startswith("the second step read 6,800 of its 6,800 prompt tokens again: the "
                              "server didn't reuse its cache")


def test_check_refuses_a_template_that_rerenders_drops_thinking_or_closes_it():
    msgs_text = lambda body: "".join(  # noqa: E731
        f"<{m['role']}>{m.get('reasoning_content', '')}|{m.get('content', '')}"
        for m in body["messages"]) + "<assistant>"

    def good(base, key, body):
        return msgs_text(body)

    def rerenders(base, key, body):        # the last reply's thinking shown, earlier ones not
        ms = body["messages"]
        last = max(i for i, m in enumerate(ms) if m["role"] == "assistant")
        return "".join(f"<{m['role']}>{m.get('reasoning_content', '') if i == last else ''}|"
                       for i, m in enumerate(ms)) + "<assistant>"

    def closes(base, key, body):
        return msgs_text(body) + "<think>\n\n</think>\n\n"

    def drops(base, key, body):
        return "".join(f"<{m['role']}>|{m.get('content', '')}" for m in body["messages"]) + \
            "<assistant>"
    assert ar.template_line("http://x/v1", "k", apply=good) == ""
    assert "renders the conversation differently once step 2 follows" in ar.template_line(
        "http://x/v1", "k", apply=rerenders)
    assert "thinking is off" in ar.template_line("http://x/v1", "k", apply=closes)
    assert "leaves out the thinking of step 1, 2, 3" in ar.template_line("http://x/v1", "k",
                                                                          apply=drops)


def test_check_runs_both_and_says_the_estimate(server_world, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(ar, "template_line", lambda base, key: "")
    monkeypatch.setattr(ar, "cache_line", lambda relay, model, config: (
        "", {"prefill": 900.0, "decode": 85.0, "reread": 900, "whole": 50000}))
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--check"]) == 0
    out = capsys.readouterr().out
    assert "it read 900 prompt tokens a second and wrote 85: about " in out
    assert out.rstrip().splitlines()[-1].startswith("ready — 10 task(s)")
    monkeypatch.setattr(ar, "cache_line", lambda relay, model, config: ("the second step read "
                                                                        "50,000 …", {}))
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "10", "--check"]) == 2
    assert "refused — the second step read 50,000" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 15. an agent run holds the model: the board says so
# ---------------------------------------------------------------------------

def busy_run(rdir: Path, model: str, at: float) -> None:
    import import_agent
    run_json(rdir, ["inv__repo1-1001"], model=model)
    (rdir / "progress.json").write_text(json.dumps({
        "line": "1 of 10 · 0 resolved · 30 min a task · about 4 h left", "at": at,
        "until": time.time() + 4 * 3600}))
    assert import_agent.main(["--progress", str(rdir)]) == 0


def test_the_board_knows_which_model_an_agent_run_holds(svc):  # noqa: F811
    from service import agent_runs, config
    root = Path(config.BENCH_ROOT) / "agent-runs"
    busy_run(root / "swebench-multilingual__held__k1", "served/held", time.time())
    busy_run(root / "swebench-multilingual__gone__k1", "served/gone", time.time() - 3600)
    got = agent_runs.busy()
    assert set(got) == {"served/held"}                    # a runner silent for an hour: not
    assert got["served/held"]["until"] > time.time() + 3 * 3600
    import service.app as appmod
    assert set(appmod.agent_busy()["models"]) == {"served/held"}


# ---------------------------------------------------------------------------
# 16. the container hardened
# ---------------------------------------------------------------------------

def test_the_task_container_drops_its_capabilities_and_privileges():
    y = ar.OVERRIDE_YAML
    assert "no-new-privileges:true" in y and "cap_drop:\n      - ALL" in y
    assert y.split("cap_add:")[1].split() == ["-", "CHOWN", "-", "DAC_OVERRIDE", "-", "FOWNER"]
    ok = "PROBE tcp-ok\nNET lo\nPIDS 4096\nCAPS {}\nNNP {}\nDONE"
    assert hm.reach_verdict(ok.format("000000000000000b", 1)) == ""
    docker_default = "00000000a80425fb"
    assert "keeps capabilities: FSETID, KILL, SETGID, SETUID, SETPCAP, NET_BIND_SERVICE, " \
           "NET_RAW, SYS_CHROOT, MKNOD, AUDIT_WRITE, SETFCAP" in hm.reach_verdict(
               ok.format(docker_default, 1))
    assert "can gain privileges" in hm.reach_verdict(ok.format("000000000000000b", 0))
    assert ab.SAMPLING["min_p"] == 0.0


# ---------------------------------------------------------------------------
# 17. step B's server lines
# ---------------------------------------------------------------------------

def test_step_b_gives_the_servers_own_lines():
    b = DOC[DOC.index("## B."):DOC.index("## C.")]
    assert "sudo ss -ltnp 'sport = :8091'" in b and "kill <PID>" in b     # 18c: by its port
    assert 'CTX=262144 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0 > ~/lda-orig.log ' \
           '2>&1 &' in b and "~/serve-base-la0-mtp0.log" in b
    assert 'CTX=131072 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0' in b
    assert 'CTX=65536 CPU_MOE="--n-cpu-moe 21" nohup ~/lda-serve.sh base 0 0' in b
    assert "never the phone build" in b.lower() and "gemma-vllm" in b
    assert "Use the new window" not in b                    # the runner takes the window alone


# ---------------------------------------------------------------------------
# 18. the small ones
# ---------------------------------------------------------------------------

def test_the_key_is_never_printed(server_world, capsys):  # noqa: F811
    w = server_world
    import agent_run
    real = agent_run.run

    def run(cmd, cwd=None, timeout=None, env=None):
        if cmd[:len(ar.BOARD)] == ar.BOARD and cmd[len(ar.BOARD)] == "--served":
            return 0, json.dumps({**w["served"], "why": "its server doesn't answer"}) + \
                "\nWARNING: a line after the JSON\n"                       # 70df001: printed
        return real(cmd, cwd=cwd, timeout=timeout, env=env)
    agent_run.run = run
    try:
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1"]) == 2
        out = capsys.readouterr().out
        assert "its server doesn't answer" in out and "k18" not in out
        agent_run.run = lambda cmd, **kw: (1, '{"key": "k18", "base_url"\nkey=k18 went wrong')
        info, why = ar.served_info(MODEL)
        assert "k18" not in why and why.startswith("the board didn't answer")
    finally:
        agent_run.run = real


def test_a_pilot_counts_the_tasks_asked_errors_included(tmp_path):
    rd = tmp_path / "r"
    trial(rd, "inv__p-1")
    trial(rd, "inv__p-2", result="", exc="RuntimeError", started=False,
          message="Docker compose command failed: pull")
    rs = ar.results(rd, ["inv__p-1", "inv__p-2", "inv__p-3"], 1)
    s = ab.score(rs, 300)
    assert ab.score_words(s, 300) == ("50.0% resolved ± 35.4 (one standard error) · 2 tasks · "
                                      "pilot: 2 of 300 · 1 error of ours, counted not resolved")


def test_the_trial_helper_is_utc_wherever_it_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Dubai")
    time.tzset()
    try:
        t = trial(tmp_path / "r", "inv__tz-1", minutes=20.0)
        assert ab.read_trial(t)["minutes"] == 20.0                    # 70df001: 260.0
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
    assert os.environ.get("TZ") is None
