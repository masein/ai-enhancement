"""18d, part 2: agent runs, before step C and A's docs now (docs/prompts/phase-
18d-eleventh-review-scores-oracle-speed.md, points 10–19). A test a point,
each failing on 0b58df7: the reviewers' oracle_run2, oracle_givenup and
ours_leaks cases, a planted reward.json link and a sparse reward.txt, the
firewall's rules, a second runner. Harbor, Docker and the model are stand-ins;
no model runs."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_run as ar
from agent18 import FIREWALL, MODEL, run_json, trial
from test_18_agent import server_world  # noqa: F401 — server_world is the fixture

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()


def rdir_of(model: str = MODEL, oracle: bool = False, number: int = 1) -> Path:
    return ar.run_dir(Path(os.environ["BENCH_ROOT"]), "swebench-multilingual", model, 1, oracle,
                      number)


# ---------------------------------------------------------------------------
# 10. a pull locks nothing; the oracle's latest result across its runs
# ---------------------------------------------------------------------------

def test_10_a_pull_resumes_and_the_oracles_run_2_is_read(server_world, monkeypatch,  # noqa: F811
                                                         tmp_path):
    w = server_world
    monkeypatch.setattr(ar, "runner_commit", lambda: "a" * 40)
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0
    monkeypatch.setattr(ar, "runner_commit", lambda: "b" * 40)       # a pull, the files the same
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0
    # oracle_run2: run 1 says a task doesn't pass, run 2 (after a pull) that it does
    root = Path(os.environ["BENCH_ROOT"])
    names = w["names"]
    one = rdir_of("", True)
    run_json(one, names[:2], model="")
    trial(one, names[0], result="unresolved", agent="oracle", exit_status="")
    trial(one, names[1], agent="oracle", exit_status="")
    two = rdir_of("", True, 2)
    run_json(two, names[:1], model="")
    trial(two, names[0], agent="oracle", exit_status="")
    got = ar.oracle_states(root, "swebench-multilingual", [])
    assert got["left_out"] == [] and got["unknown"] == []     # 0b58df7: run 1 alone, left out
    assert [p.name for p in ar.oracle_dirs(root, "swebench-multilingual")] == [one.name, two.name]


# ---------------------------------------------------------------------------
# 11. an oracle task given up, or erring after it ran: left out by name
# ---------------------------------------------------------------------------

def test_11_a_given_up_oracle_task_is_left_out_by_name_and_never_blocks_all(server_world, capsys):  # noqa: F811
    w = server_world
    root = Path(os.environ["BENCH_ROOT"])
    names = w["names"]
    od = rdir_of("", True)
    run_json(od, names, model="")
    for t in names[2:]:
        trial(od, t, agent="oracle", exit_status="")
    for k in range(ar.TRIES):                                 # oracle_givenup: three before
        trial(od, names[0], result="", exc="RuntimeError", started=False, stamp=1000 + k,
              message="Docker compose command failed: pull")
    trial(od, names[1], result="", exc="CleanupFailed", agent="oracle", exit_status="",
          message="the clean-up …: not verified")            # an error after it ran
    got = ar.oracle_states(root, "swebench-multilingual", [])
    assert got["unverified"] == sorted(names[:2]) and got["unknown"] == []   # 0b58df7: unknown
    assert sorted(got["left_out"]) == sorted(names[:2])
    capsys.readouterr()
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--only", ",".join(names[:4]),
                    "--no-board"]) == 0
    out = capsys.readouterr().out
    assert "2 of these tasks are left out: they couldn't be checked here (the oracle run's " \
           "errors of ours; --oracle --run N+1 --only them asks them again)" in out
    assert not {h["task"] for h in w["harbor"]} & set(names[:2])
    # --all isn't blocked for ever: every task has a result or is left out by name
    w["harbor"].clear()
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--all", "--no-board"]) == 0


# ---------------------------------------------------------------------------
# 12–13. no result, or a stop, after the agent ran: final
# ---------------------------------------------------------------------------

def test_12_a_job_with_no_result_after_the_agent_ran_is_final(tmp_path):
    job = tmp_path / "jobs" / "inv__e-1__a1__1"
    t = job / "inv__e-1__abc"
    (t / ab.HOST_DIR).mkdir(parents=True)
    (t / ab.HOST_DIR / ab.TRAJECTORY).write_text("{}")         # the agent had run
    r = ab.no_result(job, None)
    assert (r["ours"], r["final"]) == ("after", True)          # 0b58df7: before, asked again
    assert "never asked again" in r["why"]
    job2 = tmp_path / "jobs" / "inv__e-2__a1__1"
    (job2 / "inv__e-2__abc").mkdir(parents=True)               # nothing of the agent's
    assert (ab.no_result(job2, None)["ours"], ab.no_result(job2, None)["final"]) == (
        "before", False)


def test_13_ours_leaks_four_stops_after_the_agent_then_successes_score_nothing(tmp_path):
    rd = tmp_path / "r"
    tasks = [f"inv__s-{i}" for i in range(4)]
    for i, t in enumerate(tasks):
        trial(rd, t, result="", exc="CancelledError", stamp=1000 + i)   # stopped while it ran
        trial(rd, t, stamp=2000 + i)                                     # then resolved
    rs = ar.results(rd, tasks, 1)
    assert all(r["result"] == "error" and r["final"] for r in rs)
    s = ab.score(rs, 4)
    assert s["k"] == 0 if "k" in s else s["score"] == 0.0      # 0b58df7: 100%
    assert all(ar.state_of(rd, t, 1)[0] == "done" for t in tasks)
    # before the agent started, a stop is asked again
    r = ab.read_trial(trial(rd, "inv__s-9", result="", exc="KeyboardInterrupt", started=False))
    assert (r["ours"], r["final"]) == ("before", False)


# ---------------------------------------------------------------------------
# 14. the reward file, read on the host: no link followed, capped
# ---------------------------------------------------------------------------

def test_14_safe_read_follows_no_link_and_reads_no_fifo_or_large_file(tmp_path):
    d = tmp_path / "verifier"
    d.mkdir()
    secret = tmp_path / "config.json"
    secret.write_text('{"auths": {"x": "a-docker-login-shaped-secret"}}')
    (d / "reward.json").symlink_to(secret)
    assert hm.safe_read(d, "reward.json") == ""                # 0b58df7's Harbor: quoted it
    os.mkfifo(d / "reward.txt")
    assert hm.safe_read(d, "reward.txt") == ""                 # never waited on
    (d / "reward.txt").unlink()
    with open(d / "reward.txt", "wb") as fh:                   # sparse: 150 GB, no disk
        fh.truncate(150 * 1024 ** 3)
    assert hm.safe_read(d, "reward.txt") == ""
    (d / "reward.txt").unlink()
    (d / "reward.txt").write_text("1.0\n")
    assert hm.safe_read(d, "reward.txt") == "1.0\n"
    os.link(d / "reward.txt", tmp_path / "elsewhere")          # a hard link to it: not read
    assert hm.safe_read(d, "reward.txt") == ""
    assert hm.safe_read(d, "../config.json") == ""


def test_14_harbors_verifier_reads_through_it(tmp_path, monkeypatch):
    class Empty(Exception):
        pass

    class Parse(Exception):
        pass

    class Verifier:
        def _parse_reward_text(self):
            raise AssertionError("Harbor's own read")

        _parse_reward_json = _parse_reward_text
    mod = types.ModuleType("harbor.verifier.verifier")
    mod.Verifier, mod.RewardFileEmptyError, mod.VerifierOutputParseError = Verifier, Empty, Parse
    for name, m in (("harbor", types.ModuleType("harbor")),
                    ("harbor.verifier", types.ModuleType("harbor.verifier")),
                    ("harbor.verifier.verifier", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    assert hm.patch_reward_reading() is True
    d = tmp_path / "verifier"
    d.mkdir()
    v = Verifier()
    v.trial_paths = types.SimpleNamespace(reward_text_path=d / "reward.txt",
                                          reward_json_path=d / "reward.json")
    secret = tmp_path / "config.json"
    secret.write_text('{"reward": "a-docker-login-shaped-secret"}')
    (d / "reward.json").symlink_to(secret)
    with pytest.raises(Empty) as e:
        v._parse_reward_json()
    assert "secret" not in str(e.value)
    (d / "reward.json").unlink()
    (d / "reward.json").write_text('{"reward": "a-docker-login-shaped-secret"}')
    with pytest.raises(Parse) as e:
        v._parse_reward_json()
    assert "secret" not in str(e.value)                        # 0b58df7's Harbor: quoted it
    (d / "reward.json").write_text('{"reward": 1.0}')
    assert v._parse_reward_json() == {"reward": 1.0}
    # Multilingual writes reward.txt alone: a reward.json is never read
    monkeypatch.setenv(hm.REWARD_ONLY, "reward.txt")
    (d / "reward.txt").write_text("0")
    assert v._parse_reward_json() == {"reward": 0.0}
    assert ab.BENCHES["swebench-multilingual"]["reward_file"] == "reward.txt"
    assert ar.job_env(ab.BENCHES["swebench-multilingual"])["EVALBOARD_REWARD_FILE"] == \
        "reward.txt"
    assert "EVALBOARD_REWARD_FILE" not in ar.job_env(ab.BENCHES["deepswe"])


def test_14_the_run_folder_is_its_owners_alone(server_world):  # noqa: F811
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1", "--no-board"]) == 0
    assert rdir_of().stat().st_mode & 0o777 == 0o700           # 0b58df7: world-readable


# ---------------------------------------------------------------------------
# 15. the default bridge's firewall: DeepSWE refuses without it
# ---------------------------------------------------------------------------

def test_15_the_firewalls_rules_are_read_and_each_missing_one_named(monkeypatch):
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (0, FIREWALL))
    assert ar.firewall_missing() == ""
    partial = "\n".join(x for x in FIREWALL.splitlines()
                        if "INPUT -i" not in x and ar.TAILNET_RANGE not in x)
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (0, partial))
    why = ar.firewall_missing()
    assert "INPUT from docker0" in why and f"DOCKER-USER to {ar.TAILNET_RANGE}" in why
    assert "a reboot loses them" in why
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (1, "Permission denied (you must be root)"))
    assert "run this line in its sudo form" in ar.firewall_missing()


def test_15_every_deepswe_line_refuses_while_the_rules_are_missing(server_world, capsys):  # noqa: F811
    w = server_world
    w["firewall"] = ""                                         # after a reboot
    assert ar.main(["deepswe", "--oracle", "--tasks", "3", "--no-board"]) == 2
    assert "refused — the firewall rules for Docker's default bridge aren't in place" in \
        capsys.readouterr().out                                # 0b58df7: ran
    assert not w["harbor"]


def test_15_the_firewall_step_comes_before_the_tasks_that_fetch(capsys):
    a = DOC[DOC.index("## A."):DOC.index("## B.")]
    fw = a[a.index("\n9. "):a.index("\n10. ")]
    # the rules printed by the runner, their addresses never in the repo
    assert 'agent_run.py deepswe --firewall-lines' in fw
    assert ar.main(["deepswe", "--firewall-lines"]) == 0
    assert capsys.readouterr().out.splitlines() == list(ar.FIREWALL_LINES)
    assert all(x.startswith("sudo iptables -I ") for x in ar.FIREWALL_LINES)
    assert "sudo docker network inspect bridge --format '{{range .Containers}}{{.Name}} {{end}}'" \
        in fw and "must print nothing" in fw
    assert "A\n   reboot loses the rules" in fw or "reboot loses the rules" in fw
    assert "sudo iptables -S INPUT | grep docker0; sudo iptables -S DOCKER-USER | grep docker0" \
        in fw
    assert "no tailnet address but Tailscale's resolver on port 53" in " ".join(fw.split())
    assert "step 7 again" in fw
    assert a.index("\n9. ") < a.index("--only fetching") < a.index("deepswe --oracle")


# ---------------------------------------------------------------------------
# 16. one runner a run
# ---------------------------------------------------------------------------

def test_16_a_second_runner_on_the_same_run_is_refused(server_world, capsys):  # noqa: F811
    import fcntl
    w = server_world
    rd = rdir_of()
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "run.lock", "a") as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)      # the first, running
        (rd / "jobs").mkdir(exist_ok=True)
        (rd / "jobs" / "inv__x__a1__1.running").write_text("1")
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2",
                        "--no-board"]) == 2
        assert "refused — another runner is on this run" in capsys.readouterr().out
        assert (rd / "jobs" / "inv__x__a1__1.running").exists()   # 0b58df7: deleted it
        assert not w["harbor"]
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board"]) == 0


# ---------------------------------------------------------------------------
# 17. the cache check: against the first request's whole prompt
# ---------------------------------------------------------------------------

class Relay:
    url = "http://127.0.0.1:9/v1"

    def __init__(self, first: dict, second: dict):
        base = {"trial": ar.CACHE_TRIAL, "status": 200, "prompt_ms": 5000, "predicted_n": 300,
                "predicted_ms": 10000}
        self.records = [{**base, **first}, {**base, **second}]


def test_17_a_long_first_reply_never_refuses_a_healthy_server():
    ask = lambda *a, **k: {"role": "assistant", "tool_calls": [{"id": "c1"}]}  # noqa: E731
    # the first request's 4,000; its 2,000-token thinking never cached: the
    # second step reads 2,300 of 6,300 again — 18c refused it (over a fifth)
    why, sp = ar.cache_line(Relay({"prompt_n": 4000, "cache_n": 0},
                                  {"prompt_n": 2300, "cache_n": 4000}), "m", "x", ask=ask, first=[])
    assert why == "" and sp["first"] == 4000 and sp["reused"] == 4000   # 0b58df7: refused
    why, _ = ar.cache_line(Relay({"prompt_n": 4000, "cache_n": 0},
                                 {"prompt_n": 6300, "cache_n": 0}), "m", "x", ask=ask, first=[])
    assert why.startswith("the second step took 0 tokens from the server's cache, under 80% of "
                          "the first request's 4,000")


# ---------------------------------------------------------------------------
# 18. the small ones
# ---------------------------------------------------------------------------

def test_18_the_server_going_down_twice_gives_a_task_up(tmp_path):
    rd = tmp_path / "r"
    trial(rd, "inv__d-1", result="", exc="ServerDown", stamp=1000)
    assert ar.state_of(rd, "inv__d-1", 1)[0] == "todo"
    trial(rd, "inv__d-1", result="", exc="ServerDown", stamp=2000)
    assert ar.state_of(rd, "inv__d-1", 1)[0] == "given up"     # 0b58df7: asked for ever


def test_18_a_trials_project_is_named_as_harbor_names_it(monkeypatch):
    task = "a" * 31 + "-" + "b" * 10                            # its 32nd character a dash
    assert ar.trial_head(task) == "a" * 31 + "__"              # 0b58df7: "aaa…a-__"
    seen = []
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (seen.append(cmd), (
        0, f"c1 {'a' * 31}__abc1234 \nc2 other__x") if cmd[:2] == ["docker", "ps"] else (0, ""))[1])
    ar.leftovers(task)
    assert ["docker", "rm", "-f", "c1"] in seen


def test_18_the_build_caches_listing_is_read_without_its_star(monkeypatch):
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (0, json.dumps(
        [{"ID": "abc123*"}, {"ID": "def456"}])))
    assert ar.cache_ids() == {"abc123", "def456"}              # 0b58df7: "abc123*"
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (1, "Error response from daemon"))
    assert ar.cache_ids() is None


def test_18_a_listing_that_cant_be_read_is_said_once_a_run(server_world, capsys):  # noqa: F811
    ar._CACHE_SAID.clear()                  # the stand-in's docker system df says nothing
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "3", "--no-board"]) == 0
    assert capsys.readouterr().out.count("Docker's build cache couldn't be listed") == 1


# ---------------------------------------------------------------------------
# 19. the steps
# ---------------------------------------------------------------------------

def test_19_steps_b0_b5_b6_and_c1():
    b = DOC[DOC.index("## B."):DOC.index("## C.")]
    b0 = b[b.index("\n0. "):b.index("\n1. ")]
    assert "sudo ss -tn state established '( sport = :8091 )'" in b0
    assert "dport" not in b0                                   # 0b58df7: missed the board's
    b5 = b[b.index("\n5. "):b.index("\n6. ")]
    assert 'must start "ready — N task(s)", N being 10' in b5 and "step A8's oracle" in b5
    b6 = b[b.index("\n6. "):b.index("\n7. ")]
    assert b6.index("pgrep -af scripts/agent_run.py") < b6.index("tmux new -s agent")
    assert "before any\n   resume" in b6 or "before any resume" in b6
    c = DOC[DOC.index("## C."):]
    assert "A `git pull` between A and C is fine" in c
