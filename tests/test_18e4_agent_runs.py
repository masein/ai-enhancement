"""18e, part 3: agent runs, before step A (docs/prompts/phase-18e-twelfth-
review-fetch-scores-agent-page.md, points 11–16). A test a point, each
failing on 518e4a1: the firewall's order (tailscaled's forward rule, an
earlier ACCEPT), a tailnet peer tried and never written down, a second
runner refused before the task copies are touched, an oracle run's later
"couldn't be checked", a stop's cost, its kinds. Harbor, Docker, iptables and
the model are stand-ins; no model runs."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_run as ar
import import_agent as ia
from agent18 import FIREWALL, MODEL, QUAD, run_json, trial
from test_18_agent import server_world  # noqa: F401 — server_world is the fixture

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()
RANGE = ".".join(["100", "64", "0", "0"]) + "/10"
PEER = ".".join(str(x) for x in (100, 101, 7, 9))     # built here: the mirror's guard


def rdir_of(model: str = MODEL, oracle: bool = False, number: int = 1) -> Path:
    return ar.run_dir(Path(os.environ["BENCH_ROOT"]), "swebench-multilingual", model, 1, oracle,
                      number)


# ---------------------------------------------------------------------------
# 11. the firewall: in the raw table, first; a tailnet peer tried
# ---------------------------------------------------------------------------

# the 18d rules in INPUT and DOCKER-USER, after tailscaled put its own
# forward rule first again — 518e4a1's check passed these
FILTER_ONLY = "\n".join([
    "-P INPUT ACCEPT", "-P FORWARD DROP", "-N DOCKER-USER", "-N ts-forward",
    "-A INPUT -i docker0 -j DROP", "-A FORWARD -j ts-forward", "-A FORWARD -j DOCKER-USER",
    f"-A DOCKER-USER -d {QUAD}/32 -i docker0 -p udp -m udp --dport 53 -j ACCEPT",
    *[f"-A DOCKER-USER -d {n} -i docker0 -j DROP" for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", RANGE, "169.254.0.0/16")],
    "-A ts-forward -o tailscale0 -j ACCEPT"]) + "\n"


def with_rules(monkeypatch, text: str) -> list[list[str]]:
    asked: list[list[str]] = []

    def run(cmd, **kw):
        asked.append(cmd)
        return 0, text
    monkeypatch.setattr(ar, "run", run)
    return asked


def test_11_rules_tailscaled_can_put_its_forward_before_are_refused(monkeypatch):
    asked = with_rules(monkeypatch, FILTER_ONLY)
    why = ar.firewall_missing()
    assert "aren't in place" in why, why                       # 518e4a1: "" — they passed
    assert asked[0] == ["iptables", "-t", "raw", "-S", "PREROUTING"]
    with_rules(monkeypatch, FIREWALL)
    assert ar.firewall_missing() == ""


def test_11_an_accept_before_the_drops_is_refused(monkeypatch):
    lines = FIREWALL.splitlines()
    early = [lines[0], "-A PREROUTING -i docker0 -j ACCEPT", *lines[1:]]
    with_rules(monkeypatch, "\n".join(early) + "\n")
    why = ar.firewall_missing()
    assert "another rule comes before the bridge's" in why and "rule 1, -j ACCEPT" in why, why
    # one for another interface, or a DROP, changes nothing
    for other in ("-A PREROUTING -i eth0 -j ACCEPT", "-A PREROUTING ! -i docker0 -j ACCEPT",
                  "-A PREROUTING -d 172.17.0.2/32 -j DROP"):
        with_rules(monkeypatch, "\n".join([lines[0], other, *lines[1:]]) + "\n")
        assert ar.firewall_missing() == "", other
    # the resolver's ACCEPT after the tailnet's DROP: no name resolves
    dns = next(x for x in lines if "--dport 53" in x)
    late = [x for x in lines if x != dns] + [dns]
    with_rules(monkeypatch, "\n".join(late) + "\n")
    assert "comes after the tailnet's DROP" in ar.firewall_missing()


def test_11_the_lines_print_the_check_their_help_promises(capsys):
    assert ar.main(["deepswe", "--firewall-lines"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == list(ar.FIREWALL_LINES)
    assert out[-1] == "sudo iptables -t raw -S PREROUTING"     # 518e4a1: promised, not printed
    assert all(" -t raw -I PREROUTING -i docker0 " in x for x in ar.FIREWALL_LINES)
    assert "--dst-type LOCAL -j DROP" in ar.FIREWALL_LINES[0]
    assert ar.FIREWALL_LINES[-1].endswith("--dport 53 -j ACCEPT")     # inserted last: first


def test_11_a_tailnet_peer_is_tried_and_never_written_to_a_file(monkeypatch, tmp_path):
    status = {"Self": {"TailscaleIPs": ["x"]}, "Peer": {
        "a": {"Online": False, "TailscaleIPs": ["10.1.1.1"]},
        "b": {"Online": True, "TailscaleIPs": ["fd7a::1", PEER]}}}
    seen: list[list[str]] = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[:3] == ["tailscale", "status", "--json"]:
            return 0, json.dumps(status)
        if cmd[:2] == ["docker", "exec"]:
            return 0, "REACHED a tailnet peer\n"               # past a rule out of order
        if cmd[:3] == ["docker", "compose", "-p"] and "exec" in cmd:
            return 0, "PROBE tcp-ok\nNET lo\nPIDS 4096\nCAPS 0\nNNP 1\nDONE\n"
        if "build" in cmd:
            return 0, "PROBE tcp-ok\nDONE\n"
        return 0, ""
    monkeypatch.setattr(ar, "run", run)
    monkeypatch.setattr(ar, "cache_ids", lambda: set())
    monkeypatch.setattr(ar, "prune_since", lambda before: 0)
    assert ar.tailnet_peer() == PEER                           # 518e4a1: no peer tried
    rdir = tmp_path / "reach"
    rdir.mkdir()
    ar.check_reach(rdir, ["1.1.1.1:443"], PEER)
    exec_ = next(c for c in seen if "exec" in c and c[:2] != ["docker", "exec"])
    assert exec_[exec_.index("-e") + 1] == f"PEER={PEER}"
    why, open_ = ar.check_build_reach(rdir, ["1.1.1.1:443"], PEER)
    assert "a tailnet peer" in open_
    bridge = next(c for c in seen if c[:2] == ["docker", "exec"])
    assert f"PEER={PEER}" in bridge and "--build-arg" not in sum(seen, [])
    # the address in no file the checks wrote
    for p in rdir.rglob("*"):
        if p.is_file():
            assert PEER not in p.read_text(errors="replace"), p


def test_11_the_peers_check_counts_a_refusal_as_reached():
    """its ports aren't known: a refused connection means it is reachable"""
    got = subprocess.run(["bash", "-c", hm.PEER_SCRIPT], env={"PEER": "127.0.0.1",
                         "PATH": os.environ["PATH"]}, capture_output=True, text=True, timeout=30)
    assert "REACHED a tailnet peer" in got.stdout             # nothing listens on its port 22
    got = subprocess.run(["bash", "-c", hm.PEER_SCRIPT], env={"PATH": os.environ["PATH"]},
                         capture_output=True, text=True, timeout=30)
    assert got.stdout == ""                                    # no peer given: nothing tried
    assert hm.PEER_SCRIPT in hm.reach_script(["1.1.1.1:443"])


def test_11_step_a9_gives_the_raw_tables_check():
    a = DOC[DOC.index("## A."):DOC.index("## B.")]
    fw = a[a.index("\n9. "):a.index("\n10. ")]
    assert "sudo iptables -t raw -S PREROUTING" in fw and "raw table" in fw
    assert "sudo iptables -S INPUT | grep docker0" not in fw
    assert 'not "a tailnet peer"' in fw


# ---------------------------------------------------------------------------
# 12. the run lock before anything is fetched or copied, --check too
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("check", [False, True])
def test_12_a_second_runner_is_refused_before_the_task_copies(server_world, monkeypatch,  # noqa: F811
                                                              capsys, check):
    fetched: list[int] = []
    real = ar.fetch_tasks
    monkeypatch.setattr(ar, "fetch_tasks", lambda *a, **k: fetched.append(1) or real(*a, **k))
    rd = rdir_of()
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "run.lock", "a") as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)      # the first, running
        assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "2", "--no-board",
                        *(["--check"] if check else [])]) == 2
        assert "refused — another runner is on this run" in capsys.readouterr().out
        assert fetched == []                                   # 518e4a1: fetched and copied first


# ---------------------------------------------------------------------------
# 13. the oracle: a real result stands; a model run keeps its list; run 2 asks the rest
# ---------------------------------------------------------------------------

def test_13_a_later_couldnt_be_checked_never_replaces_a_result(server_world):  # noqa: F811
    root = Path(os.environ["BENCH_ROOT"])
    names = server_world["names"]
    one = rdir_of("", True)
    run_json(one, names[:1], model="")
    trial(one, names[0], agent="oracle", exit_status="")      # run 1: resolved
    two = rdir_of("", True, 2)
    run_json(two, names[:1], model="")
    for k in range(ar.TRIES):                                 # run 2: given up
        trial(two, names[0], result="", exc="RuntimeError", started=False, stamp=1000 + k,
              message="Docker compose command failed: pull")
    got = ar.oracle_states(root, "swebench-multilingual", [])
    assert got["left_out"] == [] and got["unverified"] == []   # 518e4a1: left out


def test_13_a_model_run_keeps_the_list_it_started_with(server_world):  # noqa: F811
    root = Path(os.environ["BENCH_ROOT"])
    names = server_world["names"]
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--only", ",".join(names[:2]),
                    "--no-board"]) == 0
    rd = rdir_of()
    assert json.loads((rd / "run.json").read_text())["left_out"] == []
    before = ia.summary(rd)
    # an oracle run after it: one of its tasks doesn't pass
    od = rdir_of("", True)
    run_json(od, names[:1], model="")
    trial(od, names[0], result="unresolved", agent="oracle", exit_status="")
    assert ar.oracle_states(root, "swebench-multilingual", [])["left_out"] == [names[0]]
    after = ia.summary(rd)
    assert after["left_out"] == [] and after["score"] == before["score"]   # 518e4a1: changed


def test_13_a_later_oracle_run_over_every_task_asks_only_the_rest(server_world, capsys):  # noqa: F811
    w = server_world
    names = w["names"]
    one = rdir_of("", True)
    run_json(one, names[:3], model="")
    for t in names[:3]:
        trial(one, t, agent="oracle", exit_status="")
    assert ar.main(["swebench-multilingual", "--oracle", "--run", "2", "--all",
                    "--no-board"]) == 0
    asked = {h["task"] for h in w["harbor"]}
    assert not asked & set(names[:3]) and asked == set(names[3:])   # 518e4a1: every task
    assert "3 of these tasks have a result from another oracle run: not asked again here" in \
        capsys.readouterr().out


def test_13_c1_says_what_a_pull_costs():
    c = " ".join(DOC[DOC.index("## C."):].split())
    assert "A `git pull` between A and C is fine" not in c
    assert "refuses `--oracle --all` on step A8's run. Then give `--run 2`: it asks only the " \
           "tasks no other oracle run has a result for" in c


# ---------------------------------------------------------------------------
# 14–15. what stopping costs; a stop and a failure after the agent ran, by name
# ---------------------------------------------------------------------------

def test_14_the_docs_say_what_stopping_costs():
    doc = " ".join(DOC.split())
    assert "after a kill, the same command goes on." not in doc  # 518e4a1: a kill asked again
    assert "**stopping costs the tasks in flight.** Ctrl-C or a kill stops every task running " \
           "then" in doc
    assert "the run stopped while it ran." not in doc


def test_15_a_stop_and_harbors_failures_after_the_agent_have_their_own_kinds(tmp_path):
    rd = tmp_path / "r"
    kinds = {}
    for name, kw in {
            "stop-after": {"exc": "CancelledError"},
            "stop-before": {"exc": "KeyboardInterrupt", "started": False},
            "cleanup": {"exc": "CleanupFailed", "message": "the clean-up failed"},
            "tests-in": {"exc": "AddTestsDirError", "message": "couldn't copy"}}.items():
        r = ab.read_trial(trial(rd, f"inv__{name}", result="", agent="host-mini-swe-agent",
                                **kw))
        kinds[name] = ab.again_kind(r["why"])
    assert kinds == {                                          # 518e4a1: each "Docker couldn't…"
        "stop-after": "the run was stopped after the agent ran",
        "stop-before": "the run was stopped before the agent started",
        "cleanup": "Harbor's clean-up failed after the agent ran",
        "tests-in": "Harbor couldn't copy the tests in after the agent ran"}
    assert ab.again_kind("Docker compose command failed: pull") == \
        "Docker couldn't pull, build or start its container"


# ---------------------------------------------------------------------------
# 16. the small ones
# ---------------------------------------------------------------------------

def test_16_an_empty_build_cache_is_empty(monkeypatch):
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (0, "null\n"))
    assert ar.cache_ids() == set()                             # 518e4a1: None, "couldn't be listed"
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: (1, ""))
    assert ar.cache_ids() is None


def test_16_the_cache_rule_is_the_codes():
    doc = " ".join(DOC.split())
    assert "a fifth" not in doc                                # 518e4a1: two old rules
    assert "less than 80% of the first request's prompt from the server's cache" in doc


def test_16_tasks_the_oracle_couldnt_check_are_said_so():
    s = {**ab.score([], 10), "left_out": 3, "unchecked": 1}
    assert ab.score_words(s, 10).endswith("3 tasks left out: 2 whose reference solution doesn't "
                                          "pass here, 1 the oracle couldn't check here")
    s.update(unchecked=3)
    assert ab.score_words(s, 10).endswith("left out: the oracle couldn't check them here")
    s.update(unchecked=0)
    assert ab.score_words(s, 10).endswith("left out: their reference solution doesn't pass here")


def test_16_the_reward_read_quotes_nothing_and_opens_no_device(tmp_path, monkeypatch):
    import types
    hv = types.SimpleNamespace(VerifierOutputParseError=type("Parse", (Exception,), {}))
    monkeypatch.setitem(__import__("sys").modules, "harbor", types.ModuleType("harbor"))
    monkeypatch.setitem(__import__("sys").modules, "harbor.verifier",
                        types.SimpleNamespace(verifier=hv))
    with pytest.raises(hv.VerifierOutputParseError) as e:
        hm._reward("a-docker-login-shaped-secret", "/x/reward.txt")
    assert e.value.__cause__ is None and e.value.__suppress_context__   # 518e4a1: from e
    # a FIFO is looked at, never opened
    d = tmp_path / "verifier"
    d.mkdir()
    os.mkfifo(d / "reward.txt")
    opened: list = []
    real = os.open
    monkeypatch.setattr(os, "open", lambda p, *a, **k: opened.append(p) or real(p, *a, **k))
    assert hm.safe_read(d, "reward.txt") == ""
    assert "reward.txt" not in opened                          # 518e4a1: opened, then read ''
