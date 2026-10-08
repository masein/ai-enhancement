"""18b part 2: what step A needs — the docs' --check-reach line runs, the
agent venv is a hashed lock the runner holds it to, the venv steps work on
a stock server and are numbered, verification needs no network (a copy of
each task installs what it would fetch), the container's check fails
closed, and the docker group is masein's choice, said plainly.

Invented tasks; no model, no benchmark's task, nothing fetched."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

import agent_bench as ab
import agent_host_mini as hm
import agent_run as ar
from agent18 import MODEL, tasks_folder
from test_18_agent import server_world  # noqa: F401 — server_world is the fixture

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "AGENT-RUNS.md").read_text()


def doc_line(needle: str) -> str:
    """the docs' command line holding `needle`"""
    return next(x for x in DOC.splitlines() if needle in x and x.strip().startswith("cd "))


# ---------------------------------------------------------------------------
# 4. --check-reach needs neither --as nor --oracle
# ---------------------------------------------------------------------------

def test_the_docs_check_reach_line_runs(server_world, monkeypatch, capsys):  # noqa: F811
    line = doc_line("--check-reach")
    argv = shlex.split(line.split("scripts/agent_run.py", 1)[1])
    seen = {}
    monkeypatch.setattr(ar, "check_reach", lambda rdir, targets: seen.setdefault("t", targets)
                        and "")
    assert ar.main(argv) == 0                                    # 70df001: argparse, exit 2
    out = capsys.readouterr().out
    assert out.startswith("a task's container reaches nothing of: ")
    assert "its only interface is the loopback" in out
    assert any(t.endswith(":8090") for t in seen["t"])
    assert not server_world["harbor"]


# ---------------------------------------------------------------------------
# 5. a full lock, with hashes, the runner holds the venv to
# ---------------------------------------------------------------------------

def test_the_lock_pins_every_package_with_its_hashes():
    text = (ROOT / "docs" / "agent-requirements.txt").read_text()
    lock = ar.lock()
    assert len(lock) > 50                                     # 70df001: two lines, no hashes
    assert lock["harbor"] == ab.HARBOR_VERSION and lock["mini-swe-agent"] == ab.MINI_VERSION
    assert "litellm" in lock and lock["litellm"] not in ("1.82.7", "1.82.8")
    blocks = re.split(r"\n(?=[a-z0-9])", text.split("\n\n", 1)[1].strip())
    assert len(blocks) == len(lock)
    for b in blocks:
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", b), b.splitlines()[0]
    # the doc installs it only that way
    assert "pip install --require-hashes --no-deps -r ~/benchmarks/aienh/docs/agent-" \
           "requirements.txt" in DOC


def test_the_runner_refuses_a_venv_that_differs_from_its_lock(monkeypatch):
    from importlib import metadata
    lock = ar.lock()
    real = dict(lock)
    real["litellm"] = "1.82.8"                                  # one package moved on
    del real["openai"]                                          # one missing
    monkeypatch.setattr(metadata, "version", lambda name: real[name] if name in real else (
        _ for _ in ()).throw(metadata.PackageNotFoundError(name)))
    v = ar.versions()
    why = ar.check_versions(v, oracle=False)
    assert why.startswith("2 packages in the agent venv differ from its lock: ")
    assert "litellm 1.82.8 (the lock: " in why and "openai missing (the lock: " in why
    assert "make the venv again from the lock (docs/AGENT-RUNS.md § A, steps 2–4)" in why
    monkeypatch.setattr(metadata, "version", lambda name: lock[name] if name in lock else "x")
    assert ar.check_versions(ar.versions(), oracle=False) == ""


# ---------------------------------------------------------------------------
# 6. the venv steps work on a stock server, numbered, referred to by number
# ---------------------------------------------------------------------------

def section(name: str) -> str:
    start = DOC.index(f"## {name}.")
    nxt = DOC.find("\n## ", start + 4)
    return DOC[start:nxt if nxt > 0 else None]


def test_step_a_is_numbered_and_works_on_a_stock_server():
    a = section("A")
    steps = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\. ", a)]
    assert steps == list(range(1, len(steps) + 1)) and len(steps) >= 10
    assert "python3 -m venv" not in a                       # 70df001: needs python3-venv
    assert "uv venv --python 3.12 --seed ~/agent-venv" in a   # pip in it
    # uv itself: a step of its own, checked against its published checksum
    assert re.search(r"releases/download/0\.12\.18/uv-x86_64-unknown-linux-gnu\.tar\.gz.*"
                     r"echo \"[0-9a-f]{64}  uv-x86_64-unknown-linux-gnu\.tar\.gz\" \| sha256sum -c",
                     a)
    # every "step N" names a step that exists, in A or B
    numbers = set(steps) | {int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\. ", section("B"))}
    for m in re.finditer(r"steps? (?:A|B)?(\d+)", DOC):
        assert int(m.group(1)) in numbers, m.group(0)
    assert "step 4's" not in DOC and "Run step 4" not in DOC


# ---------------------------------------------------------------------------
# 7. verification with no network: a copy of each task installs what it fetches
# ---------------------------------------------------------------------------

PARSER = '''#!/bin/bash
set -uo pipefail -x
cd /testbed
git apply --verbose --reject - <<'EOF_1'
invented
EOF_1
{fetch}
: '>>>>> Start Test Output'
invented-test-runner --run one
: '>>>>> End Test Output'

cd ..
cat > parser.py <<EOF
# /// script
# requires-python = ">=3.11"
# dependencies = ["invented-parser==1.0", "invented-data==2.0"]
# ///
import sys
EOF
uv run parser.py
'''


def invented_tasks(d: Path) -> list[str]:
    names = tasks_folder(d, 4)
    for i, n in enumerate(names):
        (d / n / "tests").mkdir()
        (d / n / "tests" / "test.sh").write_text(PARSER.replace(
            "{fetch}", ["", "npm install", "cargo update x@1.0 --precise 0.9 2>/dev/null || true",
                        ""][i]))
    # one verified in a separate container, as DeepSWE's are
    (d / names[3] / "task.toml").write_text('[verifier]\nnetwork_mode = "no-network"\n'
                                            'environment_mode = "separate"\n')
    (d / names[3] / "tests" / "Dockerfile").write_text("FROM invented/verifier:1\n")
    return names


def test_a_tasks_copy_installs_what_its_verification_would_fetch(tmp_path):
    src = tmp_path / "src"
    names = invented_tasks(src)
    before = {n: (src / n / "environment" / "Dockerfile").read_text() for n in names}
    b = ab.BENCHES["swebench-multilingual"]
    out, info = ar.offline_tasks(tmp_path, b, src, names)
    assert out.name.endswith("+offline")
    for n in names:                                             # the fetched tasks untouched
        assert (src / n / "environment" / "Dockerfile").read_text() == before[n]
    plain = (out / names[0] / "environment" / "Dockerfile").read_text()
    assert plain.startswith(before[names[0]]) and ar.OFFLINE_MARK in plain
    # the parser's own header, run once at build time; uv offline after
    run = next(x for x in plain.splitlines() if "uv run parser.py" in x)
    assert "'# dependencies = [\"invented-parser==1.0\", \"invented-data==2.0\"]'" in run
    assert "'# requires-python = \">=3.11\"'" in run and run.startswith("RUN cd / && ")
    assert "ENV UV_OFFLINE=1" in plain and "evalboard-warm" not in plain
    # a task whose tests fetch: those lines, in a throwaway copy; the tool offline
    npm = (out / names[1] / "environment" / "Dockerfile").read_text()
    assert ("RUN cp -a /testbed /tmp/evalboard-warm && cd /tmp/evalboard-warm && "
            "( npm install ) ; cd / && rm -rf /tmp/evalboard-warm") in npm
    assert "ENV npm_config_offline=true" in npm
    cargo = (out / names[2] / "environment" / "Dockerfile").read_text()
    assert "( cargo update x@1.0 --precise 0.9 2>/dev/null || true ) ; ( cargo fetch )" in cargo
    assert "ENV CARGO_NET_OFFLINE=true" in cargo
    # a separate verifier gets the agent's limits; its Dockerfile is left alone
    sep = out / names[3] / "tests"
    assert (sep / "docker-compose.yaml").read_text() == ar.OVERRIDE_YAML
    assert "network_mode: none" in ar.OVERRIDE_YAML
    assert (sep / "Dockerfile").read_text() == "FROM invented/verifier:1\n"
    assert info["fetching"] == names[1:3] and len(info["sha256"]) == 64
    # made again each run, the same
    assert ar.offline_tasks(tmp_path, b, src, names)[1] == info


def test_the_run_asks_the_copies_and_only_fetching_picks_its_tasks(  # noqa: F811
        server_world, tmp_path, capsys):  # noqa: F811
    w = server_world
    tdir = ar.tasks_dir(tmp_path, ab.BENCHES["swebench-multilingual"])
    for i, n in enumerate(w["names"]):
        (tdir / n / "tests").mkdir(exist_ok=True)
        (tdir / n / "tests" / "test.sh").write_text(PARSER.replace(
            "{fetch}", "composer update" if i in (5, 17) else ""))
    assert ar.main(["swebench-multilingual", "--oracle", "--only", "fetching",
                    "--no-board"]) == 0
    asked = sorted(h["task"] for h in w["harbor"])
    assert asked == sorted([w["names"][5], w["names"][17]])
    assert all("+offline" in h["cmd"][h["cmd"].index("-p") + 1] for h in w["harbor"])
    out = capsys.readouterr().out
    assert "2 of these tasks fetch packages in their tests" in out
    run = json.loads((ar.run_dir(tmp_path, "swebench-multilingual", "", 1, True) / "run.json")
                     .read_text())
    assert run["offline"]["fetching"] == asked and run["lock_sha256"]


# ---------------------------------------------------------------------------
# 8. the container's check fails closed, and tries more
# ---------------------------------------------------------------------------

def test_the_containers_check_fails_closed():
    ok = "PROBE tcp-ok\nNET lo \nPIDS 4096\nDONE\n"
    assert hm.reach_verdict(ok) == ""
    cases = {
        "PROBE tcp-ok\nNET lo \nPIDS 4096\n": "didn't run to its end",       # cut short
        "PROBE no /dev/tcp: bash: /dev/tcp/127.0.0.1/1: No such file or directory\nNET lo"
        "\nPIDS 4096\nDONE": "couldn't try a connection",                     # 70df001: passed
        "PROBE missing sleep\nPROBE tcp-ok\nNET lo\nPIDS 4096\nDONE": "missing sleep",
        "NET lo\nPIDS 4096\nDONE": "couldn't try a connection",
        "PROBE tcp-ok\nNET eth0 lo \nPIDS 4096\nDONE": "has a network interface: eth0",
        "PROBE tcp-ok\nNET \nPIDS 4096\nDONE": "interfaces couldn't be read",
        "PROBE tcp-ok\nREACHED dns pypi.org\nNET lo\nPIDS 4096\nDONE": "reached dns pypi.org",
        "PROBE tcp-ok\nNET lo\nPIDS max\nDONE": "no process limit"}
    for out, words in cases.items():
        assert words in hm.reach_verdict(out), out
    script = hm.reach_script(["10.0.0.1:22", "2606:4700:4700::1111:443"])
    assert "timeout " not in script                              # 70df001: `timeout` or pass
    assert "/sys/class/net" in script and "getent hosts" in script and "echo DONE" in script


def test_the_check_runs_in_bash_and_refuses_a_machine_with_a_network():
    out = subprocess.run(["bash", "-c", hm.reach_script(["127.0.0.1:1"])], capture_output=True,
                         text=True, timeout=120).stdout
    assert "DONE" in out and "PROBE tcp-ok" in out
    assert hm.reach_verdict(out) != ""          # this machine has an interface besides lo


def test_the_targets_cover_ipv6_names_the_gateways_the_tailnet_and_the_relay(monkeypatch):
    tailnet = ".".join(["100", "70", "1", "2"])       # built: no tracked file holds one
    answers = {"hostname": (0, f"192.168.1.20 {tailnet} 2001:db8::20\n"),
               "network": (0, "172.17.0.1\n"), "ip": (0, "default via 192.168.1.1 dev eth0\n")}
    monkeypatch.setattr(ar, "run", lambda cmd, **kw: answers.get(cmd[1] if cmd[0] == "docker"
                                                                 else cmd[0], (1, "")))
    t = ar.reach_targets(llama="http://172.17.0.1:8090/v1", relay_port=40123)
    for want in ("172.17.0.1:8090", "192.168.1.20:8899", f"{tailnet}:22", "2001:db8::20:8090",
                 "host.docker.internal:40123", "192.168.1.1:80", "127.0.0.11:53",
                 f"{ar.tailnet_dns()}:53", "2606:4700:4700::1111:443", "127.0.0.1:40123"):
        assert want in t, want


# ---------------------------------------------------------------------------
# 9. the docker group, said plainly; sudo the other way
# ---------------------------------------------------------------------------

def test_the_docker_group_is_said_plainly_with_sudo_beside_it(  # noqa: F811
        server_world, monkeypatch, capsys):  # noqa: F811
    a = section("A")
    assert "root without a password" in a and "sudo gpasswd -d $USER docker" in a
    assert "(b) sudo, each time." in a and "owned by root" in a and "--by masein" in a
    # under sudo, the Runs row says who ran sudo
    monkeypatch.setenv("SUDO_USER", "masein")
    monkeypatch.setenv("USER", "root")
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1", "--no-board"]) == 0
    run = json.loads(next((Path(ar.bench_root()) / "agent-runs").glob("*/run.json")).read_text())
    assert run["by"] == "masein"
    # a refusal names both ways
    server_world["docker"] = (1, "permission denied")
    assert ar.main(["swebench-multilingual", "--as", MODEL, "--tasks", "1"]) == 2
    out = capsys.readouterr().out
    assert "run this with sudo, or add yourself to the docker group" in out


@pytest.mark.parametrize("name", ["deepswe", "swebench-multilingual"])
def test_the_agents_override_and_the_separate_verifiers_are_one(tmp_path, name):
    assert ar.override(tmp_path).read_text() == ar.OVERRIDE_YAML
    assert "pids_limit: 4096" in ar.OVERRIDE_YAML and ab.bench(name)
