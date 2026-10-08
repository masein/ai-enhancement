"""18: what a task's container can reach (docs/AGENT-RUNS-design.md § 4) — a
container started as Harbor starts one (its prebuilt compose file) with the
runner's override merged over it tries the host's other ports, the tailnet's
range, the internet and a host folder from inside, and the test fails if
any answers or the check couldn't try (18b: it fails closed, and wants the
loopback as the only interface). The same container without the override
is refused, and on Linux reaches the host: the check sees it. Needs Docker;
pulls only a small bash image."""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import threading

import pytest

import agent_host_mini as hm
import agent_run as ar


def docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not docker_ok(), reason="no Docker here (CI has it)")
IMAGE = "bash:5.2"
BASE = f"""services:
  main:
    image: {IMAGE}
    command: ["sh", "-c", "sleep infinity"]
"""


def compose(tmp_path, project, *files, args=()):
    cmd = ["docker", "compose", "-p", project]
    for f in files:
        cmd += ["-f", str(f)]
    return subprocess.run([*cmd, *args], capture_output=True, text=True, timeout=600)


def probe(tmp_path, project, files, script):
    up = compose(tmp_path, project, *files, args=["up", "-d"])
    assert up.returncode == 0, up.stderr
    try:
        got = compose(tmp_path, project, *files, args=["exec", "-T", "main", "bash", "-c", script])
        return got.stdout + got.stderr
    finally:
        compose(tmp_path, project, *files, args=["down", "-v", "--remove-orphans"])


def test_a_tasks_container_reaches_nothing_and_has_no_host_folder(tmp_path):
    srv = socket.socket()
    srv.bind(("0.0.0.0", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: [srv.accept()[0].close() for _ in range(8)],
                     daemon=True).start()
    gw = subprocess.run(["docker", "network", "inspect", "bridge", "-f",
                         "{{(index .IPAM.Config 0).Gateway}}"], capture_output=True,
                        text=True).stdout.strip() or "172.17.0.1"
    # the host's port, the tailnet's own DNS address (Tailscale's, the same on
    # every tailnet — built here so no tracked file holds a tailnet address),
    # the internet
    tailnet_dns = ".".join(["100"] * 4)
    targets = [f"{gw}:{port}", f"{tailnet_dns}:53", "1.1.1.1:443", "8.8.8.8:53"]
    marker = tmp_path / "host-marker-18"
    marker.write_text("x")
    script = hm.reach_script(targets) + f"\n[ -e {marker} ] && echo 'REACHED host folder'"
    base = tmp_path / "docker-compose-prebuilt.yaml"
    base.write_text(BASE)
    out = probe(tmp_path, "agent18-reach", [base, ar.override(tmp_path)], script)
    assert hm.reach_verdict(out) == "", out                     # nothing answered
    assert "PIDS 4096" in out and "PROBE tcp-ok" in out and "NET lo" in out, out
    # without the override the same container has a network: the check refuses
    # it (18b: its interface), and on Linux sees the host's port answer
    open_ = probe(tmp_path, "agent18-open", [base], script)
    assert "has a network interface" in hm.reach_verdict(open_) or "reached" in \
        hm.reach_verdict(open_), open_
    if platform.system() == "Linux":
        assert f"REACHED {gw}:{port}" in open_, open_
    srv.close()
