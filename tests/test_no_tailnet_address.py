"""The repo has a public mirror (masein/ai-enhancement), so no tracked file may
hold an address on the tailnet: Tailscale's IPv4 range (the 100.64/10 block) or its
IPv6 prefix (fd7a:115c:a1e0::/48). The board's address was in the docs and the
client until this test: they say <board> now, and the client finds the board
from --base, BENCH_URL or the board it was downloaded from. A test that needs
such an address builds a made-up one at runtime."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# an address, not the range's name: Tailscale's IPv6 prefix written out alone
# (fd7a:115c:a1e0::, as the raw export's scrubber describes it) is public knowledge
TAILNET = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"
                     r"|\bfd7a:115c:a1e0:[0-9a-f]{1,4}:[0-9a-f:]*", re.I)


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
                         check=True).stdout.decode().split("\0")
    return [REPO / n for n in out if n]


def test_no_tracked_file_holds_a_tailnet_address():
    found = []
    for p in tracked():
        if not p.is_file() or p.stat().st_size > 30_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if TAILNET.search(line):
                # the file and line only: the address itself is never repeated
                found.append(f"{p.relative_to(REPO)}:{n}")
    assert not found, "a tailnet address in a tracked file (the mirror is public): " + ", ".join(found)


def test_the_check_would_catch_one():
    made_up = ".".join(["100", "101", "7", "9"])
    assert TAILNET.search(f"http://{made_up}:8899/")
    assert TAILNET.search(":".join(["fd7a", "115c", "a1e0", "ab12", "", "1"]))
    assert not TAILNET.search("the tailnet's IPv6 (fd7a:115c:a1e0::)")
    assert not TAILNET.search("http://100.10.1.1/ and 10.0.0.1")      # outside the range


# ---------------------------------------------------------------------------
# the client finds the board without an address in the repo
# ---------------------------------------------------------------------------

@pytest.fixture
def bc(monkeypatch):
    import importlib
    sys.path.insert(0, str(REPO / "clients"))
    import bench_client
    monkeypatch.delenv("BENCH_URL", raising=False)
    return importlib.reload(bench_client)


def test_the_client_takes_base_then_bench_url_then_the_board_it_came_from(bc, monkeypatch):
    with pytest.raises(bc.BenchError, match="Where is the board"):
        bc.Bench()
    monkeypatch.setattr(bc, "DEFAULT_BASE", "http://came-from:8899")
    assert bc.Bench().base == "http://came-from:8899"
    monkeypatch.setenv("BENCH_URL", "http://from-env:8899/")
    assert bc.Bench().base == "http://from-env:8899"
    assert bc.Bench("http://given:8899").base == "http://given:8899"


def test_the_cli_says_where_to_set_the_board(bc, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench_client.py", "queue"])
    assert bc.main() == 2
    err = capsys.readouterr().err
    assert "--base http://<board>:8899" in err and "BENCH_URL" in err


def test_a_downloaded_client_knows_the_board_it_came_from(tmp_path, monkeypatch):
    from conftest import make_service
    client, _, _ = make_service(tmp_path, monkeypatch, tree=False)
    try:
        text = client.get("/client").text
        assert 'DEFAULT_BASE = "http://testserver"' in text
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("base,ok", [
    ("http://testserver", True), ("http://board.example:8899", True),
    ("https://[fd00::1]:8899", True),
    ('http://x"; import os; "', False), ("http://x/../y", False), ("javascript:alert(1)", False),
    ("http://x:8899\nDEFAULT_BASE = 'y'", False), ("http://x:8899\n", False)])
def test_only_an_address_is_written_into_the_client(base, ok):
    """the request's own Host: anything but a scheme, a host and a port is left out"""
    from service import app
    assert bool(app._BASE_OK.match(base)) is ok
