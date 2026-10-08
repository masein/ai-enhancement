"""18b part 5: the export (points 27 to 30). public/ opens only with
SCRUB_HOSTS and SCRUB_ACCOUNTS given; no name is taken from a request's
Host header; the server's own words never go out; folder names, the quote
check's gaps and the README's own list. Runs invented on the board; nothing
is uploaded."""

from __future__ import annotations

import base64
import builtins
import socket
from pathlib import Path

import export_devicemark_raw as dmx
import export_frontier_raw as efr
import export_safe as es
from service import config, db
from service import frontier as sf
from test_12q_devicemark_runs import svc  # noqa: F401 — svc is the fixture
from test_17_gguf_box import SERVED, box  # noqa: F401
from test_17g_review import LONG
from test_17h_review import _export_world
from test_17j_export import ACCT, exported_log

TAILNET = ".".join(["100", "74", "89", "105"])            # built: no tracked file holds one


# ---------------------------------------------------------------------------
# 27. public/ only with both variables given
# ---------------------------------------------------------------------------

def test_27_public_is_refused_without_both_variables_whatever_a_request_says(  # noqa: F811
        box, tmp_path, monkeypatch, capsys):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    monkeypatch.delenv("SCRUB_HOSTS", raising=False)
    monkeypatch.delenv("SCRUB_ACCOUNTS", raising=False)
    from fastapi.testclient import TestClient

    import service.app as appmod
    TestClient(appmod.app, base_url="http://bench").get("/api/served")  # 70df001: it opened

    why = es.refused_public()
    assert "SCRUB_HOSTS isn't given" in why and "SCRUB_ACCOUNTS isn't given" in why, why
    db.public_set(SERVED, True, "masein")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    out = tmp_path / "raw"
    assert efr.main(["--run", str(sid), "--out", str(out)]) == 1
    assert not (out / "public").exists()
    # one given, the other not: still refused
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01")
    assert "SCRUB_ACCOUNTS isn't given" in es.refused_public()
    # both: public/ may open; the names worked out add to them
    monkeypatch.setenv("SCRUB_ACCOUNTS", ACCT)
    monkeypatch.setattr(socket, "gethostname", lambda: "c0ffee123456")
    assert es.refused_public() == ""
    got = es.scrub_names()
    assert "evalsrv01" in got["hosts"] and ACCT in got["accounts"]


# ---------------------------------------------------------------------------
# 28. a Host header is never a word the scrub removes
# ---------------------------------------------------------------------------

def test_28_no_name_is_taken_from_a_request(svc, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SCRUB_HOSTS", "evalsrv01")
    monkeypatch.setenv("SCRUB_ACCOUNTS", ACCT)
    from fastapi.testclient import TestClient

    import service.app as appmod
    for host in ("the", "run", "done", "hle", "sha256", "board.tailnet-name.ts.net"):
        TestClient(appmod.app, base_url=f"http://{host}").get("/api/served")
    hosts = [h.lower() for h in es.scrub_names()["hosts"]]
    for word in ("the", "run", "done", "hle", "sha256"):
        assert word not in hosts, hosts                          # 70df001: each, for good
    line = "2026-10-07 09:00:09 the run: done · hle 12 of 540 · sha256 0123abcd"
    assert dmx.scrub(line) == line                               # 70df001: [host] [host]: …
    assert not (Path(config.BENCH_ROOT) / "seen-hosts.json").exists()
    assert "public_files.seen" not in Path(appmod.__file__).read_text()


# ---------------------------------------------------------------------------
# 29. the server's own words, in a stop reason or a line with a free slot
# ---------------------------------------------------------------------------

WORDS = "HTTP 400: model gemma-judge at judge-box rejected prompt containing secret-words"


def test_29_the_servers_words_never_go_out(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    text = exported_log(
        sid, tmp_path,
        "[frontier] GPQA Diamond: the server refused a question: " + WORDS
        + " · the answers it gave are kept: the next run asks only the rest (" + WORDS + ")",
        "[frontier] GPQA Diamond: the server failed on 6 questions, asked its own way and "
        "without its chat parsing (" + WORDS + "): stopped · the answers it gave are kept: the "
        "next run asks only the rest (" + WORDS + ")",
        # a reason with a newline: its second line is the server's
        "[frontier] GPQA Diamond: the server failed on every question it was asked (3), asked "
        "its own way and without its chat parsing (HTTP 500: Internal",
        "secret-words in the second line): stopped",
        f"===== [{sid + 7}] a header ConnectionRefusedError at judge-box =====",
        "2026-10-07 09:00:09 the run: done · judge-box TimeoutError")
    for word in ("secret-words", "judge-box", "gemma-judge", "ConnectionRefusedError",
                 "TimeoutError"):
        assert word not in text, (word, text)                    # 70df001: each went out
    assert ("[frontier] GPQA Diamond: the server refused a question · the answers it gave are "
            "kept: the next run asks only the rest\n") in text, text
    assert ("[frontier] GPQA Diamond: the server failed on 6 questions, asked its own way and "
            "without its chat parsing: stopped · the answers it gave are kept: the next run asks "
            "only the rest\n") in text, text


def test_29_a_stop_reason_holds_the_servers_status_alone():
    assert sf.status_of(WORDS) == "HTTP 400"
    assert sf.status_of("no answer within 600 s, twice") == "no answer within 600 s"
    assert sf.status_of("the model spoke at length") == "no status given"
    src = Path(sf.__file__).read_text()
    assert 'f"the server refused a question: {x}"' not in src           # 70df001
    assert "chat parsing ({a.error.get('chat')})" not in src
    assert 'log(f"[frontier] {line} ({e.why})")' not in src


# ---------------------------------------------------------------------------
# 30. folder names, the quote check's gaps, the README's own list
# ---------------------------------------------------------------------------

def test_30_a_folder_is_named_by_its_run(box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    dest = efr.export_run(sid, tmp_path / "raw")
    assert dest.name == f"frontier-run-{sid}"                   # 70df001: the row's name
    readme = (dest / "README.md").read_text()
    assert "the tailnet's IPv6 (fd7a:115c:a1e0::)" in readme     # 70df001: ([address])
    # a DeviceMark row named after an address: the scrub reads it as an id
    plain = f"served/{TAILNET}"
    assert dmx.scrub(plain) != plain or dmx.es_risky(plain)


def test_30_the_quote_checks_gaps_are_closed():
    q = es.Questions(["Which enzyme splits the lactose sugar found in fresh milk products?"])
    for line in ("[frontier] enzyme splits the lactose",                         # four words
                 "[frontier] which um enzyme um splits um the um lactose",        # a filler
                 "[frontier] " + base64.b64encode(b"splits the lactose sugar").decode()):
        assert q.quotes(line), line                                # 70df001: through
    assert not q.quotes("[frontier] gpqa_diamond_epoch: 120 of 198 answered")
    for line in ("[frontier] key " + "a3f9" * 10,                             # 40 hex, no prefix
                 "[frontier] at %2Fhome%2Fmasein%2Fbenchmarks"):             # URL-encoded home
        assert es._RISKY.search(line), line                         # 70df001: through
    assert not es._RISKY.search("[frontier] sha256 " + "ab" * 32)  # a sha256 has its slot


def test_30_four_words_of_a_gated_question_in_an_export_are_left_out(  # noqa: F811
        box, tmp_path, monkeypatch):  # noqa: F811
    sid = _export_world(box, monkeypatch)
    words = LONG[2]["question"].split()
    four = " ".join(words[:4])
    text = exported_log(sid, tmp_path, f"[frontier] note: {four}")
    assert four not in text, text
