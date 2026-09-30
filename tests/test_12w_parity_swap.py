"""12w: the MTP parity check with one llama-server on the card at a time.

The check asks two setups, the one with MTP and then the one without, and
only one llama-server fits on the card beside the judge's model. The run
took "nothing answers" on the second setup for a server that had stopped: it
waited two minutes (SERVED_RETRY_S) and failed, and a resubmit needed the
first setup's server up again just to start.

- A setup with items still to answer is waited for (DM_SWAP_WAIT_S, 30 min):
  the run's line says which server to start, and that the other can stop.
- A setup whose 50 answers are kept needs no server, and the run starts with
  either one up.
- A server up with another file than the one registered is not waited for.

Fake llama-servers write every reply: no model runs."""

from __future__ import annotations

import json

import devicemark as dm
from fake_openai import FakeServer
from service import config, db, served
from service import devicemark as sdm
from test_12q_devicemark_runs import queue, register, replies, row_dir, svc  # noqa: F401 — svc is the fixture


def _two():
    a, b = FakeServer(), FakeServer()
    replies(a)
    replies(b)
    return a, b, register(a, "Qwen3.6 phone MTP", "--spec-type mtp"), register(b, "Qwen3.6 phone")


def _lines(monkeypatch) -> list[str]:
    """every line the run's row showed"""
    seen, real = [], db.update

    def update(sid, **kw):
        if kw.get("progress"):
            seen.append(kw["progress"])
        return real(sid, **kw)
    monkeypatch.setattr(db, "update", update)
    return seen


def test_the_run_waits_for_the_second_setup_and_says_which_to_start(svc, monkeypatch):  # noqa: F811
    a, b, mtp, plain = _two()
    try:
        # the setup without MTP isn't up until the third look: the servers are being swapped
        looks, real = [], served.check_pin

        def check_pin(rec):
            if rec["id"] != plain:
                return real(rec)
            looks.append(len(a.requests))
            return f"Nothing answered at {rec['base_url']}: Connection refused" if len(looks) < 3 \
                else real(rec)
        monkeypatch.setattr(served, "check_pin", check_pin)
        monkeypatch.setattr(config, "DM_SWAP_WAIT_S", 2)
        seen = _lines(monkeypatch)
        row = queue(mtp, "parity", pair=plain)
        assert row["status"] == "done", row["error"]
        assert "parity 50/50 answers the same, 50/50 outputs identical" in row["progress"]
        # the MTP setup's 50 were in before the other was looked for
        assert looks == [50, 50, 50] and len(b.requests) == 50
        assert ("devicemark parity · start Qwen3.6 phone's server now (the other can stop): "
                "waiting 1 min more") in seen
        [log_file] = config.LOGS_DIR.glob(f"service_{row['id']}_*.log")
        assert f"[devicemark] waiting for {plain} at {b.base}" in log_file.read_text()
    finally:
        a.close()
        b.close()


def test_it_gives_up_after_the_wait_and_the_resubmit_needs_only_the_other_server(svc, monkeypatch):  # noqa: F811
    a, b, mtp, plain = _two()
    try:
        b.close()                                           # the setup without MTP isn't running
        monkeypatch.setattr(config, "DM_SWAP_WAIT_S", 0.3)
        row = queue(mtp, "parity", pair=plain)
        assert row["status"] == "failed"
        assert row["error"].startswith("Qwen3.6 phone: its server didn't answer in 1 min (Nothing "
                                       "answered at ")
        assert row["error"].endswith(" · the answers so far are kept")
        kept = row_dir(mtp) / "devicemark_parity_mtp_answers.jsonl"
        assert len(kept.read_text().splitlines()) == 50
        assert not (row_dir(plain) / dm.PARITY_NAME).exists()
        # swapped: the MTP server is stopped, the other is up. The run starts without the
        # first — its 50 are kept — and asks only the second
        a.close()
        b2 = FakeServer(port=b.port)
        try:
            replies(b2)
            monkeypatch.setattr(config, "DM_SWAP_WAIT_S", 5)
            row = queue(mtp, "parity", pair=plain)
            assert row["status"] == "done", row["error"]
            assert "parity 50/50" in row["progress"] and len(b2.requests) == 50
            assert json.loads((row_dir(plain) / dm.PARITY_NAME).read_text())["passes"] is True
        finally:
            b2.close()
    finally:
        a.close()


def test_a_server_up_with_another_file_is_not_waited_for(svc, monkeypatch):  # noqa: F811
    a, b, mtp, plain = _two()
    try:
        real = served.check_pin
        monkeypatch.setattr(served, "check_pin",
                            lambda rec: served.CHANGED_LINE if rec["id"] == plain else real(rec))
        monkeypatch.setattr(config, "DM_SWAP_WAIT_S", 600)
        seen = _lines(monkeypatch)
        row = queue(mtp, "parity", pair=plain)
        assert row["status"] == "failed"
        assert row["error"] == (f"Qwen3.6 phone: {served.CHANGED_LINE} · the answers so far "
                                f"are kept")
        assert not any("start Qwen3.6 phone's server now" in x for x in seen)
        assert b.requests == []
    finally:
        a.close()
        b.close()


def test_every_other_part_still_needs_its_server_at_the_start(svc, monkeypatch):  # noqa: F811
    a, b, mtp, plain = _two()
    try:
        a.close()
        for part in ("full", "pilot", "speed"):
            row = queue(mtp, part)
            assert row["status"] == "failed" and row["error"].startswith("Nothing answered at ")
    finally:
        b.close()


def test_kept_counts_only_this_setups_answers_to_these_items(svc):  # noqa: F811
    a, b, mtp, plain = _two()
    try:
        keys, s = dm.keys_for("parity"), sdm.settings(False)
        rec = served.get(mtp)
        store = row_dir(mtp) / "devicemark_parity_mtp_answers.jsonl"
        assert sdm.kept(rec, keys, s, store) == {}
        queue(mtp, "parity", pair=plain)
        assert set(sdm.kept(rec, keys, s, store)) == set(keys)
        # thinking on is other settings: none of them counts
        assert sdm.kept(rec, keys, sdm.settings(True), store) == {}
    finally:
        a.close()
        b.close()
