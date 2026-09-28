"""12o.4: the judge calibration sheet shows each answer's question to the
person marking it, so export takes the Knowledge exam's diagnose half only —
the report half is never listed. The board's owner may add it with
--include-report-half: a one-line warning, and the export logged before the
sheet is written — who, when, which topics — with the audits Data & sources
lists.

Fixtures and the stand-in judge only."""

from __future__ import annotations

import csv
import sys
import time

import pytest

import judge_calibrate as jc
from conftest import make_service
from service import config, db


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client
    client.__exit__(None, None, None)


def judged():
    return {r["id"]: r for r in jc._judged_rows(config.OUT_DIR, set())}


def sheet(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["judge_calibrate.py", *argv])
    return jc.main()


def test_the_sheet_is_the_diagnose_half_by_default(svc, tmp_path):
    rows = judged()
    assert {"report", "diagnose"} <= {r["half"] for r in rows.values()}   # both on file
    out = tmp_path / "cal.csv"
    n = jc.export(config.OUT_DIR, out, [], 200, 7)
    got = sheet(out)
    assert n == len(got) > 0
    assert {rows[r["id"]]["half"] for r in got} == {"diagnose"}
    # no report-half question anywhere in it
    report = {r["prompt"] for r in rows.values() if r["half"] == "report"}
    assert not report & {r["prompt"] for r in got}
    assert db.hidden_audits() == []                                       # nothing to log


def test_a_row_whose_half_cant_be_told_stays_out(svc, tmp_path, monkeypatch):
    real = jc._judged_rows
    monkeypatch.setattr(jc, "_judged_rows", lambda *a: [
        {**r, "half": None} if r["category"] == "Law" else r for r in real(*a)])
    out = tmp_path / "cal.csv"
    jc.export(config.OUT_DIR, out, [], 400, 7)
    assert "Law" not in {r["category"] for r in sheet(out)}


def test_the_owner_adds_the_report_half_and_the_export_is_logged(svc, tmp_path):
    rows = judged()
    out = tmp_path / "cal.csv"
    before = time.time()
    n = jc.export(config.OUT_DIR, out, [], 200, 7, include_report_half=True,
                  by=f" {config.BOARD_OWNER.upper()} ")
    got = sheet(out)
    assert n == len(got)
    shown = [rows[r["id"]] for r in got if rows[r["id"]]["half"] != "diagnose"]
    assert shown and {r["half"] for r in shown} == {"report"}
    [log] = db.hidden_audits()
    topics = sorted({r["category"] for r in shown})
    # who, when, which topics, and how many of its questions
    assert log["by"] == config.BOARD_OWNER.upper()
    assert log["group"] == "judge calibration export · the exam's report half · " + ", ".join(topics)
    assert log["n"] == len({(r["task"], r["id"].rsplit("|", 1)[1]) for r in shown})
    assert before <= log["at"] <= time.time()
    # listed where the page's audits are
    assert svc.get("/api/everyday/audits").json()["audits"] == [log]


@pytest.mark.parametrize("by", ["", "someone-else"])
def test_nobody_else_takes_the_report_half(svc, tmp_path, by):
    out = tmp_path / "cal.csv"
    with pytest.raises(PermissionError, match="Only the board's owner"):
        jc.export(config.OUT_DIR, out, [], 50, 7, include_report_half=True, by=by)
    assert not out.exists() and db.hidden_audits() == []


def test_no_log_no_sheet(svc, tmp_path, monkeypatch):
    def down(*a):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "hidden_audit_add", down)
    out = tmp_path / "cal.csv"
    with pytest.raises(RuntimeError):
        jc.export(config.OUT_DIR, out, [], 50, 7, include_report_half=True,
                  by=config.BOARD_OWNER)
    assert not out.exists()


def test_the_command_line(svc, tmp_path, monkeypatch, capsys):
    out = tmp_path / "cal.csv"
    assert run(monkeypatch, "export", str(config.OUT_DIR), "--out", str(out), "--n", "40") == 0
    said = capsys.readouterr()
    assert said.err == "" and f"wrote 40 rows to {out}, the diagnose half only — " in said.out
    # the flag without the owner's name: refused, nothing written
    out.unlink()
    assert run(monkeypatch, "export", str(config.OUT_DIR), "--out", str(out),
               "--include-report-half") == 2
    assert "Only the board's owner" in capsys.readouterr().err and not out.exists()
    # the owner: one line of warning, and the log
    assert run(monkeypatch, "export", str(config.OUT_DIR), "--out", str(out), "--n", "200",
               "--include-report-half", "--by", config.BOARD_OWNER) == 0
    said = capsys.readouterr()
    assert said.err.splitlines() == [jc.REPORT_WARNING]
    assert f"wrote 200 rows to {out} — " in said.out
    assert len(db.hidden_audits()) == 1
