"""r25 (masein, 14 Nov): a run of a served model on this server keeps, in its record, the
launch registered when it started — its flags and environment, and "How it's served" — as a
DeviceMark run's setup does, so a run here can be checked later as a box's bundle can. A row
from before this keeps nothing, and nothing is filled in for it.

Fake llama-servers: no model runs."""

from __future__ import annotations

import json
import sqlite3

from service import config, db, served
from test_12q_devicemark_runs import ME, fake, queue, svc  # noqa: F401 — svc and fake are fixtures

FLAGS = "-ngl 99 --n-cpu-moe 21 --no-mmap --jinja"
ENV = "LOOKAHEAD_MOE_ROUTER_NORM=1 GGML_CPU_DISABLE_FUSION=1"
HOW = "step8804.gguf: the lda build at a4a76a751 with the router norm, CPU fusion off"


def register(server, flags: str = FLAGS, env: str = ENV) -> str:
    return served.register({"name": "step8804 (lda runtime)", "base_url": server.base, "how": HOW,
                            "flags": flags, "env": env, "thinking": "off", "phone": True}, ME)["id"]


def test_a_served_run_here_keeps_the_launch_registered_when_it_started(svc, fake):  # noqa: F811
    mid = register(fake)
    row = queue(mid, "pilot")
    assert row["status"] == "done", row["error"]
    assert json.loads(row["launch"]) == {"flags": FLAGS, "env": ENV, "how": HOW}
    # the registration changes later: the run keeps what it started with, and the next
    # run takes the launch registered then
    served.set_launch(mid, "-ngl 99 --jinja", "GGML_CPU_DISABLE_FUSION=1")
    assert json.loads(db.get(row["id"])["launch"])["env"] == ENV
    again = queue(mid, "pilot")
    assert again["status"] == "done", again["error"]
    assert json.loads(again["launch"]) == {"flags": "-ngl 99 --jinja",
                                           "env": "GGML_CPU_DISABLE_FUSION=1", "how": HOW}
    # a run queued and not yet started has no launch yet
    waiting = db.add(mid, "instruct", "devicemark", ME, "", part="pilot")
    assert db.get(waiting)["launch"] == ""


def test_a_served_mobile_run_keeps_it_too(svc, fake, monkeypatch):  # noqa: F811
    # the runs it was asked for: Mobile benchmarks of a served model on this server (step8804's,
    # 29 Oct) — the launch is kept when the run starts, before any question is asked
    from service import runner
    monkeypatch.setattr(runner, "_ask_served", lambda *a, **k: ("done", False))
    mid = register(fake)
    sid = db.add(mid, "instruct", "mobile", ME, "", part="mmlu")
    runner.run_submission(db.get(sid))
    assert json.loads(db.get(sid)["launch"]) == {"flags": FLAGS, "env": ENV, "how": HOW}


def test_a_row_from_before_keeps_nothing(svc, tmp_path, monkeypatch):  # noqa: F811
    # a database made before the column existed: its rows get none, and nothing is guessed
    old = tmp_path / "old.sqlite3"
    with sqlite3.connect(old) as c:
        c.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY AUTOINCREMENT, hf_id TEXT NOT NULL, "
                  "kind TEXT NOT NULL DEFAULT 'auto', suite TEXT NOT NULL DEFAULT 'full', "
                  "submitter TEXT DEFAULT '', note TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'queued', "
                  "progress TEXT DEFAULT '', error TEXT DEFAULT '', created_at REAL NOT NULL)")
        c.execute("INSERT INTO submissions (hf_id, suite, status, created_at) "
                  "VALUES ('served/step8804-lda-2610', 'mobile', 'done', 1.0)")
    monkeypatch.setattr(config, "DB_PATH", old)
    db.init()
    with sqlite3.connect(old) as c:
        assert c.execute("SELECT launch FROM submissions").fetchall() == [("",)]
