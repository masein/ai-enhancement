"""12f.0: two fixes from 2026-09-26.

1. The judge-test sample was mostly base models looping ("The following is
   the following: …"), which any judge marks 0. A new sample draws mostly
   instruct and chat models' answers, loops at most a tenth; it is a new
   judge-test version, and the old one — masein's 100 marks with it — stays,
   under History.
2. The server's disk filled and nothing said so: the status dot goes amber
   under 10 GB and red under 3 GB, and below red a run doesn't start."""

from __future__ import annotations

import collections
import json
import shutil

import pytest

from conftest import fresh, make_service
from service import config, db, disk, judge_test, runner

LOOP = " ".join(["The following is the following:"] * 12)


def row(i, model, answer, question="Why do prices rise when supply falls?"):
    return {"key": f"exam:{model}|{i}", "kind": "exam", "model": model, "task": "exam_economics",
            "topic": f"Economics {i % 5}", "question": question, "reference": "r",
            "answer": answer, "judge": 0}


def test_what_counts_as_a_loop():
    assert judge_test.degenerate(LOOP)
    assert judge_test.degenerate("supply falls prices rise", "Why do prices rise when supply falls?")
    assert judge_test.degenerate("   ")
    assert not judge_test.degenerate(
        "When supply falls and demand holds, buyers compete for fewer goods, so sellers can "
        "raise prices until the market clears again.", "Why do prices rise when supply falls?")
    # a real answer that repeats a phrase once is not a loop
    assert not judge_test.degenerate(
        "Prices rise because fewer goods meet the same demand. Prices rise because fewer goods "
        "meet the same demand, and sellers know buyers will pay more for what is left.")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch, judge_model="stub")
    yield client, appmod
    client.__exit__(None, None, None)


def test_a_sample_from_mostly_loops_is_mostly_instruct_answers_and_loops_at_most_a_tenth(
        svc, monkeypatch):
    rows = [row(i, f"base/b{i % 3}", LOOP) for i in range(60)]                       # 60% loops
    rows += [row(100 + i, f"org/chat-{i % 4}-it", f"A real answer about supply, number {i}, "
             "with its own reasoning about scarcity and what buyers do.") for i in range(25)]
    rows += [row(200 + i, f"base/b{i % 3}", f"A base model's plain answer {i} about scarcity "
             "and prices going up when goods are short.") for i in range(15)]
    monkeypatch.setattr(judge_test, "_exam_rows", lambda: rows)
    monkeypatch.setattr(judge_test, "_everyday_rows", lambda: [])
    monkeypatch.setattr(judge_test, "_instruct", lambda: {f"org/chat-{i}-it" for i in range(4)})
    got = judge_test.answers(n=50, rebuild=True)
    # 40 good answers and room for 5 loops: a sample of 45 rather than more loops
    assert len(got) == 45
    loops = [a for a in got if judge_test.degenerate(a["answer"], a["question"])]
    assert len(loops) <= 5                                            # at most a tenth
    kinds = collections.Counter("instruct" if a["model"].endswith("-it") else
                                "loop" if a in loops else "base" for a in got)
    assert kinds["instruct"] == 25 and kinds["base"] == 15 and kinds["loop"] == 5
    # instruct first: every instruct answer on file is in before any base one
    assert json.loads(judge_test._sample_path().read_text())["builder"] == judge_test.BUILDER


def test_a_new_sample_is_a_new_version_and_the_old_marks_stay_with_the_old(svc, monkeypatch):
    client, appmod = svc
    old = [row(i, "org/chat-1-it", f"an old answer {i} about scarcity") for i in range(100)]
    p = judge_test._sample_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # a sample drawn by 12i.1's builder: no builder, no version
    p.write_text(json.dumps({"n": 100, "at": 1.0, "answers": old}), encoding="utf-8")
    for a in old:
        db.jt_mark("person", a["key"], 3, "claude")
    assert judge_test.version() == "v1" and judge_test.person() == "person"
    new = [row(500 + i, "org/chat-2-it", f"a new answer {i} about scarcity") for i in range(100)]
    monkeypatch.setattr(judge_test, "_exam_rows", lambda: new)
    monkeypatch.setattr(judge_test, "_everyday_rows", lambda: [])
    page = client.get("/api/judge-test").json()
    assert page["version"] != "v1" and page["version"].startswith("v")
    assert judge_test.person() == "person@" + page["version"]
    assert page["progress"]["marked"] == 0 and page["changed"] is True
    h = page["history"]
    assert [x["version"] for x in h] == ["v1"] and h[0]["marked"] == 100 and h[0]["by"] == ["claude"]
    # not one mark deleted: they are still masein's, on the old sample
    assert len(db.jt_marks("person")) == 100
    # a mark on the new sample: it is no longer "changed"
    client.post("/api/judge-test/mark", json={"key": page["answers"][0]["key"], "mark": 2,
                                              "by": "masein"})
    again = client.get("/api/judge-test").json()
    assert again["changed"] is False and again["progress"]["marked"] == 1
    assert len(db.jt_marks("person")) == 100


# ---------------------------------------------------------------------------
# 2. the disk
# ---------------------------------------------------------------------------

def free(monkeypatch, gb):
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10 ** 12, 10 ** 12 - int(gb * 1e9),
                                                              int(gb * 1e9)))


def test_the_disk_goes_amber_then_red_on_the_status_dot(svc, monkeypatch):
    client, appmod = svc
    free(monkeypatch, 50)
    assert disk.status_check() is None
    assert not any(c["key"] == "disk" for c in client.get("/api/results").json()["checks"])
    free(monkeypatch, 5.04)
    c = disk.status_check()
    assert c["severity"] == "warning" and c["short"] == \
        "The server's disk has 5.0 GB free. Runs may fail to save."
    fresh(appmod)
    free(monkeypatch, 2.1)
    checks = client.get("/api/results").json()["checks"]
    assert checks[0]["key"] == "disk" and checks[0]["severity"] == "error"
    assert checks[0]["short"] == "The server's disk has 2.1 GB free. Runs may fail to save."
    assert (config.DISK_AMBER_GB, config.DISK_RED_GB) == (10, 3)


def test_a_red_disk_stops_a_run_from_starting_in_the_same_words(svc, monkeypatch):
    client, _ = svc
    free(monkeypatch, 50)
    sid = client.post("/api/submissions", json={"hf_id": "org/m", "suite": "full"}).json()["id"]
    free(monkeypatch, 2.1)
    r = client.post("/api/submissions", json={"hf_id": "org/m2", "suite": "full"})
    assert r.status_code == 409
    assert r.json()["detail"] == "The server's disk has 2.1 GB free. Runs may fail to save."
    # one queued before the disk filled doesn't start either
    runner.run_submission(db.get(sid))
    got = db.get(sid)
    assert got["status"] == "failed"
    assert got["error"] == "The server's disk has 2.1 GB free. Runs may fail to save."
