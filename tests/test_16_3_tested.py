"""16.3: Models' Tested filter reads the date of each model's latest finished
run. lm_eval's results carry a date; a judged run its judged_at; but a model
tested only by DeviceMark, a served run, a GGUF run or Everyday had none,
and could never be kept by a date filter. The runs table has every kind:
/api/results carries each model's latest finished one as testedAt."""

from __future__ import annotations

import pytest

from conftest import make_service
from service import db, served

T1, T2 = 1_759_000_000.0, 1_759_400_000.0


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, tree = make_service(tmp_path, monkeypatch)
    yield client, appmod


def done(model, suite, at, status="done"):
    sid = db.add(model, "instruct", suite, "masein", "", status=status)
    db.update(sid, finished_at=at)
    return sid


def models(client, appmod):
    appmod._cache.update(key=None, payload=None, at=0.0)
    return {m["id"]: m for m in client.get("/api/results").json()["models"]}


def test_the_latest_finished_run_of_any_kind_by_model(svc):
    done("fx/good-750m", "full", T1)
    done("fx/good-750m", "devicemark", T2)
    done("fx/good-750m", "everyday", T2 + 50, status="failed")      # not finished: failed
    sid = db.add("fx/good-750m", "instruct", "gguf", "masein", "")  # queued: not finished
    assert sid
    assert db.last_done()["fx/good-750m"] == T2


def test_a_served_model_tested_only_by_devicemark_has_a_tested_date(svc):
    client, appmod = svc
    rec = {"id": "served/k4-lda", "name": "k4-LDA phone build", "base_url": "http://x:8090/v1",
           "how": "k4", "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off", "pin": {}}
    db.served_put(rec)
    served.write_meta(rec)
    before = models(client, appmod)["served/k4-lda"]
    assert before["date"] is None and "testedAt" not in before          # no lm_eval date
    done("served/k4-lda", "devicemark", T1)
    after = models(client, appmod)["served/k4-lda"]
    assert after["testedAt"] == T1
    # the payload kept a few seconds still learns of a run that moved no file
    done("served/k4-lda", "everyday", T2)
    appmod._cache["at"] = 0.0
    got = {m["id"]: m for m in client.get("/api/results").json()["models"]}
    assert got["served/k4-lda"]["testedAt"] == T2


def test_a_model_with_no_finished_run_has_none(svc):
    client, appmod = svc
    done("fx/good-750m", "full", T1, status="failed")
    m = models(client, appmod)["fx/good-750m"]
    assert m.get("testedAt") is None and m["date"]                     # its results' own date
