"""12f.2: On phone — numbers measured on the phone, typed in, for a served
model registered as a phone build. The kind exists only when a phone build
does; every number carries who measured it and when; and none of them ever
reaches a board column, a results file or an average."""

from __future__ import annotations

import json

from service import config, db, phone, runner
from test_served_12f1 import BASE, HOW, ME, SID, fake, register, svc  # noqa: F401

REPORT = {**phone.README, "model": SID, "by": "Sam", "date": "2026-09-25", "entered_by": ME}


def fresh(appmod):
    appmod._cache.update(key=None, payload=None, at=0.0)


def test_the_on_phone_kind_exists_only_for_a_phone_build(svc, fake):  # noqa: F811
    client, appmod, _ = svc
    assert client.get("/api/phone").json()["builds"] == []
    # a served model that isn't a phone build: no card
    assert register(client, fake).status_code == 200
    assert client.get("/api/phone").json()["builds"] == []
    fresh(appmod)
    assert client.get("/api/results").json()["served"][SID]["phone"] is False
    # its box ticked
    assert register(client, fake, phone=True).status_code == 200
    assert [b["id"] for b in client.get("/api/phone").json()["builds"]] == [SID]
    fresh(appmod)
    assert client.get("/api/results").json()["served"][SID]["phone"] is True
    # or its How text says so
    assert register(client, fake, name="LDA build two", how="phone build · " + HOW,
                    phone=False).status_code == 200
    assert {b["id"] for b in client.get("/api/phone").json()["builds"]} == \
        {SID, "served/LDA-build-two"}


def test_reported_numbers_carry_who_measured_them_and_when(svc, fake):  # noqa: F811
    client, _, _ = svc
    assert register(client, fake, phone=True).status_code == 200
    add = lambda **over: client.post("/api/phone/reports", json={**REPORT, **over})  # noqa: E731
    for over, why in (({"by": ""}, "Who measured it"), ({"date": ""}, "The date it was measured"),
                      ({"date": "25/09/2026"}, "The date it was measured"),
                      ({"decode_median": None}, "the median"),
                      ({"decode_best": 12.0}, "under the median"),
                      ({"model": BASE}, "go with a phone build")):
        r = add(**over)
        assert r.status_code == 422 and why in r.json()["detail"], (over, r.text)
    r = add()
    assert r.status_code == 200, r.text
    got = r.json()["report"]
    assert (got["by"], got["date"], got["entered_by"]) == ("Sam", "2026-09-25", ME)
    assert (got["device"], got["chip"], got["ram_gb"]) == ("OnePlus 15", "Snapdragon 8 Elite Gen 5",
                                                           16.0)
    assert (got["decode_median"], got["decode_best"]) == (13.5, 16.0)
    assert got["quality"] == [{"name": "MMLU", "value": "81.98%",
                               "note": "all 14,042, measured on Metal"}]
    assert got["source"].startswith("the fork's README")
    [b] = client.get("/api/phone").json()["builds"]
    assert b["reports"] == [got]
    # a second report is the newest; the first stays
    add(decode_median=14.1, decode_best=16.2, date="2026-09-27")
    [b] = client.get("/api/phone").json()["builds"]
    assert [x["decode_median"] for x in b["reports"]] == [14.1, 13.5]


def test_reported_numbers_never_enter_an_average_or_a_measured_column(svc, fake):  # noqa: F811
    client, appmod, _ = svc
    assert register(client, fake, phone=True).status_code == 200
    sid = db.add(SID, "instruct", "everyday", ME, "")
    runner.run_submission(db.get(sid))
    assert db.get(sid)["status"] == "done", db.get(sid)["error"]
    fresh(appmod)
    before = client.get("/api/results").json()
    assert client.post("/api/phone/reports", json=REPORT).status_code == 200
    fresh(appmod)
    after = client.get("/api/results").json()
    # the board's payload is the same with the report as without it
    assert after["models"] == before["models"] and after["cells"] == before["cells"]
    assert after["everyday"] == before["everyday"]
    text = json.dumps(after)
    assert "81.98" not in text and "0.8198" not in text and "decode_median" not in text
    assert SID not in (after["cells"].get("mmlu") or {})
    row = next(m for m in after["models"] if m["id"] == SID)
    assert row["avg"] is None and row["partialAvg"] is None
    # nor in any file the board reads its scores from
    for f in config.OUT_DIR.rglob("*.json*"):
        assert "81.98" not in f.read_text(encoding="utf-8", errors="replace"), f
