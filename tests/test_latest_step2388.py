"""step2388 on Models ▸ Latest: three versions of one file, side by side — High (thinking on,
the 6-expert path), Medium (thinking off, the 6-expert path at its file's presence penalty 0.75)
and Instant (thinking off, the 3-expert path), each named by the report ("step2388 · High") with
"k3k6 runtime" under it. Only runs made with the file's own settings: the presence-0 Medium row
stays off the report. The world is 21c's; nothing is fetched, run or graded."""

from __future__ import annotations

import json

import pytest

from conftest import make_service
from service import config, reported
from test_21c_page import world

FILE = reported.REPORTS_DIR / "step-checkpoints.json"
STEP2388 = [("step2388 · High", "served/step2388-k3k6-2611 · thinking"),
            ("step2388 · Medium", "served/step2388-k3k6-2611-medium-075"),
            ("step2388 · Instant", "served/step2388-k3k6-2611")]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    yield client
    client.__exit__(None, None, None)


def test_the_report_names_step2388s_three_versions():
    """last among ours (the other rows keep their order), on Latest and not in the picker,
    each with its name and its line; the presence-0 Medium row isn't one of them, and the
    report never says "Masein model\""""
    rep = json.loads(FILE.read_text(encoding="utf-8"))
    head = rep["ours"][-3:]
    assert [(o["label"], o["row"]) for o in head] == STEP2388
    for o in head:
        assert o["name"] == o["label"] and o["line"] == "k3k6 runtime"
        assert o["glance"] is True and o["pick"] is False and o["colour"] is None
        assert "step" not in o        # not a point on Across training
    rows = [o["row"] for o in rep["ours"]]
    assert "served/step2388-k3k6-2611-medium" not in rows
    assert "masein model" not in FILE.read_text(encoding="utf-8").lower()
    # the other rows keep the board's own names
    assert not any("name" in o or "line" in o for o in rep["ours"][:-3])


def test_latest_carries_their_names_and_lines(svc):
    """the report's payload gives the three their name and line, and Latest's rows end
    with them; a row with nothing imported in the world says so instead of a number"""
    world()
    got = svc.get("/api/reports/step-checkpoints").json()
    mine = {r["label"]: r for r in got["ours"]}
    for label, row in STEP2388:
        r = mine[label]
        assert (r["row"], r["name"], r["line"], r["glance"], r["pick"]) == (
            row, label, "k3k6 runtime", True, False)
        assert {c["state"] for c in r["cells"].values()} == {"none"}
    assert not any("name" in r or "line" in r for r in got["ours"]
                   if not r["label"].startswith("step2388"))
    from service import app
    rows = app.results_payload()["glance"]["rows"]
    assert rows[-3:] == [row for _, row in STEP2388]
