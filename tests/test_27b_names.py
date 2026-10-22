"""27b, Part 0: the names file holds Kevin's mapping (brief 26, Part 2) — the
phone app's model-001 to model-004 by name, what each is where he said, and
no board row or sha256 yet (a row only when the file's sha256 matches its
pin, never by name); the tiers T0 to T2 by their RAM rule; no engines. A
test of model-004 reads as step8804, on no model page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from service import config, telemetry
from test_26_telemetry import a_test, keep

REAL = Path(telemetry.__file__).resolve().parent.parent / "eval_tasks" / "phone" / "names.json"


@pytest.fixture
def real(tmp_path, monkeypatch):
    """the repository's own names file, a temporary telemetry folder"""
    monkeypatch.setattr(config, "BENCH_ROOT", tmp_path)
    for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(config, k, "")
    monkeypatch.setattr(telemetry, "NAMES", REAL)
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
    yield tmp_path
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)


def test_the_names_file_holds_kevins_mapping():
    names = json.loads(REAL.read_text(encoding="utf-8"))
    assert {k: v["name"] for k, v in names["models"].items()} == {
        "model-001": "Qwen3.6-35B-A3B k4-LDA, UD-Q4_K_XL", "model-002": "step4000",
        "model-003": "step4000-denseq8", "model-004": "step8804"}
    assert names["models"]["model-001"]["what"] == "the phone build of the original model"
    assert names["models"]["model-003"]["what"] == (
        "step4000 with its non-expert weights in q8_0 instead of q5_K")
    # no board row and no sha256 until Kevin's sha256 matches a row's pin
    assert all(m["row"] == "" and m["sha256"] == "" for m in names["models"].values())
    assert names["tiers"] == {"T0": "RAM 14 GB or more", "T1": "RAM 10 to 14 GB",
                              "T2": "RAM under 10 GB"}
    assert names["engines"] == {}


def test_a_test_of_model_004_shows_step8804(real):
    t = telemetry.read(a_test("t1"))                    # the attached test: model-004, T0
    assert (t["model"]["id"], t["model"]["name"], t["model"]["named"]) == (
        "model-004", "step8804", True)
    assert t["tier"]["name"] == "RAM 14 GB or more"
    # on the testers' place under its name, its medians by phone; on no model page
    keep(real, a_test("t1"))
    ts, _ = telemetry.tests()
    (p,) = telemetry.per_phone(ts)
    assert (p["model"]["name"], p["phone"]["name"]) == ("step8804", "OnePlus 15")
    assert telemetry.by_row() == {}
