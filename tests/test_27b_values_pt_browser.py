"""27b: no number changes on On our testers' phones and the On-device
report's phone column — 26's world (nine made-up tests, a phone build with a
speed typed in by hand), each value the old renderers drew recorded
(`RECORD27B=1`) and checked against modelTable's tables now. See values27b.py."""

from __future__ import annotations

import os

import pytest

import values27b as v
from test_26_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
VIEWS = {"testers": ("[data-pt-per] tbody tr", [["per", "[data-pt-per]"]]),
         "on-device": ("[data-od-table] tbody tr", [["od", "[data-od-table]"]])}


@pytest.mark.skipif(not os.environ.get("RECORD27B"), reason="records the old renderers' values")
def test_record_the_values_before(live, page):
    got = v.record(page, live, "pt", VIEWS)
    assert all(got[r][t] for r in VIEWS for t, _ in VIEWS[r][1]), got.keys()


def test_no_number_changes(live, page):
    out = v.lost(page, live, "pt", VIEWS)
    assert out == [], "\n".join(map(str, out[:40]))
