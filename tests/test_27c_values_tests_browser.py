"""27c: no number changes on "Every test" (On our testers' phones) — 26's
world (nine made-up tests from made-up testers), each value the old table
drew recorded (`RECORD27B=1`) and checked against the new one: grouped
columns, the engine build, thermal max, charging and first load moved to the
test's panel and its row's hover. See values27b.py."""

from __future__ import annotations

import os

import pytest

import values27b as v
from test_26_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
VIEWS = {"testers": ("[data-pt-table] tbody tr", [["tests", "[data-pt-table]"]])}


@pytest.mark.skipif(not os.environ.get("RECORD27B"), reason="records the old renderers' values")
def test_record_the_values_before(live, page):
    got = v.record(page, live, "pt-tests", VIEWS)
    assert got["testers"]["tests"], got


def test_no_number_changes(live, page):
    out = v.lost(page, live, "pt-tests", VIEWS)
    assert out == [], "\n".join(map(str, out[:40]))
