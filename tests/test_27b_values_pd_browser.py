"""27b: no number changes on the two builds' table — 24's world (invented
questions and answers, worked out by the stats worker), each value the old
renderer drew recorded (`RECORD27B=1`) and checked against scoreCell's cells
now. See values27b.py."""

from __future__ import annotations

import os

import pytest

import values27b as v
from test_24_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
VIEWS = {"paired": ("[data-pd-table] .pdx-row[data-pd-row] [data-pd-diff]",
                    [["pd", "[data-pd-table]"]])}


@pytest.mark.skipif(not os.environ.get("RECORD27B"), reason="records the old renderers' values")
def test_record_the_values_before(live, page):
    got = v.record(page, live, "pd", VIEWS)
    assert all(got[r][t] for r in VIEWS for t, _ in VIEWS[r][1]), got.keys()


def test_no_number_changes(live, page):
    out = v.lost(page, live, "pd", VIEWS)
    assert out == [], "\n".join(map(str, out[:40]))
