"""27b: no number changes on the On-device and Step checkpoints reports —
25c's world (21c's scorecard, four builds with their files' sizes, every card
entered), each value the old renderers drew recorded (`RECORD27B=1`) and
checked against modelTable's tables now. See values27b.py."""

from __future__ import annotations

import os

import pytest

import values27b as v
from test_25c_browser import board  # noqa: F401  the world

pytestmark = pytest.mark.dashboard
VIEWS = {"on-device": ("[data-od-table] tbody tr", [["od", "[data-od-table]"]]),
         "step-checkpoints": ("[data-rp-ours] tbody tr", [["ours", "[data-rp-ours]"],
                                                         ["every", "[data-rp-every]"],
                                                         ["ranout", "[data-rp-ranout]"]])}


@pytest.mark.skipif(not os.environ.get("RECORD27B"), reason="records the old renderers' values")
def test_record_the_values_before(live, page):
    got = v.record(page, live, "od", VIEWS)
    assert all(got[r][t] for r in VIEWS for t, _ in VIEWS[r][1]), got.keys()


def test_no_number_changes(live, page):
    out = v.lost(page, live, "od", VIEWS)
    assert out == [], "\n".join(map(str, out[:40]))
