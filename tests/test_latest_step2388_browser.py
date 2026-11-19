"""step2388 on Models ▸ Latest, in a real browser at 1280 and 400 px, light and dark: its three
versions side by side, each named by the report ("step2388 · High") with "k3k6 runtime" on the
line under it, beside our other rows and the published ones. The world is 21c's, with the three
rows served and scored here (invented numbers); nothing is fetched, run or graded."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from test_21c_page import world
from test_21d_browser import LATEST, go, no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "latest-step2388"
# label, row, the board's name for it, thinking, invented GPQA and MMLU-Pro shares
THREE = [("step2388 · High", "served/step2388-k3k6-2611 · thinking",
          "step2388 (k3k6 runtime)", True, 0.712, 0.801),
         ("step2388 · Medium", "served/step2388-k3k6-2611-medium-075",
          "step2388 Medium, presence 0.75 (k3k6 runtime)", False, 0.633, 0.744),
         ("step2388 · Instant", "served/step2388-k3k6-2611",
          "step2388 (k3k6 runtime)", False, 0.598, 0.731)]


def _results(row: Path, mid: str, task: str, v: float, thinking: bool) -> None:
    d = row / f"{task}_0shot" / "frontier"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-11-12T07-20-00.000000.json").write_text(json.dumps({
        "results": {task: {"alias": task, "acc,none": v, "acc_stderr,none": 0.021}},
        "config": {"model": "frontier",
                   "model_args": f"pretrained={mid},enable_thinking={thinking}"},
        "date": time.time(),
        "frontier": {"version": "frontier-2", "answers": 792, "ran_out": 0,
                     "scored_by": "code", "where": "rented GPU · RTX 5090"}}))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import db, reports
    world()
    for _, row, name, thinking, gpqa, mmlupro in THREE:
        mid = row.removesuffix(" · thinking")
        db.served_put({"id": mid, "name": name, "base_url": "", "key": "", "how": "rented GPU",
                       "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                       "pin": {"size": 15994966752}, "by": "masein", "at": 0})
        d = reports.row_dir(row)
        d.mkdir(parents=True, exist_ok=True)
        _results(d, mid, "gpqa_diamond_epoch", gpqa, thinking)
        _results(d, mid, "mmlupro_tiger", mmlupro, thinking)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_latest_shows_step2388s_three_versions(live, page, width, scheme):
    go(page, live, width=width, scheme=scheme)
    if width >= 640:
        rows = {r["labels"][0]: r for r in page.evaluate(LATEST)}
        for label, _, _, _, gpqa, _ in THREE:
            r = rows[label]
            assert r["name"] == label and r["ours"] == "true" and r["group"] == "ours"
            assert r["cells"]["gpqa"] == f"{100 * gpqa:.1f}"
            line = page.locator(f"[data-latest-row='{label}'] .mt-line").inner_text()
            assert "k3k6 runtime" in line and "16.0 GB" in line, line
            assert ("thinking on" in line) == (label == "step2388 · High"), line
            assert r["src"].startswith("Our run")
        # never the team's model by another name
        assert "masein model" not in page.locator("[data-latest-table]").inner_text().lower()
        assert no_sideways(page)
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"latest-{width}-{scheme}.png"), full_page=True)
