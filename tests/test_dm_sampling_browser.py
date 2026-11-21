"""19 Nov (masein) on the page: Mobile ▸ DeviceMark hides the greedy runs of our
own models until "Show greedy runs of our models" is on, gives every row its own
sampling (a Sampling column and the hover), says so in the caption instead of
"greedy" for every row, and names step2388's two k3 runs apart — greedy (run
419) beside the file's sampling (run 428). The table view and Mobile ▸ All read
the same switch. Fixtures only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from test_12q_devicemark_board import _row
from test_dm_sampling import ACC, RUNS

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "dm-sampling"
GREEDY = "served/step2388-k3k6-2611"                    # as run 419
FILE = "served/step2388-k3k6-2611-file"                 # as run 428
PHONE = "served/Qwen3.6-k4-LDA-lookahead"
STEP8804_ON = "served/step8804-lda-2610 · thinking"
KEEP = (GREEDY, FILE, PHONE, STEP8804_ON)


@pytest.fixture
def step(live):
    import service.app as appmod
    out = Path(live["root"]) / "results" / "full"
    for rid in KEEP:
        su, thinks = RUNS[rid]
        _row(out, rid.removesuffix(" · thinking"), ACC, su, thinking=thinks)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield out
    for rid in KEEP:
        for d in out.glob(rid.removesuffix(" · thinking").replace("/", "__") + "*"):
            shutil.rmtree(d, ignore_errors=True)
    appmod._cache.update(key=None, payload=None, at=0.0)


def open_chart(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=ondevice")
    page.wait_for_selector("[data-dm-chart]")


LINE_LABELS = """() => Object.fromEntries([...document.querySelectorAll('[data-dm-line-label]')]
  .map(t => [t.dataset.dmLineLabel, t.textContent]))"""


@pytest.mark.parametrize("width", [1400, 375])
def test_greedy_runs_of_our_models_are_hidden_until_switched_on(live, page, step, width):
    open_chart(page, live, width)
    t = page.locator("[data-dm-table]")
    assert t.locator(f"tr[data-dm-row='{GREEDY}']").count() == 0
    assert t.locator(f"tr[data-dm-row='{STEP8804_ON}']").count() == 0
    assert t.locator(f"tr[data-dm-row='{FILE}']").count() == 1
    assert t.locator(f"tr[data-dm-row='{PHONE}']").count() == 1        # Qwen3.6's builds stay
    assert "Show greedy runs of our models (2)" in page.locator("[data-dm-greedy]").inner_text()
    # every row its own sampling: ours, and DeviceMark's own as their protocol samples
    f = page.locator(f"[data-dm-sampling='{FILE}']")
    assert f.inner_text() == "the file's sampling"
    assert json.loads(f.get_attribute("data-tip")) == [
        "the model file's own sampling for expert path k3: nothing sent but the path and seed 0"]
    assert page.locator(f"[data-dm-sampling='{PHONE}']").inner_text() == "greedy"
    theirs = page.locator("tr[data-dm-external] [data-dm-sampling]")
    assert theirs.count() > 0 and set(theirs.all_inner_texts()) == {"greedy"}
    # the caption no longer says every row is greedy
    cap = page.locator("[data-ondevice-caption]").inner_text()
    assert "greedy, a cap of" not in cap
    assert "Each row says how its answers were sampled (Sampling)" in cap
    assert "Greedy runs of our own models (step2388, step8804) are hidden" in cap
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-ondevice]").screenshot(path=SCREENS / f"greedy-hidden-{width}.png")

    page.locator("[data-dm-greedy-toggle]").check()
    # under 640 px the table's rows are drawn as cards, the row itself kept but not shown
    page.wait_for_selector(f"tr[data-dm-row='{GREEDY}']", state="attached")
    assert page.locator(f"[data-dm-sampling='{GREEDY}']").inner_text() == "greedy · hidden by default"
    assert page.locator(f"[data-dm-rank='{GREEDY}']").inner_text().strip() == ""    # never ranked
    # 419 and 428 apart, on the chart's lines and in their rows
    labels = page.evaluate(LINE_LABELS)
    assert labels[GREEDY] == "step2388 · k3 · lookahead · greedy"
    assert labels[FILE] == "step2388 · k3 · lookahead · the file’s sampling"
    full = page.locator(f"tr[data-dm-row='{GREEDY}']").get_attribute("data-dm-row")
    assert full == GREEDY
    page.locator("[data-ondevice]").screenshot(path=SCREENS / f"greedy-shown-{width}.png")
    assert page.errors == []


def test_the_hover_says_the_sampling_and_why_hidden(live, page, step):
    open_chart(page, live)
    page.locator("[data-dm-greedy-toggle]").check()
    page.wait_for_selector(f"tr[data-dm-row='{GREEDY}']")
    tip = page.evaluate(f"dmTip(dmRows().find(r => r.id === {json.dumps(GREEDY)}))")
    assert "sampling: greedy, DeviceMark's protocol: temperature 0, seed 0, presence penalty 0" in tip
    assert any("a greedy run of our model step2388" in x for x in tip)
    ftip = page.evaluate(f"dmTip(dmRows().find(r => r.id === {json.dumps(FILE)}))")
    assert any(x.startswith("sampling: the model file's own sampling for expert path k3") for x in ftip)
    assert not any("greedy run of our model" in x for x in ftip)


def test_the_table_view_reads_the_same_switch(live, page, step):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=devicemark")
    page.wait_for_selector("[data-dm-greedy]")
    g, f = json.dumps(GREEDY), json.dumps(FILE)
    assert page.evaluate(f"dmVal({g}, 'composite')") is None
    assert page.evaluate(f"dmVal({f}, 'composite')") is not None
    assert page.evaluate("dmHiddenN()") == 2
    assert "Show greedy runs of our models (2)" in page.locator("[data-dm-greedy]").inner_text()
    page.locator("[data-dm-greedy-toggle]").check()
    page.wait_for_function(f"dmVal({g}, 'composite') !== null")
    assert page.errors == []
