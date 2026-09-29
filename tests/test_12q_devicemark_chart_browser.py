"""12q.B on the page: Models ▸ On-device chart — DeviceMark's board as
published and our rows by their protocol, on their axes. A row with no speed
measured on a device is a dashed line, never a point (and never at the
server's speed); their rows are read-only, the credit and the note on them;
our run of one of their models sits in their row and their point's hover,
never plotted at their device's speed; the table ranks everyone together; a
device's speed is entered here; the accuracy-against-budget chart draws the
ticked rows. Fixtures only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import set_name
from test_12q_devicemark_board import SERVED, _row

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12q"
MTP = "served/Qwen3.6-k4-LDA-MTP"
ORIG = "served/Qwen3.6-35B-A3B-Q4-original-k-8"
NEMO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
THEIR_NEMO = "nemotron-4b__int8hu__aimodel"


@pytest.fixture
def ours(live):
    out = Path(live["root"]) / "results" / "full"
    _row(out, MTP, {"ifeval": 0.7, "mmlu_pro": 0.65, "math": 0.6},
         {**SERVED, "phone": True, "name": "Qwen3.6 k4-LDA MTP"}, speed=31.5,
         device={"tok_s": 14.2, "device": "iPhone 17 Pro",
                 "source": "measured by a colleague, 29 Sep 2026", "at": 1.0})
    _row(out, ORIG, {"ifeval": 0.8, "mmlu_pro": 0.7, "math": 0.7},
         {**SERVED, "phone": False, "name": "Qwen3.6 original k=8"}, speed=12.0)
    _row(out, NEMO, {"ifeval": 0.5, "mmlu_pro": 0.6, "math": 0.8},
         {"runtime": "hf transformers (lm_eval)", "dtype": "bfloat16", "phone": False,
          "lookahead": False})
    made = [out / m.replace("/", "__") for m in (MTP, ORIG, NEMO)]
    yield out
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


def open_chart(page, live, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&chip=ondevice")
    page.wait_for_selector("[data-dm-chart]")


def test_the_chart_draws_rows_on_a_device_as_points_and_the_rest_as_lines(live, page, ours):
    page.goto(live["base"] + "/")
    set_name(page, "masein")
    open_chart(page, live)
    chart = page.locator("[data-dm-chart]")
    # theirs: nine measured on the iPhone, the two cloud APIs and the built-in model as lines
    assert chart.locator("[data-dm-point][data-dm-external]").count() == 9
    assert chart.locator("[data-dm-line][data-dm-external]").count() == 3
    # ours: a point only with a speed from a device — never at the server's
    assert chart.locator(f"[data-dm-point='{MTP}']").count() == 1
    assert chart.locator(f"[data-dm-point='{ORIG}']").count() == 0
    assert chart.locator(f"[data-dm-line='{ORIG}']").count() == 1
    # our run of one of their models: never plotted — in their point's hover
    assert chart.locator(f"[data-dm-line='{NEMO}'], [data-dm-point='{NEMO}']").count() == 0
    nemo = json.loads(chart.locator(f"[data-dm-point='{THEIR_NEMO}']").get_attribute("data-tip"))
    assert nemo[1] == "theirs (int8, iPhone): composite 61.4 [57.8, 65.1]"
    assert nemo[3].startswith("ours (bf16, our battery): composite ") \
        and nemo[3].endswith(" · calibration")
    assert "never plotted at their device speed" in nemo[5]
    assert chart.locator(f"[data-dm-whisker='{MTP}']").count() == 1
    tip = json.loads(chart.locator(f"[data-dm-point='{MTP}']").get_attribute("data-tip"))
    assert "14.2 tok/s decode on iPhone 17 Pro · measured by a colleague, 29 Sep 2026" in tip
    lfm = json.loads(chart.locator("[data-dm-point='lfm2.5-1.2b__int8hu__aimodel']")
                     .get_attribute("data-tip"))
    assert "DeviceMark (devicemark.github.io), CC-BY-4.0" in lfm
    assert any("don't reproduce these numbers" in x and "board as published" in x for x in lfm)
    cap = page.locator("[data-ondevice-caption]").inner_text()
    assert "the same 300 IFEval items" in cap and "our draw of the same design" in cap
    assert "never on the chart" in cap
    assert "CC-BY-4.0" in page.locator("[data-ondevice-credit]").inner_text()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-ondevice]").screenshot(path=SCREENS / "on-device-1400.png")
    # their point's hover, with ours in it
    page.locator(f"[data-dm-point='{THEIR_NEMO}']").hover()
    page.wait_for_function("getComputedStyle(document.querySelector('#tip')).opacity === '1'")
    assert "ours (bf16, our battery): composite" in page.locator("#tip").inner_text()
    page.locator("#tip").screenshot(path=SCREENS / "their-point-hover-1400.png")
    assert page.errors == []


def test_the_table_ranks_everyone_theirs_read_only(live, page, ours):
    open_chart(page, live)
    t = page.locator("[data-dm-table]")
    assert t.locator("tbody tr").count() == 14                  # ours of theirs: in their row
    lfm = t.locator("[data-dm-row='lfm2.5-1.2b__int8hu__aimodel']")
    assert "68.2 ±3.8" in lfm.inner_text()
    assert lfm.locator("[data-dm-whose='theirs']").inner_text() == "DeviceMark"
    assert lfm.locator("[data-dm-device-edit]").count() == 0             # theirs: read-only
    assert lfm.locator("[data-dm-server='lfm2.5-1.2b__int8hu__aimodel']").inner_text() == "n/a"
    mine = t.locator(f"[data-dm-row='{MTP}']")
    assert mine.locator("[data-dm-whose='ours']").inner_text() == "ours"
    assert mine.locator(f"[data-dm-server='{MTP}']").inner_text() == "31.5"
    # a phone build over the original: the same items, thinking, no lookahead
    assert mine.locator(f"[data-dm-retention='{MTP}']").inner_text() == "88% · 93% · 86%"
    assert t.locator(f"[data-dm-row='{NEMO}']").count() == 0
    nemo = t.locator(f"[data-dm-row='{THEIR_NEMO}']")
    comp = nemo.locator(f"[data-dm-composite='{THEIR_NEMO}']")
    assert comp.locator(f"[data-dm-theirs='{THEIR_NEMO}']").inner_text() == \
        "theirs (int8, iPhone): 61.4 ±3.6"
    assert comp.locator(f"[data-dm-ours='{NEMO}']").inner_text().startswith(
        "ours (bf16, our battery): ")
    assert nemo.locator(f"[data-dm-ours='{NEMO}']").count() == 6    # composite, 3 benches, 2 more
    assert nemo.locator("[data-dm-whose]").inner_text() == "DeviceMark · and ours"
    assert nemo.locator("[data-dm-device-edit]").count() == 0
    ranks = t.locator("[data-dm-rank]").all_inner_texts()
    assert ranks[0] == "=1" and all(r for r in ranks)
    assert page.errors == []


def test_a_speed_from_a_device_is_entered_and_puts_the_row_on_the_chart(live, page, ours):
    page.goto(live["base"] + "/")
    set_name(page, "masein")
    open_chart(page, live)
    page.locator(f"[data-dm-device-edit='{ORIG}']").click()
    form = page.locator(f"[data-dm-device-form='{ORIG}']")
    form.locator("[data-dm-device='tok_s']").fill("9.5")
    form.locator("[data-dm-device='device']").fill("Pixel 10")
    form.locator("[data-dm-device-save]").click()
    # the server's own words, once it has answered
    page.locator("[data-dm-device-note]:has-text('where it came from')").wait_for()
    form.locator("[data-dm-device='source']").fill("measured by a colleague, 30 Sep 2026")
    form.locator("[data-dm-device-save]").click()
    page.wait_for_selector(f"[data-dm-point='{ORIG}']")
    assert page.locator(f"[data-dm-line='{ORIG}']").count() == 0
    saved = json.loads((ours / ORIG.replace("/", "__") / "devicemark_device.json").read_text())
    assert (saved["tok_s"], saved["device"], saved["by"]) == (9.5, "Pixel 10", "masein")
    # the one refusal asked for above, and nothing else
    assert [e for e in page.errors if "status of 422" not in e] == []


def test_accuracy_against_budget_draws_the_ticked_rows(live, page, ours):
    open_chart(page, live)
    box = page.locator("[data-dm-budget-box]")
    assert box.locator("[data-dm-budget-line]").count() == 2           # ours, ticked at first
    page.locator("[data-dm-sel='lfm2.5-1.2b__int8hu__aimodel']").check()
    assert box.locator("[data-dm-budget-line='lfm2.5-1.2b__int8hu__aimodel']").count() == 1
    # a row of theirs with our run of it: both drawn
    page.locator(f"[data-dm-sel='{THEIR_NEMO}']").check()
    assert box.locator(f"[data-dm-budget-line='{THEIR_NEMO}']").count() == 1
    assert box.locator(f"[data-dm-budget-line='{NEMO}']").count() == 1
    box.screenshot(path=SCREENS / "budget-1400.png")
    assert page.errors == []


def test_frontier_links_to_it(live, page, ours):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=models&chip=frontier")
    page.locator("[data-frontier-ondevice-go]").click()
    page.wait_for_selector("[data-dm-chart]")
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_at_400px(live, page, ours, scheme):
    page.emulate_media(color_scheme=scheme)
    open_chart(page, live, width=400)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    page.locator("[data-ondevice]").screenshot(path=SCREENS / f"on-device-400-{scheme}.png")
    assert page.errors == []
