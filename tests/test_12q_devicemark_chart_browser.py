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
    assert nemo[1] == "theirs (int8, scored on a Mac; speed on iPhone 17 Pro): composite 61.4 [57.8, 65.1]"
    # 12q.D: not calibration any more — it can't run here — though a run of ours still shows
    assert nemo[3].startswith("ours (bf16, our battery): composite ") \
        and not nemo[3].endswith(" · calibration")
    assert not any("can't run here" in x for x in nemo)
    assert any("never plotted at their device speed" in x for x in nemo)
    # 12q.E: this made-up run's answers are far shorter than theirs: another mode, said so
    assert any(x.startswith("the modes differ: ours answered ") for x in nemo)
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
        "theirs (int8, scored on a Mac; speed on iPhone 17 Pro): 61.4 ±3.6"
    assert comp.locator(f"[data-dm-ours='{NEMO}']").inner_text().startswith(
        "ours (bf16, our battery): ")
    assert nemo.locator(f"[data-dm-ours='{NEMO}']").count() == 6    # composite, 3 benches, 2 more
    assert nemo.locator("[data-dm-whose]").inner_text() == "DeviceMark · and ours"
    assert nemo.locator("[data-dm-device-edit]").count() == 0
    # 12q.B2: the cloud APIs unranked (☁), as on DeviceMark's board — LFM2.5-1.2B is =1 again
    rank = {r.get_attribute("data-dm-rank"): r.inner_text()
            for r in t.locator("[data-dm-rank]").all()}
    assert rank["gemini-flash__api__api"] == rank["gemini-pro__api__api"] == "☁"
    assert rank["lfm2.5-1.2b__int8hu__aimodel"] == "=1"
    assert all(rank.values())
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


# ---------------------------------------------------------------------------
# 12q.B2: every label clear, the chart at its card's width, ours by its setup
# ---------------------------------------------------------------------------

CLASHES = """() => {
  const svg = document.querySelector('[data-dm-chart]');
  const hit = (a, b) => a.x < b.x + b.width - 0.5 && b.x < a.x + a.width - 0.5
    && a.y < b.y + b.height - 0.5 && b.y < a.y + a.height - 0.5;
  const labels = [...svg.querySelectorAll('[data-dm-label]')].map(t =>
    ({ id: t.dataset.dmLabel, b: t.getBBox() }));
  const points = [...svg.querySelectorAll('[data-dm-point]')].map(c => ({ id: c.dataset.dmPoint,
    b: { x: +c.getAttribute('cx') - 5, y: +c.getAttribute('cy') - 5, width: 10, height: 10 } }));
  const lines = [...svg.querySelectorAll('[data-dm-line]')].map(l => ({ id: l.dataset.dmLine,
    b: { x: +l.getAttribute('x1'), y: +l.getAttribute('y1') - 0.6,
         width: +l.getAttribute('x2') - +l.getAttribute('x1'), height: 1.2 } }));
  const whiskers = [...svg.querySelectorAll('[data-dm-whisker]')].map(l => ({
    id: l.dataset.dmWhisker + ' (whisker)', b: { x: +l.getAttribute('x1') - 0.6,
      y: Math.min(+l.getAttribute('y1'), +l.getAttribute('y2')), width: 1.2,
      height: Math.abs(+l.getAttribute('y2') - +l.getAttribute('y1')) } }));
  const out = [];
  labels.forEach((a, i) => {
    labels.slice(i + 1).forEach(b => { if (hit(a.b, b.b)) out.push(`${a.id} × ${b.id}`); });
    for (const o of [...points, ...lines, ...whiskers])
      if (hit(a.b, o.b)) out.push(`${a.id} × ${o.id}`);
  });
  return { n: labels.length, out, w: svg.getBoundingClientRect().width,
           card: document.querySelector('[data-ondevice]').clientWidth };
}"""


@pytest.mark.parametrize("width", [1400, 400])
def test_no_label_sits_on_another_a_point_a_whisker_or_a_line(live, page, ours, width):
    open_chart(page, live, width=width)
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-dm-chart]").screenshot(path=SCREENS / f"chart-labels-{width}.png")
    got = page.evaluate(CLASHES)
    assert got["n"] == 10                                  # their nine on the iPhone, and ours
    assert got["out"] == []
    if width == 1400:
        assert got["w"] > 0.9 * got["card"] - 60           # the card's width, not 900
    else:
        assert got["w"] == 900                             # scrolls on a phone
    assert page.errors == []


def test_our_rows_are_named_by_their_setup_on_the_chart(live, page, ours):
    open_chart(page, live)
    chart = page.locator("[data-dm-chart]")
    assert chart.locator(f"[data-dm-line-label='{ORIG}']").text_content() == \
        "original (k=8) · MTP · thinking off"
    assert chart.locator(f"[data-dm-label='{MTP}']").text_content() == \
        "phone build (k4-LDA) · MTP · thinking off · iPhone 17 Pro"
    # the row's name stays in the table
    assert page.locator(f"[data-dm-row='{ORIG}'] td").nth(1).inner_text().startswith(
        "Qwen3.6 original k=8")
    tip = json.loads(chart.locator(f"[data-dm-line='{ORIG}']").get_attribute("data-tip"))
    assert tip[:2] == ["Qwen3.6 original k=8", "original (k=8) · MTP · thinking off"]
    assert page.errors == []
