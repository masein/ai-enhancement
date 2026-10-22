"""27c, Parts 4 and 6 in a real browser: our testers' phones on the model
pages, and the reliability bar's words.

- Part 6: an app model's tests show on every model page of its checkpoint
  (names.json's `checkpoint`), thinking on or off, under "On our testers'
  phones" with the line "the app's step8804 file — not yet checked to be this
  exact file"; with Kevin's sha256 matching the row's pin, "the same file
  (sha256 checked)". A variant (model-003, step4000-denseq8) shows on
  step4000's page under a heading of its own, never mixed with model-002's.
  The On-device report's "On a phone" says the same on hover.
- Part 4: a reliability bar's words sit inside a segment only when they fit.

The world is 23b's (21c's scorecard; invented graded answers for step4000 and
Q4, worked out by the stats worker), with step8804 measured here thinking on
and off, a names file of the app's four models, and made-up tests from made-up
testers on two phones."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from service import config, db, reports, telemetry
from test_21c_page import results
from test_21d_browser import no_sideways
from test_23b_beyond import GPQA, OTIS, write
from test_23b_browser import board, runs  # noqa: F401  the world
from test_26_telemetry import a_test, keep

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27c"
S88 = "served/step8804-lda-2610"
S40 = "served/step4000-lda-2610"
SHA = "c" * 64
NAMES = {
    "models": {
        "model-002": {"name": "step4000", "row": "", "what": "", "checkpoint": "step4000",
                      "sha256": ""},
        "model-003": {"name": "step4000-denseq8", "row": "",
                      "what": "step4000 with its non-expert weights in q8_0 instead of q5_K",
                      "checkpoint": "step4000", "sha256": ""},
        "model-004": {"name": "step8804", "row": "", "what": "", "checkpoint": "step8804",
                      "sha256": ""}},
    "tiers": {"T0": "RAM 14 GB or more"}, "engines": {},
    "devices": {"CPH2747": "OnePlus 15", "SM-S938B": "Galaxy S25 Ultra"}}
# (test id, model, phone, decode)
TESTS = [("a1", "model-004", None, 13.1), ("a2", "model-004", None, 12.9),
         ("a3", "model-004", "SM-S938B", 9.8), ("b1", "model-002", None, 14.4),
         ("c1", "model-003", None, 11.2), ("c2", "model-003", None, 11.6)]


@pytest.fixture(scope="module", autouse=True)
def phones(board, tmp_path_factory):  # noqa: F811
    import service.app as appmod
    from service import report_stats
    # step8804 measured here, thinking on and off, as the board has it
    db.served_put({"id": S88, "name": "step8804-lda-2610", "base_url": "", "key": "",
                   "how": "rented GPU", "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "on",
                   "pin": {}, "by": "masein", "at": 0, "file_sha256": {"sha256": SHA}})
    for row in (S88 + " · thinking", S88):
        d = reports.row_dir(row)
        results(d, S88, "gpqa_diamond_epoch", 0.70, 0.02, 792, 0)
        if not row.endswith(" · thinking"):
            for f in d.glob("*/frontier/results_*.json"):
                f.write_text(f.read_text().replace(",enable_thinking=True", ""))
    # its graded answers: OTIS's bar has a narrow "sometimes" and "never"
    write("step8804", GPQA, runs(30, 4, 18, 3))
    write("step8804", OTIS, runs(9, 8, 6, 1))
    names = tmp_path_factory.mktemp("names27c") / "names.json"
    names.write_text(json.dumps(NAMES))
    saved = telemetry.NAMES
    telemetry.NAMES = names
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
    for i, (tid, model, device, decode) in enumerate(TESTS):
        t = a_test(tid, install=f"00000000-0000-4000-8000-{i:012d}", model=model, device=device,
                   decode=decode, received=f"2026-10-2{i % 3}T10:00:00Z")
        t["label"] = "tester-" + tid[0]
        keep(Path(config.BENCH_ROOT), t)
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield names
    telemetry.NAMES = saved
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)


def page_of(page, live, mid, width=1280, scheme="dark"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(mid, safe=""))
    page.wait_for_selector("[data-kind-block='testers']", timeout=30000)


def test_the_mapping_names_each_checkpoints_rows():
    got = {e["model"]["id"]: e for e in telemetry.by_checkpoint()}
    assert got["model-004"]["rows"] == [S88] and not got["model-004"]["variant"]
    assert got["model-002"]["rows"] == [S40] and not got["model-002"]["variant"]
    assert got["model-003"]["rows"] == [S40] and got["model-003"]["variant"]
    assert [p["n"] for p in got["model-004"]["phones"]] == [1, 2]      # Galaxy, OnePlus
    assert telemetry.checked_line(got["model-004"], S88) == (
        "the app's step8804 file — not yet checked to be this exact file")


@pytest.mark.parametrize("mid", [S88 + " · thinking", S88])
def test_step8804s_pages_both_settings_show_its_tests(live, page, mid):
    page_of(page, live, mid)
    blk = page.locator("[data-kind-block='testers']")
    assert blk.locator(".kblock-k").inner_text() == "On our testers’ phones"
    assert blk.locator("[data-phone-cp-line='model-004']").inner_text() == (
        "the app’s step8804 file — not yet checked to be this exact file")
    rows = blk.locator("[data-phone-tested]").evaluate_all("rs => rs.map(r => r.dataset.phoneTested)")
    assert sorted(rows) == ["Galaxy S25 Ultra", "OnePlus 15"]
    assert blk.locator("[data-phone-tested-decode='OnePlus 15']").inner_text() == "13.0 tok/s"
    assert page.locator("[data-phone-variant]").count() == 0
    assert no_sideways(page) and page.errors == []


def test_a_matching_sha256_says_the_same_file(live, page, phones):
    names = json.loads(phones.read_text())
    names["models"]["model-004"]["sha256"] = SHA
    phones.write_text(json.dumps(names))
    try:
        page_of(page, live, S88 + " · thinking")
        assert page.locator("[data-phone-cp-line='model-004']").inner_text() == (
            "the same file (sha256 checked)")
    finally:
        phones.write_text(json.dumps(NAMES))
    assert page.errors == []


def test_a_variant_has_its_own_heading_on_step4000s_page(live, page):
    page_of(page, live, S40 + " · thinking")
    blk = page.locator("[data-kind-block='testers']")
    # model-002's own, first: step4000's file, one test on OnePlus 15
    assert blk.locator("[data-phone-cp-line='model-002']").inner_text().startswith(
        "the app’s step4000 file")
    own = blk.locator("[data-phone-tested-table]").first
    assert own.locator("[data-phone-tested-decode='OnePlus 15']").inner_text() == "14.4 tok/s"
    # model-003's under its own heading, never mixed in
    var = blk.locator("[data-phone-variant='model-003']")
    assert var.locator("h4").inner_text() == (
        "step4000-denseq8 (step4000 with its non-expert weights in q8_0 instead of q5_K)")
    assert var.locator("[data-phone-tested-decode='OnePlus 15']").inner_text() == "11.4 tok/s"
    assert "11.4" not in own.inner_text() and "14.4" not in var.inner_text()
    assert page.errors == []


def test_on_devices_phone_column_says_the_same_on_hover(live, page):
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=on-device")
    page.wait_for_selector("[data-od-table]", state="attached", timeout=30000)
    cell = page.locator("td[data-od-fact='step8804|phone']")
    assert cell.locator(".mt-n").inner_text() == "13.0 tok/s"
    assert "the app's step8804 file — not yet checked to be this exact file" in cell.get_attribute(
        "data-tip")
    # step4000's is model-002's, never the variant's
    assert page.locator("td[data-od-fact='step4000|phone'] .mt-n").inner_text() == "14.4 tok/s"
    assert page.errors == []


SEGS = """() => [...document.querySelectorAll('svg.by-stack')].flatMap(svg =>
  [...svg.querySelectorAll('[data-by-seg-text]')].map(t => {
    const r = svg.querySelector(`[data-by-seg="${t.dataset.bySegText}"]`).getBBox();
    const b = t.getBBox();
    return { text: t.textContent, in: b.x >= r.x - 0.5 && b.x + b.width <= r.x + r.width + 0.5 };
  }))"""


@pytest.mark.parametrize("width", [1280, 400])
def test_step8804s_page_its_phone_block_and_reliability_bars(live, page, width):
    page_of(page, live, S88 + " · thinking", width=width)
    blk = page.locator("[data-kind-block='beyond']")
    blk.locator("summary").click()
    page.wait_for_selector("[data-by-part='reliability'] svg.by-stack", timeout=30000)
    segs = page.evaluate(SEGS)
    assert segs and all(s["in"] for s in segs), segs
    # OTIS: "sometimes" and "never" too narrow for their words: the key gives them
    otis = page.locator("[data-by-part='reliability'] svg.by-stack").nth(1)
    assert otis.locator("[data-by-seg-text]").count() < otis.locator("[data-by-seg]").count()
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(300)
    page.screenshot(path=str(SCREENS / f"step8804-page-{width}-dark.png"), full_page=True)
