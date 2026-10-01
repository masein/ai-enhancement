"""12z B on the page: Test a model closes when the page changes (B3), "← Back
to …" goes back and names where (B4), the chip in use is in sight and the
On-device chart has no Benchmarks picker (B5), and "sub=" opens a Models
view (B6) — at 1400 and 375 px. The GGUF block and the Measure dialog's
defaults (B1, B2) are in test_12n1_browser, test_gguf_12f3_browser and
test_12f4_browser, beside their fixtures. Fixtures only."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12z"
MODEL = "fx/good-750m"
WIDTHS = [1400, 375]


def shot(loc, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    loc.screenshot(path=SCREENS / name)


def start(page, live, width, at="#tab=home"):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    if at != "#tab=home":
        page.goto(live["base"] + "/" + at)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_closes_when_the_page_changes(live, page, width):
    start(page, live, width, "#tab=models")
    page.wait_for_selector(f"tr[data-lb-row='{MODEL}']")
    page.locator(f"tr[data-lb-row='{MODEL}'] td.num").first.click()
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    assert dlg.locator("[data-ms='submit'] input").input_value() == MODEL
    # Back: the dialog was this page's
    page.go_back()
    page.wait_for_selector("[data-lb-table]")
    assert dlg.count() == 0 and page.evaluate("state.testOpen") is False
    # an address typed in, or a link followed, leaves it too
    page.go_forward()
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-test-model]").click()
    dlg.wait_for()
    page.evaluate("location.hash = '#tab=home'")
    page.wait_for_selector("[data-dialog='test']", state="detached")
    assert page.evaluate("state.tab") == "overview"
    no_sideways(page)
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_back_goes_back_and_names_where_it_goes(live, page, width):
    # from Models ▸ Math, the link says so, and is Back
    start(page, live, width, "#tab=models&chip=math")
    page.wait_for_selector(f"tr[data-lb-row='{MODEL}']")
    page.locator(f"tr[data-lb-row='{MODEL}'] td.num").first.click()
    page.wait_for_selector("[data-model-hero]")
    back = page.locator(".backlink")
    assert back.inner_text() == "← Back to Models · Math"
    assert back.get_attribute("data-back-to") == "history"
    n = page.evaluate("history.length")
    back.click()
    page.wait_for_function("location.hash === '#tab=models&chip=math'")
    assert page.evaluate("history.length") == n
    # from another model's page, by a link: back to that model, by its name
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    other = page.evaluate(f"DATA.models.find(m => m.id !== '{MODEL}' && !m.duplicateOf).id")
    name = page.evaluate(f"DATA.models.find(m => m.id === '{MODEL}').name")
    page.evaluate("id => { const a = document.createElement('a'); a.href = '#model=' + "
                  "encodeURIComponent(id); a.id = 'go-other'; document.querySelector('#view')"
                  ".prepend(a); a.textContent = 'x'; a.click(); }", other)
    page.wait_for_function(f"state.model === {other!r}")
    back = page.locator(".backlink")
    assert back.inner_text() == f"← Back to {name}"
    back.click()
    page.wait_for_function(f"state.model === {MODEL!r}")
    # opened from nowhere here: the view it sits under, by its name
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + MODEL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    back = page.locator(".backlink")
    assert back.get_attribute("data-back-to") == "view"
    assert back.inner_text() == "← Back to Home"
    no_sideways(page)
    assert page.errors == []


def test_back_names_each_kind_of_address(live, page):
    start(page, live, 1400)
    words = page.evaluate("""() => ['tab=models&chip=devicemark', 'tab=models&view=everyday',
        'tab=models&view=exam&chip=judged', 'tab=benchmarks&sub=everyday', 'tab=improve&sub=model',
        'tab=runs', '', 'tab=models&view=compare&m=a,b'].map(hashWords)""")
    assert words == ["Models · DeviceMark", "Models · Everyday tasks", "Models · Knowledge exam",
                     "Benchmarks · Everyday tasks", "Improve · By model", "All runs", "Home",
                     "Compare"]
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("chip", ["ondevice", "frontier", "lm"])
def test_the_chip_in_use_is_in_sight(live, page, width, chip):
    start(page, live, width, f"#tab=models&chip={chip}")
    on = page.locator(f"[data-lb-chips] [data-chip='{chip}']")
    on.wait_for()
    assert "on" in on.get_attribute("class").split()
    page.wait_for_timeout(100)
    inside = page.evaluate("""() => {
        const row = document.querySelector('[data-lb-chips]'), on = row.querySelector('.chip-btn.on');
        const r = row.getBoundingClientRect(), c = on.getBoundingClientRect();
        return c.left >= r.left - 1 && c.right <= r.right + 1; }""")
    assert inside
    # the row says there is more to one side or the other when it scrolls
    wide = page.evaluate("(r => r.scrollWidth > r.clientWidth + 1)(document.querySelector('[data-lb-chips]'))")
    if wide:
        row = page.locator("[data-lb-chips]")
        assert "1" in (row.get_attribute("data-less"), row.get_attribute("data-more"))
    # the On-device chart and Frontier have no benchmark columns to pick
    assert page.locator("[data-benchmarks-menu]").count() == (1 if chip == "lm" else 0)
    no_sideways(page)
    if chip == "ondevice":
        shot(page.locator(".lbbar"), f"chips-{chip}-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("sub,view", [("everyday", "everyday"), ("exam", "exam"),
                                      ("knowledge", "exam"), ("standard", "standard")])
def test_a_models_view_has_an_address_by_sub(live, page, width, sub, view):
    start(page, live, width, f"#tab=models&sub={sub}")
    page.wait_for_selector("[data-lb-table], [data-lb-everyday]")
    assert page.evaluate("lbS().view") == view
    # the address bar shows it the way the board writes it
    want = "#tab=models" + ("" if view == "standard" else f"&view={view}")
    page.wait_for_function(f"location.hash === {want!r}")
    no_sideways(page)
    assert page.errors == []
