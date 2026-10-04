"""16.1 on the page: Size's five buckets, more than one ticked, and "Size not
recorded (n)" last; a size filter never hides a row silently ("N models have
no size recorded and are hidden · show them"); "35B · 3B active" with where it
came from on hover; the size entered on a model's page; and the registration
form's suggestion from a name, behind a button for the person to confirm. At
1400 and 375 px. A fake server; nothing runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import choose_where, open_filters, set_name
from fake_openai import FakeServer
from test_14_3_browser import go, no_sideways, steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_1"
ME = "masein"


def api(live, path, body):
    req = urllib.request.Request(live["base"] + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


@pytest.fixture(scope="module", autouse=True)
def rows(live):
    """two Qwen3.6-35B-A3B builds served elsewhere: one with its size entered,
    one with none (as the 9 were before the deploy step's command)"""
    import service.app as appmod
    fake = FakeServer()
    sized = api(live, "/api/served", {"name": "Q36 LDA sized", "base_url": fake.base,
                                      "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                      "thinking": "off", "size": "35B", "active": "3B",
                                      "by": ME})["model"]["id"]
    bare = api(live, "/api/served", {"name": "Q36 LDA unsized", "base_url": fake.base,
                                     "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "k4",
                                     "thinking": "off", "by": ME})["model"]["id"]
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"sized": sized, "bare": bare}
    fake.close()


def kept(page) -> set[str]:
    """the rows the toolbar's filters keep — tested or listed as not tested yet"""
    return set(page.evaluate("lbFilter(DATA.models).map(m => m.id)"))


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


@pytest.mark.parametrize("width", [1400, 375])
def test_the_size_menu_and_a_filter_never_hides_a_row_silently(live, page, rows, width):
    go(page, live, "tab=leaderboard", "[data-lb-table] tbody tr[data-lb-row]", width)
    open_filters(page)
    page.locator("#pill-size").click()
    menu = page.locator("#pop-size")
    choices = menu.locator("[data-choice]").evaluate_all("xs => xs.map(x => x.textContent)")
    assert choices[:6] == ["All", "< 200M", "200M–1B", "1–3B", "3–9B", "> 9B"]
    assert choices[-1].startswith("Size not recorded (") and choices[-1].endswith(")")
    shot(menu, f"size-menu-{width}.png")
    menu.locator("[data-choice='xx']").click()
    page.wait_for_selector("#pill-size[data-value='xx']")
    # the 35B build is > 9B: its total, never its 3B active
    assert rows["sized"] in kept(page)
    hidden = page.locator("[data-size-hidden]")
    n = int(hidden.get_attribute("data-size-hidden"))
    assert n >= 1 and hidden.inner_text().startswith(
        f"{n} model{'s have' if n > 1 else ' has'} no size recorded and")
    assert rows["bare"] not in kept(page)
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"size-filter-{width}.png")
    hidden.locator("[data-size-show]").click()
    page.wait_for_selector("#pill-size[data-value='xx,none']")
    assert rows["bare"] in kept(page)
    assert page.locator("[data-size-hidden]").count() == 0
    # more than one ticked, in the address
    assert "size=xx%2Cnone" in page.evaluate("location.hash") or "size=xx,none" in page.evaluate(
        "location.hash")
    assert page.errors == []


def test_an_old_address_for_over_3b_opens_both_new_buckets(live, page):
    go(page, live, "tab=leaderboard&size=xl", "[data-lb-table]")
    open_filters(page)
    assert page.locator("#pill-size").get_attribute("data-value") == "x,xx"
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_model_page_says_its_size_and_where_it_came_from(live, page, rows, width):
    go(page, live, "model=" + quote(rows["sized"], safe=""), "[data-model-hero]", width)
    size = page.locator(f"[data-model-size='{rows['sized']}']")
    assert size.inner_text() == "35B · 3B active"
    assert size.get_attribute("title") == "entered by masein"
    no_sideways(page)
    shot(page.locator("[data-model-hero]"), f"model-size-{width}.png")
    assert page.errors == []


def test_a_size_not_recorded_is_entered_on_the_models_page(live, page, rows):
    width = 1400
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, ME)
    # the unsized one: said so, and entered here — its base's name only suggests
    go(page, live, "model=" + quote(rows["bare"], safe=""), "[data-model-hero]", width)
    assert page.locator(f"[data-model-size='{rows['bare']}']").inner_text() == "size not recorded"
    page.locator(f"[data-size-edit='{rows['bare']}']").click()
    form = page.locator(f"[data-size-form='{rows['bare']}']")
    assert form.locator("[data-size-input='total']").input_value() == "35B"     # suggested
    assert form.locator("[data-size-input='active']").input_value() == "3B"
    shot(form, f"size-edit-{width}.png")
    form.locator(f"[data-size-save='{rows['bare']}']").click()
    page.wait_for_function(f"document.querySelector(\"[data-model-size='{rows['bare']}']\")"
                           "?.textContent === '35B · 3B active'")
    assert page.errors == []


def test_the_registration_form_suggests_a_size_from_the_name_behind_a_button(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, ME)
    page.locator("[data-test-model]").click()
    choose_where(page, "server")
    page.locator("#srv-name").fill("Qwen3.6-35B-A3B k4-LDA (phone build)")
    page.locator("#srv-based_on").fill("Qwen/Qwen3.6-35B-A3B")
    page.locator("#srv-name").press("Tab")
    btn = page.locator("[data-size-suggest='srv']")
    btn.wait_for()
    assert btn.inner_text() == "Use 35B · 3B active (from its name)"
    # nothing is filled until the person says so
    assert page.locator("#srv-size").input_value() == ""
    btn.click()
    assert (page.locator("#srv-size").input_value(), page.locator("#srv-active").input_value()) == (
        "35B", "3B")
    shot(page.locator("[data-size-fields='srv']").locator("xpath=.."), "register-size-1400.png")
    assert page.errors == []
