"""16b's thinking rule on the page: said on the cards of the suites it decides
(MobileAIBench, Mobile-MMLU, Instruction & maths, Frontier's two), on the row
("thinking off" where the rule decides; a thinking row's badge says it too),
and in Test a model — the rule above the parts, the choice of a thinking run
for a served model, and each part's time at the pace of the setting asked.
At 1400 and 375 px. A fake server and answers written in; nothing runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import mobileaibench as mab
from conftest import set_name
from fake_openai import FakeServer
from service import config, served
from test_14_1_mab_text import sit
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16b_thinking"
WIDTHS = [1400, 375]
RULE = ("Thinking off for a model that can turn it off, served models too, unless a thinking "
        "run is asked for: that is a row of its own (“· thinking”).")


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def lda(live):
    """a served model that decides for itself, measured with its thinking off;
    its Mobile answers in its row, and its thinking row's"""
    import service.app as appmod
    fake = FakeServer()
    got = api(live, "/api/served", {"name": "LDA auto", "base_url": fake.base, "how": "k4",
                                    "thinking": "auto", "by": "masein"})
    mid = (got.get("model") or got)["id"]
    served.record_speed(mid, 4.0, 80, "off")
    row = config.OUT_DIR / mid.replace("/", "__")
    think = row.with_name(row.name + "__thinking")
    sit(row, mab.HOTPOT)
    sit(think, mab.HOTPOT, answer=lambda q: "<think>hm</think>\n\nan answer of six words")
    # as a thinking run's results name it (served._write)
    for f in think.rglob("results*.json"):
        blob = json.loads(f.read_text())
        blob["config"]["model_args"] = f"pretrained={mid},enable_thinking=True"
        f.write_text(json.dumps(blob))
    (think / "model_meta.json").write_text(json.dumps(
        {"model": mid + " · thinking", "base_model": mid, "kind": "instruct", "params": None,
         **served.archinfo(served.get(mid))}))
    for d in (row, think):
        mab.write(d, mab.mark(d))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield mid
    fake.close()


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_the_rule_is_on_the_cards_of_the_suites_it_decides(live, page, lda):
    go(page, live, "tab=benchmarks", "[data-catalog]")
    assert page.evaluate("THINKING_RULE_WORDS") == RULE
    for key in ("mobileaibench", "ifeval", "mmlu_pro", "hendrycks_math500", "gpqa", "simpleqa"):
        card = page.locator(f"[data-cat-card='{key}']")
        assert card.locator(f"[data-cat-protocol='{key}']").inner_text() == RULE, key
    mm = page.locator("[data-cat-protocol='mobile_mmlu']").inner_text()
    assert mm.startswith("A Hugging Face model is asked with no chat template") and mm.endswith(RULE)
    # Everyday, Trust & safety and the exam keep their template's default: not said there
    for key in ("everyday", "bbq"):
        loc = page.locator(f"[data-cat-card='{key}'] [data-cat-protocol]")
        assert not loc.count() or RULE not in loc.inner_text(), key
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-cat-card='mobileaibench']"), SCREENS / "card-1400.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_rule_is_on_the_row_where_it_decides(live, page, lda, width):
    go(page, live, "tab=models&chip=mobile", "[data-lb-table]", width)
    off = page.locator(f"[data-thinking-off='{lda}']")
    off.wait_for(state="attached")
    assert off.inner_text() == "thinking off"
    assert off.get_attribute("title") == "Asked with thinking off.\n\n" + RULE
    on = page.locator(f"[data-thinking-badge='{lda} · thinking']")
    assert on.inner_text() == "thinking" and on.get_attribute("title").endswith("\n\n" + RULE)
    # the name's tooltip says it too (the badge gives way before its row's
    # name is cut), and the two rows never read alike
    cell = page.locator(f"[data-lb-table] td.model[data-model='{lda}']")
    assert cell.get_attribute("title").endswith("\n\nAsked with thinking off. " + RULE)
    names = page.evaluate("DATA.models.filter(m => m.id.startsWith(%r)).map(m => m.name)" % lda)
    assert sorted(names) == ["LDA auto", "LDA auto · thinking"]
    no_sideways(page)
    if width == 1400:
        steady_shot(page.locator("[data-lb-card]"), SCREENS / "mobile-rows-1400.png")
    # Standard's All: its rule isn't the suite's, so nothing is said there
    go(page, live, "tab=models", "[data-lb-table]", width)
    assert page.locator("[data-thinking-off]").count() == 0
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_says_the_rule_and_the_time_at_the_settings_pace(live, page, lda, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(lda, safe=""), f"[data-test-model][data-test-this='{lda}']",
       width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    page.locator("[data-mab-opts]").wait_for()
    assert page.locator("[data-thinking-rule='mobile']").inner_text() == RULE
    box = page.locator("[data-mab-think] input[type='checkbox']")
    assert not box.is_checked()
    est = page.locator("[data-mab-part-est='none']")
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='none']\")"
                           "?.textContent.includes('answers')")
    assert est.inner_text().strip() == ("· 5,000 answers, about 5.6 h, at its measured 4.0 s an "
                                        "answer with thinking off")
    no_sideways(page)
    steady_shot(page.locator("[data-dialog='test'] .dlg"), SCREENS / f"test-off-{width}.png")
    # a thinking run: its own pace, not measured yet — said, never the other's
    box.check()
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='none']\")"
                           "?.textContent.includes('not measured yet')")
    assert est.inner_text().strip().endswith("— with thinking on not measured yet; with "
                                             "thinking off it took 4.0 s an answer")
    assert "5.6 h" not in est.inner_text()
    no_sideways(page)
    steady_shot(page.locator("[data-dialog='test'] .dlg"), SCREENS / f"test-on-{width}.png")
    assert page.errors == []
