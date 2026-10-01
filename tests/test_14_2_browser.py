"""14.2 on the page: Trust & safety's three new columns — Adversarial
Instruction, Privacy Leakage, Social Chemistry 101 ("agrees with crowd
judgements") — in Columns ▾, its default view as it was; their tooltips; the
model page's line in the Trust & safety block; Privacy Leakage's questions as
ids and verdicts only; Test a model's trust part. At 1400 and 375 px.
Invented Privacy Leakage rows, answers written in, the fake judge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import pytest

import mobileaibench as mab
from conftest import set_name, show_all_columns
from test_14_1_mab_text import sit
from test_14_2_mab_trust import FIXTURE

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_2"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live, tmp_path_factory):
    """the invented Privacy Leakage on this server; good-750m's replies marked
    by the fake judge, skewed-360m's awaiting it"""
    import judge as _judge
    import service.app as appmod
    from service import config
    d = tmp_path_factory.mktemp("private")
    (d / "privacy_leakage.csv").write_bytes(FIXTURE.read_bytes())
    real = mab.manifest

    def pinned():
        m = real()
        raw = FIXTURE.read_bytes()
        m["files"]["privacy_leakage"].update(sha256=hashlib.sha256(raw).hexdigest(),
                                             bytes=len(raw), n=6)
        return m
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MAB_PRIVATE_DIR", str(d))
        mp.setattr(config, "MAB_PRIVATE_DIR", d)
        mp.setattr(mab, "manifest", pinned)
        tree = live["tree"]
        good, skewed = tree["models"][GOOD]["dir"], tree["models"][SKEWED]["dir"]
        for t in mab.TRUST:
            sit(good, t, docs=mab.load(t), answer=lambda q: f"{q.get('answer', '')} — I won't "
                "share anyone's address" if not q.get("original") else "ANSWER")
            sit(skewed, t, docs=mab.load(t), answer=lambda q: "Sure: someone@example-corp.test")
        mp.setattr(_judge, "is_stub", lambda: True)
        mab.start_judge(good)
        mp.setattr(_judge, "is_stub", lambda: False)
        mp.setattr(_judge, "blocked", lambda: "no judge is configured on this server")
        mab.start_judge(skewed)
        mp.undo()
        mp.setenv("MAB_PRIVATE_DIR", str(d))
        mp.setattr(config, "MAB_PRIVATE_DIR", d)
        mp.setattr(mab, "manifest", pinned)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def cols(page):
    return page.locator("[data-lb-table] thead th[data-col]").evaluate_all(
        "xs => xs.map(x => x.dataset.col)")


@pytest.mark.parametrize("width", WIDTHS)
def test_trust_and_safetys_view_is_as_it_was_and_columns_offers_the_three(live, page, width):
    go(page, live, "tab=models&chip=trust", "[data-lb-table]", width)
    assert not [c for c in cols(page) if c.startswith("mab_")]           # its default view
    show_all_columns(page)
    page.wait_for_selector("th[data-col='mab_socchem']")
    assert [c for c in cols(page) if c.startswith("mab_")] == ["mab_adv", "mab_privacy",
                                                               "mab_socchem"]
    # the marked shares, and what waits for the judge
    assert page.locator(f"tr[data-lb-row='{GOOD}'] td[data-watch$='|mab_privacy']").inner_text() \
        .startswith("100.0")
    wait = page.locator(f"tr[data-lb-row='{SKEWED}'] [data-mab-awaiting-task='mab_socchem']")
    assert wait.inner_text() == "awaiting judge"
    assert wait.get_attribute("title").startswith("500 of 500 replies wait for the judge")
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"trust-columns-{width}.png")
    assert page.errors == []


def test_social_chemistrys_label_and_tooltip(live, page):
    go(page, live, "tab=models&chip=trust", "[data-lb-table]")
    show_all_columns(page)
    th = page.locator("th[data-col='mab_socchem']")
    th.wait_for()
    assert th.locator(".hname").text_content() == "Agrees with crowd"
    tip = json.loads(th.get_attribute("data-tip"))
    assert tip[0] == "Social Chemistry 101 (MobileAIBench) — agrees with crowd judgements"
    assert tip[1] == ("contested everyday moral judgements: agreement with the majority label, "
                      "not right or wrong.")
    assert any("CC BY-SA 4.0" in x for x in tip)
    priv = json.loads(page.locator("th[data-col='mab_privacy']").get_attribute("data-tip"))
    assert priv[0] == ("Privacy Leakage (MobileAIBench) — share of replies that don’t reveal the "
                       "address")
    assert "higher is better" in priv[1] and "ids and verdicts only" in priv[1]
    adv = json.loads(page.locator("th[data-col='mab_adv']").get_attribute("data-tip"))
    assert adv[0] == "Adversarial Instruction (MobileAIBench) — share judged correct"
    assert "embedding similarity: another measure" in adv[1]
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_model_pages_trust_block_has_one_more_line(live, page, width):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    line = page.locator(f"[data-mab-trust-line='{GOOD}']")
    line.wait_for()
    text = line.inner_text()
    assert text.startswith("MobileAIBench: judged correct on reworded instructions ")
    assert "keeps an address private 100%" in text and "agrees with crowd judgements" in text
    no_sideways(page)
    shot(line, f"trust-line-{width}.png")
    go(page, live, "model=" + quote(SKEWED, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    w = page.locator(f"[data-mab-trust-line='{SKEWED}']")
    w.wait_for()
    assert "Privacy: 6 of 6 awaiting judge" in w.inner_text()
    assert w.locator(f"[data-mab-judge='{SKEWED}']").inner_text() == "Judge now"
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_privacy_leakages_questions_are_ids_and_verdicts_only(live, page, width):
    go(page, live, f"tab=benchmarks&q=mab_privacy&models={ids(GOOD, SKEWED)}", "[data-qx-q]",
       width)
    body = page.locator("body").inner_text()
    for q in mab.load(mab.PRIVACY) or []:
        assert q["prompt"] not in body
    import csv
    for row in csv.DictReader(FIXTURE.read_text().splitlines()):
        assert row["prompt"] not in body                     # never a person's name
    assert "@" not in body and "example-corp" not in body
    first = page.locator("[data-qx-q]").first
    assert first.inner_text().startswith("privacy-")
    assert first.locator(f"[data-qx-res='{GOOD}'] .qx-verdict").inner_text().strip() == \
        "· kept it private"
    assert first.locator(f"[data-qx-res='{SKEWED}'] .qx-verdict").inner_text().strip() == \
        "· awaiting judge"
    no_sideways(page)
    shot(page.locator("[data-qx-q]").first, f"privacy-question-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_offers_the_trust_part(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='trust']\")"
                           "?.textContent.includes('answers')")
    assert page.locator("[data-mab-part-est='trust']").inner_text().strip().startswith(
        "· 1,250 answers")
    jl = page.locator("[data-mab-part='trust'] [data-mab-judge-est]").inner_text()
    assert jl.startswith("The judge (") and "1200 judgements · about " in jl
    assert page.locator("[data-mab-part='trust'] input").is_enabled()
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] .dlg"), f"test-trust-part-{width}.png")
    assert page.errors == []


def test_without_privacy_leakage_the_trust_part_says_why(live, page):
    def missing(route):
        r = route.fetch()
        body = r.json()
        body["mab"]["privacyMissing"] = ("Privacy Leakage isn't on this server: fetch it with the "
                                        "data step (scripts/fetch_data.py)")
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", missing)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    box = page.locator("[data-mab-part='trust'] input")
    box.wait_for()
    assert box.is_disabled()
    assert page.locator("[data-mab-part-est='trust']").inner_text().strip() == (
        "· Privacy Leakage isn't on this server: fetch it with the data step "
        "(scripts/fetch_data.py)")
    assert page.errors == []
