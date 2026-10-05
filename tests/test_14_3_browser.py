"""14.3 on the page: Mobile-MMLU-Pro in the Mobile tasks chip, "provisional
key" on its column, a labeller's row "labelled the key"; the model page's
score, its 9 categories and the portal's file; the question browser's
question, our key and how the labellers decided it, credited to the authors;
Test a model's multiple-choice part; AI models' card for the key — its
labellers, the dry run, Start, the counts and the paper's checks. At 1400 and
375 px. Invented rows, picks written in: nothing runs, nothing calls
OpenRouter (Start is a route stub)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

import mobile_mmlu as mmp
from conftest import set_name
from test_14_3_mobile_mmlu import FIXTURE, RIGHT, a_key, fake_pin, pin_fixture, sit_picks

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase14_3"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
NEW = "fx/chance-160m"                      # no Mobile-MMLU picks
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """the invented rows on this server and a key built from them; good-750m
    misses one kept question, skewed-360m — set as the third labeller —
    "labelled the key" """
    import service.app as appmod
    from service import config, db
    with pytest.MonkeyPatch.context() as mp:
        d = Path(config.MMP_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
        pin_fixture(mp)
        a_key()
        tree = live["tree"]
        for mid, picks in ((GOOD, dict(RIGHT, inv00001="C")), (SKEWED, RIGHT)):
            md = tree["models"][mid]["dir"]
            sit_picks(md, picks)
            mmp.collect(md)
        db.ai_set("labeller:third", fake_pin(SKEWED), "masein")
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
        db.ai_set("labeller:third", None, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(part, name, screens=None):
    (screens or SCREENS).mkdir(parents=True, exist_ok=True)
    steady_shot(part, (screens or SCREENS) / name)


def steady_shot(part, path):
    """a part's screenshot. The page renders again when its 5-second poll
    lands, and a tall part (AI models' key card at 375 px) can take longer to
    shoot than is left before it: a part replaced mid-shot is taken again,
    straight after the render that replaced it"""
    from playwright.sync_api import Error
    for attempt in range(4):
        try:
            part.screenshot(path=path)
            return
        except Error as e:
            if "not attached" not in str(e) or attempt == 3:
                raise


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


@pytest.mark.parametrize("width", WIDTHS)
def test_the_mobile_chip_has_it_provisional_and_a_labeller_unranked(live, page, width):
    go(page, live, "tab=models&view=mobile&group=mmlu", "th[data-col='mobile_mmlu_pro']", width)
    th = page.locator("th[data-col='mobile_mmlu_pro']")
    assert th.locator(".hname").text_content() == "Mobile-MMLU-Pro"
    assert th.locator("[data-mmp-provisional]").text_content() == "provisional key"
    good = page.locator(f"tr[data-lb-row='{GOOD}'] [data-mmp-cell='{GOOD}']")
    assert good.inner_text().startswith("90.0")              # 9 of the 10 kept
    assert json.loads(good.get_attribute("data-tip"))[-1] == ("on 10 of 10 kept questions · "
                                                              "provisional key · Internal use")
    lab = page.locator(f"tr[data-lb-row='{SKEWED}'] [data-mmp-labelled='{SKEWED}']")
    assert lab.inner_text() == "labelled the key"
    assert lab.get_attribute("title") == "100.0% on the key it labelled: never ranked"
    no_sideways(page)
    shot(page.locator("[data-lb-card]"), f"chip-{width}.png")
    assert page.errors == []


def test_its_tooltip_says_how_it_is_scored_on_whose_key_and_the_checks(live, page):
    go(page, live, "tab=models&view=mobile&group=mmlu", "th[data-col='mobile_mmlu_pro']")
    tip = json.loads(page.locator("th[data-col='mobile_mmlu_pro']").get_attribute("data-tip"))
    # 14.4.4: its badge beside its name, then its sentence
    assert tip[0] == ("Mobile-MMLU-Pro (MBZUAI) — accuracy on our answer key’s kept questions · "
                      "provisional key · Internal use")
    assert tip[1] == ("CC BY-ND 4.0. Its scores may be used. The questions and our answer key "
                      "must not be published.")
    assert tip[2].startswith("asked as MMLU is, 0-shot: the letters’ log-likelihood")
    assert "strong models of different makers agreed on" in tip[3]
    assert tip[4].startswith("our key ") and "10 of 12 questions kept" in tip[4]
    assert tip[5].startswith("Provisional key — Qwen2.5-3B-Instruct not run (paper 60.6)")
    assert any("CC BY-ND 4.0, used here and never published" in x for x in tip)
    assert tip[-1] == "never in any average · never a training target"
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_model_page_has_the_score_its_categories_and_the_portals_file(live, page, width):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    line = page.locator(f"[data-mmp-line='{GOOD}']")
    line.wait_for()
    assert line.locator("p").first.inner_text() == ("Mobile-MMLU-Pro Internal use 90.0% on 10 "
                                                    "kept questions · provisional key")
    line.locator(f"[data-mmp-cats='{GOOD}'] > summary").click()
    cats = line.locator("[data-mmp-cat]")
    names = cats.evaluate_all("xs => xs.map(x => x.dataset.mmpCat)")
    assert names == [c for c in mmp.CATEGORIES if c in names] and len(names) >= 8
    hs = line.locator("[data-mmp-cat='Health & Safety']")
    assert hs.inner_text().split() == ["Health", "&", "Safety", "0.0%", "n", "1"]
    dl = line.locator(f"[data-mmp-download='{GOOD}']")
    assert dl.get_attribute("href") == "api/mobile-mmlu/predictions?model=fx%2Fgood-750m"
    assert "never our key" in line.locator(f"[data-mmp-portal='{GOOD}']").inner_text()
    no_sideways(page)
    shot(line, f"model-line-{width}.png")
    go(page, live, "model=" + quote(SKEWED, safe=""), "[data-model-hero]", width)
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    sk = page.locator(f"[data-mmp-line='{SKEWED}']")
    sk.wait_for()
    assert "not ranked: it labelled the key" in sk.inner_text()
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_the_question_browser_shows_the_question_our_key_and_how_it_was_decided(live, page,
                                                                               width):
    go(page, live, f"tab=benchmarks&q=mobile_mmlu_pro&models={ids(GOOD, SKEWED)}", "[data-qx-q]",
       width)
    src = page.locator("[data-qx-source]").inner_text()
    assert "MBZUAI" in src and "CC BY-ND 4.0 — used here, never published" in src
    assert "scored on our own answer key" in src
    first = page.locator("[data-qx-q]").first
    qid = first.get_attribute("data-qx-q")
    key = RIGHT[qid]
    right = first.locator("li.qx-right")
    assert right.count() == 1 and right.get_attribute("data-qx-opt") == str("ABCD".index(key))
    assert first.locator(f"[data-qx-key='{qid}']").inner_text() == (
        f"Our key: kept by agreement (both chose {key})")
    good = "C" if qid == "inv00001" else key
    assert first.locator(f"[data-qx-res='{GOOD}'] .qx-verdict").inner_text().strip() == \
        f"· picked {good} · our key {key}"
    # the key is never offered as a file
    assert page.locator("a[href*='key'], a[download]").count() == 0
    no_sideways(page)
    shot(first, f"question-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_test_a_model_offers_the_multiple_choice_part(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    # 14.4: a model that hasn't sat it (one that has would have nothing to ask)
    go(page, live, "model=" + quote(NEW, safe=""), "[data-model-hero]", width)
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    page.wait_for_function("document.querySelector(\"[data-mab-part-est='mmlu']\")"
                           "?.textContent.includes('answers')")
    part = page.locator("[data-mab-part='mmlu']")
    assert part.inner_text().strip().startswith(
        "multiple choice — Mobile-MMLU-Pro Internal use · 12 answers")
    assert part.locator("input").is_enabled()
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] .dlg"), f"test-mmlu-part-{width}.png")
    assert page.errors == []


def test_without_the_file_the_part_says_why(live, page):
    def missing(route):
        r = route.fetch()
        body = r.json()
        body["mmp"]["missing"] = ("Mobile-MMLU-Pro isn't on this server: fetch it with the data "
                                  "step (scripts/fetch_data.py)")
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", missing)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]")
    page.locator("[data-test-model]").click()
    page.locator("[data-dialog='test'] [data-select='suite']").click()
    page.locator("[role='option'][data-value='mobile']").click()
    box = page.locator("[data-mab-part='mmlu'] input")
    box.wait_for()
    assert box.is_disabled()
    assert page.locator("[data-mab-part-est='mmlu']").inner_text().strip() == (
        "· Mobile-MMLU-Pro isn't on this server: fetch it with the data step "
        "(scripts/fetch_data.py)")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_ai_models_has_the_keys_card(live, page, width):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])", width)
    card = page.locator("[data-mmp-key]")
    labs = card.locator("[data-mmp-labeller]")
    assert labs.evaluate_all("xs => xs.map(x => x.dataset.mmpLabeller)") == ["first", "second",
                                                                             "third"]
    assert card.locator("[data-mmp-labeller-now='first']").inner_text() == "GPT-6 Sol"
    assert card.locator("[data-mmp-labeller-now='second']").inner_text() == "Gemini 3.1 Pro"
    # a labeller scored on this set is refused, and says why
    assert "has a Mobile-MMLU score on the board" in card.locator(
        "[data-mmp-refused='third']").inner_text()
    est = card.locator("[data-mmp-est='first']").inner_text().split("\t")
    assert est[0].strip() == "GPT-6 Sol" and est[1].strip() == "12"
    assert card.locator("[data-mmp-counts]").inner_text().startswith("10 of 12 kept · ")
    assert card.locator("[data-mmp-checks]").get_attribute("data-mmp-checks") == "provisional"
    assert card.locator("[data-mmp-check]").count() == 3
    start_btn = card.locator("[data-mmp-start]")
    assert start_btn.is_disabled()                       # the third labeller can't label it
    assert "has a Mobile-MMLU score" in card.locator("[data-mmp-why]").inner_text()
    no_sideways(page)
    shot(card, f"key-card-{width}.png")
    assert page.errors == []


def test_start_sends_the_rest_once_the_labellers_can_label(live, page):
    from service import db
    import service.app as appmod
    db.ai_set("labeller:third", fake_pin("anthropic/claude-sonnet-5.5"), "masein")
    try:
        posted = []

        def start(route):
            posted.append(json.loads(route.request.post_data))
            route.fulfill(json={"sent": {"first": {"n": 12}, "second": {"n": 12}},
                                "page": json.loads(route.fetch(method="GET", url=live["base"]
                                                   + "/api/mobile-mmlu/key").text())})
        page.route("**/api/mobile-mmlu/key/start", start)
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=home")
        set_name(page, "masein")
        go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])")
        btn = page.locator("[data-mmp-start]")
        # the key has labels — 16c: and the button says what Carry on sends, and its cost
        assert btn.is_enabled() and btn.inner_text().startswith("Carry on: about $")
        assert btn.inner_text().endswith(" · Pro only")
        btn.click()
        page.wait_for_function("document.querySelector('[data-toast]')")
        assert posted == [{"by": "masein"}]
        assert "24 questions sent" in page.locator("[data-toast]").last.inner_text()
        assert page.errors == []
    finally:
        db.ai_set("labeller:third", fake_pin(SKEWED), "masein")
        appmod._cache.update(key=None, payload=None, at=0.0)
