"""16c, part 1, on the page: a Results row opens the model's Answers tab on
that benchmark — the address says which, Back returns to Scores, the keyboard
works — its answers lowest score first, each with its own score, the long
article folded, a filter, a search, 50 a page, and a second model under the
first. The Results table names each row in plain words. At 1400 and 375 px.
Answers written in; nothing runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

import mobileaibench as mab
from test_12n2 import sit_gpqa
from test_14_1_mab_text import sit

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16c_answers"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
WIDTHS = [1400, 375]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """CNN/DailyMail answered by both: the good one closer to the reference"""
    import service.app as appmod
    tree = live["tree"]
    for mid, words in ((GOOD, 60), (SKEWED, 8)):
        d = tree["models"][mid]["dir"]
        sit(d, mab.CNNDM, answer=lambda q, w=words: " ".join(q["answer"].split()[:w]) or "none")
        mab.write(d, mab.mark(d))
    sit_gpqa(tree["models"][GOOD]["dir"])
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def results(page, live, width=1400):
    go(page, live, "model=" + quote(GOOD, safe=""), "[data-model-hero]", width)
    if page.locator("[data-mtab='scores'][aria-selected='false']").count():
        page.locator("[data-mtab='scores']").click()
    if page.locator("[data-kind-block='standard']:not([open])").count():
        page.locator("[data-kind-block='standard'] > summary").first.click()
    page.locator("[data-result-row='mab_cnndm']").wait_for()


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def shot(part, name, height=1100):
    """the top of a part: its header, its tools and the first questions"""
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.scroll_into_view_if_needed()
    box = part.bounding_box()
    page = part.page
    y = box["y"] + page.evaluate("scrollY")
    page.screenshot(path=SCREENS / name, full_page=True,
                    clip={"x": box["x"], "y": y, "width": box["width"],
                          "height": min(box["height"], height)})


@pytest.mark.parametrize("width", WIDTHS)
def test_a_results_row_opens_its_answers_lowest_first_and_back_returns(live, page, width):
    results(page, live, width)
    row = page.locator("[data-result-row='mab_cnndm']")
    name = row.locator("[data-result-name='mab_cnndm']")
    assert name.inner_text() == "CNN/DailyMail · ROUGE-L"
    assert name.get_attribute("title").startswith("mab_cnndm")
    assert row.evaluate("r => getComputedStyle(r).cursor") == "pointer"
    row.click()
    view = page.locator("[data-bench-answers='mab_cnndm']")
    view.wait_for()
    assert "answers=mab_cnndm" in page.url
    assert page.locator("[data-mtab='answers']").get_attribute("aria-selected") == "true"
    page.wait_for_selector("[data-answers-q]")
    # what can't be read, once at the top
    counts = page.locator("[data-answers-counts]")
    listed, other = map(int, counts.get_attribute("data-answers-counts").split("|"))
    assert other > 0 and counts.inner_text() == (
        f"{listed:,} of {listed + other:,} can be read here. The other {other:,} are held back, "
        "so they can never reach training data or a question writer.")
    # lowest ROUGE-L first, each question with its own
    scores = [float(x) for x in page.locator(f"[data-answers-res='{GOOD}']").evaluate_all(
        "xs => xs.map(x => x.dataset.answersScore)")]
    assert len(scores) == 50 and scores == sorted(scores)
    first = page.locator("[data-answers-q]").first
    assert first.locator(f"[data-answers-score-words='{GOOD}']").inner_text().startswith(
        "ROUGE-L ")
    # the article folded, with its length
    ctx = first.locator("[data-answers-context]")
    assert ctx.evaluate("e => e.tagName") == "DETAILS"
    assert ctx.locator("summary").inner_text().startswith("Show the article ▸ ")
    assert first.locator("[data-answers-ref]").inner_text().startswith("Reference: ")
    # no Wrong or Right for a summary: overlap with one reference is no verdict
    assert page.locator("[data-answers-filter]").count() == 0
    no_sideways(page)
    shot(page.locator("[data-model-answers]"), f"cnndm-{width}.png")
    # Back: Scores, where it was
    page.go_back()
    page.wait_for_selector("[data-result-row='mab_cnndm']")
    assert "answers=" not in page.url
    assert page.locator("[data-mtab='scores']").get_attribute("aria-selected") == "true"
    assert page.errors == []


def test_the_address_opens_it_and_a_second_model_sits_under_the_first(live, page):
    go(page, live, "model=" + quote(GOOD, safe="") + "&answers=mab_cnndm",
       "[data-bench-answers='mab_cnndm'] [data-answers-q]")
    page.locator("[data-answers-vs]").click()
    page.locator(f"[role='option'][data-value='{SKEWED}']").click()
    page.wait_for_function(f"document.querySelectorAll(\"[data-answers-res='{SKEWED}']\")"
                           ".length === 50")
    assert "vs=" in page.url and quote(SKEWED, safe="") in page.url
    q = page.locator("[data-answers-q]").first
    names = q.locator("[data-answers-res] .qx-m").all_inner_texts()
    assert len(names) == 2 and names[0] != names[1]
    # a reload keeps both
    page.reload()
    page.wait_for_function(f"document.querySelectorAll(\"[data-answers-res='{SKEWED}']\")"
                           ".length === 50")
    shot(page.locator("[data-bench-answers]"), "cnndm-compare-1400.png")
    assert page.errors == []


def test_a_row_from_the_keyboard_the_wrong_ones_and_gpqa_says_why(live, page):
    results(page, live)
    page.locator("[data-result-row='mmlu']").focus()
    page.keyboard.press("Enter")
    page.wait_for_selector("[data-bench-answers='mmlu'] [data-answers-q]")
    page.locator("[data-answers-filter-pick='wrong']").click()
    page.wait_for_selector("[data-answers-filter='wrong']")
    page.wait_for_function("document.querySelector('[data-answers-q]')")
    marks = page.locator(f"[data-answers-res='{GOOD}'] .qx-no, [data-answers-res='{GOOD}'] "
                         ".qx-ok").all_inner_texts()
    assert marks and all(x.strip() == "✗" for x in marks)
    assert page.locator(f"[data-answers-score-words='{GOOD}']").first.inner_text() == "wrong"
    # GPQA: the row opens, and says why
    results(page, live)
    page.locator("[data-result-row='gpqa_diamond_cot_zeroshot']").click()
    why = page.locator("[data-answers-why='gpqa_diamond_cot_zeroshot']")
    why.wait_for()
    assert why.inner_text() == "GPQA Diamond’s questions are never shown, as its authors ask."
    assert page.errors == []
