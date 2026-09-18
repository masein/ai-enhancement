"""Playwright smoke against the frozen report built from the fixture.

The page is one file with the payload embedded, so file:// is the honest way
to load it — nothing is fetched in that mode. Every test collects console
errors and uncaught exceptions and fails on any. Screenshots land in
tests/_screens/ (gitignored; CI uploads them) so a PR can show the page in
light and dark at desktop and phone width.

    pytest -q -m dashboard        # needs: playwright install chromium
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pytest

pytestmark = pytest.mark.dashboard

# the tabs the FROZEN page has (the live one adds Training and Submit & Queue)
FROZEN_TABS = ["Overview", "Leaderboard", "Tasks", "Perplexity & Loss", "Evals"]
SCREENS = Path(__file__).resolve().parent / "_screens"


def model_link(mid: str) -> str:
    return "#model=" + quote(mid, safe="")           # what encodeURIComponent writes


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


class Surface:
    def __init__(self, page, url):
        self.page, self.url, self.errors = page, url, []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: self.errors.append(f"console.error: {m.text}")
                if m.type == "error" else None)

    def open(self, frag: str = ""):
        self.page.goto(self.url + frag)
        self.page.wait_for_selector("#view > *")
        return self.page

    def tab(self, label: str):
        self.page.get_by_role("tab", name=label, exact=True).click()
        self.page.wait_for_selector("#view > *")

    def selected_tab(self) -> str:
        return self.page.locator("#tabs button[aria-selected='true']").inner_text()

    def fits(self) -> bool:
        return self.page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth")


@pytest.fixture
def surface(browser, tree, request):
    width = getattr(request, "param", 1240)
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    s = Surface(ctx.new_page(), tree["report"].as_uri())
    yield s
    ctx.close()


def test_every_tab_renders_with_zero_console_errors(surface):
    pg = surface.open()
    assert surface.selected_tab() == "Overview"
    for label in FROZEN_TABS:
        surface.tab(label)
        assert surface.selected_tab() == label
        assert pg.locator("#view > *").count() > 0, label
        assert pg.evaluate("location.hash") == "#tab=" + {
            "Perplexity & Loss": "perplexity", "Evals": "runs"}.get(label, label.lower())
    assert surface.errors == []


def _mmlu_details(pg, card):
    return card.locator("details.dx", has=pg.locator(".dxname", has_text=re.compile(r"^mmlu[^_]", re.I))).first


def test_model_page_shows_the_diagnose_card(surface, diag):
    pg = surface.open(model_link("fx/skewed-360m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
    assert card.count() == 1
    assert "No per-item diagnosis on file" not in card.text_content()
    # the finding planted in this model is the one the card leads with
    lead = card.locator(".dxlead").first.text_content()
    assert "of its answers land on" in lead and "cannot score above" in lead
    assert "answer positions" in card.locator(".dxflag").all_text_contents()
    # the ceiling on the page is 1 - TVD from the diagnosis file, to the shown precision
    mmlu = _mmlu_details(pg, card)
    m = re.search(r"Ceiling for this answer distribution: ([\d.]+)%", mmlu.text_content())
    assert m, "no ceiling sentence for mmlu"
    tvd = diag["fx/skewed-360m"]["tasks"]["mmlu"]["answers"]["pick_skew"]
    assert abs(float(m.group(1)) - 100 * (1 - tvd)) < 0.15
    # the examples are labelled as the diagnosis half, and there are some
    assert mmlu.locator("details.dxex summary").text_content().endswith("(diagnosis half only)")
    assert mmlu.locator("details.dxex li").count() > 0
    assert surface.errors == []


def test_model_without_a_diagnosis_says_so(surface, tree):
    pg = surface.open(model_link(tree["nodiag"]))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
    assert card.count() == 1
    assert "No per-item diagnosis on file for this model" in card.inner_text()
    assert card.locator("details.dx").count() == 0
    assert surface.errors == []


def test_deep_link_and_back_forward(surface):
    pg = surface.open("#tab=leaderboard")
    assert surface.selected_tab() == "Leaderboard"
    surface.tab("Tasks")
    assert pg.evaluate("location.hash") == "#tab=tasks"
    pg.go_back()
    pg.wait_for_function("location.hash === '#tab=leaderboard'")
    assert surface.selected_tab() == "Leaderboard"
    assert pg.locator("#view > *").count() > 0
    pg.go_forward()
    pg.wait_for_function("location.hash === '#tab=tasks'")
    assert surface.selected_tab() == "Tasks"

    # into a model page from wherever the board links one, and Back out again
    surface.tab("Leaderboard")
    link = pg.locator('#view a[href^="#model="]').first
    href = link.get_attribute("href")
    link.click()
    pg.wait_for_selector(".backlink")
    assert pg.evaluate("location.hash") == href
    assert pg.locator("#tabs button[aria-selected='true']").count() == 0
    pg.go_back()
    pg.wait_for_function("location.hash === '#tab=leaderboard'")
    assert pg.locator(".backlink").count() == 0
    assert surface.selected_tab() == "Leaderboard"
    assert surface.errors == []


def test_unknown_model_link_falls_back_to_the_board(surface):
    surface.open(model_link("nobody/nothing"))
    assert surface.selected_tab() == "Overview"
    assert surface.errors == []


@pytest.mark.parametrize("surface", [430], indirect=True)
def test_no_horizontal_scroll_at_phone_width(surface):
    pg = surface.open()
    for label in FROZEN_TABS:
        surface.tab(label)
        assert surface.fits(), f"{label} overflows 430px"
    surface.open(model_link("fx/skewed-360m"))
    pg.locator("details.dx > summary").first.click()               # open a task's detail
    assert surface.fits(), "model page overflows 430px"
    assert surface.errors == []


def test_categories_first_subjects_on_expand(surface, diag):
    pg = surface.open(model_link("fx/good-750m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
    mmlu = _mmlu_details(pg, card)
    mmlu.locator("> summary").click()
    cats = mmlu.locator("details.dxcat")
    expected = diag["fx/good-750m"]["tasks"]["mmlu"]["categories"]
    assert cats.count() == len(expected)
    # weakest first; among the categories above the noise floor the planted
    # gap (econometrics) puts economics lowest — the greyed one-subject rows
    # may land anywhere, which is exactly why they are greyed
    scores = [float(x.rstrip("%")) for x in cats.locator("> summary > .num").all_text_contents()]
    assert scores == sorted(scores)
    solid = mmlu.locator("details.dxcat:not(.dim) .dxcname").all_text_contents()
    assert solid[0] == "economics"
    # the noise floor is visible: one-subject categories are greyed, two-subject ones are not
    dim = mmlu.locator("details.dxcat.dim .dxcname").all_text_contents()
    assert "economics" not in dim and "medicine & health" not in dim and len(dim) == 4
    assert "under 30, noise" in mmlu.locator("details.dxcat.dim").first.text_content()
    # subjects live under the category, not beside it, until you open one
    econ = mmlu.locator("details.dxcat[data-cat='economics']")
    assert econ.locator("table.dxsub").is_hidden()
    econ.locator("summary").click()
    rows = econ.locator("tbody tr td:first-child").all_text_contents()
    assert set(rows) == {"econometrics", "high_school_macroeconomics"}
    assert "Weakest groups first" not in mmlu.text_content()        # replaced, not doubled
    assert "Mapping gap" not in mmlu.text_content()
    assert surface.errors == []


def test_categories_under_a_score_at_chance_are_not_a_claim(surface):
    pg = surface.open(model_link("fx/chance-160m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
    mmlu = _mmlu_details(pg, card)
    assert "how the model guesses per category" in mmlu.text_content()


def test_permutation_control_sentence(surface):
    def perm_text(mid):
        pg = surface.open(model_link(mid))
        card = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
        p = card.locator(".dxperm")
        return p.text_content() if p.count() else None
    skewed = perm_text("fx/skewed-360m")
    assert "The format was hiding measurable knowledge" in skewed
    assert "mmlu — options as published" in skewed and "options rotated by item" in skewed
    assert re.search(r"\d+\.\d standard errors", skewed) and "±" in skewed
    chance = perm_text("fx/chance-160m")
    assert "The knowledge is not there to hide" in chance
    good = perm_text("fx/good-750m")
    assert "Both posings clear chance" in good
    assert perm_text("fx/below-135m-it") is None            # no control run: no section
    assert surface.errors == []


def test_leaderboard_by_category_view(surface, diag):
    pg = surface.open("#tab=leaderboard")
    pg.get_by_role("button", name="MMLU by category", exact=True).click()
    table = pg.locator("table.lbcats")
    heads = table.locator("thead th").all_text_contents()
    assert any(h.startswith("economics") for h in heads) and any(h.startswith("mmlu") for h in heads)
    assert table.locator("tbody tr").count() == len(diag)     # diagnosed models only
    assert "1 model without a diagnosis on file" in pg.locator("#view").text_content()
    assert table.locator("td.dim").count() > 0 and table.locator("td.num.best").count() > 0
    # the control column carries its warning in the task view
    pg.get_by_label("leaderboard columns").get_by_role("button", name="tasks", exact=True).click()
    th = pg.locator("table.lb thead th", has_text=re.compile(r"^mmlu_perm"))
    assert "CONTROL" in th.get_attribute("title")
    assert surface.errors == []


def test_judged_section_and_the_control_sentence(surface, tree):
    pg = surface.open(model_link("fx/skewed-360m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert card.count() == 1
    text = card.text_content()
    assert "Calibrated." in text and "Cohen's κ" in text
    assert "STUB grader" in text                                  # never mistaken for a judgement
    assert "Knew it, couldn't pick it" in text
    m = re.search(r"of the (\d+) control items this model got wrong as multiple choice, it answered (\d+)", text)
    assert m and int(m.group(2)) / int(m.group(1)) >= 0.5
    assert "By category (0–4)" in text and "Score against answer length" in text
    assert card.locator("table.jd").count() == 3
    surface.open(model_link("fx/chance-160m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert "Didn't know it either way" in card.text_content()
    surface.open(model_link(tree["nodiag"]))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert "Not judged" in card.text_content()
    assert surface.errors == []


def test_judged_columns_appear_once_calibrated(surface):
    pg = surface.open("#tab=leaderboard")
    heads = pg.locator("table.lb thead tr").first.locator("th").all_text_contents()
    judged = [h for h in heads if "κ" in h]
    assert len(judged) == 5 and any(h.startswith("Judged avg") for h in judged)
    assert not any(h.startswith("fr_") for h in heads)           # never as a task column
    row = pg.locator("table.lb tbody tr", has_text="good-750m").first
    assert "/4" in row.text_content()
    assert surface.errors == []


def test_what_the_training_taught(surface, tree):
    pg = surface.open(model_link("fx/good-750m-tuned-test"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="What the training taught"))
    assert card.count() == 1
    text = card.text_content()
    assert "before — good-750m" in text and "after — good-750m-tuned-test" in text
    assert "leaderboard half (never in the training data)" in text
    v = card.locator("[data-verdict]")
    assert v.get_attribute("data-verdict") == "test" and "warn" in v.get_attribute("class")
    assert "The training taught the test" in text and "Ratio of the two deltas" in text
    assert "By category — the half we never touched" in text
    assert card.locator("table.jd").nth(1).locator("tbody tr").count() >= 4
    # the category rows in Diagnose carry the same deltas
    det = pg.locator("details.dx", has=pg.locator(".dxname", has_text=re.compile(r"^mmlu[^_]"))).first
    det.locator("> summary").click()
    assert det.locator(".taintdelta").count() >= 4
    assert "vs parent: lb" in det.locator(".taintdelta").first.text_content()
    # the badge and the head sentence
    head = pg.locator("#view .card").first
    assert head.locator(".badge.taint").count() == 1
    assert "excluded from its official average" in head.text_content()

    surface.open(model_link("fx/good-750m-tuned-skill"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="What the training taught"))
    v = card.locator("[data-verdict]")
    assert v.get_attribute("data-verdict") == "skill" and "calm" in v.get_attribute("class")
    assert "The training taught the skill" in card.text_content()
    surface.open(model_link("fx/good-750m"))
    assert pg.locator(".card", has=pg.locator("h2", has_text="What the training taught")).count() == 0
    assert surface.errors == []


def test_screenshots_for_the_pr(surface):
    """Not an assertion beyond 'it rendered': the pictures a reviewer wants."""
    SCREENS.mkdir(exist_ok=True)
    pg = surface.page
    for scheme in ("light", "dark"):
        pg.emulate_media(color_scheme=scheme)
        for width in (1240, 430):
            pg.set_viewport_size({"width": width, "height": 900})
            surface.open("#tab=overview")
            pg.screenshot(path=SCREENS / f"overview-{scheme}-{width}.png", full_page=True)
            surface.open(model_link("fx/skewed-360m"))
            pg.locator("details.dx > summary").first.click()
            pg.locator("details.dxcat > summary").first.click()
            pg.screenshot(path=SCREENS / f"model-skewed-{scheme}-{width}.png", full_page=True)
            surface.open("#tab=leaderboard")
            pg.get_by_role("button", name="MMLU by category", exact=True).click()
            pg.screenshot(path=SCREENS / f"leaderboard-categories-{scheme}-{width}.png",
                          full_page=True)
            surface.open(model_link("fx/good-750m-tuned-test"))
            pg.locator(".card", has=pg.locator("h2", has_text="What the training taught")) \
              .screenshot(path=SCREENS / f"taught-the-test-{scheme}-{width}.png")
    assert len(list(SCREENS.glob("*.png"))) >= 16
    assert surface.errors == []
