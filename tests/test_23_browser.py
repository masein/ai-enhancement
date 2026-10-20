"""23 in a real browser, at 1280 and 400 px, light and dark: two builds,
question by question — the picker and the swap, the chart and the table
changing with the pair, a count opening its questions and an answer, the
link with a= and b= opening the same pair, the same build twice refused,
nothing scrolling sideways. The world is tests/test_23_paired.py's (invented
questions and answers), every pair worked out by the stats worker first."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_21d_browser import no_sideways
from test_23_paired import EARLY, ORN, Q4, S4, world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase23"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import paired, report_stats
    world()
    paired._tables.clear()
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def open_pair(page, live, hash_: str = "", width: int = 1280, scheme: str = "light") -> None:
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=paired" + (("&" + hash_) if hash_ else ""))
    page.wait_for_selector("[data-pd-table]", timeout=30000)


def picked(page, which: str) -> str:
    return page.locator(f"[data-pd-pill='{which}']").get_attribute("data-value")


def choose(page, which: str, build: str) -> None:
    """a build from a pill's menu"""
    page.locator(f"[data-pd-pill='{which}']").click()
    page.locator(f"#pop-pd-{which} [data-pd-choice='{build}']").click()


def diffs(page) -> dict[str, str]:
    return dict(page.eval_on_selector_all(
        "[data-pd-diff]", "ts => ts.map(t => [t.dataset.pdDiff, t.textContent])"))


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_pair_its_chart_and_table(live, page, width, scheme):
    open_pair(page, live, width=width, scheme=scheme)
    assert picked(page, "a") == S4 and picked(page, "b") == Q4
    assert page.locator("select").count() == 0                  # pills, as Models' pickers
    page.locator("[data-pd-pill='a']").click()
    opts = page.eval_on_selector_all("#pop-pd-a [data-pd-choice]",
                                     "os => os.map(o => [o.dataset.pdChoice, o.disabled])")
    assert {x for x, _ in opts} == {S4, Q4, EARLY, ORN}         # those with answers only
    assert dict(opts)[Q4] is True and dict(opts)[EARLY] is False   # B's isn't offered to A
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"picker-{width}-{scheme}.png"),
                    clip={"x": 0, "y": 0, "width": width, "height": 520})
    page.keyboard.press("Escape")
    s = page.locator("[data-pd-sentence]").inner_text()
    assert s.startswith(f"{S4} is higher than {Q4} on ") and "can't be told apart on" in s
    assert page.locator("[data-pd-settings]").count() == 0     # the same runtime
    # a bar a benchmark compared, labelled; the two sides named under the axis
    bars = page.locator("[data-pd-bar]")
    assert bars.count() == 5                                    # HLE and MATH: no answers
    assert page.locator("[data-pd-sides]").inner_text().replace("\n", " ") == \
        f"← {Q4} higher {S4} higher →"
    # the table: MMLU-Pro 9 against 1 of 50, A higher
    assert page.locator("[data-pd-verdict='mmlupro']").inner_text() == "A higher"
    assert diffs(page)["mmlupro"].startswith("+16.0 (")
    assert page.locator("[data-pd-count='mmlupro:a']").inner_text() == "9"
    assert page.locator("[data-pd-differ='mmlupro']").inner_text() == "20.0%"
    assert page.locator("[data-pd-row='hle']").inner_text().count("no scored answers") == 1
    line = page.locator("[data-pd-self='gpqa']").inner_text()
    assert line.startswith(f"Each build also disagrees with itself: between two of its own "
                           f"runs, {S4} gives a different result on ") and \
        "% of GPQA Diamond questions, " in line and ". The two builds disagree on " in line
    # at 400 px each label sits above its bar, the bar the width of the card
    bar = page.locator("[data-pd-bar='gpqa'] + svg").bounding_box()
    assert bar["width"] > (200 if width == 400 else 600), bar
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(SCREENS / f"paired-{width}-{scheme}.png"), full_page=True)


def test_the_swap_flips_the_numbers_without_asking_again(live, page):
    open_pair(page, live)
    before = diffs(page)
    asked: list[str] = []
    page.on("request", lambda r: asked.append(r.url) if "/api/reports/paired" in r.url else None)
    page.locator("[data-pd-swap]").click()
    page.wait_for_function("document.querySelector(\"[data-pd-pill='a']\").dataset.value"
                           f" === {Q4!r}")
    after = diffs(page)
    for k, v in before.items():
        assert after[k].split(" ")[0] == ("−" + v[1:] if v.startswith("+") else
                                          "+" + v[1:] if v.startswith("−") else v).split(" ")[0]
    assert page.locator("[data-pd-verdict='mmlupro']").inner_text() == "B higher"
    assert not asked                                            # flipped, never asked again
    assert "a=Q4" in page.evaluate("decodeURIComponent(location.hash)")


def test_another_pair_a_count_its_list_and_an_answer(live, page):
    open_pair(page, live)
    choose(page, "b", ORN)
    page.wait_for_function("document.querySelector('[data-pd-settings]') !== null")
    assert page.locator("[data-pd-settings]").inner_text() == (
        f"{S4} ran on the lda runtime; {ORN} on its own card's settings.")
    assert "b=Ornith" in page.evaluate("decodeURIComponent(location.hash)")
    # back to Q4, and MMLU-Pro's count of A higher opened
    choose(page, "b", Q4)
    page.wait_for_selector("[data-pd-count='mmlupro:a']")
    page.locator("[data-pd-count='mmlupro:a']").click()
    head = page.locator("[data-pd-list-head]")
    head.wait_for()
    words = head.inner_text()
    assert words.startswith("9 questions; ") and words.endswith(" in the hidden half.")
    items = page.locator("[data-pd-item]")
    listed = int(words.split("; ")[1].split(" ")[0])
    assert items.count() == listed
    first = items.first.get_attribute("data-pd-item")
    page.locator(f"[data-pd-read='A:{first}']").click()
    page.wait_for_selector("[data-pd-answer] [data-pd-reply]")
    assert page.locator("[data-pd-question]").first.inner_text().startswith(
        "Invented MMLU-Pro question")
    assert no_sideways(page) and page.errors == []


def test_the_link_opens_the_same_pair_and_the_same_build_twice_is_refused(live, page):
    open_pair(page, live, f"a={EARLY.replace(' ', '%20').replace(',', '%2C')}&b={Q4}")
    assert picked(page, "a") == EARLY
    assert page.locator("[data-pd-settings]").inner_text() == (
        f"{EARLY} ran on Qwen's card settings; {Q4} on the lda runtime.")
    # the same build twice — only a link can ask it — is refused
    page.goto(live["base"] + f"/#tab=reports&report=paired&a={Q4}&b={Q4}")
    page.wait_for_selector("[data-pd-same]")
    assert "Pick two different builds" in page.locator("[data-pd-same]").inner_text()
    assert page.locator("[data-pd-table]").count() == 0
    # the scorecard's link under Our runs
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    link = page.locator("[data-rp-paired='paired']")
    link.wait_for(timeout=30000)
    assert link.inner_text() == "Compare two of these question by question ▸"
    assert "report=paired&a=step8804&b=step4000" in link.get_attribute("href")   # 23b
    assert page.errors == []
