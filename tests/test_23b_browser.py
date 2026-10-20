"""23b in a real browser, at 1280 and 400 px, light and dark: a model page's
block "Beyond the score" opening and drawing its four parts, and the
scorecard's new section, its "Score against thinking" chart and its loop
column; nothing scrolling sideways. The world is 21c's scorecard, with
invented graded answers for step4000 and Q4 on GPQA, OTIS, HLE and SimpleQA,
worked out by the stats worker."""

from __future__ import annotations

import random
from pathlib import Path
from urllib.parse import quote

import pytest

from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_23b_beyond import GPQA, HLE, LOOP, OTIS, SQA, hle_rows, write

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase23b"
S4, Q4 = "step4000", "Q4 (UD-Q4_K_XL)"
S4_ROW = "served/step4000-lda-2610 · thinking"


def runs(n: int, k: int, always: int, never: int) -> list[dict]:
    """n questions of k runs: `always` right every time, `never` never, the
    rest right on about half"""
    out = []
    for q in range(n):
        oks = [True] * k if q < always else [False] * k if q < always + never else \
            [e % 2 == 0 for e in range(k)]
        out.append({"runs": [{"epoch": e, "ok": x, "ran_out": False} for e, x in enumerate(oks)],
                    "texts": ["<think>\nworking\n</think>\n\nANSWER: A"] * k})
    return out


def sqa(a: int, b: int, c: int, out: int) -> list[dict]:
    rows = []
    for letter, n in (("A", a), ("B", b), ("C", c)):
        rows += [{"runs": [{"epoch": 0, "ok": letter == "A", "ran_out": False,
                            "grade": {"grade": letter}}], "texts": ["It is X."]}] * n
    rows += [{"runs": [{"epoch": 0, "ok": False, "ran_out": True}],
              "texts": ["<think>" + LOOP]}] * out
    return rows


def answers() -> None:
    rng = random.Random(7)
    for label, (alw, nev), (a, b, c, o) in ((S4, (18, 3), (14, 70, 12, 4)),
                                            (Q4, (15, 6), (12, 60, 26, 2))):
        write(label, GPQA, runs(30, 4, alw, nev))
        write(label, OTIS, runs(9, 8, alw // 4, nev // 2))
        write(label, HLE, hle_rows(260, rng))
        write(label, SQA, sqa(a, b, c, o))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import report_stats
    scorecard()
    answers()
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_model_pages_block_opens_and_draws_its_four_parts(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(S4_ROW, safe=""))
    blk = page.locator("[data-kind-block='beyond']")
    blk.wait_for(timeout=30000)
    head = page.locator("[data-by-head]")
    page.wait_for_function("document.querySelector('[data-by-head]').textContent"
                           ".startsWith('calibration error ')", timeout=30000)
    assert blk.get_attribute("open") is None                      # closed unless opened
    blk.locator("summary").click()
    page.wait_for_selector("[data-by-part='loops']")
    for k in ("reliability", "calibration", "simpleqa", "loops"):
        assert page.locator(f"[data-by-part='{k}']").count() == 1, k
    assert page.locator("[data-by-rel='gpqa_diamond_epoch']").inner_text() == (
        "Right every time on 60% of GPQA Diamond questions, never on 10%, and sometimes on 30% "
        "(4 runs each).")
    assert page.locator("[data-by-cal]").inner_text().startswith("Says it is ")
    assert "Calibration error " in page.locator("[data-by-cal]").inner_text()
    assert page.locator("[data-by-sqa]").inner_text() == (
        "Tries 84% of questions; right on 14.0% of all and 16.7% of the ones it tries; F 15.2.")
    assert page.locator("[data-by-bin]").count() == 10
    assert page.locator("[data-by-loop='simpleqa_epoch']").inner_text().count("%") == 2
    assert head.inner_text().startswith("calibration error ")
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    blk.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 70)")
    page.screenshot(path=str(SCREENS / f"model-block-{width}-{scheme}.png"), full_page=True)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_scorecards_section_chart_and_loop_column(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    sec = page.locator("[data-report-section='beyond']")
    sec.wait_for(timeout=30000)
    rows = page.eval_on_selector_all("[data-by-row]", "rs => rs.map(r => r.dataset.byRow)")
    assert rows[0] == "step8804" and S4 in rows and Q4 in rows
    assert page.locator(f"[data-by-cell='{S4}:gpqa']").inner_text() == "60%"
    assert page.locator(f"[data-by-cell='{S4}:gpqa']").get_attribute("title").startswith(
        "Right every time on 60% of GPQA Diamond questions")
    assert page.locator("[data-by-cell='step8804:gpqa']").inner_text() == "—"
    assert page.locator(f"[data-by-cell='{S4}:f']").inner_text() == "15.2"
    # Thinking tokens: the loop column, and the score against thinking
    page.wait_for_selector("[data-rp-tok-col='Ended in a loop']", timeout=30000)
    assert page.locator(f"[data-rp-tok-loop='{S4}|simpleqa_epoch']").inner_text().endswith("%")
    page.wait_for_selector("[data-by-think-chart]", timeout=30000)
    assert page.locator("[data-by-think]").count() >= 3
    assert page.locator("[data-by-dot]").count() >= 6
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    sec.screenshot(path=str(SCREENS / f"report-beyond-{width}-{scheme}.png"))
    page.locator("[data-by-think-chart]").screenshot(
        path=str(SCREENS / f"report-think-{width}-{scheme}.png"))
