"""Phase 10c on the page: 36 topics on every screen that lists them, and the
five findings of the phase-9 live check."""

from __future__ import annotations

import re
import socket

import pytest

from conftest import go_tab

pytestmark = pytest.mark.dashboard


def _score(text: str) -> float | None:
    m = re.match(r"\s*(\d+(?:\.\d+)?) / 4", text or "")
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 7. thirty-six topics
# ---------------------------------------------------------------------------

def test_the_loop_board_is_weakest_first_searchable_and_paged(live, page):
    page.goto(live["base"] + "/#tab=loop")
    page.wait_for_selector("table[data-loop-table] tr[data-loop-row]")
    rows = page.locator("table[data-loop-table] tr[data-loop-row]")
    assert rows.count() == 25                              # the shared pager, at 25
    assert page.locator("[data-pager='loop'] [data-page-range]").get_attribute(
        "data-page-range") == "1-25"
    # weakest first, for the model in "Results for"
    scores = [_score(t) for t in page.locator("[data-loop-score]").all_text_contents()]
    judged = [s for s in scores if s is not None]
    assert len(judged) >= 10 and judged == sorted(judged)
    assert scores[:len(judged)] == judged                  # the ones it has not sat come after
    # Arts, delivered empty, is in the fold and not a row
    assert "Arts" in page.locator("tr[data-empty-topics]").text_content()
    # a search narrows it
    page.get_by_label("find a topic").fill("law")
    page.wait_for_function("document.querySelectorAll('tr[data-loop-row]').length === 1")
    assert page.locator("tr[data-loop-row='law']").count() == 1
    assert "match" in page.locator("[data-loop-count]").text_content()
    page.get_by_label("find a topic").fill("")
    page.wait_for_function("document.querySelectorAll('tr[data-loop-row]').length === 25")
    assert page.errors == []


def test_the_topic_boxes_filter_tick_all_or_none_and_say_what_they_cost(live, page):
    page.goto(live["base"] + "/#topic=law")
    picker = page.locator("[data-picker='sit']")
    picker.wait_for()
    count = picker.locator("[data-pick-count]")
    law_items = int(re.search(r"about (\d+) answers", count.text_content()).group(1))
    assert count.text_content().startswith("1 of 36 topics · about ")
    assert law_items > 0
    # the minutes come from the last judged runs: none on this board ran on a GPU
    assert "no judged run yet to time it by" in count.text_content()
    picker.get_by_label("find a topic").fill("econ")
    page.wait_for_function("document.querySelectorAll(\"[data-picker='sit'] "
                           "input[type=checkbox]\").length === 1")
    picker.locator("[data-pick-all]").click()
    assert count.text_content().startswith("2 of 36 topics")
    picker.locator("[data-pick-none]").click()                 # none of what is shown
    assert count.text_content().startswith("1 of 36 topics")
    picker.get_by_label("find a topic").fill("")
    page.wait_for_function("document.querySelectorAll(\"[data-picker='sit'] "
                           "input[type=checkbox]\").length > 30")
    picker.locator("[data-pick-all]").click()
    assert count.text_content().startswith("36 of 36 topics")
    assert page.locator("[data-picker='sit'] input:checked").count() >= 36
    picker.locator("[data-pick-none]").click()
    assert count.text_content().startswith("0 of 36 topics")
    assert page.errors == []


def test_the_model_page_picks_a_judged_topic_from_a_searchable_select(live, page):
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    sel = page.locator("select[data-topic-switch]")
    sel.wait_for()
    assert sel.locator("option").count() >= 30
    labels = sel.locator("option").all_text_contents()
    scores = [_score(t.split(" — ")[-1]) for t in labels]
    assert scores == sorted(scores)                           # weakest first, with the score
    page.get_by_label("find a judged topic").fill("law")
    page.wait_for_function("document.querySelector('select[data-topic-switch]').options.length <= 3")
    sel.select_option("exam_law")
    page.wait_for_selector("table[data-criteria-table='Law']")
    assert page.errors == []


def test_the_leaderboard_shows_the_judged_average_and_hides_the_topic_columns(live, page):
    page.goto(live["base"] + "/#tab=leaderboard")
    page.wait_for_selector("table.lb")
    heads = page.locator("table.lb thead tr").first.locator("th").all_text_contents()
    assert any(h.startswith("Judged avg") for h in heads)       # shown by default
    assert not any(h.startswith("MMLU ") for h in heads)        # categories: hidden
    assert not any("κ" in h and not h.startswith("Judged avg") for h in heads)   # topics: hidden
    avg = page.locator("table.lb tbody tr[data-lb-row='fx/good-750m'] [data-judged-avg]")
    assert avg.count() == 1 and "/4" in avg.text_content()
    menu = page.locator("[data-columns-menu]")
    menu.locator("summary").click()
    assert menu.locator("[data-column-group='judged']").count() == 1
    menu.locator("[data-column-group-all='cats']").click()
    page.wait_for_function("[...document.querySelectorAll('table.lb thead th')]"
                           ".some(th => th.textContent.startsWith('MMLU Economics'))")
    menu = page.locator("[data-columns-menu]")
    if menu.get_attribute("open") is None:
        menu.locator("summary").click()
    menu.locator("[data-column-group-none='cats']").click()
    page.wait_for_function("![...document.querySelectorAll('table.lb thead th')]"
                           ".some(th => th.textContent.startsWith('MMLU '))")
    assert page.errors == []


def test_the_rubrics_table_is_searchable_and_paged(live, page):
    page.goto(live["base"] + "/#tab=exam")
    panel = page.locator("[data-panel='rubrics']")
    panel.locator("tr[data-rubric-row]").first.wait_for()
    assert panel.locator("tr[data-rubric-row]").count() == 25
    assert panel.locator("[data-pager='rubrics']").count() == 1
    panel.get_by_label("find a rubric").fill("medicine")
    page.wait_for_function("document.querySelectorAll(\"[data-panel='rubrics'] "
                           "tr[data-rubric-row]\").length === 1")
    bank = panel.locator("tr[data-rubric-row='Medicine & Clinical Health'] [data-bank]")
    assert re.fullmatch(r"\d+ — \d+ report / \d+ diagnose.*", bank.text_content())
    assert page.errors == []


# ---------------------------------------------------------------------------
# 8. the phase-9 live check
# ---------------------------------------------------------------------------

@pytest.fixture
def judge_down(live, monkeypatch):
    from service import app, config
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
    monkeypatch.setattr(config, "JUDGE_MODEL", "chat")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", url)
    app._JUDGE_HEALTH.update(at=0.0, value=None)
    yield url
    app._JUDGE_HEALTH.update(at=0.0, value=None)


def test_a_judge_that_is_down_is_on_the_board_and_stops_queue_this_run(live, page, judge_down):
    page.goto(live["base"] + "/#topic=law")
    page.wait_for_selector("[data-topic-page='law'] [data-judge-offline]")
    btn = page.locator("button[data-sit]")
    assert btn.is_disabled()
    why = page.locator("[data-why='sit']").text_content()
    assert why.startswith("The grading model isn't answering at " + judge_down)
    # the header's live dot covers the judge
    stamp = page.locator("[data-stamp]")
    page.wait_for_function("document.querySelector('[data-stamp]').dataset.fresh === "
                           "'judge-offline'")
    assert "judge offline" in stamp.text_content()
    page.goto(live["base"] + "/#tab=loop")
    page.wait_for_selector("[data-judge-offline-why]")
    assert judge_down in page.locator("[data-judge-offline-why]").text_content()
    assert page.errors == []


def test_a_row_whose_grading_failed_reads_plainly_and_retries_the_grading(live, page,
                                                                         monkeypatch):
    from service import app, config, db
    monkeypatch.setattr(config, "JUDGE_MODEL", "stub")
    app._JUDGE_HEALTH.update(at=0.0, value=None)
    sid = db.add("fx/good-750m", "auto", "judged", "omar", "")
    db.update(sid, status="failed", progress="failed on: judge", tasks='["exam_law"]',
              error="judge: LocalUnreachable: nothing is answering at "
                    "http://host.docker.internal:8000/v1 — ssh -L 8000:localhost:8000")
    page.goto(live["base"] + "/#tab=queue")
    row = page.locator(f"tr[data-queue-row='{sid}']")
    row.wait_for()
    plain = row.locator("[data-judge-failed] .down").text_content()
    assert plain == ("The grading model isn't running. Start it, then Retry grading — the "
                     "answers are kept.")
    details = row.locator("[data-judge-failed] details")
    assert not details.locator(".se").is_visible()           # the raw text is behind details
    details.locator("summary").click()
    assert "host.docker.internal" in details.text_content()
    row.locator(f"[data-row-regrade='{sid}']").click()
    page.wait_for_selector("[data-toast='regrade']")
    assert f"Grading #{sid}'s answers again" in page.locator("[data-toast='regrade']").text_content()
    new = max(r["id"] for r in db.recent(50))
    assert new > sid and db.get(new)["tasks"] == '["exam_law"]'
    page.wait_for_selector(f"tr.landed[data-queue-row='{new}']")
    assert page.errors == []


def test_a_toast_with_a_link_stays_eight_seconds_and_while_it_is_pointed_at(live, page):
    page.goto(live["base"] + "/#tab=overview")
    page.wait_for_selector("#tabs #moreBtn")
    page.evaluate("toast('plain', { key: 'plain' })")
    page.evaluate("toast('with a link', { key: 'linked', go: () => {}, link: 'see it' })")
    page.wait_for_timeout(5000)
    assert page.locator("[data-toast='plain']").count() == 0          # four seconds
    linked = page.locator("[data-toast='linked']")
    assert linked.count() == 1                                        # eight
    linked.hover()
    page.wait_for_timeout(5000)                                       # past eight, pointed at
    assert linked.count() == 1
    page.mouse.move(5, 5)
    page.wait_for_timeout(3000)
    assert linked.count() == 1                                        # the clock starts again
    page.wait_for_selector("[data-toast='linked']", state="detached", timeout=8000)
    assert page.errors == []


def _fill_queue(n=30):
    from service import db
    ids = []
    for i in range(n):
        s = db.add(f"org/filler-{i}", "auto", "quick", "omar", "")
        db.update(s, status="done", progress="all 4 tasks done")
        ids.append(s)
    return ids


def test_the_queue_starts_on_page_one_and_marks_a_row_you_queued(live, page):
    from service import db
    _fill_queue()
    mine = db.add("org/mine", "auto", "quick", "me", "")
    page.goto(live["base"] + "/#tab=queue")
    pager = page.locator("[data-pager='queue']")
    pager.locator("[data-page='2']").click()
    page.wait_for_function("document.querySelector(\"[data-pager='queue'] [data-page-range]\")"
                           ".dataset.pageRange.startsWith('26-')")
    go_tab(page, "Loop")
    go_tab(page, "Queue")                                   # coming back: page 1
    page.wait_for_function("document.querySelector(\"[data-pager='queue'] [data-page-range]\")"
                           ".dataset.pageRange.startsWith('1-')")
    # a row queued from this browser changes status while page 2 is open
    page.evaluate(f"rememberQueued({mine})")
    pager.locator("[data-page='2']").click()
    page.wait_for_function("document.querySelector(\"[data-pager='queue'] [data-page-range]\")"
                           ".dataset.pageRange.startsWith('26-')")
    db.update(mine, status="running", progress="1/4 · hellaswag")
    page.wait_for_selector(f"tr.landed[data-queue-row='{mine}']", timeout=15000)
    assert page.locator("[data-pager='queue'] [data-page-range]").get_attribute(
        "data-page-range").startswith("1-")
    assert page.errors == []


def test_the_queue_pager_is_built_once_and_updated_in_place(live, page):
    from service import db
    ids = _fill_queue()
    page.goto(live["base"] + "/#tab=queue")
    page.wait_for_selector("[data-pager='queue'] [data-page='2']")
    page.evaluate("window._qp = document.querySelector(\"[data-pager='queue']\");"
                  "window._qb = document.querySelector(\"[data-pager='queue'] [data-page='2']\")")
    db.update(ids[-1], progress="rewritten, so the poll redraws the queue")   # page 1
    page.wait_for_function("[...document.querySelectorAll('[data-queue-table] td')]"
                           ".some(td => td.textContent.includes('rewritten'))", timeout=15000)
    assert page.evaluate("document.querySelector(\"[data-pager='queue']\") === window._qp")
    assert page.evaluate("document.querySelector(\"[data-pager='queue'] [data-page='2']\") "
                         "=== window._qb")
    page.locator("[data-pager='queue'] [data-page-next]").click()
    page.wait_for_function("document.querySelector(\"[data-pager='queue'] [data-page-range]\")"
                           ".dataset.pageRange.startsWith('26-')")
    assert page.errors == []
