"""12i.1 on the page: AI models under the name menu — each job's model, its
price and provider, change ▾ with Local first and the suggested model marked,
the warnings that hold, the monthly spend and its limit, and a new judge
asking first before it re-judges. The judge test: one answer at a time, keys
0–4 (or P and F) and S, marks that save themselves, and the judges' marks
never shown while marking. With a fake OpenRouter: nothing leaves the machine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import set_name
from fake_openrouter import FakeOpenRouter

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12i1"


def shot(page_or_part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page_or_part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture(autouse=True)
def clean_ai(live):
    """every test starts from the environment's models, no spend, no marks"""
    from service import db
    import service.app as appmod
    for job in ("judge", "writer", "data", "checker"):
        db.ai_set("job:" + job, None)
    db.ai_set("spend_limit", 20)
    db.ai_set("judge_test:candidates", {})
    import sqlite3
    from service import config
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("DELETE FROM ai_spend")
        c.execute("DELETE FROM judge_test_marks")
    (config.BENCH_ROOT / "ai" / "openrouter_models.json").unlink(missing_ok=True)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def ai_page(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=ai")
    page.wait_for_selector("[data-ai-page] [data-ai-jobs]")


def test_ai_models_is_under_the_name_menu_with_one_row_per_job(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    set_name_first(page, live["base"])
    page.locator("#who button.who").click()
    page.locator("#pop-who [data-menu='ai']").click()
    page.wait_for_selector("[data-ai-page] [data-ai-jobs]")
    assert page.evaluate("location.hash") == "#tab=ai"
    jobs = [r.get_attribute("data-ai-job") for r in page.locator("[data-ai-job]").all()]
    assert jobs == ["judge", "writer", "data", "checker"]
    assert page.locator("[data-ai-job='judge'] td").first.inner_text().startswith("Judge\n")
    # change ▾: Local first, then the suggested model, marked, with why
    page.locator("[data-ai-change-menu='judge']").click()
    menu = page.locator("#pop-ai-judge")
    menu.wait_for()
    picks = [b.get_attribute("data-ai-pick") for b in menu.locator("[data-ai-pick]").all()]
    assert picks[:2] == ["local", "deepseek/deepseek-v4.1-flash"]
    assert menu.locator("[data-ai-suggested]").inner_text() == "suggested"
    assert menu.locator("[data-ai-why='judge']").inner_text() == \
        "cheap and strong as a judge when given a reference answer"
    assert "$0.14 in · $0.42 out" in menu.locator(
        "[data-ai-pick='deepseek/deepseek-v4.1-flash']").inner_text()
    shot(page, "12i1-ai-models-change-1400-light.png")
    # choosing it: pinned, with its provider, and the judge asks first
    menu.locator("[data-ai-pick='deepseek/deepseek-v4.1-flash']").click()
    page.wait_for_selector("[data-ai-rejudge]")
    row = page.locator("[data-ai-job='judge']")
    assert row.locator("[data-ai-now]").inner_text() == "DeepSeek V4.1 Flash"
    assert row.locator("[data-ai-provider]").inner_text() == \
        "on InferenceNet · fp8 · deepseek/deepseek-v4.1-flash-20260910"
    assert row.locator("[data-ai-price]").inner_text() == "$0.10 in · $0.40 out"
    box = page.locator("[data-ai-rejudge]")
    assert box.locator("p").first.inner_text().startswith("Re-judge the ")
    assert " answers on file with the new judge? About $" in box.inner_text()
    shot(page, "12i1-ai-models-rejudge-1400-light.png", full_page=True)
    # Later: the old scores stay in History
    box.locator("[data-ai-rejudge-later]").click()
    assert page.locator("[data-ai-rejudge]").count() == 0
    assert page.errors == []


def set_name_first(page, base):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")


def test_a_new_judge_moves_old_judged_scores_to_history(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    from service import ai_models
    ai_models.save("judge", "deepseek/deepseek-v4.1-flash", "masein")
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-mtab='history']").click()
    card = page.locator("[data-judged-earlier]")
    card.wait_for()
    assert card.locator("[data-judged-by]").inner_text().startswith("judged by ")
    assert "in no table and no comparison until they are judged again" in card.inner_text()
    shot(card, "12i1-history-judged-by-1400-light.png")
    assert page.errors == []


def test_warnings_show_only_when_they_hold(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    from service import ai_models
    ai_models.save("judge", "z-ai/glm-5.3-flash", "masein")
    ai_models.save("data", "z-ai/glm-5.3", "masein")
    ai_page(page, live["base"])
    warns = page.locator("[data-ai-warning]").all_inner_texts()
    assert warns == ["The judge and the training-data writer are both GLM: a judge tends to "
                     "favour its own family's style."]
    ai_models.save("judge", "deepseek/deepseek-v4.1-flash", "masein")
    page.reload()
    page.wait_for_selector("[data-ai-jobs]")
    assert page.locator("[data-ai-warning]").count() == 0
    assert page.errors == []


def test_at_the_limit_the_page_says_ai_jobs_wait(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    from service import db
    db.ai_set("spend_limit", 5)
    db.spend_add("judge", "deepseek/deepseek-v4.1-flash", "InferenceNet", 10, 10, 5.5)
    ai_page(page, live["base"])
    assert page.locator("[data-ai-spend]").get_attribute("data-ai-spend") == "5.5"
    assert page.locator("[data-ai-waiting]").inner_text() == (
        "waiting: this month's AI spend has reached its $5.00 limit — raise it on AI models, or "
        "wait for next month")
    assert page.errors == []


def test_with_no_key_only_local_is_offered(live, page, monkeypatch):
    from service import config
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    ai_page(page, live["base"])
    assert page.locator("[data-ai-no-key]").inner_text().startswith(
        "OpenRouter has no key on this server, so only the local model is offered.")
    page.locator("[data-ai-change-menu='writer']").click()
    menu = page.locator("#pop-ai-writer")
    menu.wait_for()
    assert [b.get_attribute("data-ai-pick") for b in menu.locator("[data-ai-pick]").all()] == \
        ["local"]
    shot(page, "12i1-ai-models-no-key-1400-light.png")
    assert page.errors == []


def test_improve_names_each_jobs_model_and_where_to_change_it(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    from service import ai_models
    ai_models.save("data", "z-ai/glm-5.3", "masein")
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=improve")
    line = page.locator("[data-imp-ai]")
    line.wait_for()
    judge = ai_models.label("judge")
    assert line.inner_text() == f"AI: judge {judge} · writer GLM 5.3 · change"
    # a new judge: the answers on file wait for it, and Improve says so
    ai_models.save("judge", "deepseek/deepseek-v4.1-flash", "masein")
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.reload()
    empty = page.locator("[data-pipeline='none'] [data-empty]")
    empty.wait_for()
    assert empty.locator("p").inner_text() == (
        "Nothing to improve yet: the exam answers on file were judged by another judge, and a "
        "score compares only with its own judge’s.")
    empty.locator("[data-empty-action]").click()
    page.wait_for_selector("[data-ai-page]")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the judge test
# ---------------------------------------------------------------------------

def test_the_judge_test_marks_one_answer_at_a_time_and_saves_as_it_goes(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    set_name_first(page, live["base"])
    ai_page(page, live["base"])
    page.locator("[data-jt-mark]").click()
    page.wait_for_selector("[data-jt-answer]")
    assert page.evaluate("location.hash") == "#tab=ai&sub=mark"
    ans = page.locator("[data-jt-answer]")
    total = int(ans.locator("[data-jt-at]").get_attribute("data-jt-at").split("|")[1])
    # the question, what a full answer contains, and the answer — never a judge's mark
    assert ans.locator(".evq").inner_text() and ans.locator("[data-jt-text]").inner_text()
    assert ans.locator("[data-jt-criteria], [data-jt-rubric]").count() == 1
    held = page.evaluate("JSON.stringify(state.ai.jt.answers)")
    assert '"judge"' not in held
    shot(page, "12i1-judge-test-mark-1400-light.png", full_page=True)
    # keys: a mark, then a skip; each moves on and is saved
    first = ans.get_attribute("data-jt-answer")
    scale = ans.get_attribute("data-jt-scale")
    page.keyboard.press("3" if scale == "0-4" else "p")
    page.wait_for_function(f"document.querySelector('[data-jt-answer]').dataset.jtAnswer !== "
                           f"{json.dumps(first)}")
    page.keyboard.press("s")
    page.wait_for_function("state.ai.jt.progress.skipped === 1")
    from service import db, judge_test
    got = db.jt_marks(judge_test.PERSON)
    assert got[first] == (3 if scale == "0-4" else 4) and list(got.values()).count(None) == 1
    # a reload lands where he stopped: the third answer, one marked
    page.reload()
    page.wait_for_selector("[data-jt-answer]")
    assert page.locator("[data-jt-at]").get_attribute("data-jt-at") == f"3|{total}"
    assert page.locator("[data-jt-at]").inner_text() == f"3 of {total} · 1 marked"
    page.locator("[data-jt-back]").click()
    page.wait_for_selector("[data-jt-progress]")
    assert page.evaluate("location.hash") == "#tab=ai"
    assert page.locator("[data-jt-progress]").get_attribute("data-jt-progress") == f"1|{total}"
    assert page.locator("[data-jt-mark]").inner_text() == "Continue marking"
    assert page.errors == []


def test_the_result_table_marks_the_best_and_offers_it_as_the_judge(live, page, monkeypatch):
    FakeOpenRouter.install(monkeypatch)
    rows = [{"key": "cur", "name": "overlap-v1", "current": True, "id": "stub/overlap-v1",
             "n": 110, "exact": 0.52, "within1": 0.9, "kappa": 0.64, "per_1000": None},
            {"key": "ds", "name": "DeepSeek V4.1 Flash", "current": False,
             "id": "deepseek/deepseek-v4.1-flash", "provider": "InferenceNet", "n": 110,
             "exact": 0.61, "within1": 0.95, "kappa": 0.78, "per_1000": 0.4, "best": True}]
    page.route("**/api/judge-test/result", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"rows": rows, "person": {"total": 110, "marked": 110, "skipped": 0,
                                                  "next": None},
                         "kappa_min": 0.7, "n_min": 100})))
    ai_page(page, live["base"])
    table = page.locator("[data-jt-result]")
    table.wait_for()
    assert table.locator("thead th").all_text_contents()[:5] == [
        "Judge", "Same mark as you", "Within 1 point", "Agreement (0–1)",     # 12i.3
        "Cost per 1,000 answers"]
    best = table.locator("tr[data-jt-row='ds']")
    assert best.locator("[data-jt-best]").inner_text() == "best"
    assert best.locator("[data-jt-kappa]").inner_text() == "0.78 · 110"
    assert best.locator("[data-jt-use]").inner_text() == "Use this judge"
    assert table.locator("tr[data-jt-row='cur'] [data-jt-use]").count() == 0
    shot(page.locator("[data-judge-test]"), "12i1-judge-test-result-1400-light.png")
    assert page.errors == []


@pytest.mark.parametrize("width", [400])
def test_the_pages_at_phone_width(live, page, monkeypatch, width):
    FakeOpenRouter.install(monkeypatch)
    ai_page(page, live["base"], width)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12i1-ai-models-{width}-light.png", full_page=True)
    page.locator("[data-jt-mark]").click()
    page.wait_for_selector("[data-jt-answer]")
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12i1-judge-test-mark-{width}-light.png", full_page=True)
    assert page.errors == []


def test_a_checked_judge_says_what_it_rests_on(live, page, monkeypatch):
    """κ 0.7 or more on 100 answers or more takes "provisional" off, and the
    badge says what that rests on"""
    FakeOpenRouter.install(monkeypatch)
    from service import config
    import service.app as appmod
    p = config.OUT_DIR / "judge_calibration.json"
    before = p.read_text(encoding="utf-8") if p.exists() else None
    # the judge that marked the board's answers
    head = json.loads((live["tree"]["models"]["fx/good-750m"]["dir"] / "judge.json")
                      .read_text(encoding="utf-8"))["judge"]
    p.write_text(json.dumps({
        "kappa": 0.78, "n": 120, "exact": 0.6, "within1": 0.95, "kappa_min": 0.7, "n_min": 100,
        "calibrated": True, "by": "masein",
        "method": "judge test: weighted kappa (quadratic) against a person's marks",
        "judge": {"id": head["id"], "version": head["version"]["key"]}}), encoding="utf-8")
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=models&view=exam")
        badge = page.locator("[data-lb-card] h2 [data-judge-checked]")
        badge.wait_for()
        assert badge.inner_text() == "judge checked against masein on 120 answers · κ 0.78"
        assert page.locator("[data-exam-off]").count() == 0
        shot(page.locator("[data-lb-card]"), "12i1-exam-judge-checked-1400-light.png")
        ai_page(page, live["base"])
        assert page.locator("[data-jt-calibration]").inner_text() == \
            "The judge now: checked against masein on 120 answers · κ 0.78"
        assert page.errors == []
    finally:
        if before is None:
            p.unlink()
        else:
            p.write_text(before, encoding="utf-8")
        appmod._cache.update(key=None, payload=None, at=0.0)
