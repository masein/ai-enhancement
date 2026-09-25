"""12a.5b on the page: IFEval, MMLU-Pro and MATH-500 in the submit form's own
group, "Slow · instruct models only", none ticked; the time the run will hold
the GPU said before Submit, from this server's past runs or as a rough guess,
changing with every box and model, and one amber line over an hour; MMLU-Pro
the seeded 1,200 unless Full is chosen, each with its time; a subset labelled
on the board as a column of its own; and a cancelled run saying what it kept."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generative_fixture import write_run

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12a5b"
MODEL = "fx/below-135m-it"          # the fixture's instruct model
QWEN = "Qwen/Qwen3.5-2B"
# measured here: seconds per item per billion parameters
PACE = {"tasks": {"ifeval": {"per_b": 1.0, "runs": 3}, "mmlu_pro": {"per_b": 0.33, "runs": 2},
                  "hendrycks_math500": {"per_b": 2.0, "runs": 1}},
        "items": {"ifeval": 541, "mmlu_pro": 12032, "hendrycks_math500": 500}, "subset": 1200}


@pytest.fixture(scope="module", autouse=True)
def subset_run(live):
    """the fixture's instruct model sat MMLU-Pro as a subset"""
    import service.app as appmod
    write_run(live["tree"]["out_dir"], MODEL, thinking=False, subset=1200)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield


def shot(page_or_part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page_or_part.screenshot(path=SCREENS / name, **kw)


def the_form(page, base, pace=None, width=1400):
    if pace is not None:
        page.route("**/api/gen/pace", lambda r: r.fulfill(
            status=200, content_type="application/json", body=json.dumps(pace)))
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(base + "/#tab=home")
    page.locator("header [data-test-model]").click()
    dlg = page.locator("[data-dialog='test']")
    dlg.wait_for()
    page.get_by_label("suite").click()
    opt = page.locator("#pop-sel-submit-suite [role=option][data-value='generative']")
    assert opt.inner_text().startswith("Instruction & maths — slow, instruct models only")
    opt.click()
    dlg.locator("[data-gen-opts]").wait_for()
    return dlg


def pick(page, model=QWEN, params=2e9):
    page.evaluate(f"state.sub.hf_id = {json.dumps(model)}; state.sub.params = {params}; render()")


def tick(dlg, task):
    dlg.locator(f"[data-gen-pick='{task}'] input").check()


def test_the_three_sit_unticked_in_their_own_group(live, page):
    dlg = the_form(page, live["base"])
    group = dlg.locator("[data-gen-opts]")
    assert group.locator("legend").text_content() == "Slow · instruct models only"
    boxes = group.locator("[data-gen-pick]")
    assert [b.get_attribute("data-gen-pick") for b in boxes.all()] == \
        ["ifeval", "hendrycks_math500", "mmlu_pro"]
    assert [b.locator("input").is_checked() for b in boxes.all()] == [False] * 3
    assert dlg.locator("[data-gen-estimate]").inner_text() == "Tick the ones to run."
    # MMLU-Pro's size shows once it is ticked
    assert dlg.locator("[data-mmlu-sizes]").is_hidden()
    # Submit with none ticked is refused, and says why
    pick(page)
    dlg.get_by_role("button", name="Submit model").click()
    assert "tick at least one of IFEval, MATH-500 and MMLU-Pro" in dlg.inner_text()
    page.keyboard.press("Escape")
    assert page.errors == []


def test_before_any_run_here_the_time_is_a_rough_guess(live, page):
    dlg = the_form(page, live["base"], pace={**PACE, "tasks": {}})
    pick(page)
    tick(dlg, "ifeval")
    # 7.4 s per item per billion parameters × 2 B × 541 items
    assert dlg.locator("[data-gen-estimate]").inner_text() == \
        "Time on the GPU: about 2 h, a rough guess"
    assert dlg.locator("[data-gen-long]").inner_text() == \
        "This holds the GPU for about 2 h 13 min; other runs wait."
    page.keyboard.press("Escape")
    assert page.errors == []


def test_the_time_follows_the_boxes_and_the_model_and_warns_over_an_hour(live, page):
    dlg = the_form(page, live["base"], pace=PACE)
    est, long = dlg.locator("[data-gen-estimate]"), dlg.locator("[data-gen-long]")
    tick(dlg, "ifeval")
    # no size yet: the page says what it needs
    assert est.inner_text() == "The time needs the model’s size: pick it from the list."
    pick(page)
    assert est.inner_text() == "Time on the GPU: about 18 min"                # 1.0 × 2 × 541
    assert long.is_hidden()
    tick(dlg, "mmlu_pro")
    sizes = dlg.locator("[data-mmlu-sizes]")
    assert sizes.is_visible()
    # the seeded 1,200 unless Full is chosen, each with its time
    assert sizes.locator("[data-mmlu-size='subset'] input").is_checked()
    words = lambda loc: " ".join(loc.inner_text().split())               # noqa: E731
    assert words(sizes.locator("[data-mmlu-size='subset']")) == "Subset (1,200) · about 13 min"
    assert words(sizes.locator("[data-mmlu-size='full']")) == "Full (12,032) · about 2 h 12 min"
    assert est.inner_text() == "Time on the GPU: about 31 min"
    sizes.locator("[data-mmlu-size='full'] input").check()
    assert est.inner_text() == "Time on the GPU: about 2 h 30 min"
    assert long.is_visible() and long.inner_text() == \
        "This holds the GPU for about 2 h 30 min; other runs wait."
    shot(dlg.locator(".dlg"), "12a5b-form-1400-light.png")
    # a bigger model takes longer
    pick(page, "Qwen/Qwen3.5-4B", 4e9)
    assert est.inner_text() == "Time on the GPU: about 5 h 1 min"
    # what goes in: the ticked ones, and Full
    sent = []

    def queued(route):
        sent.append(json.loads(route.request.post_data))
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"id": 991, "status": "queued"}))
    page.route("**/api/submissions", lambda r: queued(r) if r.request.method == "POST"
               else r.continue_())
    dlg.get_by_role("button", name="Submit model").click()
    page.wait_for_function("() => !document.querySelector(\"[data-dialog='test']\")")
    assert sent[0]["suite"] == "generative" and sent[0]["tasks"] == ["ifeval", "mmlu_pro"]
    assert sent[0]["subset"] == 0
    assert page.errors == []


def test_the_subset_is_the_default_that_goes_in(live, page):
    dlg = the_form(page, live["base"], pace=PACE)
    pick(page)
    tick(dlg, "mmlu_pro")
    sent = []
    page.route("**/api/submissions", lambda r: (sent.append(json.loads(r.request.post_data)),
                                                r.fulfill(status=200, content_type="application/json",
                                                          body='{"id": 992, "status": "queued"}'))
               if r.request.method == "POST" else r.continue_())
    dlg.get_by_role("button", name="Submit model").click()
    page.wait_for_function("() => !document.querySelector(\"[data-dialog='test']\")")
    assert sent[0]["tasks"] == ["mmlu_pro"] and sent[0]["subset"] == 1200
    assert page.errors == []


def test_a_subset_is_labelled_and_a_column_of_its_own(live, page):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    page.locator("[data-chip='instruction']").click()
    page.wait_for_selector("th[data-col='mmlu_pro_subset']")
    heads = [h.lower() for h in page.locator("[data-lb-table] thead tr.names th .hname")
             .all_inner_texts()]
    assert "mmlu-pro subset" in heads and "mmlu-pro" not in heads      # nobody sat the full one
    tip = json.loads(page.locator("th[data-col='mmlu_pro_subset']").get_attribute("data-tip"))
    assert any("never averaged or compared with the full MMLU-Pro" in t for t in tip)
    cell = page.locator(f"tr[data-lb-row='{MODEL}'] [data-subset='mmlu_pro_subset']")
    assert cell.text_content() == "subset"
    assert page.errors == []


def test_a_cancelled_run_says_what_it_kept(live, page):
    from service import db
    sid = db.add(QWEN, "instruct", "generative", "masein", "")
    db.update(sid, status="canceled",
              progress="cancelled during MMLU-Pro · IFEval and MATH-500 kept")
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=runs")
    row = page.locator(f"tr[data-queue-row='{sid}']")
    row.wait_for()
    assert "cancelled during MMLU-Pro · IFEval and MATH-500 kept" in row.inner_text()
    assert page.errors == []


@pytest.mark.parametrize("width", [400])
def test_the_form_at_phone_width(live, page, width):
    dlg = the_form(page, live["base"], pace=PACE, width=width)
    pick(page)
    tick(dlg, "ifeval")
    tick(dlg, "mmlu_pro")
    dlg.locator("[data-mmlu-size='full'] input").check()
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12a5b-form-{width}-light.png")
    page.keyboard.press("Escape")
    assert page.errors == []
