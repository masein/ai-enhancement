"""12n.1 on the page: Benchmarks ▸ Everyday tasks' readers. A group's name
opens every practice question of the group, each once, with each chosen
model's answer beside it — its mark, the check that decided it, the answer
with the thinking folded — "models disagree first", and its own address. A
count opens that model's practice answers under a heading that says what the
number was. The hidden half is the owner's, after a warning, logged and
listed under Data & sources. And a question is edited where it is read: its
impact, then saved, the answers marked again, and undone. Fixtures only."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import go_tab, set_name
from test_everyday_improve_12g2 import leaks

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12n1"
GOOD, SKEWED = "fx/good-750m", "fx/skewed-360m"
G = "writing"
WARNING = ("These questions are the test. Don’t train on them or write questions toward them. "
           "This opening is logged.")


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def everyday(page, live, extra="", width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=benchmarks&sub=everyday&models={ids(GOOD, SKEWED)}" + extra)
    page.wait_for_selector("[data-everyday-table]")


def open_group(page, g=G):
    page.locator(f"a[data-read-open='group:{g}']").click()
    page.wait_for_selector("#reader[data-kind='group'] [data-grp-q]")


def test_a_groups_name_opens_its_practice_questions_with_each_models_answer(live, page):
    everyday(page, live)
    set_name(page, "sam")
    open_group(page)
    assert "read=group:writing" in page.evaluate("location.hash")
    want = page.evaluate(f"evdQs({json.dumps(G)}).map(q => q.id)")
    shown = page.locator("[data-grp-q]").evaluate_all("xs => xs.map(x => x.dataset.grpQ)")
    assert sorted(shown) == sorted(want) and len(shown) == len(set(shown))    # each once
    for qid in shown:
        cards = page.locator(f"[data-grp-answers='{qid}'] [data-grp-answer]").evaluate_all(
            "xs => xs.map(x => x.dataset.grpAnswer.split('|')[0])")
        assert sorted(cards) == sorted([GOOD, SKEWED]), qid
    # a card: its mark, what decided it, the whole answer, the thinking folded
    card = page.locator(f"[data-grp-answer='{GOOD}|{shown[0]}']")
    assert card.locator("[data-grp-decided]").inner_text()
    assert card.locator("[data-grp-text]").inner_text() == page.evaluate(
        f"evdItem(evdOf({json.dumps(GOOD)}), {json.dumps(shown[0])}).answer_text")
    # the thinking, where there is some, folded
    think = page.evaluate(f"""evdQs({json.dumps(G)}).map(q => q.id).find(id =>
      (evdItem(evdOf({json.dumps(GOOD)}), id) || {{}}).had_reasoning)""")
    if think:
        fold = page.locator(f"[data-grp-answer='{GOOD}|{think}'] details.evthink")
        assert fold.count() == 1 and fold.get_attribute("open") is None
    # the hidden half: its count and a line, for anyone but the owner
    assert page.locator(f"[data-grp-hidden-line='{G}']").inner_text() == \
        f"The {page.evaluate('evdHidden(' + json.dumps(G) + ')')} hidden ones score it and are not shown."
    assert page.locator("[data-audit-open]").count() == 0
    # models disagree first
    page.locator("[data-grp-sort]").click()
    split = page.locator("[data-grp-q]").evaluate_all("xs => xs.map(x => !!x.dataset.grpSplit)")
    n = sum(split)
    assert n and split == [True] * n + [False] * (len(split) - n)
    shot(page.locator("#reader aside"), "everyday-group-reader.png")
    # the address opens it
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=benchmarks&sub=everyday&models={ids(GOOD, SKEWED)}"
              + "&read=group:writing")
    page.wait_for_selector("#reader[data-kind='group'] [data-grp-q]")
    assert leaks(page.content()) == []
    assert page.errors == []


def test_a_count_says_what_its_number_was(live, page):
    everyday(page, live)
    page.locator(f"[data-evd-cell='{GOOD}|{G}']").click()
    head = page.locator(f"[data-evd-count-head='{GOOD}|{G}']")
    h = page.evaluate(f"evdOf({json.dumps(GOOD)}).groups[{json.dumps(G)}]")
    p = page.evaluate(f"evdOf({json.dumps(GOOD)}).practice[{json.dumps(G)}]")
    assert head.inner_text() == (f"Scored: {h['passed']} of {h['total']} hidden (not shown) · "
                                 f"Practice below: {p['passed']} of {p['total']}")
    assert page.errors == []


def test_the_hidden_half_is_the_owners_after_the_warning_and_logged(live, page):
    everyday(page, live)
    set_name(page, "masein")
    open_group(page)
    page.locator(f"[data-audit-open='{G}']").click()
    dlg = page.locator(f"[data-audit-dialog='{G}']")
    assert dlg.locator("[data-audit-warning]").inner_text() == WARNING
    dlg.locator("[data-dialog-cancel]").click()
    audits = page.evaluate("api('api/everyday/audits')")
    assert audits["audits"] == []                         # a warning closed is no opening
    page.locator(f"[data-audit-open='{G}']").click()
    page.locator("[data-audit-go]").click()
    banner = page.locator(f"[data-audit-banner='{G}']")
    banner.wait_for()
    assert WARNING in banner.inner_text()
    n = page.evaluate(f"evdHidden({json.dumps(G)})")
    assert page.locator("[data-grp-q]").count() == n
    assert page.locator("[data-grp-answer]").count() == 2 * n                # side by side
    assert "audit" not in page.evaluate("location.hash")                     # nothing links to it
    shot(page.locator("#reader aside"), "everyday-audit.png")
    page.locator("[data-audit-close]").click()
    page.wait_for_selector(f"[data-audit-banner='{G}']", state="detached")
    # every opening, under Data & sources
    page.locator("[data-reader-close]").click()
    page.wait_for_selector("#reader", state="detached")
    go_tab(page, "Provenance")
    row = page.locator("[data-audits='1'] [data-audit-row]")
    row.wait_for()
    assert row.inner_text().startswith("masein · ") and row.inner_text().endswith(
        f" · Writing · {n} questions")
    assert page.errors == []


def test_a_question_is_edited_where_it_is_read(live, page):
    everyday(page, live)
    set_name(page, "masein")
    open_group(page)
    # a question the two models split on, marked by its script checks alone
    qid = page.evaluate(f"""evdQs({json.dumps(G)}).filter(q => !q.judged).map(q => q.id).find(id => {{
      const a = evdItem(evdOf({json.dumps(GOOD)}), id), b = evdItem(evdOf({json.dumps(SKEWED)}), id);
      return a && b && a.pass === true && b.pass === false; }})""")
    assert qid
    page.locator(f"[data-qe-open='{qid}']").click()
    form = page.locator(f"[data-qe-form='{qid}']")
    form.wait_for()
    form.locator("[data-qe='checks']").fill(json.dumps([{"type": "contains_any",
                                                         "values": ["not sure"]}]))
    form.locator("[data-qe-save]").click()
    page.wait_for_selector("[data-qe-error]")
    assert page.locator("[data-qe-error]").inner_text() == "Why: every edit keeps its reason"
    form.locator("[data-qe='why']").fill("doubt is the right answer here")
    form.locator("[data-qe-impact-btn]").click()
    imp = page.locator("[data-qe-impact]")
    imp.wait_for()
    counts = json.loads(imp.get_attribute("data-qe-impact"))
    assert counts["flips"] >= 2 and imp.inner_text().startswith(f"{counts['flips']} would flip: ")
    assert "good-750m ✓→✗" in imp.inner_text() and "skewed-360m ✗→✓" in imp.inner_text()
    shot(page.locator(f"[data-grp-q='{qid}']"), "everyday-edit-impact.png")
    form.locator("[data-qe-save]").click()
    page.wait_for_selector(f"[data-grp-answer='{GOOD}|{qid}'][data-mark='no']")
    assert page.locator(f"[data-grp-answer='{SKEWED}|{qid}']").get_attribute("data-mark") == "ok"
    assert page.locator(f"[data-grp-edited='{qid}']").inner_text().endswith(
        "by masein: doubt is the right answer here")
    # Undo, from its history
    page.locator(f"[data-qe-open='{qid}']").click()
    page.locator("[data-qe-undo]").click()
    page.wait_for_selector(f"[data-grp-answer='{GOOD}|{qid}'][data-mark='ok']")
    assert page.locator(f"[data-grp-answer='{SKEWED}|{qid}']").get_attribute("data-mark") == "no"
    # the refusal's own 422, as the browser logs it, and nothing else
    assert [e for e in page.errors if "status of 422" not in e] == []
