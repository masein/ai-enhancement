"""11d: the Overview's highlight cards and a tidy-up of the model page.

Every verdict is derived from the data under it — the best model's lead is
the z-test's call, not an adjective — and no average of provisional scores
appears anywhere, on a card, an area header or a sentence. 12b.2: Home is
rebuilt and the highlight cards are gone; the lead's verdict is the model
page's, on its Standard block, and the exam's sections are in its block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import open_kind

pytestmark = pytest.mark.dashboard
MODEL = "fx/good-750m"


def served(page, edit):
    """The same board as served, edited on the way to the page."""
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def ranked(body):
    return sorted((m for m in body["models"] if m.get("avg") is not None and not m.get("duplicateOf")),
                  key=lambda m: -m["avg"])


def ranked_names(page):
    return page.evaluate("""() => DATA.models.filter(m => officialAvg(m) != null && !m.duplicateOf)
      .sort((a, b) => officialAvg(b) - officialAvg(a)).map(m => m.name)""")


def lead_verdict(page, base, mid):
    """the top model's lead, where 12b.2 says it: its page's Standard block"""
    open_model(page, base, mid)
    open_kind(page, "standard")
    return page.locator("[data-avg-verdict]").text_content()


# ---------------------------------------------------------------------------
# the highlight cards
# ---------------------------------------------------------------------------

def test_a_lead_inside_the_noise_says_so(live, page):
    def close(body):
        top, nxt = ranked(body)[:2]
        top["avg"], nxt["avg"] = 0.900, 0.893             # still the top two
        top["avgSe"] = nxt["avgSe"] = 0.02
    served(page, close)
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-best='standard']")
    # Home's card: the number, and the name on its own line under it
    assert page.locator("[data-best-value='standard']").text_content() == "90.0"
    top = ranked_names(page)[0]
    assert page.locator("[data-best-name='standard']").text_content() == top
    mid = page.evaluate(f"DATA.models.find(m => m.name === {top!r}).id")
    v = lead_verdict(page, live["base"], mid)
    assert "by 0.7 points — within noise." in v and v.startswith("Leads ")
    assert page.errors == []


def test_a_lead_outside_the_noise_says_so(live, page):
    def apart(body):
        top, nxt = ranked(body)[:2]
        top["avg"], nxt["avg"] = 0.95, 0.75
        top["avgSe"] = nxt["avgSe"] = 0.02
    served(page, apart)
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-best='standard']")
    mid = page.evaluate(f"DATA.models.find(m => m.name === {ranked_names(page)[0]!r}).id")
    # (0.95 − 0.75) / √(0.02² + 0.02²) = 7.1: past 1.96, a real gap
    assert "by 20.0 points — a real gap." in lead_verdict(page, live["base"], mid)
    assert page.errors == []


def test_with_nothing_judged_home_has_no_exam_card_and_the_tile_says_so(live, page):
    """12b.2: a kind with no data has no card on Home; on a model page it is a
    tile reading Not tested · Test."""
    def nothing(body):
        for m in body["models"]:
            m["judge"] = None
            m["judgedAvg"] = None
    served(page, nothing)
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-best-by-kind]")
    assert page.locator("[data-best='exam']").count() == 0
    assert page.locator("[data-best='standard']").count() == 1
    open_model(page, live["base"])
    tile = page.locator("[data-kind-tile='exam']")
    assert "Not tested · Test" in " ".join(tile.inner_text().split())
    assert page.locator("[data-kind-block='exam']").count() == 0
    assert page.errors == []


def test_a_static_report_has_no_running_now(page, tree):
    """a static report has no queue: Home is what needs you and the best"""
    page.goto(tree["report"].as_uri())
    page.wait_for_selector("[data-best-by-kind]")
    assert page.locator("[data-running-now]").count() == 0
    assert page.locator("[data-needs-you]").count() == 1
    assert page.locator("[data-best='standard']").count() == 1
    assert page.errors == []


# ---------------------------------------------------------------------------
# the model page
# ---------------------------------------------------------------------------

def open_model(page, base, mid=MODEL):
    page.goto(base + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")


def test_the_preliminary_note_appears_exactly_once(live, page):
    def soft(body):
        m = next(x for x in body["models"] if x["id"] == MODEL)
        for v in (m["judge"] or {}).get("tasks", {}).values():
            v["propose"] = {"ok": False, "hard": [], "caution": None,
                            "soft": [{"short": "the judge is not calibrated"}],
                            "why": "the judged suite is preliminary"}
    served(page, soft)
    open_model(page, live["base"])
    open_kind(page, "exam")
    card = page.locator("#sec-judged")
    assert card.locator("[data-prelim-note]").count() == 1
    assert card.get_by_text("the topic page says what proposing would mean").count() == 1
    # the row's action is a small text button, one per row
    rows = card.locator("[data-judged-topics] tbody tr[data-topic]").count()
    # 11k: it opens the New proposal dialog in place, so it is a button now
    assert card.locator("[data-judged-topics] .propose").count() == rows
    assert card.locator("[data-judged-topics] .propose").first.text_content() == "Propose →"
    assert page.errors == []


def test_topics_are_grouped_under_their_areas_weakest_first(live, page):
    open_model(page, live["base"])
    open_kind(page, "exam")
    table = page.locator("[data-judged-topics]")
    table.wait_for()
    groups = page.evaluate("""() => { const out = []; let cur = null;
      for (const tr of document.querySelectorAll('[data-judged-topics] tbody tr')) {
        if (tr.dataset.areaRow) { cur = {area: tr.dataset.areaRow, head: tr.textContent, v: []};
          out.push(cur); }
        else if (cur) cur.v.push(parseFloat(tr.children[1].textContent)); }
      return out; }""")
    areas = page.evaluate("Object.keys(DATA.meta.areas)")
    assert [g["area"] for g in groups] == [a for a in areas if any(g["area"] == a for g in groups)]
    for g in groups:
        assert g["v"] == sorted(g["v"]), g["area"]              # weakest first, inside the area
        assert "topics judged" in g["head"]
    assert page.errors == []


def test_the_length_table_has_one_row_per_topic(live, page):
    open_model(page, live["base"])
    open_kind(page, "exam")
    page.wait_for_selector("[data-length-table]", state="attached")
    n = page.evaluate(f"""() => {{ const m = DATA.models.find(x => x.id === '{MODEL}');
      return Object.entries(m.judge.tasks).filter(([t, v]) => t.startsWith('exam_')
        && pubScore(v) != null && (v.score_vs_length || []).length).length; }}""")
    rows = page.locator("[data-length-table] tbody tr[data-length-row]")
    assert rows.count() == n and n > 0
    assert len(set(rows.all_text_contents())) == n                # one row each
    assert page.locator("[data-length-table] thead th").count() == 5
    # more than ten topics: it starts folded
    if n > 10:
        fold = page.locator("[data-length-fold]")
        assert fold.get_attribute("open") is None
        assert f"({n} topics)" in fold.locator("summary").text_content()
    # a cell under five answers is muted and says so; the rest are grey by the
    # mean, never a rank tint
    assert page.locator("[data-length-table] td.lencell[data-step]").count() == 0
    assert page.errors == []


def test_no_average_of_provisional_scores_appears_anywhere(live, page):
    def provisional(body):
        body["judged"]["calibration"] = None
        for m in body["models"]:
            if m.get("judge"):
                m["judge"]["judge"]["provisional"] = True
                m["judge"]["judge"]["provisional_reason"] = "a local model was the judge"
                m["judgeState"] = {**(m.get("judgeState") or {}), "ok": False,
                                   "reasons": ["a local judge — provisional"]}
            m["judgedAvg"] = None                                  # judged_avg() says None
    served(page, provisional)
    open_model(page, live["base"])
    open_kind(page, "exam")
    page.wait_for_selector("[data-judged-topics]")
    text = page.locator("#view").text_content()
    assert page.locator("[data-area-mean]").count() == 0
    assert "Judged average" not in text
    assert "demo only, not ranked" in text
    # the tile has no number to show, and Home no Knowledge exam card
    assert page.locator("[data-kind-value='exam']").text_content() == "—"
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-best-by-kind]")
    assert page.locator("[data-best='exam']").count() == 0
    assert "mean" not in page.locator("#view").text_content().lower()
    assert page.errors == []


def test_screenshots_for_the_pr(live, page):
    out = Path(__file__).resolve().parent / "_screens" / "phase11"
    out.mkdir(parents=True, exist_ok=True)
    for width in (1280, 400):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-best-by-kind]")
        page.wait_for_timeout(300)
        page.locator("[data-best-by-kind]").screenshot(path=out / f"overview-highlights-{width}-light.png")
        open_model(page, live["base"])
        page.wait_for_timeout(300)
        page.locator("[data-model-hero]").screenshot(path=out / f"model-hero-{width}-light.png")
    assert page.errors == []
