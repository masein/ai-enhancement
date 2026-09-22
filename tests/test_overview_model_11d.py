"""11d: the Overview's highlight cards and a tidy-up of the model page.

Every verdict on a highlight card is derived from the data under it — the
best model's lead is the z-test's call, not an adjective — and no average of
provisional scores appears anywhere, on a card, an area header or a sentence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def verdict(page, key):
    return page.locator(f"[data-verdict='{key}']").text_content()


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
    page.wait_for_selector("[data-highlights]")
    v = verdict(page, "best")
    assert "by 0.7 points — within noise." in v and v.startswith("Leads ")
    # 11e: the value is the number; the name is its own line under it
    assert page.locator("[data-hl-value='best']").text_content() == "90.0"
    assert page.locator("[data-hl-name='best']").text_content() == ranked_names(page)[0]
    assert page.errors == []


def test_a_lead_outside_the_noise_says_so(live, page):
    def apart(body):
        top, nxt = ranked(body)[:2]
        top["avg"], nxt["avg"] = 0.95, 0.75
        top["avgSe"] = nxt["avgSe"] = 0.02
    served(page, apart)
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights]")
    v = verdict(page, "best")
    # (0.95 − 0.75) / √(0.02² + 0.02²) = 7.1
    assert "by 20.0 points — a real gap (z = 7.1)." in v
    assert page.errors == []


def test_with_nothing_judged_the_cards_say_so(live, page):
    def nothing(body):
        for m in body["models"]:
            m["judge"] = None
            m["judgedAvg"] = None
    served(page, nothing)
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights]")
    assert verdict(page, "weakest") == "No model has been judged yet — Loop ▸ Sit the exam."
    n = page.evaluate("DATA.judged.exam.length")
    assert page.locator("[data-hl-value='loop']").text_content() == f"0 / {n}"
    assert page.locator("[data-hl-name='loop']").text_content() == "topics judged"
    assert "No judged run yet." in verdict(page, "judge")
    assert page.errors == []


def test_a_static_report_has_only_the_best_model_card(page, tree):
    page.goto(tree["report"].as_uri())
    page.wait_for_selector("[data-highlights]")
    assert page.locator("[data-hl]").count() == 1
    assert page.locator("[data-hl='best']").count() == 1
    assert page.errors == []


def test_the_live_cards_are_derived_from_the_board(live, page):
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights] [data-hl='judge']")
    facts = page.evaluate("""() => { const m = loopModel();
      const xs = Object.entries(m.judge.tasks).filter(([t]) => t.startsWith('exam_'))
        .map(([t, v]) => [t, pubScore(v)]).filter(x => x[1] != null).sort((a, b) => a[1] - b[1]);
      return {name: m.name, weakest: frName(xs[0][0]), v: xs[0][1], n: xs.length,
              exam: DATA.judged.exam.length,
              canary: m.judge.canary}; }""")
    assert page.locator("[data-hl-name='weakest']").text_content() == facts["weakest"]
    assert page.locator("[data-hl-value='weakest']").text_content() == f"{facts['v']:.2f} / 4"
    assert verdict(page, "weakest").startswith(f"{facts['name']}, {facts['n']} of {facts['exam']} topics judged")
    assert verdict(page, "loop").startswith(f"Last judged {facts['name']}")
    cn = facts["canary"]
    assert page.locator("[data-hl-value='judge']").text_content() == f"{cn['graded']} / {cn['n']}"
    assert page.locator("[data-hl-name='judge']").text_content() == \
        ("moved" if cn["drifted"] else "steady")
    # the tiles are one line under the hero now
    line = page.locator("[data-statline]").text_content()
    assert "models · " in line and "gaps are real" in line and "of evaluation" in line
    assert page.locator(".tiles").count() == 0
    # Top models: the table component, the rank, the tint, and a way on
    top = page.locator("[data-top-models]")
    # 11f: the Leaderboard's rule — the leaders bold and tinted, the rest plain
    assert 1 <= top.locator("td[data-lead]").count() < top.locator("tbody tr").count()
    assert top.locator("td[data-lead] b").count() == top.locator("td[data-lead]").count()
    assert top.locator("[data-see-leaderboard]").count() == 1
    assert page.locator("[data-biggest-gap]").count() in (0, 1)
    # the sections run 01 Highlights, 02 Top models, 03 The loop
    ix = page.evaluate("""() => [...document.querySelectorAll('#view > .card h2[data-ix]')]
      .map(h => h.dataset.ix + ' ' + h.textContent.trim())""")
    assert ix[:3] == ["01 Highlights", "02 Top models", "03 The loop"]
    assert page.errors == []


# ---------------------------------------------------------------------------
# the model page
# ---------------------------------------------------------------------------

def open_model(page, base, mid=MODEL):
    page.goto(base + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")


def test_the_model_page_leads_with_a_hero(live, page):
    open_model(page, live["base"])
    hero = page.locator("[data-model-hero]")
    r = page.evaluate(f"rankOf(DATA.models.find(m => m.id === '{MODEL}'))")
    assert hero.locator("[data-model-eyebrow]").text_content() == f"model · base · #{r['n']} of {r['of']}"
    assert hero.locator("h1.mtitle").text_content() == "good-750m"
    assert [c.get_attribute("data-hl") for c in hero.locator("[data-hl]").all()] == \
        ["params", "avg", "tasks"]
    assert hero.locator("[data-verdict='avg']").text_content().endswith(
        ("within noise.", "a real gap.", "no standard error to test it."))
    assert page.errors == []


def test_the_preliminary_note_appears_exactly_once(live, page):
    def soft(body):
        m = next(x for x in body["models"] if x["id"] == MODEL)
        for v in (m["judge"] or {}).get("tasks", {}).values():
            v["propose"] = {"ok": False, "hard": [], "caution": None,
                            "soft": [{"short": "the judge is not calibrated"}],
                            "why": "the judged suite is preliminary"}
    served(page, soft)
    open_model(page, live["base"])
    card = page.locator(".card", has=page.locator("h2", has_text="Judged free response"))
    card.wait_for()
    assert card.locator("[data-prelim-note]").count() == 1
    assert card.get_by_text("the topic page says what proposing would mean").count() == 1
    # the row's action is a small text button, one per row
    rows = card.locator("[data-judged-topics] tbody tr[data-topic]").count()
    assert card.locator("[data-judged-topics] a.propose").count() == rows
    assert card.locator("[data-judged-topics] a.propose").first.text_content() == "Propose →"
    assert page.errors == []


def test_topics_are_grouped_under_their_areas_weakest_first(live, page):
    open_model(page, live["base"])
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
    page.wait_for_selector("[data-judged-topics]")
    text = page.locator("#view").text_content()
    assert page.locator("[data-area-mean]").count() == 0
    assert "Judged average" not in text
    assert "demo only, not ranked" in text
    # the Overview: the weakest-topic card is one model's own topic, labelled
    # provisional, and no card holds an area mean
    page.goto(live["base"] + "/")
    page.wait_for_selector("[data-highlights] [data-hl='weakest']")
    assert "provisional, local judge" in verdict(page, "weakest")
    assert "mean" not in page.locator("[data-highlights]").text_content().lower()
    assert page.errors == []


def test_the_section_nav_lights_the_section_in_view(live, page):
    page.set_viewport_size({"width": 1280, "height": 800})
    open_model(page, live["base"])
    nav = page.locator("[data-model-nav]")
    ix = [a.get_attribute("data-ix") for a in nav.locator("a[data-nav]").all()]
    assert ix == [f"{i:02d}" for i in range(1, len(ix) + 1)]
    for sec in ("results", "provenance"):
        page.evaluate(f"document.getElementById('sec-{sec}').scrollIntoView({{block: 'start'}})")
        page.wait_for_selector(f"[data-model-nav] a[data-nav='{sec}'][aria-current='true']")
        assert nav.locator("a[aria-current='true']").count() == 1
    # it sticks under the bar
    top = nav.bounding_box()["y"]
    assert 50 <= top <= 60, top
    assert page.errors == []


def test_screenshots_for_the_pr(live, page):
    out = Path(__file__).resolve().parent / "_screens" / "phase11"
    out.mkdir(parents=True, exist_ok=True)
    for width in (1280, 400):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(live["base"] + "/")
        page.wait_for_selector("[data-highlights]")
        page.wait_for_timeout(300)
        page.locator("[data-highlights]").screenshot(path=out / f"overview-highlights-{width}-light.png")
        open_model(page, live["base"])
        page.wait_for_timeout(300)
        page.locator("[data-model-hero]").screenshot(path=out / f"model-hero-{width}-light.png")
    assert page.errors == []
