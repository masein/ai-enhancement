"""Playwright smoke against the frozen report built from the fixture.

The page is one file with the payload embedded, so file:// is the honest way
to load it — nothing is fetched in that mode. Every test collects console
errors and uncaught exceptions and fails on any. Screenshots land in
tests/_screens/ (gitignored; CI uploads them) so a PR can show the page in
light and dark at desktop and phone width.

    pytest -q -m dashboard        # needs: playwright install chromium
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import go_tab

pytestmark = pytest.mark.dashboard

# the tabs the FROZEN page has (the live one adds Training and Submit & Queue)
FROZEN_TABS = ["Overview", "Models", "Leaderboard", "Tasks", "Perplexity & Loss",
                "Provenance"]
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
        go_tab(self.page, label)                      # one of the six, or under More ▾
        self.page.wait_for_selector("#view > *")

    def selected_tab(self) -> str:
        # a tab under More shows on the More button itself: "Provenance ▾"
        return self.page.locator("#tabs button[aria-selected='true']").inner_text() \
            .removesuffix(" ▾")

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
        # every tab's hash is its own label's slug — one name per tab
        assert pg.evaluate("location.hash") == "#tab=" + {
            "Perplexity & Loss": "perplexity"}.get(label, label.lower())
    assert surface.errors == []


def test_old_hashes_still_land_where_they_used_to(surface):
    """A link someone pasted in a message last month must not silently drop
    the reader on Overview."""
    pg = surface.page
    for old_hash, label in (("runs", "Provenance"), ("evals", "Provenance"),
                            ("ppl", "Perplexity & Loss")):
        surface.open("#tab=" + old_hash)
        assert surface.selected_tab() == label, old_hash
    # and a hash that means nothing leaves you where you were, not blank
    surface.open("#tab=nonsense")
    assert pg.locator("#view > *").count() > 0
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


def test_the_model_page_leads_with_the_exam(surface):
    """The exam is the instrument, so it comes first; the multiple-choice
    results and the per-item diagnosis follow as the second opinion."""
    pg = surface.open(model_link("fx/good-750m"))
    heads = [h.strip() for h in pg.locator("#view .card h2").all_text_contents()]
    assert heads[0].startswith("good-750m") or heads[0] == ""      # the head card has no h2 title
    order = [h for h in heads if h]
    assert order.index("Judged free response — the exam") < order.index("Results")
    assert order.index("Results") < order.index("Diagnose")
    assert order[-1] == "Provenance"
    dx = pg.locator(".card", has=pg.locator("h2", has_text="Diagnose"))
    assert "The second opinion, free" in dx.text_content()
    assert surface.errors == []


def test_judged_section_and_the_control_sentence(surface, tree):
    pg = surface.open(model_link("fx/skewed-360m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert card.count() == 1
    text = card.text_content()
    assert "Counts." in text and "Cohen's κ" in text and "for judge stub/overlap-v1" in text
    assert "Canary steady." in text and "fixed scripts re-graded" in text
    assert card.locator("[data-canary='steady']").count() == 1
    assert "STUB grader" in text                                  # never mistaken for a judgement
    assert "Knew it, couldn't pick it" in text
    m = re.search(r"of the (\d+) control items this model got wrong as multiple choice, it answered (\d+)", text)
    assert m and int(m.group(2)) / int(m.group(1)) >= 0.5
    assert "By topic (0–4), weakest first — report half" in text
    assert "Score against answer length" in text and "economics" in text
    # topics, score-vs-length, the control — plus one per-criterion table for
    # every topic graded criterion by criterion, and one breakdown table per
    # metadata field those topics' banks carry
    crit = card.locator("table.jd[data-criteria-table]").count()
    breakdown = card.locator("table.jd[data-breakdown-table]").count()
    assert crit >= 1 and breakdown >= crit
    assert card.locator("table.jd[data-breakdown-table='acuity']").count() >= 1
    assert card.locator("table.jd").count() == 3 + crit + breakdown
    assert card.locator("table.jd[data-criteria-table='medicine & health']").count() == 1
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
    assert len(judged) >= 4 and any(h.startswith("Judged avg") for h in judged)
    assert any(h.startswith("economics") for h in judged)
    assert not any(h.startswith(("fr_", "exam_")) for h in heads)   # never as a task column
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
    assert "derived from mmlu diagnostics" in head.text_content()
    assert "never ranked" in head.text_content()

    surface.open(model_link("fx/good-750m-tuned-skill"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="What the training taught"))
    v = card.locator("[data-verdict]")
    assert v.get_attribute("data-verdict") == "skill" and "calm" in v.get_attribute("class")
    assert "The training taught the skill" in card.text_content()
    surface.open(model_link("fx/good-750m"))
    assert pg.locator(".card", has=pg.locator("h2", has_text="What the training taught")).count() == 0
    assert surface.errors == []


@pytest.fixture(scope="module")
def local_judged(tmp_path_factory) -> Path:
    """The fixture board with one model's exam graded by a local judge — the
    judge.json scripts/judge.py writes for JUDGE_PROVIDER=local."""
    import judge as jd
    import make_fixture
    root = tmp_path_factory.mktemp("local-judge")
    tree = make_fixture.build(root)
    d = tree["models"]["fx/good-750m"]["dir"]
    reqs, plan = jd.plan_requests(d, "gemma")
    results = jd.stub_results(reqs)
    # plant one critical safety failure on a criteria-graded item, so the page
    # has both wordings to show: the fixture's own answers trip none
    med = [m for m in plan["tasks"]["exam_medicine_health"] if m["half"] == "diagnose"]
    spec = jd.rubric_for("exam_medicine_health").criteria
    results[med[0]["cid"]] = type(results[med[0]["cid"]])(text=json.dumps({
        "flags": {fid: True for fid in jd.flag_ids(spec)},
        "criteria": {cid: 1.0 for cid in jd.criteria_ids(spec)},
        "justification": "told an emergency to wait until morning"}))
    plan["provisional"] = {"provisional": True,
                           "provisional_reason": "graded by a local model — not a pinned benchmark",
                           "base_url": "http://localhost:8000/v1", "served_model": "chat",
                           "weights": "google/gemma-4-E4B-it"}
    ident = {"provider": "local", "model": "chat", "id": "local/chat", "family": "chat"}
    jd.write_judge(d, jd.assemble(plan, results, ident, "local_0123456789ab",
                                  tree["out_dir"], 0.5, False, record=False))
    return make_fixture.frozen_report(root, root / "report.html")


def test_a_local_judge_is_greyed_labelled_and_never_ranked(browser, local_judged):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    s = Surface(ctx.new_page(), local_judged.as_uri())
    try:
        pg = s.open(model_link("fx/good-750m"))
        card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
        banner = card.locator("[data-provisional='judge']")
        assert banner.count() == 1
        text = banner.text_content()
        assert text.startswith("Provisional. Graded by a local model — not a pinned benchmark")
        assert "whose id cannot be pinned" in card.locator("p.sub").text_content()
        assert "chat at http://localhost:8000/v1 (weights google/gemma-4-E4B-it)" in text
        assert "Preliminary." in card.text_content() and "Judged average" not in card.text_content()
        # the second stamp, independent of the judge: a rubric its author has
        # not signed off. The fixture's medicine topic is graded by one.
        draft = card.locator("[data-rubric='draft']")
        assert draft.count() == 1
        assert "medicine & health is graded against a rubric its author has not signed off" \
            in draft.text_content()
        assert "changes its sha" in draft.text_content()
        # greyed: every topic row, in the muted colour rather than the text colour
        rows = card.locator("table.jd").first.locator("tbody tr")
        assert rows.count() > 0
        assert all("dim" in (rows.nth(i).get_attribute("class") or "") for i in range(rows.count()))
        style = "e => getComputedStyle(e).color"
        assert rows.first.locator("td").nth(1).evaluate(style) != card.locator("h2").evaluate(style)
        SCREENS.mkdir(exist_ok=True)
        for scheme in ("light", "dark"):
            pg.emulate_media(color_scheme=scheme)
            card.screenshot(path=SCREENS / f"local-judge-provisional-{scheme}.png")
        # never ranked: its judged cells on the board are blank, with the reason on hover
        s.open("#tab=leaderboard")
        row = pg.locator("table.lb tbody tr", has_text="good-750m").first
        assert "/4" not in row.text_content()
        assert any("graded by a local model" in (c.get_attribute("title") or "")
                   for c in row.locator("td").all())
        assert "were graded by a local model — not a pinned benchmark" in \
            pg.locator("#warnings").text_content()
        assert s.errors == []
    finally:
        ctx.close()


def test_a_criteria_graded_topic_shows_its_criteria_failures_and_acuities(browser, local_judged):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    s = Surface(ctx.new_page(), local_judged.as_uri())
    try:
        pg = s.open(model_link("fx/good-750m"))
        card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
        head = card.locator("[data-criteria='medicine & health']")
        assert head.count() == 1 and "by criterion (0–1), weakest first" in head.text_content()
        # every criterion in THIS topic's file has a row, weakest first
        table = card.locator("table.jd[data-criteria-table='medicine & health']")
        rows = table.locator("tr[data-criterion]")
        assert rows.count() == 15                      # the author's own file, verbatim
        means = [float(rows.nth(i).locator("td").nth(1).inner_text())
                 for i in range(rows.count())
                 if rows.nth(i).locator("td").nth(1).inner_text() != "—"]
        assert means == sorted(means)
        # the flag is stated in words, not only coloured, and says what it did
        flag = card.locator("[data-flag='critical_safety_failure'][data-flag-topic='medicine & health']")
        assert flag.count() == 1
        text = flag.text_content()
        assert text.startswith("Critical safety failure.")
        assert "were flagged" in text and "sets the whole score to 0" in text
        assert "applied here and not by the judge" in text
        # law's two flags say different things, because its file does
        law = card.locator("[data-flag-topic='law']")
        assert law.count() == 2
        assert "caps the whole score at 1 of 4" in law.nth(1).text_content()
        # a table per metadata field, acuity first, with a column per flag
        fields = card.locator("table.jd[data-breakdown-topic='medicine & health']")
        assert [fields.nth(i).get_attribute("data-breakdown-table")
                for i in range(fields.count())] == ["acuity", "difficulty", "intent"]
        acuity = fields.nth(0).locator("tr[data-value]")
        assert acuity.count() >= 3
        assert acuity.first.get_attribute("data-value") == "emergency"   # most severe first
        assert "Critical safety failure" in fields.nth(0).locator("thead").text_content()
        # and the difficulty table carries the question its author asked of it
        assert card.locator("[data-difficulty-note]").count() >= 1
        assert "only on basic questions" in card.locator("[data-difficulty-note]").first.text_content()
        SCREENS.mkdir(exist_ok=True)
        card.screenshot(path=SCREENS / "criteria-medicine.png")
        assert s.errors == []
    finally:
        ctx.close()


@pytest.fixture(scope="module")
def demo_report(tmp_path_factory) -> Path:
    """A real demo run's own page, built by the demo itself — the medicine
    bank, so the judged section has criteria to show."""
    import os
    import subprocess
    import sys
    root = tmp_path_factory.mktemp("demo-bench")
    env = {**os.environ, "BENCH_ROOT": str(root),
           "LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1",
           "JUDGE_PROVIDER": "fake", "JUDGE_MODEL": "fake-judge-20250101",
           "EXAM_PROVIDER": "", "EXAM_MODEL": ""}
    for k in ("LLM_API_KEY", "EXAM_API_KEY", "JUDGE_API_KEY"):
        env.pop(k, None)
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "demo_loop.py"), "--topic", "medicine & health",
         "--import", str(repo / "eval_tasks" / "fr" / "medicine_v2.json"),
         "--approver", "Dr. Hossein", "--sit", "stub", "--count", "4", "--keep"],
        capture_output=True, text=True, timeout=300, env=env, cwd=repo)
    assert r.returncode == 0, r.stdout + r.stderr
    page = root / "demo" / "report.html"
    assert page.is_file()
    return page


def test_the_demo_page_says_what_it_is_and_shows_the_criteria(browser, demo_report):
    """P5a: the first time a person can open what the demo produced. The
    judged section has to hold up on the page, not only in judge.json."""
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    s = Surface(ctx.new_page(), demo_report.as_uri())
    try:
        pg = s.open()
        banner = pg.locator(".pagebanner")
        assert banner.count() == 1
        assert "DEMO RUN" in banner.text_content()
        assert "nothing on this page is on the leaderboard" in banner.text_content()
        assert banner.locator("a").get_attribute("href") == "/"      # back to the real board
        pg.goto(demo_report.as_uri() + model_link("EleutherAI/pythia-160m"))
        pg.wait_for_selector("#view > *")
        card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
        text = card.text_content()
        # the two stamps, in words
        assert "Provisional." not in text                 # the fake judge is not local…
        assert "Draft rubric." in text                    # …but the rubric is still a draft
        assert "medicine & health is graded against a rubric its author has not signed off" in text
        # the criteria row, the flag in words, the tables by acuity and by
        # difficulty — the author's own file decides all three
        med = card.locator("table.jd[data-criteria-table='medicine & health']")
        assert card.locator("[data-criteria='medicine & health']").count() == 1
        assert med.locator("tr[data-criterion]").count() == 15
        assert card.locator("[data-flag='critical_safety_failure']").count() == 1
        assert "critical safety failure" in text.lower()
        assert card.locator("table.jd[data-breakdown-table='acuity'] tr[data-value]"
                            ).count() == 5
        assert card.locator("table.jd[data-breakdown-table='difficulty'] tr[data-value]"
                            ).count() == 3
        # the folded score is shown and not counted: 100 questions clear the
        #30-item floor, so the row is no longer greyed for being thin — the
        # whole suite is still preliminary, because nothing is calibrated
        assert "Preliminary." in text and "never ranked, never averaged" in text
        row = card.locator("tr[data-topic='medicine & health']")
        assert row.count() == 1 and "dim" not in (row.get_attribute("class") or "")
        assert "· under 30" not in row.text_content()
        SCREENS.mkdir(exist_ok=True)
        card.screenshot(path=SCREENS / "demo-judged-medicine.png")
        pg.goto(demo_report.as_uri())
        pg.wait_for_selector("#view > *")
        pg.screenshot(path=SCREENS / "demo-report-top.png", clip={"x": 0, "y": 0,
                                                                  "width": 1240, "height": 420})
        assert s.errors == []
    finally:
        ctx.close()


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


def test_the_models_tab_lists_every_model_and_filters_it(surface):
    """Phase 8e P6b: thirty-three models and no list was the complaint. The
    two filter rows that floated above the tabs live here now, where what
    they filter is on screen under them."""
    pg = surface.open("#tab=models")
    table = pg.locator("table.jd[data-models-table]")
    rows = table.locator("tbody tr[data-model-row]")
    n = rows.count()
    assert n == pg.evaluate("DATA.models.length") and n > 3
    assert pg.locator("[data-model-count]").first.get_attribute("data-model-count") == str(n)
    # the old floating filters are gone from the shell
    assert pg.locator("#kindSeg").count() == 0 and pg.locator("#srcSeg").count() == 0
    # every filter narrows this list, and says so by the count
    pg.locator("[data-filter='kind:instruct']").click()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length < n", arg=n)
    instruct = pg.locator("tbody tr[data-model-row]").count()
    assert 0 < instruct < n
    assert all("instruct" in pg.locator("tbody tr[data-model-row]").nth(i).text_content()
               for i in range(instruct))
    pg.locator("[data-filter='kind:all']").click()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length === n", arg=n)
    # a judged-run filter, because 'which of these sat the exam' is a question
    pg.locator("input[data-filter='judged']").check()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length <= n", arg=n)
    judged = pg.locator("tbody tr[data-model-row]").count()
    assert judged >= 1
    pg.locator("input[data-filter='judged']").uncheck()
    # search
    pg.get_by_label("filter models").fill("good")
    pg.wait_for_function("() => document.querySelectorAll('tr[data-model-row]').length >= 1")
    assert all("good" in pg.locator("tbody tr[data-model-row]").nth(i).text_content()
               for i in range(pg.locator("tbody tr[data-model-row]").count()))
    pg.get_by_label("filter models").fill("")
    # sort on a column, both ways
    pg.locator("th[data-sort='params']").click()
    first = pg.locator("tbody tr[data-model-row]").first.get_attribute("data-model-row")
    pg.locator("th[data-sort='params']").click()
    assert pg.locator("tbody tr[data-model-row]").first.get_attribute("data-model-row") != first
    SCREENS.mkdir(exist_ok=True)
    pg.screenshot(path=SCREENS / "models-tab.png", full_page=True)
    assert surface.errors == []


def test_the_leaderboards_compare_ticks_are_the_only_ones(surface):
    """The radar belongs to the Leaderboard. Its ticks pick up to five models,
    say how many slots are used, redraw the radar, and drop the oldest when a
    sixth arrives. The Models tab has no tick at all: two tables sharing one
    selection is what broke this."""
    pg = surface.open("#tab=models")
    assert pg.locator("table.jd[data-models-table] input[type=checkbox][aria-label^='compare']"
                      ).count() == 0
    surface.tab("Leaderboard")
    boxes = pg.locator("table.lb tbody input[type=checkbox]")
    assert boxes.count() > 5
    radar = pg.locator(".card", has=pg.locator("h2", has_text="Capability profile"))
    assert radar.count() == 1 and radar.locator("svg").count() == 1
    assert "comparing 5 of 5 slots" in radar.text_content()   # the default profile
    drawn = radar.locator("svg").inner_html()
    assert len(drawn) > 200
    # untick one: the count follows and the radar is redrawn without it
    first_on = next(i for i in range(boxes.count()) if boxes.nth(i).is_checked())
    boxes.nth(first_on).uncheck()
    assert "comparing 4 of 5 slots" in radar.text_content()
    assert radar.locator("svg").inner_html() != drawn
    # tick one that was not in the set: back to five, and it is on the radar
    off = next(i for i in range(boxes.count()) if not boxes.nth(i).is_checked())
    mid = pg.locator("table.lb tbody tr").nth(off).locator("td.model").first.get_attribute(
        "data-model")
    boxes.nth(off).check()
    assert "comparing 5 of 5 slots" in radar.text_content()
    assert "full, the next tick replaces the oldest" in radar.text_content()
    assert mid.split("/")[-1] in radar.text_content()          # it is on the radar now
    # a sixth lands and the oldest leaves — the newest click always wins
    spare = next(i for i in range(boxes.count()) if not boxes.nth(i).is_checked())
    boxes.nth(spare).check()
    assert "comparing 5 of 5 slots" in radar.text_content()
    assert "dropped" in radar.text_content()
    assert sum(1 for i in range(boxes.count()) if boxes.nth(i).is_checked()) == 5
    assert surface.errors == []


def DATA_MODELS(pg):
    return pg.evaluate("DATA.models.map(m => ({id: m.id, name: m.name}))")


def test_a_model_page_has_a_sub_nav_and_its_sections(surface):
    pg = surface.open(model_link("fx/good-750m"))
    nav = pg.locator("[data-model-nav]")
    assert nav.count() == 1
    labels = [nav.locator("a[data-nav]").nth(i).text_content()
              for i in range(nav.locator("a[data-nav]").count())]
    assert "Judged" in labels and "Provenance" in labels
    for a in labels:
        anchor = nav.locator("a[data-nav]", has_text=a).first.get_attribute("data-nav")
        assert pg.locator(f"#sec-{anchor}").count() == 1
    assert surface.errors == []


@pytest.fixture(scope="module")
def zero_criterion(tmp_path_factory) -> Path:
    """A board where one criterion scored exactly 0 on every answer it
    applied to — physics's "Uncertainty and calibration", 5 of 100."""
    import judge as jd
    import make_fixture
    root = tmp_path_factory.mktemp("zero-crit")
    tree = make_fixture.build(root)
    d = tree["models"]["fx/good-750m"]["dir"]
    j = json.loads((d / "judge.json").read_text(encoding="utf-8"))
    task = "exam_medicine_health"
    spec = jd.rubric_for(task).criteria
    zero = jd.criteria_ids(spec)[0]
    t = j["tasks"][task]
    for it in t["items"]:
        if it.get("criteria"):
            it["criteria"][zero] = 0.0
    t["criteria_mean"][zero] = 0.0
    (d / "judge.json").write_text(json.dumps(j), encoding="utf-8")
    return make_fixture.frozen_report(root, root / "report.html")


def test_a_criterion_that_scored_zero_says_zero(browser, zero_criterion):
    """It rendered as an empty cell — and weakest-first put that empty row at
    the top of the table, so the worst finding looked like a missing one."""
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    s = Surface(ctx.new_page(), zero_criterion.as_uri())
    try:
        pg = s.open(model_link("fx/good-750m"))
        table = pg.locator("table.jd[data-criteria-table='medicine & health']")
        first = table.locator("tr[data-criterion]").first
        mean = first.locator("td").nth(1)
        assert mean.inner_text().strip() == "0.00"        # not "", not "0"
        assert first.locator("td").nth(2).inner_text().strip() != "0"   # it was scored
        # the bar for zero is a hairline, not an empty track
        bar = first.locator("[data-bar]")
        assert bar.count() == 1 and bar.get_attribute("data-bar") == "0.00"
        assert bar.evaluate("e => e.getBoundingClientRect().width") > 0
        # and every other row still prints two decimals
        others = table.locator("tr[data-criterion] td.num:nth-child(2)")
        for i in range(min(4, others.count())):
            assert re.fullmatch(r"\d\.\d\d", others.nth(i).inner_text().strip())
        assert s.errors == []
    finally:
        ctx.close()
