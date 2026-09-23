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

from conftest import go_tab, show_all_columns

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
    ctx = browser.new_context(viewport={"width": width, "height": 900}, reduced_motion="reduce")
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
    assert mmlu.locator("details.dxex summary").text_content().endswith("(practice half only)")
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
    # gap (econometrics) puts Economics lowest — the greyed one-subject rows
    # may land anywhere, which is exactly why they are greyed
    scores = [float(x.rstrip("%")) for x in cats.locator("> summary > .num").all_text_contents()]
    assert scores == sorted(scores)
    solid = mmlu.locator("details.dxcat:not(.dim) .dxcname").all_text_contents()
    assert solid[0] == "Economics"
    # the noise floor is visible: one-subject categories are greyed, two-subject ones are not
    dim = mmlu.locator("details.dxcat.dim .dxcname").all_text_contents()
    # the fixture's four one-subject topics: Mathematics & Statistics, Law,
    # Political Science & International Relations, Ethics & Religion
    assert "Economics" not in dim and "Medicine & Clinical Health" not in dim and len(dim) == 4
    assert "under 30, noise" in mmlu.locator("details.dxcat.dim").first.text_content()
    # subjects live under the category, not beside it, until you open one
    econ = mmlu.locator("details.dxcat[data-cat='Economics']")
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


def test_leaderboard_knowledge_shows_mmlu_by_area_and_by_topic(surface, diag):
    """11c: the Knowledge chip replaced the "MMLU by category" view — MMLU, its
    eight areas, and the per-topic columns one tick away under Columns."""
    pg = surface.open("#tab=leaderboard")
    pg.locator("[data-chip='knowledge']").click()
    pg.wait_for_selector("table.lb thead th[data-area]")
    # 11e: the areas some model here has a number for (the fixture's MMLU
    # subjects reach five of the eight)
    assert pg.locator("table.lb thead th[data-area]").count() == pg.evaluate(
        "Object.keys(DATA.meta.areas).filter(a => visible().some(m => areaMmlu(m, a))).length")
    assert pg.locator("table.lb thead th[data-area]").count() >= 1
    assert pg.locator("table.lb thead th[data-task='mmlu']").count() == 1
    # the per-topic columns, one tick away: a topic under 30 items is greyed
    pg.locator("[data-columns-menu]").click()
    pg.locator("#pop-columns [data-column-group-all='cats']").click()
    pg.wait_for_selector("table.lb thead th[data-col='cat:Economics']")
    pg.keyboard.press("Escape")
    table = pg.locator("table.lb")
    assert table.locator("td.dim").count() > 0 and table.locator("td.lead").count() > 0
    # a model without a diagnosis on file has no area to show, and says so with a dash
    rows = pg.evaluate("""() => [...document.querySelectorAll('table.lb tbody tr[data-lb-row]')]
      .filter(tr => !mmluCats(DATA.models.find(m => m.id === tr.dataset.lbRow))).length""")
    assert rows >= 1
    # the control column carries its warning in All tasks
    pg.locator("[data-chip='all']").click()
    show_all_columns(pg)                             # six task columns by default (9c)
    th = pg.locator("table.lb thead th[data-task='mmlu_perm']")
    assert "CONTROL" in th.get_attribute("data-tip")          # 11f: the name's tooltip
    assert surface.errors == []


def test_the_model_page_leads_with_the_exam(surface):
    """The exam is the instrument, so it comes first; the multiple-choice
    results and the per-item diagnosis follow as the second opinion."""
    pg = surface.open(model_link("fx/good-750m"))
    # 11d: the name is the hero's h1; the first section under it is the exam
    assert pg.locator("[data-model-hero] h1").text_content() == "good-750m"
    heads = [h.strip() for h in pg.locator("#view .card h2").all_text_contents()]
    order = [h for h in heads if h]
    assert order[0] == "Judged free response — the exam"
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
    # 11h: plain words — κ is "agreement with a person"
    assert "Counts." in text and "Agreement with a person:" in text and "stub/overlap-v1" in text
    assert "Canary steady." in text and "fixed scripts re-graded" in text
    assert card.locator("[data-canary='steady']").count() == 1
    assert "STUB grader" in text                                  # never mistaken for a judgement
    assert "Knew it, couldn't pick it" in text
    m = re.search(r"of the (\d+) control items this model got wrong as multiple choice, it answered (\d+)", text)
    assert m and int(m.group(2)) / int(m.group(1)) >= 0.5
    assert "By topic (0–4), weakest first within each area — hidden questions" in text
    assert "Score against answer length" in text and "Economics" in text
    # topics, score-vs-length and the control, and nothing else: the
    # by-criterion block, its picker and its breakdown tables went in 11m
    # (masein did not want them); each answer card keeps its criteria strip
    assert card.locator("table.jd").count() == 3
    assert card.locator("[data-criteria-table], [data-breakdown-table], "
                        "[data-topic-switch]").count() == 0
    surface.open(model_link("fx/chance-160m"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert "Didn't know it either way" in card.text_content()
    surface.open(model_link(tree["nodiag"]))
    card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
    assert "Not judged" in card.text_content()
    assert surface.errors == []


def test_judged_columns_appear_once_calibrated(surface):
    pg = surface.open("#tab=leaderboard")
    show_all_columns(pg)
    # 11f: one line of names; the κ and the 0–4 scale are each judged
    # column's tooltip
    ths = pg.locator("table.lb thead tr.names th").evaluate_all(
        "ts => ts.map(t => [t.textContent.trim(), t.dataset.tip || ''])")
    heads = [h for h, _ in ths]
    judged = [(h, tip) for h, tip in ths if "0–4" in tip]
    assert len(judged) >= 4 and any(h == "Judged" for h, _ in judged)
    assert any(h.startswith("Economics") and "agreement" in tip for h, tip in judged)
    assert not any(h.startswith(("fr_", "exam_")) for h in heads)   # never as a task column
    row = pg.locator("table.lb tbody tr[data-lb-row='fx/good-750m']")
    assert float(row.locator("[data-judged-avg]").text_content()) >= 0
    assert surface.errors == []


def test_what_the_training_taught(surface, tree):
    pg = surface.open(model_link("fx/good-750m-tuned-test"))
    card = pg.locator(".card", has=pg.locator("h2", has_text="What the training taught"))
    assert card.count() == 1
    text = card.text_content()
    assert "before — good-750m" in text and "after — good-750m-tuned-test" in text
    assert "hidden half (never in the training data)" in text
    v = card.locator("[data-verdict]")
    assert v.get_attribute("data-verdict") == "test" and "warn" in v.get_attribute("class")
    assert "The training taught the test" in text and "Ratio of the two changes" in text
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


# The five topics' files as first delivered, retired when the 37-topic exam
# replaced them. Every rubric delivered with the 37 is signed off and has one
# zeroing flag, so a page test about a DRAFT rubric's stamp, or about a file
# whose flags do different things (law v2: one zeroes, one caps at 1 of 4),
# grades that topic with the retired pair instead — installed in
# $BENCH_ROOT/rubrics, where the page's rubric upload puts a file and where
# the judge looks first.
RETIRED_RUBRICS = Path(__file__).resolve().parents[1] / "eval_tasks" / "fr" / "retired" / "rubrics"


def install_retired_rubrics(store: Path, pairs: dict[str, str]) -> None:
    """pairs: the new topic's slug -> the retired file's name."""
    store.mkdir(parents=True, exist_ok=True)
    for slug, old in pairs.items():
        for suffix in (".md", ".criteria.json"):
            (store / f"{slug}{suffix}").write_bytes(
                (RETIRED_RUBRICS / f"{old}{suffix}").read_bytes())


@pytest.fixture(scope="module")
def local_judged(tmp_path_factory) -> Path:
    """The fixture board with one model's exam graded by a local judge — the
    judge.json scripts/judge.py writes for JUDGE_PROVIDER=local. Medicine and
    law are graded by their retired pairs (see RETIRED_RUBRICS): both drafts,
    and law's with a capping flag beside its zeroing one."""
    import judge as jd
    import make_fixture
    from service import config
    root = tmp_path_factory.mktemp("local-judge")
    install_retired_rubrics(root / "rubrics", {"medicine_clinical_health": "medicine_health",
                                               "law": "law"})
    saved = config.BENCH_ROOT
    config.BENCH_ROOT = root
    try:
        tree = make_fixture.build(root)
        d = tree["models"]["fx/good-750m"]["dir"]
        reqs, plan = jd.plan_requests(d, "gemma")
        results = jd.stub_results(reqs)
        # plant one critical safety failure on a criteria-graded item, so the page
        # has both wordings to show: the fixture's own answers trip none
        med = [m for m in plan["tasks"]["exam_medicine_clinical_health"]
               if m["half"] == "diagnose"]
        spec = jd.rubric_for("exam_medicine_clinical_health").criteria
        results[med[0]["cid"]] = type(results[med[0]["cid"]])(text=json.dumps({
            "flags": {fid: True for fid in jd.flag_ids(spec)},
            "criteria": {cid: 1.0 for cid in jd.criteria_ids(spec)},
            "justification": "told an emergency to wait until morning"}))
        plan["provisional"] = {"provisional": True,
                               "provisional_reason": "graded by a local model — not a pinned "
                                                     "benchmark",
                               "base_url": "http://localhost:8000/v1", "served_model": "chat",
                               "weights": "google/gemma-4-E4B-it"}
        ident = {"provider": "local", "model": "chat", "id": "local/chat", "family": "chat"}
        jd.write_judge(d, jd.assemble(plan, results, ident, "local_0123456789ab",
                                      tree["out_dir"], 0.5, False, record=False))
        return make_fixture.frozen_report(root, root / "report.html")
    finally:
        config.BENCH_ROOT = saved


def test_a_local_judge_is_greyed_labelled_and_never_ranked(browser, local_judged):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900}, reduced_motion="reduce")
    s = Surface(ctx.new_page(), local_judged.as_uri())
    try:
        pg = s.open(model_link("fx/good-750m"))
        card = pg.locator(".card", has=pg.locator("h2", has_text="Judged free response"))
        banner = card.locator("[data-provisional='judge']")
        assert banner.count() == 1
        text = banner.text_content()
        assert text.startswith("Provisional. Graded by a local model — not a pinned benchmark")
        # 11h: the card says what it is in one line; how it works is behind a click
        assert "whose id cannot be pinned" in card.locator("[data-how-judged]").text_content()
        assert "chat at http://localhost:8000/v1 (weights google/gemma-4-E4B-it)" in text
        assert "Preliminary." in card.text_content() and "Judged average" not in card.text_content()
        # the second stamp, independent of the judge: a rubric its author has
        # not signed off. This board's medicine topic is graded by one.
        draft = card.locator("[data-rubric='draft']")
        assert draft.count() == 1
        assert "Medicine & Clinical Health is graded against a rubric its author has not " \
            "signed off" in draft.text_content()
        assert "changes its sha" in draft.text_content()
        # greyed: every topic row, in the muted colour rather than the text colour
        # the topic rows, not the area headers between them (11d)
        rows = card.locator("table.jd").first.locator("tbody tr[data-topic]")
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
        show_all_columns(pg)
        row = pg.locator("table.lb tbody tr", has_text="good-750m").first
        assert "/4" not in row.text_content()
        assert any("graded by a local model" in (c.get_attribute("title") or "")
                   for c in row.locator("td").all())
        assert "were graded by a local model — not a pinned benchmark" in \
            pg.locator("#warnings").text_content()
        assert s.errors == []
    finally:
        ctx.close()




@pytest.fixture(scope="module")
def demo_report(tmp_path_factory) -> Path:
    """A real demo run's own page, built by the demo itself — the medicine
    bank, so the judged section has criteria to show. The retired medicine &
    health bank and its DRAFT rubric pair (see RETIRED_RUBRICS), so the page
    has both a draft stamp and five acuities to show; installed in the demo's
    own $BENCH_ROOT/rubrics, which is $BENCH_ROOT/demo/rubrics."""
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
    install_retired_rubrics(root / "demo" / "rubrics",
                            {"medicine_clinical_health": "medicine_health"})
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "demo_loop.py"),
         "--topic", "Medicine & Clinical Health",
         "--import", str(repo / "eval_tasks" / "fr" / "retired" / "medicine_v2.json"),
         "--approver", "Dr. Hossein", "--sit", "stub", "--count", "4", "--keep"],
        capture_output=True, text=True, timeout=300, env=env, cwd=repo)
    assert r.returncode == 0, r.stdout + r.stderr
    page = root / "demo" / "report.html"
    assert page.is_file()
    return page


def test_the_demo_page_says_what_it_is_and_what_its_scores_are(browser, demo_report):
    """P5a: the first time a person can open what the demo produced. The
    judged section has to hold up on the page, not only in judge.json."""
    ctx = browser.new_context(viewport={"width": 1240, "height": 900}, reduced_motion="reduce")
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
        assert "Medicine & Clinical Health is graded against a rubric its author has not " \
            "signed off" in text
        # 11m: no by-criterion block, no breakdown tables — the criteria are
        # in each answer card on the live board, which a static report has not
        assert card.locator("[data-criteria-table], [data-breakdown-table]").count() == 0
        # the folded score is shown and not counted: 100 questions clear the
        #30-item floor, so the row is no longer greyed for being thin — the
        # whole suite is still preliminary, because nothing is calibrated
        assert "Preliminary." in text and "never ranked, never averaged" in text
        row = card.locator("tr[data-topic='Medicine & Clinical Health']")
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
            pg.locator("[data-chip='knowledge']").click()
            pg.wait_for_selector("table.lb thead th[data-area]")
            pg.screenshot(path=SCREENS / f"leaderboard-knowledge-{scheme}-{width}.png",
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
    # 11c: the filters are the Leaderboard's "Label: Value ▾" pills
    pg.locator("#pill-mkind").click()
    pg.locator("#pop-mkind [data-choice='instruct']").click()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length < n", arg=n)
    instruct = pg.locator("tbody tr[data-model-row]").count()
    assert 0 < instruct < n
    assert all("instruct" in pg.locator("tbody tr[data-model-row]").nth(i).text_content()
               for i in range(instruct))
    assert pg.locator("#pill-mkind").text_content().startswith("Kind: instruct")
    pg.locator("#pill-mkind").click()
    pg.locator("#pop-mkind [data-choice='all']").click()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length === n", arg=n)
    # a judged-run filter, because 'which of these sat the exam' is a question
    pg.locator("#pill-mshow").click()
    pg.locator("#pop-mshow input[data-filter='judged']").check()
    pg.wait_for_function("n => document.querySelectorAll('tr[data-model-row]').length <= n", arg=n)
    judged = pg.locator("tbody tr[data-model-row]").count()
    assert judged >= 1
    assert pg.locator("#pill-mshow").get_attribute("data-show-filters") == "judged"
    pg.locator("#pop-mshow input[data-filter='judged']").uncheck()
    pg.keyboard.press("Escape")
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


def test_the_radars_model_chips_are_the_only_comparison(surface):
    """11c: the radar's chips replaced the compare column. Up to five models,
    added from a search and removed with ×; a sixth waits for a free slot,
    and says so. The Models tab has no tick at all."""
    pg = surface.open("#tab=models")
    assert pg.locator("table.jd[data-models-table] input[type=checkbox][aria-label^='compare']"
                      ).count() == 0
    surface.tab("Leaderboard")
    assert pg.locator("table.lb thead th.cmp").count() == 0
    assert pg.locator("table.lb tbody input[type=checkbox]").count() == 0
    radar = pg.locator("[data-radar]")
    assert "Add up to 5 models" in radar.locator("[data-radar-prompt]").text_content()
    ids = [m["id"] for m in DATA_MODELS(pg)][:6]
    for mid in ids[:5]:
        pg.locator("#pill-radar-add").click()
        pg.locator(f"#pop-radar-add [data-radar-pick='{mid}']").click()
        pg.wait_for_selector(f"[data-radar-chip='{mid}']")
    assert radar.locator("[data-radar-chip]").count() == 5
    drawn = radar.locator("svg.radar").inner_html()
    assert len(drawn) > 200
    # full: Add is off and says why, and so is a row's own Add to radar
    add = pg.locator("#pill-radar-add")
    assert add.is_disabled() and "remove one first" in add.get_attribute("title")
    pg.locator(f"tr[data-lb-row='{ids[5]}'] button.disclose").click()
    row_add = pg.locator(f"[data-add-radar='{ids[5]}']")
    assert row_add.is_disabled() and "remove one first" in row_add.get_attribute("title")
    # remove one: the radar is redrawn without it, and Add comes back
    radar.locator(f"[data-radar-chip='{ids[0]}'] button.xbtn").click()
    pg.wait_for_selector(f"[data-radar-chip='{ids[0]}']", state="detached")
    assert radar.locator("svg.radar").inner_html() != drawn
    assert pg.locator("#pill-radar-add").is_enabled()
    pg.locator(f"[data-add-radar='{ids[5]}']").click()
    pg.wait_for_selector(f"[data-radar-chip='{ids[5]}']")
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


