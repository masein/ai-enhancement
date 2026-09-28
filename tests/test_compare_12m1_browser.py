"""12m.1 on the page: Compare — any two to eight models on one table,
benchmarks down the side, grouped by how each number was measured, and only
like compared with like; the shape chart one method at a time, with real
names, drawing only shared axes; and Benchmarks ▸ Standard following the
same Models ▾ choice, with up to three highlighted. With them, two fixes to
the same rows: a GGUF's own row is its "as built" results alone, and the run
lists name a GGUF run with its setup and every suite by the board's name."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import gguf_data as gd
import gguf_worker as gw
from conftest import set_name
from fake_openai import FakeServer
from service import config, db, runner
from test_gguf_12f3 import docs_of, fake_binary

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12m1"
PHONE = "served/k4-LDA"
BASE = "Qwen/Qwen3.6-35B-A3B"
GOOD, SMALL, SKEWED = "fx/good-750m", "fx/below-135m-it", "fx/skewed-360m"
LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def board(live, tmp_path_factory):
    """the phone build — served, its GGUF measured as built and in lookahead,
    its Everyday answered through its server — and a GGUF file measured in
    lookahead only"""
    import service.app as appmod
    tmp = tmp_path_factory.mktemp("cmp")
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    lda, lone = tmp / "lda.gguf", tmp / "lone.gguf"
    lda.write_bytes(b"GGUF" + b"\1" * 400)
    lone.write_bytes(b"GGUF" + b"\2" * 400)
    fake = FakeServer()
    api(live, "/api/served", {"name": "k4-LDA", "base_url": fake.base, "based_on": BASE,
                              "how": "llama.cpp fork, k=4 + LDA", "thinking": "off", "phone": True,
                              "gguf_path": str(lda), "gguf_setups": LOOK, "by": "masein"})
    lone_m = api(live, "/api/gguf/models", {"name": "Qwen3.6 k=8", "path": str(lone),
                                            "based_on": BASE, "how": "k=8", "setups": LOOK,
                                            "by": "masein"})["model"]
    binary = fake_binary(tmp)
    gw._stop["why"] = ""
    os.environ["FAKE_PPL_ACC_LOOKAHEAD"] = "0.8"
    look = next(x["id"] for x in lone_m["setups"] if x["name"] == "lookahead 1")
    for body in ({"model": PHONE, "by": "masein"},
                 {"model": lone_m["id"], "setups": [look], "by": "masein"}):
        for _ in api(live, "/api/gguf/runs", body)["ids"]:
            gw.Worker(config.RESULTS_ROOT, binary, poll=0.05).once()
    saved = runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    config.JUDGE_MODEL = "stub"
    try:
        sid = db.add(PHONE, "instruct", "everyday", "masein", "")
        runner.run_submission(db.get(sid))
    finally:
        runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL = saved
    api(live, "/api/submissions")
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - 120}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"lone": lone_m["id"]}
    fake.close()
    os.environ.pop("FAKE_PPL_ACC_LOOKAHEAD", None)


def served(page, edit):
    """the payload as the server sends it, edited — a cell with another shot
    count, two scores with set errors"""
    def handle(route):
        r = route.fetch()
        body = r.json()
        edit(body)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/results*", handle)


def compare(page, live, ids, width=1400, scheme="light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=compare&m=" + ",".join(quote(i, safe="") for i in ids))
    page.wait_for_selector("[data-compare]")


def tip(loc) -> list[str]:
    return json.loads(loc.get_attribute("data-tip"))


# ---------------------------------------------------------------------------
# the Compare table
# ---------------------------------------------------------------------------

def test_compare_groups_by_method_and_ranks_only_like_with_like(live, page, board):
    def edit(b):
        # one model's MMLU asked 0-shot: another method, in the same row
        b["cells"]["mmlu"][SKEWED].update(shots=0)
        # and one asked 50 of the hidden Everyday questions so far
        e = b["everyday"]["models"]
        e[SMALL] = {**e[GOOD], "passed": 30, "total": 50, "unasked": 129, "provisional": False}
    served(page, edit)
    compare(page, live, [GOOD, SMALL, SKEWED, PHONE])
    assert page.locator("[data-cmp-model]").evaluate_all("xs => xs.map(x => x.dataset.cmpModel)") \
        == [GOOD, SMALL, SKEWED, PHONE]
    assert page.locator(f"[data-cmp-kind='{PHONE}']").inner_text().startswith(" · phone build")
    assert page.locator(f"[data-cmp-kind='{GOOD}']").inner_text() == " · base · 750M"
    groups = page.locator("[data-cmp-group]").evaluate_all("xs => xs.map(x => x.dataset.cmpGroup)")
    assert groups[:2] == ["standard", "gguf"] and "everyday" in groups and "exam" in groups
    row = page.locator("[data-cmp-row='mmlu']")
    assert row.locator("[data-cmp-main='mmlu']").inner_text() == "lm_eval · 5-shot"
    off = row.locator(f"[data-cmp-cell='{SKEWED}']")
    assert "cmp-off" in off.get_attribute("class") and off.get_attribute("data-best") is None
    assert "lm_eval · 0-shot" in off.inner_text()
    assert "measured another way: not ranked against this row" in tip(off)
    # the best of the 5-shot cells is bold, and only one of them
    vals = {m: row.locator(f"[data-cmp-cell='{m}']") for m in (GOOD, SMALL)}
    best = max(vals, key=lambda m: float(tip(vals[m])[1].split()[0]))
    assert vals[best].get_attribute("data-best") == "1" and vals[best].locator("b").count() == 1
    assert row.locator("[data-best]").count() == 1
    # the phone build has no lm_eval MMLU
    assert row.locator(f"[data-cmp-cell='{PHONE}']").inner_text() == "not measured"
    # GGUF: only the phone build has it, so the group folds, saying so
    g = page.locator("[data-cmp-group='gguf']")
    assert "1 of 4 models measured" in g.inner_text()
    assert page.locator("[data-cmp-row='gguf:mmlu']").count() == 0
    g.locator("[data-cmp-fold]").click()
    page.wait_for_selector("[data-cmp-row='gguf:mmlu']")
    # Everyday: its method is the count its scores are over — 12n.1: the
    # hidden half's (179), from the result, never the bank's 340 — and a
    # model not asked them all is another method
    E = page.evaluate("evdHidden()")
    assert page.locator("[data-cmp-main='evd']").inner_text() == f"{E} questions, our checks"
    part = page.locator(f"[data-cmp-row='evd'] [data-cmp-cell='{SMALL}']")
    assert "cmp-off" in part.get_attribute("class") and " scored, our checks" in part.inner_text()
    shot(page.locator("[data-compare]"), "compare-1400-light.png")
    assert page.errors == []


def test_two_models_get_a_delta_by_the_z_test(live, page, board):
    def edit(b):
        for t, (a, c) in {"mmlu": (0.600, 0.572), "hellaswag": (0.600, 0.616)}.items():
            b["cells"][t][GOOD].update(v=a, se=0.01, shots=5)
            b["cells"][t][SKEWED].update(v=c, se=0.01, shots=5)
    served(page, edit)
    compare(page, live, [GOOD, SKEWED])
    assert page.locator("[data-cmp-delta-head]").inner_text() == "Δ"
    assert page.locator("[data-cmp-delta='mmlu']").inner_text() == "+2.8 · clear"
    assert page.locator("[data-cmp-delta='hellaswag']").inner_text() == \
        "−1.6 · not a clear difference"
    # three models: no Δ
    compare(page, live, [GOOD, SKEWED, SMALL])
    assert page.locator("[data-cmp-delta-head], [data-cmp-delta]").count() == 0
    assert page.errors == []


def test_trust_and_safety_is_its_own_group_and_over_refusal_is_never_best(live, page, board):
    def edit(b):
        for mid, (dna, over) in {GOOD: (0.9, 0.2), SKEWED: (0.8, 0.05)}.items():
            m = next(x for x in b["models"] if x["id"] == mid)
            m["trust"] = {"refuses": {"v": dna, "n": int(dna * 20), "of": 20},
                          "xstest": None, "over": {"v": over, "n": int(over * 20), "of": 20},
                          "fair": None, "waiting": 0, "provisional": False}
    served(page, edit)
    compare(page, live, [GOOD, SKEWED])
    assert page.locator("[data-cmp-main='trust:dna']").inner_text() == "judged 0–2"
    over = page.locator("[data-cmp-row='trust:over']")
    assert over.locator("[data-cmp-lower]").inner_text() == " · lower is better"
    assert over.locator("[data-best]").count() == 0
    assert page.locator("[data-cmp-row='trust:dna'] [data-best]").count() == 1
    assert page.errors == []


def test_compare_is_reached_from_ticks_the_models_menu_and_the_model_page(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-table]")
    assert page.locator("[data-cmp-go]").count() == 0
    for mid in (GOOD, SKEWED):
        page.locator(f"[data-cmp-tick='{mid}']").check()
    go = page.locator("[data-cmp-go]")
    assert go.inner_text() == "Compare 2 ▸"
    go.click()
    page.wait_for_selector("[data-compare='2']")
    assert "view=compare&m=" + quote(GOOD, safe="") in page.evaluate("location.hash")
    # Back returns to the table
    page.go_back()
    page.wait_for_selector("[data-lb-table]")
    # the Models menu: Compare these
    page.locator("#pill-models").click()
    page.locator("[data-models-clear]").click()
    for mid in (GOOD, SMALL, PHONE):
        page.locator(f"#pop-models [data-model-pick='{mid}']").check()
    page.locator("[data-models-compare]").click()
    page.wait_for_selector("[data-compare='3']")
    # the model page: Compare with…, then Add a model
    page.goto(live["base"] + "/#model=" + quote(PHONE, safe=""))
    page.locator(f"[data-compare-with='{PHONE}']").click()
    page.wait_for_selector("[data-compare='1']")
    assert page.locator("[data-cmp-few]").inner_text() == \
        "Add at least one more model to compare: Add a model ▾."
    page.locator("#pill-cmp-add").click()
    page.locator(f"#pop-cmp-add [data-cmp-pick='{GOOD}']").click()
    page.wait_for_selector("[data-compare='2']")
    assert page.errors == []


def test_a_comparison_is_saved_as_a_view(live, page, board):
    compare(page, live, [GOOD, PHONE])
    set_name(page, "masein")
    page.locator("[data-save-view]").click()
    page.fill("[data-view-name]", "Phone vs good")
    page.locator("[data-action='view-save']").click()
    page.wait_for_selector("[data-toast]")
    page.goto(live["base"] + "/#tab=models")
    chip = page.locator("[data-saved-view]", has_text="Phone vs good")
    chip.click()
    page.wait_for_selector("[data-compare='2']")
    assert page.locator("[data-cmp-chip]").evaluate_all("xs => xs.map(x => x.dataset.cmpChip)") \
        == [GOOD, PHONE]
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", [(1400, "dark"), (400, "light"), (400, "dark")])
def test_the_screens(live, page, board, width, scheme):
    compare(page, live, [GOOD, SKEWED, PHONE], width, scheme)
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth + 1")
    shot(page, f"compare-{width}-{scheme}.png", full_page=width == 400)
    assert page.errors == []


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def test_the_shape_takes_one_method_draws_shared_axes_and_names_them(live, page, board):
    compare(page, live, [GOOD, SKEWED])
    card = page.locator("[data-shape-card='cmp']")
    assert card.locator("[data-shape-src='tasks']").get_attribute("aria-pressed") == "true"
    labels = card.locator("svg.radar text").evaluate_all("xs => xs.map(x => x.textContent)")
    assert "MMLU" in labels and "HellaSwag" in labels and "mmlu" not in labels
    assert not [t for t in labels if "_" in t]                     # never a task id
    # the phone build with its base: no Standard it has, so its own kinds —
    # Everyday groups shared with good-750m
    compare(page, live, [GOOD, PHONE])
    card = page.locator("[data-shape-card='cmp']")
    assert card.locator("[data-shape-src='everyday']").get_attribute("aria-pressed") == "true"
    groups = page.evaluate("evdGroups().map(([, l]) => l)")
    assert set(card.locator("svg.radar text").evaluate_all("xs => xs.map(x => x.textContent)")) <= set(groups)
    # alone, its own axes: its GGUF, before its Everyday
    compare(page, live, [PHONE])
    assert page.locator("[data-shape-card='cmp'] [data-shape-src='gguf']").get_attribute(
        "aria-pressed") == "true"
    # two models with nothing measured the same way: a line, not an empty web
    compare(page, live, [board["lone"] + " · lookahead 1", SKEWED])
    assert page.locator("[data-shape-card='cmp'] [data-shape-none]").inner_text() == \
        "Nothing measured the same way for these models yet"
    assert page.locator("[data-shape-card='cmp'] svg.radar").count() == 0
    shot(page.locator("[data-shape-card='cmp']"), "shape-none.png")
    assert page.errors == []


def test_the_phone_builds_own_shape_is_its_gguf(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&view=compare&m=" + quote(PHONE, safe="")
              + "," + quote(PHONE + " · lookahead 1", safe=""))
    card = page.locator("[data-shape-card='cmp']")
    card.wait_for()
    card.locator("[data-shape-src='gguf']").click()
    page.wait_for_selector("[data-shape-card='cmp'] [data-shape-src='gguf'][aria-pressed='true']")
    labels = page.locator("[data-shape-card='cmp'] svg.radar text").evaluate_all("xs => xs.map(x => x.textContent)")
    assert {"MMLU", "ARC-C", "TruthfulQA"} <= set(labels)
    shot(card, "shape-phone-gguf.png")
    assert page.errors == []


def test_the_insights_radar_has_the_method_sources(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-radar]")
    srcs = page.locator("[data-radar-src]").evaluate_all("xs => xs.map(x => x.textContent)")
    assert srcs == ["Standard", "GGUF", "Everyday groups", "Instruction & maths", "MMLU by area",
                    "Judged by area"]
    page.evaluate(f"state.cmpSel = [{json.dumps(GOOD)}, {json.dumps(SKEWED)}]; render()")
    labels = page.locator("[data-radar] svg.radar text").evaluate_all("xs => xs.map(x => x.textContent)")
    assert "MMLU" in labels and not [t for t in labels if "_" in t]
    assert page.errors == []


# ---------------------------------------------------------------------------
# Benchmarks ▸ Standard: the chosen models, and the highlighted ones
# ---------------------------------------------------------------------------

def test_benchmarks_follow_the_models_choice_and_a_highlight(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks&sub=standard&models="
              + ",".join(quote(m, safe="") for m in (GOOD, SMALL, SKEWED))
              + "&hl=" + quote(SMALL, safe=""))
    page.wait_for_selector("[data-bench-pick='3']")
    for p in page.locator("[data-panel]").all():
        assert int(p.get_attribute("data-panel-models")) <= 3
    head = page.locator("[data-panel-method='mmlu']").inner_text()
    assert head.startswith("lm_eval · 5-shot · 3 models · higher is better")
    # the highlighted model in colour, the rest grey, in every panel
    bar = page.locator(f"[data-panel='mmlu'] path.bar[data-model='{SMALL}']")
    assert bar.get_attribute("data-hl") == "0"
    assert page.locator(f"[data-panel='mmlu'] path.bar[data-model='{GOOD}']").get_attribute(
        "opacity") == "0.35"
    # the choice is Models' own: going there, the address carries it
    page.locator("#tabs [role=tab][data-tab='models']").click()
    page.wait_for_selector("[data-lb-table]")
    assert page.locator("#pill-models").inner_text() == "Models: 3 ▾"
    # a change here is in the address, and every panel follows
    page.goto(live["base"] + "/#tab=benchmarks&sub=standard")
    page.wait_for_selector("[data-bench-pick]")
    page.locator("#pill-models").click()
    page.locator("[data-models-clear]").click()
    for mid in (GOOD, SKEWED):
        page.locator(f"#pop-models [data-model-pick='{mid}']").check()
    page.keyboard.press("Escape")
    page.wait_for_selector("[data-bench-pick='2']")
    assert "models=" + quote(GOOD, safe="") in page.evaluate("location.hash")
    page.locator("#pill-highlight").click()
    page.locator(f"#pop-highlight [data-hl-pick='{GOOD}']").check()
    page.wait_for_function("location.hash.includes('&hl=')")
    page.reload()
    page.wait_for_selector("[data-bench-pick='2']")
    assert page.locator(f"[data-panel='mmlu'] path.bar[data-model='{GOOD}']").get_attribute(
        "data-hl") == "0"
    assert page.errors == []


def test_a_gguf_panel_sits_beside_its_lm_eval_panel(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks&sub=standard")
    page.wait_for_selector("[data-panel='gguf:mmlu']")
    keys = page.locator("[data-panel]").evaluate_all("xs => xs.map(x => x.dataset.panel)")
    assert keys[keys.index("mmlu") + 1] == "gguf:mmlu"
    assert page.locator("[data-panel='gguf:mmlu'] h3").inner_text() == "MMLU · measured on the GGUF"
    assert page.locator("[data-panel-method='gguf:mmlu']").inner_text().startswith(
        "llama.cpp · 0-shot · ")
    shot(page.locator("[data-panel='gguf:mmlu']"), "gguf-panel.png")
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", [(1400, "light"), (400, "dark")])
def test_benchmarks_screens(live, page, board, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=benchmarks&sub=standard&hl={quote(GOOD, safe='')}")
    page.wait_for_selector("[data-bench-pick]")
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth + 1")
    shot(page, f"benchmarks-{width}-{scheme}.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# the two fixes
# ---------------------------------------------------------------------------

def test_a_ggufs_own_row_says_not_measured_yet_without_as_built(live, page, board):
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&cols=gguf:mmlu,gguf:arc_easy")
    lone = board["lone"]
    page.wait_for_selector(f"tr[data-lb-row='{lone} · lookahead 1']")
    own = page.locator(f"tr[data-lb-row='{lone}'] [data-gguf-cell='mmlu']")
    assert own.inner_text() == "not measured yet"
    setup = page.locator(f"tr[data-lb-row='{lone} · lookahead 1'] [data-gguf-cell='mmlu']")
    assert setup.inner_text().strip() not in ("", "—", "not measured yet")
    # the phone build's own row is its as-built result, not its lookahead one
    assert page.evaluate(f"ggufOf({json.dumps(PHONE)}, 'mmlu').setup") == "as built"
    assert page.evaluate(f"ggufOf({json.dumps(PHONE + ' · lookahead 1')}, 'mmlu').setup") == \
        "lookahead 1"
    assert page.errors == []


def test_run_lists_name_a_gguf_runs_setup_and_suites_by_the_boards_names(live, page, board):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("tr[data-queue-row]")
    rows = api(live, "/api/submissions")
    look = next(r for r in rows if r["suite"] == "gguf" and r["hf_id"] == board["lone"])
    name = page.locator(f"tr[data-queue-row='{look['id']}'] td").nth(2).inner_text()
    assert name.startswith("Qwen3.6 k=8 · GGUF · lookahead 1")
    suite = page.locator(f"tr[data-queue-row='{look['id']}'] [data-suite-cell]").inner_text()
    assert suite.startswith("Measured on the GGUF · ") and "MMLU" in suite
    built = next(r for r in rows if r["suite"] == "gguf" and r["hf_id"] == PHONE
                 and "as built" in r["note"])
    assert page.evaluate(f"runName({json.dumps(built)})") == "k4-LDA · GGUF · as built"
    # never an id: gguf, generative, safety
    words = page.evaluate("['gguf', 'generative', 'safety', 'everyday', 'full', 'judged']"
                          ".map(s => suiteWords({suite: s}))")
    assert words == ["Measured on the GGUF", "Instruction & maths", "Trust & safety",
                     "Everyday tasks", "Standard", "Knowledge exam"]
    # and the run line (Home ▸ Running now, the Runs menu) uses the same
    ev = next(r for r in rows if r["suite"] == "everyday" and r["hf_id"] == PHONE)
    line = page.evaluate(f"runLine({json.dumps(ev)}).textContent")
    assert "k4-LDA" in line and "Everyday tasks" in line and "everyday" not in line.replace(
        "Everyday", "")
    assert page.errors == []
