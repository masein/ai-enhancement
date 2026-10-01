"""12n.1 on the page: all · none and a three-state box in every picker
group; reported scores in a Frontier view of their own — credited once,
ranked only within one setting, ours tagged beside them — and as reference
ticks, never as rows or columns in our tables; Compare's Reported group down
to what the chosen models share; a served model and its GGUF one model; the
Everyday count from the result. And the live check's fixes: every chosen
model accounted for in a table someone built, labels that keep their end,
empty panels folded, Everyday following the choice, and a Model column as
wide as someone drags it.

Fixtures only: Epoch AI's trimmed file, Artificial Analysis's canned
response, cards typed in, a fake server and a fake llama-perplexity."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import gguf_data as gd
import gguf_worker as gw
from fake_openai import FakeServer
from service import config, reported
from test_gguf_12f3 import docs_of, fake_binary
from test_reported_12m2 import KEY, canned, epoch_fetch, epoch_zip

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12n1"
GOOD, SMALL, SKEWED = "fx/good-750m", "fx/below-135m-it", "fx/skewed-360m"
F55 = "reported/openai/frontier-test-5.5"
CLAUDE = "reported/anthropic/claude-test-5"
LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"
NAME = "Qwen3.6-35B-A3B k=8 original"
SERVED, SERVED_LA, SERVED_MTP = ("served/Qwen3.6-35B-A3B-k-8-original",
                                 "served/Qwen3.6-35B-A3B-k-8-original-lookahead-1",
                                 "served/Qwen3.6-35B-A3B-k-8-original-MTP")
FR_CAL = ("Same questions, different prompt and settings. A large gap means our method differs, "
          "not the model.")


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def tip(loc) -> list[str]:
    return json.loads(loc.get_attribute("data-tip"))


@pytest.fixture(scope="module", autouse=True)
def board(live, tmp_path_factory):
    """what others report — Epoch's trimmed file, Artificial Analysis's canned
    response, two cards, one of them a board model's by an alias — and a
    GGUF file measured as built and in lookahead, with three served entries
    whose server serves that same file: plain, lookahead 1 and MTP"""
    import service.app as appmod
    saved = config.AA_API_KEY
    config.AA_API_KEY = KEY
    try:
        reported.import_aa(fetch=canned())
    finally:
        config.AA_API_KEY = saved
    reported.import_epoch(fetch=epoch_fetch(epoch_zip()))
    reported.card_add({"model": "Frontier Test 5.5", "maker": "OpenAI", "benchmark": "MMLU",
                       "value": "91.2%", "setting": "5-shot", "url": "https://example.org/card",
                       "date": "2026-09-20"}, "masein")
    # a board model reported elsewhere too: its calibration cell
    reported.alias_set("Good 750M", GOOD, "masein")
    reported.card_add({"model": "Good 750M", "maker": "", "benchmark": "MMLU", "value": "55%",
                       "setting": "5-shot", "url": "https://example.org/good", "date": "2026-09-21"},
                      "masein")
    tmp = tmp_path_factory.mktemp("n1")
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    f = tmp / "k8-original.gguf"
    f.write_bytes(b"GGUF" + b"\3" * 400)
    g = api(live, "/api/gguf/models", {"name": NAME + " (GGUF)", "path": str(f),
                                       "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama.cpp, k=8",
                                       "setups": LOOK, "by": "masein"})["model"]
    binary = fake_binary(tmp)
    gw._stop["why"] = ""
    os.environ["FAKE_PPL_ACC_LOOKAHEAD"] = "0.8"
    look = next(x["id"] for x in g["setups"] if x["name"] == "lookahead 1")
    for _ in api(live, "/api/gguf/runs", {"model": g["id"], "setups": ["as-built", look],
                                          "by": "masein"})["ids"]:
        gw.Worker(config.RESULTS_ROOT, binary, poll=0.05).once()
    fake = FakeServer()
    fake.model_path, fake.size = str(f), f.stat().st_size
    for name, sid in ((NAME, SERVED), (NAME + " · lookahead 1", SERVED_LA),
                      (NAME + " · MTP", SERVED_MTP)):
        got = api(live, "/api/served", {"name": name, "base_url": fake.base, "how": "llama.cpp, k=8",
                                        "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off",
                                        "by": "masein"})["model"]
        assert got["id"] == sid
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - 120}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"gguf": g["id"]}
    fake.close()
    os.environ.pop("FAKE_PPL_ACC_LOOKAHEAD", None)


ELSEWHERE = {"model": CLAUDE, "source": "epoch", "benchmark": "MMLU", "value": 0.883, "se": None,
             "unit": "share", "setting": "from Stanford CRFM Leaderboard, 5-shot",
             "url": "https://crfm.stanford.edu/helm/", "date": "", "by": ""}


def elsewhere(page):
    """a value Epoch took from another source, as their file has them (the
    trimmed file's belong to models the import doesn't keep)"""
    def handle(route):
        r = route.fetch()
        body = r.json()
        body["scores"].append(ELSEWHERE)
        route.fulfill(response=r, body=json.dumps(body))
    page.route("**/api/reported*", handle)


def go(page, live, hash_, sel, width=1400, scheme="light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


# ---------------------------------------------------------------------------
# 1. all · none in every picker group
# ---------------------------------------------------------------------------

def test_every_picker_group_has_all_none_and_a_three_state_box(live, page):
    go(page, live, "tab=models", "[data-lb-table]")
    page.locator("#pill-models").click()
    box = page.locator("#pop-models [data-model-group-box='base']")
    box.wait_for()
    assert box.get_attribute("data-group-state") == "all"
    page.locator("#pop-models [data-model-group-none='base']").click()
    page.wait_for_selector("#pop-models [data-model-group-box='base'][data-group-state='none']")
    assert "models=" in page.evaluate("location.hash")
    base_ids = page.locator("#pop-models [data-model-group='base'] ~ label [data-model-pick]") \
        .evaluate_all("xs => xs.map(x => x.dataset.modelPick)")
    first = page.locator("#pop-models [data-model-pick]").evaluate_all(
        "xs => xs.filter(x => DATA.models.find(m => m.id === x.dataset.modelPick && m.kind === 'base'"
        " && m.source !== 'artifact')).map(x => x.dataset.modelPick)")
    assert first and base_ids
    page.locator(f"#pop-models [data-model-pick='{first[0]}']").check()
    page.wait_for_selector("#pop-models [data-model-group-box='base'][data-group-state='some']")
    assert page.locator("#pop-models [data-model-group-box='base']").evaluate("x => x.indeterminate")
    # the box of a mixed group ticks them all; again, none
    page.locator("#pop-models [data-model-group-box='base']").click()
    page.wait_for_selector("#pop-models [data-model-group-box='base'][data-group-state='all']")
    page.locator("#pop-models [data-model-group-box='base']").click()
    page.wait_for_selector("#pop-models [data-model-group-box='base'][data-group-state='none']")
    page.locator("#pop-models [data-model-group-all='base']").click()
    page.wait_for_selector("#pop-models [data-model-group-box='base'][data-group-state='all']")
    # every model ticked is "all", no choice at all; the instruct ones off is one
    assert page.evaluate("lbS().models") is None
    page.locator("#pop-models [data-model-group-none='instruct']").click()
    page.wait_for_selector("#pop-models [data-model-group-box='instruct'][data-group-state='none']")
    # Reported (not run here) is folded until opened, one Google among its makers
    assert page.locator("#pop-models [data-rep-fold]").get_attribute("aria-expanded") == "false"
    assert page.locator("#pop-models [data-model-group^='reported · ']").count() == 0
    page.locator("#pop-models [data-rep-fold]").click()
    makers = page.locator("#pop-models [data-model-group^='reported · ']").evaluate_all(
        "xs => xs.map(x => x.dataset.modelGroup)")
    assert "reported · Google" in makers and "reported · Google DeepMind" not in makers
    gem = page.locator("#pop-models [data-pick-maker][title]").first
    assert "Google DeepMind" in gem.get_attribute("title")
    page.keyboard.press("Escape")
    chosen = page.evaluate("lbS().models")
    # Benchmarks ▾'s groups
    page.locator("#pill-benchmarks").click()
    gb = page.locator("#pop-benchmarks [data-bench-group-box='commonsense']")
    gb.wait_for()
    page.locator("#pop-benchmarks [data-bench-group-none='commonsense']").click()
    page.wait_for_selector("#pop-benchmarks [data-bench-group-box='commonsense']"
                           "[data-group-state='none']")
    page.locator("#pop-benchmarks [data-bench-group-box='commonsense']").click()
    page.wait_for_selector("#pop-benchmarks [data-bench-group-box='commonsense']"
                           "[data-group-state='all']")
    page.keyboard.press("Escape")
    # the choice is the address's, shared with Benchmarks
    page.locator("#tabs [role=tab][data-tab='benchmarks']").click()
    page.locator("[data-subswitch] [data-sub='standard']").click()
    page.wait_for_selector("[data-bench-pick]")
    assert page.evaluate("lbS().models") == chosen
    assert "models=" in page.evaluate("location.hash")
    assert int(page.locator("[data-bench-pick]").get_attribute("data-bench-pick")) == len(
        [m for m in chosen if not m.startswith("reported/")])
    assert page.errors == []


def test_the_columns_groups_have_the_box_too(live, page):
    go(page, live, "tab=models", "[data-lb-table]")
    page.locator("[data-filters]").click()
    page.locator("#pill-columns").click()
    box = page.locator("#pop-columns [data-column-group-box='tasks']")
    box.wait_for()
    page.locator("#pop-columns [data-column-group-none='tasks']").click()
    page.wait_for_selector("#pop-columns [data-column-group-box='tasks'][data-group-state='none']")
    page.locator("#pop-columns [data-column-group-box='tasks']").click()
    page.wait_for_selector("#pop-columns [data-column-group-box='tasks'][data-group-state='all']")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. the Frontier view
# ---------------------------------------------------------------------------

def frkey(b):
    return re.sub(r"\s+", " ", b.lower()).strip()


def test_the_frontier_view_follows_the_coverage_rule_and_credits_once(live, page):
    elsewhere(page)
    view = api(live, "/api/reported")
    view["scores"].append(ELSEWHERE)
    n = len(view["models"])
    have = {}
    for s in view["scores"]:
        have.setdefault(frkey(s["benchmark"]), set()).add(s["model"])
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    assert page.locator("[data-chip='frontier']").get_attribute("aria-pressed") == "true"
    cols = page.locator("[data-fr-col]").evaluate_all("xs => xs.map(x => x.dataset.frCol)")
    wide = {k for k, ms in have.items() if 2 * len(ms) >= n}
    assert wide <= set(cols)
    for k in set(cols) - wide:
        # a column under half the imported models is one measured here too
        assert page.locator(f"[data-fr-here][data-fr-cell='{k}'], "
                            f"[data-fr-cal][data-fr-cell='{k}']").count(), k
    assert "mmlu" in cols                                       # measured here, and reported
    # All N benchmarks
    page.locator("[data-frontier-all]").click()
    page.wait_for_selector(f"[data-frontier-cols$='|{len(have)}']")
    assert page.locator("[data-fr-col]").count() == len(have)
    page.locator("[data-frontier-all]").click()
    # grouped by what they test: GPQA under Knowledge & reasoning, Aider under Other
    groups = page.locator("[data-fr-group]").evaluate_all("xs => xs.map(x => x.dataset.frGroup)")
    assert groups[0] == "Knowledge & reasoning" and groups[-1] == "Other"
    # the credit once, in the header; a cell is a plain number
    credit = page.locator("[data-frontier-credit]").inner_text()
    assert credit.startswith("Data: Epoch AI, CC BY 4.0 · imported ")
    assert "Data: Artificial Analysis · imported " in credit
    assert page.locator("[data-frontier]").inner_text().count("Data: Epoch AI") == 1
    page.locator("[data-frontier-all]").click()
    cell = page.locator(f"[data-fr-rep='{F55}'][data-fr-cell='mmlu-pro']")
    assert re.fullmatch(r"\d+\.\d", cell.inner_text())
    t = tip(cell)
    assert "reported by Artificial Analysis · Artificial Analysis's own run" in t[1]
    # a value Epoch took from another source: marked, the source in its tooltip
    mark = page.locator(f"[data-fr-rep='{CLAUDE}'][data-fr-cell='mmlu'] sup.fr-mark")
    assert mark.inner_text() == "†"
    assert mark.get_attribute("title") == "from Stanford CRFM Leaderboard, 5-shot"
    # a card's number is the card's own: no mark
    assert page.locator(f"[data-fr-rep='{F55}'][data-fr-cell='mmlu'] sup").count() == 0
    # the rows by maker: one Google
    makers = page.locator("[data-fr-maker]").evaluate_all("xs => xs.map(x => x.dataset.frMaker)")
    assert "Google" in makers and "Google DeepMind" not in makers
    assert makers[-1] == "Measured here"
    shot(page.locator("[data-frontier]"), "frontier-1400-light.png")
    assert page.errors == []


def test_the_frontier_ranks_only_within_one_setting(live, page):
    elsewhere(page)
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    page.locator("[data-frontier-all]").click()
    bad = page.evaluate("""() => {
      const out = [];
      const cols = [...document.querySelectorAll('[data-fr-col]')].map(x => x.dataset.frCol);
      for (const k of cols) {
        const groups = {};
        // every number in the column, a calibration cell's two parts apart
        for (const x of document.querySelectorAll(`[data-fr-cell="${CSS.escape(k)}"] [data-fr-set],
            [data-fr-cell="${CSS.escape(k)}"][data-fr-set]`)) {
          const g = x.dataset.frSet, v = +x.dataset.frV;
          (groups[g] = groups[g] || []).push({ v, lead: !!x.dataset.frLead });
        }
        for (const [g, xs] of Object.entries(groups)) {
          const best = Math.max(...xs.map(x => x.v));
          if (xs.length < 2 && xs.some(x => x.lead)) out.push(`${k}: ${g} alone and bold`);
          if (xs.length >= 2 && !xs.some(x => x.lead && x.v === best)) out.push(`${k}: ${g} best not bold`);
        }
      }
      return out; }""")
    assert bad == []
    # a setting of one (the value from another source) is never bold
    lone = page.locator(f"[data-fr-rep='{CLAUDE}'][data-fr-cell='mmlu']")
    assert lone.locator("sup.fr-mark").count() == 1 and lone.locator("b").count() == 0
    assert page.errors == []


def test_our_cells_are_tagged_and_a_model_in_both_shows_both(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    ours = page.locator("[data-fr-here][data-fr-cell='mmlu']")
    assert ours.count() >= 2
    for i in range(ours.count()):
        tag = ours.nth(i).locator(".fr-tag").inner_text()
        # 12n.2: the GGUF's MMLU is ours too, by its own method
        assert tag.startswith(("measured here · lm_eval, ", "measured here · llama.cpp, ")), tag
        assert "ranked only with the cells measured the same way here" in tip(ours.nth(i))
    # ours are never ranked with the reported: bold only for the best of ours
    here = page.locator("[data-fr-cell='mmlu'] [data-fr-set^='here|'], "
                        "[data-fr-cell='mmlu'][data-fr-set^='here|']")
    vals = [float(here.nth(i).get_attribute("data-fr-v")) for i in range(here.count())]
    best = [v for i, v in enumerate(vals) if here.nth(i).get_attribute("data-fr-lead")]
    assert best and max(vals) in best
    # the calibration line: measured here and the card's, with the one-line tooltip
    cal = page.locator(f"[data-fr-cal='{GOOD}'][data-fr-cell='mmlu']")
    assert re.fullmatch(r"measured here \d+\.\d · card 55\.0", cal.inner_text().replace("\n", ""))
    assert tip(cal) == [FR_CAL]
    # sortable: the column, then a model with no number goes to the bottom
    page.locator("[data-fr-col='mmlu']").click()
    rows = page.locator("[data-fr-row]").evaluate_all(
        "xs => xs.map(x => (x.querySelector('[data-fr-cell=\"mmlu\"]') || {}).textContent || '')")
    have = [r for r in rows if r.strip()]
    assert rows[:len(have)] == have and not any(r.strip() for r in rows[len(have):])
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. reported scores everywhere else
# ---------------------------------------------------------------------------

def test_other_chips_never_show_a_reported_row_or_column(live, page):
    go(page, live, f"tab=models&models={ids(GOOD, F55)}", "[data-lb-table]")
    assert page.locator(f"tr[data-lb-row='{F55}']").count() == 0
    assert page.locator("[data-rep2-cell]").count() == 0
    groups = page.locator("[data-lb-table] thead tr.grp th").all_inner_texts()
    assert not [g for g in groups if g.upper().startswith("REPORTED")]
    line = page.locator("[data-rep-hidden='1']")
    assert line.inner_text() == ("1 reported model isn’t shown here: nothing is measured this way "
                                 "for it · Frontier ▸")
    # a table someone built: only the columns chosen, never a reported one
    go(page, live, f"tab=models&cols=hellaswag&models={ids(GOOD, SKEWED)}", "[data-lb-table]")
    cols = page.locator("[data-lb-table] thead tr.names th[data-col]").evaluate_all(
        "xs => xs.map(x => x.dataset.col)")
    assert set(cols) <= {"rank", "name", "params", "cavg", "hellaswag"}, cols
    go(page, live, f"tab=models&models={ids(GOOD, F55)}", "[data-rep-hidden]")
    page.locator("[data-rep-hidden-go]").click()
    page.wait_for_selector(f"[data-frontier-table] [data-fr-row='{F55}']")
    assert page.locator("[data-fr-row]").count() == 2
    assert page.errors == []


def test_compare_shows_only_the_reported_benchmarks_they_share(live, page):
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, F55)}", "[data-compare='2']")
    # Artificial Analysis reports nothing of the board model: nothing shared
    g = page.locator("[data-cmp-group='rep:aa']")
    assert "none shared by two of these" in g.inner_text()
    assert page.locator("[data-cmp-row^='rep:aa:']").count() == 0
    assert page.locator("[data-cmp-credit='rep:aa']").inner_text() == " · Data: Artificial Analysis"
    more = page.locator("[data-cmp-show-all='rep:aa']")
    n = int(re.search(r"Show all (\d+) reported benchmarks", more.inner_text()).group(1))
    more.click()
    page.wait_for_selector("[data-cmp-row^='rep:aa:']")
    assert page.locator("[data-cmp-row^='rep:aa:']").count() == n
    # the card's MMLU: the board measures MMLU too, so it is shared
    assert page.locator("[data-cmp-row='rep:card:MMLU']").count() == 1
    # the credit once, in the group's head: never in a row's tooltip
    c = page.locator(f"[data-cmp-row='rep:card:MMLU'] [data-cmp-cell='{F55}']")
    assert not [x for x in tip(c) if "model card or paper reports it" in x]
    # only reported models: the Frontier view's default set
    go(page, live, f"tab=models&view=compare&m={ids(F55, CLAUDE)}", "[data-compare='2']")
    rows = page.locator("[data-cmp-row^='rep:']").evaluate_all(
        "xs => xs.map(x => x.dataset.cmpRow.split(':').slice(2).join(':'))")
    dflt = page.evaluate("frColumns().filter(c => c.dflt).map(c => c.key)")
    assert rows and all(frkey(r) in dflt for r in rows)
    assert page.errors == []


def test_benchmarks_draws_reported_numbers_as_ticks_on_their_benchmark_only(live, page):
    go(page, live, "tab=benchmarks&sub=standard", "[data-panel='mmlu']")
    mmlu = page.locator("[data-panel='mmlu']")
    ticks = mmlu.locator("g.reftick")
    assert ticks.count() >= 1
    labels = ticks.locator("text").all_text_contents()
    assert any(re.fullmatch(r".+ \d+\.\d · (Epoch|card|AA)", x) for x in labels), labels
    # a tick, never a bar: no reported model has a bar here
    assert mmlu.locator("path.bar[data-model^='reported/']").count() == 0
    assert ticks.first.locator("line").get_attribute("stroke-dasharray")
    # only on a benchmark reported elsewhere
    assert page.locator("[data-panel='hellaswag'] g.reftick").count() == 0
    # the frontier chip from what was imported
    chip = page.locator("[data-frontier-chip='mmlu']")
    assert chip.inner_text() == "frontier 91.2%"
    assert "Frontier Test 5.5, reported by model card · 5-shot" in chip.get_attribute("title")
    # a Frontier group: a panel a default benchmark, bars of one setting
    fr = page.locator("[data-frontier-panels]")
    assert fr.locator("[data-panel='fr:gpqa diamond']").count() == 1
    assert fr.locator("[data-panel='fr:gpqa diamond'] path.bar").count() >= 2
    shot(page.locator("[data-panel='mmlu']"), "mmlu-reference-ticks.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 4. one model, and the Everyday count
# ---------------------------------------------------------------------------

def test_a_served_model_and_its_gguf_are_one_model(live, page, board):
    go(page, live, "tab=models", "[data-lb-table]")
    rows = page.locator("[data-lb-row]").evaluate_all("xs => xs.map(x => x.dataset.lbRow)")
    assert board["gguf"] not in rows and not [r for r in rows if r.startswith(board["gguf"] + " ·")]
    assert SERVED in rows and SERVED_LA in rows
    served = page.locator(f"tr[data-lb-row='{SERVED}'] [data-gguf-cell]").first
    assert re.fullmatch(r"\d+\.\d", served.inner_text())
    la = page.locator(f"tr[data-lb-row='{SERVED_LA}'] [data-gguf-cell='hellaswag']")
    look = page.evaluate(f"G().setups[{json.dumps(SERVED)}].find(x => x.name === 'lookahead 1')"
                         ".benches.hellaswag.v")
    assert la.inner_text() == f"{100 * look:.1f}"               # its setup's, lookahead 1
    # MTP has no GGUF counterpart: its own row, no GGUF cell of its own
    assert page.evaluate(f"!!ggufOf({json.dumps(SERVED_MTP)}, 'hellaswag')") is False
    # one Compare column
    go(page, live, f"tab=models&view=compare&m={ids(board['gguf'], GOOD)}", "[data-compare='2']")
    assert page.locator("[data-cmp-col]").evaluate_all("xs => xs.map(x => x.dataset.cmpCol)") \
        == [SERVED, GOOD]
    page.locator("[data-cmp-fold='gguf']").click()
    assert page.locator(f"[data-cmp-row='gguf:hellaswag'] [data-cmp-cell='{SERVED}']").count() == 1
    # one page: the GGUF's address opens the served model's, with both provenances
    go(page, live, "model=" + quote(board["gguf"], safe=""), "[data-model-hero]")
    assert page.evaluate("state.model") == SERVED
    assert page.locator(f"[data-served-head='{SERVED}']").count() == 1
    assert page.locator(f"[data-gguf-joined='{board['gguf']}']").inner_text().startswith(
        f"Its file, measured by llama.cpp as {NAME} (GGUF): ")
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_ggufs_address_opens_its_block_with_measure(live, page, board, width):
    """12z B1: the GGUF's own address lands on the served page at its GGUF's
    block, open, Measure in it — the numbers were inside a closed Standard"""
    go(page, live, "model=" + quote(board["gguf"], safe=""), "[data-model-hero]", width=width)
    blk = page.locator("[data-kind-block='gguf']")
    blk.wait_for()
    assert blk.get_attribute("open") is not None
    assert blk.locator(f"[data-gguf-part='{SERVED}'] [data-gguf-row]").count() > 0
    assert blk.locator(f"[data-gg-measure='{SERVED}']").is_visible()
    assert page.locator("[data-kind-block='standard'] [data-gguf-part]").count() == 0
    # the tile says how many, and opens it
    tile = page.locator("[data-kind-tile='gguf']")
    assert tile.locator(".ktile-k").inner_text().lower() == "on its gguf · llama.cpp"
    assert re.fullmatch(r"\d+ of \d+", tile.locator("[data-kind-value='gguf']").inner_text())
    # Measure opens the dialog for the GGUF, from here
    blk.locator(f"[data-gg-measure='{SERVED}']").click()
    page.locator(f"[data-gg-measure-dialog='{board['gguf']}']").wait_for()
    page.keyboard.press("Escape")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(blk, f"gguf-block-{width}.png")
    assert page.errors == []


def test_compares_everyday_count_is_the_results(live, page):
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, SKEWED)}", "[data-compare='2']")
    n = page.evaluate(f"evdOf({json.dumps(GOOD)}).total")
    assert n == page.evaluate("evdHidden()") and n != page.evaluate("evdAll()")
    assert page.locator("[data-cmp-main='evd']").inner_text() == f"{n} questions, our checks"
    assert page.errors == []


# ---------------------------------------------------------------------------
# the live check's fixes
# ---------------------------------------------------------------------------

LOGLIK = "mmlu,hellaswag,piqa,winogrande,arc_challenge,arc_easy"


def test_every_chosen_model_is_a_row_or_says_why_not(live, page):
    chosen = (SERVED, SERVED_LA, SERVED_MTP)
    go(page, live, f"tab=models&cols={LOGLIK}&models={ids(*chosen)}", "[data-lb-table]")
    empty = page.locator("[data-none-has]")
    assert empty.locator("p").inner_text() == (
        "None of the 3 has any of these 6. 3 can’t be measured this way (served or GGUF) · "
        "Show their llama.cpp columns")
    # under the line, open from the start: the table has no rows
    assert page.locator("[data-not-tested-toggle]").get_attribute("aria-expanded") == "true"
    for m in chosen:
        assert page.locator(f"[data-not-tested-row='{m}']").count() == 1, m
    why = page.locator(f"[data-not-here='{SERVED}']").inner_text()
    assert why.startswith(" · served · a server can’t give the log-likelihoods these benchmarks "
                          "use · measured by llama.cpp instead: ")
    assert re.search(r"HellaSwag \d+\.\d", why)
    assert page.locator(f"[data-not-here='{SERVED_MTP}']").inner_text() == \
        " · served · a server can’t give the log-likelihoods these benchmarks use"
    # "Show their llama.cpp columns": the GGUF benchmarks that ask the same questions
    want = page.evaluate("""llamaCounterparts('mmlu,hellaswag,piqa,winogrande,arc_challenge,arc_easy'
      .split(',').map(t => ({ task: t }))).map(b => 'gguf:' + b)""")
    assert want and "gguf:piqa" not in want
    page.locator("[data-llama-switch='all']").click()
    page.wait_for_selector(f"tr[data-lb-row='{SERVED}']")
    assert page.evaluate("lbS().cols") == want
    assert page.locator(f"tr[data-lb-row='{SERVED_LA}']").count() == 1
    assert page.errors == []


def test_the_empty_table_offers_every_set_that_covers_them(live, page):
    """12x: not only llama.cpp's — DeviceMark's too, where a chosen model has a row"""
    import service.app as appmod
    from test_12q_devicemark_board import SERVED as DM_SETUP, _row
    _row(config.OUT_DIR, SERVED_MTP, {"ifeval": 0.7, "mmlu_pro": 0.6, "math": 0.5},
         {**DM_SETUP, "name": NAME + " · MTP"})
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        go(page, live, f"tab=models&cols={LOGLIK}&models={ids(SERVED, SERVED_MTP)}",
           "[data-none-has]")
        empty = page.locator("[data-none-has]")
        assert empty.locator("p").inner_text() == (
            "None of the 2 has any of these 6. 2 can’t be measured this way (served or GGUF) · "
            "Show their llama.cpp columns · Show their DeviceMark columns")
        assert empty.locator("[data-other-set]").evaluate_all(
            "bs => bs.map(b => b.dataset.otherSet)") == ["gguf", "devicemark"]
        shot(empty, "empty-offers.png")
        empty.locator("[data-other-set='devicemark']").click()
        page.wait_for_selector(f"tr[data-lb-row='{SERVED_MTP}'] [data-dm-cell='composite']")
        # the GGUF entry joined to the plain setup has no DeviceMark row: under the line
        assert page.locator(f"tr[data-lb-row='{SERVED}']").count() == 0
        # llama.cpp's columns chosen, DeviceMark's is still offered for the one without
        go(page, live, f"tab=models&cols=gguf:mmlu&models={ids(SERVED_MTP)}", "[data-lb-card]")
        assert page.locator("[data-other-set='devicemark']").count() == 1
        assert page.locator("[data-other-set='gguf']").count() == 0
    finally:
        for f in (config.OUT_DIR / SERVED_MTP.replace("/", "__")).glob("devicemark*"):
            f.unlink()
        appmod._cache.update(key=None, payload=None, at=0.0)
    assert page.errors == []


def test_labels_keep_what_tells_them_apart(live, page):
    go(page, live, "tab=benchmarks&sub=standard", "[data-bench-pick]")
    names = ["Qwen3.6-35B-A3B k4-LDA · lookahead 1", "Qwen3.6-35B-A3B original · lookahead 1",
             "Qwen3.6-35B-A3B k4-LDA", "Qwen3.6-35B-A3B original", "SmolLM2-360M"]
    got = page.evaluate(f"Object.fromEntries(shortNames({json.dumps(names)}, 22))")
    assert got == {names[0]: "k4-LDA · lookahead 1", names[1]: "original · lookahead 1",
                   names[2]: "k4-LDA", names[3]: "original", names[4]: "SmolLM2-360M"}
    # none too long: nothing cut
    assert page.evaluate("Object.fromEntries(shortNames(['Qwen3-1.7B', 'Qwen3-0.6B'], 22))") == {
        "Qwen3-1.7B": "Qwen3-1.7B", "Qwen3-0.6B": "Qwen3-0.6B"}
    assert page.evaluate("wrap2('Qwen3.6-35B-A3B k=8 original · lookahead 1', 24)") == [
        "Qwen3.6-35B-A3B k=8", "original · lookahead 1"]
    # in a panel: the whole name, in two lines, and on hover
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(SERVED, SERVED_LA)}",
       "[data-panel='gguf:hellaswag']")
    lab = page.locator(f"[data-panel='gguf:hellaswag'] text.blab[data-full-name][data-model='{SERVED_LA}']")
    assert lab.get_attribute("data-lines") == "2"
    assert "".join(lab.locator("tspan").all_text_contents()).replace(" ", "") == \
        (NAME + " · lookahead 1").replace(" ", "")
    assert lab.get_attribute("data-full-name") == NAME + " · lookahead 1"
    assert page.errors == []


def test_panels_with_no_number_for_the_chosen_fold_into_a_line(live, page):
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(SERVED, SERVED_LA)}",
       "[data-empty-panels]")
    assert page.locator("[data-panel='hellaswag'], [data-panel='mmlu']").count() == 0
    line = page.locator("[data-empty-panels]").first
    assert line.inner_text().startswith("No numbers for the chosen models: ")
    assert line.inner_text().endswith("(lm_eval) · served and GGUF models can’t be measured this way")
    # they come back with a model that has a number
    go(page, live, f"tab=benchmarks&sub=standard&models={ids(SERVED, GOOD)}", "[data-panel='mmlu']")
    assert "mmlu" not in (page.locator("[data-empty-panels]").first.get_attribute(
        "data-empty-panels") or "").split(",") if page.locator("[data-empty-panels]").count() else True
    assert page.errors == []


def test_everyday_and_the_exam_follow_the_choice(live, page):
    go(page, live, f"tab=benchmarks&sub=everyday&models={ids(GOOD, SKEWED)}",
       "[data-everyday-table]")
    cols = page.locator("[data-evd-model]").evaluate_all("xs => xs.map(x => x.dataset.evdModel)")
    assert sorted(cols) == sorted([GOOD, SKEWED])
    assert f"models={ids(GOOD, SKEWED)}" in page.evaluate("location.hash")
    assert page.locator("[data-bench-pick='2']").count() == 1
    page.locator("[data-subswitch] [data-sub='exam']").click()
    page.wait_for_function("location.hash.includes('sub=exam')")
    assert f"models={ids(GOOD, SKEWED)}" in page.evaluate("location.hash")
    page.locator("[data-subswitch] [data-sub='standard']").click()
    page.wait_for_selector("[data-bench-pick='2']")
    assert page.errors == []


def test_the_model_column_widens_fits_resets_and_wraps_on_a_phone(live, page):
    # 12o.1: the Model column's width is one of every column's, kept per table
    key = "bench-layout-models:standard:all"
    kept_w = f"(JSON.parse(localStorage.getItem('{key}') || '{{}}').w || {{}}).name"
    go(page, live, "tab=models", "[data-lb-table]")
    page.evaluate(f"localStorage.removeItem('{key}')")
    th = page.locator("[data-lb-table] thead th.model")
    w0 = th.bounding_box()["width"]
    grip = th.locator("[data-col-grip='name']")
    b = grip.bounding_box()
    page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    page.mouse.down()
    page.mouse.move(b["x"] + 120, b["y"] + b["height"] / 2, steps=6)
    page.mouse.up()
    w1 = th.bounding_box()["width"]
    assert w1 > w0 + 80
    kept = int(page.evaluate(kept_w))
    assert abs(kept - w1) <= 2
    # the sort did not change: the handle is not the header's click
    page.reload()
    page.wait_for_selector("[data-lb-table] thead th.model[data-lw]")
    # measured in the page: a poll may draw the header again between a wait and a read
    assert abs(page.evaluate("document.querySelector('[data-lb-table] thead th.model')"
                             ".getBoundingClientRect().width") - w1) <= 2
    # the keyboard: ← narrower
    page.locator("[data-lb-table] thead th.model [data-col-grip='name']").focus()
    page.keyboard.press("ArrowLeft")
    assert int(page.evaluate(kept_w)) == kept - 16
    # a double-click fits the longest name shown: none is cut
    page.locator("[data-lb-table] thead th.model [data-col-grip='name']").dblclick()
    cut = page.locator("[data-lb-table] tbody .mname").evaluate_all(
        "xs => xs.filter(x => x.scrollWidth > x.clientWidth + 1).map(x => x.textContent)")
    assert cut == []
    # Reset layout, in the header's ⋯
    page.locator("[data-lb-table] thead th.model [data-col-more='name']").click()
    page.locator("[data-layout-reset='models:standard:all']").click()
    page.wait_for_selector("[data-lb-table] thead th.model:not([data-lw])")
    assert page.evaluate(f"localStorage.getItem('{key}')") is None
    assert abs(page.evaluate("document.querySelector('[data-lb-table] thead th.model')"
                             ".getBoundingClientRect().width") - w0) <= 2
    # a phone: no handle, and a long name wraps to two lines
    go(page, live, f"tab=models&models={ids(SERVED_LA, GOOD)}", "[data-lb-table]", width=400)
    assert not page.locator("[data-lb-table] [data-col-grip='name']").is_visible()
    name = page.locator(f"tr[data-lb-row='{SERVED_LA}'] .mname")
    lh = float(name.evaluate("x => parseFloat(getComputedStyle(x).lineHeight) || 16"))
    assert name.bounding_box()["height"] > 1.5 * lh
    # what it keeps is its end, the setup; the whole name is its label
    assert name.inner_text().replace("\n", " ").endswith("original · lookahead 1")
    assert name.get_attribute("aria-label") == NAME + " · lookahead 1"
    shot(page.locator("[data-lb-card]"), "models-400-wraps.png")
    assert page.errors == []


def test_the_frontier_view_at_400px_and_in_the_dark(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]", width=400, scheme="dark")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-frontier]"), "frontier-400-dark.png")
    go(page, live, f"tab=models&view=compare&m={ids(GOOD, F55)}", "[data-compare='2']", width=400,
       scheme="dark")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-compare]"), "compare-400-dark.png")
    assert page.errors == []
