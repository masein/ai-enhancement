"""12q.C on the page: a DeviceMark run on its model's page — a card for each
thinking mode (the composite with its interval, the rank, the three benches,
answered, median tokens, device and server tok/s, and its row on the
On-device chart), the pilot, the parity check and the speed test in it; a
DeviceMark column in the setups of the file; "Open results" opening the
run's own result; and the Answers tab's DeviceMark items, no answer and wrong
first. Two served setups of one file, against a fake server: nothing runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import devicemark as dm
from fake_openai import FakeServer
from service import config, db
from test_12q_devicemark_board import HF, _row
from test_12q_devicemark_model_page import (ITEMS, MTP, NEMO, PLAIN, QUESTIONS, _dir, _write,
                                            runs_here)

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12qc"
THEIR_NEMO = "nemotron-4b__int8hu__aimodel"


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def shot(part, name):
    """a screenshot, taken again when the page drew itself anew under it"""
    # imported here, not at the top: deploy step 3 collects this module inside
    # the image, which has no Playwright (tests/test_step3_collects.py)
    from playwright.sync_api import Error as PlaywrightError
    SCREENS.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        try:
            part.screenshot(path=SCREENS / name)
            return
        except PlaywrightError as e:
            if "not attached" not in str(e) or i == 2:
                raise


@pytest.fixture(scope="module")
def runs(live):
    """the two setups registered through the API, each with a score, and every
    kind of DeviceMark run on the one with MTP — and our run of one of
    DeviceMark's models on hf, which has no page of its own"""
    import service.app as appmod
    from service import served
    fake = FakeServer()
    got = {}
    for name, extra in (("DM phone build", {"phone": True, "how": "llama.cpp fork, k=4 + LDA"}),
                        ("DM phone build, MTP", {"how": "llama.cpp fork, k=4 + LDA, MTP 3",
                                                 # 12z A1: MTP from its launch, not its words
                                                 "flags": "--spec-type draft-mtp"})):
        r = api(live, "/api/served", {"name": name, "base_url": fake.base, "thinking": "off",
                                      "based_on": "Qwen/Qwen3.6-35B-A3B", "by": "masein", **extra})
        got[name] = (r.get("model") or r)["id"]
    assert sorted(got.values()) == sorted([MTP, PLAIN]), got
    for sid in (MTP, PLAIN):
        d = config.OUT_DIR / sid.replace("/", "__") / "ifeval_0shot" / "served"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results_2026-09-29T10-00-00.000000.json").write_text(json.dumps({
            "results": {"ifeval": {"alias": "ifeval", "prompt_level_strict_acc,none": 0.62}},
            "config": {"model": "local-chat-completions", "model_args": f"pretrained={sid},x=1"},
            "served": served.view(served.get(sid)), "date": 2.0}))
    runs_here(config.OUT_DIR)
    config.DM_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    config.DM_ITEMS.write_text(json.dumps({"version": dm.VERSION}) + "\n"
                               + "".join(json.dumps(q) + "\n" for q in QUESTIONS))
    _write(_dir(config.OUT_DIR, MTP), dm.ITEMS_NAME, ITEMS)
    _row(config.OUT_DIR, NEMO, {"ifeval": 0.6, "mmlu_pro": 0.6, "math": 0.8}, HF)
    sids = {}
    for key, model, kw in (("full", MTP, {"part": "full"}), ("pilot", MTP, {"part": "pilot"}),
                           ("parity", MTP, {"part": "parity", "pair": PLAIN}),
                           ("speed", MTP, {"part": "speed"}),
                           ("thinking", MTP, {"part": "full", "thinking": True}),
                           ("theirs", NEMO, {"part": "full"})):
        sids[key] = db.add(model, "instruct", "devicemark", "masein", "", **kw)
        db.update(sids[key], status="done", progress="done")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield sids
    fake.close()
    for model in (MTP, PLAIN, NEMO):
        for suffix in ("", "__thinking"):
            d = config.OUT_DIR / (model.replace("/", "__") + suffix)
            for f in d.glob("devicemark*"):
                f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


def model_page(page, live, mid, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + mid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")


def open_block(page, kind):
    page.locator(f"[data-kind-tile='{kind}']").click()
    page.wait_for_selector(f"[data-kind-block='{kind}'][open] [data-dm-card]")
    return page.locator(f"[data-kind-block='{kind}']")


# on screen, below the bar that stays at the top
SEEN = """sel => {{ const e = document.querySelector(sel); if (!e) return false;
  const r = e.getBoundingClientRect(), bar = document.getElementById('bar');
  const top = bar ? bar.getBoundingClientRect().bottom : 0;
  return r.top >= top - 1 && r.top < innerHeight; }}"""


def wait_seen(page, sel):
    page.wait_for_function(SEEN.replace("{{", "{").replace("}}", "}"), arg=sel)


# ---------------------------------------------------------------------------
# 1. the cards, and the setups of the file
# ---------------------------------------------------------------------------

def test_a_card_for_each_thinking_mode_with_its_row_and_its_runs(live, page, runs):
    model_page(page, live, MTP)
    off = json.loads((_dir(config.OUT_DIR, MTP) / dm.OUT_NAME).read_text())
    comp = f"{100 * off['composite']['value']:.1f} ±" \
           f"{50 * (off['composite']['ci'][1] - off['composite']['ci'][0]):.1f}"
    assert page.locator("[data-kind-tile='dm'] [data-kind-value='dm']").inner_text() == comp
    assert page.locator("[data-kind-tile='dm_thinking']").count() == 1
    blk = open_block(page, "dm")
    line = blk.locator("[data-dm-card-line='off']").inner_text()
    assert line.startswith(f"{comp} composite · ") and (
        " among ranked rows (cloud lines aren’t ranked) · phone build · MTP · thinking off · "
        "its row on the On-device chart") in line
    t = blk.locator("[data-dm-card-table='off']")
    assert t.locator("[data-dm-card-bench='ifeval']").inner_text().startswith("70.0 ±")
    assert t.locator("[data-dm-card-device='off']").inner_text() == "14.2 · iPhone 17 Pro"
    assert t.locator("[data-dm-card-server='off']").inner_text() == "31.5"
    # the pilot, the parity check and the speed test, in the card
    assert blk.locator("[data-dm-cap-check='off']").inner_text().startswith(
        "Cap check: thinking on, max_tokens 64: stopped at 64 tokens (length)")
    assert blk.locator("[data-dm-pilot-line='off']").inner_text().startswith("composite ")
    assert blk.locator("[data-dm-parity-line='off']").inner_text() == (
        f"{MTP} (MTP) against {PLAIN}, thinking off: identical 47/50 · same answer 49/50 · "
        "passes (48 or more): the setup without MTP takes its quality from the MTP run")
    diff = blk.locator("[data-dm-parity-differ='off'] tbody tr")
    assert diff.count() == 3
    assert diff.nth(0).inner_text().split("\t") == ["MATH · m1", "the answer", "2 · 3", "90 · 95"]
    assert diff.nth(1).inner_text().split("\t")[1] == "the tokens, not the answer"
    assert blk.locator("[data-dm-speed-line='off']").inner_text().startswith(
        "31.5 tok/s decode, the mean of the timed trials · 128 prompt tokens, 256 decoded")
    trials = blk.locator("[data-dm-speed-trials='off'] tbody tr").all_inner_texts()
    assert trials[0].startswith("warm-up (not counted)") and trials[2].startswith("trial 2")
    shot(blk, "devicemark-card-1400.png")
    # the thinking card: its own row, no speed test (the server's speed is the one)
    th = open_block(page, "dm_thinking")
    assert th.locator("[data-dm-card-line='on']").count() == 1
    assert th.locator("[data-dm-part='speed']").count() == 0
    # its row on the On-device chart
    blk.locator(f"[data-dm-card-chart='{MTP}']").click()
    page.wait_for_selector(f"tr.dmfocus[data-dm-row='{MTP}']")
    assert page.evaluate("location.hash").startswith("#tab=models")
    assert page.errors == []


def test_the_setups_of_the_file_have_a_devicemark_column(live, page, runs):
    model_page(page, live, PLAIN)
    table = page.locator("[data-served-setups]")
    table.wait_for()
    assert "DeviceMark" in table.locator("thead").text_content()
    # 12z A2: a row for each setup and thinking mode, each with its own number
    mtp = table.locator(f"[data-served-dm='{MTP}|off']").inner_text()
    on = table.locator(f"[data-served-dm='{MTP}|on']").inner_text()
    assert mtp.count("±") == 1 and on.count("±") == 1 and mtp != on
    names = table.locator("tbody tr td:first-child").all_inner_texts()
    assert names.count("DM phone build, MTP") == 1 and names.count("DM phone build, MTP · thinking") == 1
    plain = table.locator(f"[data-served-dm='{PLAIN}|off']")
    assert plain.inner_text() == mtp                               # its MTP partner's, said so
    assert plain.get_attribute("title") == "quality from MTP run, parity 49/50"
    shot(table, "setups-devicemark-column.png")
    blk = open_block(page, "dm")
    assert blk.locator("[data-dm-card-inherited='off']").inner_text().startswith(
        "quality from MTP run, parity 49/50")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 2. "Open results" opens the run's own result
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,kind,part", [("full", "dm", None), ("pilot", "dm", "pilot"),
                                           ("parity", "dm", "parity"), ("speed", "dm", "speed"),
                                           ("thinking", "dm_thinking", None)])
def test_open_results_opens_the_runs_own_result(live, page, runs, key, kind, part):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=queue")
    page.locator(f"[data-row-open='{runs[key]}']").click()
    page.wait_for_selector(f"[data-kind-block='{kind}'][open] [data-dm-card]")
    assert page.evaluate("location.hash") == "#model=" + MTP.replace("/", "%2F")
    if part:
        sel = f"[data-kind-block='{kind}'] [data-dm-part='{part}']"
        page.wait_for_selector(sel + ".dmfocus")
        wait_seen(page, sel)
        if part == "parity":                               # all of it, below the bar
            page.set_viewport_size({"width": 1400, "height": 2400})
            page.evaluate("scrollTo(0, 0)")
            shot(page.locator(f"[data-kind-block='{kind}']"), "open-results-parity.png")
    assert page.errors == []


def test_open_results_on_a_model_with_no_page_opens_its_row_on_the_chart(live, page, runs):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=queue")
    page.locator(f"[data-row-open='{runs['theirs']}']").click()
    # our run of their model: in their row
    page.wait_for_selector(f"tr.dmfocus[data-dm-row='{THEIR_NEMO}']")
    wait_seen(page, f"tr[data-dm-row='{THEIR_NEMO}']")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 3. the Answers tab
# ---------------------------------------------------------------------------


# 12z C1: the mode a switch, the benches chips 8px apart, the rows 12px apart
GAPS = """sel => { const xs = [...document.querySelectorAll(sel)].map(e => e.getBoundingClientRect());
  const out = []; for (let i = 1; i < xs.length; i++)
    if (Math.abs(xs[i].top - xs[i - 1].top) < 2) out.push(xs[i].left - xs[i - 1].right);
  return out; }"""
VGAP = """([a, b]) => document.querySelector(b).getBoundingClientRect().top
  - document.querySelector(a).getBoundingClientRect().bottom"""


@pytest.mark.parametrize("width", [1400, 375])
def test_devicemarks_answers_have_a_mode_switch_and_spaced_chips(live, page, runs, width):
    model_page(page, live, MTP, width=width)
    page.locator("[data-mtab='answers']").click()
    modes = page.locator("[data-dm-ans-modes]")
    modes.wait_for()
    assert "seg" in modes.get_attribute("class").split() and modes.locator(".chip-btn").count() == 0
    page.wait_for_selector("[data-dm-ans-benches] .chip-btn")
    gaps = page.evaluate(GAPS, "[data-dm-ans-benches] .chip-btn")
    assert gaps and min(gaps) >= 7.5, gaps
    assert page.evaluate(VGAP, ["[data-dm-ans-modes]", "[data-dm-ans-benches]"]) >= 11.5
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-dm-ans-benches]"), f"answers-devicemark-chips-{width}.png")
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_setups_table_has_no_drafts_column_until_one_is_reported(live, page, runs, width):
    """12z C3: "not reported" on every row is a column of nothing"""
    model_page(page, live, PLAIN, width=width)
    table = page.locator("[data-served-setups]")
    table.wait_for()
    heads = table.locator("thead th").all_text_contents()
    assert "MTP drafts accepted" not in heads and table.locator("[data-served-draft]").count() == 0
    assert heads[-1] == "Ran out"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert page.errors == []


def test_the_answers_tab_has_every_item_no_answer_and_wrong_first(live, page, runs):
    model_page(page, live, MTP)
    page.locator("[data-mtab='answers']").click()
    # DeviceMark's is this model's only kind of written answer: no chips to choose it
    assert page.locator("[data-answers-kind]").count() == 0
    page.wait_for_selector("[data-dm-ans]")
    ids = page.locator("[data-dm-ans]").evaluate_all("xs => xs.map(x => x.dataset.dmAns)")
    assert ids == ["math:test/algebra/1.json", "mmlu_pro:p2", "ifeval:i1", "mmlu_pro:p1",
                   "math:test/algebra/2.json"]
    marks = page.locator("[data-dm-ans]").evaluate_all("xs => xs.map(x => x.dataset.dmAnsMark)")
    assert marks == ["no answer", "wrong", "right", "right", "right"]
    # ran out of room, still thinking; the thinking folded
    first = page.locator("[data-dm-ans='math:test/algebra/1.json']")
    assert first.locator("[data-dm-ans-capped]").inner_text() == "ran out of room"
    assert first.locator("[data-dm-ans-out]").inner_text() == \
        "No answer: it was still thinking when it ran out of room."
    assert first.locator("[data-dm-ans-read]").inner_text() == \
        "read as: nothing (no answer) · the answer: 2"
    wrong = page.locator("[data-dm-ans='mmlu_pro:p2']")
    assert wrong.locator("[data-dm-ans-read]").inner_text() == "read as: A · the answer: B"
    assert wrong.locator("[data-dm-ans-q]").inner_text() == "Which is even?\n\nA. 3\nB. 4"
    right = page.locator("[data-dm-ans='mmlu_pro:p1']")
    fold = right.locator("[data-dm-ans-thinking]")
    assert fold.get_attribute("open") is None and right.locator("[data-dm-ans-out]").inner_text() \
        == "The answer is (C)"
    ife = page.locator("[data-dm-ans='ifeval:i1'] [data-dm-ans-read]").inner_text()
    assert ife.startswith("strict: 1 of 1 instructions followed")
    page.set_viewport_size({"width": 1400, "height": 2200})         # all of it, below the bar
    shot(page.locator("[data-model-answers]"), "answers-devicemark-1400.png")
    page.set_viewport_size({"width": 1400, "height": 1000})
    # one bench, with its counts
    chips = page.locator("[data-dm-ans-benches]").inner_text()
    assert "MATH · 2 · 1 no answer" in chips and "MMLU-Pro · 2 · 1 wrong" in chips
    page.locator("[data-dm-ans-bench='math']").click()
    page.wait_for_function("document.querySelectorAll('[data-dm-ans]').length === 2")
    # thinking on: its own answers, fifty at a time (the bench chosen stays chosen)
    page.locator("[data-dm-ans-mode='on']").click()
    page.wait_for_function("document.querySelector('[data-dm-ans-shown]')?.dataset.dmAnsShown"
                           " === '10'")
    page.locator("[data-dm-ans-bench='all']").click()
    page.wait_for_selector("[data-dm-ans-more]")
    page.wait_for_function("document.querySelector('[data-dm-ans-shown]')?.dataset.dmAnsShown"
                           " === '50'")
    page.locator("[data-dm-ans-more]").click()
    page.wait_for_function("document.querySelector('[data-dm-ans-shown]')?.dataset.dmAnsShown"
                           " === '60'")
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_at_400px(live, page, runs, scheme):
    page.emulate_media(color_scheme=scheme)
    model_page(page, live, MTP, width=400)
    blk = open_block(page, "dm")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(blk, f"devicemark-card-400-{scheme}.png")
    page.locator("[data-mtab='answers']").click()
    page.wait_for_selector("[data-dm-ans]")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    shot(page.locator("[data-dm-ans]").first, f"answer-400-{scheme}.png")
    assert page.errors == []
