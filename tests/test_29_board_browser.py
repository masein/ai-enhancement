"""29 in a real browser, at 1280 and 400 px, light and dark: NVIDIA Nemotron
3.5 Bolt E4B measured here on our own V100s — the On-device report's
measured Q4_K_M row (a filled dot in NVIDIA's colour, "measured here on a
V100", what it keeps "of our Q8_0, standing in for BF16"), NVIDIA's own row
as published beside it; Models ▸ Frontier's two measured rows in the Open
group, its 64K cap on hover and said once; one model page holding both
builds and NVIDIA's numbers, told apart; its speed on a V100 with "own box"
for dollars; and Runs' paused own box. The world is 25c's scorecard with
every card entered by masein, and test_29_board's invented Bolt scores;
nothing measured, nothing fetched."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_23b_browser import runs
from test_23c_browser import timed
from test_29_board import CAPPED, PUB, Q4, Q4_ID, Q8, Q8_ID, bolt_world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase29"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]
GPQA = "gpqa_diamond_epoch"
V100 = "Tesla V100-DGXS-32GB"
WHY = ("GPU 0 of D3 has another process (pid 4242, python, 1,234 MiB): paused after the "
       "answers in flight; it carries on when the GPU is free")


def card(alone: float, who: str) -> dict:
    return {"name": V100, "alone_out": alone, "alone_prompt": 64 * alone,
            "long_prompt": 40 * alone, "long_out": alone - 10, "together_out": 5 * alone,
            "together_n": 8, "box": who}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import enter_report_cards as erc
    import service.app as appmod
    from service import reports
    scorecard()
    assert erc.main(["on-device", "--by", "masein"]) == 0
    bolt_world()
    # its GPQA Diamond from two of dgx-03's GPUs, own boxes: no price to rent
    d = reports.row_dir(Q4_ID + " · thinking")
    (d / "frontier_imports.json").write_text(json.dumps({"tasks": {GPQA: {"shard_bundles": [
        {"shard": 1, "box": "D3.0", "answers": 396, "energy_wh": 120.0, "server_s": 3600.0,
         "card": card(60.0, "D3.0"), "own_box": True},
        {"shard": 2, "box": "D3.1", "answers": 396, "energy_wh": 126.0, "server_s": 3700.0,
         "card": card(64.0, "D3.1"), "own_box": True}]}}}))
    # and its answers' timings, as the stats worker keeps them
    from service import beyond
    from service import frontier as sf
    rng = random.Random(29)
    rows = timed(runs(30, 4, 12, 6), rng, (30.0, 120.0))
    t = sf.task_dir(d, GPQA)
    t.mkdir(parents=True, exist_ok=True)
    (t / f"samples_{GPQA}_2026-10-26T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": f"q{k}"}, "resps": [[x] for x in r["texts"]],
                    "filtered_resps": [""] * len(r["texts"]), "frontier": r["runs"]}) + "\n"
        for k, r in enumerate(rows)))
    (t / "setup.json").write_text(json.dumps({"concurrency": {"in_flight": 8, "server_slots": 8}}))
    beyond.kept(d, GPQA, compute=True)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_: str, width: int, scheme: str, sel: str) -> None:
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel, timeout=30000)


def shot(page, loc, path: Path) -> None:
    SCREENS.mkdir(parents=True, exist_ok=True)
    loc.evaluate("e => window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 70)")
    loc.screenshot(path=str(path))


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_measured_q4_k_m_row_on_the_on_device_report(live, page, width, scheme):
    go(page, live, "tab=reports&report=on-device", width, scheme,
       "[data-report-section='od-keep']")
    if width > 700:
        row = page.locator(f"tr[data-od-row='{Q4}']")
        assert row.get_attribute("data-mt-group") == "open"
        assert row.get_attribute("data-od-measured") == "1"
        mark = row.locator("[data-mt-mark]")
        assert mark.get_attribute("data-mt-mark") == "here"           # a filled dot
        assert "var(--fam-nvidia)" in mark.get_attribute("style")
        line = row.locator("[data-mt-line]").inner_text()
        assert "measured here on a V100" in line and "published by" not in line
        g = page.locator(f"[data-od-cell='{Q4}|gpqa']")
        assert g.locator(".mt-n").inner_text() == "52.0"              # no letter: measured
        keep = g.locator("[data-od-keep]")
        assert keep.inner_text() == "94.5% of our Q8_0"
        assert keep.get_attribute("title") == "of our Q8_0, standing in for BF16"
        assert CAPPED in g.get_attribute("data-tip")
        mm = page.locator(f"[data-od-cell='{Q4}|mmlupro']")
        assert mm.inner_text().strip() == "–" and mm.get_attribute("title") == "not run yet"
        assert page.locator(f"[data-od-fact='{Q4}|memory'] .mt-n").inner_text().startswith(
            "6.18 GB")
        # NVIDIA's own row, as published: its letter, its base, never the measured one's
        pub = page.locator(f"[data-od-cell='{PUB}|ifbench']")
        assert pub.locator(".mt-n").inner_text() == "64.0N"
        assert page.locator(f"[data-od-keep='{PUB}|ifbench']").inner_text() == (
            "150.0% of its base")
        assert page.locator(f"tr[data-od-row='{Q8}']").count() == 0      # a base, never a row
    # What each keeps: the Q4_K_M a filled dot in NVIDIA's colour, of its Q8_0
    dot = page.locator(f"[data-od-keep-dot='gpqa:{Q4}']")
    assert dot.evaluate("e => e.tagName") == "circle"
    assert dot.get_attribute("fill") == "var(--fam-nvidia)"
    base = page.locator(f"[data-od-base='gpqa:{Q4}']")
    assert base.evaluate("e => [...e.querySelectorAll('tspan')].map(t => t.textContent)"
                         ".join(' ')") == "of our Q8_0, standing in for BF16"
    # MMLU-Pro, not run yet: a "–" in its panel, said on hover
    none = page.locator(f"[data-od-keep-none='mmlupro:{Q4}']")
    assert none.evaluate("e => e.firstChild.textContent") == "–" and none.locator(
        "title").text_content() == (
        f"{Q4}: not run yet")
    note = page.locator("[data-od-measured-note]").inner_text()
    assert "standing in for BF16" in note and "measured here" in note
    # the memory chart: its dot, and its name in the legend, as ours are
    assert page.locator(f"[data-od-mem-dot='gpqa:{Q4}']").evaluate("e => e.tagName") == "circle"
    assert page.locator(f"[data-od-leg='{Q4}']").inner_text().strip() == "Bolt E4B · Q4_K_M"
    assert no_sideways(page) and page.errors == []
    for sec in ("od-table", "od-keep", "od-memory"):
        shot(page, page.locator(f"[data-report-section='{sec}']"),
             SCREENS / f"{sec}-{width}-{scheme}.png")


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark")])
def test_frontier_shows_both_measured_rows_their_cap_on_hover_and_once(live, page, width,
                                                                       scheme):
    go(page, live, "tab=models&view=frontier&cols=gpqa,ifeval,ifbench", width, scheme,
       "[data-frontier-table]")
    for mid in (Q4_ID, Q8_ID):
        row = page.locator(f"tr[data-fr-row='{mid} · thinking']")
        assert row.count() == 1, mid
        assert row.get_attribute("data-mt-group") == "open"
        # NVIDIA's colour, as on the On-device report
        assert "var(--fam-nvidia)" in row.locator("[data-mt-mark]").get_attribute("style")
        cell = row.locator("[data-fr-here]").first
        assert cell.get_attribute("data-fr-capped") == "1"
        assert CAPPED in cell.get_attribute("data-tip")
    names = page.eval_on_selector_all("tr[data-fr-row*='Bolt-E4B'] [data-mt-name]",
                                      "ns => ns.map(n => n.textContent)")
    assert sorted(names) == [Q4, Q8]
    # NVIDIA's published numbers stay their own row
    assert page.locator("[data-fr-row^='reported/'][data-fr-row*='nemotron']").count() == 1
    note = page.locator("[data-frontier-cap-note]")
    assert note.count() == 1
    assert note.inner_text().startswith("NVIDIA Nemotron 3.5 Bolt E4B holds 64K tokens of context")
    assert f"“{CAPPED}”" in note.inner_text()
    # no other row says it
    others = page.eval_on_selector_all("[data-fr-here][data-fr-capped]",
                                       "cs => cs.map(c => c.dataset.frHere)")
    assert others and all("Bolt-E4B" in x for x in others)
    assert no_sideways(page) and page.errors == []
    # the two rows in view, in the table's own scrolling box
    page.locator(f"tr[data-fr-row='{Q4_ID} · thinking']").evaluate(
        "e => e.scrollIntoView({block: 'center'})")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / f"frontier-{width}-{scheme}.png"))


@pytest.mark.parametrize("width,scheme", SIZES)
def test_one_model_page_holds_both_builds_and_the_published_numbers(live, page, width, scheme):
    go(page, live, "model=" + quote(Q4_ID + " · thinking", safe=""), width, scheme,
       "[data-builds]")
    page.wait_for_selector("[data-builds-table]", state="attached", timeout=30000)
    card_ = page.locator("[data-builds]")
    assert card_.locator("h2").inner_text() == ("NVIDIA Nemotron 3.5 Bolt E4B: its builds, and "
                                                "as NVIDIA publishes it")
    said = card_.locator("[data-builds-said]").inner_text()
    assert said.startswith("Measured here on a V100: Q4_K_M and Q8_0 (standing in for BF16).")
    assert "never mixed with ours" in said and "65,536" in said
    if width > 700:
        rows = page.eval_on_selector_all("[data-builds-table] tr[data-builds-row]",
                                         "rs => rs.map(r => r.dataset.buildsRow)")
        assert rows == [Q4_ID, Q8_ID, "published"]
        this = page.locator("tr[data-builds-this]")
        assert this.get_attribute("data-builds-row") == Q4_ID
        q8 = page.locator(f"tr[data-builds-row='{Q8_ID}'] [data-mt-line]").inner_text()
        assert "standing in for BF16" in q8 and "measured here on a V100" in q8
        g = page.locator(f"[data-builds-cell='{Q4_ID}|gpqa']")
        assert g.locator(".mt-n").inner_text() == "52.0" and CAPPED in g.get_attribute("data-tip")
        pub = page.locator("[data-builds-cell='published|ifbench']")
        assert pub.locator(".mt-n.mt-pub").inner_text() == "64.0N"
        assert page.locator("tr[data-builds-row='published'] [data-mt-mark]").get_attribute(
            "data-mt-mark") == "pub"
        assert page.locator("[data-builds-cell='published|gpqa']").inner_text().strip() == "–"
    assert no_sideways(page) and page.errors == []
    shot(page, card_, SCREENS / f"model-builds-{width}-{scheme}.png")
    # the published model's own page holds the same card
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote("reported/nvidia/nvidia-nemotron-3.5-bolt-e4b",
                                                safe=""))
    page.wait_for_selector("[data-reported-model] h1", timeout=30000)
    page.wait_for_selector("[data-builds-table]", state="attached", timeout=30000)
    assert page.locator("h1.mtitle").inner_text() == PUB
    assert page.locator("[data-builds-this]").count() == 0
    assert no_sideways(page) and page.errors == []
    if width == 1280 and scheme == "light":
        page.screenshot(path=str(SCREENS / "reported-page-1280-light.png"), full_page=True)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (400, "dark")])
def test_its_speed_is_a_v100s_and_its_dollars_own_box(live, page, width, scheme):
    go(page, live, "model=" + quote(Q4_ID + " · thinking", safe=""), width, scheme,
       "[data-kind-block='beyond']")
    blk = page.locator("[data-kind-block='beyond']")
    blk.locator("summary").click()
    page.wait_for_selector("[data-by-part='speed']", timeout=30000)
    part = page.locator("[data-by-part='speed']")
    said = part.locator("[data-by-speed-sentence]").inner_text()
    assert said.startswith(f"On a {V100}, NVIDIA Nemotron 3.5 Bolt E4B"), said
    heads = part.locator("h4").all_inner_texts()
    assert heads[0] == f"Speed on a {V100}"
    assert heads[1] == f"Each benchmark, with 8 answers at once on a {V100}"
    usd = page.locator(f"[data-by-usd='{GPQA}']")
    assert usd.inner_text() == "own box" and usd.get_attribute("data-by-own-box") == "1"
    assert "$" not in part.inner_text()
    assert no_sideways(page) and page.errors == []
    shot(page, part, SCREENS / f"speed-{width}-{scheme}.png")


def boxes_reply(now: float) -> dict:
    """what Runs reads of the boxes, as the board serves it: one own box of
    dgx-03 paused for someone else's process, with its why"""
    box = {"label": "D3.0", "step": "D3.0-1", "status": "paused", "words": f"paused: {WHY}",
           "n": 120, "of": 792, "shard": None, "notes": [], "heard": now, "behind": False}
    return {"boxes": [], "posted_at": now, "every": 180, "stale_min": None,
            "next_at": now + 180,
            "runs": [{"id": f"rented:{Q4_ID}|{GPQA}|on", "hf_id": Q4_ID, "tasks": GPQA,
                      "parity": False, "thinking": "on", "status": "paused",
                      "status_words": f"paused: {WHY}", "n": 120, "of": 792, "benchmarks": [],
                      "boxes_n": 1, "finish": None, "attention": [], "slow": [],
                      "heard": now, "behind": False, "started_at": "2026-10-26T06:00:00Z",
                      "own": True, "where": "our own GPU · box D3.0", "boxes": [box]}]}


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light")])
def test_runs_shows_the_own_box_paused_waiting_not_failed(live, page, width, scheme):
    reply = json.dumps(boxes_reply(time.time()))
    page.route("**/api/frontier/boxes*", lambda r: r.fulfill(
        status=200, content_type="application/json", body=reply))
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    row = page.locator(f"[data-rented-run='rented:{Q4_ID}|{GPQA}|on']")
    row.wait_for(timeout=30000)
    chip = row.locator("[data-stage]")
    assert chip.get_attribute("data-stage") == "paused" and chip.inner_text() == "paused"
    assert "st-paused" in chip.get_attribute("class") and "st-failed" not in chip.get_attribute(
        "class")
    assert chip.get_attribute("title") == WHY
    assert row.locator("[data-rented-paused]").inner_text() == f"waiting for its GPU: {WHY}"
    if width > 700:
        own = row.locator("[data-rented-own]")
        assert own.inner_text() == "own box" and "$" not in row.inner_text()
        assert row.locator("[data-run-where]").inner_text() == "our own GPU · box D3.0"
    # a paused run is among the active ones, never the failed
    page.evaluate("() => { state.qStatus = 'active'; state.queueRedraw(); }")
    row.wait_for()
    assert no_sideways(page) and page.errors == []
    shot(page, page.locator("table[data-queue-table]"), SCREENS / f"runs-{width}-{scheme}.png")
