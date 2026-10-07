"""17g on the page: a benchmark whose two rows are scored by different
graders says so on both rows' cells in the Frontier view, and AI models'
grading card offers the other row's regrade with its price — taken, the dry
run shows it, and Undo takes it back. Answers and grades invented on disk;
nothing is sent."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
import make_fixture as mf
from conftest import set_name
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase17g"
SQA = "simpleqa_epoch"
ROWS = {"served/split-a": "openai/gpt-4.1-2025-04-14",
        "served/split-b": "google/gemini-2.5-flash-20250617"}
# 17g, point 22: two models whose names differ at their front
NAMES = {"served/split-a": "Bonsai 2 27B (ternary, PQ2_0)",
         "served/split-b": "Qwen3.6-35B-A3B-K4-LDA (phone build)"}
HOW = {"served/split-a": ("the ternary PQ2_0 GGUF on llama-server b6500, flash attention, a q8_0 "
                          "cache, 8 slots of 16,384, served over its OpenAI endpoint"),
       "served/split-b": ("built from the k4 branch with LDA routing, MTP off, flash attention, a "
                          "q8_0 cache, --cpu-moe, as the phone runs it, over its OpenAI endpoint")}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """two rows of SimpleQA, every answer graded — one by GPT-4.1, the other by
    Gemini 2.5 Flash, the grader chosen now"""
    import service.app as appmod
    from service import ai_models, config, db, served
    from service import frontier as sf
    from service import frontier_grade as fgr
    saved_key, was = config.OPENROUTER_API_KEY, db.ai_get(fgr._setting("simpleqa"))
    config.OPENROUTER_API_KEY = "test-key"
    items = [{"id": str(k), "question": f"Split fact {k}?", "answer": f"A{k}", "subject": ""}
             for k in range(4)]
    p = fb._cache(config.BENCH_ROOT, SQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(have or items))
    items = have or items
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cached = json.loads(p.read_text()) if p.exists() else {"models": []}
    known = {m["id"] for m in cached.get("models") or []}
    cached["models"] = (cached.get("models") or []) + [m for m in [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": ROWS["served/split-a"],
         "price_in": 2.0, "price_out": 8.0, "reasons": False},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash",
         "version": ROWS["served/split-b"], "price_in": 0.3, "price_out": 2.5,
         "reasons": True}] if m["id"] not in known]
    cached["at"] = time.time()
    p.write_text(json.dumps(cached))
    for sid, version in ROWS.items():
        db.served_put({"id": sid, "name": NAMES[sid], "base_url": "", "key": "",
                       "how": HOW[sid], "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                       "phone": sid.endswith("-b"), "pin": {"file": f"{sid[-1]}.gguf"},
                       "by": "masein", "at": 0})
        row = Path(config.OUT_DIR) / sid.replace("/", "__")
        d = sf.task_dir(row, SQA)
        d.mkdir(parents=True, exist_ok=True)
        served.write_meta(db.served_get(sid))              # its name, and how it's served
        answers = {it["id"]: f"I think {it['answer']}." for it in items}
        (d / sf.ANSWERS).write_text("".join(json.dumps(
            {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n"
            for q, a in answers.items()))
        who = fgr.grader_record("simpleqa", {"id": version.rsplit("-", 3)[0], "version": version,
                                             "provider": "p", "provider_name": "Prov"}, "masein")
        (d / sf.GRADES).write_text(json.dumps({"grader": who, "graders": [who], "refused": {},
                                               "items": {f"{q}#0": {
                                                   "ok": True, "words": "correct",
                                                   "by": version,
                                                   "prompt_sha256": who["prompt_sha256"],
                                                   "answer_sha256": sf.answer_sha(a), "at": 0}
                                                   for q, a in answers.items()}}))
        sf.score_task(row, SQA, db.served_get(sid))
        # a benchmark Compare lists, for its table to have a row
        mf.write_task(Path(config.OUT_DIR), sid, "instruct", 0, "good_skill", "arc_easy",
                      mf.make_docs("arc_easy"), 7)
    db.size_set("served/split-a", 27e9, None, "masein")
    db.size_set("served/split-b", 35e9, 3e9, "masein")
    # 90 runs, a few of them imports — every one listed (the latest 100 are):
    # Runs' pages, its where filter, and "#1", which begins #10 … #19
    for k in range(90):
        sid = db.add("served/split-a", "instruct", "frontier", "masein", f"run {k}",
                     status="done")
        if k % 10 == 0:
            db.update(sid, where_ran="rented GPU · RTX 5090 · box A1")
    db.ai_set(fgr._setting("simpleqa"), {
        "kind": "openrouter", "id": "google/gemini-2.5-flash", "version": ROWS["served/split-b"],
        "name": "Google: Gemini 2.5 Flash", "provider": "p", "provider_name": "Prov",
        "price_in": 0.3, "price_out": 2.5}, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    # the board as the other tests expect it: no regrade asked, the grader as it was
    fgr.regrade_row("served__split-a", SQA, "masein", undo=True)
    db.ai_set(fgr._setting("simpleqa"), was, "masein")
    config.OPENROUTER_API_KEY = saved_key
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def test_both_rows_say_their_graders_differ_in_the_frontier_view(live, page):
    go(page, live, "tab=models&chip=frontier", "[data-frontier-table]")
    page.wait_for_function("document.querySelectorAll('[data-fr-grader-differs]').length >= 2")
    said = {page.locator("[data-fr-grader-differs]").nth(i).get_attribute(
        "data-fr-grader-differs"): page.locator("[data-fr-grader-differs]").nth(i).inner_text()
        for i in range(page.locator("[data-fr-grader-differs]").count())}
    assert set(ROWS) <= set(said), said
    assert said["served/split-a"].startswith(f"graded by {ROWS['served/split-a']}; ")
    assert f"{NAMES['served/split-b']} by {ROWS['served/split-b']}" in said["served/split-a"]
    assert all("one benchmark, two graders" in w for w in said.values())
    assert page.errors == []


def test_the_card_offers_the_other_rows_regrade_priced_and_undo_takes_it_back(live, page):
    go(page, live, "tab=ai", "[data-frontier-grading]:not([data-frontier-grading='loading'])")
    set_name(page, "masein")
    card = page.locator("[data-frontier-grading]")
    page.wait_for_selector(f"[data-frontier-mismatch='{SQA}']")
    offer = card.locator(f"[data-frontier-offer='{SQA}|served__split-a']")
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(card.locator(f"[data-frontier-mismatch='{SQA}']"), SCREENS / "grading-offer.png")
    text = offer.inner_text()
    assert text.startswith(f"{NAMES['served/split-a']}: grade it again by "
                           f"{ROWS['served/split-b']}, as the others are — 4 answers, about $"), text
    # the row graded by the grader chosen now isn't offered
    assert card.locator(f"[data-frontier-offer='{SQA}|served__split-b']").count() == 0
    offer.locator("button").click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    assert "at the next Start" in card.locator(
        f"[data-frontier-offer='{SQA}|served__split-a']").inner_text()
    # the dry run: the row, graded again whole, priced before Start
    est = card.locator("[data-frontier-est='simpleqa|served/split-a']")
    assert est.count() == 1 and "so that the rows compared share" in est.inner_text()
    # Undo: nothing waits for it any more
    card.locator(f"[data-frontier-offer='{SQA}|served__split-a'] button").click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    assert card.locator("[data-frontier-est='simpleqa|served/split-a']").count() == 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 18, 21: Runs — the boxes' list redrawn by the poll, "#12" is run 12, and the
# where filter back to page 1
# ---------------------------------------------------------------------------

def test_18_the_poll_redraws_the_boxes_list(live, page):
    import import_frontier as imf
    now = time.time()
    step = {"label": "A5", "model": "served/split-b", "step": "A5-1", "thinking": "on",
            "tasks": ["mmlupro_tiger"], "state": "asking", "at": now - 60, "seen_at": now,
            "line": "MMLU-Pro 3,000 of 12,032 · 5.3 s an answer · 13.3 h left", "sessions": 1,
            "reachable": True, "safe": False}
    imf.store_boxes([step])
    run = "[data-rented-run='rented:served/split-b|mmlupro_tiger|on'] [data-rented-count]"
    go(page, live, "tab=runs", run)
    imf.store_boxes([step, {**step, "label": "A6", "step": "A6-1"}])
    # the poll's redraw: the table — and, 17g, the list (308fcf3: only a full render).
    # 17i: the second box merged into its run's row
    page.evaluate("state.boxesAt = 0; state.queueRedraw()")
    page.wait_for_function(f"(document.querySelector(\"{run}\") || {{}}).textContent"
                           " === '6,000 of 24,064 · 2 boxes'", timeout=10000)
    assert page.errors == []


def test_21_run_12_is_run_12_and_the_where_filter_goes_back_to_page_1(live, page):
    go(page, live, "tab=runs", "[data-queue-table]")
    listed = page.evaluate("state.queue.map(r => r.id)")
    twelve = next(n for n in sorted(listed)
                  if any(str(o).startswith(str(n)) and o != n for o in listed))
    page.fill("input[aria-label='filter queue']", f"#{twelve}")
    page.wait_for_function("document.querySelectorAll('[data-queue-row]').length === 1")
    assert page.locator(f"[data-queue-row='{twelve}']").count() == 1   # 308fcf3: #120 … too
    page.fill("input[aria-label='filter queue']", "")
    page.wait_for_function("document.querySelectorAll('[data-queue-row]').length > 1")
    page.locator("[data-pager='queue'] [data-page-next]").first.click()
    page.locator("[data-pager='queue'] [data-page-next]").first.click()
    assert page.evaluate("state.pg.queue.page") == 3
    page.get_by_label("where filter").click()
    page.locator("[role=listbox][aria-label='where filter'] [data-value='rented']").click()
    page.wait_for_function("state.qWhere === 'rented'")
    assert page.evaluate("state.pg.queue.page") == 1                   # 308fcf3: 3
    assert page.locator("[data-queue-row]").count() > 0
    assert page.errors == []


# ---------------------------------------------------------------------------
# 22: Compare — a column's name whole when it has room, cut to what tells it
# apart when it hasn't; the lines above say the kind and the size
# ---------------------------------------------------------------------------

def heads(page) -> dict:
    return {h.get_attribute("data-cmp-head"): h.inner_text()
            for h in page.locator("[data-cmp-head]").all()}


def test_22_compare_names_whole_with_room_and_cut_at_the_front_that_tells_them_apart(live, page):
    ids = ",".join(NAMES)
    go(page, live, f"tab=models&view=compare&m={ids}", "[data-compare='2']", width=1400)
    page.wait_for_selector("[data-cmp-head]")
    page.wait_for_timeout(200)
    got = heads(page)
    # wide: each whole (308fcf3: "…27B (ternary, PQ2_0)", "…K4-LDA (phone build)")
    assert {k: v.lower() for k, v in got.items()} == {k: v.lower() for k, v in NAMES.items()}
    # the lines above: the kind and the size, how it's served on hover
    kinds = {k.get_attribute("data-cmp-kind"): k for k in page.locator("[data-cmp-kind]").all()}
    assert kinds["served/split-a"].inner_text() == " · served · 27B"
    assert kinds["served/split-b"].inner_text() == " · phone build · 35B · 3B active"
    assert kinds["served/split-b"].get_attribute("title") == HOW["served/split-b"]
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(page.locator("[data-compare='2']"), SCREENS / "compare-wide.png")
    # narrower: cut, each keeping its own front
    go(page, live, f"tab=models&view=compare&m={ids}", "[data-compare='2']", width=720)
    page.wait_for_selector("[data-cmp-head]")
    page.wait_for_timeout(200)
    got = heads(page)
    a, b = got["served/split-a"], got["served/split-b"]
    assert a.endswith("…") and b.endswith("…"), got
    assert a.lower().startswith("bonsai 2") and b.lower().startswith("qwen3.6"), got
    assert NAMES["served/split-a"].lower().startswith(a[:-1].lower())
    steady_shot(page.locator("[data-compare='2']"), SCREENS / "compare-720.png")
    # a phone's width: shorter still, and still each its own front
    go(page, live, f"tab=models&view=compare&m={ids}", "[data-compare='2']", width=420)
    page.wait_for_selector("[data-cmp-head]")
    page.wait_for_timeout(200)
    got = heads(page)
    assert got["served/split-a"].lower().startswith("b") and \
        got["served/split-b"].lower().startswith("q"), got
    assert page.errors == []


def test_22_head_names_drop_what_they_share_at_either_end(live, page):
    go(page, live, "tab=models", "[data-lb-card]")
    got = page.evaluate("""() => {
      const fits = n => x => x.length <= n;
      const two = ['Bonsai 2 27B (ternary, PQ2_0)', 'Qwen3.6-35B-A3B-K4-LDA (phone build)'];
      const one = ['Qwen3.6-35B-A3B Q4_K_M · setup fast-decode',
                   'Qwen3.6-35B-A3B Q4_K_M · setup long-context'];
      return [[...headNames(two, fits(60)).values()], [...headNames(two, fits(16)).values()],
              [...headNames(one, fits(20)).values()]];
    }""")
    assert got[0] == ["Bonsai 2 27B (ternary, PQ2_0)", "Qwen3.6-35B-A3B-K4-LDA (phone build)"]
    assert got[1][0].startswith("Bonsai 2") and got[1][1].startswith("Qwen3.6")
    assert got[2] == ["…fast-decode", "…long-context"]
