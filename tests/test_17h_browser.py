"""17h on the page, part 1, point 4: a grader chosen again that graded
everything has its Start — "nothing to send, the grades switched" — and the
dry run says what Start leaves the score as. Answers and grades invented on
disk; nothing is sent."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import frontier as fb
from conftest import set_name

pytestmark = pytest.mark.dashboard
SQA = "simpleqa_epoch"
SID = "served/switch-box"
GPT_V, GEMINI_V = "openai/gpt-4.1-2025-04-14", "google/gemini-2.5-flash-20250617"


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """one row of SimpleQA graded whole by Gemini 2.5 Flash, GPT-4.1's grades
    of every answer kept under its name, and GPT-4.1 the grader chosen now"""
    import service.app as appmod
    from service import ai_models, config, db, served
    from service import frontier as sf
    from service import frontier_grade as fgr
    saved = config.OPENROUTER_API_KEY
    config.OPENROUTER_API_KEY = "test-key"
    items = [{"id": str(k), "question": f"Switch fact {k}?", "answer": f"A{k}", "subject": ""}
             for k in range(4)]
    p = fb._cache(config.BENCH_ROOT, SQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items))
    p = ai_models._cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": time.time(), "models": [
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "version": GPT_V,
         "price_in": 2.0, "price_out": 8.0, "reasons": False},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash",
         "version": GEMINI_V, "price_in": 0.3, "price_out": 2.5, "reasons": True}]}))
    db.served_put({"id": SID, "name": "switch box", "base_url": "", "key": "", "how": "x",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
                   "pin": {"file": "s.gguf"}, "by": "masein", "at": 0})
    rec = db.served_get(SID)
    served.write_meta(rec)
    row = Path(config.OUT_DIR) / SID.replace("/", "__")
    d = sf.task_dir(row, SQA)
    d.mkdir(parents=True, exist_ok=True)
    answers = {it["id"]: f"I think {it['answer']}." for it in items}
    (d / sf.ANSWERS).write_text("".join(json.dumps(
        {"id": q, "epoch": 0, "answer": a, "finish": "stop"}) + "\n" for q, a in answers.items()))

    def grades(version):
        who = fgr.grader_record("simpleqa", {"id": version.rsplit("-", 3)[0],
                                             "version": version}, "masein")
        return who, {f"{q}#0": {"ok": True, "words": "correct", "by": version,
                                "prompt_sha256": who["prompt_sha256"],
                                "answer_sha256": sf.answer_sha(a), "at": 0}
                     for q, a in answers.items()}
    gem, now = grades(GEMINI_V)
    gpt, kept = grades(GPT_V)
    (d / sf.GRADES).write_text(json.dumps({
        "grader": gem, "graders": [gpt, gem], "items": now, "refused": {},
        "kept": {fgr._key((GPT_V, gpt["prompt_sha256"])): {"items": kept, "refused": {}}}}))
    sf.score_task(row, SQA, rec)
    db.ai_set(fgr._setting("simpleqa"), {
        "kind": "openrouter", "id": "openai/gpt-4.1", "version": GPT_V, "name": "OpenAI: GPT-4.1",
        "provider": "p", "provider_name": "Prov", "price_in": 2.0, "price_out": 8.0}, "masein")
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield d
    config.OPENROUTER_API_KEY = saved


def test_4_a_switch_with_nothing_to_send_has_its_start_and_says_what_it_leaves(live, page,
                                                                             board):
    from service import frontier as sf
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=ai")
    page.wait_for_selector("[data-frontier-grading]:not([data-frontier-grading='loading'])")
    set_name(page, "masein")
    card = page.locator("[data-frontier-grading]")
    row = card.locator(f"[data-frontier-est='simpleqa|{SID}']")
    row.wait_for()
    assert "4 grades it gave before are used again" in row.inner_text()
    assert card.locator("[data-frontier-after='final']").inner_text() == \
        "Start makes its score: final"
    start = card.locator("[data-frontier-start]")                 # 0adb522: no button
    assert start.inner_text() == "Start grading: nothing to send, the grades switched"
    start.click()
    page.wait_for_function("!state.ai.frgBusy", timeout=15000)
    fs = sorted(board.glob("results_*.json"))
    assert json.loads(fs[-1].read_text())["frontier"]["grader"]["version"] == GPT_V
    assert sf.read_grades(board)["items"]["0#0"]["by"] == GPT_V
    assert page.errors == []


def test_13_the_poll_brings_the_boxes_list_up_to_date_on_its_own(live, page):
    """part 2, point 13: the list is read again every half minute while it is
    on the page — the poll redrew it only when a run on the board changed"""
    import import_frontier as imf
    now = time.time()
    step = {"label": "A5", "model": "served/switch-box", "step": "A5-1", "thinking": "on",
            "tasks": ["mmlupro_tiger"], "state": "asking", "at": now - 60, "seen_at": now,
            "line": "MMLU-Pro 3,000 of 12,032 · 13.3 h left", "sessions": 1, "reachable": True,
            "safe": False, "box_id": "a"}
    imf.store_boxes({"steps": [step], "asked": ["a"]})
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-rented-box='A5|A5-1']")
    imf.store_boxes({"steps": [step, {**step, "label": "A6", "step": "A6-1", "box_id": "b"}],
                     "asked": ["a", "b"]})
    page.evaluate("state.boxesAt = 0")                     # half a minute on
    page.wait_for_selector("[data-rented-box='A6|A6-1']", timeout=12000)  # 0adb522: never
    assert page.errors == []


def test_19_a_models_page_shows_its_public_mark_and_clears_it(live, page):
    """part 4, point 19: --public-weights given by mistake is seen on the
    model's page, and cleared there; marking asks first"""
    from service import db
    db.public_set(SID, True, "masein")
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#model={SID}")
    set_name(page, "masein")
    line = page.locator("[data-public-weights]")
    line.wait_for()                                         # 0adb522: nothing shows it
    assert line.get_attribute("data-public-weights") == "1"
    assert line.inner_text().startswith("Public weights — marked by masein on ")
    page.locator("[data-public-weights-toggle='clear']").click()
    page.wait_for_selector("[data-public-weights='0']")
    assert SID not in db.public_all()
    # marking asks first, in the page
    page.locator("[data-public-weights-toggle='set']").click()
    page.locator("[data-public-weights-yes]").click()
    page.wait_for_selector("[data-public-weights='1']")
    assert SID in db.public_all()
    assert page.errors == []


def test_24_three_columns_cut_alike_keep_their_middles_and_twins_are_numbered(live, page):
    """part 5, point 24: names that differ only in their middle, cut to a
    narrow column, keep the middle; two names alike to the letter are
    numbered — never a fixed 22 characters, never one label for two"""
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("[data-lb-card]")
    got = page.evaluate("""() => {
      const fits = n => x => x.length <= n;
      const mid = ['Qwen3.6-35B-A3B-k4LDA-phone', 'Qwen3.6-35B-A3B-k8LDA-phone',
                   'Qwen3.6-35B-A3B-k16LDA-phone'];
      const twins = ['Qwen3.6 build lookahead', 'Qwen3.6 build lookahead', 'Gemma 4 E2B'];
      return [[...headNames(mid, fits(5)).values()], [...headNames(twins, fits(18)).values()],
              [...headNames(mid, fits(40)).values()]];
    }""")
    # cut to the same end, they keep what differs between them
    assert got[0] == ["…k4L…", "…k8L…", "…k16…"]            # 0adb522: one label, 22 long
    assert got[1][:2] == ["Qwen3.6 build… #1", "Qwen3.6 build… #2"] and got[1][2] == "Gemma 4 E2B"
    assert got[2] == ["Qwen3.6-35B-A3B-k4LDA-phone", "Qwen3.6-35B-A3B-k8LDA-phone",
                      "Qwen3.6-35B-A3B-k16LDA-phone"]
    assert page.errors == []
