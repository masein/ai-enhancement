"""28 in a real browser: the assistant's panel at 1280 and 400 wide, light
and dark — a question that opens a page, one that draws a chart, one that
clarifies, one that refuses; ⌘K / Ctrl+K; the thumbs; "slower now" while a
run uses the model, and the model down; nothing for a name not in
ASSISTANT_USERS; and 28's sort and thinking in a Models link. The model is a
fake answering each bank question with its expected action; the world is
tests/assistant_world.py's."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import assistant_world as aw
from conftest import set_name
from service import assistant, config, telemetry
from test_21d_browser import no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase28"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]
BANK = {b["q"]: b for b in (json.loads(x) for x in assistant.BANK_FILE.read_text(
    encoding="utf-8").splitlines() if x.strip())}
CTRL = {"slow": 0.0, "busy": False, "down": False, "calls": 0}
SIX = ["GPQA Diamond", "OTIS Mock AIME", "MMLU-Pro", "HLE", "IFEval", "IFBench"]
# 28g: the panel's own questions, each with what the model writes — names
# as written (the board matches them)
PANEL = {
    "step8804 on the frontier benchmarks": {"action": "compare", "models": ["step8804"],
                                            "benchmarks": SIX},
    "Ternary Bonsai 2 2B": {"action": "compare", "models": ["step8804", "Ternary Bonsai 2 2B"],
                            "benchmarks": SIX},
    "q2's gpqa?": {"action": "compare", "models": ["q2"], "benchmarks": ["GPQA Diamond"]},
    "how does the step one do on HLE": {"action": "compare", "models": ["the step one"],
                                        "benchmarks": ["HLE"]},
    # the miss the chips are for: "sorted by OTIS" dropped
    "frontier, last 7 days, sorted by OTIS": {"action": "open_page", "page": "models_table",
                                              "view": "frontier", "when": "7d"},
    "best of ours on HLE": {"action": "best_on", "benchmark": "HLE", "among": "ours"},
    "step8804, step4000, step1000, q4 and q2 on gpqa and hle": {"action": "compare", "models": [
        "step8804", "step4000", "step1000", "q4", "q2"], "benchmarks": ["GPQA Diamond", "HLE"]},
    "step8804 vs q4 on gpqa and hle": {"action": "compare", "models": ["step8804", "q4"],
                                       "benchmarks": ["GPQA Diamond", "HLE"]},
}


def fake_call(system, user, sch, prov=None):
    """the bank's action for its question, or the panel's own; a clicked
    choice opens that model"""
    CTRL["calls"] += 1
    if CTRL["down"]:
        raise assistant.Down("connection refused")
    time.sleep(CTRL["slow"])
    q = user.rsplit("Message: ", 1)[1]
    b = BANK.get(q)
    want = b["want"] if b else PANEL.get(q) or {"action": "open_page", "page": "model_page",
                                                "model": q}
    first = {k: v for k, v in want.items() if k not in ("also", "then")}
    return {"raw": json.dumps({"do": first, **({"then": want["then"]} if want.get("then")
                                               else {})}), "ms": 5, "usage": {}}


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    mp = pytest.MonkeyPatch()
    saved = telemetry.NAMES
    aw.world()
    mp.setattr(config, "ASSISTANT_USERS", "masein")
    mp.setattr(assistant, "call", fake_call)
    mp.setattr(assistant, "busy", lambda: CTRL["busy"])
    mp.setattr(assistant, "up", lambda: not CTRL["down"])
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    aw.unwind(saved)
    mp.undo()
    appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.fixture(autouse=True)
def calm():
    CTRL.update(slow=0.0, busy=False, down=False)
    yield
    CTRL.update(slow=0.0, busy=False, down=False)


def board_at(page, live, width=1280, scheme="light", hash_="#tab=home", name="masein"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/" + hash_)
    page.locator("#view").wait_for()
    set_name(page, name)


def answered(page, n: int):
    """the panel's n-th answer is in (28g: older ones fold, so cards aren't counted)"""
    page.wait_for_function(f"!ASK.waiting && ASK.msgs.filter(m => m.role === 'bot').length >= {n}",
                           timeout=20000)
    return page.locator("[data-ask-bot]").last


def ask(page, q: str, kind: str):
    n = page.evaluate("ASK.msgs.filter(m => m.role === 'bot').length")
    box = page.locator("[data-ask-input]")
    box.fill(q)
    box.press("Enter")
    got = answered(page, n + 1)
    assert got.get_attribute("data-ask-bot") == kind, (q, got.get_attribute("data-ask-bot"))
    return got


def shot(page, name: str):
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-ask-panel]").screenshot(path=str(SCREENS / name))


@pytest.mark.parametrize("width,scheme", SIZES)
def test_the_panel_opens_a_page_draws_clarifies_and_refuses(live, page, width, scheme):
    board_at(page, live, width, scheme)
    if width > 720:
        page.locator("[data-ask-open]").click()
    else:                                       # a phone: Ask is in Menu ▾, with the places
        page.wait_for_function("ASK.on")
        page.locator("#menuBtn").click()
        page.locator("[data-place='ask']").click()
    panel = page.locator("[data-ask-panel]")
    panel.wait_for()
    # 28g: 400 wide over the page; a phone's whole width
    w = panel.evaluate("e => e.getBoundingClientRect().width")
    assert (w == 400) if width > 700 else (w == width), w
    # examples in four groups, two at most each, every one the bank's
    groups = page.locator("[data-ask-group]")
    assert groups.count() == 4
    assert [g.get_attribute("data-ask-group") for g in groups.all()] == [
        "Numbers", "Pages", "Phones", "Runs"]
    for g in groups.all():
        qs = g.locator("[data-ask-example]").all_inner_texts()
        assert 1 <= len(qs) <= 2 and all(q in BANK for q in qs), qs
    shot(page, f"empty-{width}-{scheme}.png")
    # a page: the board's own link, opened; the panel stays
    got = ask(page, "frontier tab, ours only, sorted by HLE", "open")
    page.wait_for_function("location.hash === '#tab=models&view=frontier&only=ours&sort=hle'")
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    assert got.locator("[data-ask-copy]").get_attribute("data-ask-link") == (
        "#tab=models&view=frontier&only=ours&sort=hle")
    assert got.locator("[data-ask-say]").inner_text() == "Opened Models · Frontier"
    assert got.locator("[data-ask-chip]").count() == 3      # the tab, ours only, the sort
    assert panel.is_visible()
    # a chart, from Across training's own panel
    got = ask(page, "chart GPQA across the checkpoints", "chart")
    got.locator("[data-ask-chart='checkpoints'] svg").wait_for(timeout=20000)
    # a table of 27's cells, models across and benchmarks down
    got = ask(page, "step8804 vs step4000 on GPQA and MMLU-Pro", "table")
    assert got.locator("[data-ask-table] [data-ask-cell]").count() == 4
    assert got.locator("[data-ask-col]").count() == 2
    # a question back, its choices buttons
    got = ask(page, "show me the qwen one", "clarify")
    assert got.locator("[data-ask-choice]").count() >= 2
    # two requests in one: the first done, the second a whole action behind a button
    got = ask(page, "q4's gpqa and then open the on-device report", "table")
    assert got.get_attribute("data-ask-card") == "number"
    assert got.locator("[data-ask-then]").inner_text() == "Also: Open On-device models"
    n = page.evaluate("ASK.msgs.filter(m => m.role === 'bot').length")
    got.locator("[data-ask-then]").click()
    answered(page, n + 1)
    page.wait_for_function("location.hash === '#tab=reports&report=on-device'")
    assert page.locator("[data-ask-then]").count() == 0
    # a refusal, and the button to where it's done
    got = ask(page, "start a run on step8804", "text")
    assert got.locator("[data-ask-say]").inner_text() == "I can't start or queue runs."
    assert got.locator("[data-ask-main]").get_attribute("href") == "#tab=runs"
    # older answers fold: every card but the last two
    assert page.locator("[data-ask-fold]").count() == page.locator("[data-ask-you]").count() - 2
    wide = page.evaluate("""() => [...document.querySelectorAll('body *')].filter(e => {
        const r = e.getBoundingClientRect(); return r.right > innerWidth + 1 && r.width > 0; })
        .slice(0, 8).map(e => e.tagName + '.' + e.className + ' ' + Math.round(
          e.getBoundingClientRect().right) + ' in ' + (e.closest('[data-ask-panel]') ? 'panel' : 'page'))""")
    assert no_sideways(page) and page.errors == [], wide
    shot(page, f"replies-{width}-{scheme}.png")


def test_a_choice_asks_again_and_the_thumbs_are_logged(live, page):
    board_at(page, live)
    page.locator("[data-ask-open]").click()
    got = ask(page, "show me the qwen one", "clarify")
    first = got.locator("[data-ask-choice]").first
    label = first.locator(".ask-cname").inner_text()
    first.click()
    answered(page, 2)
    assert page.locator("[data-ask-q]").nth(1).inner_text() == label
    page.wait_for_function("location.hash.startsWith('#model=')")
    bot = page.locator("[data-ask-bot]").nth(1)
    bot.locator("[data-ask-up]").click()
    assert bot.locator("[data-ask-up]").get_attribute("aria-pressed") == "true"
    # 👎: what did you want — a reason, a few words
    bot.locator("[data-ask-down]").click()
    form = bot.locator("[data-ask-form]")
    form.locator("[data-ask-reason='Wrong model']").click()
    assert form.locator("[data-ask-reason='Wrong model']").get_attribute("aria-checked") == "true"
    form.locator("[data-ask-note]").fill("the BF16 build")
    form.locator("[data-ask-fsend]").click()
    page.locator("text=Thanks — noted.").wait_for()
    time.sleep(0.3)
    lines = [json.loads(x) for x in assistant.log_path().read_text().splitlines()]
    th = [x for x in lines if x.get("thumbs")]
    assert (th[-1]["thumbs"], th[-1]["note"], th[-1]["reason"]) == (
        "down", "the BF16 build", "Wrong model")
    assert th[-2]["thumbs"] == "up"
    # Clear empties it
    page.locator("[data-ask-clear]").click()
    assert page.locator("[data-ask-bot]").count() == 0
    assert page.locator("[data-ask-group]").count() == 4
    assert page.errors == []


def test_the_shortcut_and_escape(live, page):
    board_at(page, live)
    page.locator("[data-ask-open]").wait_for()
    page.keyboard.press("Control+k")
    page.locator("[data-ask-panel]").wait_for()
    assert page.evaluate("document.activeElement.dataset.askInput") == "1"
    page.keyboard.press("Escape")
    page.locator("[data-ask-panel]").wait_for(state="detached")
    page.keyboard.press("Meta+k")
    page.locator("[data-ask-panel]").wait_for()
    page.keyboard.press("Meta+k")
    page.locator("[data-ask-panel]").wait_for(state="detached")
    assert page.errors == []


def test_slower_while_a_run_uses_the_model_and_down(live, page):
    board_at(page, live)
    page.locator("[data-ask-open]").click()
    CTRL.update(slow=3.0, busy=True)
    box = page.locator("[data-ask-input]")
    box.fill("what's running?")
    box.press("Enter")
    page.locator("[data-ask-waiting]").wait_for()
    assert page.locator("[data-ask-waiting]").inner_text().startswith("Reading the board…")
    # 28g: the status says it — a dot and words, never colour alone
    page.locator("[data-ask-status='slow']").wait_for(timeout=5000)
    assert page.locator("[data-ask-status]").inner_text() == "Q4 · slower: a run is using it"
    shot(page, "slow-1280-light.png")
    page.locator("[data-ask-bot='list']").wait_for(timeout=20000)
    # down: said, and nothing else breaks
    CTRL.update(slow=0.0, busy=False, down=True)
    ask(page, "what's running?", "down")
    assert "isn’t answering" in page.locator("[data-ask-bot='down']").inner_text() or \
        "isn't answering" in page.locator("[data-ask-bot='down']").inner_text()
    shot(page, "down-1280-light.png")
    page.locator("#tabs [data-tab='models']").click()
    page.locator("[data-lb-card], [data-latest-table]").first.wait_for(timeout=20000)
    assert page.errors == []


def test_nothing_for_a_name_not_set_up(live, page):
    board_at(page, live, name="someone")
    page.wait_for_timeout(800)
    assert page.locator("[data-ask-open]").count() == 0
    page.keyboard.press("Control+k")
    page.wait_for_timeout(300)
    assert page.locator("[data-ask-panel]").count() == 0
    assert page.errors == []


def test_a_models_link_keeps_its_sort_and_thinking(live, page):
    board_at(page, live, hash_="#tab=models&view=frontier&only=ours&sort=hle:asc")
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    assert page.evaluate("JSON.stringify(state.fr.sort)") == (
        '{"key":"hle","dir":1,"link":true}')
    page.reload()
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    assert "sort=hle:asc" in page.evaluate("location.hash")
    page.goto(live["base"] + "/#tab=models&view=frontier&thinking=off")
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    assert page.evaluate("lbS().thinking") == "off"
    assert "thinking=off" in page.evaluate("location.hash")
    assert page.errors == []


def test_a_row_measured_before_its_family_went_to_published_numbers_says_so(live, page):
    """28d: Bonsai and Edge0 are on the board from their published numbers only
    (25c); a row of theirs measured here before that says so, on its line and
    its name's hover"""
    board_at(page, live, hash_="#tab=models")
    page.wait_for_function("DATA && DATA.models.length")
    got = page.evaluate("""() => {
      const m = DATA.models.find(x => x.id.startsWith('served/'));
      const at = Date.UTC(2026, 9, 7, 12) / 1000;
      const b = mtRowOf({ ...m, family: 'bonsai', testedAt: at });
      const o = mtRowOf({ ...m, family: 'qwen', testedAt: at });
      return [b.line, b.full, o.line + o.full]; }""")
    note = "measured here on 7 Oct, before Bonsai went to published numbers only"
    assert note in got[0].split(" · ") and f"It was {note}." in got[1]
    assert "published numbers only" not in got[2]
    assert page.errors == []
