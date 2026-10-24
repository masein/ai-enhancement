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
CTRL = {"slow": 0.0, "busy": False, "down": False}


def fake_call(system, user, sch, prov=None):
    """the bank's action for its question; a clicked choice opens that model"""
    if CTRL["down"]:
        raise assistant.Down("connection refused")
    time.sleep(CTRL["slow"])
    q = user.rsplit("Question: ", 1)[1]
    b = BANK.get(q)
    want = b["want"] if b else {"action": "open_page", "page": "model", "models": [q]}
    return {"raw": json.dumps(want), "ms": 5, "usage": {}}


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


def ask(page, q: str, kind: str):
    n = page.locator("[data-ask-bot]").count()
    box = page.locator("[data-ask-input]")
    box.fill(q)
    box.press("Enter")
    page.locator("[data-ask-bot]").nth(n).wait_for(timeout=20000)
    got = page.locator("[data-ask-bot]").nth(n)
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
    # 380 wide over the page; a phone's whole width
    w = panel.evaluate("e => e.getBoundingClientRect().width")
    assert (w == 380) if width > 700 else (w == width), w
    # three example questions from the bank, each its own
    ex = page.locator("[data-ask-example]").all_inner_texts()
    assert len(ex) == 3 and all(q in BANK for q in ex)
    shot(page, f"empty-{width}-{scheme}.png")
    # a page: the board's own link, opened; the panel stays
    got = ask(page, "frontier tab, ours only, sorted by HLE", "open")
    page.wait_for_function("location.hash === '#tab=models&view=frontier&only=ours&sort=hle'")
    page.locator("[data-frontier-table]").wait_for(state="attached", timeout=20000)
    assert got.locator("[data-ask-link]").get_attribute("href") == (
        "#tab=models&view=frontier&only=ours&sort=hle")
    assert panel.is_visible()
    # a chart, from Across training's own panel
    got = ask(page, "chart GPQA across the checkpoints", "chart")
    got.locator("[data-ask-chart='checkpoints'] svg").wait_for(timeout=20000)
    # a table of 27's cells
    got = ask(page, "step8804 vs step4000 on GPQA and MMLU-Pro", "table")
    assert got.locator("[data-ask-table] [data-ask-cell]").count() == 4
    # a clarify, its choices buttons
    got = ask(page, "show me the qwen one", "clarify")
    assert got.locator("[data-ask-choice]").count() >= 2
    # a refusal, and where the button is
    got = ask(page, "start a run on step8804", "text")
    assert got.locator("[data-ask-say]").inner_text() == "I can't start or queue runs."
    assert got.locator("[data-ask-link]").get_attribute("href") == "#tab=runs"
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
    label = first.inner_text()
    first.click()
    page.locator("[data-ask-bot]").nth(1).wait_for(timeout=20000)
    assert page.locator("[data-ask-you]").nth(1).inner_text() == label
    page.wait_for_function("location.hash.startsWith('#model=')")
    bot = page.locator("[data-ask-bot]").nth(1)
    bot.locator("[data-ask-up]").click()
    assert "on" in bot.locator("[data-ask-up]").get_attribute("class")
    bot.locator("[data-ask-down]").click()
    note = bot.locator("[data-ask-note]")
    note.fill("the BF16 build")
    note.press("Enter")
    page.locator("text=Thanks — noted.").wait_for()
    time.sleep(0.3)
    lines = [json.loads(x) for x in assistant.log_path().read_text().splitlines()]
    th = [x for x in lines if x.get("thumbs")]
    assert th[-1]["thumbs"] == "down" and th[-1]["note"] == "the BF16 build"
    assert th[-2]["thumbs"] == "up"
    # Clear empties it
    page.locator("[data-ask-clear]").click()
    assert page.locator("[data-ask-bot]").count() == 0
    assert page.locator("[data-ask-example]").count() == 3
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
    assert page.locator("[data-ask-waiting]").inner_text() == "Thinking…"
    page.locator("[data-ask-waiting='slow']").wait_for(timeout=5000)
    assert page.locator("[data-ask-waiting]").inner_text() == (
        "Slower now: a run is using the model…")
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
