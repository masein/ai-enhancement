"""12d.1 on the page: Playground between Models and Improve; a chat that
streams, with its stats, copy and again; Settings ▸ and the amber line; a
practice question marked; thinking folded; base models left out with the
line; a closed tab that frees the model; and the phone's Chats ▾."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

import everyday as ev
from conftest import pg_choose, set_name
from service import chat, config

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12d1"
SMALL, THINKER = "fx/below-135m-it", "fx/thinker-0.6b"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


@pytest.fixture(autouse=True)
def fake(live, monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "fake")
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.005)
    monkeypatch.setattr(chat.FakeBackend, "responder", None)
    monkeypatch.setattr(chat, "ENGINE", chat.Engine())
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {
        "remote_code": False, "revision": None, "archinfo": {}})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 40 * 10 ** 9)
    d = config.OUT_DIR / THINKER.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({
        "model": THINKER, "kind": "instruct", "params": 600_000_000, "ctx": 4096,
        "tmpl_sha": "abc", "reasoning_template": True, "thinking": "always"}), encoding="utf-8")
    yield


def pg(page, base, width=1400, name="masein"):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(base + "/#tab=home")
    set_name(page, name)
    page.goto(base + "/#tab=playground")
    page.wait_for_selector("[data-pg-main]")


def test_the_header_reads_home_models_playground_improve_benchmarks(live, page):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    tabs = page.locator("#tabs [role=tab]").all_inner_texts()
    assert tabs == ["Home", "Models", "Playground", "Improve", "Benchmarks"]
    page.locator("#tabs [data-tab='playground']").click()
    page.wait_for_selector("[data-pg]")
    assert page.evaluate("location.hash") == "#tab=playground"
    # one line at 400 px: the places are in Menu ▾
    page.set_viewport_size({"width": 400, "height": 800})
    page.goto(live["base"] + "/#tab=home")
    bar = page.locator("#bar").bounding_box()
    assert bar["height"] < 70, bar
    page.locator("#menuBtn").click()
    assert "Playground" in page.locator("#pop-places [data-place]").all_inner_texts()
    assert page.errors == []


def test_a_chat_streams_with_its_stats_copy_and_again(live, page):
    pg(page, live["base"])
    empty = page.locator("[data-pg-empty]")
    assert empty.locator("h2").inner_text() == "Ask anything"
    # 12d.3: practice questions are offered once, beside the composer
    assert page.locator("[data-pg-practice]").count() == 1
    pg_choose(page, SMALL)
    page.locator("[data-pg-input]").fill("can u make this shorter pls")
    page.locator("[data-pg-send]").click()
    stats = page.locator("[data-pg-reply='1'] [data-pg-stats]")
    stats.wait_for()
    assert page.locator("[data-pg-reply='1'] [data-pg-text]").inner_text().startswith(
        "You asked: can u make this shorter pls")
    import re
    assert re.match(r"\d\d:\d\d · \d+ words · [\d.]+s · [\d.]+ w/s · copy · again",
                    stats.inner_text())                         # 16.6: when, first
    did = page.evaluate("state.pg.id")
    assert page.evaluate("location.hash") == f"#tab=playground&chat={did}"
    assert page.locator(f"[data-pg-chat='{did}'] .pgtitle").inner_text() == "can u make this shorter pls"
    shot(page, "12d1-chat-1400-light.png")
    # the runs popover says what the Playground holds, quietly
    page.wait_for_function("(state.pgStatus || {loaded: []}).loaded.length === 1", timeout=20000)
    page.locator("#runs button").click()
    assert page.locator("#pop-runs [data-pg-loaded]").inner_text() == \
        "Playground: below-135m-it loaded"
    page.keyboard.press("Escape")
    stats.locator("[data-pg-again]").click()
    page.wait_for_selector("[data-pg-alt='2|2']")
    # the chat is the server's: a reload lands on it
    page.reload()
    page.wait_for_selector("[data-pg-reply='1'] [data-pg-alt='2|2']")
    assert page.errors == []


def test_settings_changed_show_the_amber_line_and_reset_puts_them_back(live, page):
    pg(page, live["base"])
    pg_choose(page, SMALL)
    # 12d.3: the settings are a side panel; a badge beside the composer says which
    assert page.locator("[data-pg-badge]").inner_text() == "scored settings ✓"
    page.locator("[data-pg-gear]").click()
    page.wait_for_selector("[data-pg-settings]")
    assert page.locator("[data-pg-not-scored]").count() == 0
    page.locator("[data-pg-temperature]").fill("0.7")
    page.locator("[data-pg-temperature]").dispatch_event("change")
    line = page.locator("[data-pg-not-scored]")
    line.wait_for()
    assert line.inner_text() == "custom settings · reset"
    shot(page.locator("[data-pg-main]"), "12d1-settings-1400-light.png")
    line.locator("[data-pg-reset]").click()
    page.wait_for_selector("[data-pg-not-scored]", state="detached")
    assert page.errors == []


def test_a_practice_question_sent_unedited_is_marked_as_the_run_marks(live, page, monkeypatch):
    q = next(x for x in ev.load_bank() if ev.half(x) == ev.PRACTICE
             and not any(c["type"] == "judge" for c in x["checks"]))
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: q["reference"]))
    pg(page, live["base"])
    pg_choose(page, SMALL)
    page.locator("[data-pg-practice]").click()
    page.locator(f"[data-pg-pq='everyday|{q['id']}']").click()
    assert page.locator("[data-pg-input]").input_value() == q["prompt"]     # unsent
    assert page.locator("[data-pg-reply]").count() == 0
    page.locator("[data-pg-send]").click()
    mark = page.locator("[data-pg-reply='1'] [data-pg-mark]")
    mark.wait_for()
    assert mark.inner_text() == "✓ passes"
    shot(page.locator("[data-pg-main]"), "12d1-practice-marked-1400-light.png")
    assert page.errors == []


def test_thinking_is_folded_above_the_reply(live, page):
    pg(page, live["base"])
    pg_choose(page, THINKER)
    page.locator("[data-pg-input]").fill("what is 2+2")
    page.locator("[data-pg-send]").click()
    page.locator("[data-pg-reply='1'] [data-pg-stats]").wait_for()
    fold = page.locator("[data-pg-reply='1'] [data-pg-think]")
    assert fold.get_attribute("open") is None
    assert fold.locator("summary").inner_text() == "Thinking ▸"
    assert "Let me think" not in page.locator("[data-pg-reply='1'] [data-pg-text]").inner_text()
    assert page.errors == []


def test_base_models_are_left_out_with_the_line(live, page):
    pg(page, live["base"])
    page.locator("[data-pg-model]").click()
    opts = page.locator("#pop-pg-model-a [data-pg-option]").evaluate_all(
        "os => os.map(o => o.dataset.pgOption)")
    assert SMALL in opts and "fx/good-750m" not in opts
    assert page.locator("#pop-pg-model-a [data-pg-left-out]").first.inner_text() == \
        "Base models aren't listed: they have no chat format."
    assert page.errors == []


def test_closing_the_tab_frees_the_model(live, monkeypatch):
    """a reader who goes away mid-reply: the reply stops, the model is free"""
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.05)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(["w"] * 400)))
    base = live["base"]

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    c = post("/api/playground/chats", {"model": SMALL, "by": "masein"})
    r = post(f"/api/playground/chats/{c['id']}/messages", {"text": "go", "by": "masein"})
    resp = urllib.request.urlopen(base + f"/api/playground/streams/{r['stream']}", timeout=10)
    resp.readline()
    resp.close()                                       # the tab is gone
    st = chat.ENGINE.streams[r["stream"]]
    end = time.time() + 10
    while time.time() < end and not st.done:
        time.sleep(0.05)
    assert st.done and st.stop.is_set()
    assert not chat.ENGINE.loaded[SMALL].busy.locked()


def test_on_a_phone_the_chat_list_folds_into_chats(live, page):
    pg(page, live["base"], width=400)
    assert page.locator("[data-pg-list]").count() == 0
    page.locator("[data-pg-chats]").click()
    page.wait_for_selector("#pop-pg-chats [data-pg-new]")
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, "12d1-phone-400-light.png", full_page=True)
    assert page.errors == []


def test_a_reply_a_run_cut_short_says_the_chat_is_paused_and_kept(live, page):
    """what a run does to a reply on the GPU (chat.yield_gpu): it keeps what it
    wrote, marked, and the chat says why once"""
    from service import db
    c = chat.new_chat(SMALL, "masein")
    c["messages"] = [{"role": "user", "text": "tell me a long story", "at": 1.0},
                     {"role": "assistant", "shown": 0, "replies": [{
                         "text": "Once upon a time there was", "thinking": "", "cut": "a run started",
                         "device": "cuda", "secs": 1.2, "words": 6, "wps": 5.0, "mark": None}]}]
    c["title"] = "tell me a long story"
    db.chat_put(c)
    pg(page, live["base"])
    page.goto(live["base"] + f"/#tab=playground&chat={c['id']}")
    page.wait_for_selector("[data-pg-paused]")
    assert page.locator("[data-pg-paused]").inner_text() == \
        "Paused: a run started. Your conversation is kept."
    assert page.locator("[data-pg-cut]").inner_text() == "cut short: a run started"
    assert page.locator("[data-pg-text]").inner_text() == "Once upon a time there was"
    assert page.errors == []
