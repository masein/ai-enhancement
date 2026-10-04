"""12d.2 on the page: + Compare and two columns (side by side, stacked on a
phone); the model page's Chat tab on every model, a base model's with its
line; "Ask it again ▸" from Answers, in the input, unsent; a trained model
under its base, and Compare suggesting that base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import everyday as ev
from conftest import pg_choose, set_name
from service import chat, config, db

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12d2"
SMALL, BIG, TUNED = "fx/below-135m-it", "fx/chat-1.7b-it", "local/chat-1.7b-tuned"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def plant(mid, params, **arch):
    d = config.OUT_DIR / mid.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({
        "model": mid, "kind": "instruct", "params": params, "ctx": 4096, "tmpl_sha": "abc",
        **arch}), encoding="utf-8")


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
    plant(BIG, 1_700_000_000)
    plant(TUNED, 1_700_000_000)
    db.trained_from_set(TUNED, BIG, "masein")
    yield


def pg(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")
    page.goto(base + "/#tab=playground")
    page.wait_for_selector("[data-pg-main]")


@pytest.mark.parametrize("width", [1400, 400])
def test_compare_two_models_side_by_side(live, page, width):
    pg(page, live["base"], width)
    pg_choose(page, SMALL)
    page.locator("[data-pg-compare]").click()
    pg_choose(page, BIG, second=True)
    page.locator("[data-pg-input]").fill("say hi to both")
    page.locator("[data-pg-send]").click()
    a = page.locator("[data-pg-pair='1'] [data-pg-col='a'] [data-pg-stats]")
    b = page.locator("[data-pg-pair='1'] [data-pg-col='b'] [data-pg-stats]")
    a.wait_for()
    b.wait_for()
    names = page.locator("[data-pg-pair='1'] .pgwho").all_inner_texts()
    assert names == ["below-135m-it", "chat-1.7b-it"]
    ra = page.locator("[data-pg-pair='1'] [data-pg-col='a']").bounding_box()
    rb = page.locator("[data-pg-pair='1'] [data-pg-col='b']").bounding_box()
    if width > 720:
        assert rb["x"] > ra["x"] + ra["width"] - 1 and abs(rb["y"] - ra["y"]) < 2   # side by side
    else:
        assert rb["y"] >= ra["y"] + ra["height"] - 1                                 # stacked
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"12d2-compare-{width}-light.png", full_page=width < 720)
    if width > 720:
        did = page.evaluate("state.pg.id")
        item = page.locator(f"[data-pg-chat='{did}'] [data-pg-names] .pgnm")   # 16.6: its time beside
        assert item.inner_text() == "below-135m-it vs chat-1.7b-it"
    assert page.errors == []


def test_a_trained_model_sits_under_its_base_and_compare_suggests_it(live, page):
    pg(page, live["base"])
    page.locator("[data-pg-model]").click()
    opts = page.locator("#pop-pg-model-a [data-pg-option]").evaluate_all(
        "os => os.map(o => [o.dataset.pgOption, o.querySelector('.pgoname').textContent, "
        "o.querySelector('[data-pg-tag=trained]')?.textContent || ''])")
    ids = [v for v, _, _ in opts]
    assert ids.index(TUNED) == ids.index(BIG) + 1
    import re
    name, tag = next((n, t) for v, n, t in opts if v == TUNED)
    assert name == "↳ chat-1.7b-tuned" and re.fullmatch(r"trained · \d{4}-\d\d-\d\d", tag)
    page.keyboard.press("Escape")
    pg_choose(page, TUNED)
    sug = page.locator(f"[data-pg-suggest='{BIG}']")
    assert sug.inner_text() == "Compare with chat-1.7b-it (before training)"
    sug.click()
    assert page.locator("[data-pg-model2]").get_attribute("data-pg-model2") == BIG
    shot(page.locator("[data-pg-main]"), "12d2-trained-suggest-1400-light.png")
    assert page.errors == []


def test_the_chat_tab_is_on_every_model_page(live, page):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#model=" + SMALL.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    tabs = page.locator("[data-mtab]").all_inner_texts()
    assert tabs[:3] == ["Scores", "Answers", "Chat"] and tabs[-1] == "History"
    page.locator("[data-mtab='chat']").click()
    card = page.locator(f"[data-model-chat='{SMALL}']")
    card.wait_for()
    assert card.locator("[data-pg-open]").inner_text() == "Open in Playground →"
    assert page.locator("[data-pg-model]").inner_text() == "below-135m-it"      # fixed
    page.locator("[data-pg-input]").fill("hi from the model page")
    page.locator("[data-pg-send]").click()
    page.locator("[data-pg-reply='1'] [data-pg-stats]").wait_for()
    shot(page, "12d2-chat-tab-1400-light.png")
    # a base model: the tab is there, with its one line
    page.goto(live["base"] + "/#model=fx%2Fchance-160m")
    page.wait_for_selector("[data-model-hero]")
    assert "Chat" in page.locator("[data-mtab]").all_inner_texts()
    page.locator("[data-mtab='chat']").click()
    why = page.locator("[data-model-chat-why]")
    why.wait_for()
    assert why.inner_text() == ("This is a base model: it has no chat format, so there’s "
                                "nothing to chat with.")
    assert page.errors == []


def test_ask_it_again_puts_the_question_in_the_chat_tab_unsent(live, page):
    # an Everyday answer of a model the Playground offers: fx/good-750m, given a template
    mid = "fx/good-750m"
    p = config.OUT_DIR / "fx__good-750m" / "model_meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps({**meta, "tmpl_sha": "abc"}), encoding="utf-8")
    try:
        page.set_viewport_size({"width": 1400, "height": 900})
        page.goto(live["base"] + "/#tab=home")
        set_name(page, "masein")
        page.goto(live["base"] + "/#model=" + mid.replace("/", "%2F"))
        page.wait_for_selector("[data-model-hero]")
        page.locator("[data-mtab='answers']").click()
        if page.locator("[data-answers-kind='everyday']").count():
            page.locator("[data-answers-kind='everyday']").click()
        btn = page.locator("[data-ask-again]").first
        btn.wait_for()
        qid = btn.get_attribute("data-ask-again")
        q = next(x for x in ev.load_bank() if x["id"] == qid)
        assert ev.half(q) == ev.PRACTICE
        btn.click()
        page.wait_for_selector("[data-mtab-panel='chat'] [data-pg-input]")
        assert page.locator("[data-mtab='chat']").get_attribute("aria-selected") == "true"
        assert page.locator("[data-pg-input]").input_value() == q["prompt"]
        assert page.locator("[data-pg-reply]").count() == 0                    # not sent
        assert page.evaluate("JSON.stringify(state.pg.ref)") == \
            json.dumps({"kind": "everyday", "id": qid}, separators=(",", ":"))
    finally:
        p.write_text(json.dumps(meta), encoding="utf-8")
    assert page.errors == []
