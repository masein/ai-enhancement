"""12d.3 on the page: the Playground as a chat app — the chats down the left,
Today / Yesterday / Earlier, titles on two lines; a searchable model picker
grouped as Models ▾ is, with each model's tags and Everyday score; a centred
conversation over a composer pinned at the foot (Enter sends, Shift+Enter a
new line, Send is Stop while a reply streams); "Ask anything" with practice
questions to start from; the settings in a side panel with a badge; and the
models served elsewhere — the phone build and the original — streaming, their
thinking folded, MTP's drafts under the reply, side by side in Compare, and
waiting while a run tests them. Screenshots at 1280 and 400 px, light and
dark, are CI's artifacts (tests/_screens/phase12d3/)."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import pytest

import everyday as ev
from conftest import pg_choose, set_name
from fake_openai import FakeServer
from service import chat, config, runner

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12d3"
SMALL = "fx/below-135m-it"
LDA, ORIG = "served/k4-LDA-MTP", "served/Qwen3.6-original"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json", "X-Who": "masein"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def servers(live):
    lda, orig = FakeServer(), FakeServer()
    lda.reply = lambda body: ("The 9:40 from Dubai Marina arrives at 10:05 on platform 2; "
                              "the next one is at 10:10.")
    lda.reasoning = "The timetable lists 9:40 and 10:10; the first arrives at 10:05."
    lda.timings = lambda body: {"predicted_n": 38, "draft_n": 165, "draft_n_accepted": 135}
    orig.model_path = "/home/masein/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    orig.reply = lambda body: "It arrives at 10:05, platform 2."
    for fake, name, how in ((lda, "k4-LDA MTP", "llama.cpp fork, k=4 + LDA, MTP n_max 3, phone build"),
                            (orig, "Qwen3.6 original", "llama.cpp, unsloth UD-Q4_K_XL")):
        api(live, "/api/served", {"name": name, "base_url": fake.base, "key": "",
                                  "based_on": "Qwen/Qwen3.6-35B-A3B", "how": how,
                                  "thinking": "on", "by": "masein"})
    yield lda, orig
    lda.close()
    orig.close()


@pytest.fixture(autouse=True)
def fake(live, servers, monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "fake")
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.005)
    monkeypatch.setattr(chat.FakeBackend, "responder", None)
    monkeypatch.setattr(chat, "ENGINE", chat.Engine())
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {
        "remote_code": False, "revision": None, "archinfo": {}})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 40 * 10 ** 9)
    for s in servers:
        s.stream_delay_s = 0.0
    yield


def pg(page, base, width=1280, theme=None):
    page.set_viewport_size({"width": width, "height": 860})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")
    page.goto("about:blank")
    page.goto(base + "/#tab=playground")
    page.wait_for_selector("[data-pg-main]")
    if theme:
        page.evaluate(f"applyTheme('{theme}')")


def send(page, text):
    box = page.locator("[data-pg-input]")
    box.fill(text)
    box.press("Enter")


def test_a_served_model_streams_its_thinking_folded_and_mtp_under_it(live, page, servers):
    lda, _ = servers
    pg(page, live["base"])
    pg_choose(page, LDA)
    assert page.locator("[data-pg-model] [data-pg-tag]").all_inner_texts() == ["served", "phone build"]
    n = len(lda.requests)
    send(page, "When does the next train from Dubai Marina arrive?")
    stats = page.locator("[data-pg-reply='1'] [data-pg-stats]")
    stats.wait_for()
    assert page.locator("[data-pg-reply='1'] [data-pg-text]").inner_text().startswith("The 9:40")
    fold = page.locator("[data-pg-reply='1'] [data-pg-think]")
    assert fold.get_attribute("open") is None and fold.locator("summary").inner_text() == "Thinking ▸"
    fold.locator("summary").click()
    assert "timetable lists" in fold.inner_text()
    import re
    txt = stats.inner_text()
    assert re.fullmatch(r"\d+ tokens · [\d.]+ s · MTP: 135 of 165 drafts kept · copy · again", txt), txt
    assert "w/s" not in txt                     # its server's speed is not the phone's
    body = lda.requests[n]
    assert body["stream"] is True and body["chat_template_kwargs"] == {"enable_thinking": True}
    assert page.errors == []


def test_the_chat_tab_of_a_served_model_chats(live, page, servers):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#model=" + ORIG.replace("/", "%2F"))
    page.wait_for_selector(f"[data-served-head='{ORIG}']")
    page.locator("[data-mtab='chat']").click()
    page.wait_for_selector("[data-mtab-panel='chat'] [data-pg-input]")
    assert page.locator("[data-model-chat-none]").count() == 0
    send(page, "When does it arrive?")
    page.locator("[data-pg-reply='1'] [data-pg-stats]").wait_for()
    assert page.locator("[data-pg-reply='1'] [data-pg-text]").inner_text() == \
        "It arrives at 10:05, platform 2."
    assert page.errors == []


def test_while_a_run_tests_it_the_chat_waits_and_sends_nothing(live, page, servers):
    lda, _ = servers
    sub = api(live, "/api/submissions", {"hf_id": LDA, "suite": "everyday"})
    from service import db
    db.update(sub["id"], status="running", progress="140 of 388 · 4.1 s an answer · about 20 min left")
    runner.LOCK.mkdir(parents=True, exist_ok=True)
    (runner.LOCK / "submission").write_text(str(sub["id"]))
    try:
        pg(page, live["base"])
        pg_choose(page, LDA)
        n = len(lda.requests)
        # 16.2: Not now before anything is sent — Send held back, the reason beside it
        line = page.locator(f"[data-pg-notnow='{LDA}']")
        line.wait_for()
        assert line.inner_text() == (f"k4-LDA MTP: not now. Being tested right now (run "
                                     f"#{sub['id']}, about 20 min left). Chat starts when it's done.")
        assert page.locator("[data-pg-send]").is_disabled()
        send(page, "Are you there?")
        assert page.locator("[data-pg-reply]").count() == 0
        assert len(lda.requests) == n
    finally:
        shutil.rmtree(runner.LOCK, ignore_errors=True)
        db.update(sub["id"], status="canceled")
    assert page.errors == []


def test_compare_the_phone_build_and_the_original_side_by_side(live, page, servers):
    pg(page, live["base"])
    page.locator("[data-pg-compare]").click()
    pg_choose(page, LDA)
    pg_choose(page, ORIG, second=True)
    heads = page.locator("[data-pg-colhead]")
    assert heads.count() == 2
    assert "Everyday" in heads.nth(0).locator("[data-pg-meta]").inner_text()
    send(page, "When does the next train arrive?")
    a = page.locator("[data-pg-pair='1'] [data-pg-col='a'] [data-pg-stats]")
    b = page.locator("[data-pg-pair='1'] [data-pg-col='b'] [data-pg-stats]")
    a.wait_for()
    b.wait_for()
    assert "MTP: 135 of 165 drafts kept" in a.inner_text() and "MTP" not in b.inner_text()
    shot(page, "playground-compare-1280-light.png")
    # a local and a served model, too
    page.locator("[data-pg-new]").first.click()
    page.locator("[data-pg-compare]").click()
    pg_choose(page, SMALL)
    pg_choose(page, ORIG, second=True)
    send(page, "hi to both")
    page.locator("[data-pg-pair='1'] [data-pg-col='a'] [data-pg-stats]").wait_for()
    page.locator("[data-pg-pair='1'] [data-pg-col='b'] [data-pg-stats]").wait_for()
    assert page.errors == []


def test_enter_sends_shift_enter_is_a_new_line_and_stop_cancels(live, page, servers, monkeypatch):
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.05)
    monkeypatch.setattr(chat.FakeBackend, "responder",
                        staticmethod(lambda m, msgs, s: " ".join(["word"] * 300)))
    pg(page, live["base"])
    pg_choose(page, SMALL)
    box = page.locator("[data-pg-input]")
    box.click()
    box.type("line one")
    box.press("Shift+Enter")
    box.type("line two")
    assert box.input_value() == "line one\nline two"
    assert page.locator("[data-pg-reply]").count() == 0                   # nothing sent
    h1 = box.bounding_box()["height"]
    box.press("Enter")
    stop = page.locator("[data-pg-stop]")
    stop.wait_for()
    assert box.input_value() == ""                                         # sent, so gone
    assert page.locator("[data-pg-send]").count() == 0                     # Send is Stop
    assert page.locator("[data-pg-you]").inner_text() == "you:\nline one\nline two"
    stop.click()
    cut = page.locator("[data-pg-reply='1'] [data-pg-cut]")
    cut.wait_for()
    assert cut.inner_text() == "stopped"
    page.wait_for_selector("[data-pg-send]")
    # the composer grows with the text, up to about eight lines
    box.fill("\n".join(f"line {i}" for i in range(20)))
    box.dispatch_event("input")
    h = box.bounding_box()["height"]
    assert h > h1 * 2 and h < 260, (h1, h)
    assert page.errors == []


def test_the_empty_state_offers_practice_questions_only_and_one_practice_button(live, page):
    pg(page, live["base"])
    empty = page.locator("[data-pg-empty]")
    assert empty.locator("h2").inner_text() == "Ask anything"
    chips = empty.locator("[data-pg-chip]").evaluate_all("xs => xs.map(x => x.dataset.pgChip)")
    assert 3 <= len(chips) <= 4
    practice = {q["id"] for q in ev.load_bank() if ev.half(q) == ev.PRACTICE}
    hidden = {q["id"] for q in ev.load_bank() if ev.half(q) != ev.PRACTICE}
    offered = api(live, "/api/playground/practice")
    known = {x["id"] for x in offered["knowledge"]}
    for c in chips:
        kind, qid = c.split("|", 1)
        assert (qid in practice and qid not in hidden) if kind == "everyday" else qid in known
    assert page.locator("[data-pg-practice]").count() == 1              # once, by the composer
    first = empty.locator("[data-pg-chip]").first
    kind, qid = chips[0].split("|", 1)
    first.click()
    want = next(x for x in offered[kind] if x["id"] == qid)["prompt"]
    assert page.locator("[data-pg-input]").input_value() == want
    # a new chat draws again
    assert page.evaluate("JSON.stringify(state.pg.ref)") == json.dumps(
        {"kind": kind, "id": qid}, separators=(",", ":"))
    assert page.errors == []


def test_the_picker_is_searchable_grouped_with_tags_and_one_score(live, page):
    pg(page, live["base"])
    page.locator("[data-pg-model]").click()
    menu = page.locator("#pop-pg-model-a")
    heads = menu.locator(".pghead").all_inner_texts()
    assert heads[:3] == ["Phone builds", "Served elsewhere", "Instruct"], heads
    opt = menu.locator(f"[data-pg-option='{LDA}']")
    assert opt.locator("[data-pg-tag]").all_inner_texts() == ["served", "phone build"]
    assert opt.locator("[data-pg-score]").inner_text().startswith("Everyday")
    assert menu.locator("[data-pg-left-out]").first.inner_text() == \
        "Base models aren't listed: they have no chat format."
    menu.locator("[data-pg-search]").fill("original")
    assert menu.locator("[data-pg-option]").evaluate_all("xs => xs.map(x => x.dataset.pgOption)") == [ORIG]
    menu.locator("[data-pg-search]").press("ArrowDown")
    page.keyboard.press("Enter")
    page.wait_for_selector(f"[data-pg-model='{ORIG}']")
    assert page.errors == []


def test_settings_are_a_side_panel_and_the_badge_says_which(live, page):
    pg(page, live["base"])
    pg_choose(page, SMALL)
    badge = page.locator("[data-pg-badge]")
    assert badge.inner_text() == "scored settings ✓"
    page.locator("[data-pg-gear]").click()
    panel = page.locator("[data-pg-settings]")
    panel.wait_for()
    main = page.locator("[data-pg-main]").bounding_box()
    box = panel.bounding_box()
    assert abs(box["x"] + box["width"] - main["x"] - main["width"]) < 3      # at the right
    page.locator("[data-pg-temperature]").fill("0.7")
    page.locator("[data-pg-temperature]").dispatch_event("change")
    page.wait_for_selector("[data-pg-badge='custom']")
    assert badge.inner_text() == "custom settings · reset"
    shot(page, "playground-settings-1280-light.png")
    page.keyboard.press("Escape") if page.locator("[data-pg-settings] :focus").count() else None
    page.locator("[data-pg-panel-close]").click() if panel.count() else None
    page.locator("[data-pg-reset]").click()
    page.wait_for_selector("[data-pg-badge='scored']")
    assert page.errors == []


def test_the_chat_list_groups_by_day_wraps_titles_and_hides_delete(live, page):
    from service import db
    import time
    now = time.time()
    for title, at in (("Train arrival times from Dubai Marina to the airport, and what if it is late",
                       now - 60), ("k4 vs original", now - 86400 - 3600), ("an old one", now - 9 * 86400)):
        c = chat.new_chat(SMALL, "masein")
        c.update(title=title, updated_at=at)
        # chat_put stamps now: the day is set as the database keeps it
        from contextlib import closing
        with closing(db._conn()) as cn:
            cn.execute("UPDATE chats SET data=?, updated_at=? WHERE id=?", (json.dumps(c), at, c["id"]))
            cn.commit()
    pg(page, live["base"])
    groups = page.locator("[data-pg-group]").evaluate_all("xs => xs.map(x => x.dataset.pgGroup)")
    assert groups == ["Today", "Yesterday", "Earlier"]
    long = page.locator("[data-pg-list] .pgtitle", has_text="Train arrival").first
    lines = long.evaluate("e => Math.round(e.getBoundingClientRect().height / parseFloat(getComputedStyle(e).lineHeight))")
    assert lines == 2
    item = page.locator("[data-pg-chat]").first
    dl = item.locator("[data-pg-delete]")
    assert dl.evaluate("e => getComputedStyle(e).opacity") == "0"
    item.hover()
    page.wait_for_function("e => getComputedStyle(e).opacity === '1'", arg=dl.element_handle())
    assert page.errors == []


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1280, 400])
def test_screenshots_for_the_pr(live, page, servers, theme, width):
    """the new layout, as masein sees it before deploying: a chat with the
    phone build (thinking folded, MTP under the reply), and the empty state"""
    pg(page, live["base"], width=width, theme=theme)
    if width > 720:
        shot(page, f"playground-empty-{width}-{theme}.png")
    pg_choose(page, LDA)
    send(page, "When does the next train from Dubai Marina arrive?")
    page.locator("[data-pg-reply='1'] [data-pg-stats]").wait_for()
    page.wait_for_timeout(150)
    # no sideways scroll, at 400 px too
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"playground-{width}-{theme}.png")
    assert page.errors == []
