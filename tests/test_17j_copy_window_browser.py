"""17j on the page, part 5, points 36 and 37: copying works where the board is
opened over http on a name that isn't localhost (the tailnet's: no secure
context, no navigator.clipboard), at the Playground's copy and every other;
and a served model's page offers "Use the new window" when its server was
started again with a larger -c. A stand-in llama-server; no model runs."""

from __future__ import annotations

import json
import time
from contextlib import closing

import pytest

from conftest import set_name
from fake_openai import FakeServer
from test_playground_12d1_browser import fake  # noqa: F401 — the fake chat backend

pytestmark = pytest.mark.dashboard
SMALL = "fx/below-135m-it"
REPLY = "The train leaves at 10:05 from platform two."
# every copy event's text, in the order they happen — what the clipboard got
LISTEN = """window.__copied = [];
document.addEventListener('copy', e => {
  const t = e.target;
  window.__copied.push(t && typeof t.value === 'string' ? t.value : String(getSelection()));
}, true);"""


@pytest.fixture
def tailnet(live, browser):
    """the board as a person on the tailnet opens it: http, on a name that
    isn't localhost — Chromium's own resolver sends the name to the board"""
    port = live["base"].rsplit(":", 1)[1]
    other = browser.browser_type.launch(
        args=["--host-resolver-rules=MAP board-17j.example 127.0.0.1"])
    ctx = other.new_context(viewport={"width": 1400, "height": 900}, reduced_motion="reduce")
    ctx.add_init_script(LISTEN)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.errors = errors
    page.base = f"http://board-17j.example:{port}"
    yield page
    ctx.close()
    other.close()


def a_chat() -> str:
    from service import chat, db
    c = chat.new_chat(SMALL, "masein")
    c.update(title="Trains", messages=[
        {"role": "user", "text": "when is the next train", "at": time.time() - 60},
        {"role": "assistant", "shown": 0, "replies": [
            {"text": REPLY, "words": len(REPLY.split()), "secs": 1.2, "wps": 3.3, "device": "cpu",
             "at": time.time() - 30}]}])
    with closing(db._conn()) as cn:
        cn.execute("UPDATE chats SET data=? WHERE id=?", (json.dumps(c), c["id"]))
        cn.commit()
    return c["id"]


def test_36_copy_works_over_http_on_a_name_that_isnt_localhost(live, tailnet, fake):  # noqa: F811
    page = tailnet
    page.goto(page.base + "/#tab=home")
    set_name(page, "masein")
    # the case itself: no secure context, no clipboard API
    assert page.evaluate("window.isSecureContext") is False
    assert page.evaluate("navigator.clipboard === undefined") is True
    cid = a_chat()
    page.goto("about:blank")
    page.goto(page.base + f"/#tab=playground&chat={cid}")
    page.locator("[data-pg-copy]").first.click()
    toast = page.locator("[data-toast='pg']")
    toast.wait_for()
    assert toast.inner_text().startswith("Copied the reply"), toast.inner_text()  # 0abb757: failed
    assert page.evaluate("window.__copied") == [REPLY]
    # and a copy of copyText's (a run's ⋯ ▸ Copy id), the same way
    page.evaluate("copyText('--gap-dataset 17', 'the flag')")
    page.wait_for_function("window.__copied.length === 2")
    assert page.evaluate("window.__copied[1]") == "--gap-dataset 17"
    assert page.errors == []


def test_37_the_models_page_offers_the_new_window_without_the_key(live, page):
    import service.app as appmod
    from service import db, served
    srv = FakeServer()
    srv.key = "kept-key-17j"
    try:
        mid = "served/window-page-17j"
        db.served_put({"id": mid, "name": "window page 17j", "base_url": srv.base,
                       "key": srv.key, "based_on": "Qwen/Qwen3.6-35B-A3B",
                       "how": "llama-server -c 16384", "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, srv.key)), "by": "masein",
                       "at": 0})
        served.write_meta(served.get(mid))
        srv.ctx = 65536                                    # started again with -c 65536
        appmod._cache.update(key=None, payload=None, at=0.0)
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + f"/#model={mid}")
        set_name(page, "masein")
        line = page.locator(f"[data-served-head='{mid}'] [data-served-pin-now]")
        line.wait_for()                                    # 0abb757: nothing, then a "file"
        assert line.get_attribute("data-served-pin-now") == "window"
        words = line.inner_text()
        assert "context window of 65,536 tokens, not the 16,384" in words, words
        assert "different file" not in words
        page.locator(f"[data-served-use-window='{mid}']").click()
        page.wait_for_function(f"!document.querySelector(\"[data-served-head='{mid}'] "
                               "[data-served-pin-now]\")", timeout=15000)
        assert served.get(mid)["pin"]["ctx"] == 65536
        assert served.get(mid)["key"] == "kept-key-17j"
        assert page.errors == []
    finally:
        srv.close()
