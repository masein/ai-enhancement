"""18b part 3 on the page: a served model an agent run holds says so on its
page and in the Playground, at 1440 and 400 px, light and dark (point 15);
a run's page says what "tokens in" adds up and shows the last prompt, and
the catalogue says what each "±" is (point 18). A stand-in llama-server;
invented runs; nothing is fetched, no model runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent18 import run_json, trial
from conftest import set_name
from fake_openai import FakeServer
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18b"
MID = "served/agent-held-18b"


@pytest.fixture
def held(live):
    """a served model, and an agent run on it that posted a minute ago"""
    import import_agent
    import service.app as appmod
    from service import config, db, served
    srv = FakeServer()
    srv.key = "k-18b"
    db.served_put({"id": MID, "name": "agent held 18b", "base_url": srv.base, "key": srv.key,
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 262144",
                   "thinking": "auto", "phone": False,
                   "pin": served.pin_of(served.probe(srv.base, srv.key)), "by": "masein", "at": 0})
    served.write_meta(served.get(MID))
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / "swebench-multilingual__agent-held-18b__k1"
    run_json(rdir, ["inv__repo1-1001"], model=MID)
    trial(rdir, "inv__repo1-1001")
    (rdir / "progress.json").write_text(json.dumps({
        "line": "1 of 10 · 1 resolved · 30 min a task · about 5 h left", "at": time.time(),
        "until": time.time() + 5 * 3600}))
    assert import_agent.main(["--progress", str(rdir)]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield rdir
    srv.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_models_page_and_the_playground_say_an_agent_run_holds_it(live, page, held, scheme):
    from service import chat
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + f"/#model={MID}")
        set_name(page, "masein")
        line = page.locator(f"[data-agent-busy='{MID}'][data-agent-busy-where='model']")
        line.wait_for()                                           # 70df001: nothing said
        words = line.inner_text()
        assert words.startswith("Busy with an agent run until about "), words
        assert "The Playground and runs on this model wait behind it" in words
        assert line.evaluate("e => e.scrollWidth <= e.clientWidth + 1")      # no overflow
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(line, SCREENS / f"busy-model-{width}-{scheme}.png")
    c = chat.new_chat(MID, "masein")
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=playground&chat={c['id']}")
    pg = page.locator(f"[data-agent-busy='{MID}'][data-agent-busy-where='playground']")
    pg.wait_for()
    assert "busy with an agent run until about " in pg.inner_text()
    assert "A reply here waits behind its requests" in pg.inner_text()
    pg.locator(f"[data-agent-busy-open='{MID}']").wait_for()
    assert page.errors == []


def test_a_runs_page_says_what_tokens_in_is_and_the_cards_each_plus_minus(live, page, held):
    key = held.name
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=runs&agent={key}")
    page.locator(f"[data-agent-page='{key}']").wait_for()
    note = page.locator(f"[data-agent-tokens-note='{key}']").inner_text()
    assert note == ("Tokens in adds up every step’s whole prompt; the last prompt is how big the "
                    "conversation grew.")
    assert page.locator("[data-agent-col='Last prompt']").count() == 1
    assert page.locator("[data-agent-last-prompt='inv__repo1-1001']").inner_text() == "1,600"
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks")
    pm = page.locator("[data-agent-card='deepswe'] [data-agent-pm]")
    pm.wait_for()
    assert pm.inner_text() == "± is a 95% interval over 4 runs; ours is one standard error"
    ours = page.locator(f"[data-agent-words='{key}']")
    assert "(one standard error)" in ours.inner_text()
    assert page.errors == []
