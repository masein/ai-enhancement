"""18c part 3 on the page (point 13): a job whose batch waits at a run of
refusals has a Carry on button on the AI models page, at 400 and 1440 px,
light and dark; pressing it lifts the halt and the button goes. An invented
batch; nothing calls OpenRouter."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from conftest import set_name
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18c"
BID = "or_" + "c0ffee18c3aa"


def old_halt(d: Path) -> None:
    """a halt as 70df001 wrote it: no hold — held now, waiting for Carry on"""
    (d / "halt.json").write_text(json.dumps({
        "why": "waiting: OpenRouter refused 20 requests in a row — the key's limit. It tries "
               "again in 10 minutes", "status": 402, "at": time.time() - 3600, "n": 20}))


@pytest.fixture
def halted(live, monkeypatch):
    import service.app as appmod
    from service import config, db, llm_poller
    # the board's poller would take the batch up with no judge configured
    # here, and fail it: a judge on OpenRouter keeps it waiting
    monkeypatch.setattr(llm_poller, "tick", lambda: 0)
    d = Path(config.BENCH_ROOT) / "llm_batches" / "openrouter" / BID
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text(json.dumps({"custom_id": "q1", "body": {}}) + "\n")
    old_halt(d)
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("DELETE FROM llm_batches WHERE batch_id=?", (BID,))
    db.batch_add(BID, "judge", 0, 1, "openrouter", "openai/gpt-4.1")
    appmod._cache.update(key=None, payload=None, at=0.0)
    try:
        yield d
    finally:                                    # the other pages never see it
        with sqlite3.connect(config.DB_PATH) as c:
            c.execute("DELETE FROM llm_batches WHERE batch_id=?", (BID,))
        shutil.rmtree(d, ignore_errors=True)
        appmod._cache.update(key=None, payload=None, at=0.0)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_a_halted_batch_has_carry_on_on_the_ai_models_page(live, page, halted, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    for width in (1440, 400):
        old_halt(halted)
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + "/#tab=ai")
        btn = page.locator("[data-ai-carry-on='judge']")
        btn.wait_for()                                                    # d94f7f6: none
        assert btn.inner_text() == "Carry on"
        assert "It waits for Carry on" in page.locator(
            "[data-ai-last-batch='judge']").inner_text()
        # on the page, whatever re-render came between (a locator re-resolves)
        box = btn.evaluate("e => { const r = e.getBoundingClientRect(); "
                           "return {x: r.x, width: r.width}; }")
        assert box["x"] >= 0 and box["x"] + box["width"] <= width + 1
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(page.locator("[data-ai-job='judge']"), SCREENS / f"carry-on-{width}-"
                    f"{scheme}.png")
        btn.click()
        page.wait_for_selector("[data-ai-carry-on='judge']", state="detached")
        assert not (halted / "halt.json").exists()                        # lifted
    assert page.errors == []
