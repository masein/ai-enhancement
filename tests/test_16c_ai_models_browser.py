"""16c on the page, parts 2 to 4 and 7: the key's card says what a labeller's
last batch did and why it failed, offers "Pro only" or "Pro, then the full
set" and prices what Start sends; AI models shows what the OpenRouter key may
still spend and refreshes with the page's tick without losing a value being
typed; a served model's "How it's served" is corrected from its page. At 1400
and 375 px. Batches written on disk, a fake key allowance; nothing calls
OpenRouter."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import set_name
from test_14_3_browser import steady_shot
from test_14_3_mobile_mmlu import FIXTURE, fake_pin, pin_fixture

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16c"
WIDTHS = [1400, 375]
ME = "masein"
LIMIT_BODY = json.dumps({"error": {"code": 402, "message": "Key limit exceeded",
                                   "metadata": {"limit_source": "openrouter_key_limit"}}})
LIMIT_WORDS = ("OpenRouter refused the key: it has reached its own spending limit, or the account "
               "is out of credit. That is fixed on OpenRouter’s side: raise the key’s limit or add "
               "credit there")


def fake_batch(kind: str, n: int, fail: int, slot: str = "", status: str = "done") -> str:
    """a batch as OpenRouterChat leaves it on disk: n sent, `fail` refused
    for the key's limit"""
    from service import config, db, mmp_key
    import mobile_mmlu as mmp
    bid = "or_" + uuid.uuid4().hex[:12]
    d = config.BENCH_ROOT / "llm_batches" / "openrouter" / bid
    d.mkdir(parents=True)
    lids = [q["lid"] for q in mmp.pool()][:n] if kind == mmp_key.KIND else [str(i) for i in range(n)]
    cids = [f"mmpk:{slot}:{k}" if kind == mmp_key.KIND else f"j:{k}" for k in lids]
    (d / "requests.jsonl").write_text("".join(json.dumps({"custom_id": c, "system": "", "user": "q",
                                                          "max_tokens": 10}) + "\n" for c in cids))
    now = time.time()
    (d / "results.jsonl").write_text("".join(json.dumps(
        {"custom_id": c, "text": "" if i < fail else '{"answer": "A"}', "at": now + i,
         "error": (f"POST https://openrouter.test/api/v1/chat/completions: HTTP 402: {LIMIT_BODY}"
                   if i < fail else ""), "status": 402 if i < fail else None}) + "\n"
        for i, c in enumerate(cids)))
    db.batch_add(bid, kind, 0, n, "openrouter", "x/y")
    db.batch_finish(bid, status, LIMIT_BODY if status == "failed" else "")
    if kind == mmp_key.KIND:
        p = mmp_key._bdir() / f"{bid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"slot": slot, "pin": fake_pin("x/y"), "ids": lids, "by": ME,
                                 "at": now}))
    return bid


@pytest.fixture(scope="module", autouse=True)
def board(live):
    """Pro's invented rows; three labellers pinned; the second's last batch
    refused for the key's limit part-way, a judge batch too; the key with
    $0.40 left of its $50"""
    import service.app as appmod
    from service import ai_models, config, db, served
    import mobile_mmlu as mmp
    saved = config.OPENROUTER_API_KEY
    with pytest.MonkeyPatch.context() as mp:
        d = Path(config.MMP_DIR)
        d.mkdir(parents=True, exist_ok=True)
        (d / "mobile-mmlu-pro.csv").write_bytes(FIXTURE.read_bytes())
        pin_fixture(mp)
        config.OPENROUTER_API_KEY = "test-key"
        for s in mmp.SLOTS:
            db.ai_set(f"labeller:{s}", fake_pin(mmp.DEFAULT_LABELLERS[s]["id"]), ME)
        fake_batch("mmpk", 3, 1, slot="second")
        fake_batch("judge", 4, 4, status="failed")
        db.ai_set("job:judge", fake_pin("deepseek/deepseek-v4.1-flash"), ME)
        mp.setattr(ai_models, "drifted", lambda pin: "")
        ai_models._KEY.update(got={"limit": 50.0, "remaining": 0.4, "usage": 49.6, "reset": None},
                              at=time.time() + 3600, why="", asking=False)
        # a served model whose description says a flag its launch doesn't
        rec = {"id": "served/orig", "name": "Qwen3.6 original", "base_url": "http://127.0.0.1:9/v1",
               "how": "k4, --cpu-moe", "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "auto",
               "pin": {"model": "m", "file": "f.gguf"}, "flags": "--n-cpu-moe 21", "env": ""}
        db.served_put(rec)
        served.write_meta(rec)
        appmod._cache.update(key=None, payload=None, at=0.0)
        yield
        config.OPENROUTER_API_KEY = saved
        for s in mmp.SLOTS:
            db.ai_set(f"labeller:{s}", None, ME)
        ai_models._KEY.update(got=None, at=0.0)
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def no_sideways(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


@pytest.mark.parametrize("width", WIDTHS)
def test_the_key_card_says_what_failed_offers_pro_only_and_prices_what_start_sends(
        live, page, width):
    go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])", width)
    card = page.locator("[data-mmp-key]")
    # the second labeller's last batch, in OpenRouter's words
    assert card.locator("[data-mmp-last='second']").inner_text() == (
        "Second labeller’s last batch: 3 sent · 2 answered · 1 failed. The first failure: "
        + LIMIT_WORDS + ".")
    # Pro only, the default; what Start sends, priced
    scope = card.locator("[data-mmp-scope]")
    assert scope.get_attribute("data-mmp-scope") == "pro"
    assert card.locator("[data-mmp-scope-pick='pro']").is_checked()
    usd = page.evaluate("state.ai.mmp.sends.usd")
    cost = f"${usd:,.2f}"
    assert card.locator("[data-mmp-start]").inner_text() == (
        f"Start labelling: about {cost} · Pro only")
    assert card.locator("[data-mmp-total]").inner_text().startswith(
        f"Start: about {cost} · Pro only · this month ")
    # the key's own allowance, on the spend line; a job dearer than it is warned
    assert "the OpenRouter key may still spend $0.40 of its $50.00" in page.locator(
        "[data-ai-key-left]").inner_text()
    assert card.locator("[data-mmp-key-warning]").count() == (1 if usd > 0.4 else 0)
    # the judge's last batch failed whole: said under its model
    assert page.locator("[data-ai-last-batch='judge']").inner_text() == (
        "Its last batch: 4 sent · 0 answered · 4 failed. The first failure: " + LIMIT_WORDS + ".")
    no_sideways(page)
    shot(card, f"key-card-{width}.png")
    shot(page.locator("[data-ai-page]"), f"ai-page-{width}.png")
    assert page.errors == []


def test_choosing_pro_then_the_full_set_is_kept_with_who_and_when(live, page):
    from service import db
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, ME)
    go(page, live, "tab=ai", "[data-mmp-key]:not([data-mmp-key='loading'])")
    try:
        page.locator("[data-mmp-scope-pick='all']").check()
        page.wait_for_selector("[data-mmp-scope='all'] [data-mmp-scope-by]")
        assert page.locator("[data-mmp-scope-by]").inner_text().startswith(f"· chosen by {ME}, ")
        assert page.locator("[data-mmp-start]").inner_text().endswith("· Pro, then the full set")
        assert db.ai_get("mmp:scope") == "all"
    finally:
        db.ai_set("mmp:scope", "pro", ME)
    assert page.errors == []


def test_ai_models_refreshes_with_the_tick_and_a_value_being_typed_survives_it(live, page):
    from service import db
    go(page, live, "tab=ai", "[data-ai-spend]")
    before = page.locator("[data-ai-spend]").get_attribute("data-ai-spend")
    db.spend_add("judge", "x/y", "prov", 1000, 100, 1.25, "")
    # the figure changes on its own, within a tick or two — no reload
    page.wait_for_function(f"document.querySelector('[data-ai-spend]').dataset.aiSpend !== "
                           f"'{before}'", timeout=15000)
    # typing a new limit: the next ticks never take the value away
    page.locator("[data-ai-limit-edit]").click()
    box = page.locator("[data-ai-limit-input]")
    box.fill("77")
    db.spend_add("judge", "x/y", "prov", 1000, 100, 0.5, "")
    page.wait_for_timeout(11000)
    assert box.input_value() == "77"
    assert page.evaluate("document.activeElement.dataset.aiLimitInput") == "1"
    page.locator("[data-ai-limit-cancel]").click()
    page.wait_for_selector("[data-ai-limit-edit]")
    assert page.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_how_its_served_is_corrected_from_its_page(live, page, width):
    import service.app as appmod
    from service import db, served
    rec = {**served.get("served/orig"), "how": "k4, --cpu-moe", "how_was": []}
    db.served_put(rec)
    served.write_meta(rec)
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, ME)
    go(page, live, "model=" + quote("served/orig", safe=""), "[data-served-head]", width)
    warn = page.locator("[data-served-mismatch='served/orig']")
    assert warn.inner_text().startswith("“How it’s served” says --cpu-moe; the launch flags "
                                        "registered don’t")
    page.locator("[data-served-mismatch-edit='served/orig']").click()
    form = page.locator("[data-served-edit-form='served/orig']")
    form.wait_for()
    assert form.locator("[data-served-edit='how']").input_value() == served.get(
        "served/orig")["how"]
    no_sideways(page)
    shot(form, f"served-edit-{width}.png")
    if width == 375:
        form.locator("[data-served-edit-cancel]").click()
        page.wait_for_selector("[data-served-edit-form]", state="detached")
        return
    form.locator("[data-served-edit='how']").fill("k4, --n-cpu-moe 21")
    form.locator("[data-served-edit-save]").click()
    page.wait_for_selector("[data-served-edit-form]", state="detached")
    page.wait_for_function("!document.querySelector(\"[data-served-mismatch='served/orig']\")",
                           timeout=15000)
    assert "k4, --n-cpu-moe 21" in page.locator("[data-served-how='served/orig']").inner_text()
    assert served.get("served/orig")["how_was"][0]["how"] == "k4, --cpu-moe"
    assert page.errors == []
