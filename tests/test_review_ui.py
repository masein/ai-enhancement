"""The Review tab and the propose buttons, against a live service with the
fake LLM: the LIVE dashboard fetches, so this needs a real HTTP server."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

import make_fixture

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens"


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    import uvicorn

    from service import config, llm, llm_poller, worker
    import service.app as appmod
    root = tmp_path_factory.mktemp("live")
    tree = make_fixture.build(root)
    saved = {k: getattr(config, k) for k in (
        "BENCH_ROOT", "RESULTS_ROOT", "OUT_DIR", "DB_PATH", "ARTIFACTS_DIR", "LOGS_DIR",
        "DATASETS_DIR", "SUBMIT_TOKEN", "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_POLL_S")}
    for k, v in {"BENCH_ROOT": root, "RESULTS_ROOT": root / "results",
                 "OUT_DIR": root / "results" / "full", "DB_PATH": root / "service.sqlite3",
                 "ARTIFACTS_DIR": root / "artifacts", "LOGS_DIR": root / "logs",
                 "DATASETS_DIR": root / "datasets", "SUBMIT_TOKEN": "",
                 "LLM_PROVIDER": "fake", "LLM_MODEL": "fake-1", "LLM_API_KEY": "",
                 "LLM_POLL_S": 0.3}.items():
        setattr(config, k, v)
    worker_start = worker.start
    worker.start = lambda: None
    llm.reset()
    appmod._cache.update(key=None, payload=None, at=0.0)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("live server did not come up")
    yield {"base": base, "tree": tree, "root": root}
    server.should_exit = True
    th.join(5)
    llm_poller.stop()
    worker.start = worker_start
    for k, v in saved.items():
        setattr(config, k, v)
    llm.reset()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1240, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.errors = errors
    yield pg
    ctx.close()


def model_url(base, mid):
    return f"{base}/#model={quote(mid, safe='')}"


def open_mmlu(pg, base, mid):
    pg.goto(model_url(base, mid))
    pg.wait_for_selector("details.dx")
    det = pg.locator("details.dx", has=pg.locator(".dxname", has_text=re.compile(r"^mmlu[^_]"))).first
    det.locator("> summary").click()
    return det


def test_propose_buttons_carry_their_reasons(live, page):
    base = live["base"]
    det = open_mmlu(page, base, "fx/good-750m")
    econ = det.locator("details.dxcat[data-cat='economics']")
    assert econ.locator("button.propose").is_enabled()
    assert econ.locator(".propwhy").count() == 0
    law = det.locator("details.dxcat[data-cat='law']")
    assert law.locator("button.propose").is_disabled()
    assert "noise floor" in law.locator(".propwhy").text_content()
    for mid, needle in (("fx/chance-160m", "not cleared chance"),
                        ("fx/skewed-360m", "answer positions"),
                        ("fx/below-135m-it", "confidently wrong")):
        det = open_mmlu(page, base, mid)
        btns = det.locator("button.propose")
        assert btns.count() > 0 and all(not b.is_enabled() for b in btns.all()), mid
        assert needle in det.locator(".propwhy").first.text_content(), mid
    assert page.errors == []


def test_review_flow_in_the_browser(live, page):
    base = live["base"]
    # the LLM card says what is configured and what today has cost
    page.goto(base + "/#tab=review")
    page.wait_for_selector(".card h2:has-text('Review')")
    assert "fake/fake-1" in page.locator("#view").text_content()
    assert "Nothing waiting" in page.locator("#view").text_content()
    # a name, remembered for every decision on this page
    page.get_by_label("your name").first.fill("Omar")

    det = open_mmlu(page, base, "fx/good-750m")
    det.locator("details.dxcat[data-cat='economics'] button.propose").click()
    page.wait_for_selector(".card h2:has-text('Review')")
    assert "proposal #" in page.locator("#view").text_content()
    card = page.locator(".rv[data-proposal]").first
    card.locator("textarea").wait_for(timeout=20000)          # the poller and the 5 s poll
    text = card.text_content()
    assert "introductory economics" in text and "diagnosis-half items the LLM saw" in text
    assert "economics" in text and "ceiling" in text and "leaderboard-half items" in text
    assert "fx/good-750m · mmlu · economics" in text
    assert card.locator(".dxlead").count() >= 1                # the model's own findings
    card.locator("details summary").first.click()
    assert card.locator(".ex").count() == 8
    # approve, edited
    ta = card.locator("textarea")
    ta.fill(ta.input_value() + " Emphasise direction of effect.")
    card.get_by_label("your name").fill("Omar")
    card.get_by_role("button", name="Approve this spec").click()
    page.wait_for_selector(".rv[data-proposal] :text('Approved as edited')", timeout=10000)
    card = page.locator(".rv[data-proposal]").first
    assert "approved by Omar" in card.text_content()
    # generate
    card.get_by_label("item count").fill("20")
    card.get_by_label("your name").fill("Omar")
    card.get_by_role("button", name="Generate data").click()
    page.wait_for_selector(".rv[data-dataset]:has-text('ready')", timeout=20000)
    ds = page.locator(".rv[data-dataset]").first
    ds.locator("summary").click()
    dtext = ds.text_content()
    assert "20 kept of 20 generated" in dtext and "Contamination gate" in dtext
    assert "Provenance, in full" in dtext and "approver" in dtext and "items_sha256" in dtext
    href = ds.locator("a:has-text('items.jsonl')").get_attribute("href")
    with urllib.request.urlopen(f"{base}/{href}") as r:
        items = [json.loads(x) for x in r.read().decode().splitlines()]
    assert len(items) == 20
    # taint it through the API the way a training run would, then look at the board
    did = int(re.search(r"dataset #(\d+)", dtext).group(1))
    req = urllib.request.Request(f"{base}/api/truns", data=json.dumps(
        {"name": "gap-run", "datasets": [did]}).encode(),
        headers={"Content-Type": "application/json"})
    rid = json.loads(urllib.request.urlopen(req).read())["id"]
    urllib.request.urlopen(urllib.request.Request(f"{base}/api/truns/{rid}/event", data=json.dumps(
        {"step": 1, "detail": "fx/good-750m"}).encode(), headers={"Content-Type": "application/json"}))
    time.sleep(5.2)                                            # the payload debounce
    # the live page re-fetches results when a queue job finishes, not on a taint
    # change — a reload is how a viewer sees it, so that is what the test does
    page.goto(base + "/#tab=leaderboard")
    page.reload()
    page.wait_for_selector("table.lb")
    row = page.locator("table.lb tbody tr",
                       has=page.locator("a.mname", has_text=re.compile(r"^good-750m$"))).first
    assert "trained on mmlu diagnostics" in row.locator(".badge.taint").text_content()
    page.goto(model_url(base, "fx/good-750m"))
    page.wait_for_selector(".backlink")
    head = page.locator("#view .card").first.text_content()
    assert "excluded from its official average" in head and "carries no rank" in head
    assert "trained on data derived from this task" in page.locator("#view").text_content()
    # screenshots for the PR: the Review tab, light and dark, desktop and phone
    SCREENS.mkdir(exist_ok=True)
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        for width in (1240, 430):
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(base + "/#tab=review")
            page.wait_for_selector(".rv[data-dataset]")
            page.locator(".rv[data-dataset] summary").first.click()
            page.screenshot(path=SCREENS / f"review-{scheme}-{width}.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.errors == []
