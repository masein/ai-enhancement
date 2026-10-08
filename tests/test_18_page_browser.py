"""18 on the page: Benchmarks ▸ Agent tasks, a card each with our run's score
(a pilot says so, never ranked) and the published numbers as reference; an
agent run on Runs, one line like its neighbours; a run's page with its
tasks and the failures in one click; a task's conversation with HTML in it,
shown as text. Invented runs; nothing is fetched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent18 import run_json, trial
from test_14_3_browser import steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18"
KEY = "swebench-multilingual__invented-agent__k1"
NAMES = [f"inv__repo{i}-{1000 + i}" for i in range(10)]


@pytest.fixture
def agent_run(live):
    import import_agent
    import service.app as appmod
    from service import config
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / KEY
    if not (rdir / "run.json").exists():
        run_json(rdir, NAMES)
        for i, t in enumerate(NAMES):
            trial(rdir, t, result="resolved" if i < 6 else "unresolved", html=(i == 7),
                  minutes=18.0 + i)
        (rdir / "progress.json").write_text(json.dumps({"line": "10 of 10 · 6 resolved"}))
        assert import_agent.main([str(rdir)]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    return rdir


def test_the_catalogue_has_agent_tasks_with_our_pilot_and_the_published_numbers(live, page,
                                                                              agent_run):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks")
    sec = page.locator("[data-cat-section='agent']")
    sec.wait_for()
    assert sec.locator("[data-agent-card]").count() == 2
    card = sec.locator("[data-agent-card='swebench-multilingual']")
    card.locator(f"[data-agent-run='{KEY}']").wait_for()
    words = card.locator(f"[data-agent-words='{KEY}']").inner_text()
    assert words == "60.0% resolved ± 15.5 · 10 tasks · pilot: 10 of 300"
    assert "never averaged" in card.locator("[data-agent-fact='avg']").inner_text()
    assert card.locator("[data-agent-ref='Qwen3.6-35B-A3B']").inner_text() == "Qwen3.6-35B-A3B 67.2"
    groups = card.locator(".agentref").all_inner_texts()
    assert len(groups) == 2 and groups[0].startswith("reference Qwen (the model's vendor) · ")
    assert "Gemini 3 Flash 72.7 · Claude Opus 4.6 72.0 · GLM-5 69.7" in groups[1]
    deep = sec.locator("[data-agent-card='deepswe']")
    assert "Not run here yet." in deep.inner_text()
    assert "74 ± 4" in deep.locator("[data-agent-published]").inner_text()
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": 1400, "height": 2400})        # no header over the cards
    steady_shot(sec, SCREENS / "catalogue.png")
    assert page.errors == []


def test_an_agent_run_is_one_line_of_runs_and_opens_its_page(live, page, agent_run):
    from service import db
    sid = next(r["id"] for r in db.recent(200) if r.get("suite") == "agent")
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.get_by_label("filter queue").fill("agent on this server")
    row = page.locator(f"[data-queue-row='{sid}']")
    row.wait_for()
    assert "Agent tasks · SWE-bench Multilingual" in row.inner_text()
    assert "pilot: 10 of 300" in row.inner_text()
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(row, SCREENS / "run-line.png")
    page.locator(f"[data-agent-row-open='{KEY}']").click()
    page.wait_for_selector(f"[data-agent-page='{KEY}']")
    tasks = page.locator(f"[data-agent-tasks='{KEY}'] tbody tr")
    assert tasks.count() == 10
    page.locator("[data-agent-filter='failures']").click()          # the failures, one click
    page.wait_for_function(f"document.querySelectorAll(\"[data-agent-tasks='{KEY}'] tbody tr\")"
                           ".length === 4")
    assert all(x == "unresolved" for x in page.locator("[data-agent-task]").evaluate_all(
        "xs => xs.map(x => x.dataset.agentResult)"))
    assert page.locator("[data-agent-result='unresolved'] .st").first.inner_text() == \
        "Not resolved"
    steady_shot(page.locator(f"[data-agent-page='{KEY}']"), SCREENS / "run-page.png")
    assert page.errors == []


def test_a_tasks_conversation_shows_html_as_text(live, page, agent_run):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=runs&agent={KEY}&task={NAMES[7]}")
    conv = page.locator(f"[data-agent-conv='{NAMES[7]}']")
    conv.wait_for()
    assert page.evaluate("location.hash") == f"#tab=runs&agent={KEY}&task={NAMES[7]}"
    said = conv.locator("[data-agent-said='1']")
    assert said.inner_text() == "<script>alert('x')</script> <b>bold</b>"   # text, not HTML
    for part in ("said", "thought", "ran"):                          # no element of theirs
        assert conv.locator(f"[data-agent-{part}] *").count() == 0, part
    assert conv.locator("[data-agent-ran='1']").first.inner_text().startswith("$ ls -la && echo "
                                                                              "'<img")
    # long output folded; the thought folded; the patch and the verifier's output
    fold = conv.locator("details.agentout").first
    assert fold.get_attribute("open") is None and "63 lines" in fold.locator("summary").inner_text()
    assert conv.locator("[data-agent-thought='1']").text_content() == "thinking about <i>it</i>"
    assert conv.locator(f"[data-agent-patch='{NAMES[7]}']").inner_text().rstrip().endswith("+<new>")
    assert page.locator("[data-agent-exit]").last.get_attribute("data-agent-exit") == "Submitted"
    SCREENS.mkdir(parents=True, exist_ok=True)
    conv.locator("details.agentthought").first.locator("summary").click()
    steady_shot(conv, SCREENS / "conversation.png")
    assert page.errors == []
