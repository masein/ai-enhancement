"""18c part 4 on the page (points 18 to 22), at 400 and 1440 px, light and
dark: the header's list holds the rented runs it counts; the busy notice
comes and goes on an open page; the model's reply wraps and keeps its lines;
the agent pages name the model as Runs does; an error of ours said once; a
busy end a week out says its date; Copy id gives focus back to its ⋯; a
rented run's times and notes in words, a refused import said; a run here's
full id in its hover; the window's line on its own page. Invented runs,
boxes and conversations; a stand-in llama-server; nothing is fetched."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent18 import run_json, trial
from conftest import set_name
from fake_openai import FakeServer
from test_14_3_browser import steady_shot
from test_17i_runs_browser import post, run_row, step

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18c"
HLE, MMLU = "hle_text_cais", "mmlupro_tiger"
AGENT = "served/agent-18c4"
KEY = "swebench-multilingual__agent-18c4__k1"
LONG = "x" * 700                                            # one word, longer than any card
SAID = f"First line of the reply.\nSecond line.\n{LONG}"


def rented_world():
    """three rented runs: Humanity's Last Exam on four boxes (A3's and A4's
    part ending when their box frees), one box's MMLU-Pro running beside a
    step abandoned, and one done whose import the board refused"""
    import import_frontier as imf
    imf.boxes_path().unlink(missing_ok=True)
    now = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 3 * 3600))
    post([
        # A1's box has a second step after this one (its plan's): its box frees later
        *[step("served/18c4-bf16", f"A{k}", f"A{k}-1", shard=f"{k}/4", started_at=started,
               line=f"Humanity's Last Exam {100 + k} of 540 · 30 s an answer · 2.0 h left")
          for k in range(1, 5)],
        step("served/18c4-lda", "A6", "A6-1", tasks=(MMLU,), started_at=started,
             line="MMLU-Pro 3,000 of 12,032 · 5.3 s an answer · 13.3 h left"),
        step("served/18c4-lda", "A6", "A6-2", state="abandoned", tasks=(HLE,),
             started_at=started),
        step("served/18c4-k8", "A5", "A5-1", state="whole", tasks=(MMLU,), started_at=started,
             line="MMLU-Pro: 12,032 of 12,032 answered",
             import_refused="its GGUF isn't the file registered for served/18c4-k8")])


def runs_page(page, live, width, q="18c4"):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=runs")
    page.wait_for_selector("[data-queue-table]")
    page.get_by_label("filter queue").fill(q)
    page.wait_for_function(f"state.qQ === {q!r}")


def no_overflow(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


# ---------------------------------------------------------------------------
# 18 and 22: the header's list; a rented run's words; a run here's hover
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_18_22_the_header_lists_what_it_counts_and_rented_runs_in_words(live, page, scheme):
    from service import db
    rented_world()
    sid = db.add("Qwen/Qwen3.5-2B-18c4", "instruct", "frontier", "masein", "",
                 tasks=["gpqa_diamond_epoch"], status="done")
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        runs_page(page, live, width)
        hle = run_row(page, "served/18c4-bf16", HLE)
        hle.wait_for()
        # 18: the pill's count and its list agree, the rented runs listed
        page.wait_for_function("+document.querySelector('[data-runs]').dataset.runs === "
                               "runsNow().running.length + rentedRunning()")
        page.locator("[data-runs] ").first.click()
        lst = page.locator("#pop-runs [data-runs-list]")
        lst.wait_for()
        for rid in ("rented:served/18c4-bf16|hle_text_cais|on",
                    "rented:served/18c4-lda|mmlupro_tiger|on"):
            assert lst.locator(f"[data-run-rented='{rid}']").count() == 1, rid   # d94f7f6: none
        assert lst.locator("[data-nothing-running]").count() == 0
        n = int(page.locator("[data-runs]").first.get_attribute("data-runs"))
        running_lines = lst.locator(".runline:not([data-run-done])").evaluate_all(
            "xs => xs.filter(x => !/queued|waiting/.test(x.firstChild.textContent)).length")
        assert running_lines == n, (running_lines, n)
        page.keyboard.press("Escape")
        # 22: a one-box run's own row: done about; the four-box run: this part
        lda = run_row(page, "served/18c4-lda", MMLU)
        assert lda.locator("[data-rented-finish]").inner_text().startswith("done about ")
        assert hle.locator("[data-rented-finish]").inner_text().startswith("this part done ")
        assert "a step abandoned" in lda.inner_text() and "A6-2" not in lda.inner_text()
        # A3's and A4's boxes free when their part is done (as the reviewers saw
        # them): the time said once
        page.evaluate("""() => { const r = state.boxes.runs.find(x => x.hf_id ===
            'served/18c4-bf16'); for (const b of r.boxes) if (['A3', 'A4'].includes(b.label))
            b.box_finish = b.step_finish; }""")
        hle.locator("[data-rented-open]").click()
        page.wait_for_selector("[data-rented-boxes]")
        lines = {x.get_attribute("data-rented-box").split("|")[0]: x.inner_text()
                 for x in page.locator("[data-rented-box]").all()}
        assert "box free" in lines["A1"]                       # its box frees later
        for b in ("A3", "A4"):
            assert "this part done" in lines[b] and "box free" not in lines[b], lines[b]
        # d94f7f6: "this part done Thu 20:32 · box free Thu 20:32"
        # 22: a refused import said, never "not imported yet"
        k8 = run_row(page, "served/18c4-k8", MMLU)
        assert k8.locator("[data-rented-refused]").inner_text() == (
            "its import was refused: its GGUF isn't the file registered for served/18c4-k8")
        assert k8.locator("[data-rented-not-imported]").count() == 0
        # 22: a run here's whole id in its model cell's hover
        here = page.locator(f"[data-queue-row='{sid}'] .qc-model")
        assert here.get_attribute("title").startswith("Qwen/Qwen3.5-2B-18c4")
        assert no_overflow(page)
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(page.locator("table[data-queue-table]"),
                    SCREENS / f"runs-{width}-{scheme}.png")
    assert page.errors == []


# ---------------------------------------------------------------------------
# 19 and 22: the busy notice on an open page; its date a week out
# ---------------------------------------------------------------------------

@pytest.fixture
def busy_world(live):
    """a served model, and an agent run on it whose last line is old"""
    import import_agent
    import service.app as appmod
    from service import config, db, served
    srv = FakeServer()
    mid = "served/busy-18c4"
    db.served_put({"id": mid, "name": "busy 18c4", "base_url": srv.base, "key": "",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 262144",
                   "thinking": "auto", "phone": False,
                   "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein", "at": 0})
    served.write_meta(served.get(mid))
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / "swebench-multilingual__busy-18c4__k1"
    run_json(rdir, ["inv__repo1-1001"], model=mid)
    trial(rdir, "inv__repo1-1001")
    prog = {"line": "1 of 10 · 1 resolved", "at": time.time() - 3600,
            "until": time.time() + 8 * 86400}
    (rdir / "progress.json").write_text(json.dumps(prog))
    assert import_agent.main(["--progress", str(rdir)]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield mid, rdir, prog
    srv.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_19_the_busy_notice_comes_and_goes_on_an_open_page(live, page, busy_world, scheme):
    mid, rdir, prog = busy_world
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        (rdir / "progress.json").write_text(json.dumps({**prog, "at": time.time() - 3600}))
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + f"/#model={mid}")
        set_name(page, "masein")
        page.wait_for_selector("[data-test-model]")
        page.wait_for_function("state.agentBusyAt > 0")
        assert page.locator(f"[data-agent-busy='{mid}']").count() == 0   # not posting: free
        # the run posts again, nothing else on the page moves: a minute on, said
        (rdir / "progress.json").write_text(json.dumps({**prog, "at": time.time()}))
        page.evaluate("state.agentBusyAt = 1")                  # the minute, gone by
        line = page.locator(f"[data-agent-busy='{mid}']")
        line.wait_for(timeout=20000)                            # d94f7f6: never
        words = line.inner_text()
        until = time.strftime("%b", time.localtime(prog["until"]))
        assert until in words, words                            # a week out: its date
        assert no_overflow(page)
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(line, SCREENS / f"busy-{width}-{scheme}.png")
        # it stops posting: the notice goes
        (rdir / "progress.json").write_text(json.dumps({**prog, "at": time.time() - 3600}))
        page.evaluate("state.agentBusyAt = 1")
        line.wait_for(state="detached", timeout=20000)
    assert page.errors == []


# ---------------------------------------------------------------------------
# 20 and 22: the reply wraps; the names; an error of ours said once
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_world(live):
    import import_agent
    import service.app as appmod
    from service import config, db
    db.served_put({"id": AGENT, "name": "agent 18c4", "base_url": "http://127.0.0.1:9/v1",
                   "key": "", "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "", "thinking": "auto",
                   "phone": False, "pin": {"model": "x", "ctx": 262144}, "by": "masein", "at": 0})
    rdir = Path(config.BENCH_ROOT) / "agent-runs" / KEY
    if not (rdir / "run.json").exists():
        run_json(rdir, ["inv__said-1", "inv__ours-1"], model=AGENT)
        t = trial(rdir, "inv__said-1")
        conv = json.loads((t / "agent-host" / "mini-swe-agent.trajectory.json").read_text())
        conv["messages"][2]["content"] = SAID
        (t / "agent-host" / "mini-swe-agent.trajectory.json").write_text(json.dumps(conv))
        for k in range(3):                                    # three errors of ours, the same
            trial(rdir, "inv__ours-1", stamp=1_700_000_000_000 + k, result="", exc="RuntimeError",
                  started=False, message="Docker compose command failed for environment x")
        (rdir / "progress.json").write_text(json.dumps({"line": "2 of 2"}))
        assert import_agent.main([str(rdir)]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    return rdir


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_20_22_the_reply_wraps_and_the_run_names_its_model(live, page, agent_world, scheme):
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + f"/#tab=runs&agent={KEY}")
        page.wait_for_selector(f"[data-agent-page='{KEY}']")
        h2 = page.locator(f"[data-agent-page-name='{KEY}']")
        name = page.evaluate(f"runName({{hf_id: {AGENT!r}, suite: 'agent'}})")    # as Runs says it
        assert not name.startswith("served/")
        assert h2.inner_text() == f"SWE-bench Multilingual · {name}"     # d94f7f6: served/…
        assert h2.get_attribute("title") == AGENT
        row = page.locator("[data-agent-task='inv__ours-1']")
        text = row.inner_text()
        assert text.count("Docker compose command failed") == 1, text     # d94f7f6: 3
        assert row.locator("[data-agent-tries='inv__ours-1']").inner_text() == (
            "asked 3 times, each: RuntimeError: Docker compose command failed for "
            "environment x")
        # 20: the model's reply wraps inside its card and keeps its lines
        page.goto("about:blank")
        page.goto(live["base"] + f"/#tab=runs&agent={KEY}&task=inv__said-1")
        said = page.locator("[data-agent-said='1']")
        said.wait_for()
        assert said.evaluate("e => getComputedStyle(e).whiteSpace") == "pre-wrap"
        assert said.inner_text().split("\n")[:2] == ["First line of the reply.", "Second line."]
        card = page.locator("[data-agent-conv='inv__said-1']")
        assert said.evaluate("(e) => e.getBoundingClientRect().right") <= card.evaluate(
            "(e) => e.getBoundingClientRect().right") + 1                  # d94f7f6: far past
        assert no_overflow(page)
        SCREENS.mkdir(parents=True, exist_ok=True)
        steady_shot(said, SCREENS / f"said-{width}-{scheme}.png")
    # 22: the card's run line by its board name too
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=benchmarks")
    got = page.locator(f"[data-agent-run-name='{KEY}']")
    got.wait_for()
    assert got.inner_text().startswith(name + " ·") and got.get_attribute("title") == AGENT
    assert page.errors == []


# ---------------------------------------------------------------------------
# 22: Copy id gives focus back to its ⋯; the window's line on its own page
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_22_copy_id_gives_focus_back_to_its_menu_button(live, page, scheme):
    from service import db
    sid = db.add("served/copy-18c4", "instruct", "frontier", "masein", "", tasks=[HLE],
                 status="done")
    page.emulate_media(color_scheme=scheme)
    for width in (1440, 400):
        runs_page(page, live, width, q="copy-18c4")
        btn = page.locator(f"[data-row-menu='q{sid}']")
        btn.wait_for()
        btn.click()
        page.locator("[data-act='copy-id']").first.click()
        page.wait_for_function(f"document.activeElement && document.activeElement.dataset"
                               f".rowMenu === 'q{sid}'")                 # d94f7f6: the body
    assert page.errors == []


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_22_the_window_line_on_the_models_own_page(live, page, scheme):
    import service.app as appmod
    from service import db, served
    srv = FakeServer()
    try:
        mid = "served/window-18c4"
        db.served_put({"id": mid, "name": "window 18c4", "base_url": srv.base, "key": "",
                       "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 16384",
                       "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, "")), "by": "masein",
                       "at": 0})
        served.write_meta(served.get(mid))
        srv.ctx = 65536
        page.emulate_media(color_scheme=scheme)
        for width in (1440, 400):
            appmod._cache.update(key=None, payload=None, at=0.0)
            page.set_viewport_size({"width": width, "height": 1000})
            page.goto("about:blank")
            page.goto(live["base"] + f"/#model={mid}")
            set_name(page, "masein")
            line = page.locator("[data-served-pin-now='window']")
            line.wait_for()
            words = line.inner_text()
            assert "on the model's page" not in words, words                # d94f7f6: there
            assert page.locator(f"[data-served-window-note='{mid}']").inner_text().strip() == (
                "If you use it, a run of this model that stopped part-way starts again from "
                "its first question.")
            assert no_overflow(page)
            SCREENS.mkdir(parents=True, exist_ok=True)
            steady_shot(line, SCREENS / f"window-{width}-{scheme}.png")
        assert page.errors == []
    finally:
        srv.close()
