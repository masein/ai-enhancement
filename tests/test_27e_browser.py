"""27e in a real browser (docs/prompts/phase-27e-top-bar-q2-baseline-and-small-fixes.md):
a place in the top bar, or in its menu, opens its front page whatever is
open inside it, and Back returns to what was open; On-device with Q2, our
same-size baseline, and on IFEval and IFBench our Q4 standing in for BF16,
said on every row and once in the panel's note, Q2 a "–"; "What each keeps"'s
axis labels never touching, at 1280 and 400; Step checkpoints' baseline
switch, Q2 by default. The world is 21c's scorecard with 24b's file sizes,
the cards entered by masein, IFEval and IFBench invented for step8804,
step4000 and Q4 (our BF16 hasn't run them), and invented HLE answers."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from service import db, reports
from service import frontier as sf
from test_21c_page import SPEC, results
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways
from test_24b_browser import FILES

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27e"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]
Q2, Q4 = "Q2 (UD-Q2_K_XL)", "Q4 (UD-Q4_K_XL)"
HLE = "hle_text_cais"
# invented, so the shares come out as the brief's: {label: (IFEval, IFBench)}
IF = {Q4: (0.900, 0.600), "step8804": (0.883, 0.568), "step4000": (0.8577, 0.520)}
# invented HLE answers: right on the first k of 60
RIGHT = {"step1000": 3, "step4000": 12, "step8804": 13, Q4: 22, Q2: 15}
STAND = "of our Q4, standing in for BF16"


def follow(label: str, task: str, v: float, se: float, n: int) -> None:
    """an invented IFEval or IFBench score with its four numbers, as an import keeps them"""
    row = SPEC[label]["row"]
    results(reports.row_dir(row), row.removesuffix(" · thinking"), task, v, se, n, 0)
    f = next((reports.row_dir(row) / f"{task}_0shot" / "frontier").glob("results_*.json"))
    got = json.loads(f.read_text())
    x = {"v": v, "se": se}
    got["frontier"]["follow"] = {"metric": "strict" if task == "ifeval_google" else "loose",
                                 "prompt_strict": x, "prompt_loose": x, "inst_strict": x,
                                 "inst_loose": x, "prompts": n, "instructions": n}
    f.write_text(json.dumps(got))


def errors(page) -> list[str]:
    """the page's errors, but a request this world has no answer for (an
    invented agent run, a benchmark's questions it hasn't cached)"""
    return [e for e in page.errors if "Failed to load resource" not in e]


def hle(label: str, k: int, n: int = 60) -> None:
    d = sf.task_dir(reports.row_dir(SPEC[label]["row"]), HLE)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"samples_{HLE}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": q, "doc": {"id": f"h{q}"}, "resps": [["Exact Answer: x"]],
                    "filtered_resps": ["x"],
                    "frontier": [{"epoch": 0, "ok": q < k, "ran_out": False}]}) + "\n"
        for q in range(n)))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import enter_report_cards as erc
    import service.app as appmod
    from service import report_stats
    scorecard()
    for label, more in FILES.items():
        mid = SPEC[label]["row"].removesuffix(" · thinking")
        db.served_put({**(db.served_get(mid) or {}), **more})
    assert erc.main(["on-device", "--by", "masein"]) == 0
    for label, (ife, ifb) in IF.items():
        follow(label, "ifeval_google", ife, 0.014, 541)
        follow(label, "ifbench_test", ifb, 0.028, 300)
    for label, k in RIGHT.items():
        hle(label, k)
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def shot(page, loc, path: Path) -> None:
    for _ in range(20):
        page.wait_for_timeout(100)
        b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + "
                         "scrollX, y: r.top + scrollY, width: r.width, height: r.height}; }")
        if b["width"] > 0 and b["height"] > 0:
            break
    page.screenshot(path=str(path), full_page=True, clip=b)


def at(page, live, hash_: str, wait: str, width: int = 1280, scheme: str = "light"):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/" + hash_)
    page.locator(wait).first.wait_for(timeout=30000)


def hash_(page) -> str:
    return page.evaluate("location.hash")


# ---------------------------------------------------------------------------
# Part 1: a place in the top bar opens its front page; Back returns
# ---------------------------------------------------------------------------

def place(page, pid: str) -> None:
    page.locator(f"#tabs [role=tab][data-tab='{pid}']").click()


def back_to(page, want: str, wait: str) -> None:
    page.go_back()
    page.locator(wait).first.wait_for(timeout=30000)
    assert hash_(page) == want


def test_reports_a_report_then_reports_opens_the_list_and_back_the_report(live, page):
    inner = "#tab=reports&report=step-checkpoints"
    at(page, live, inner, "[data-report='step-checkpoints']")
    place(page, "reports")
    page.locator("[data-reports-list]").wait_for(timeout=30000)
    assert hash_(page) == "#tab=reports"
    assert page.locator("[data-report='step-checkpoints']").count() == 0
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENS / "reports-front-1280-light.png"))
    back_to(page, inner, "[data-report='step-checkpoints']")
    # and once more from the list: Reports again is still the list, no new entry
    place(page, "reports")
    page.locator("[data-reports-list]").wait_for()
    n = page.evaluate("history.length")
    place(page, "reports")
    page.wait_for_timeout(200)
    assert hash_(page) == "#tab=reports" and page.evaluate("history.length") == n
    assert page.errors == []


def test_a_two_builds_report_and_a_ran_out_reader_close_too(live, page):
    # its two builds' picks, and its reader on a question
    pd = "#tab=reports&report=paired&a=step4000&b=Q4%20(UD-Q4_K_XL)"
    at(page, live, pd, "[data-report='paired']")
    place(page, "reports")
    page.locator("[data-reports-list]").wait_for(timeout=30000)
    assert hash_(page) == "#tab=reports"
    page.go_back()
    page.locator("[data-report='paired']").wait_for(timeout=30000)
    assert "a=step4000" in hash_(page) and "b=Q4" in hash_(page)
    # the ran-out report's model and benchmark
    ro = "#tab=reports&report=ran-out-of-room&model=step4000"
    at(page, live, ro, "[data-report='ran-out-of-room']")
    place(page, "reports")
    page.locator("[data-reports-list]").wait_for(timeout=30000)
    assert hash_(page) == "#tab=reports"
    page.go_back()
    page.locator("[data-report='ran-out-of-room']").wait_for(timeout=30000)
    assert "model=step4000" in hash_(page)
    assert page.errors == []


def test_every_place_opens_its_front_page_and_back_returns(live, page):
    import time
    mid = SPEC[Q4]["row"]
    # a chat as kept (a served row of this world isn't offered to chat with)
    c = {"id": "c27e00000001", "who": "masein", "by": "masein", "model": mid, "model2": "",
         "settings": {}, "title": "Invented chat", "messages": [], "created_at": time.time(),
         "updated_at": time.time()}
    db.chat_put(c)
    model = "#model=" + quote(mid, safe="")
    cases = [
        # (inside, its marker, the place, its front page's hash, its marker)
        (f"#tab=playground&chat={c['id']}", "#view", "playground", "#tab=playground", "#view"),
        ("#tab=benchmarks&q=gpqa_diamond_epoch", "[data-qx-back]", "benchmarks",
         "#tab=benchmarks", "[data-catalog]"),
        ("#tab=improve&sub=training", "#view h2:text-is('Training runs')", "improve",
         "#tab=improve&sub=model", "#view"),
        (model, "h1.mtitle", "home", "#tab=home", "[data-needs-you]"),
        (model, "h1.mtitle", "models", "#tab=models", "[data-models-view]"),
    ]
    for inner, mark, pid, front, fmark in cases:
        at(page, live, inner, mark)
        was = hash_(page)
        place(page, pid)
        page.locator(fmark).first.wait_for(timeout=30000)
        page.wait_for_timeout(150)
        got = hash_(page)
        assert got == front or got.startswith(front + "&"), (pid, got)
        if pid != "improve":
            assert got != was, pid
        assert page.locator("h1.mtitle").count() == 0 or pid not in ("home", "models"), pid
        if pid == "benchmarks":
            assert page.locator("[data-qx-back]").count() == 0
        if pid == "playground":
            assert "chat=" not in got
        page.go_back()
        page.locator(mark).first.wait_for(timeout=30000)
        assert hash_(page) == was, (pid, hash_(page), was)
    assert errors(page) == []


def test_runs_all_runs_closes_an_opened_run_and_back_opens_it_again(live, page):
    inner = "#tab=runs&agent=invented-run"
    at(page, live, inner, "[data-agent-back]")
    page.locator("#runs [data-runs]").click()
    page.locator("[data-all-runs-link]").click()
    page.locator("[data-all-runs]").first.wait_for(timeout=30000)
    assert hash_(page) == "#tab=runs"
    assert page.locator("[data-agent-back]").count() == 0
    back_to(page, inner, "[data-agent-back]")
    assert errors(page) == []


def test_on_a_phone_the_menus_reports_opens_the_list(live, page):
    inner = "#tab=reports&report=step-checkpoints"
    at(page, live, inner, "[data-report='step-checkpoints']", width=400)
    page.locator("#menuBtn").click()
    page.locator("[data-place='reports']").click()
    page.locator("[data-reports-list]").wait_for(timeout=30000)
    assert hash_(page) == "#tab=reports"
    back_to(page, inner, "[data-report='step-checkpoints']")
    assert page.errors == []


# ---------------------------------------------------------------------------
# Part 2: On-device — Q2, and our Q4 standing in for BF16
# ---------------------------------------------------------------------------

def ticks_apart(page) -> dict[str, list[float]]:
    """each What-each-keeps panel's gaps between neighbouring axis labels, in px"""
    return page.evaluate("""() => Object.fromEntries([...document.querySelectorAll(
        '[data-od-keep-panel]')].map(p => {
          const r = [...p.querySelectorAll('[data-od-keep-tick]')]
            .map(t => t.getBoundingClientRect()).sort((a, b) => a.left - b.left);
          return [p.dataset.odKeepPanel, r.slice(1).map((x, i) => x.left - r[i].right)];
        }))""")


@pytest.mark.parametrize("width,scheme", SIZES)
def test_on_device_has_q2_and_the_stand_in(live, page, width, scheme):
    at(page, live, "#tab=reports&report=on-device", "[data-report-section='od-keep']", width,
       scheme)
    # the table (a phone shows its cards): Q2 a row of ours, its line under its name
    if width > 700:
        row = page.locator(f"tr[data-od-row='{Q2}']")
        assert row.get_attribute("data-od-ours") == "true"
        line = row.locator("[data-mt-line]")
        assert "the original, squeezed to 2 bits — our same-size baseline" in (
            (line.get_attribute("title") or "") + line.inner_text())
        assert page.locator(f"[data-od-fact='{Q2}|memory'] .mt-n").inner_text() == "13.0 GB"
        # a stand-in share says whose, the whole on hover
        k = page.locator("[data-od-keep='step8804|ifeval']")
        assert k.inner_text() == "98.1% of our Q4"
        assert k.get_attribute("title").startswith(STAND + " (Q4 keeps ")
        for lb, b, want in (("step4000", "ifbench", "86.7"), ("step8804", "ifbench", "94.7"),
                            ("step4000", "ifeval", "95.3")):
            assert page.locator(f"[data-od-keep='{lb}|{b}']").inner_text() == (
                f"{want}% of our Q4")
        # GPQA Diamond: Q2 of our BF16, like the others
        assert page.locator(f"[data-od-keep='{Q2}|gpqa']").inner_text().endswith(
            "% of its base")
    assert page.locator(f"[data-od-keep-dot='gpqa:{Q2}']").count() == 1
    # What each keeps: IFEval and IFBench of our Q4 — each row's line, the note once
    for b in ("ifeval", "ifbench"):
        panel = page.locator(f"[data-od-keep-panel='{b}']")
        note = panel.locator(f"[data-od-stand-note='{b}']")
        assert note.count() == 1
        assert "standing in for BF16 (Q4 keeps " in note.inner_text()
        assert "of BF16 elsewhere)" in note.inner_text()
        for lb in ("step8804", "step4000"):
            base = panel.locator(f"[data-od-base='{b}:{lb}']")
            assert base.get_attribute("data-od-stand-in") == b
            assert " ".join(base.evaluate(
                "e => [...e.querySelectorAll('tspan')].map(t => t.textContent)")) == STAND
            assert base.locator("title").text_content().startswith(STAND + " (Q4 keeps ")
        # Q2 hasn't run it: a "–", not run yet on hover
        none = panel.locator(f"[data-od-keep-none='{b}:{Q2}']")
        assert none.text_content().startswith("–")
        assert none.locator("title").text_content() == f"{Q2}: not run yet"
        assert panel.locator(f"[data-od-keep-dot='{b}:{Q4}']").count() == 0   # the base itself
    # the others' panels have no note
    assert page.locator("[data-od-keep-panel='gpqa'] [data-od-stand-note]").count() == 0
    # Q2 in the memory chart and its legend
    assert page.locator(f"[data-od-mem-dot='gpqa:{Q2}']").count() == 1
    assert page.locator(f"[data-od-leg='{Q2}']").count() == 1
    # Part 3: the axis labels never touch
    for b, gaps in ticks_apart(page).items():
        assert gaps and min(gaps) > 0, (b, gaps)
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    for sec in ("od-table", "od-keep", "od-memory"):
        shot(page, page.locator(f"[data-report-section='{sec}']"),
             SCREENS / f"{sec}-{width}-{scheme}.png")


# ---------------------------------------------------------------------------
# Part 2: Step checkpoints — the baseline switch, Q2 by default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,scheme", SIZES)
def test_step_checkpoints_baseline_switch(live, page, width, scheme):
    at(page, live, "#tab=reports&report=step-checkpoints", "[data-report-section='training']",
       width, scheme)
    sw = page.locator("[data-rp-baseline]")
    assert sw.get_attribute("data-rp-baseline") == Q2
    assert page.locator(f"[data-rp-base='{Q2}']").get_attribute("aria-checked") == "true"
    assert page.locator(f"[data-rp-base='{Q2}']").inner_text() == "Q2 · 13.0 GB"
    assert page.locator(f"[data-rp-base='{Q4}']").inner_text() == "Q4 · 22.0 GB"
    tr = page.locator("[data-report-section='training']")
    against = tr.locator("[data-tr-against]").inner_text()
    assert against.startswith("Against Q2, step8804 ")
    assert tr.locator("[data-tr-ref-label='hle:Q2']").count() == 1
    assert tr.locator("[data-tr-ref-label='hle:Q4']").count() == 0
    assert tr.locator(f"[data-tr-key='{Q2}']").count() == 1
    fl = page.locator("[data-fl-sentence]").inner_text()
    assert "against Q2 (UD-Q2_K_XL) on IFEval" in fl
    numbers = page.locator("[data-report-section='ours'] [data-mt-row]").all_inner_texts()
    segs = tr.locator("[data-tr-seg]").all_inner_texts()
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-report='step-checkpoints']"),
         SCREENS / f"switch-head-{width}-{scheme}.png")
    shot(page, tr, SCREENS / f"switch-training-{width}-{scheme}.png")
    # Q4: the lines follow; the numbers don't change
    page.locator(f"[data-rp-base='{Q4}']").click()
    page.locator(f"[data-rp-baseline='{Q4}']").wait_for()
    assert "base=Q4" in hash_(page)
    assert tr.locator("[data-tr-against]").inner_text().startswith("Against Q4, step8804 ")
    assert tr.locator("[data-tr-ref-label='hle:Q4']").count() == 1
    assert "against Q4 (UD-Q4_K_XL) on IFEval" in page.locator("[data-fl-sentence]").inner_text()
    assert page.locator("[data-report-section='ours'] [data-mt-row]").all_inner_texts() == numbers
    assert tr.locator("[data-tr-seg]").all_inner_texts() == segs
    # a reload keeps it; Back returns to Q2
    page.reload()
    page.locator(f"[data-rp-baseline='{Q4}']").wait_for(timeout=30000)
    page.go_back()
    page.locator(f"[data-rp-baseline='{Q2}']").wait_for(timeout=30000)
    assert "base=" not in hash_(page)
    assert page.errors == []
