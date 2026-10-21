"""24 in a real browser, at 1280 and 400 px, light and dark: the two-builds
report as one table-chart — a row a benchmark, HLE included and waiting for
grading, one scale for every row's difference, a sure verdict's bar in the
accent colour, the split bar's parts in proportion — and its reader: a part
opens the questions, a card at a time with both answers side by side,
Previous/Next and the arrow keys, the link opening the same card, GPQA's
subject without its text, MMLU-Pro's question and options, a run each with
its mark, the filters; and the small fixes: round ticks, two reference
labels never on top of each other, an all-empty column hidden. Invented
questions and answers, worked out by the stats worker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frontier as fb
from service import config, reports
from service import frontier as sf
from test_21d_browser import no_sideways
from test_23_paired import MMLU, OPTS, Q4, S4, row_dir, world
from test_24_paired import subjects

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase24"
CATS = ("math", "law", "physics")
N, A_ONLY, B_ONLY = 200, 60, 30


def scores() -> None:
    """each build's score on the board, from its samples: the mean of its
    questions' means, its standard error"""
    import statistics

    from service import db, paired
    from test_21c_page import results
    for label in (S4, Q4):
        mid = row_dir(label)
        sid = next(o["row"] for o in reports.files()["paired"]["ours"] if o["label"] == label)
        base = sid.removesuffix(" · thinking")
        db.served_put({"id": base, "name": base.split("/")[1], "base_url": "", "key": "",
                       "how": "rented GPU", "based_on": "Qwen/Qwen3.6-35B-A3B",
                       "thinking": "on", "pin": {}, "by": "masein", "at": 0})
        for task in fb.TASKS:
            f = reports.samples_file(mid, task)
            if f is None:
                continue
            t = paired.table_of(f, task)
            per = [sum(x["ok"]) / len(x["ok"]) for x in t.values()]
            se = statistics.stdev(per) / len(per) ** 0.5 if len(per) > 1 else 0.0
            results(mid, base, task, sum(per) / len(per), se,
                    sum(len(x["ok"]) for x in t.values()), 0)


def big_mmlu() -> None:
    """200 questions in three categories: 60 step4000 alone gets right, 30 Q4
    alone, the rest the same; a third of Q4's wrong ones ran out of room —
    each reply saying what it chose, as the scorer reads it"""
    ids = [str(2000 + k) for k in range(N)]
    p = fb._cache(config.BENCH_ROOT, MMLU)
    p.write_text(json.dumps({"items": [{"id": q, "question": f"Invented MMLU-Pro question {q}?",
                                        "options": OPTS, "answer": "A", "category": CATS[k % 3]}
                                       for k, q in enumerate(ids)], "extra": {"shots": {}}}))

    def answer(q: str, ok: bool, out: bool) -> tuple[str, str | None]:
        if out:
            return ("<think>\nThe options, one by one. Option C reads well. "
                    + "Let me check option C against the question once more. " * 40, None)
        pick = "A" if ok else "C"
        return (f"<think>\nWeighing the options for question {q}. Option A fits the first part. "
                f"So the answer is ({pick}).\nLet me check that once more against the others."
                f"\n</think>\n\nThe best fit is ({pick}).\nANSWER: {pick}", pick)
    for label, flip in ((S4, False), (Q4, True)):
        lines = []
        for k, q in enumerate(ids):
            same = k % 2 == 0
            ra = k < A_ONLY or (k >= A_ONLY + B_ONLY and same)
            rb = A_ONLY <= k < A_ONLY + B_ONLY or (k >= A_ONLY + B_ONLY and same)
            ok = rb if flip else ra
            out = flip and not ok and k % 3 == 0
            text, read = answer(q, ok, out)
            lines.append(json.dumps({"doc_id": k, "doc": {"id": q}, "resps": [[text]],
                                     "filtered_resps": [read or ""],
                                     "frontier": [{"epoch": 0, "ok": ok, "ran_out": out,
                                                   "read": read, "tokens": 1200 if not out
                                                   else 32768}]}))
        d = sf.task_dir(row_dir(label), MMLU)
        for old in d.glob(f"samples_{MMLU}_*.jsonl"):
            old.unlink()
        (d / f"samples_{MMLU}_2026-10-21T08-00-00.000000.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import paired, report_stats
    world()
    subjects()
    big_mmlu()
    d = sf.task_dir(row_dir(S4), "hle_text_cais")
    d.mkdir(parents=True, exist_ok=True)
    (d / sf.ANSWERS).write_text("".join(json.dumps({"id": f"h{k}", "epoch": 0, "answer": "x",
                                                    "finish": "stop"}) + "\n"
                                        for k in range(2098)))
    paired._tables.clear()
    scores()
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def open_pair(page, live, hash_: str = "", width: int = 1280, scheme: str = "light") -> None:
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#tab=reports&report=paired&a={S4}&b={Q4}"
              + (("&" + hash_) if hash_ else ""))
    page.wait_for_selector("[data-pd-table]", timeout=30000)


def shot(page, loc, path: Path) -> None:
    """the element, laid out (an answer landing draws the reader again), the
    sticky header kept off it"""
    for _ in range(20):
        page.wait_for_timeout(100)
        b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + "
                         "scrollX, y: r.top + scrollY, width: r.width, height: r.height}; }")
        if b["width"] > 0 and b["height"] > 0:
            break
    page.screenshot(path=str(path), full_page=True, clip=b)


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_table_chart(live, page, width, scheme):
    open_pair(page, live, width=width, scheme=scheme)
    rows = page.eval_on_selector_all("[data-pd-row]", "rs => rs.map(r => r.dataset.pdRow)")
    assert rows == ["gpqa", "hle", "otis", "math", "mmlupro", "simpleqa", "arc",   # HLE in
                    "ifeval", "ifbench"]                                           # 25
    assert page.locator("[data-pd-wait='hle']").inner_text() == (
        f"waiting for grading — 2,098 answers from {S4}")
    assert page.locator("[data-pd-wait='math']").inner_text() == f"no answers from {S4}"
    # the old chart and the Differ column are gone
    assert page.locator("[data-report-section='pd-chart']").count() == 0
    assert page.locator("[data-pd-count]").count() == 0 and "Differ" not in page.locator(
        "[data-pd-table]").inner_text().split("\n")[0:8]
    # one scale for every row: the zero line at the same place in each, under the header's 0
    zeros = page.eval_on_selector_all("[data-pd-diff] .pdx-zero", "ls => ls.map(l => Math.round("
                                      "l.getBoundingClientRect().left))")
    assert len(zeros) == 5 and len(set(zeros)) == 1, zeros
    if width == 1280:
        z = page.locator("[data-pd-scale] span:not(.lo):not(.hi)").bounding_box()
        assert abs(z["x"] + z["width"] / 2 - zeros[0]) <= 2, (z, zeros)
    assert int(page.locator("[data-pd-scale]").get_attribute("data-pd-scale")) >= 10
    # a sure verdict's bar in the accent colour, a can't-tell one muted
    for k in ("gpqa", "otis", "mmlupro", "simpleqa", "arc"):
        verdict = page.locator(f"[data-pd-verdict='{k}']").inner_text()
        bar = page.locator(f"[data-pd-diff='{k}'] [data-pd-bar]").get_attribute("data-pd-bar")
        assert bar == ("tie" if verdict == "can't tell" else "sure"), (k, verdict, bar)
    assert page.locator("[data-pd-verdict='mmlupro']").inner_text() == f"{S4} higher"
    # where they differ: Q4's 30 on the left, step4000's 60 on the right, in proportion
    left = page.locator("[data-pd-part='mmlupro:b']")
    right = page.locator("[data-pd-part='mmlupro:a']")
    assert (left.inner_text(), right.inner_text()) == ("30 ▸", "60 ▸")
    assert right.get_attribute("title") == (f"See the 60 questions {S4} got right and {Q4} got "
                                            "wrong, with both answers")
    w = left.bounding_box()["width"] / right.bounding_box()["width"]
    assert 0.4 < w < 0.6, w
    assert page.locator("[data-pd-differ='mmlupro']").inner_text() == "on 45.0% of questions"
    assert page.locator("[data-pd-differ='gpqa']").inner_text().startswith(
        "a run of each differs ")
    if width == 400:                                # stacked: the difference under the name
        name = page.locator("[data-pd-row='mmlupro'] .pdx-name").bounding_box()
        diff = page.locator("[data-pd-diff='mmlupro']").bounding_box()
        assert diff["y"] >= name["y"] + name["height"] - 1
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    shot(page, page.locator("[data-report-section='pd-table']"),
         SCREENS / f"table-{width}-{scheme}.png")


def card_pos(page) -> str:
    return page.locator("[data-pd-pos]").inner_text()


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_reader_mmlu_pro(live, page, width, scheme):
    open_pair(page, live, width=width, scheme=scheme)
    page.locator("[data-pd-part='mmlupro:a']").click()
    rd = page.locator("[data-pd-reader='mmlupro:a']")
    rd.wait_for()
    assert rd.locator("[data-pd-reader-title]").inner_text() == (
        f"MMLU-Pro — 60 questions {S4} got right and {Q4} got wrong")
    page.wait_for_selector("[data-pd-list-head]")
    head = page.locator("[data-pd-list-head]").inner_text()
    shown = [str(2000 + k) for k in range(A_ONLY) if reports.shown_half(MMLU, str(2000 + k))]
    assert head == f"{len(shown)} shown here; {A_ONLY - len(shown)} in the hidden half, counted " \
                   "but never shown"
    assert rd.locator("[data-ro-warning]").inner_text().startswith("For insight only")
    page.wait_for_selector("[data-pd-question]")
    assert card_pos(page) == f"1 of {len(shown)}"
    assert page.locator("[data-pd-card]").get_attribute("data-pd-card") == shown[0]
    assert page.locator("[data-pd-question]").inner_text().startswith(
        f"Invented MMLU-Pro question {shown[0]}?")
    assert page.locator("[data-pd-question] li").count() == 10
    assert page.locator("[data-pd-right]").inner_text() == "The right answer: (A) invented option A"
    page.wait_for_selector(f"[data-pd-col-of='{Q4}'] [data-pd-ok]")
    assert page.locator(f"[data-pd-col-of='{S4}'] [data-pd-ok]").get_attribute("data-pd-ok") == "true"
    assert page.locator(f"[data-pd-col-of='{Q4}'] [data-pd-ok]").get_attribute("data-pd-ok") == "false"
    assert page.locator("[data-pd-thinking] summary").first.inner_text().startswith(
        "Show its thinking (")
    # Previous and Next, and the arrow keys
    page.locator("[data-pd-next]").click()
    page.wait_for_function(f"document.querySelector('[data-pd-pos]').textContent === '2 of {len(shown)}'")
    page.locator("body").press("ArrowRight")
    page.wait_for_function(f"document.querySelector('[data-pd-pos]').textContent === '3 of {len(shown)}'")
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(f"document.querySelector('[data-pd-pos]').textContent === '2 of {len(shown)}'")
    h = page.evaluate("decodeURIComponent(location.hash)")
    assert "bench=mmlupro&side=a&q=2" in h
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_selector(f"[data-pd-col-of='{Q4}'] [data-pd-reply]")
    shot(page, rd, SCREENS / f"card-mmlupro-{width}-{scheme}.png")
    # the link opens the same card
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + h.split("#", 1)[-1])
    page.wait_for_selector("[data-pd-card]", timeout=30000)
    assert page.locator("[data-pd-card]").get_attribute("data-pd-card") == shown[1]
    assert card_pos(page) == f"2 of {len(shown)}"
    # the filters change the count: a subject, then ran out only
    page.select_option("[data-pd-subject]", "law")
    page.wait_for_function("document.querySelector('[data-pd-pos]').textContent.endsWith(' shown')")
    law = [q for q in shown if CATS[(int(q) - 2000) % 3] == "law"]
    assert card_pos(page) == f"1 of {len(law)} shown"
    page.select_option("[data-pd-subject]", "")
    page.locator("[data-pd-ranout]").check()
    out = [q for q in shown if (int(q) - 2000) % 3 == 0]
    page.wait_for_function(f"document.querySelector('[data-pd-pos]').textContent === '1 of {len(out)} shown'")
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_reader_gpqa_and_its_runs(live, page, width, scheme):
    open_pair(page, live, width=width, scheme=scheme)
    page.locator("[data-pd-part='gpqa:a']").click()
    page.wait_for_selector("[data-pd-withheld]", timeout=30000)
    card = page.locator("[data-pd-card]")
    assert page.locator("[data-pd-withheld]").inner_text() == (
        "The question isn’t shown: the board keeps GPQA Diamond’s questions unlisted.")
    assert page.locator("[data-pd-subject-of]").inner_text() in ("Quantum Mechanics",
                                                                 "Organic Chemistry")
    assert "invented question" not in card.inner_text().lower()
    right = page.locator("[data-pd-right]").inner_text()
    assert right.startswith("The right answer: (") and right.endswith(")")   # the letter alone
    # four runs a build, each its mark; each column opens on the run that differs
    page.wait_for_selector(f"[data-pd-runs='{Q4}']")
    assert page.locator(f"[data-pd-runs='{S4}'] button").count() == 4
    on_a = page.locator(f"[data-pd-runs='{S4}'] button[aria-pressed='true']").inner_text()
    on_b = page.locator(f"[data-pd-runs='{Q4}'] button[aria-pressed='true']").inner_text()
    assert (on_a, on_b) == ("✓", "✗")
    # another run, clicked
    page.locator(f"[data-pd-runs='{S4}'] button[aria-pressed='false']").first.click()
    page.wait_for_function(f"document.querySelectorAll(\"[data-pd-runs='{S4}'] "
                           "button[aria-pressed='true']\").length === 1")
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.wait_for_selector(f"[data-pd-col-of='{Q4}'] [data-pd-reply]")
    shot(page, page.locator("[data-pd-reader]"), SCREENS / f"card-gpqa-{width}-{scheme}.png")


def test_a_hidden_half_question_cant_be_reached_by_the_link(live, page):
    hidden = [str(2000 + k) for k in range(A_ONLY) if not reports.shown_half(MMLU, str(2000 + k))]
    open_pair(page, live, "bench=mmlupro&side=a&q=999")
    page.wait_for_selector("[data-pd-card]", timeout=30000)
    assert page.locator("[data-pd-card]").get_attribute("data-pd-card") not in hidden
    for _ in range(3):
        page.locator("[data-pd-next]").click()
        assert page.locator("[data-pd-card]").get_attribute("data-pd-card") not in hidden
    r = page.request.get(live["base"] + "/api/reports/paired/answer", params={
        "model": S4, "task": MMLU, "id": hidden[0], "epoch": 0})
    assert r.status == 403


def test_the_small_fixes(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_function("typeof trTicks === 'function'")
    # round ticks: 62.4 … 85.1 reads 60, 70, 80, 90
    assert page.evaluate("trTicks(0.624, 0.851)") == [60, 70, 80, 90]
    assert page.evaluate("trTicks(0.97, 0.992)") == [97, 98, 99, 100]
    # Q4 and BF16 0.3 points apart: their labels one above the other, never on top
    boxes = page.evaluate("""() => {
      const T = { checkpoints: [{ label: 's1', step: 1000 }, { label: 's2', step: 4000 }],
        references: ['Q4 (x)', 'BF16'], segments: { k: [] }, label: 'Training steps', paired: null };
      const byL = { s1: { cells: { k: { v: .62, se: .02 } } }, s2: { cells: { k: { v: .70, se: .02 } } },
        'Q4 (x)': { cells: { k: { v: .851 } } }, BF16: { cells: { k: { v: .848 } } } };
      const p = trPanel({ ceiling: {} }, T, { key: 'k', label: 'K' }, byL);
      p.style.width = '280px';                    // a panel's width on the page
      document.body.append(p);
      const out = [...p.querySelectorAll('[data-tr-ref-label]')].map(t => {
        const r = t.getBoundingClientRect(); return [r.top, r.bottom, t.textContent]; });
      const w = p.querySelector('svg').getBoundingClientRect().width;
      p.remove();
      return { out, w };
    }""")
    (t1, b1, _), (t2, b2, _) = sorted(boxes["out"])
    assert b1 <= t2 + 0.5, boxes                                 # no overlap
    assert boxes["w"] >= 260                                     # 12-unit text at least 12 px
    # an all-empty column: gone, and its words with it
    got = page.evaluate("""() => {
      const d = { picks: ['x'], ours: [{ label: 'x', colour: 0 }], speed: {},
        beyond: { x: { gpqa: { reliability: { n: 3, always_share: .5, never_share: .2,
          sometimes_share: .3, runs: 4 } } } } };
      const s = rpBeyond(d);
      return { cols: [...s.querySelectorAll('th')].map(t => t.textContent),
               how: s.querySelector('[data-by-how]').textContent };
    }""")
    assert "Output tok/s alone" not in got["cols"] and "GPQA Diamond energy an answer" not in got[
        "cols"]
    assert got["cols"] == ["Build", "GPQA always right"]
    assert "how fast its card" not in got["how"] and "energy" not in got["how"]
    assert page.errors == []
