"""23d in a real browser, at 1280 and 400 px, light and dark: the
scorecard's "How it thinks" inside Thinking tokens — its pills change the
table and the bars, its sentence for the first pick against Q4 — and the
model page's part in Beyond the score; nothing scrolls sideways. The world
is 21c's scorecard (its MMLU-Pro answers), with invented GPQA Diamond
questions and answers for step4000 and Q4, worked out by the stats worker."""

from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import quote

import pytest

import frontier as fb
from service import config, ran_out, reports
from service import frontier as sf
from test_21c_page import world as scorecard
from test_21d_browser import no_sideways

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase23d"
GPQA = "gpqa_diamond_epoch"
S4, Q4 = "step4000", "Q4 (UD-Q4_K_XL)"
ROWS = {S4: "served/step4000-lda-2610 · thinking",
        Q4: "served/Qwen3.6-35B-A3B-Q4-lda-2610 · thinking"}
ITEMS = [{"id": f"g{k}", "question": f"Invented question {k}?", "right": f"right answer {k}",
          "wrong": [f"wrong answer {k}{c}" for c in "abc"]} for k in range(30)]


def answers(where: float, right: int, lost: int, rng: random.Random) -> list[tuple]:
    """30 questions: `right` right, stating it about `where` into the
    thinking; the rest wrong, `lost` of them having stated it first; the last
    two ran out, one with it stated"""
    out = []
    for k, it in enumerate(ITEMS):
        letter = ran_out.right_of(GPQA, it)["letter"]
        other = next(c for c in "ABCD" if c != letter)
        say = f"So the answer is ({letter})."
        n = rng.randint(4000, 9000)
        pos = max(0.02, min(0.95, rng.gauss(where, 0.08)))
        body = ("Let me work through it. " * 40)[:int(1200 * pos)]
        rest = ("Wait, let me check that again. " * 60)[:int(1200 * (1 - pos))]
        if k >= 28:
            text = "<think>" + (say if k == 28 else "") + " looping. " * 300
            out.append((it["id"], True, False, text, 32768))
        elif k < right:
            out.append((it["id"], False, True, f"<think>{body}{say}{rest}</think>\n\nANSWER: "
                        f"{letter}", n))
        else:
            had = k < right + lost
            out.append((it["id"], False, False, f"<think>{body}{say if had else ''}"
                        f"{rest}</think>\n\nANSWER: {other}", n))
    return out


def gpqa(label: str, runs: list[tuple]) -> None:
    d = sf.task_dir(reports.row_dir(ROWS[label]), GPQA)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"samples_{GPQA}_*.jsonl"):
        old.unlink()
    (d / f"samples_{GPQA}_2026-10-21T08-00-00.000000.jsonl").write_text("".join(
        json.dumps({"doc_id": k, "doc": {"id": q}, "resps": [[t]], "filtered_resps": [""],
                    "frontier": [{"epoch": 0, "ok": ok, "ran_out": out, "tokens": n}]}) + "\n"
        for k, (q, out, ok, t, n) in enumerate(runs)))


@pytest.fixture(scope="module", autouse=True)
def board(live):
    import service.app as appmod
    from service import report_stats
    scorecard()
    p = fb._cache(config.BENCH_ROOT, GPQA)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"items": ITEMS, "extra": {}}))
    rng = random.Random(23)
    gpqa(S4, answers(0.41, 18, 4, rng))
    gpqa(Q4, answers(0.33, 21, 1, rng))
    report_stats.work(say=lambda s: None)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    appmod._cache.update(key=None, payload=None, at=0.0)


def table(page) -> list[list[str]]:
    return page.eval_on_selector_all("[data-th-table] tr", "rs => rs.map(r => [...r.children]"
                                     ".map(c => c.innerText.trim()))")


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_scorecards_how_it_thinks(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=step-checkpoints")
    part = page.locator("[data-th-part]")
    part.wait_for(timeout=30000)
    page.wait_for_selector("[data-th-table='gpqa']", timeout=30000)
    pills = page.eval_on_selector_all("[data-th-bench]", "bs => bs.map(b => b.dataset.thBench)")
    assert pills == ["gpqa", "mmlupro", "math", "otis"]
    said = page.locator("[data-th-sentence]").inner_text()
    assert said.startswith("On GPQA Diamond, step4000 first states the right answer ") and \
        "; Q4: " in said and "Of its wrong answers, 40% had stated the right answer first." in said
    rows = table(page)
    assert [r[0] for r in rows[1:]] == [S4, Q4]
    # 18 of 18 right answers state it; 4 of its 10 wrong answers had it first; one of the two
    # that ran out had it
    s4 = rows[1]
    # 23e: the two that open their answers end in ▸
    assert s4[1] == "100%" and s4[5] == "40% ▸" and s4[6] == "50% ▸"
    assert page.locator("[data-th-bar]").count() == 2
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=str(SCREENS / f"report-thinks-gpqa-{width}-{scheme}.png"))
    if (width, scheme) == (1280, "light"):
        (SCREENS / "table-gpqa.txt").write_text("\n".join(" | ".join(r) for r in rows) + "\n")
    # the pills change the table and the bars: MMLU-Pro, 21c's answers — step4000's all ran
    # out stating it, Q4's all right without stating it in the thinking
    page.locator("[data-th-bench='mmlupro']").click()
    page.wait_for_selector("[data-th-table='mmlupro']")
    m = table(page)
    assert [r[0] for r in m[1:]] == [S4, Q4]
    assert m[1][6] == "100% ▸" and m[2][1] == "0%"
    assert page.locator("[data-th-bar]").count() == 0          # no right answer states it
    assert no_sideways(page) and page.errors == []
    part.screenshot(path=str(SCREENS / f"report-thinks-mmlupro-{width}-{scheme}.png"))


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (1280, "dark"), (400, "light"),
                                          (400, "dark")])
def test_the_model_pages_part(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + quote(ROWS[S4], safe=""))
    blk = page.locator("[data-kind-block='beyond']")
    blk.wait_for(timeout=30000)
    blk.locator("summary").click()
    part = page.locator("[data-by-part='thinks']")
    part.wait_for(timeout=30000)
    assert part.locator("[data-th-sentence]").inner_text().startswith(
        "On GPQA Diamond, step4000 first states the right answer ")
    assert page.eval_on_selector_all("[data-th-model] tbody tr",
                                     "rs => rs.map(r => r.dataset.thRow)") == [GPQA, "mmlupro_tiger"]
    assert part.locator("[data-th-bar]").count() == 1          # GPQA's; MMLU-Pro states none
    assert no_sideways(page) and page.errors == []
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=str(SCREENS / f"model-thinks-{width}-{scheme}.png"))
