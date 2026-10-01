"""12x: DeviceMark's rows as columns on Models.

Every model in the asked-for view has a DeviceMark row, and the Benchmarks
picker had no DeviceMark columns: filtering to them left an empty table.

- A "DeviceMark" chip, and a "DeviceMark protocol" group in Benchmarks ▾:
  the composite with half its 95% interval, IFEval, MMLU-Pro and MATH (each
  with its interval), answered %, median tokens. Each model's newest row, as
  the On-device chart has it; thinking on a row of its own, "· thinking".
- A served setup and a Hugging Face model both fill them: the setup's
  thinking row, which the board has no other row for, is made beside it.
- Never in an average: chosen alone, there is no Avg column.

Fixtures only: the rows are written as runs write them."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from fake_openai import FakeServer
from service import config
from test_12q_devicemark_board import HF, SERVED, _row

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12x"
QWEN = "Qwen/Qwen3.5-2B"
SETUP = "served/DM-k4-LDA-phone-build-MTP"
OFF, ON = {"ifeval": 0.7, "mmlu_pro": 0.65, "math": 0.6}, {"ifeval": 0.5, "mmlu_pro": 0.7, "math": 0.8}
Q_OFF, Q_ON = {"ifeval": 0.6, "mmlu_pro": 0.4, "math": 0.3}, {"ifeval": 0.55, "mmlu_pro": 0.5, "math": 0.7}
DM_COLS = ["DeviceMark", "IFEval (DM)", "MMLU-Pro (DM)", "MATH (DM)", "Answered", "Tokens"]


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def ids(*ms):
    return ",".join(quote(m, safe="") for m in ms)


def _dm_results(model: str, thinking: bool) -> None:
    """what lm_eval leaves of a Hugging Face model's DeviceMark run: the row on
    Models is made from it, the scores from devicemark.json"""
    d = config.OUT_DIR / (model.replace("/", "__") + ("__thinking" if thinking else ""))
    out = d / "dm_ifeval_0shot" / ("pretrained__" + model.replace("/", "__"))
    out.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps(
        {"model": model + (" · thinking" if thinking else ""), "kind": "instruct",
         "params": 2_000_000_000, **({"base_model": model} if thinking else {})}))
    (out / "results_2026-09-30T10-00-00.000000.json").write_text(json.dumps({
        "results": {"dm_ifeval": {"alias": "dm_ifeval", "bypass,none": 999}},
        "config": {"model": "hf", "model_args": f"pretrained={model},dtype=bfloat16"},
        "chat_template": True, "date": 3.0}))


@pytest.fixture(scope="module")
def rows(live):
    """a served phone build with MTP, thinking off and on; and our hf run of
    Qwen3.5-2B, thinking off and on"""
    import service.app as appmod
    fake = FakeServer()
    got = api(live, "/api/served", {"name": "DM k4-LDA phone build, MTP", "base_url": fake.base,
                                    "thinking": "off", "based_on": "Qwen/Qwen3.6-35B-A3B",
                                    "how": "llama.cpp fork, k=4 + LDA, MTP 3", "phone": True,
                                    "by": "masein"})
    assert (got.get("model") or got)["id"] == SETUP
    _row(config.OUT_DIR, SETUP, OFF, {**SERVED, "phone": True, "name": "DM k4-LDA phone build, MTP"})
    _row(config.OUT_DIR, SETUP, ON, {**SERVED, "phone": True, "name": "DM k4-LDA phone build, MTP"},
         thinking=True)
    for th, acc in ((False, Q_OFF), (True, Q_ON)):
        _dm_results(QWEN, th)
        _row(config.OUT_DIR, QWEN, acc, HF, thinking=th)
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    fake.close()
    for model in (SETUP, QWEN):
        for suffix in ("", "__thinking"):
            d = config.OUT_DIR / (model.replace("/", "__") + suffix)
            for f in d.glob("devicemark*"):
                f.unlink()
    appmod._cache.update(key=None, payload=None, at=0.0)


def go(page, live, hash_, sel, width=1400):
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#" + hash_)
    page.wait_for_selector(sel)


def headers(page) -> list[str]:
    return page.locator("[data-lb-table] thead tr:last-child th").evaluate_all(
        "ths => ths.map(t => (t.querySelector('.hname') || t).textContent.replace(/[▼▲]/g, '').trim())")


def cell(page, model, f) -> str:
    return page.locator(f"tr[data-lb-row='{model}'] [data-dm-cell='{f}']").inner_text().strip()


def test_the_chip_is_offered_only_while_a_model_has_a_row(live, page):
    # first in this file: before its rows are written there is no DeviceMark row, and no chip
    go(page, live, "tab=models", "[data-lb-table]")
    assert page.evaluate("dmAny()") is False
    assert page.locator("[data-chip='devicemark']").count() == 0
    assert page.locator("[data-chip='ondevice']").count() == 1


def test_the_chip_shows_each_models_rows_both_modes_sorted_by_the_composite(live, page, rows):
    go(page, live, "tab=models&chip=devicemark", "[data-lb-table]")
    assert page.locator("[data-chip='devicemark'][aria-pressed='true']").count() == 1
    assert headers(page)[-6:] == DM_COLS and "Avg" not in headers(page)
    shown = page.locator("tr[data-lb-row]").evaluate_all("rs => rs.map(r => r.dataset.lbRow)")
    want = {SETUP, SETUP + " · thinking", QWEN, QWEN + " · thinking"}
    assert want <= set(shown)
    # sorted by the composite, highest first
    comp = {m: page.evaluate(f"dmVal({json.dumps(m)}, 'composite')") for m in want}
    order = [m for m in shown if m in want]
    assert order == sorted(want, key=lambda m: -comp[m])
    # the numbers are the row's: the composite and each bench with half its interval
    off = page.evaluate(f"dmRowById({json.dumps(SETUP)})")
    half = 50 * (off["composite"]["ci"][1] - off["composite"]["ci"][0])
    assert cell(page, SETUP, "composite").startswith(f"{100 * off['composite']['value']:.1f}")
    assert cell(page, SETUP, "ifeval").startswith(f"{100 * OFF['ifeval']:.1f}")
    assert cell(page, SETUP + " · thinking", "math").startswith(f"{100 * ON['math']:.1f}")
    assert cell(page, QWEN, "mmlu_pro").startswith(f"{100 * Q_OFF['mmlu_pro']:.1f}")
    assert cell(page, SETUP, "tokens") == f"{round(off['median_tokens']):,}"
    assert cell(page, SETUP, "answered") == f"{100 * off['answered_pct']:.1f}%"          # 12z C7
    tip = json.loads(page.locator(f"tr[data-lb-row='{SETUP}'] [data-dm-cell='composite']")
                     .get_attribute("data-tip"))
    assert tip[0] == f"{100 * off['composite']['value']:.1f} ± {half:.1f}"
    assert tip[-1].startswith("the ± is half the 95% interval · ")
    assert tip[-1].endswith(" among ranked rows (cloud lines aren’t ranked)")
    # the served setup's thinking row is made beside it, badged, and opens the setup's page
    row = page.locator(f"tr[data-lb-row='{SETUP} · thinking']")
    assert row.locator("[data-thinking-badge]").count() == 1
    # and tagged as its setup is: a phone build
    base = page.locator(f"tr[data-lb-row='{SETUP}']")
    assert base.locator("[data-phone-tag]").inner_text() == "phone build"
    assert row.locator("[data-phone-tag]").inner_text() == "phone build"
    assert row.locator("a").first.get_attribute("href") == "#model=" + quote(SETUP, safe="")
    # the best in a column is found among the made rows too, and the line says the sort
    best = page.locator("[data-dm-cell='composite'][data-lead]").evaluate_all(
        "tds => tds.map(t => t.closest('tr').dataset.lbRow)")
    assert best[0] == max(want, key=lambda m: comp[m])
    assert "sorted by DeviceMark composite ▼" in page.locator("[data-lb-card]").inner_text()
    # never bold for answered or tokens: they are not scores
    assert page.locator("[data-dm-cell='tokens'][data-lead], [data-dm-cell='answered'][data-lead]"
                        ).count() == 0
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=SCREENS / "chip.png")
    assert page.errors == []


def test_the_picker_has_the_group_and_chosen_models_bring_their_thinking_rows(live, page, rows):
    cols = ",".join(f"dm:{f}" for f in ("composite", "ifeval", "mmlu_pro", "math", "answered",
                                        "tokens"))
    go(page, live, f"tab=models&cols={cols}&models={ids(SETUP, QWEN)}", "[data-lb-table]")
    shown = page.locator("tr[data-lb-row]").evaluate_all("rs => rs.map(r => r.dataset.lbRow)")
    assert set(shown) == {SETUP, SETUP + " · thinking", QWEN, QWEN + " · thinking"}
    # DeviceMark's alone: no Avg column, and nothing averaged
    assert not any(h.startswith("Avg") for h in headers(page))
    assert headers(page)[-6:] == DM_COLS
    page.locator("#pill-benchmarks").click()
    group = page.locator("[data-bench-group='devicemark']")
    group.wait_for()
    assert group.locator("[data-bench-row]").evaluate_all("rs => rs.map(r => r.dataset.benchRow)") \
        == [f"dm:{f}" for f in ("composite", "ifeval", "mmlu_pro", "math", "answered", "tokens")]
    assert group.locator("[data-bench]:checked").count() == 6
    assert "MMLU-Pro (DeviceMark protocol)" in group.inner_text()
    # one unticked goes from the table, the rest stay
    group.locator("[data-bench='dm:tokens']").uncheck()
    page.wait_for_function("!lbS().cols.includes('dm:tokens')")
    assert "Tokens" not in headers(page)
    # with an lm_eval column beside them, the Avg is that one's alone
    go(page, live, f"tab=models&cols=hellaswag,dm:composite&models={ids(QWEN)},fx%2Fgood-750m",
       "[data-lb-table]")
    assert page.evaluate("lbColumns(DATA.models).find(c => c.key === 'cavg').ts") == ["hellaswag"]
    assert page.errors == []


def test_with_another_columns_chosen_the_empty_table_offers_devicemarks(live, page, rows):
    go(page, live, f"tab=models&cols=ifeval&models={ids(SETUP, QWEN)}", "[data-lb-table]")
    empty = page.locator("[data-none-has]")
    empty.wait_for()
    assert empty.locator("[data-other-set='devicemark']").inner_text() == \
        "Show their DeviceMark columns"
    assert empty.locator("[data-other-set='gguf']").count() == 0        # no GGUF file here
    empty.locator("[data-other-set='devicemark']").click()
    page.wait_for_selector(f"tr[data-lb-row='{SETUP}'] [data-dm-cell='composite']")
    assert page.evaluate("lbS().cols") == [f"dm:{f}" for f in ("composite", "ifeval", "mmlu_pro",
                                                               "math", "answered", "tokens")]
    assert page.errors == []


def test_at_400px(live, page, rows):
    go(page, live, f"tab=models&chip=devicemark&models={ids(SETUP, QWEN)}", "[data-lb-table]",
       width=400)
    assert page.locator(f"tr[data-lb-row='{SETUP} · thinking']").count() == 1
    # the chosen two are tested, here: not "2 chosen · 0 tested"
    line = page.locator("[data-lb-card]").inner_text()
    assert "tested" not in line
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=SCREENS / "chip-400.png")
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_devicemark_view_names_rows_whole_and_drops_standards_notions(live, page, rows, width):
    """12z C7: names cut to "Qwen3.6-35B-…" couldn't tell the original from the
    phone build; Params "—" for a served setup; "0 ranked" and "prelim 0/7"
    are Standard's; answered is a share"""
    go(page, live, "tab=models&chip=devicemark", "[data-lb-table]", width=width)
    assert "dmview" in page.locator("[data-lb-table]").get_attribute("class").split()
    # the whole name, in two lines at most, nothing cut, at 1400; a phone wraps it already
    if width == 1400:
        name = page.locator(f"tr[data-lb-row='{SETUP} · thinking'] .mname")
        assert name.inner_text() == "DM k4-LDA phone build, MTP · thinking"
        fit = name.evaluate("e => [e.scrollHeight, e.clientHeight, e.scrollWidth, e.clientWidth, "
                            "e.getBoundingClientRect().width, e.closest('td').getBoundingClientRect().width]")
        assert fit[0] <= fit[1] + 1 and fit[2] <= fit[3] + 1, fit
    # Params from what it is based on: Qwen3.6-35B-A3B
    params = page.locator(f"tr[data-lb-row='{SETUP}'] td[data-params-from]")
    assert params.get_attribute("data-params-from") == "Qwen/Qwen3.6-35B-A3B"
    assert params.inner_text().startswith("35B")
    # none of Standard's: no prelim badge, no count of ranked models
    assert page.locator("[data-lb-table] .badge.prelim").count() == 0
    assert " ranked" not in page.locator("[data-statusline='models']").inner_text()
    # answered, with its % sign
    assert cell(page, QWEN, "answered").endswith("%")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    SCREENS.mkdir(parents=True, exist_ok=True)
    page.locator("[data-lb-card]").screenshot(path=SCREENS / f"dmview-{width}.png")
    assert page.errors == []
