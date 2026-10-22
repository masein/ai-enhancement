"""27: one table for every Models tab — Latest, Standard, Mobile (DeviceMark's
table too), Everyday and Frontier drawn through modelTable, one row design and
one cell design. The world is 27's values world (tests/test_27_values_browser):
the fixture board, 21c's scorecard with our rows named as on the board, the
On-device cards, MobileAIBench, Mobile-MMLU-Pro and DeviceMark; and the Q4
baseline sat on Everyday and DeviceMark too, so its name can be read on every
tab. Every model, question and number is invented."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from service import config, reports
from test_21c_page import SPEC
from test_21d_browser import no_sideways
from test_27_values_browser import VIEWS, world27

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase27"
Q4 = SPEC["Q4 (UD-Q4_K_XL)"]["row"]                  # served/…-Q4-lda-2610 · thinking
TABS = {"latest": "view=latest", "standard": "view=standard&group=all",
        "mobile": "view=mobile&group=devicemark", "everyday": "view=everyday",
        "frontier": "view=frontier"}
# the Models views and DeviceMark's table: none of them builds a table of its own
RENDERERS = ["lbLatest", "vLeaderboard", "lbEveryday", "lbFrontier", "lbOnDevice", "dmSpec",
             "mmfCard"]


@pytest.fixture(scope="module", autouse=True)
def board(live):
    from service import db
    from test_12q_devicemark_board import SERVED, _row
    mp = pytest.MonkeyPatch()
    world27(live, mp)
    # the Q4 baseline on Everyday (good-750m's marks) and DeviceMark too
    tree = live["tree"]
    d = reports.row_dir(Q4)
    marks = json.loads((Path(tree["models"]["fx/good-750m"]["dir"]) / "everyday.json").read_text())
    (d / "everyday.json").write_text(json.dumps({**marks, "model": Q4}))
    _row(config.OUT_DIR, Q4.removesuffix(" · thinking"), {"ifeval": 0.8, "mmlu_pro": 0.7,
         "math": 0.6}, {**SERVED, "name": "Qwen3.6-35B-A3B Q4 (lda runtime)"}, thinking=True)
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    db.ai_set("labeller:third", None, "masein")
    mp.undo()


def go(page, live, hash_, width=1280, scheme="dark", thinking="default", tint=False):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(live["base"] + "/")
    page.evaluate(f"""() => {{ localStorage.setItem('bench-mt-thinking', '{thinking}');
      localStorage.setItem('bench-lb-tint', '{'on' if tint else 'off'}'); }}""")
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&" + hash_)
    page.wait_for_selector("[data-lb-card] [data-mt]", timeout=30000)
    page.wait_for_timeout(200)


def shot(page, loc, path: Path) -> None:
    SCREENS.mkdir(parents=True, exist_ok=True)
    b = None
    for _ in range(20):
        page.wait_for_timeout(100)
        b = loc.evaluate("e => { const r = e.getBoundingClientRect(); return {x: r.left + "
                         "scrollX, y: r.top + scrollY, width: r.width, height: r.height}; }")
        if b["width"] > 0 and b["height"] > 0:
            break
    page.screenshot(path=str(path), full_page=True, clip=b)


def body_of(js: str, name: str) -> str:
    """a top-level function's source, to its closing brace"""
    m = re.search(rf"^(?:async )?function {name}\(", js, re.M)
    assert m, name
    end = re.search(r"^}\n", js[m.start():], re.M)
    return js[m.start():m.start() + end.end()]


def test_one_renderer():
    import report_lm_eval as rep
    for name in RENDERERS:
        assert "el('table'" not in body_of(rep.JS, name), f"{name} builds its own table"
    assert len(re.findall(r"^function modelTable\(", rep.JS, re.M)) == 1


def test_every_models_table_is_the_one(live, page):
    for v in VIEWS:
        go(page, live, v, width=1400)
        tables = page.eval_on_selector_all("[data-lb-card] table",
                                           "ts => ts.map(t => t.dataset.modelTable || '')")
        assert tables and all(tables), (v, tables)
        assert page.errors == [], v


def test_the_same_name_everywhere(live, page):
    want = "Qwen3.6-35B-A3B · Q4"
    go(page, live, TABS["latest"])
    assert page.locator("[data-latest-row='Q4 (UD-Q4_K_XL)'] [data-mt-name]").inner_text() == want
    for tab in ("mobile", "everyday", "frontier"):
        go(page, live, TABS[tab])
        row = page.locator(f"tr[data-lb-row='{Q4}']:not(.mt-sub-row)")
        assert row.locator("[data-mt-name]").inner_text() == want, tab
    page.goto(live["base"] + "/#model=" + Q4.replace("/", "%2F").replace(" ", "%20"))
    page.wait_for_selector(".mtitle")
    assert page.locator(".mtitle").first.inner_text() == want
    # the older run of the same quant keeps its tag; two rows never share a name
    go(page, live, TABS["frontier"])
    names = page.eval_on_selector_all("tbody tr.mt-row:not(.mt-sub-row) [data-mt-name]",
                                      "ns => ns.map(n => n.textContent)")
    assert "Qwen3.6-35B-A3B · Q4 (original, k=8)" in names
    assert len(names) == len(set(names))


def test_display_names_follow_the_rules(live, page):
    go(page, live, TABS["latest"])
    got = page.evaluate("""() => [
      ['Qwen3.6-35B-A3B Q4 (lda runtime)', {}], ['Qwen3.6-35B-A3B Q4 (original, k=8)', {}],
      ['Qwen3.6-35B-A3B BF16 (unsloth) · thinking', {}], ['step8804 (lda runtime) · thinking', {}],
      ['Qwen3.6 k4-LDA · lookahead + MTP', { basedOn: 'Qwen/Qwen3.6-35B-A3B' }],
      ['Qwen3.6-35B-A3B k4-LDA (phone build)', {}],
      ['Ornith-1.5-35B-A3B Q4_K_M (card settings)', {}],
      ['qwen35-delta-moe-7d560104-step945-v2', {}],
      ['o4-mini (high)', { published: true }], ['Edge0-35B-A3B-preview', { published: true }],
      ['Bonsai 27B (1-bit)', { published: true }], ['Qwen3-30B-A3B Thinking', { published: true }],
      ['Ornith-1.5-35B-A3B, its card', { published: true }],
      ['Claude 3.7 Sonnet (64K thinking)', { published: true }]].map(([n, o]) => {
        const p = nameParts(n, { measured: !o.published, basedOn: o.basedOn });
        return [mtNameOf(p), p.line.join(', '), p.thinking || ''];
      })""")
    assert got == [
        ["Qwen3.6-35B-A3B · Q4", "lda runtime", ""],
        ["Qwen3.6-35B-A3B · Q4 (original, k=8)", "", ""],
        ["Qwen3.6-35B-A3B · BF16 (unsloth)", "", "on"],
        ["step8804", "lda runtime", "on"],
        ["Qwen3.6-35B-A3B k4-LDA · lookahead + MTP", "", ""],
        ["Qwen3.6-35B-A3B k4-LDA", "phone build", ""],
        ["Ornith-1.5-35B-A3B · Q4_K_M", "on its card’s settings", ""],
        ["qwen35-delta-moe-7d560104-step945-v2", "", ""],
        ["o4-mini · high", "", ""],
        ["Edge0-35B-A3B-preview", "", ""],
        ["Bonsai 27B", "1-bit", ""],
        ["Qwen3-30B-A3B", "", "on"],
        ["Ornith-1.5-35B-A3B", "", ""],
        ["Claude 3.7 Sonnet · 64K thinking", "", ""]]


def test_frontier_has_one_column_a_benchmark(live, page):
    go(page, live, TABS["frontier"])
    page.locator("[data-mt-columns]").click()
    page.locator("[data-show-all]").click()
    page.wait_for_selector("[data-frontier-table]")
    heads = page.eval_on_selector_all("[data-frontier-table] thead tr.mt-hrow th[data-fr-col]",
                                      "ts => ts.map(t => t.dataset.frCol)")
    assert heads.count("hle") == 1 and heads.count("mmlu-pro") == 1
    assert not any("humanity" in h or h == "mmlu pro" for h in heads)
    # the header's first line is never cut
    firsts = page.eval_on_selector_all("[data-frontier-table] thead tr.mt-hrow th[data-fr-col] .mt-h1",
                                       "ts => ts.map(t => t.scrollWidth <= t.clientWidth + 1)")
    assert all(firsts)
    row = page.locator("tr[data-fr-row='reported/google/gemma-4-e4b']")
    cell = row.locator("td[data-mt-col='mmlu-pro']")
    assert cell.locator(".mt-n").evaluate("n => n.firstChild.textContent") == "69.4"
    assert cell.locator("[data-mt-src]").inner_text() == "G"
    # the company is in its line; Ours first
    assert "Google · open" in row.locator("[data-mt-line]").inner_text()
    groups = page.eval_on_selector_all("[data-frontier-table] [data-mt-group-head]",
                                       "gs => gs.map(g => g.dataset.mtGroupHead)")
    assert groups[0] == "ours"


def test_no_blanks(live, page):
    for tab, v in TABS.items():
        go(page, live, v)
        blank = page.eval_on_selector_all(
            "[data-lb-card] table[data-model-table] td[data-mt-col]",
            "ts => ts.filter(t => !t.textContent.trim() && !t.querySelector('input, button'))"
            ".map(t => t.dataset.mtCol)")
        assert blank == [], (tab, blank[:5])
        dashes = page.eval_on_selector_all("[data-lb-card] td.mt-none", "ts => ts.map(t => t.textContent)")
        assert all(d == "–" or d.startswith("– ") for d in dashes), tab


def test_everyday_shows_the_share_big_and_the_count_under(live, page):
    go(page, live, TABS["everyday"])
    assert page.evaluate("JSON.stringify(evdCell(22, 27))") == json.dumps(
        {"v": 22 / 27, "text": "81", "sub": "22 of 27"}, separators=(",", ":"))
    for td in page.locator("[data-lb-everyday] td[data-mt-col^='g:'], [data-lb-everyday] "
                           "td[data-mt-col='total']").all():
        sub = td.locator("[data-mt-sub]").inner_text()
        a, b = map(int, re.match(r"(\d+) of (\d+)", sub).groups())
        assert td.locator(".mt-n").inner_text() == str(round(100 * a / b))
    # the highest Total first, in each group
    for g in ("ours", "open"):
        tot = page.eval_on_selector_all(
            f"[data-lb-everyday] tr.mt-row[data-mt-group='{g}']:not(.mt-sub-row) td[data-mt-col='total']",
            "ts => ts.map(t => Number(t.dataset.mtV))")
        assert tot == sorted(tot, reverse=True), g


@pytest.mark.parametrize("tab", list(TABS))
def test_tint_on_tints_every_tab_and_off_gives_bars(live, page, tab):
    go(page, live, TABS[tab], tint=True)
    table = page.locator("[data-lb-card] table[data-model-table]").first
    assert table.locator("td[data-mt-heat]").count() > 0
    assert table.locator(".mt-bar").count() == 0
    go(page, live, TABS[tab], tint=False)
    table = page.locator("[data-lb-card] table[data-model-table]").first
    assert table.locator("td[data-mt-heat]").count() == 0
    assert table.locator(".mt-bar").count() > 0


def test_the_thinking_filter(live, page):
    base = "Qwen/Qwen3.5-2B"
    go(page, live, TABS["mobile"], thinking="both")
    row = page.locator(f"tr[data-lb-row='{base}']")
    sub = page.locator(f"tr.mt-sub-row[data-lb-row='{base} · thinking']")
    assert row.count() == 1 and sub.count() == 1
    assert sub.locator("[data-mt-name]").inner_text() == "thinking on"
    go(page, live, TABS["mobile"], thinking="off")
    assert page.locator(f"tr[data-lb-row='{base} · thinking']").count() == 0
    comp = page.locator(f"tr[data-lb-row='{base}'] td[data-mt-col='dm:composite'] .mt-n")
    assert comp.inner_text() == "43.3"                          # its thinking-off number
    go(page, live, TABS["mobile"], thinking="on")
    assert page.locator(f"tr[data-lb-row='{base}']").count() == 0
    comp = page.locator(f"tr[data-lb-row='{base} · thinking'] td[data-mt-col='dm:composite'] .mt-n")
    assert comp.inner_text() == "57.8"


def test_a_row_with_no_score_shown_is_hidden_and_counted(live, page):
    # IFEval alone: most of Frontier's models have no number in it
    page.goto(live["base"] + "/")
    page.evaluate("() => localStorage.setItem('bench-mt-cols', JSON.stringify({ frontier: ['ifeval'] }))")
    go(page, live, TABS["frontier"])
    line = page.locator("[data-mt-empty]")
    n = int(line.get_attribute("data-mt-empty"))
    assert n > 0 and line.inner_text() == f"{n} more models have no score in these columns · show them"
    shown = page.locator("[data-frontier-table] tbody tr.mt-row").count()
    page.locator("[data-mt-empty-toggle]").click()
    page.wait_for_function(f"document.querySelectorAll('[data-frontier-table] tbody tr.mt-row')"
                           f".length === {shown + n}")


@pytest.mark.parametrize("width,scheme", [(1280, "dark"), (1280, "light"), (400, "dark"),
                                          (400, "light")])
def test_every_tab_in_the_browser(live, page, width, scheme):
    for tab, v in TABS.items():
        go(page, live, v, width=width, scheme=scheme)
        assert no_sideways(page), (tab, width)
        assert page.errors == [], tab
        legend = page.locator("[data-mt-legend]").first.inner_text()
        assert legend.startswith("● measured here · ○ published · ± standard error · bar = out of "
                                 "100 · bold = best in the column")
        if width < 640:
            # a card a model: its mark, name, line and four scores; tapping opens its page
            assert page.locator("[data-mt-cards]").first.is_visible()
            assert not page.locator("[data-lb-card] .mt-wide").first.is_visible()
            card = page.locator("[data-mt-card]").first
            assert card.locator(".mt-mark").count() == 1
            assert 1 <= card.locator("[data-mt-card-col]").count() <= 4
        else:
            assert not page.locator("[data-mt-cards]").first.is_visible()
            # 14 px on every tab
            size = page.locator("[data-lb-card] table[data-model-table] td.mt-name .mname").first \
                .evaluate("e => getComputedStyle(e.closest('table')).fontSize")
            assert size == "14px", tab
        if scheme == "dark" and (width == 1280 or tab in ("latest", "everyday")):
            shot(page, page.locator("[data-lb-card]").first, SCREENS / f"{tab}-{width}-{scheme}.png")


def test_the_header_and_the_names_stay_put(live, page):
    for tab in ("standard", "frontier"):
        go(page, live, TABS[tab], width=900)
        box = page.locator("[data-lb-card] .mt-box").first
        box.evaluate("b => { b.scrollTop = 120; b.scrollLeft = 400; }")
        page.wait_for_timeout(100)
        got = box.evaluate("""b => { const r = b.getBoundingClientRect();
          const th = b.querySelector('thead tr.mt-hrow th:last-child').getBoundingClientRect();
          const nm = b.querySelector('tbody tr.mt-row td.mt-name').getBoundingClientRect();
          return { scrolled: [b.scrollTop, b.scrollLeft], headTop: Math.round(th.bottom - r.top),
            nameLeft: Math.round(nm.left - r.left) }; }""")
        assert got["scrolled"][0] > 0 and got["scrolled"][1] > 0, (tab, got)
        assert 0 < got["headTop"] < 120, (tab, got)               # the header still in view
        assert got["nameLeft"] <= 41, (tab, got)                  # the names at the left edge
