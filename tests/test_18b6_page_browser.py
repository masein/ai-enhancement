"""18b part 6 on the page (points 31 to 34): the Runs table at 400 px and
1440, light and dark, with a rented run opened — no two cells overlap and no
cell holds more than it shows; one model, one name; times said in words;
the header counts the rented runs; one wording for no contact; a done
rented run says it isn't imported yet; a copy gives focus back. The boxes
are invented, as the fetch posts them; nothing is fetched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_14_3_browser import steady_shot
from test_17i_runs_browser import run_row
from test_17j_page_browser import HLE, runs, world

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase18b"

# every row shown: no cell overlaps another, no cell holds more than it shows,
# and no row runs into the next
LAYOUT = """() => {
  const bad = [];
  const rows = [...document.querySelectorAll('table[data-queue-table] tbody tr')]
    .filter(tr => tr.getClientRects().length);
  const box = e => e.getBoundingClientRect();
  rows.forEach((tr, k) => {
    const tds = [...tr.children].filter(td => td.getClientRects().length && box(td).height > 0);
    for (const td of tds) {
      if (td.scrollHeight > td.clientHeight + 1)
        bad.push(['taller than its cell', tr.dataset.queueRow || tr.dataset.rentedBoxes,
                  td.className, td.scrollHeight, td.clientHeight]);
    }
    for (let i = 0; i < tds.length; i++) for (let j = i + 1; j < tds.length; j++) {
      const a = box(tds[i]), b = box(tds[j]);
      const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
      const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (w > 1 && h > 1) bad.push(['cells overlap', tr.dataset.queueRow, tds[i].className,
                                    tds[j].className]);
    }
    const next = rows[k + 1];
    if (next && box(tr).bottom > box(next).top + 1)
      bad.push(['row runs into the next', tr.dataset.queueRow || tr.dataset.rentedBoxes]);
  });
  return bad;
}"""


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_31_the_runs_table_holds_its_lines_at_400_and_1440(live, page, scheme):
    world()
    page.emulate_media(color_scheme=scheme)
    for width in (400, 1440):
        runs(page, live, width=width)
        hle = run_row(page, "served/17j-bf16", HLE)
        hle.wait_for()
        hle.locator("[data-rented-open]").click()                 # a row opened
        page.wait_for_selector("[data-rented-boxes]")
        bad = page.evaluate(LAYOUT)
        assert bad == [], (width, scheme, bad)                     # 70df001: overlaps at 400
        if width == 1440:
            # "4 boxes ▸" on one line
            btn = hle.locator("[data-rented-open]")
            lh = float(btn.evaluate("e => parseFloat(getComputedStyle(e).lineHeight) || 20"))
            assert btn.bounding_box()["height"] < 2 * lh, btn.bounding_box()
        else:
            SCREENS.mkdir(parents=True, exist_ok=True)
            page.set_viewport_size({"width": 400, "height": 2600})   # no header over the rows
            assert page.evaluate(LAYOUT) == []
            steady_shot(page.locator("table[data-queue-table]"),
                        SCREENS / f"runs-400-{scheme}.png")
        assert page.errors == []


def test_32_to_34_names_times_count_and_words(live, page):
    from service import db
    world()
    # the same model run here as on the rented boxes
    sid = db.add("served/17j-bf16", "instruct", "frontier", "masein", "", tasks=[HLE],
                 status="running")
    db.update(sid, progress="Humanity's Last Exam 10 of 540")
    runs(page, live)
    here = page.locator(f"[data-queue-row='{sid}']")
    hle = run_row(page, "served/17j-bf16", HLE)
    hle.wait_for()
    name_here = here.locator("[data-run-name]").evaluate("e => e.firstChild.textContent")
    name_rented = hle.locator("[data-run-name]").evaluate("e => e.firstChild.textContent")
    assert name_here == name_rented == "17j-bf16"                   # 70df001: served/… there
    assert page.locator("table[data-queue-table] .qc-model .badge").count() == 0
    # 33: no time in the model cell; a box's times in words, no step id
    for row in page.locator("[data-rented-run]").all():
        assert "its box" not in row.locator(".qc-model").inner_text()
    hle.locator("[data-rented-open]").click()
    line = page.locator("[data-rented-box]").first
    line.wait_for()
    text = line.inner_text()
    assert "this part done " in text and "→" not in text, text
    assert "A1-1" not in text and text.startswith("A1 · shard 1/4"), text
    assert hle.locator("[data-rented-finish]").inner_text().startswith("this part done ")
    # 34: the header counts the rented runs running beside the runs here
    here_running = page.evaluate("runsNow().running.length")
    page.wait_for_function(f"+document.querySelector('[data-runs]').dataset.runs === "
                           f"{here_running + 3}")                    # 70df001: here alone
    # a done rented run says it isn't imported yet
    done = run_row(page, "served/17j-k8", "mmlupro_tiger")
    assert "not imported yet" in done.inner_text()
    # one wording for no contact
    assert "heard " not in page.locator("table[data-queue-table]").inner_text()
    # a copy by hand gives focus back to what had it
    assert page.evaluate("""() => { const b = document.querySelector('[data-rented-open]');
        b.focus(); copyByHand('x'); return document.activeElement === b; }""")
    assert page.errors == []


def test_35_the_window_button_says_a_run_left_part_way_starts_again(live, page):
    from fake_openai import FakeServer
    from service import db, served

    import service.app as appmod
    from conftest import set_name
    srv = FakeServer()
    srv.key = "k-18b6"
    try:
        mid = "served/window-18b6"
        db.served_put({"id": mid, "name": "window 18b6", "base_url": srv.base, "key": srv.key,
                       "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "llama-server -c 16384",
                       "thinking": "auto", "phone": False,
                       "pin": served.pin_of(served.probe(srv.base, srv.key)), "by": "masein",
                       "at": 0})
        served.write_meta(served.get(mid))
        srv.ctx = 65536
        appmod._cache.update(key=None, payload=None, at=0.0)
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto("about:blank")
        page.goto(live["base"] + f"/#model={mid}")
        set_name(page, "masein")
        note = page.locator(f"[data-served-window-note='{mid}']")
        note.wait_for()
        assert "starts again from its first question" in note.inner_text()
        # a window the board can't take: no button, said
        srv.ctx = 512
        got = json.loads(json.dumps(appmod.served_pin(id=mid)))
        assert got["window"] is None and ("doesn't report a usable context window (it says "
                                          "512)") in got["why"]           # 18c: its words
        assert page.errors == []
    finally:
        srv.close()
