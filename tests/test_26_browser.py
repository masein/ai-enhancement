"""26 in a real browser, at 1280 and 400 px, light and dark: On our testers'
phones under Reports — the tiles, every test newest first with a flagged one
greyed and its reason on hover, a test opened to its checks, the medians per
phone and model, what failed — and the phone build's model page, its
measured numbers above the ones typed in by hand, and its column in the
On-device report. Nine made-up tests in the attached test's shape (made-up
ids, testers and replies), as today: three server crashes, two on one
Xiaomi; no R2 keys, so the sync says it isn't set up."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from service import config, db, reports, served, telemetry
from test_21c_page import results
from test_21d_browser import no_sideways
from test_26_telemetry import a_test, keep

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase26"
SIZES = [(1280, "light"), (1280, "dark"), (400, "light"), (400, "dark")]
PHONE = "served/fx-k4-phone-build"
GPQA = next(b["task"] for b in json.loads((reports.REPORTS_DIR / "on-device.json").read_text())
            ["benchmarks"] if b["key"] == "gpqa")
NAME = "fx k4 phone build"
# (test id, tester, received, what the case changes)
TESTS = [
    ("t1", "tester-a", "2026-10-20T17:55:16Z", {}),
    ("t2", "tester-a", "2026-10-19T09:10:00Z", {"decode": 13.6, "temp_end": 34.1}),
    ("t3", "tester-b", "2026-10-18T12:00:00Z", {"decode": 12.4, "temp_end": 33.9}),
    ("t4", "tester-b", "2026-10-18T15:30:00Z", {"decode": 400}),          # impossible
    ("t5", "tester-c", "2026-10-17T08:00:00Z", {"charging": True, "temp_end": 39.0}),
    ("t6", "tester-c", "2026-10-16T20:45:00Z", {"device": "SM-S938B", "decode": 9.8}),
    ("x1", "tester-d", "2026-10-16T10:00:00Z", {"device": "23078PND5G", "outcome": "server_crash"}),
    ("x2", "tester-d", "2026-10-15T10:00:00Z", {"device": "23078PND5G", "outcome": "server_crash"}),
    ("x3", "tester-c", "2026-10-14T10:00:00Z", {"device": "SM-S938B", "outcome": "server_crash"}),
]


@pytest.fixture(scope="module", autouse=True)
def board(live, tmp_path_factory):
    names = tmp_path_factory.mktemp("names") / "names.json"
    names.write_text(json.dumps({
        "models": {"model-004": {"name": "k4 phone build, Q4 (Qwen3.6-35B-A3B)", "row": PHONE}},
        "tiers": {"T0": "tier 0"}, "engines": {"e-00000001": "engine, made-up build"},
        "devices": {"CPH2747": "OnePlus 15", "SM-S938B": "Galaxy S25 Ultra"}}))
    saved = telemetry.NAMES
    telemetry.NAMES = names
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)
    db.served_put({"id": PHONE, "name": NAME, "base_url": "", "key": "", "how": "phone build",
                   "based_on": "Qwen/Qwen3.6-35B-A3B", "thinking": "off", "phone": True,
                   "pin": {}, "by": "masein", "at": 0})
    # a score, so the build is a row on the board (invented), and how it was
    # served, as a run through it keeps it
    d = reports.row_dir(PHONE)
    results(d, PHONE, GPQA, 0.70, 0.02, 100, 0)
    for f in d.glob("*/frontier/results_*.json"):
        f.write_text(f.read_text().replace(",enable_thinking=True", ""))
    (d / "model_meta.json").write_text(json.dumps({"model": PHONE,
                                                   "served": served.view(db.served_get(PHONE))}))
    db.phone_add(PHONE, {"device": "OnePlus 15", "decode_median": 13.5, "by": "Sam",
                         "date": "2026-10-01", "entered_by": "masein", "at": 0})
    for tid, who, at, kw in TESTS:
        t = a_test(tid, install="00000000-0000-4000-8000-0000000000" + tid[-1].zfill(2),
                   received=at, **kw)
        t["label"] = who
        keep(Path(config.BENCH_ROOT), t)
    import service.app as appmod
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    telemetry.NAMES = saved
    telemetry._memo.update(stamp=None)
    telemetry._names_memo.update(stamp=None)


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


def place(page, live, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=testers")
    # 27c: 27's table — on a phone, its cards; the table is in the page all the same
    page.wait_for_selector("[data-pt-table]", state="attached", timeout=30000)


@pytest.mark.parametrize("width,scheme", SIZES)
def test_on_our_testers_phones(live, page, width, scheme):
    place(page, live, width, scheme)
    assert page.locator("[data-pt-sync]").inner_text() == \
        "Phone checks: not set up — R2 keys missing in .env"
    tiles = page.eval_on_selector_all("[data-pt-tile]", "ts => ts.map(t => [t.dataset.ptTile, "
                                      "t.querySelector('.value').textContent])")
    assert tiles == [["tests", "9"], ["phones", "3"], ["testers", "4"], ["unfinished", "3"]]
    # newest first, as Kevin's
    rows = page.eval_on_selector_all("[data-pt-row]", "rs => rs.map(r => r.dataset.ptRow)")
    assert rows == ["t1", "t2", "t4", "t3", "t5", "t6", "x1", "x2", "x3"]
    # 27c: 27's table, grouped under Speed, Memory and Battery; the engine build,
    # thermal max, charging and first load are in a test's panel
    heads = page.eval_on_selector_all("[data-pt-table] thead tr.mt-hrow th[data-mt-col]",
                                      "ts => ts.map(t => t.dataset.mtCol)")
    assert heads == ["name", "phone", "app", "model", "result", "answers", "ttft", "prefill",
                     "decode", "sustained", "engine", "swap", "memmin", "battery", "rise"]
    cell = lambda t, k: page.locator(f"[data-pt-cell='{t}|{k}']")       # noqa: E731
    first = lambda t, k: cell(t, k).evaluate(                           # noqa: E731
        "c => (c.querySelector('.mt-n') || c).textContent")
    # the attached test, read as his dashboard reads it
    assert [first("t1", k) for k in ("ttft", "prefill", "decode", "sustained", "engine",
                                      "swap", "memmin", "battery", "rise", "answers")] == [
        "5.0 s", "36.2", "13.0", "−4.5%", "4,565 MB", "6,452 MB", "5,553 MB",
        "32.3 → 33.7", "+1.4 °C", "8/8"]
    parts = lambda t, k: cell(t, k).evaluate(                           # noqa: E731
        "c => [...c.querySelectorAll('.mt-n, .mt-sub')].map(x => x.textContent)")
    assert parts("t1", "phone") == ["OnePlus 15", "16 GB RAM"]
    assert parts("t1", "model")[0] == "k4 phone build, Q4 (Qwen3.6-35B-A3B)"
    assert first("x1", "phone") == "23078PND5G (not named yet)"
    assert first("x1", "result") == "server_crash"
    # flagged: the impossible one greyed whole, its reason on hover; the
    # charging one greyed in its battery column only
    t4 = page.locator("[data-pt-row='t4']")
    assert "pt-flag" in t4.get_attribute("class")
    # 27c: the "flagged" word under its date gives the reason on hover
    assert page.locator("[data-pt-flag-why='t4']").get_attribute("title") == (
        "decode 400 tok/s: no phone does that with a 30B-class model (above 60)")
    assert "pt-off" in cell("t5", "battery").get_attribute("class")
    assert "pt-off" not in cell("t5", "decode").get_attribute("class")
    assert cell("t5", "battery").get_attribute("title").startswith("it ran while charging")
    # per phone and model: OnePlus 15's medians, without the impossible one
    op = page.locator("[data-pt-per-row='CPH2747|model-004']")
    # 27b: modelTable's row and cells
    assert op.locator("[data-pt-per-n] .mt-n").evaluate("c => c.textContent") == "4 of 5"
    assert op.locator("[data-pt-per-decode] .mt-n").evaluate("c => c.textContent") == "13.0"
    assert op.locator("[data-pt-per-rise] .mt-n").evaluate("c => c.textContent") == (
        "+1.6 °C")                                                         # not t5's, charging
    # what failed, by phone and app version
    failed = page.eval_on_selector_all("[data-pt-failed]", "ls => ls.map(l => l.dataset.ptFailed)")
    assert failed == ["23078PND5G|1.4", "SM-S938B|1.4"]
    assert no_sideways(page) and page.errors == []
    tag = f"{width}-{scheme}"
    shot(page, page.locator("[data-report]"), SCREENS / f"testers-head-{tag}.png")
    shot(page, page.locator("[data-report-section='pt-tests']"), SCREENS / f"testers-tests-{tag}.png")
    shot(page, page.locator("[data-report-section='pt-per-phone']"),
         SCREENS / f"testers-per-phone-{tag}.png")
    shot(page, page.locator("[data-report-section='pt-failed']"), SCREENS / f"testers-failed-{tag}.png")
    # a test opened, in its panel (27c): every check, its time, speeds and reply
    page.locator("[data-pt-row='t1'] td.mt-name" if width >= 640 else "[data-mt-card='t1']").click()
    page.wait_for_selector("[data-pt-check='sustain']")
    checks = page.eval_on_selector_all("[data-pt-check]", "rs => rs.map(r => r.dataset.ptCheck)")
    assert checks == ["speed-cold", "speed-warm", "id-who", "math-fast", "think-math", "web-no",
                      "web-yes", "id-probe", "about-tf", "sustain"]
    assert page.locator("[data-pt-reply='speed-cold']").inner_text() == "Ready"
    assert no_sideways(page) and page.errors == []
    shot(page, page.locator("[data-pt-test]"), SCREENS / f"testers-test-{tag}.png")


def test_the_reports_list_says_what_is_in(live, page):
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports")
    page.wait_for_selector("[data-report-card='testers']", timeout=30000)
    assert page.locator("[data-report-models='testers']").inner_text() == \
        "9 tests on 3 phones by 4 testers"
    assert page.locator("[data-report-changed='testers']").inner_text() == \
        "Phone checks: not set up — R2 keys missing in .env"
    assert page.errors == []


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (400, "dark")])
def test_the_phone_builds_page_measured_beside_reported_by_hand(live, page, width, scheme):
    page.emulate_media(color_scheme=scheme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#model=" + PHONE.replace("/", "%2F"))
    tile = page.locator("[data-kind-tile='phone']")
    tile.wait_for(timeout=30000)
    page.wait_for_function("() => document.querySelector(\"[data-kind-value='phone']\")"
                           ".textContent === '13.0'")
    assert "measured on our testers’ phones" in tile.inner_text()
    tile.click()
    block = page.locator(f"[data-phone-testers='{PHONE}']")
    block.wait_for()
    rows = page.eval_on_selector_all("[data-phone-tested]", "rs => rs.map(r => r.dataset.phoneTested)")
    assert rows == ["Galaxy S25 Ultra", "OnePlus 15"]
    assert page.locator("[data-phone-tested-decode='OnePlus 15']").inner_text() == "13.0 tok/s"
    # typed in by hand: its own block, said so, never mixed in
    rep = page.locator(f"[data-phone-reported='{PHONE}']")
    assert rep.locator("h3").inner_text() == "Reported by hand"
    assert "13.5 tok/s median" in rep.inner_text() and "13.0" not in rep.inner_text()
    assert "13.5" not in block.inner_text()
    assert no_sideways(page) and page.errors == []
    shot(page, page.locator("[data-kind-block='phone']"),
         SCREENS / f"phone-build-page-{width}-{scheme}.png")


def test_the_on_device_report_has_our_testers_column(live, page):
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=reports&report=on-device")
    page.wait_for_selector("[data-od-table]", timeout=30000)
    row = f"{NAME} (phone build)"
    # 27b: one column, On a phone — our testers' median wins it: the row's dot,
    # decode tok/s, the phone and its tests under it; every phone on hover
    cell = page.locator(f"[data-od-fact='{row}|phone']")
    assert cell.locator("[data-mt-cell-mark='here']").count() == 1
    assert cell.locator(".mt-n").inner_text() == "13.0 tok/s"
    assert cell.locator(".mt-sub").inner_text() == "OnePlus 15 · 4 tests · +1 phone"
    tip = cell.get_attribute("data-tip")
    assert "Galaxy S25 Ultra: 9.8 tok/s · engine memory 4,565 MB, median of 1 test, 2026-10-16" in tip
    assert "OnePlus 15: 13.0 tok/s · engine memory 4,565 MB, median of 4 tests, 2026-10-17 to " \
        "2026-10-20" in tip
    # the hand-typed one never mixed in: on hover, said so
    assert "reported by hand: 13.5 tok/s decode, median on OnePlus 15 by Sam" in tip
    # a published model's: its own speed, in italics with its letter, the phone under it
    b27 = page.locator("[data-od-fact='Bonsai 27B (1-bit)|phone']")
    assert b27.locator(".mt-n.mt-pub").inner_text() == "11 tok/sP"
    assert b27.locator(".mt-sub").inner_text() == "iPhone 17 Pro Max"
    assert page.errors == []
    shot(page, page.locator("[data-report-section='od-table']"), SCREENS / "on-device-testers-1280.png")
