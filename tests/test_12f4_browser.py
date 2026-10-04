"""12f.4 on the page: Measure on the GGUF is a dialog — from Test a model's
list of GGUF files, from a GGUF-only model page's header ("Measure this
model") and from its GGUF section: the benchmarks with how many questions
each, the setups, the full sets or a subset, what it takes, and Start."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import gguf_data as gd
from conftest import open_add, set_name
from service import config
from test_gguf_12f3 import docs_of

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f4"
LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"


def shot(part, name, **kw):
    SCREENS.mkdir(parents=True, exist_ok=True)
    part.screenshot(path=SCREENS / name, **kw)


def api(live, path, body=None):
    req = urllib.request.Request(live["base"] + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def gid(live, tmp_path_factory):
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    f = tmp_path_factory.mktemp("gg") / "orig.gguf"
    f.write_bytes(b"GGUF" + b"\1" * 400)
    return api(live, "/api/gguf/models", {"name": "Qwen3.6 original", "path": str(f),
                                          "based_on": "fx/good-750m", "how": "unsloth UD-Q4_K_XL",
                                          "setups": LOOK, "by": "masein"})["model"]["id"]


def home(page, live, width=1280):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")


def test_measure_from_test_a_model_is_a_dialog_that_queues(live, page, gid):
    home(page, live)
    open_add(page, live["base"], "here")
    page.locator(f"[data-gg-open='{gid}']").click()
    dlg = page.locator("[data-dialog='gguf-measure']")
    dlg.wait_for()
    assert page.locator("[data-dialog='test']").count() == 1          # it opens over Test a model
    assert dlg.locator("h2").inner_text() == "Measure on the GGUF"
    rows = dlg.locator("[data-gg-bench-row]")
    assert rows.count() == 7                                          # 12n.2: GPQA Diamond
    n = gd.manifest_of(config.RESULTS_ROOT / "gguf_data") if hasattr(gd, "manifest_of") else json.loads(
        (config.RESULTS_ROOT / "gguf_data" / "manifest.json").read_text())["benchmarks"]
    assert dlg.locator("[data-gg-count='mmlu']").inner_text() == f"{n['mmlu']['n']:,}"
    assert dlg.locator("[data-gg-setup]").evaluate_all("xs => xs.map(x => x.dataset.ggSetup)") \
        == ["as-built", dlg.locator("[data-gg-setup]").nth(1).get_attribute("data-gg-setup")]
    # 12z B2: as built only, every benchmark it has no number for, each with its time
    assert dlg.locator("[data-gg-setup]:checked").evaluate_all(
        "xs => xs.map(x => x.dataset.ggSetup)") == ["as-built"]
    assert dlg.locator("[data-gg-bench]:checked").count() == dlg.locator(
        "[data-gg-bench]:not([disabled])").count() > 0
    assert dlg.locator("[data-gg-start]").inner_text() == "Start"
    page.wait_for_function("document.querySelector('[data-gg-estimate]').textContent.startsWith('It takes')")
    assert dlg.locator("[data-gg-estimate]").inner_text().endswith("a rough guess.")
    for t in dlg.locator("[data-gg-time]").all_inner_texts():
        assert t.startswith("about ") and t.endswith((" min", " h")), t
    assert dlg.locator("[data-gg-setup-time='as-built']").inner_text().startswith("about ")
    assert dlg.locator("[data-gg-setup-time]").nth(1).inner_text() == ""     # not ticked
    shot(dlg.locator(".dlg"), "measure-dialog-default-1280-light.png")      # 12z: as it opens
    dlg.locator("[data-gg-setup]").nth(1).check()
    assert dlg.locator("[data-gg-start]").inner_text() == "Start 2 runs"
    assert dlg.locator("[data-gg-estimate]").inner_text().endswith("a rough guess, for the 2 setups.")
    assert "The GGUF worker isn't running." in dlg.locator("[data-gguf-worker-down]").inner_text()
    shot(dlg.locator(".dlg"), "measure-dialog-1280-light.png")
    # a subset: each benchmark's count says so; one setup: one run
    dlg.locator("[data-gg-size='subset']").check()
    dlg.locator("[data-gg-n]").fill("3")
    dlg.locator("[data-gg-n]").dispatch_event("change")
    assert dlg.locator("[data-gg-count='mmlu']").inner_text() == f"3 of {n['mmlu']['n']:,}"
    dlg.locator("[data-gg-setup]").nth(1).uncheck()
    dlg.locator("[data-gg-bench='truthfulqa']").uncheck()
    assert dlg.locator("[data-gg-start]").inner_text() == "Start"
    before = {r["id"] for r in api(live, "/api/submissions")}
    dlg.locator("[data-gg-start]").click()
    page.wait_for_selector("[data-dialog='gguf-measure']", state="detached")
    page.wait_for_selector("[data-toast='gguf']")
    new = [r for r in api(live, "/api/submissions") if r["id"] not in before]
    assert len(new) == 1 and new[0]["suite"] == "gguf" and new[0]["hf_id"] == gid
    assert new[0]["subset"] == 3 and "truthfulqa" not in json.loads(new[0]["tasks"])
    assert new[0]["note"] == "setup: as built"
    assert page.errors == []


def test_a_gguf_only_page_measures_from_its_header_and_its_section(live, page, gid):
    home(page, live)
    page.goto(live["base"] + "/#model=" + gid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    btn = page.locator("[data-test-model]")
    assert btn.locator(".t-full").inner_text() == "Measure this model"
    btn.click()
    page.wait_for_selector("[data-dialog='gguf-measure']")
    assert page.locator("[data-dialog='test']").count() == 0           # not Test a model
    page.keyboard.press("Escape")
    page.wait_for_selector("[data-dialog='gguf-measure']", state="detached")
    # and from its Scores, which say it isn't measured yet
    assert page.locator("[data-scores-none]").inner_text().startswith("Not measured on its GGUF yet.")
    page.locator(f"[data-gg-measure='{gid}']").click()
    page.wait_for_selector("[data-dialog='gguf-measure']")
    page.keyboard.press("Escape")
    # a model that isn't a GGUF keeps Test this model
    page.goto(live["base"] + "/#model=fx%2Fgood-750m")
    page.wait_for_selector("[data-model-hero]")
    assert page.locator("[data-test-model] .t-full").inner_text() == "Test this model"
    assert page.errors == []


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_dialog_fits_a_phone(live, page, gid, theme):
    home(page, live, width=400)
    page.goto(live["base"] + "/#model=" + gid.replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    page.evaluate(f"applyTheme('{theme}')")
    page.locator("[data-test-model]").click()
    dlg = page.locator("[data-dialog='gguf-measure']")
    dlg.wait_for()
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    shot(page, f"measure-dialog-400-{theme}.png")
    assert page.errors == []
