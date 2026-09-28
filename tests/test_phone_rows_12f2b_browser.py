"""12f.2b on the page: phone builds and their setups are ordinary rows on
Models with a "phone build" tag — a served setup of the build's file, and a
GGUF setup named "… · lookahead 1"; Models ▾ has "phone builds" and "served"
next to instruct and base; the phone's reported numbers are a column group of
their own, "On the phone · reported", shown only while a row shown has them,
never averaged; and a served model has only the columns it can have — no row
of dashes, and no row at all where it can have none."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pytest

import gguf_data as gd
import gguf_worker as gw
from fake_openai import FakeServer
from service import config
from test_gguf_12f3 import docs_of, fake_binary

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f2b"
PHONE = "served/k4-LDA"
MTP = "served/k4-LDA-MTP"
OTHER = "served/Qwen3.6-original"
BASE = "Qwen/Qwen3.6-35B-A3B"
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
def board(live, tmp_path_factory):
    import service.app as appmod
    tmp = tmp_path_factory.mktemp("rows")
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    lda = tmp / "lda.gguf"
    lda.write_bytes(b"GGUF" + b"\1" * 400)
    fake = FakeServer()
    api(live, "/api/served", {"name": "k4-LDA", "base_url": fake.base, "based_on": BASE,
                              "how": "llama.cpp fork, k=4 + LDA", "thinking": "off", "phone": True,
                              "gguf_path": str(lda), "gguf_setups": LOOK, "by": "masein"})
    api(live, "/api/served", {"name": "k4-LDA · MTP", "base_url": fake.base, "based_on": BASE,
                              "how": "llama.cpp fork, k=4 + LDA, MTP n_max 3", "thinking": "off",
                              "by": "masein"})
    # a served model on another file is served, not a phone build
    other = FakeServer()
    other.model_path = "/home/masein/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    api(live, "/api/served", {"name": "Qwen3.6 original", "base_url": other.base,
                              "based_on": BASE, "how": "unsloth UD-Q4_K_XL", "thinking": "off",
                              "by": "masein"})
    binary = fake_binary(tmp)
    gw._stop["why"] = ""
    os.environ["FAKE_PPL_ACC_LOOKAHEAD"] = "0.8"
    for _ in api(live, "/api/gguf/runs", {"model": PHONE, "by": "masein"})["ids"]:
        gw.Worker(config.RESULTS_ROOT, binary, poll=0.05).once()
    api(live, "/api/phone/reports", {"model": PHONE, "device": "OnePlus 15", "decode_median": 13.5,
                                     "decode_best": 16.0, "date": "2026-09-25", "by": "Sam",
                                     "quality": [{"name": "MMLU", "value": "81.98%",
                                                  "note": "all 14,042"}], "entered_by": "masein"})
    api(live, "/api/submissions")
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - 120}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield
    fake.close()
    other.close()
    os.environ.pop("FAKE_PPL_ACC_LOOKAHEAD", None)


def models(page, live, hash_="#tab=models", width=1600):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto("about:blank")
    page.goto(live["base"] + "/" + hash_)
    page.wait_for_selector("table[data-lb-table]")


def test_phone_builds_and_their_setups_are_rows_with_a_tag(live, page, board):
    models(page, live)
    for rid in (PHONE, f"{PHONE} · lookahead 1"):
        row = page.locator(f"tr[data-lb-row='{rid}']")
        row.wait_for()
        assert row.locator("[data-phone-tag]").inner_text() == "phone build"
        # a phone build is served: one tag says both
        assert row.locator("[data-served-tag]").count() == 0
    # the name is whole, not cut to fit the tags
    assert page.locator(f"tr[data-lb-row='{PHONE} · lookahead 1'] .mname").evaluate(
        "x => x.scrollWidth <= x.clientWidth")
    assert page.locator(f"tr[data-lb-row='{PHONE} · lookahead 1'] .mname").inner_text() == \
        "k4-LDA · lookahead 1"
    # a setup row opens its model's page
    page.locator(f"tr[data-lb-row='{PHONE} · lookahead 1'] .mname").click()
    page.wait_for_selector(f"[data-served-head='{PHONE}']")
    assert page.errors == []


def test_models_menu_groups_phone_builds_and_served_next_to_instruct_and_base(live, page, board):
    models(page, live)
    page.locator("[data-models-menu]").click()
    groups = page.locator("[data-model-group]").evaluate_all(
        "xs => xs.map(x => x.dataset.modelGroup)")
    # 12o.1: a served model and its GGUF by model and setup — the phone build first
    assert groups[0].startswith("model:") and "instruct" in groups and "base" in groups
    assert "phone builds" not in groups and "served" not in groups
    phone = groups[0]
    assert page.locator(f"[data-model-group='{phone}']").inner_text().strip().startswith(
        "phone build · ")
    # a served entry on the build's own file is one of its setups
    picks = page.locator("#pop-models").evaluate(
        """p => { const out = {}; let g = null;
                  for (const x of p.querySelectorAll('[data-model-group], [data-setup-pick]'))
                    if (x.dataset.modelGroup) g = x.dataset.modelGroup;
                    else for (const id of JSON.parse(x.dataset.ids)) out[id] = g;
                  return out; }""")
    assert picks[PHONE] == picks[MTP] == phone and picks[OTHER] not in (None, phone)
    page.locator(f"[data-model-group-only='{phone}']").click()
    page.keyboard.press("Escape")
    ids = page.locator("tr[data-lb-row]").evaluate_all("xs => xs.map(x => x.dataset.lbRow)")
    assert set(ids) == {PHONE, f"{PHONE} · lookahead 1"}
    # the old On phone link lands on the same
    models(page, live, "#tab=models&view=phone")
    page.wait_for_function("location.hash.includes('models=')")
    assert "view=phone" not in page.evaluate("location.hash")
    shot(page.locator("[data-lb-card]"), "phone-builds-only.png")
    assert page.errors == []


def test_the_reported_numbers_are_a_group_of_their_own_never_averaged(live, page, board):
    models(page, live)
    groups = page.locator("table[data-lb-table] tr.grp th").evaluate_all(
        "xs => xs.map(x => x.textContent)")
    assert groups[-1] == "On the phone · reported"
    row = page.locator(f"tr[data-lb-row='{PHONE}']")
    assert row.locator("[data-rep-cell='rep:median']").inner_text() == "13.5\nreported by Sam"
    assert row.locator("[data-rep-cell='rep:best']").inner_text() == "16.0\nreported by Sam"
    assert row.locator("[data-rep-cell='rep:q:MMLU']").inner_text().startswith("81.98%")
    # a served row has only the columns it can have: blank, not dashes, and no Avg
    na = row.locator("td[data-na]")
    assert na.count() > 0 and set(na.all_inner_texts()) == {""}
    assert "Multiple-choice benchmarks need the model loaded here" in na.first.get_attribute("title")
    assert row.locator("td[data-watch$='|avg']").count() == 0 or \
        row.locator("td.na").count() >= 1
    shot(page.locator("[data-lb-card]"), "models-phone-rows-1600.png")
    page.locator("table[data-lb-table]").evaluate("""t => {
        let p = t.parentElement;
        while (p && getComputedStyle(p).overflowX === 'visible') p = p.parentElement;
        p.scrollLeft = p.scrollWidth; }""")
    shot(page.locator("[data-lb-card]"), "models-reported-group-1600.png")
    # only while a row shown has them: base models alone, and the group goes
    page.locator("[data-models-menu]").click()
    page.locator("[data-model-group-only='base']").click()
    page.keyboard.press("Escape")
    assert page.locator("th[data-col^='rep:']").count() == 0
    # never a benchmark to average: not in Benchmarks ▾
    models(page, live)
    page.locator("[data-benchmarks-menu]").click()
    assert "tok/s" not in page.locator("#pop-benchmarks").inner_text()
    assert page.errors == []


@pytest.mark.parametrize("chip", ["knowledge", "commonsense"])
def test_a_chip_a_served_model_can_have_nothing_in_leaves_it_out(live, page, board, chip):
    # MMLU, its areas and topics, and the commonsense tasks all need the model
    # loaded here: a served model is neither a row nor "not tested" on them
    models(page, live, f"#tab=models&chip={chip}")
    toggle = page.locator("[data-not-tested-toggle]")
    if toggle.count():
        toggle.click()
    for rid in (PHONE, MTP, OTHER, f"{PHONE} · lookahead 1"):
        assert page.locator(f"tr[data-lb-row='{rid}']").count() == 0
        assert page.locator(f"tr[data-not-tested-row='{rid}']").count() == 0
    assert page.errors == []
