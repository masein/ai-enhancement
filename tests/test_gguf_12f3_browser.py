"""12f.3 on the page: the "Measured on the GGUF · llama.cpp, 0-shot" group on
Models, only while a model has a result, with its tooltip; a custom table that
refuses to average GGUF columns with lm_eval ones; the model page's GGUF
section with the pairing and "not a clear difference"; the On phone card's
measured MMLU beside the reported one, never merged; History's runs; and the
worker's line when it isn't running. The measurements are the host worker's,
with the fake llama-perplexity."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

import gguf_data as gd
import gguf_worker as gw
from conftest import set_name
from fake_openai import FakeServer
from service import config
from test_gguf_12f3 import docs_of, fake_binary

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f3"
PHONE = "served/LDA-phone-build"
BASE = "Qwen/Qwen3.6-35B-A3B"
TIP = ("Scored by llama.cpp's llama-perplexity on the quantised file. Not comparable with the "
       "lm_eval columns to its left: different prompts and no examples.")


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
def measured(live, tmp_path_factory):
    import os
    import service.app as appmod
    tmp = tmp_path_factory.mktemp("gguf")
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs_of())
    lda, orig = tmp / "lda.gguf", tmp / "orig.gguf"
    lda.write_bytes(b"GGUF" + b"\1" * 400)
    orig.write_bytes(b"GGUF" + b"\2" * 400)
    fake = FakeServer()
    api(live, "/api/served", {"name": "LDA phone build", "base_url": fake.base, "based_on": BASE,
                              "how": "llama.cpp fork, k=4 + LDA", "thinking": "off",
                              "phone": True, "gguf_path": str(lda), "by": "masein"})
    gid = api(live, "/api/gguf/models", {"name": "Qwen3.6 original", "path": str(orig),
                                         "based_on": BASE, "how": "unsloth MTP UD-Q4_K_XL",
                                         "by": "masein"})["model"]["id"]
    binary = fake_binary(tmp)
    gw._stop["why"] = ""
    for mid, acc in ((PHONE, "0.8"), (gid, "0.7")):
        api(live, "/api/gguf/runs", {"model": mid, "by": "masein"})
        os.environ["FAKE_PPL_ACC"] = acc
        gw.Worker(config.RESULTS_ROOT, binary, poll=0.05).once()
    os.environ.pop("FAKE_PPL_ACC", None)
    api(live, "/api/phone/reports", {"model": PHONE, "device": "OnePlus 15",
                                     "decode_median": 13.5, "date": "2026-09-25", "by": "Sam",
                                     "quality": [{"name": "MMLU", "value": "81.98%",
                                                  "note": "all 14,042, on Metal"}],
                                     "entered_by": "masein"})
    api(live, "/api/submissions")
    # the worker that ran here has gone: its heartbeat is two minutes old
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time() - 120}))
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield {"gid": gid}
    fake.close()


def test_models_has_the_gguf_group_only_for_models_with_a_result(live, page, measured):
    page.set_viewport_size({"width": 1600, "height": 900})
    page.goto(live["base"] + "/#tab=models")
    page.wait_for_selector("table[data-lb-table]")
    groups = page.locator("table[data-lb-table] tr.grp th").evaluate_all(
        "xs => xs.map(x => x.textContent)")
    assert groups[-1] == "Measured on the GGUF · llama.cpp, 0-shot", groups
    th = page.locator("th[data-col='gguf:mmlu']")
    assert "llama-perplexity" in th.get_attribute("data-tip") and TIP in th.get_attribute("data-tip")
    names = page.locator("th[data-col^='gguf:'] .hname").evaluate_all(
        "xs => xs.map(x => x.textContent)")
    assert names == ["MMLU", "HellaSwag", "Winogrande", "ARC-C", "ARC-E", "TruthfulQA"]
    cell = page.locator(f"tr[data-lb-row='{measured['gid']}'] td[data-gguf-cell='mmlu']")
    assert cell.inner_text().startswith("66.7")
    # a model with no GGUF result: a dash, not a number
    assert page.locator("tr[data-lb-row='fx/good-750m'] td").nth(-5).inner_text() in ("—", "")
    shot(page.locator("[data-lb-card]"), "models-gguf-group.png")
    assert page.errors == []


def test_the_custom_table_averages_gguf_columns_only_with_each_other(live, page, measured):
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=models&cols=mmlu,gguf:mmlu")
    page.wait_for_selector("[data-custom-line]")
    assert "no Avg: the GGUF columns average only with other GGUF columns" in \
        page.locator("[data-gguf-mix]").inner_text()
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=models&cols=gguf:mmlu,gguf:hellaswag")
    page.wait_for_selector("[data-custom-line]")
    assert page.locator("[data-gguf-mix]").count() == 0
    avg = page.locator(f"tr[data-lb-row='{measured['gid']}'] td[data-watch$='|cavg']")
    assert avg.inner_text().strip() not in ("", "—")
    assert page.errors == []


def test_the_model_page_shows_its_gguf_scores_the_pairing_and_the_phone_card(live, page,
                                                                             measured):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#model=" + PHONE.replace("/", "%2F"))
    page.locator("[data-kind-block='standard'] > summary").click()
    part = page.locator(f"[data-gguf-part='{PHONE}']")
    part.wait_for()
    assert part.locator("h3").inner_text() == "Measured on the GGUF · llama.cpp, 0-shot"
    assert part.locator("[data-gguf-row]").count() == 6
    pair = part.locator(f"[data-gguf-pair='{measured['gid']}']").inner_text()
    assert pair.startswith("LDA phone build vs Qwen3.6 original: MMLU +16.7")
    assert "(not a clear difference)" in pair
    # Measure on the GGUF ▸: the estimate from the run it had, the worker's line
    part.locator("[data-gg-measure] > summary").click()
    est = page.locator("[data-gg-estimate]")
    est.wait_for()
    assert est.inner_text().startswith("It takes about ") and "rough guess" not in est.inner_text()
    assert "The GGUF worker isn't running." in page.locator("[data-gguf-worker-down]").first \
        .inner_text()
    shot(part, "model-page-gguf.png")
    # On phone: the measured MMLU beside the reported one, each labelled, never merged
    page.locator("[data-kind-tile='phone']").click()
    card = page.locator(f"[data-phone-card='{PHONE}']")
    card.wait_for()
    measured_line = card.locator(f"[data-phone-gguf-mmlu='{PHONE}']").inner_text()
    assert measured_line == "MMLU 83.3% — measured here (llama.cpp, 0-shot, full 6)"
    reported = card.locator("[data-phone-field='q-MMLU']").inner_text()
    assert reported.startswith("81.98%") and reported.endswith("reported by Sam")
    shot(card, "on-phone-measured-and-reported.png")
    # History: the run, with what it measured
    page.locator("[data-mtab='history']").click()
    hist = page.locator(f"[data-gguf-history='{PHONE}']")
    hist.wait_for()
    assert "done" in hist.inner_text() and "lda.gguf" in hist.inner_text()
    assert page.errors == []


def test_test_a_model_has_a_gguf_file_and_says_the_worker_is_down(live, page, measured):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.locator("[data-test-model]").click()
    page.locator("[data-gguf-card] > summary").click()
    down = page.locator("[data-gguf-card] [data-gguf-worker-down]")
    down.wait_for()
    assert down.inner_text().startswith("The GGUF worker isn't running.")
    assert "scripts/gguf_worker.py --results " + str(config.RESULTS_ROOT) in down.inner_text()
    # with the worker's heartbeat, the line goes
    (config.RESULTS_ROOT / "gguf_worker.json").write_text(json.dumps({"at": time.time()}))
    page.goto("about:blank")
    page.goto(live["base"] + "/#tab=home")
    page.locator("[data-test-model]").click()
    page.locator("[data-gguf-card] > summary").click()
    page.wait_for_selector("[data-gg-list]")
    assert page.locator("[data-gguf-card] [data-gguf-worker-down]").count() == 0
    assert page.locator(f"[data-gg-row='{measured['gid']}']").count() == 1
    shot(page.locator("[data-gguf-card]"), "test-a-model-gguf.png")
    assert page.errors == []
