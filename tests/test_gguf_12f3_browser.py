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
MTP = "served/LDA-phone-build-MTP-3"
LOOK = "lookahead 1: LLAMA_MOE_ROUTE_MODE=lookahead LLAMA_MOE_ROUTE_LOOKAHEAD=1"
BASE = "Qwen/Qwen3.6-35B-A3B"
# 12y: lm_eval's own prompts now; what differs still is said
TIP = ("Scored by llama.cpp's llama-perplexity on the quantised file, with lm_eval's own prompts "
       "but no examples (0-shot), each answer by its mean log-probability a token (lm_eval's "
       "acc_norm divides by characters). Not comparable with the lm_eval columns to its left.")


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
                              "phone": True, "gguf_path": str(lda), "gguf_setups": LOOK,
                              "by": "masein"})
    # the same file, served with MTP: a setup of it
    api(live, "/api/served", {"name": "LDA phone build, MTP 3", "base_url": fake.base,
                              "based_on": BASE, "how": "llama.cpp fork, k=4 + LDA, MTP n_max 3",
                              "thinking": "off", "by": "masein"})
    gid = api(live, "/api/gguf/models", {"name": "Qwen3.6 original", "path": str(orig),
                                         "based_on": BASE, "how": "unsloth MTP UD-Q4_K_XL",
                                         "setups": LOOK, "by": "masein"})["model"]["id"]
    binary = fake_binary(tmp)
    gw._stop["why"] = ""
    for mid, acc in ((PHONE, "0.8"), (gid, "0.7")):
        ids = api(live, "/api/gguf/runs", {"model": mid, "by": "masein"})["ids"]
        os.environ["FAKE_PPL_ACC"] = acc
        os.environ["FAKE_PPL_ACC_LOOKAHEAD"] = "0.5"
        for _ in ids:
            gw.Worker(config.RESULTS_ROOT, binary, poll=0.05).once()
    for k in ("FAKE_PPL_ACC", "FAKE_PPL_ACC_LOOKAHEAD"):
        os.environ.pop(k, None)
    # Everyday through both served setups, the MTP one reporting its drafts
    from service import db, runner
    saved = runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL
    runner.acquire_lock, runner.release_lock = (lambda sid: True), (lambda: None)
    config.JUDGE_MODEL = "stub"
    try:
        for mid, t in ((MTP, {"draft_n": 12, "draft_n_accepted": 9}), (PHONE, {})):
            fake.timings = lambda body, t=t: t
            runner.run_submission(db.get(db.add(mid, "instruct", "everyday", "masein", "")))
    finally:
        runner.acquire_lock, runner.release_lock, config.JUDGE_MODEL = saved
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
    # after the others; only the phone's reported numbers come after it (12f.2b)
    assert [g for g in groups if g != "On the phone · reported"][-1] == \
        "Measured on the GGUF · llama.cpp, 0-shot", groups
    th = page.locator("th[data-col='gguf:mmlu']")
    assert "llama-perplexity" in th.get_attribute("data-tip") and TIP in th.get_attribute("data-tip")
    names = page.locator("th[data-col^='gguf:'] .hname").evaluate_all(
        "xs => xs.map(x => x.textContent)")
    # 12n.2: and GPQA Diamond
    assert names == ["MMLU", "HellaSwag", "Winogrande", "ARC-C", "ARC-E", "TruthfulQA",
                     "GPQA Diamond"]
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
    # 12o.1: one Avg a method, never one across them
    assert "no Avg across methods" in page.locator("[data-gguf-mix]").inner_text()
    assert page.locator("th[data-col='cavg'] .hname").text_content() == "Avg · lm_eval"
    assert page.locator("th[data-col='cavg:llama.cpp'] .hname").text_content() == \
        "Avg · llama.cpp"
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
    # 12z B1: its own block, not inside Standard
    page.locator("[data-kind-tile='gguf']").click()
    part = page.locator(f"[data-kind-block='gguf'][open] [data-gguf-part='{PHONE}']")
    part.wait_for()
    assert page.locator("[data-kind-block='standard'] [data-gguf-part]").count() == 0
    assert page.locator("[data-kind-tile='gguf'] [data-kind-value='gguf']").inner_text() == "7 of 7"
    assert part.locator("h3").inner_text() == "Measured on the GGUF · llama.cpp, 0-shot"
    assert part.locator("[data-gguf-row]").count() == 7                # 12n.2: GPQA Diamond too
    pairs = part.locator(f"[data-gguf-pair='{measured['gid']}']").all_inner_texts()
    assert any(p.startswith("LDA phone build vs Qwen3.6 original, as built: MMLU +16.7")
               and "(not a clear difference)" in p for p in pairs), pairs
    # the addendum: its setups side by side, and each against it as built
    assert part.locator("thead [data-gguf-setup]").evaluate_all(
        "xs => xs.map(x => x.textContent)") == ["as built", "lookahead 1"]
    own = part.locator("[data-gguf-pair='setup:lookahead 1']").inner_text()
    assert own.startswith("lookahead 1 vs as built: MMLU \u221233.3")
    shot(part, "model-page-gguf.png")
    # Measure on the GGUF: a dialog since 12f.4 — the estimate from the run it
    # had, the worker's line
    part.locator(f"[data-gg-measure='{PHONE}']").click()
    dlg = page.locator("[data-dialog='gguf-measure']")
    dlg.wait_for()
    assert dlg.locator("[data-gg-setup]").count() == 2
    assert dlg.locator("[data-gg-mtp]").inner_text() == (
        "No MTP setups: llama-perplexity only scores the choices, so there is nothing for MTP "
        "to draft.")
    # 12z B2: measured in both setups already, so nothing is ticked
    page.wait_for_function("document.querySelector('[data-gg-time=\"mmlu\"]').textContent")
    assert dlg.locator("[data-gg-bench]:checked").count() == 0
    assert dlg.locator("[data-gg-start]").is_disabled()
    assert dlg.locator("[data-gg-estimate]").inner_text() == (
        "Every benchmark is measured in every setup: tick what to measure again.")
    assert dlg.locator("[data-gg-done='mmlu']").inner_text() == "measured: as built, lookahead 1"
    dlg.locator("[data-gg-bench='mmlu']").check()
    est = dlg.locator("[data-gg-estimate]").inner_text()
    assert est.startswith("It takes about ") and "rough guess" not in est
    assert "The GGUF worker isn't running." in dlg.locator("[data-gguf-worker-down]").inner_text()
    page.keyboard.press("Escape")
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


def test_served_setups_of_one_file_sit_side_by_side_with_mtps_acceptance(live, page, measured):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#model=" + PHONE.replace("/", "%2F"))
    table = page.locator("[data-served-setups]")
    table.wait_for()
    rows = table.locator("tbody tr")
    assert rows.count() == 2
    mtp = table.locator(f"[data-served-setup='{MTP}']")
    assert mtp.locator(f"[data-served-draft='{MTP}']").inner_text() == "75.0%"
    assert table.locator(f"[data-served-draft='{PHONE}']").inner_text() == "not reported"
    cells = mtp.locator("td").all_inner_texts()
    # 12q.C: DeviceMark's column after the Knowledge exam's
    assert cells[0] == "LDA phone build, MTP 3" and " of " in cells[1] and cells[3] == "—" \
        and " of " in cells[5]
    shot(table, "served-setups-of-one-file.png")
    assert page.errors == []
