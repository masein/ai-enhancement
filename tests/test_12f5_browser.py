"""12f.5 on the page: Re-run failed benchmarks on a GGUF run — in its model's
History and on its row in All runs — queues only what the run didn't finish;
and MMLU measured the old way (each option's text, cloze) sits in History,
"cloze, not comparable". The run is the host worker's with the fake
llama-perplexity, as #90 ran: without -np, so ARC and TruthfulQA fail."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

import gguf_bench as gb
import gguf_data as gd
import gguf_worker as gw
from conftest import set_name
from service import config
from test_12f5 import docs
from test_gguf_12f3 import fake_binary

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase12f5"
FAILED = ["arc_challenge", "arc_easy", "truthfulqa"]


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
def run90(live, tmp_path_factory):
    import service.app as appmod
    tmp = tmp_path_factory.mktemp("gg12f5")
    gd.build(config.RESULTS_ROOT / "gguf_data", docs_of=docs())
    f = tmp / "orig.gguf"
    f.write_bytes(b"GGUF" + b"\3" * 400)
    gid = api(live, "/api/gguf/models", {"name": "Qwen3.6 original 12f5", "path": str(f),
                                         "based_on": "Qwen/Qwen3.6-35B-A3B", "how": "unsloth",
                                         "by": "masein"})["model"]["id"]
    sid = api(live, "/api/gguf/runs", {"model": gid, "by": "masein"})["id"]
    gw._stop["why"] = ""
    real = gb.mc_flags
    gb.mc_flags = lambda shape: []                    # the command before 12f.5
    try:
        gw.Worker(config.RESULTS_ROOT, fake_binary(tmp), poll=0.05).once()
    finally:
        gb.mc_flags = real
    # and an MMLU measured on the cloze file, as every one before 12f.5 was
    (config.RESULTS_ROOT / "gguf_results" / "9090.json").write_text(json.dumps({
        "id": "9090", "sid": 9090, "model": gid, "status": "done", "line": "1 benchmarks measured",
        "subset": 0, "finished_at": time.time() - 86400, "setup": gb.AS_BUILT,
        "file": {"name": "orig.gguf", "sha256": "e" * 64},
        "benchmarks": {"mmlu": {"status": "done", "acc": 0.425, "se": 0.004, "n": 14042}},
        "datasets": {"mmlu": {"file": "mmlu-test.bin", "sha256": "c" * 64}}}))
    row = next(r for r in api(live, "/api/submissions") if r["id"] == sid)
    assert row["status"] == "failed" and row["gguf_left"] == FAILED
    appmod._cache.update(key=None, payload=None, at=0.0)
    return {"gid": gid, "sid": sid}


def reruns(live, sid):
    return [r for r in api(live, "/api/submissions")
            if r["suite"] == "gguf" and r["note"].endswith(f"what #{sid} didn't finish")]


def test_history_re_runs_only_the_failed_benchmarks(live, page, run90):
    page.set_viewport_size({"width": 1280, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#model=" + run90["gid"].replace("/", "%2F"))
    page.wait_for_selector("[data-model-hero]")
    page.locator("[data-mtab='history']").click()
    card = page.locator(f"[data-gguf-history='{run90['gid']}']")
    card.wait_for()
    run = card.locator(f"[data-gguf-run='{run90['sid']}']")
    assert "requires a higher -np|--parallel value (at least 5)" in run.inner_text()
    btn = run.locator(f"[data-gguf-rerun='{run90['sid']}']")
    assert btn.inner_text() == "Re-run failed benchmarks"
    assert btn.get_attribute("title") == (
        f"queues ARC-C, ARC-E and TruthfulQA again, as #{run90['sid']} ran them; the finished "
        "ones are kept")
    # the cloze MMLU is there, said so, and has nothing to re-run
    old = card.locator("[data-gguf-run='9090']")
    assert "MMLU 42.5% (cloze, not comparable)" in old.inner_text()
    assert old.locator("[data-gguf-rerun]").count() == 0
    shot(card, "history-rerun-1280.png")
    btn.click()
    toast = page.locator("[data-toast='gguf-rerun']")
    toast.wait_for()
    assert toast.inner_text().startswith("Run #") and "ARC-C, ARC-E, TruthfulQA again" in \
        toast.inner_text()
    [new] = reruns(live, run90["sid"])
    assert json.loads(new["tasks"]) == FAILED and new["submitter"] == "masein"
    assert page.errors == []


def test_all_runs_re_runs_a_failed_gguf_run_from_its_row(live, page, run90):
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto(live["base"] + "/#tab=home")
    set_name(page, "masein")
    page.goto(live["base"] + "/#tab=runs")
    row = page.locator(f"[data-queue-row='{run90['sid']}']")
    row.wait_for()
    btn = row.locator(f"[data-gguf-rerun='{run90['sid']}']")
    assert btn.inner_text() == "Re-run failed benchmarks"
    # the queue's own Resubmit can't queue a GGUF run: it isn't offered
    row.locator(f"[data-row-menu='q{run90['sid']}']").click()
    menu = page.locator(f"#pop-q{run90['sid']}")
    menu.wait_for()
    assert menu.locator("[data-act='resubmit']").count() == 0
    page.keyboard.press("Escape")
    shot(row, "all-runs-row-rerun.png")
    before = len(reruns(live, run90["sid"]))
    btn.click()
    page.wait_for_selector("[data-toast='gguf-rerun']")
    assert len(reruns(live, run90["sid"])) == before + 1
    assert page.errors == []


def test_measure_doesnt_offer_mmlu_on_the_cloze_file(live, page, run90):
    mpath = config.RESULTS_ROOT / "gguf_data" / "manifest.json"
    was = mpath.read_text()
    m = json.loads(was)
    m["benchmarks"]["mmlu"].pop("format")                   # built before 12f.5
    mpath.write_text(json.dumps(m))
    try:
        page.set_viewport_size({"width": 1280, "height": 1000})
        page.goto(live["base"] + "/#tab=home")
        set_name(page, "masein")
        page.goto(live["base"] + "/#model=" + run90["gid"].replace("/", "%2F"))
        page.wait_for_selector("[data-model-hero]")
        page.locator("[data-test-model]").click()             # Measure this model
        dlg = page.locator("[data-dialog='gguf-measure']")
        dlg.wait_for()
        assert dlg.locator("[data-gg-bench='mmlu']").is_disabled()
        assert dlg.locator("[data-gg-no-data='mmlu']").inner_text() == "the old cloze file"
        assert dlg.locator("[data-gg-bench-row='mmlu']").get_attribute("title") == \
            "the old cloze file: build it again (HANDOFF § 5d)"
        assert not dlg.locator("[data-gg-bench='arc_easy']").is_disabled()
        shot(dlg.locator(".dlg"), "measure-cloze-mmlu.png")
    finally:
        mpath.write_text(was)
    assert page.errors == []
