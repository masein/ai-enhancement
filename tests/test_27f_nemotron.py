"""27f: NVIDIA-Nemotron-3.5-Bolt-E4B on the On-device report and Models,
published numbers only, as 25c's: its IFBench as NVIDIA's card gives it
for the Q4_K_M file (NVIDIA's own run under NeMo Gym, reasoning on), what it
keeps of Gemma 4 E4B-it's Q4_K_M on the same setup, that file's size from
NVIDIA's file list — the score and the memory of the same file — and
its other two numbers under "Also published" — each with its link, entered
by masein. Its GPQA, MMLU-Pro and IFEval are dashes: the card gives none.
Nothing is fetched; the card was read at its commit on 24 Oct."""

from __future__ import annotations

import json

import pytest

import enter_report_cards as erc
from conftest import make_service
from service import reported, reports

REP = json.loads((reported.REPORTS_DIR / "on-device.json").read_text())
NV = "NVIDIA-Nemotron-3.5-Bolt-E4B"
BASE = "Gemma 4 E4B-it (NVIDIA's run)"
CARD = ("https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/blob/"
        "49843eed8db1f1c616c47af75cfd36e07aec0667/README.md")
FILES = ("https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/tree/"
         "1db54d5b12533ff1bcaa8e42cf4c667988a1f1be")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    assert erc.main(["on-device", "--by", "masein"]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield client
    client.__exit__(None, None, None)


def test_the_file_holds_the_cards_numbers_each_with_its_link():
    t = {x["label"]: x for x in REP["theirs"]}
    nv, base = t[NV], t[BASE]
    assert nv["ifbench"]["value"] == 64.0 and base["ifbench"]["value"] == 42.67
    for c in (nv["ifbench"], base["ifbench"]):
        assert (c["url"], c["date"], c["published_by"]) == (CARD, "2026-10-23", "NVIDIA")
        assert "NeMo Gym" in c["setting"] and "reasoning mode on" in c["setting"]
    # the BF16 pair, on hover beside the Q4_K_M number the cell shows
    assert "Q4_K_M" in nv["ifbench"]["setting"] and "Q4_K_M" in base["ifbench"]["setting"]
    assert "BF16: 63.79" in nv["ifbench"]["setting"] and "BF16: 42.84" in base["ifbench"]["setting"]
    assert not {"gpqa", "mmlupro", "ifeval"} & set(nv)            # the card gives none
    assert base["base_only"] is True and nv["base"] == BASE
    f = nv["facts"]
    assert (f["memory"]["words"], f["memory"]["url"]) == ("6.18 GB", FILES)
    assert "6,180,446,208 bytes" in f["memory"]["how"]
    assert [(a["benchmark"], a["value"]) for a in f["also"]] == [("BFCL-v3", "68.51"),
                                                                 ("Tau2-Bench", "55.10")]
    assert nv["line"] == "trained further from Gemma 4 E4B, not compressed"
    assert nv["line_tip"] == ("its training seed data includes IFEval prompts (the model card's "
                              "data table)")


def test_its_row_and_what_it_keeps_of_gemma(svc):
    rows = {r["label"]: r for r in reports.report("on-device")["rows"]}
    r = rows[NV]
    assert r["cells"]["ifbench"]["v"] == pytest.approx(0.64)
    assert all(r["cells"][k]["v"] is None for k in ("gpqa", "mmlupro", "ifeval"))
    assert round(100 * r["retention"]["ifbench"], 1) == 150.0         # 64.00 of 42.67
    assert set(r["retention"]) == {"ifbench"}
    assert (r["source"], r["publishedBy"]) == ("published by NVIDIA", "NVIDIA")
    assert r["what"] == "trained further from Gemma 4 E4B, not compressed"
    assert r["whatTip"].startswith("its training seed data includes IFEval prompts")
    assert r["memory"]["gb"] == 6.18 and r["phone"]["none"] == "no phone numbers published"
    assert BASE not in rows                                          # a base, never a row


def test_on_models_a_reported_model_of_its_own(svc):
    v = reported.view([])
    m = next(x for x in v["models"] if x["name"] == NV)
    assert m["id"].startswith("reported/") and m["measured"] is None
    assert m["publishedBy"] == "NVIDIA"
    sc = [s for s in v["scores"] if s["model"] == m["id"]]
    assert [(s["benchmark"], s["url"]) for s in sc] == [("IFBench", CARD)]
    assert not any("NVIDIA's run" in x["name"] for x in v["models"])  # the base: the report's
