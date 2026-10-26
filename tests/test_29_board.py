"""29: NVIDIA Nemotron 3.5 Bolt E4B measured here, on our own V100s — the
board's side (docs/prompts/phase-29-own-boxes-and-bolt.md, Part 4): its two
measured rows in the Open group, one model page holding both builds and
NVIDIA's published numbers told apart, the On-device report's measured
Q4_K_M row against its measured Q8_0 ("of our Q8_0, standing in for BF16"),
its 64K context's cap said on its cells, an own box's speed labelled with
its card and never in one median with an RTX 5090's, and Runs' own box.
Invented scores and card checks; nothing is run, fetched or graded."""

from __future__ import annotations

import json

import pytest

import enter_report_cards as erc
from conftest import make_service
from service import beyond, config, db, ondevice, reported, reports
from test_21c_page import results

REP = json.loads((reported.REPORTS_DIR / "on-device.json").read_text())
TASK = {b["key"]: b["task"] for b in REP["benchmarks"]}
Q4, Q8 = "NVIDIA Nemotron 3.5 Bolt E4B · Q4_K_M", "NVIDIA Nemotron 3.5 Bolt E4B · Q8_0"
Q4_ID = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q4_K_M"
Q8_ID = "served/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0"
PUB = "NVIDIA-Nemotron-3.5-Bolt-E4B"
CAPPED = "limit capped by its 64K context"
# invented: {row: {benchmark key: (score, se)}}
BOLT = {Q4_ID: {"gpqa": (0.52, 0.02), "ifeval": (0.80, 0.017), "ifbench": (0.60, 0.028)},
        Q8_ID: {"gpqa": (0.55, 0.02), "ifeval": (0.82, 0.016), "ifbench": (0.62, 0.028)}}


def bolt_world(sizes: bool = True) -> None:
    """Bolt's two builds as the board would hold them after the import: each
    registered by its name, its thinking row scored on three benchmarks"""
    import import_frontier as imf
    for mid, cells in BOLT.items():
        name = Q4 if mid == Q4_ID else Q8
        rec = {"id": mid, "name": name, "base_url": "", "key": "",
               "how": "own box D3 · V100", "runtime": "card-bolt",
               "based_on": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "thinking": "on",
               "pin": {"ctx": 65536}, "by": "masein", "at": 0,
               **({"gguf_file": {"size": 6_180_446_208 if mid == Q4_ID
                                 else 8_031_245_312}} if sizes else {})}
        db.served_put(rec)
        row = reports.row_dir(mid + " · thinking")
        imf.write_meta(row, rec, True)                  # as the import registers it
        for k, (v, se) in cells.items():
            results(row, mid, TASK[k], v, se, 300, 0)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    client, appmod, _ = make_service(tmp_path, monkeypatch, tree=False)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert erc.main(["on-device", "--by", "masein"]) == 0
    appmod._cache.update(key=None, payload=None, at=0.0)
    yield client
    client.__exit__(None, None, None)


def test_the_file_names_the_two_builds_the_q8_0_standing_in_for_bf16():
    m = {x["label"]: x for x in REP["measured"]}
    assert set(m) == {Q4, Q8}
    assert m[Q4]["row"] == Q4_ID + " · thinking" and m[Q8]["row"] == Q8_ID + " · thinking"
    assert m[Q4]["base"] == Q8 and m[Q4]["base_line"] == "our Q8_0, standing in for BF16"
    assert m[Q8]["base_only"] is True and m[Q8]["stands_in_for"] == "BF16"
    assert {x["published"] for x in m.values()} == {PUB} and {x["card"] for x in m.values()} == {
        "V100"}
    # NVIDIA's published row stays as published: its number, its base, its link
    t = {x["label"]: x for x in REP["theirs"]}
    assert t[PUB]["ifbench"]["value"] == 64.0 and t[PUB]["base"] == "Gemma 4 E4B-it (NVIDIA's run)"
    # never two rows with one name: the measured and the published are named apart
    labels = [x["label"] for x in [*REP["ours"], *REP["measured"], *REP["theirs"]]]
    assert len(labels) == len(set(labels))


def test_the_measured_q4_k_m_row_keeps_a_share_of_the_measured_q8_0(svc):
    bolt_world()
    d = reports.report("on-device")
    rows = {r["label"]: r for r in d["rows"]}
    r = rows[Q4]
    assert (r["ours"], r["measured"], r["source"], r["card"]) == (False, True, "measured here",
                                                                   "V100")
    assert r["baseLabel"] == "our Q8_0, standing in for BF16"
    assert r["retention"] == {"gpqa": pytest.approx(0.52 / 0.55),
                              "ifeval": pytest.approx(0.80 / 0.82),
                              "ifbench": pytest.approx(0.60 / 0.62)}
    assert r["cells"]["gpqa"]["v"] == pytest.approx(0.52) and r["cells"]["gpqa"]["src"] == "here"
    assert r["cells"]["mmlupro"]["v"] is None and r["cells"]["mmlupro"]["why"] == "not run yet"
    assert r["notRun"] == ["mmlupro"]
    assert r["capped"] == CAPPED
    assert r["memory"]["words"] == "6.18 GB" and "loaded on the server" in r["memory"]["how"]
    assert r["phone"] == {"none": "not measured on a phone"} and r["testers"] is None
    assert Q8 not in rows                                         # a base, never a row
    # NVIDIA's own, unchanged: published, its IFBench and what it keeps of Gemma
    p = rows[PUB]
    assert p["source"] == "published by NVIDIA" and p["cells"]["ifbench"]["v"] == pytest.approx(0.64)
    assert round(100 * p["retention"]["ifbench"], 1) == 150.0
    assert "measured here" in d["measuredNote"] and "standing in for BF16" in d["measuredNote"]
    # after ours, before the published
    order = [x["label"] for x in d["rows"]]
    assert order.index(Q4) > order.index("Q2 (UD-Q2_K_XL)") and order.index(Q4) < order.index(PUB)


def test_before_its_answers_it_says_not_run_yet(svc):
    d = reports.report("on-device")
    r = next(x for x in d["rows"] if x["label"] == Q4)
    assert all(c["v"] is None and c["why"] == "not run yet" for c in r["cells"].values())
    # no score and no memory yet: counted under the table, as ours are; no "–" in the panels
    assert r["memory"] is None and r["retention"] == {} and r["notRun"] == []


def test_the_reports_list_counts_it_among_the_others():
    assert ondevice.listing_counts(REP)[1] == 10                     # 27f's nine, and Bolt's


def test_one_model_page_holds_both_builds_and_the_published_numbers(svc):
    bolt_world()
    g = reports.model_builds()
    assert len(g) == 1
    g = g[0]
    assert (g["name"], g["maker"], g["card"]) == ("NVIDIA Nemotron 3.5 Bolt E4B", "NVIDIA", "V100")
    assert [(b["id"], b["standsInFor"], b["capped"]) for b in g["builds"]] == [
        (Q4_ID, "", CAPPED), (Q8_ID, "BF16", CAPPED)]
    assert g["published"] == {"label": PUB, "key": "nvidia/nvidia-nemotron-3.5-bolt-e4b",
                              "page": "reported/nvidia/nvidia-nemotron-3.5-bolt-e4b",
                              "by": "NVIDIA"}
    # the board's payload carries it, and the models its context caps
    p = svc.get("/api/results").json()
    assert p["builds"] == [g]
    assert p["frontierCaps"][Q4_ID] == {"context": 65536, "words": CAPPED}
    assert set(p["frontierCaps"]) == {Q4_ID, Q8_ID}
    # the published model keeps its own page: never folded into a measured row
    v = reported.view(reported.board_ids())
    m = next(x for x in v["models"] if x["name"] == PUB)
    assert m["id"] == "reported/nvidia/nvidia-nemotron-3.5-bolt-e4b" and m["measured"] is None
    names = {x["id"]: x["name"] for x in p["models"]}
    assert names[Q4_ID + " · thinking"].startswith(Q4)


def test_the_cap_is_bolts_alone():
    assert reports.capped_words(Q4_ID + " · thinking") == CAPPED
    assert reports.capped_words(Q8_ID) == CAPPED
    assert reports.capped_words("served/step4000-lda-2610 · thinking") == ""
    assert reports.capped_words("served/Ornith-1.5-35B-A3B-Q4_K_M") == ""


def card(name: str, alone: float, who: str) -> dict:
    return {"name": name, "alone_out": alone, "alone_prompt": 64 * alone,
            "together_out": 6 * alone, "together_n": 8, "box": who}


def test_an_own_boxs_speed_is_its_cards_never_in_one_median_with_a_5090s(svc):
    gpqa = "gpqa_diamond_epoch"
    d = reports.row_dir(Q4_ID + " · thinking")
    d.mkdir(parents=True, exist_ok=True)
    v100, rtx = "Tesla V100-DGXS-32GB", "NVIDIA GeForce RTX 5090"
    parts = [{"box": "D3.0", "card": card(v100, 60.0, "D3.0"), "answers": 100, "energy_wh": 30.0,
              "own_box": True},
             {"box": "D3.1", "card": card(v100, 64.0, "D3.1"), "answers": 100, "energy_wh": 32.0,
              "own_box": True}]
    reg = {"tasks": {gpqa: {"shard_bundles": parts}}}
    (d / "frontier_imports.json").write_text(json.dumps(reg))
    sp = beyond.spent(d)
    assert sp["cards"] == [v100] and sp["probe"]["alone_out"]["median"] == 62.0
    assert sp["benches"][gpqa]["own_box"] is True and "usd_per_answer" not in sp["benches"][gpqa]
    assert sp["own_box"] is True
    # and a row whose boxes had both cards: each card's medians, never one
    reg["tasks"][gpqa]["shard_bundles"] = [*parts, {"box": "L1", "card": card(rtx, 150.0, "L1"),
                                                    "answers": 100, "energy_wh": 20.0,
                                                    "cost_usd": 0.5}]
    (d / "frontier_imports.json").write_text(json.dumps(reg))
    sp = beyond.spent(d)
    assert sp["probe"] == {} and sorted(sp["by_card"]) == ["RTX 5090", v100]
    assert sp["by_card"][v100]["probe"]["alone_out"]["median"] == 62.0
    assert sp["by_card"]["RTX 5090"]["probe"]["alone_out"]["median"] == 150.0
    assert "own_box" not in sp["benches"][gpqa] and "usd" not in sp["benches"][gpqa]
    # a report's speed column says the card with the number
    reg["tasks"][gpqa]["shard_bundles"] = parts
    (d / "frontier_imports.json").write_text(json.dumps(reg))
    cols = reports.speed_cols([{"label": Q4, "dir": str(d)}])
    assert cols[Q4]["card"] == v100 and cols[Q4]["alone_out"]["median"] == 62.0
